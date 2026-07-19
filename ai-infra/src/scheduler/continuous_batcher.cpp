#include "aiinfra/scheduler/continuous_batcher.h"

#include <algorithm>
#include <exception>
#include <memory>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace aiinfra::scheduler {

ContinuousBatcher::ContinuousBatcher(model::ModelManager& model_manager,
                                     observability::MetricsRegistry& metrics,
                                     BatcherConfig config)
    : model_manager_(model_manager),
      metrics_(metrics),
      config_(config),
      queue_(config.queue_capacity),
      workers_(config.worker_threads, config.queue_capacity) {
  scheduler_thread_ = std::thread([this]() { loop(); });
}

ContinuousBatcher::~ContinuousBatcher() { stop(); }

std::future<InferenceResponse> ContinuousBatcher::submit(InferenceRequest request) {
  request.request_id = next_request_id_.fetch_add(1, std::memory_order_relaxed);
  request.enqueue_time = Clock::now();
  request.deadline = request.enqueue_time + config_.request_timeout;

  auto ctx = std::make_shared<RequestContext>(std::move(request));
  auto future = ctx->completion.get_future();

  if (!running_.load(std::memory_order_acquire)) {
    InferenceResponse response;
    response.request_id = ctx->request.request_id;
    response.ok = false;
    response.error = "batcher is stopped";
    ctx->completion.set_value(std::move(response));
    metrics_.increment("aiinfra_requests_rejected_total");
    return future;
  }

  if (!queue_.enqueue(ctx)) {
    InferenceResponse response;
    response.request_id = ctx->request.request_id;
    response.ok = false;
    response.error = "request queue is full";
    ctx->completion.set_value(std::move(response));
    metrics_.increment("aiinfra_requests_rejected_total");
    return future;
  }

  const auto depth = queue_depth_.fetch_add(1, std::memory_order_relaxed) + 1;
  metrics_.increment("aiinfra_requests_enqueued_total");
  metrics_.set_gauge("aiinfra_queue_depth", static_cast<double>(depth));
  return future;
}

void ContinuousBatcher::stop() {
  bool expected = true;
  if (!running_.compare_exchange_strong(expected, false)) {
    return;
  }
  if (scheduler_thread_.joinable()) {
    scheduler_thread_.join();
  }
  drain_rejected("batcher stopped before dispatch");
  workers_.shutdown();
}

void ContinuousBatcher::loop() {
  RequestPtr carry;
  while (running_.load(std::memory_order_acquire)) {
    RequestPtr first;
    if (carry) {
      first = std::move(carry);
    } else if (!queue_.dequeue(first)) {
      std::this_thread::sleep_for(std::chrono::microseconds(200));
      continue;
    } else {
      queue_depth_.fetch_sub(1, std::memory_order_relaxed);
    }

    std::vector<RequestPtr> batch;
    batch.reserve(config_.max_batch_size);
    std::size_t batch_tokens = first->request.estimated_total_tokens();
    batch.push_back(std::move(first));

    const auto pressure = queue_depth_.load(std::memory_order_relaxed);
    const bool drain_immediately = pressure >= config_.max_batch_size;
    const auto window_deadline = Clock::now() + config_.batch_window;
    while (batch.size() < config_.max_batch_size &&
           (drain_immediately || Clock::now() < window_deadline)) {
      RequestPtr next;
      if (queue_.dequeue(next)) {
        queue_depth_.fetch_sub(1, std::memory_order_relaxed);
        const auto next_tokens = next->request.estimated_total_tokens();
        if (!batch.empty() && batch_tokens + next_tokens > config_.max_batch_tokens) {
          carry = std::move(next);
          break;
        }
        batch_tokens += next_tokens;
        batch.push_back(std::move(next));
      } else {
        std::this_thread::sleep_for(std::chrono::microseconds(100));
      }
    }

    metrics_.observe_us("aiinfra_batch_build_latency",
                        std::chrono::duration_cast<std::chrono::microseconds>(
                            Clock::now() - batch.front()->request.enqueue_time));
    metrics_.set_gauge("aiinfra_last_batch_size", static_cast<double>(batch.size()));
    metrics_.set_gauge("aiinfra_last_batch_tokens", static_cast<double>(batch_tokens));
    metrics_.set_gauge("aiinfra_queue_depth",
                       static_cast<double>(queue_depth_.load(std::memory_order_relaxed)));
    dispatch(std::move(batch));
  }
  if (carry) fail(std::move(carry), "batcher stopped before dispatch");
}

void ContinuousBatcher::dispatch(std::vector<RequestPtr> batch) {
  std::unordered_map<std::string, std::vector<RequestPtr>> by_model;
  const auto now = Clock::now();
  for (auto& ctx : batch) {
    if (ctx->request.deadline <= now) {
      fail(ctx, "request timed out before dispatch");
      metrics_.increment("aiinfra_requests_timeout_total");
      continue;
    }
    by_model[ctx->request.model].push_back(std::move(ctx));
  }

  for (auto& [model_name, group] : by_model) {
    auto group_ptr = std::make_shared<std::vector<RequestPtr>>(std::move(group));
    try {
      workers_.submit([this, model_name, group_ptr]() mutable {
        auto& group = *group_ptr;
        std::vector<InferenceRequest> requests;
        requests.reserve(group.size());
        for (const auto& ctx : group) {
          requests.push_back(ctx->request);
        }

        const auto infer_start = Clock::now();
        try {
          auto responses = model_manager_.infer_batch(model_name, requests);
          for (std::size_t i = 0; i < group.size(); ++i) {
            InferenceResponse response;
            if (i < responses.size()) {
              response = std::move(responses[i]);
            } else {
              response.ok = false;
              response.error = "inference engine returned fewer responses than requests";
            }
            response.request_id = group[i]->request.request_id;
            response.queue_latency =
                std::chrono::duration_cast<std::chrono::microseconds>(infer_start - group[i]->request.enqueue_time);
            group[i]->completion.set_value(std::move(response));
            if (i < responses.size() && responses[i].ok) {
              metrics_.increment("aiinfra_requests_completed_total");
            } else {
              metrics_.increment("aiinfra_requests_failed_total");
            }
          }
        } catch (const std::exception& ex) {
          for (auto& ctx : group) {
            fail(ctx, ex.what());
          }
        }
      });
    } catch (const std::exception& ex) {
      for (auto& ctx : *group_ptr) {
        fail(ctx, ex.what());
      }
    }
  }
}

void ContinuousBatcher::drain_rejected(const std::string& error) {
  RequestPtr pending;
  while (queue_.dequeue(pending)) {
    queue_depth_.fetch_sub(1, std::memory_order_relaxed);
    fail(pending, error);
  }
  metrics_.set_gauge("aiinfra_queue_depth", 0.0);
}

void ContinuousBatcher::fail(RequestPtr ctx, const std::string& error) {
  InferenceResponse response;
  response.request_id = ctx->request.request_id;
  response.ok = false;
  response.error = error;
  response.queue_latency =
      std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - ctx->request.enqueue_time);
  ctx->completion.set_value(std::move(response));
  metrics_.increment("aiinfra_requests_failed_total");
}

}  // namespace aiinfra::scheduler
