#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <future>
#include <thread>

#include "aiinfra/concurrency/mpmc_queue.h"
#include "aiinfra/concurrency/thread_pool.h"
#include "aiinfra/model/model_manager.h"
#include "aiinfra/observability/metrics.h"
#include "aiinfra/scheduler/types.h"

namespace aiinfra::scheduler {

struct BatcherConfig {
  std::size_t queue_capacity{8192};
  std::size_t max_batch_size{16};
  std::size_t max_batch_tokens{4096};
  std::chrono::microseconds batch_window{std::chrono::milliseconds(3)};
  std::chrono::milliseconds request_timeout{30000};
  std::size_t worker_threads{2};
};

class ContinuousBatcher {
 public:
  ContinuousBatcher(model::ModelManager& model_manager,
                    observability::MetricsRegistry& metrics,
                    BatcherConfig config = {});
  ~ContinuousBatcher();

  ContinuousBatcher(const ContinuousBatcher&) = delete;
  ContinuousBatcher& operator=(const ContinuousBatcher&) = delete;

  std::future<InferenceResponse> submit(InferenceRequest request);
  void stop();

 private:
  void loop();
  void dispatch(std::vector<RequestPtr> batch);
  void drain_rejected(const std::string& error);
  void fail(RequestPtr ctx, const std::string& error);

  model::ModelManager& model_manager_;
  observability::MetricsRegistry& metrics_;
  BatcherConfig config_;
  concurrency::BoundedMpmcQueue<RequestPtr> queue_;
  concurrency::ThreadPool workers_;
  std::atomic<bool> running_{true};
  std::atomic<std::uint64_t> next_request_id_{1};
  std::atomic<std::size_t> queue_depth_{0};
  std::thread scheduler_thread_;
};

}  // namespace aiinfra::scheduler
