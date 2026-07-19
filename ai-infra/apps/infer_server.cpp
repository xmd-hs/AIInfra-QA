#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <future>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>

#include "aiinfra/gpu/paged_memory_pool.h"
#include "aiinfra/model/model_manager.h"
#include "aiinfra/net/http_server.h"
#include "aiinfra/observability/gpu_probe.h"
#include "aiinfra/observability/metrics.h"
#include "aiinfra/runtime_config.h"
#include "aiinfra/scheduler/admission_control.h"
#include "aiinfra/scheduler/continuous_batcher.h"

namespace {

std::atomic<bool> g_running{true};

void signal_handler(int) {
  g_running.store(false, std::memory_order_release);
}

std::optional<std::string> extract_json_string(const std::string& body, const std::string& field);
std::optional<std::uint32_t> extract_json_u32(const std::string& body, const std::string& field);
std::uint32_t parse_u32(const std::string& value, std::uint32_t fallback);

struct CachedResponse {
  std::string text;
  std::chrono::steady_clock::time_point expires_at;
};

std::size_t env_size(const char* name, std::size_t fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') return fallback;
  try { return static_cast<std::size_t>(std::stoull(value)); }
  catch (...) { return fallback; }
}

std::string extract_prompt(const aiinfra::net::HttpRequest& req) {
  const auto query_prompt = aiinfra::net::query_value(req, "prompt", "");
  if (!query_prompt.empty()) {
    return query_prompt;
  }
  if (req.body.empty()) {
    return "";
  }

  auto prompt = extract_json_string(req.body, "prompt");
  return prompt.value_or(req.body);
}

std::optional<std::string> extract_json_string(const std::string& body, const std::string& field) {
  const std::string key = "\"" + field + "\"";
  const auto key_pos = body.find(key);
  if (key_pos == std::string::npos) {
    return std::nullopt;
  }
  const auto colon = body.find(':', key_pos + key.size());
  const auto first_quote = body.find('"', colon == std::string::npos ? key_pos : colon + 1);
  if (first_quote == std::string::npos) {
    return std::nullopt;
  }

  std::string value;
  bool escaped = false;
  for (std::size_t i = first_quote + 1; i < body.size(); ++i) {
    const char ch = body[i];
    if (escaped) {
      switch (ch) {
        case 'n':
          value.push_back('\n');
          break;
        case 'r':
          value.push_back('\r');
          break;
        case 't':
          value.push_back('\t');
          break;
        default:
          value.push_back(ch);
          break;
      }
      escaped = false;
      continue;
    }
    if (ch == '\\') {
      escaped = true;
      continue;
    }
    if (ch == '"') {
      return value;
    }
    value.push_back(ch);
  }
  return std::nullopt;
}

std::optional<std::uint32_t> extract_json_u32(const std::string& body, const std::string& field) {
  const std::string key = "\"" + field + "\"";
  const auto key_pos = body.find(key);
  if (key_pos == std::string::npos) {
    return std::nullopt;
  }
  const auto colon = body.find(':', key_pos + key.size());
  if (colon == std::string::npos) {
    return std::nullopt;
  }
  std::size_t start = colon + 1;
  while (start < body.size() && body[start] == ' ') {
    ++start;
  }
  std::size_t end = start;
  while (end < body.size() && body[end] >= '0' && body[end] <= '9') {
    ++end;
  }
  if (start == end) {
    return std::nullopt;
  }
  return parse_u32(body.substr(start, end - start), 0);
}

std::uint32_t parse_u32(const std::string& value, std::uint32_t fallback) {
  if (value.empty()) {
    return fallback;
  }
  try {
    return static_cast<std::uint32_t>(std::stoul(value));
  } catch (...) {
    return fallback;
  }
}

}  // namespace

int main() {
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);

  const auto runtime = aiinfra::load_runtime_config();

  aiinfra::gpu::PagedGpuMemoryPool memory_pool(
      runtime.gpu_page_size_bytes, runtime.gpu_page_count, runtime.gpu_high_watermark);
  aiinfra::observability::MetricsRegistry metrics;
  aiinfra::model::ModelManager model_manager(memory_pool, metrics);

  const auto register_model = [&](const std::string& name, const std::string& backend_model,
                                  std::size_t estimated_bytes) {
    aiinfra::model::ModelConfig config;
    config.name = name;
    config.path = backend_model;
    config.weight_bytes = estimated_bytes;
    config.idle_ttl = std::chrono::seconds(180);
    model_manager.register_model(std::move(config));
  };
  register_model("deepseek-chat", "deepseek-chat", 256ULL * 1024ULL * 1024ULL);

  aiinfra::scheduler::BatcherConfig batcher_config;
  batcher_config.max_batch_size = runtime.max_batch_size;
  batcher_config.max_batch_tokens = runtime.max_batch_tokens;
  batcher_config.batch_window = runtime.batch_window;
  batcher_config.worker_threads = runtime.worker_threads;
  aiinfra::scheduler::ContinuousBatcher batcher(model_manager, metrics, batcher_config);
  aiinfra::scheduler::TokenBucket limiter(runtime.rate_limit_per_second, runtime.rate_limit_burst);
  aiinfra::scheduler::CircuitBreaker breaker(8, std::chrono::seconds(5));
  std::mutex response_cache_mutex;
  std::unordered_map<std::string, CachedResponse> response_cache;
  const auto response_cache_ttl = std::chrono::seconds(env_size("AIINFRA_CACHE_TTL_SECONDS", 300));
  const auto response_cache_capacity = env_size("AIINFRA_CACHE_CAPACITY", 512);

  aiinfra::observability::GpuProbe gpu_probe(&memory_pool);
  aiinfra::net::HttpServer server(runtime.port);

  server.route("/health", [](const aiinfra::net::HttpRequest&) {
    return aiinfra::net::HttpResponse{200, "application/json", "{\"status\":\"ok\"}"};
  });

  server.route("/metrics", [&](const aiinfra::net::HttpRequest&) {
    const auto gpu = gpu_probe.sample();
    const auto pool = memory_pool.stats();
    metrics.set_gauge("aiinfra_gpu_memory_utilization", gpu.memory_utilization);
    metrics.set_gauge("aiinfra_gpu_pool_fragmentation_ratio", pool.fragmentation_ratio);
    metrics.set_gauge("aiinfra_gpu_pool_used_pages", static_cast<double>(pool.used_pages));
    metrics.set_gauge("aiinfra_gpu_pool_free_pages", static_cast<double>(pool.free_pages));
    metrics.set_gauge("aiinfra_gpu_pool_reserved_bytes", static_cast<double>(pool.reserved_bytes));
    metrics.set_gauge("aiinfra_gpu_pool_largest_free_run", static_cast<double>(pool.largest_free_run));
    return aiinfra::net::HttpResponse{200, "text/plain", metrics.to_prometheus()};
  });

  server.route("/v1/models", [&](const aiinfra::net::HttpRequest&) {
    const auto models = model_manager.snapshot();
    std::ostringstream body;
    body << "{\"models\":[";
    for (std::size_t i = 0; i < models.size(); ++i) {
      const auto& model = models[i];
      if (i > 0) {
        body << ",";
      }
      body << "{\"name\":\"" << aiinfra::net::json_escape(model.name) << "\""
           << ",\"path\":\"" << aiinfra::net::json_escape(model.path) << "\""
           << ",\"loaded\":" << (model.loaded ? "true" : "false")
           << ",\"active_inferences\":" << model.active_inferences
           << ",\"allocated_pages\":" << model.allocated_pages
           << ",\"idle_ttl_sec\":" << model.idle_ttl.count() << "}";
    }
    body << "]}";
    return aiinfra::net::HttpResponse{200, "application/json", body.str()};
  });

  server.route("/v1/infer", [&](const aiinfra::net::HttpRequest& req) {
    if (req.method != "POST") {
      return aiinfra::net::HttpResponse{400, "application/json", "{\"error\":\"POST required\"}"};
    }
    if (!limiter.allow()) {
      metrics.increment("aiinfra_requests_rate_limited_total");
      return aiinfra::net::HttpResponse{429, "application/json", "{\"error\":\"rate limited\"}"};
    }
    if (!breaker.allow()) {
      metrics.increment("aiinfra_requests_circuit_open_total");
      return aiinfra::net::HttpResponse{429, "application/json", "{\"error\":\"circuit open\"}"};
    }

    aiinfra::scheduler::InferenceRequest infer;
    infer.model = aiinfra::net::query_value(req, "model", "demo");
    infer.prompt = extract_prompt(req);
    infer.estimated_prompt_tokens = static_cast<std::uint32_t>(
        std::max<std::size_t>(1, (infer.prompt.size() + 3) / 4));
    infer.max_tokens = parse_u32(aiinfra::net::query_value(req, "max_tokens", ""), 0);
    if (infer.max_tokens == 0) {
      infer.max_tokens = extract_json_u32(req.body, "max_tokens").value_or(64);
    }

    const auto cache_key = infer.model + "\n" + std::to_string(infer.max_tokens) + "\n" + infer.prompt;
    if (response_cache_capacity > 0 && response_cache_ttl.count() > 0) {
      std::lock_guard<std::mutex> lock(response_cache_mutex);
      const auto cached = response_cache.find(cache_key);
      if (cached != response_cache.end()) {
        if (cached->second.expires_at > std::chrono::steady_clock::now()) {
          metrics.increment("aiinfra_response_cache_hits_total");
          return aiinfra::net::HttpResponse{
              200, "application/json",
              "{\"request_id\":0,\"text\":\"" + aiinfra::net::json_escape(cached->second.text) +
                  "\",\"queue_us\":0,\"infer_us\":0,\"ttft_us\":0,\"cache_hit\":true}"};
        }
        response_cache.erase(cached);
      }
      metrics.increment("aiinfra_response_cache_misses_total");
    }

    auto future = batcher.submit(std::move(infer));
    if (future.wait_for(std::chrono::seconds(35)) != std::future_status::ready) {
      breaker.record_failure();
      return aiinfra::net::HttpResponse{504, "application/json", "{\"error\":\"inference timeout\"}"};
    }

    const auto response = future.get();
    if (!response.ok) {
      breaker.record_failure();
      return aiinfra::net::HttpResponse{
          500, "application/json", "{\"error\":\"" + aiinfra::net::json_escape(response.error) + "\"}"};
    }
    breaker.record_success();

    if (response_cache_capacity > 0 && response_cache_ttl.count() > 0) {
      std::lock_guard<std::mutex> lock(response_cache_mutex);
      if (response_cache.size() >= response_cache_capacity && !response_cache.empty()) {
        response_cache.erase(response_cache.begin());
        metrics.increment("aiinfra_response_cache_evictions_total");
      }
      response_cache[cache_key] = CachedResponse{
          response.text, std::chrono::steady_clock::now() + response_cache_ttl};
      metrics.set_gauge("aiinfra_response_cache_entries", static_cast<double>(response_cache.size()));
    }

    std::ostringstream body;
    body << "{\"request_id\":" << response.request_id
         << ",\"text\":\"" << aiinfra::net::json_escape(response.text) << "\""
         << ",\"queue_us\":" << response.queue_latency.count()
         << ",\"infer_us\":" << response.inference_latency.count()
         << ",\"ttft_us\":" << response.ttft.count() << ",\"cache_hit\":false}";
    return aiinfra::net::HttpResponse{200, "application/json", body.str()};
  });

  std::thread idle_reaper([&]() {
    while (g_running.load(std::memory_order_acquire)) {
      model_manager.unload_idle(std::chrono::steady_clock::now());
      std::this_thread::sleep_for(std::chrono::seconds(5));
    }
  });

  std::exception_ptr server_error;
  std::cout << "infer_server listening on 0.0.0.0:" << runtime.port << std::endl;
  std::thread server_thread([&]() {
    try {
      server.start();
    } catch (...) {
      server_error = std::current_exception();
      g_running.store(false, std::memory_order_release);
    }
  });

  while (g_running.load(std::memory_order_acquire)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }

  server.stop();
  if (server_thread.joinable()) {
    server_thread.join();
  }

  batcher.stop();
  if (idle_reaper.joinable()) {
    idle_reaper.join();
  }
  if (server_error) {
    try {
      std::rethrow_exception(server_error);
    } catch (const std::exception& ex) {
      std::cerr << "server error: " << ex.what() << std::endl;
    }
    return 1;
  }
  return 0;
}
