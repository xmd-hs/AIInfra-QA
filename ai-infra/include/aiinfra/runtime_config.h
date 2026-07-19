#pragma once

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <string>

namespace aiinfra {

struct RuntimeConfig {
  std::uint16_t port{8080};
  std::size_t gpu_page_size_bytes{16ULL * 1024ULL * 1024ULL};
  std::size_t gpu_page_count{256};
  double gpu_high_watermark{0.92};
  std::size_t max_batch_size{16};
  std::size_t max_batch_tokens{4096};
  std::chrono::milliseconds batch_window{3};
  std::size_t worker_threads{2};
  double rate_limit_per_second{2000.0};
  double rate_limit_burst{4000.0};
};

inline std::string env_string(const char* name, const std::string& fallback) {
  const char* value = std::getenv(name);
  return value == nullptr ? fallback : value;
}

inline std::size_t env_size(const char* name, std::size_t fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return fallback;
  }
  try {
    return static_cast<std::size_t>(std::stoull(value));
  } catch (...) {
    return fallback;
  }
}

inline double env_double(const char* name, double fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return fallback;
  }
  try {
    return std::stod(value);
  } catch (...) {
    return fallback;
  }
}

inline RuntimeConfig load_runtime_config() {
  RuntimeConfig config;
  config.port = static_cast<std::uint16_t>(env_size("AIINFRA_PORT", config.port));
  config.gpu_page_size_bytes = env_size("AIINFRA_GPU_PAGE_SIZE_BYTES", config.gpu_page_size_bytes);
  config.gpu_page_count = env_size("AIINFRA_GPU_PAGE_COUNT", config.gpu_page_count);
  config.gpu_high_watermark = env_double("AIINFRA_GPU_HIGH_WATERMARK", config.gpu_high_watermark);
  config.max_batch_size = env_size("AIINFRA_MAX_BATCH_SIZE", config.max_batch_size);
  config.max_batch_tokens = env_size("AIINFRA_MAX_BATCH_TOKENS", config.max_batch_tokens);
  config.batch_window = std::chrono::milliseconds(env_size("AIINFRA_BATCH_WINDOW_MS", config.batch_window.count()));
  config.worker_threads = env_size("AIINFRA_WORKER_THREADS", config.worker_threads);
  config.rate_limit_per_second = env_double("AIINFRA_RATE_LIMIT_PER_SECOND", config.rate_limit_per_second);
  config.rate_limit_burst = env_double("AIINFRA_RATE_LIMIT_BURST", config.rate_limit_burst);
  return config;
}

}  // namespace aiinfra
