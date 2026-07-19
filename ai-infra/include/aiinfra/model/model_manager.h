#pragma once

#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "aiinfra/gpu/paged_memory_pool.h"
#include "aiinfra/model/inference_engine.h"
#include "aiinfra/observability/metrics.h"

namespace aiinfra::model {

struct ModelInfo {
  std::string name;
  std::string path;
  bool loaded{false};
  std::size_t active_inferences{0};
  std::size_t allocated_pages{0};
  std::chrono::seconds idle_ttl{0};
};

class ModelManager {
 public:
  ModelManager(gpu::PagedGpuMemoryPool& memory_pool,
               observability::MetricsRegistry& metrics,
               InferenceEngineFactory factory = make_configured_inference_engine);

  void register_model(ModelConfig config);
  std::vector<scheduler::InferenceResponse> infer_batch(
      const std::string& model_name,
      const std::vector<scheduler::InferenceRequest>& requests);

  std::size_t unload_idle(std::chrono::steady_clock::time_point now);
  std::size_t loaded_model_count() const;
  std::vector<ModelInfo> snapshot() const;

 private:
  struct Entry {
    ModelConfig config;
    std::shared_ptr<InferenceEngine> engine;
    std::vector<gpu::PageAllocation> allocations;
    std::chrono::steady_clock::time_point last_used;
    std::size_t active_inferences{0};
  };

  Entry& get_or_load_locked(const std::string& model_name);
  std::size_t loaded_model_count_locked() const;
  void finish_inference(const std::string& model_name);

  gpu::PagedGpuMemoryPool& memory_pool_;
  observability::MetricsRegistry& metrics_;
  InferenceEngineFactory factory_;
  mutable std::mutex mutex_;
  std::unordered_map<std::string, Entry> models_;
};

}  // namespace aiinfra::model
