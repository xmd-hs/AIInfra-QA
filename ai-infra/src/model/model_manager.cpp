#include "aiinfra/model/model_manager.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace aiinfra::model {
ModelManager::ModelManager(gpu::PagedGpuMemoryPool& memory_pool,
                           observability::MetricsRegistry& metrics,
                           InferenceEngineFactory factory)
    : memory_pool_(memory_pool), metrics_(metrics), factory_(factory) {}

void ModelManager::register_model(ModelConfig config) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto [it, inserted] = models_.emplace(config.name, Entry{});
  it->second.config = std::move(config);
  if (inserted) {
    it->second.last_used = std::chrono::steady_clock::now();
  }
}

std::vector<scheduler::InferenceResponse> ModelManager::infer_batch(
    const std::string& model_name,
    const std::vector<scheduler::InferenceRequest>& requests) {
  std::shared_ptr<InferenceEngine> engine;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& entry = get_or_load_locked(model_name);
    entry.last_used = std::chrono::steady_clock::now();
    ++entry.active_inferences;
    engine = entry.engine;
  }

  observability::ScopedTimer timer(metrics_, "aiinfra_model_infer_batch_latency");
  try {
    auto responses = engine->infer_batch(requests);
    finish_inference(model_name);
    return responses;
  } catch (...) {
    finish_inference(model_name);
    throw;
  }
}

std::size_t ModelManager::unload_idle(std::chrono::steady_clock::time_point now) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::size_t unloaded = 0;
  for (auto& [name, entry] : models_) {
    if (entry.engine == nullptr || !entry.engine->loaded() || entry.active_inferences > 0) {
      continue;
    }
    if (now - entry.last_used < entry.config.idle_ttl) {
      continue;
    }
    entry.engine->unload();
    entry.engine.reset();
    for (const auto& allocation : entry.allocations) {
      memory_pool_.release(allocation);
    }
    entry.allocations.clear();
    ++unloaded;
    metrics_.increment("aiinfra_model_unloads_total");
  }
  metrics_.set_gauge("aiinfra_loaded_models", static_cast<double>(loaded_model_count_locked()));
  return unloaded;
}

std::size_t ModelManager::loaded_model_count() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return loaded_model_count_locked();
}

std::vector<ModelInfo> ModelManager::snapshot() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<ModelInfo> result;
  result.reserve(models_.size());
  for (const auto& [name, entry] : models_) {
    std::size_t pages = 0;
    for (const auto& allocation : entry.allocations) {
      pages += allocation.pages.size();
    }
    result.push_back(ModelInfo{
        name,
        entry.config.path,
        entry.engine != nullptr && entry.engine->loaded(),
        entry.active_inferences,
        pages,
        entry.config.idle_ttl,
    });
  }
  return result;
}

std::size_t ModelManager::loaded_model_count_locked() const {
  std::size_t count = 0;
  for (const auto& [_, entry] : models_) {
    if (entry.engine != nullptr && entry.engine->loaded()) {
      ++count;
    }
  }
  return count;
}

void ModelManager::finish_inference(const std::string& model_name) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = models_.find(model_name);
  if (it != models_.end() && it->second.active_inferences > 0) {
    --it->second.active_inferences;
    it->second.last_used = std::chrono::steady_clock::now();
  }
}

ModelManager::Entry& ModelManager::get_or_load_locked(const std::string& model_name) {
  auto it = models_.find(model_name);
  if (it == models_.end()) {
    throw std::runtime_error("model is not registered: " + model_name);
  }

  auto& entry = it->second;
  if (entry.engine != nullptr && entry.engine->loaded()) {
    return entry;
  }

  if (memory_pool_.above_high_watermark()) {
    throw std::runtime_error("gpu memory pool is above high watermark");
  }

  auto allocation = memory_pool_.allocate(entry.config.weight_bytes, "model:" + entry.config.name);
  if (!allocation) {
    throw std::runtime_error("not enough gpu pages to load model");
  }

  entry.engine = std::shared_ptr<InferenceEngine>(factory_());
  try {
    entry.engine->load(entry.config);
    entry.allocations.push_back(std::move(*allocation));
  } catch (...) {
    memory_pool_.release(*allocation);
    entry.engine.reset();
    throw;
  }
  metrics_.increment("aiinfra_model_loads_total");
  metrics_.set_gauge("aiinfra_loaded_models", static_cast<double>(loaded_model_count_locked()));
  return entry;
}

}  // namespace aiinfra::model
