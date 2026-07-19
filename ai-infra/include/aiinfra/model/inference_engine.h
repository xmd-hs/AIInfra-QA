#pragma once

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "aiinfra/scheduler/types.h"

namespace aiinfra::model {

struct ModelConfig {
  std::string name;
  std::string path;
  std::size_t weight_bytes{256ULL * 1024ULL * 1024ULL};
  std::chrono::seconds idle_ttl{std::chrono::seconds(300)};
};

class InferenceEngine {
 public:
  virtual ~InferenceEngine() = default;

  virtual void load(const ModelConfig& config) = 0;
  virtual void unload() = 0;
  virtual bool loaded() const = 0;

  virtual std::vector<scheduler::InferenceResponse> infer_batch(
      const std::vector<scheduler::InferenceRequest>& requests) = 0;
};

using InferenceEngineFactory = std::unique_ptr<InferenceEngine> (*)();

std::unique_ptr<InferenceEngine> make_configured_inference_engine();

}  // namespace aiinfra::model
