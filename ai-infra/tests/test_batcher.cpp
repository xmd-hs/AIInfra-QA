#include <cassert>
#include <future>
#include <string>
#include <utility>
#include <vector>

#include "aiinfra/gpu/paged_memory_pool.h"
#include "aiinfra/model/model_manager.h"
#include "aiinfra/observability/metrics.h"
#include "aiinfra/scheduler/continuous_batcher.h"

namespace {
class TestEngine final : public aiinfra::model::InferenceEngine {
 public:
  void load(const aiinfra::model::ModelConfig& config) override { name_ = config.name; loaded_ = true; }
  void unload() override { loaded_ = false; }
  bool loaded() const override { return loaded_; }
  std::vector<aiinfra::scheduler::InferenceResponse> infer_batch(
      const std::vector<aiinfra::scheduler::InferenceRequest>& requests) override {
    std::vector<aiinfra::scheduler::InferenceResponse> responses;
    for (const auto& request : requests) {
      aiinfra::scheduler::InferenceResponse response;
      response.request_id = request.request_id;
      response.text = "[model=" + name_ + "] " + request.prompt;
      responses.push_back(std::move(response));
    }
    return responses;
  }
 private:
  bool loaded_{false};
  std::string name_;
};

std::unique_ptr<aiinfra::model::InferenceEngine> make_test_engine() {
  return std::make_unique<TestEngine>();
}
}  // namespace

int main() {
  aiinfra::gpu::PagedGpuMemoryPool pool(1024 * 1024, 128);
  aiinfra::observability::MetricsRegistry metrics;
  aiinfra::model::ModelManager models(pool, metrics, make_test_engine);

  aiinfra::model::ModelConfig config;
  config.name = "demo";
  config.path = "demo.gguf";
  config.weight_bytes = 4 * 1024 * 1024;
  models.register_model(config);

  aiinfra::scheduler::BatcherConfig batcher_config;
  batcher_config.max_batch_size = 4;
  batcher_config.max_batch_tokens = 12;
  batcher_config.batch_window = std::chrono::milliseconds(2);
  batcher_config.worker_threads = 2;
  aiinfra::scheduler::ContinuousBatcher batcher(models, metrics, batcher_config);

  std::vector<std::future<aiinfra::scheduler::InferenceResponse>> futures;
  for (int i = 0; i < 8; ++i) {
    aiinfra::scheduler::InferenceRequest req;
    req.model = "demo";
    req.prompt = "hello";
    req.max_tokens = 4;
    req.estimated_prompt_tokens = 2;
    futures.push_back(batcher.submit(std::move(req)));
  }

  for (auto& future : futures) {
    auto response = future.get();
    assert(response.ok);
    assert(response.text.find("model=demo") != std::string::npos);
  }

  batcher.stop();
  assert(metrics.counter("aiinfra_requests_completed_total") == 8);
  assert(metrics.gauge("aiinfra_last_batch_tokens") <= 12.0);

  aiinfra::scheduler::InferenceRequest stopped_req;
  stopped_req.model = "demo";
  stopped_req.prompt = "after stop";
  auto rejected = batcher.submit(std::move(stopped_req)).get();
  assert(!rejected.ok);
  return 0;
}
