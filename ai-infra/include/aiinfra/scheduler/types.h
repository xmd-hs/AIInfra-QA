#pragma once

#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace aiinfra::scheduler {

using Clock = std::chrono::steady_clock;

struct InferenceRequest {
  std::uint64_t request_id{0};
  std::string model;
  std::string prompt;
  std::uint32_t max_tokens{64};
  std::uint32_t estimated_prompt_tokens{0};

  std::uint32_t estimated_total_tokens() const {
    return estimated_prompt_tokens + max_tokens;
  }
  Clock::time_point enqueue_time{Clock::now()};
  Clock::time_point deadline{Clock::now() + std::chrono::seconds(30)};
};

struct InferenceResponse {
  std::uint64_t request_id{0};
  bool ok{true};
  std::string text;
  std::string error;
  std::chrono::microseconds queue_latency{0};
  std::chrono::microseconds inference_latency{0};
  std::chrono::microseconds ttft{0};
};

struct RequestContext {
  explicit RequestContext(InferenceRequest req) : request(std::move(req)) {}

  InferenceRequest request;
  std::promise<InferenceResponse> completion;
};

using RequestPtr = std::shared_ptr<RequestContext>;

}  // namespace aiinfra::scheduler
