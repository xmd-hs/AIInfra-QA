#include "gpuforge/scheduler.hpp"

#include <algorithm>

namespace gpuforge::runtime {

void Scheduler::submit(Request request) {
  std::lock_guard<std::mutex> lock(mu_);
  request.phase = Phase::Prefill;
  queue_.push_back(request);
}

void Scheduler::cancel(uint64_t request_id) {
  std::lock_guard<std::mutex> lock(mu_);
  queue_.erase(
      std::remove_if(queue_.begin(), queue_.end(),
                     [request_id](const Request& request) {
                       return request.id == request_id;
                     }),
      queue_.end());
  active_.erase(
      std::remove_if(active_.begin(), active_.end(),
                     [request_id](const Request& request) {
                       return request.id == request_id;
                     }),
      active_.end());
}

Batch Scheduler::next() {
  std::lock_guard<std::mutex> lock(mu_);
  Batch batch;
  while (active_.size() < max_batch_ && !queue_.empty()) {
    active_.push_back(queue_.front());
    queue_.pop_front();
  }
  if (active_.empty()) return batch;
  // A batch has one execution contract. Do not mix Prefill and Decode work;
  // their token shapes and kernel paths are different.
  batch.phase = active_.front().phase;
  for (const Request& request : active_) {
    if (request.phase != batch.phase) continue;
    batch.ids.push_back(request.id);
    if (request.phase == Phase::Prefill) {
      batch.tokens += request.prompt;
    } else {
      batch.tokens += 1;
    }
  }
  return batch;
}

void Scheduler::complete(uint64_t request_id, size_t tokens) {
  std::lock_guard<std::mutex> lock(mu_);
  for (Request& request : active_) {
    if (request.id != request_id) continue;
    request.generated += tokens;
    request.phase = request.generated + request.prompt >= request.total
                        ? Phase::Finished
                        : Phase::Decode;
  }
  active_.erase(
      std::remove_if(active_.begin(), active_.end(),
                     [](const Request& request) {
                       return request.phase == Phase::Finished;
                     }),
      active_.end());
}

size_t Scheduler::pending() const {
  std::lock_guard<std::mutex> lock(mu_);
  return queue_.size();
}

size_t Scheduler::active() const {
  std::lock_guard<std::mutex> lock(mu_);
  return active_.size();
}

}  // namespace gpuforge::runtime
