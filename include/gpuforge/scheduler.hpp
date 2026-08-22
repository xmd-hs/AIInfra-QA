#pragma once

#include "memory.hpp"

#include <cstddef>
#include <deque>
#include <mutex>
#include <vector>

namespace gpuforge::runtime {

enum class Phase { Prefill, Decode, Finished };

struct Request {
  uint64_t id = 0;
  size_t prompt = 0;
  size_t total = 0;
  size_t generated = 0;
  Phase phase = Phase::Prefill;
};

struct Batch {
  std::vector<uint64_t> ids;
  Phase phase = Phase::Decode;
  size_t tokens = 0;
};

class Scheduler {
 public:
  explicit Scheduler(size_t max_batch = 8) : max_batch_(max_batch) {}

  void submit(Request);
  void cancel(uint64_t);
  Batch next();
  void complete(uint64_t, size_t tokens = 1);
  size_t pending() const;
  size_t active() const;

 private:
  size_t max_batch_;
  mutable std::mutex mu_;
  std::deque<Request> queue_;
  std::vector<Request> active_;
};

}  // namespace gpuforge::runtime
