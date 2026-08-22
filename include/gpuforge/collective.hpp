#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace gpuforge::distributed {

enum class Collective { AllReduce, AllGather, ReduceScatter, Broadcast };

struct CommPlan {
  Collective op;
  int world = 1;
  int rank = 0;
  size_t bytes = 0;
  size_t chunks = 1;
  bool overlap = true;
  std::string algorithm;
};

class Communicator {
 public:
  Communicator(int world = 1, int rank = 0) : world_(world), rank_(rank) {}

  CommPlan plan(Collective, size_t bytes, bool overlap = true) const;
  std::string describe(const CommPlan&) const;
  bool valid() const { return world_ > 0 && rank_ >= 0 && rank_ < world_; }

 private:
  int world_;
  int rank_;
};

class OverlapPlanner {
 public:
  static std::vector<CommPlan> pipeline(const CommPlan&,
                                        size_t compute_bytes,
                                        int stages = 2);
};

}  // namespace gpuforge::distributed
