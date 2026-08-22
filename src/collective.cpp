#include "gpuforge/collective.hpp"

#include <algorithm>
#include <sstream>

namespace gpuforge::distributed {

CommPlan Communicator::plan(Collective operation, size_t bytes,
                             bool overlap) const {
  CommPlan plan{operation, world_, rank_, bytes,
                static_cast<size_t>(std::max(1, world_ * 2)), overlap, "ring"};
  if (world_ <= 2) plan.algorithm = "direct";
  else if (bytes > (1u << 20)) plan.algorithm = "ring-chunked";
  if (operation == Collective::AllGather) plan.chunks = static_cast<size_t>(world_);
  return plan;
}

std::string Communicator::describe(const CommPlan& plan) const {
  std::ostringstream output;
  output << "rank " << plan.rank << "/" << plan.world
         << " bytes=" << plan.bytes << " chunks=" << plan.chunks
         << " algo=" << plan.algorithm
         << " overlap=" << (plan.overlap ? "true" : "false");
  return output.str();
}

std::vector<CommPlan> OverlapPlanner::pipeline(const CommPlan& plan,
                                               size_t compute_bytes,
                                               int stages) {
  const int stage_count = std::max(1, stages);
  const size_t chunk_bytes = (plan.bytes + stage_count - 1) / stage_count;
  std::vector<CommPlan> pipeline;
  pipeline.reserve(stage_count);
  for (int stage = 0; stage < stage_count; ++stage) {
    CommPlan chunk = plan;
    chunk.bytes = chunk_bytes;
    chunk.chunks = 1;
    chunk.overlap = true;
    chunk.algorithm = plan.algorithm + "/stage" + std::to_string(stage);
    pipeline.push_back(std::move(chunk));
  }
  if (compute_bytes > plan.bytes / 2) {
    for (CommPlan& chunk : pipeline) chunk.overlap = false;
  }
  return pipeline;
}

}  // namespace gpuforge::distributed
