#include "gpuforge/planner.hpp"

#include <algorithm>
#include <sstream>

namespace gpuforge::compiler {

namespace {
bool matrix_shape(const Shape& shape) { return shape.dims.size() == 2; }
bool aligned(const Shape& shape, int tile) {
  return matrix_shape(shape) && shape.dims[0] % tile == 0 &&
         shape.dims[1] % tile == 0;
}
}

TilePlan TilePlanner::plan(const Shape& lhs, const Shape& rhs,
                           const Schedule& schedule) const {
  TilePlan plan;
  plan.tile_m = schedule.block_m;
  plan.tile_n = schedule.block_n;
  plan.tile_k = schedule.block_k;

  if (!matrix_shape(lhs) || !matrix_shape(rhs)) {
    plan.reason = "matmul requires rank-2 tensors";
    return plan;
  }
  if (lhs.dims[1] != rhs.dims[0]) {
    plan.reason = "matmul inner dimensions mismatch";
    return plan;
  }

  plan.grid_m = (static_cast<int>(lhs.dims[0]) + plan.tile_m - 1) / plan.tile_m;
  plan.grid_n = (static_cast<int>(rhs.dims[1]) + plan.tile_n - 1) / plan.tile_n;
  plan.padded = lhs.dims[0] % plan.tile_m != 0 ||
                rhs.dims[1] % plan.tile_n != 0 ||
                lhs.dims[1] % plan.tile_k != 0;
  const size_t element_bytes = lhs.type == ScalarType::F32 ? 4 : 2;
  plan.shared_bytes = static_cast<size_t>(plan.tile_k) *
                      (plan.tile_m + plan.tile_n) * element_bytes;

  const bool wmma_shape = aligned(lhs, 16) && aligned(rhs, 16) &&
                          lhs.dims[1] % 16 == 0;
  if (schedule.tensor_core && lhs.type != ScalarType::F32 && wmma_shape) {
    plan.kind = KernelKind::WmmaTensorCore;
    plan.tile_m = plan.tile_n = plan.tile_k = 16;
    plan.reason = "aligned WMMA Tensor Core path";
  } else {
    plan.kind = KernelKind::CudaTiled;
    plan.reason = plan.padded ? "boundary guarded tiled path"
                              : "shared-memory tiled path";
  }
  return plan;
}

std::vector<TilePlan> TilePlanner::candidates(const Shape& lhs,
                                              const Shape& rhs) const {
  std::vector<TilePlan> result;
  for (const int tile : {16, 32, 64, 128}) {
    Schedule schedule;
    schedule.block_m = tile;
    schedule.block_n = tile;
    schedule.block_k = std::min(tile, 32);
    schedule.tensor_core = true;
    result.push_back(plan(lhs, rhs, schedule));
  }
  return result;
}

size_t TilePlanner::workspace(const TilePlan& plan, const Shape& output) const {
  if (!plan.padded) return 0;
  const size_t element_bytes = output.type == ScalarType::F32 ? 4 : 2;
  return output.elements() * element_bytes;
}

std::string TilePlanner::explain(const TilePlan& plan) const {
  std::ostringstream text;
  text << plan.reason << " tile=" << plan.tile_m << "x" << plan.tile_n
       << "x" << plan.tile_k << " grid=" << plan.grid_m << "x"
       << plan.grid_n << " shared=" << plan.shared_bytes
       << " padded=" << (plan.padded ? "true" : "false");
  return text.str();
}

}  // namespace gpuforge::compiler
