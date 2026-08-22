#pragma once

#include "compiler.hpp"
#include "kernels.hpp"

#include <string>
#include <vector>

namespace gpuforge::compiler {

enum class KernelKind { CudaTiled, WmmaTensorCore, Fallback };

struct TilePlan {
  KernelKind kind = KernelKind::Fallback;
  int tile_m = 16;
  int tile_n = 16;
  int tile_k = 16;
  int grid_m = 0;
  int grid_n = 0;
  size_t shared_bytes = 0;
  bool padded = false;
  std::string reason;
};

class TilePlanner {
 public:
  TilePlan plan(const Shape&, const Shape&, const Schedule&) const;
  std::vector<TilePlan> candidates(const Shape&, const Shape&) const;
  size_t workspace(const TilePlan&, const Shape&) const;
  std::string explain(const TilePlan&) const;
};

}  // namespace gpuforge::compiler
