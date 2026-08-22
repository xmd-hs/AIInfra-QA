#pragma once

#include "tensor.hpp"

#include <string>

namespace gpuforge {

struct ErrorStats {
  float max_abs = 0;
  float mean_abs = 0;
  float max_rel = 0;
  size_t count = 0;

  bool pass(float tol) const { return max_abs <= tol; }
};

ErrorStats compare(const Tensor&, const Tensor&, float eps = 1e-6f);
bool tensor_core_compatible(const Tensor&, const Tensor&, int tile = 16);
std::string precision_report(const ErrorStats&);

}  // namespace gpuforge
