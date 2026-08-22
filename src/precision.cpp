#include "gpuforge/precision.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace gpuforge {

ErrorStats compare(const Tensor& expected, const Tensor& actual, float epsilon) {
  if (expected.shape() != actual.shape()) {
    throw std::invalid_argument("compare requires equal tensor shapes");
  }
  ErrorStats stats;
  stats.count = expected.numel();
  for (size_t index = 0; index < stats.count; ++index) {
    const float absolute = std::fabs(expected.at(index) - actual.at(index));
    const float relative = absolute / (std::fabs(actual.at(index)) + epsilon);
    stats.max_abs = std::max(stats.max_abs, absolute);
    stats.mean_abs += absolute;
    stats.max_rel = std::max(stats.max_rel, relative);
  }
  if (stats.count != 0) stats.mean_abs /= static_cast<float>(stats.count);
  return stats;
}

bool tensor_core_compatible(const Tensor& lhs, const Tensor& rhs, int tile) {
  if (tile <= 0 || lhs.ndim() != 2 || rhs.ndim() != 2) return false;
  return lhs.shape()[1] == rhs.shape()[0] && lhs.shape()[0] % tile == 0 &&
         lhs.shape()[1] % tile == 0 && rhs.shape()[1] % tile == 0;
}

std::string precision_report(const ErrorStats& stats) {
  std::ostringstream output;
  output << "count=" << stats.count << " max_abs=" << stats.max_abs
         << " mean_abs=" << stats.mean_abs << " max_rel=" << stats.max_rel;
  return output.str();
}

}  // namespace gpuforge
