#include "gpuforge/runtime.hpp"

#include <algorithm>
#include <chrono>

namespace gpuforge {

Sample Profiler::measure(const std::function<void()>& fn, double flops, int iters) const {
  iters = std::max(1, iters);

  for (int i = 0; i < 2; ++i) {
    fn();
  }

  const auto t = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iters; ++i) {
    fn();
  }

  const auto end = std::chrono::high_resolution_clock::now();
  const double ms =
      std::chrono::duration<double, std::milli>(end - t).count() / iters;
  return {ms, ms > 0 ? flops / (ms * 1e6) : 0};
}

GemmConfig AutoTuner::choose(size_t m, size_t n, size_t k) const {
  const auto key = std::make_tuple(m, n, k);
  const auto it = cache_.find(key);
  if (it != cache_.end()) {
    return it->second;
  }

  GemmConfig c;
  const size_t work = m * n * k;
  c.tile = work > (1 << 22) ? 32 : 16;
  c.warps = (m * n > 4096) ? 8 : 4;
  c.use_tensor_core = (k % 8 == 0);

  cache_[key] = c;
  return c;
}

}  // namespace gpuforge
