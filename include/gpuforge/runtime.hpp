#pragma once

#include "kernels.hpp"

#include <chrono>
#include <functional>
#include <map>
#include <tuple>

namespace gpuforge {

struct Sample {
  double ms = 0;
  double gflops = 0;
};

class Profiler {
 public:
  Sample measure(const std::function<void()>& fn,
                 double flops = 0,
                 int iters = 10) const;
};

class AutoTuner {
 public:
  GemmConfig choose(size_t m, size_t n, size_t k) const;

 private:
  mutable std::map<std::tuple<size_t, size_t, size_t>, GemmConfig> cache_;
};

}  // namespace gpuforge
