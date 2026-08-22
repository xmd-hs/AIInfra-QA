#pragma once

#include "telemetry.hpp"

#include <functional>
#include <string>
#include <vector>

namespace gpuforge {

struct BenchmarkCase {
  std::string name;
  double flops = 0;
  double bytes = 0;
  std::function<void()> fn;
};

struct BenchmarkResult {
  std::string name;
  double ms = 0;
  double gflops = 0;
  double gbps = 0;
  size_t iterations = 0;
};

class BenchmarkSuite {
 public:
  void add(BenchmarkCase);
  std::vector<BenchmarkResult> run(int warmup = 2, int iterations = 10) const;
  std::string markdown(const std::vector<BenchmarkResult>&) const;

 private:
  std::vector<BenchmarkCase> cases_;
};

}  // namespace gpuforge
