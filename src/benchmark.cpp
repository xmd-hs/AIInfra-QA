#include "gpuforge/benchmark.hpp"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace gpuforge {

void BenchmarkSuite::add(BenchmarkCase benchmark) {
  if (benchmark.fn) cases_.push_back(std::move(benchmark));
}

std::vector<BenchmarkResult> BenchmarkSuite::run(int warmup, int iterations) const {
  const int rounds = std::max(1, iterations);
  std::vector<BenchmarkResult> results;
  for (const BenchmarkCase& benchmark : cases_) {
    for (int i = 0; i < std::max(0, warmup); ++i) benchmark.fn();
    const auto begin = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < rounds; ++i) benchmark.fn();
    const auto end = std::chrono::high_resolution_clock::now();
    const double milliseconds =
        std::chrono::duration<double, std::milli>(end - begin).count() / rounds;
    results.push_back({benchmark.name, milliseconds,
                       milliseconds > 0 ? benchmark.flops / (milliseconds * 1e6) : 0,
                       milliseconds > 0 ? benchmark.bytes / (milliseconds * 1e6) : 0,
                       static_cast<size_t>(rounds)});
  }
  return results;
}

std::string BenchmarkSuite::markdown(const std::vector<BenchmarkResult>& results) const {
  std::ostringstream output;
  output << "| case | ms | GFLOP/s | GB/s | iterations |\n"
         << "|---|---:|---:|---:|---:|\n" << std::fixed << std::setprecision(3);
  for (const BenchmarkResult& result : results) {
    output << "| " << result.name << " | " << result.ms << " | "
           << result.gflops << " | " << result.gbps << " | "
           << result.iterations << " |\n";
  }
  return output.str();
}

}  // namespace gpuforge
