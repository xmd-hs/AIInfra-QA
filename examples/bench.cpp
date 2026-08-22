#include "gpuforge/kernels.hpp"
#include "gpuforge/runtime.hpp"
#include <cstdlib>
#include <iostream>
using namespace gpuforge;
int main(int argc, char** argv) {
  const size_t m = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 256;
  const size_t n = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : m;
  const size_t k = argc > 3 ? std::strtoull(argv[3], nullptr, 10) : m;
  const int iters = argc > 4 ? std::atoi(argv[4]) : 5;
  if (m == 0 || n == 0 || k == 0 || iters <= 0) {
    std::cerr << "usage: gpuforge-bench [M N K ITERS]\n";
    return 2;
  }
  Tensor a({m, k}), b({k, n});
  a.fill(1); b.fill(0.5f);
  AutoTuner tuner;
  const auto config = tuner.choose(m, n, k);
  Profiler profiler;
  const double flops = 2.0 * static_cast<double>(m) * n * k;
  const auto cpu = profiler.measure([&] { volatile auto x = gemm(a, b, config); }, flops, iters);
  std::cout << "backend=cpu,m=" << m << ",n=" << n << ",k=" << k
            << ",iters=" << iters << ",ms=" << cpu.ms
            << ",gflops=" << cpu.gflops << ",config=" << kernel_config(config) << "\n";
  if (cuda_available()) {
    const auto cuda = profiler.measure([&] { volatile auto x = gemm_cuda(a, b, config); }, flops, iters);
    std::cout << "backend=cuda,m=" << m << ",n=" << n << ",k=" << k
              << ",iters=" << iters << ",ms=" << cuda.ms
              << ",gflops=" << cuda.gflops << ",config=" << kernel_config(config) << "\n";
  } else {
    std::cout << "backend=cuda,status=unavailable\n";
  }
  return 0;
}
