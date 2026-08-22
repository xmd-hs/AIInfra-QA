#include "gpuforge/autotune.hpp"

#include <sstream>
#include <stdexcept>

namespace gpuforge {

GemmAutoTuner::GemmAutoTuner(AutoTuneConfig c) : cfg_(std::move(c)) {
  if (cfg_.persist && !cfg_.cache_file.empty()) {
    db_.load(cfg_.cache_file);
  }
}

TuneResult GemmAutoTuner::run(const Tensor& a, const Tensor& b) {
  if (a.ndim() != 2 || b.ndim() != 2 || a.shape()[1] != b.shape()[0]) {
    throw std::invalid_argument("autotune gemm shape");
  }

  const auto key =
      std::make_tuple(a.shape()[0], b.shape()[1], a.shape()[1]);
  const auto cached =
      db_.best(std::get<0>(key), std::get<1>(key), std::get<2>(key));
  if (cached.valid) {
    return cached;
  }

  BenchmarkSuite suite;
  const auto configs =
      Searcher::gemm_space(a.shape()[0], b.shape()[1], a.shape()[1]);
  if (configs.empty()) {
    throw std::invalid_argument("autotune has no valid GEMM configurations");
  }
  for (const auto& c : configs) {
    suite.add({
        "tile" + std::to_string(c.tile) + "w" + std::to_string(c.warps),
        2.0 * a.numel() * b.shape()[1],
        0,
        [&a, &b, c] { volatile auto x = gemm(a, b, c); },
    });
  }

  const auto rows = suite.run(cfg_.warmup, cfg_.iterations);
  TuneResult best;
  for (size_t i = 0; i < rows.size(); ++i) {
    const auto& c = configs[i];
    TuneResult r{c, rows[i].ms, rows[i].gflops, true};
    db_.record(std::get<0>(key), std::get<1>(key), std::get<2>(key), r);
    if (!best.valid || r.latency_ms < best.latency_ms) {
      best = r;
    }
  }

  if (cfg_.persist && !cfg_.cache_file.empty()) {
    db_.save(cfg_.cache_file);
  }

  std::ostringstream o;
  o << "tested=" << rows.size()
    << " best_ms=" << best.latency_ms
    << " best=" << kernel_config(best.config);
  last_report_ = o.str();
  return best;
}

std::string GemmAutoTuner::report() const {
  return last_report_;
}

}  // namespace gpuforge
