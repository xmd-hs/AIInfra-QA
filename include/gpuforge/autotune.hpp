#pragma once

#include "benchmark.hpp"
#include "tuning.hpp"

#include <string>

namespace gpuforge {

struct AutoTuneConfig {
  int warmup = 2;
  int iterations = 10;
  bool persist = true;
  std::string cache_file;
};

class GemmAutoTuner {
 public:
  explicit GemmAutoTuner(AutoTuneConfig = {});

  TuneResult run(const Tensor&, const Tensor&);
  const TuneDatabase& database() const { return db_; }
  std::string report() const;

 private:
  AutoTuneConfig cfg_;
  TuneDatabase db_;
  std::string last_report_;
};

}  // namespace gpuforge
