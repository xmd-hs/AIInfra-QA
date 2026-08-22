#pragma once

#include "kernels.hpp"

#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace gpuforge {

struct TuneResult {
  GemmConfig config;
  double latency_ms = 0;
  double throughput = 0;
  bool valid = false;
};

class TuneDatabase {
 public:
  void record(size_t m, size_t n, size_t k, TuneResult);
  TuneResult best(size_t m, size_t n, size_t k) const;
  std::vector<TuneResult> candidates(size_t m, size_t n, size_t k) const;
  std::string serialize() const;
  bool save(const std::string&) const;
  bool load(const std::string&);

 private:
  std::map<std::tuple<size_t, size_t, size_t>, std::vector<TuneResult>> table_;
};

class Searcher {
 public:
  static std::vector<GemmConfig> gemm_space(size_t m, size_t n, size_t k);
  TuneResult select(const std::vector<TuneResult>&) const;
};

}  // namespace gpuforge
