#include "gpuforge/tuning.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>

namespace gpuforge {

void TuneDatabase::record(size_t m, size_t n, size_t k, TuneResult result) {
  result.valid = true;
  auto& entries = table_[{m, n, k}];
  entries.push_back(result);
  std::sort(entries.begin(), entries.end(),
            [](const TuneResult& lhs, const TuneResult& rhs) {
              return lhs.latency_ms < rhs.latency_ms;
            });
  if (entries.size() > 32) entries.resize(32);
}

TuneResult TuneDatabase::best(size_t m, size_t n, size_t k) const {
  const auto iterator = table_.find({m, n, k});
  if (iterator == table_.end() || iterator->second.empty()) return {};
  return iterator->second.front();
}

std::vector<TuneResult> TuneDatabase::candidates(size_t m, size_t n,
                                                 size_t k) const {
  const auto iterator = table_.find({m, n, k});
  return iterator == table_.end() ? std::vector<TuneResult>{}
                                   : iterator->second;
}

std::string TuneDatabase::serialize() const {
  std::ostringstream output;
  for (const auto& [key, results] : table_) {
    for (const TuneResult& result : results) {
      output << std::get<0>(key) << ',' << std::get<1>(key) << ','
             << std::get<2>(key) << ',' << result.config.tile << ','
             << result.config.warps << ',' << result.config.use_tensor_core
             << ',' << result.latency_ms << ',' << result.throughput << '\n';
    }
  }
  return output.str();
}

bool TuneDatabase::save(const std::string& path) const {
  std::ofstream file(path);
  if (!file) return false;
  file << serialize();
  return static_cast<bool>(file);
}

bool TuneDatabase::load(const std::string& path) {
  std::ifstream file(path);
  if (!file) return false;
  std::string line;
  while (std::getline(file, line)) {
    std::stringstream input(line);
    size_t m, n, k;
    int tensor_core;
    char separator;
    TuneResult result;
    if (!(input >> m >> separator >> n >> separator >> k >> separator >>
          result.config.tile >> separator >> result.config.warps >> separator >>
          tensor_core >> separator >> result.latency_ms >> separator >>
          result.throughput)) continue;
    result.config.use_tensor_core = tensor_core != 0;
    record(m, n, k, result);
  }
  return true;
}

std::vector<GemmConfig> Searcher::gemm_space(size_t m, size_t n, size_t k) {
  std::vector<GemmConfig> candidates;
  for (const int tile : {8, 16, 32}) {
    for (const int warps : {2, 4, 8}) {
      for (const bool tensor_core : {false, true}) {
        if (tensor_core && k % 16 != 0) continue;
        if (m % tile != 0 && n % tile != 0) continue;
        candidates.push_back({tile, warps, tensor_core});
      }
    }
  }
  return candidates;
}

TuneResult Searcher::select(const std::vector<TuneResult>& results) const {
  const auto iterator = std::min_element(
      results.begin(), results.end(), [](const TuneResult& lhs, const TuneResult& rhs) {
        if (!lhs.valid) return false;
        if (!rhs.valid) return true;
        return lhs.latency_ms < rhs.latency_ms;
      });
  return iterator == results.end() ? TuneResult{} : *iterator;
}

}  // namespace gpuforge
