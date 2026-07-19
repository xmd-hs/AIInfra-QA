#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>

namespace aiinfra::observability {

class MetricsRegistry {
 public:
  void increment(const std::string& name, std::uint64_t value = 1);
  void observe_us(const std::string& name, std::chrono::microseconds value);
  void set_gauge(const std::string& name, double value);

  std::uint64_t counter(const std::string& name) const;
  double gauge(const std::string& name) const;
  std::string to_prometheus() const;

 private:
  struct Histogram {
    std::uint64_t count{0};
    std::uint64_t sum_us{0};
    std::uint64_t max_us{0};
  };

  mutable std::mutex mutex_;
  std::unordered_map<std::string, std::uint64_t> counters_;
  std::unordered_map<std::string, Histogram> histograms_;
  std::unordered_map<std::string, double> gauges_;
};

class ScopedTimer {
 public:
  ScopedTimer(MetricsRegistry& metrics, std::string name);
  ~ScopedTimer();

 private:
  MetricsRegistry& metrics_;
  std::string name_;
  std::chrono::steady_clock::time_point start_;
};

}  // namespace aiinfra::observability
