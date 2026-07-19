#include "aiinfra/observability/metrics.h"

#include <algorithm>

namespace aiinfra::observability {

void MetricsRegistry::increment(const std::string& name, std::uint64_t value) {
  std::lock_guard<std::mutex> lock(mutex_);
  counters_[name] += value;
}

void MetricsRegistry::observe_us(const std::string& name, std::chrono::microseconds value) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto& histogram = histograms_[name];
  const auto us = static_cast<std::uint64_t>(std::max<std::int64_t>(0, value.count()));
  histogram.count += 1;
  histogram.sum_us += us;
  histogram.max_us = std::max(histogram.max_us, us);
}

void MetricsRegistry::set_gauge(const std::string& name, double value) {
  std::lock_guard<std::mutex> lock(mutex_);
  gauges_[name] = value;
}

std::uint64_t MetricsRegistry::counter(const std::string& name) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto it = counters_.find(name);
  return it == counters_.end() ? 0 : it->second;
}

double MetricsRegistry::gauge(const std::string& name) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto it = gauges_.find(name);
  return it == gauges_.end() ? 0.0 : it->second;
}

std::string MetricsRegistry::to_prometheus() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::ostringstream out;
  for (const auto& [name, value] : counters_) {
    out << name << " " << value << "\n";
  }
  for (const auto& [name, histogram] : histograms_) {
    out << name << "_count " << histogram.count << "\n";
    out << name << "_sum_us " << histogram.sum_us << "\n";
    out << name << "_max_us " << histogram.max_us << "\n";
  }
  for (const auto& [name, value] : gauges_) {
    out << name << " " << value << "\n";
  }
  return out.str();
}

ScopedTimer::ScopedTimer(MetricsRegistry& metrics, std::string name)
    : metrics_(metrics), name_(std::move(name)), start_(std::chrono::steady_clock::now()) {}

ScopedTimer::~ScopedTimer() {
  const auto end = std::chrono::steady_clock::now();
  metrics_.observe_us(name_, std::chrono::duration_cast<std::chrono::microseconds>(end - start_));
}

}  // namespace aiinfra::observability
