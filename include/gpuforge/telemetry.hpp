#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace gpuforge {

struct TraceEvent {
  std::string name;
  std::string cat;
  double start_ms = 0;
  double duration_ms = 0;
  uint64_t tid = 0;
  std::map<std::string, std::string> args;
};

class TraceRecorder {
 public:
  void begin(const std::string&, const std::string& = {});
  void end(const std::string&);
  void instant(const std::string&, const std::string& = {});
  void clear();

  std::vector<TraceEvent> events() const;
  std::string chrome_json() const;
  double total_ms() const;

 private:
  std::chrono::high_resolution_clock::time_point epoch_ =
      std::chrono::high_resolution_clock::now();
  mutable std::mutex mu_;
  std::vector<TraceEvent> events_;
  std::map<std::string, std::vector<size_t>> active_;
};

struct MetricSummary {
  size_t count = 0;
  double sum = 0;
  double min = 1e30;
  double max = 0;

  double mean() const { return count ? sum / count : 0; }
};

class Metrics {
 public:
  void observe(const std::string&, double);
  MetricSummary summary(const std::string&) const;
  std::string report() const;
  void merge(const Metrics&);

 private:
  mutable std::mutex mu_;
  std::map<std::string, MetricSummary> data_;
};

class ScopedTrace {
 public:
  ScopedTrace(TraceRecorder&, std::string, std::string cat = {});
  ~ScopedTrace();

 private:
  TraceRecorder* recorder_;
  std::string name_;
};

}  // namespace gpuforge
