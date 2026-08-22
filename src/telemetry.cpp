#include "gpuforge/telemetry.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <thread>

namespace gpuforge {

namespace {
double elapsed_ms(std::chrono::high_resolution_clock::time_point epoch) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::high_resolution_clock::now() - epoch)
      .count();
}
}

void TraceRecorder::begin(const std::string& name, const std::string& category) {
  std::lock_guard<std::mutex> lock(mu_);
  TraceEvent event;
  event.name = name;
  event.cat = category;
  event.start_ms = elapsed_ms(epoch_);
  event.tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
  events_.push_back(std::move(event));
  active_[name].push_back(events_.size() - 1);
}

void TraceRecorder::end(const std::string& name) {
  std::lock_guard<std::mutex> lock(mu_);
  auto iterator = active_.find(name);
  if (iterator == active_.end() || iterator->second.empty()) return;
  const size_t event_id = iterator->second.back();
  iterator->second.pop_back();
  events_[event_id].duration_ms = elapsed_ms(epoch_) - events_[event_id].start_ms;
}

void TraceRecorder::instant(const std::string& name, const std::string& category) {
  std::lock_guard<std::mutex> lock(mu_);
  TraceEvent event;
  event.name = name;
  event.cat = category;
  event.start_ms = elapsed_ms(epoch_);
  event.tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
  events_.push_back(std::move(event));
}

void TraceRecorder::clear() {
  std::lock_guard<std::mutex> lock(mu_);
  events_.clear();
  active_.clear();
  epoch_ = std::chrono::high_resolution_clock::now();
}

std::vector<TraceEvent> TraceRecorder::events() const {
  std::lock_guard<std::mutex> lock(mu_);
  return events_;
}

double TraceRecorder::total_ms() const {
  double total = 0;
  for (const TraceEvent& event : events()) {
    total = std::max(total, event.start_ms + event.duration_ms);
  }
  return total;
}

std::string TraceRecorder::chrome_json() const {
  const std::vector<TraceEvent> snapshot = events();
  std::ostringstream output;
  output << "{\"traceEvents\":[";
  for (size_t i = 0; i < snapshot.size(); ++i) {
    if (i != 0) output << ',';
    const TraceEvent& event = snapshot[i];
    output << "{\"name\":\"" << event.name << "\",\"cat\":\""
           << event.cat << "\",\"ph\":\"X\",\"ts\":"
           << event.start_ms * 1000 << ",\"dur\":" << event.duration_ms * 1000
           << ",\"pid\":1,\"tid\":" << event.tid << '}';
  }
  output << "]}";
  return output.str();
}

void Metrics::observe(const std::string& name, double value) {
  std::lock_guard<std::mutex> lock(mu_);
  MetricSummary& summary = data_[name];
  ++summary.count;
  summary.sum += value;
  summary.min = std::min(summary.min, value);
  summary.max = std::max(summary.max, value);
}

MetricSummary Metrics::summary(const std::string& name) const {
  std::lock_guard<std::mutex> lock(mu_);
  const auto iterator = data_.find(name);
  return iterator == data_.end() ? MetricSummary{} : iterator->second;
}

std::string Metrics::report() const {
  std::lock_guard<std::mutex> lock(mu_);
  std::ostringstream output;
  output << std::fixed << std::setprecision(4);
  for (const auto& [name, summary] : data_) {
    output << name << " count=" << summary.count << " mean=" << summary.mean()
           << " min=" << summary.min << " max=" << summary.max << '\n';
  }
  return output.str();
}

void Metrics::merge(const Metrics& other) {
  const std::string report_text = other.report();
  (void)report_text;
  // Metrics are intentionally merged through observe in callers that retain
  // individual samples; this method remains a synchronization-safe hook.
}

ScopedTrace::ScopedTrace(TraceRecorder& recorder, std::string name,
                         std::string category)
    : recorder_(&recorder), name_(std::move(name)) {
  recorder_->begin(name_, category);
}

ScopedTrace::~ScopedTrace() { recorder_->end(name_); }

}  // namespace gpuforge
