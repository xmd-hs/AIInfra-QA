#include "gpuforge/executor.hpp"

#include <chrono>

namespace gpuforge::runtime {

Executor::Executor(Device device) : device_(device) {}

Stream Executor::create_stream() {
  const int id = next_stream_++;
  streams_[id] = true;
  return {id, device_};
}

void Executor::synchronize(Stream stream) {
  if (stream.id != 0) streams_[stream.id] = true;
}

ExecutionResult Executor::execute(const compiler::CompileReport& report,
                                  TraceRecorder* recorder) {
  const auto begin = std::chrono::high_resolution_clock::now();
  ExecutionResult result;
  result.ok = true;
  result.launches = report.launches.size();
  TraceRecorder local_recorder;
  TraceRecorder* trace = recorder != nullptr ? recorder : &local_recorder;

  for (const compiler::LaunchPlan& launch : report.launches) {
    ScopedTrace event(*trace, launch.kernel, "kernel");
    if (launch.threads == 0) {
      result.ok = false;
      result.error = "invalid launch dimensions";
      break;
    }
  }

  const auto end = std::chrono::high_resolution_clock::now();
  result.elapsed_ms = std::chrono::duration<double, std::milli>(end - begin).count();
  return result;
}

}  // namespace gpuforge::runtime
