#pragma once

#include "pipeline.hpp"
#include "telemetry.hpp"

#include <functional>
#include <map>
#include <string>
#include <vector>

namespace gpuforge::runtime {

enum class Device { CPU, CUDA };

struct Stream {
  int id = 0;
  Device device = Device::CPU;
};

struct ExecutionResult {
  bool ok = false;
  double elapsed_ms = 0;
  size_t launches = 0;
  std::string error;
};

class Executor {
 public:
  explicit Executor(Device d = Device::CPU);

  Stream create_stream();
  ExecutionResult execute(const compiler::CompileReport&,
                          TraceRecorder* trace = nullptr);
  void synchronize(Stream);
  Device device() const { return device_; }

 private:
  Device device_;
  int next_stream_ = 1;
  std::map<int, bool> streams_;
};

}  // namespace gpuforge::runtime
