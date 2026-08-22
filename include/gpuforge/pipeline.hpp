#pragma once

#include "compiler.hpp"
#include "tuning.hpp"

#include <string>
#include <vector>

namespace gpuforge::compiler {

struct LaunchPlan {
  int node = -1;
  std::string kernel;
  Schedule schedule;
  size_t shared_bytes = 0;
  size_t threads = 0;
  double estimated_ms = 0;
};

struct CompileReport {
  std::vector<LaunchPlan> launches;
  std::vector<std::string> diagnostics;

  std::string text() const;
};

class CompilePipeline {
 public:
  CompilePipeline();
  CompileReport run(Module&);

  void enable_fusion(bool);
  void enable_layout(bool);
  void set_tuning(TuneDatabase*);

 private:
  PassManager passes_;
  CostModel cost_;
  TuneDatabase* tuning_ = nullptr;
  bool fusion_ = true;
  bool layout_ = true;
};

}  // namespace gpuforge::compiler
