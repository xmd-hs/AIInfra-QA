#include "gpuforge/pipeline.hpp"
#include "gpuforge/shape.hpp"

#include <sstream>

namespace gpuforge::compiler {

CompilePipeline::CompilePipeline() = default;

void CompilePipeline::enable_fusion(bool enabled) { fusion_ = enabled; }
void CompilePipeline::enable_layout(bool enabled) { layout_ = enabled; }
void CompilePipeline::set_tuning(TuneDatabase* database) { tuning_ = database; }

std::string CompileReport::text() const {
  std::ostringstream output;
  for (const std::string& diagnostic : diagnostics) {
    output << "diagnostic: " << diagnostic << "\n";
  }
  for (const LaunchPlan& launch : launches) {
    output << "node=" << launch.node << " kernel=" << launch.kernel
           << " " << launch.schedule.str()
           << " shared=" << launch.shared_bytes
           << " threads=" << launch.threads
           << " est_ms=" << launch.estimated_ms << "\n";
  }
  return output.str();
}

CompileReport CompilePipeline::run(Module& module) {
  CompileReport report;
  // Validate the graph before mutating it or producing launch plans.  A
  // compiler report must never describe executable work for malformed IR.
  std::vector<std::string> errors;
  if (!validate(module, &errors)) {
    report.diagnostics = std::move(errors);
    return report;
  }
  if (layout_) passes_.infer_layout(module);
  passes_.canonicalize(module);

  std::vector<FusionGroup> groups;
  if (fusion_) {
    groups = passes_.fuse(module);
  } else {
    for (const Node& node : module.nodes()) {
      groups.push_back({{node.id}, "node_" + std::to_string(node.id), 1024});
    }
  }

  for (const FusionGroup& group : groups) {
    if (group.nodes.empty()) continue;
    const Node& root = module.node(group.nodes.front());
    Schedule schedule = passes_.schedule(root);

    if (tuning_ && root.output.shape.dims.size() == 2) {
      const auto& dims = root.output.shape.dims;
      const TuneResult best = tuning_->best(dims[0], dims[1], dims.back());
      if (best.valid) {
        schedule.block_m = best.config.tile * 8;
        schedule.block_n = best.config.tile * 8;
        schedule.warps = best.config.warps;
        schedule.tensor_core = best.config.use_tensor_core;
      }
    }

    LaunchPlan launch;
    launch.node = root.id;
    launch.kernel = group.kernel;
    launch.schedule = schedule;
    launch.threads = static_cast<size_t>(schedule.warps) * 32;
    launch.shared_bytes = static_cast<size_t>(schedule.block_k) *
                          (schedule.block_m + schedule.block_n) * 2;
    launch.estimated_ms = cost_.estimate_ms(root, schedule);
    report.launches.push_back(std::move(launch));

    if (!cost_.profitable(group)) {
      report.diagnostics.push_back("small fusion group at node " +
                                   std::to_string(root.id));
    }
  }
  return report;
}

}  // namespace gpuforge::compiler
