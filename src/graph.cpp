#include "gpuforge/graph.hpp"

#include <sstream>

namespace gpuforge {

int Graph::input(const std::string& name) {
  return add(OpKind::Input, {}, name);
}

int Graph::add(OpKind operation, std::vector<int> inputs,
               const std::string& name) {
  const int id = static_cast<int>(nodes_.size());
  nodes_.push_back({id, operation, std::move(inputs), name});
  return id;
}

void Graph::fuse_linear_bias() {
  if (!validate()) return;
  for (GraphNode& node : nodes_) {
    if (node.op != OpKind::Gemm) continue;
    for (const GraphNode& candidate : nodes_) {
      if (candidate.op != OpKind::Input ||
          candidate.name.find("bias") == std::string::npos) continue;
      if (std::find(node.inputs.begin(), node.inputs.end(), candidate.id) != node.inputs.end()) continue;
      node.op = OpKind::FusedGemmBias;
      node.inputs.push_back(candidate.id);
      node.name += "_fused_bias";
      break;
    }
  }
}

bool Graph::validate(std::string* error) const {
  for (const GraphNode& node : nodes_) {
    for (const int input : node.inputs) {
      if (input < 0 || input >= node.id) {
        if (error) *error = "node " + std::to_string(node.id) +
                            " has invalid input " + std::to_string(input);
        return false;
      }
    }
  }
  return true;
}

std::string Graph::dump() const {
  std::ostringstream output;
  for (const GraphNode& node : nodes_) {
    output << node.id << ':' << node.name << " <- ";
    for (const int input : node.inputs) output << input << ' ';
    output << '\n';
  }
  return output.str();
}

}  // namespace gpuforge
