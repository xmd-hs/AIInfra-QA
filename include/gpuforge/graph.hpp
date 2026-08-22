#pragma once

#include "kernels.hpp"

#include <functional>
#include <string>
#include <vector>

namespace gpuforge {

enum class OpKind { Input, Gemm, LayerNorm, Softmax, Attention, FusedGemmBias };

struct GraphNode {
  int id = -1;
  OpKind op = OpKind::Input;
  std::vector<int> inputs;
  std::string name;
};

class Graph {
 public:
  int input(const std::string& name);
  int add(OpKind op, std::vector<int> inputs, const std::string& name);
  void fuse_linear_bias();
  bool validate(std::string* error = nullptr) const;
  const std::vector<GraphNode>& nodes() const { return nodes_; }
  std::string dump() const;

 private:
  std::vector<GraphNode> nodes_;
};

}  // namespace gpuforge
