#pragma once

#include "tensor.hpp"

#include <array>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace gpuforge::compiler {

enum class ScalarType { F32, F16, BF16, I8 };

enum class Layout { RowMajor, ColMajor, Tiled16, Tiled32 };

enum class Op {
  Parameter,
  Constant,
  MatMul,
  Add,
  Mul,
  Relu,
  Gelu,
  LayerNorm,
  Softmax,
  ReduceSum,
  Cast,
  Reshape,
};

struct Shape {
  std::vector<int64_t> dims;
  ScalarType type = ScalarType::F32;
  Layout layout = Layout::RowMajor;

  size_t elements() const;
  std::string str() const;
  bool operator==(const Shape&) const;
};

struct Value {
  int id = -1;
  Shape shape;
  std::string name;
};

struct Node {
  int id = -1;
  Op op = Op::Parameter;
  std::vector<int> inputs;
  Value output;
  std::map<std::string, int64_t> attrs;
  std::vector<float> constant_data;
};

class Module {
 public:
  int parameter(std::string, Shape);
  int constant(std::string, Shape);
  int constant(std::string, Shape, std::vector<float> data);
  int emit(Op,
           std::vector<int>,
           Shape,
           std::map<std::string, int64_t> attrs = {},
           std::string name = {});

  Node& node(int);
  const Node& node(int) const;
  const std::vector<Node>& nodes() const;
  std::vector<Node>& mutable_nodes();

  std::vector<int> users(int) const;
  void set_outputs(std::vector<int> outputs);
  const std::vector<int>& outputs() const;
  std::string print() const;

 private:
  std::vector<Node> nodes_;
  std::vector<int> outputs_;
};

struct FusionGroup {
  std::vector<int> nodes;
  std::string kernel;
  size_t flops = 0;
};

struct Schedule {
  int block_m = 128;
  int block_n = 128;
  int block_k = 32;
  int warps = 8;
  bool tensor_core = true;
  bool async_copy = true;

  std::string str() const;
};

class PassManager {
 public:
  void constant_fold(Module&);
  void infer_layout(Module&);
  void canonicalize(Module&);
  size_t dead_code_eliminate(Module&);
  std::vector<FusionGroup> fuse(Module&) const;
  Schedule schedule(const Node&) const;
  std::string emit_cuda(const Module&, const FusionGroup&, const Schedule&) const;
};

class CostModel {
 public:
  double arithmetic_intensity(const Shape&, const Shape&, const Shape&) const;
  double estimate_ms(const Node&,
                     const Schedule&,
                     double tflops = 20.0,
                     double bandwidth = 500.0) const;
  bool profitable(const FusionGroup&) const;
};

}  // namespace gpuforge::compiler
