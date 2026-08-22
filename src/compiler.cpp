#include "gpuforge/compiler.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

namespace gpuforge::compiler {

namespace {

const char* scalar_name(ScalarType type) {
  switch (type) {
    case ScalarType::F32: return "f32";
    case ScalarType::F16: return "f16";
    case ScalarType::BF16: return "bf16";
    case ScalarType::I8: return "i8";
  }
  return "unknown";
}

const char* op_name(Op op) {
  switch (op) {
    case Op::Parameter: return "parameter";
    case Op::Constant: return "constant";
    case Op::MatMul: return "matmul";
    case Op::Add: return "add";
    case Op::Mul: return "mul";
    case Op::Relu: return "relu";
    case Op::Gelu: return "gelu";
    case Op::LayerNorm: return "layer_norm";
    case Op::Softmax: return "softmax";
    case Op::ReduceSum: return "reduce_sum";
    case Op::Cast: return "cast";
    case Op::Reshape: return "reshape";
  }
  return "unknown";
}

bool is_elementwise(Op op) {
  return op == Op::Add || op == Op::Mul || op == Op::Relu ||
         op == Op::Gelu || op == Op::Cast;
}

}  // namespace

size_t Shape::elements() const {
  if (dims.empty()) return 0;
  size_t result = 1;
  for (const int64_t dimension : dims) {
    if (dimension < 0) return 0;
    const size_t value = static_cast<size_t>(dimension);
    if (value != 0 && result > std::numeric_limits<size_t>::max() / value) return 0;
    result *= value;
  }
  return result;
}

std::string Shape::str() const {
  std::ostringstream stream;
  stream << "[";
  for (size_t i = 0; i < dims.size(); ++i) {
    if (i != 0) stream << ",";
    stream << dims[i];
  }
  stream << "]:" << scalar_name(type);
  return stream.str();
}

bool Shape::operator==(const Shape& other) const {
  return dims == other.dims && type == other.type && layout == other.layout;
}

int Module::parameter(std::string name, Shape shape) {
  return emit(Op::Parameter, {}, std::move(shape), {}, std::move(name));
}

int Module::constant(std::string name, Shape shape) {
  return emit(Op::Constant, {}, std::move(shape), {}, std::move(name));
}

int Module::constant(std::string name, Shape shape, std::vector<float> data) {
  const int id = constant(std::move(name), std::move(shape));
  node(id).constant_data = std::move(data);
  if (!node(id).constant_data.empty() &&
      node(id).constant_data.size() != node(id).output.shape.elements()) {
    throw std::invalid_argument("constant payload size mismatch");
  }
  return id;
}

int Module::emit(Op op, std::vector<int> inputs, Shape shape,
                 std::map<std::string, int64_t> attrs, std::string name) {
  const int id = static_cast<int>(nodes_.size());
  if (name.empty()) name = "v" + std::to_string(id);
  Node node;
  node.id = id;
  node.op = op;
  node.inputs = std::move(inputs);
  node.output = Value{id, std::move(shape), std::move(name)};
  node.attrs = std::move(attrs);
  nodes_.push_back(std::move(node));
  return id;
}

Node& Module::node(int id) { return nodes_.at(static_cast<size_t>(id)); }
const Node& Module::node(int id) const { return nodes_.at(static_cast<size_t>(id)); }
const std::vector<Node>& Module::nodes() const { return nodes_; }
std::vector<Node>& Module::mutable_nodes() { return nodes_; }

std::vector<int> Module::users(int value_id) const {
  std::vector<int> result;
  for (const Node& node : nodes_) {
    if (std::find(node.inputs.begin(), node.inputs.end(), value_id) != node.inputs.end()) {
      result.push_back(node.id);
    }
  }
  return result;
}

void Module::set_outputs(std::vector<int> outputs) { outputs_ = std::move(outputs); }
const std::vector<int>& Module::outputs() const { return outputs_; }

std::string Module::print() const {
  std::ostringstream stream;
  for (const Node& node : nodes_) {
    stream << node.id << " " << node.output.name << " = " << op_name(node.op)
           << " <- ";
    for (const int input : node.inputs) stream << "%" << input << " ";
    stream << node.output.shape.str() << "\n";
  }
  return stream.str();
}

void PassManager::constant_fold(Module& module) {
  for (Node& node : module.mutable_nodes()) {
    std::vector<float> folded;
    if (node.op == Op::Add || node.op == Op::Mul) {
      if (node.inputs.size() != 2) continue;
      const Node& lhs = module.node(node.inputs[0]);
      const Node& rhs = module.node(node.inputs[1]);
      if (lhs.op != Op::Constant || rhs.op != Op::Constant ||
          lhs.constant_data.empty() || lhs.constant_data.size() != rhs.constant_data.size()) continue;
      folded.resize(lhs.constant_data.size());
      for (size_t i = 0; i < folded.size(); ++i)
        folded[i] = node.op == Op::Add ? lhs.constant_data[i] + rhs.constant_data[i]
                                       : lhs.constant_data[i] * rhs.constant_data[i];
    } else if (node.op == Op::Relu || node.op == Op::Gelu || node.op == Op::Cast ||
               node.op == Op::Reshape) {
      if (node.inputs.size() != 1) continue;
      const Node& input = module.node(node.inputs[0]);
      if (input.op != Op::Constant || input.constant_data.empty()) continue;
      folded = input.constant_data;
      if (node.op == Op::Relu) {
        for (float& value : folded) value = std::max(0.0f, value);
      } else if (node.op == Op::Gelu) {
        for (float& value : folded)
          value = 0.5f * value * (1.0f + std::erf(value / std::sqrt(2.0f)));
      }
    } else {
      continue;
    }
    node.op = Op::Constant;
    node.inputs.clear();
    node.constant_data = std::move(folded);
  }
}

void PassManager::infer_layout(Module& module) {
  auto& nodes = module.mutable_nodes();
  for (Node& node : nodes) {
    if (node.op == Op::MatMul && node.output.shape.dims.size() == 2) {
      node.output.shape.layout = Layout::Tiled16;
    }
  }
}

void PassManager::canonicalize(Module& module) {
  auto& nodes = module.mutable_nodes();
  // Algebraic rewrites must preserve values. x + x is not x * x; without
  // explicit scalar algebra support, this pass intentionally remains a no-op.
  (void)nodes;
}

size_t PassManager::dead_code_eliminate(Module& module) {
  if (module.outputs().empty()) return 0;  // Conservative when outputs are unspecified.
  std::vector<bool> live(module.nodes().size(), false);
  std::vector<int> worklist = module.outputs();
  while (!worklist.empty()) {
    const int id = worklist.back();
    worklist.pop_back();
    if (id < 0 || static_cast<size_t>(id) >= module.nodes().size() || live[id]) continue;
    live[id] = true;
    for (const int input : module.node(id).inputs) worklist.push_back(input);
  }
  size_t removed = 0;
  auto& nodes = module.mutable_nodes();
  std::vector<Node> kept;
  std::vector<int> remap(nodes.size(), -1);
  kept.reserve(nodes.size());
  for (const Node& node : nodes) {
    if (live[node.id]) {
      remap[node.id] = static_cast<int>(kept.size());
      kept.push_back(node);
    } else ++removed;
  }
  for (Node& node : kept) {
    node.id = remap[node.id];
    node.output.id = node.id;
    for (int& input : node.inputs) input = remap[input];
  }
  nodes = std::move(kept);
  auto outputs = module.outputs();
  for (int& output : outputs) output = remap[output];
  module.set_outputs(std::move(outputs));
  return removed;
}

std::vector<FusionGroup> PassManager::fuse(Module& module) const {
  std::vector<FusionGroup> groups;
  std::vector<bool> visited(module.nodes().size(), false);
  for (const Node& root : module.nodes()) {
    if (visited[root.id]) continue;
    FusionGroup group;
    group.nodes.push_back(root.id);
    visited[root.id] = true;
    int current = root.id;
    while (module.users(current).size() == 1) {
      const int user_id = module.users(current).front();
      if (!is_elementwise(module.node(user_id).op) || visited[user_id]) break;
      group.nodes.push_back(user_id);
      visited[user_id] = true;
      current = user_id;
    }
    group.kernel = "fused_" + std::to_string(root.id);
    group.flops = group.nodes.size() * 1024;
    groups.push_back(std::move(group));
  }
  return groups;
}

Schedule PassManager::schedule(const Node& node) const {
  Schedule schedule;
  const auto& dims = node.output.shape.dims;
  if (dims.size() == 2 && dims[1] >= 1024) {
    schedule.block_m = 128;
    schedule.block_n = 256;
    schedule.warps = 8;
  }
  if (node.output.shape.type == ScalarType::F32) {
    schedule.tensor_core = false;
  }
  return schedule;
}

std::string PassManager::emit_cuda(const Module& module, const FusionGroup& group,
                                   const Schedule& schedule) const {
  std::ostringstream code;
  code << "// generated by GPUForge\n// nodes: ";
  for (const int id : group.nodes) code << id << " ";
  code << "\n// schedule " << schedule.str() << "\n";
  const Node& node = module.node(group.nodes.front());
  if (node.op != Op::Relu && node.op != Op::Gelu && node.op != Op::Add &&
      node.op != Op::Mul) {
    throw std::invalid_argument("CUDA codegen unsupported operator");
  }
  code << "extern \"C\" __global__ void " << group.kernel
       << "(const float* input, const float* rhs, float* output, int n) {\n"
       << "  const int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
       << "  if (i >= n) return;\n";
  if (node.op == Op::Relu)
    code << "  output[i] = input[i] > 0.0f ? input[i] : 0.0f;\n";
  else if (node.op == Op::Add)
    code << "  output[i] = input[i] + rhs[i];\n";
  else if (node.op == Op::Mul)
    code << "  output[i] = input[i] * rhs[i];\n";
  else
    code << "  output[i] = 0.5f * input[i] * (1.0f + tanhf(0.79788456f * (input[i] + 0.044715f * input[i] * input[i] * input[i])));\n";
  code << "}\n";
  return code.str();
}

std::string Schedule::str() const {
  return "bm=" + std::to_string(block_m) +
         ",bn=" + std::to_string(block_n) +
         ",bk=" + std::to_string(block_k) +
         ",warps=" + std::to_string(warps) +
         (tensor_core ? ",tc" : "") + (async_copy ? ",async" : "");
}

double CostModel::arithmetic_intensity(const Shape& a, const Shape& b,
                                       const Shape& c) const {
  if (a.dims.size() != 2 || b.dims.size() != 2 || c.dims.size() != 2) return 0;
  const double flops = 2.0 * a.dims[0] * b.dims[1] * a.dims[1];
  const double element_bytes = a.type == ScalarType::F32 ? 4.0 : 2.0;
  const double bytes = (a.elements() + b.elements() + c.elements()) * element_bytes;
  return bytes == 0 ? 0 : flops / bytes;
}

double CostModel::estimate_ms(const Node& node, const Schedule& schedule,
                              double tflops, double bandwidth) const {
  const auto& dims = node.output.shape.dims;
  if (dims.empty()) return 0;
  const double work = static_cast<double>(node.output.shape.elements()) *
                      (node.op == Op::MatMul && dims.size() > 1 ? 2 * dims.back() : 10);
  const double compute_rate = tflops * 1e12;
  const double memory_rate = bandwidth * 1e9 * std::max(1, schedule.block_k);
  return work / std::min(compute_rate, memory_rate) * 1e3;
}

bool CostModel::profitable(const FusionGroup& group) const {
  return group.nodes.size() > 1 || group.flops > 4096;
}

}  // namespace gpuforge::compiler
