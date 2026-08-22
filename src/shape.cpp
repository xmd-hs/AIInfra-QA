#include "gpuforge/shape.hpp"

#include <algorithm>

namespace gpuforge::compiler {

namespace { bool valid_dims(const std::vector<int64_t>& dims) {
  return std::all_of(dims.begin(), dims.end(), [](int64_t d) { return d >= 0; });
} }

ShapeResult infer_matmul(const Shape& lhs, const Shape& rhs) {
  if (lhs.dims.size() != 2 || rhs.dims.size() != 2 || !valid_dims(lhs.dims) || !valid_dims(rhs.dims)) {
    return {false, {}, "matmul expects rank-2 shapes"};
  }
  if (lhs.dims[1] != rhs.dims[0]) {
    return {false, {}, "matmul inner dimensions mismatch"};
  }
  return {true, {{lhs.dims[0], rhs.dims[1]}, lhs.type, Layout::Tiled16}, {}};
}

ShapeResult infer_broadcast(const Shape& lhs, const Shape& rhs) {
  if (!valid_dims(lhs.dims) || !valid_dims(rhs.dims)) return {false, {}, "broadcast dimensions must be non-negative"};
  const size_t rank = std::max(lhs.dims.size(), rhs.dims.size());
  std::vector<int64_t> dimensions(rank, 1);
  for (size_t reverse = 0; reverse < rank; ++reverse) {
    const int64_t left = reverse < lhs.dims.size()
                             ? lhs.dims[lhs.dims.size() - reverse - 1]
                             : 1;
    const int64_t right = reverse < rhs.dims.size()
                              ? rhs.dims[rhs.dims.size() - reverse - 1]
                              : 1;
    if (left != right && left != 1 && right != 1) {
      return {false, {}, "broadcast dimensions mismatch"};
    }
    dimensions[rank - reverse - 1] = std::max(left, right);
  }
  return {true, {dimensions, lhs.type, lhs.layout}, {}};
}

ShapeResult infer_reshape(const Shape& input, const std::vector<int64_t>& dims) {
  if (!valid_dims(input.dims)) return {false, {}, "input dimensions must be non-negative"};
  size_t known_elements = 1;
  int inferred_axis = -1;
  for (size_t axis = 0; axis < dims.size(); ++axis) {
    if (dims[axis] == -1) {
      if (inferred_axis >= 0) return {false, {}, "multiple inferred dimensions"};
      inferred_axis = static_cast<int>(axis);
    } else if (dims[axis] < 0) {
      return {false, {}, "reshape dimensions must be non-negative or -1"};
    } else {
      known_elements *= static_cast<size_t>(dims[axis]);
    }
  }
  if (known_elements == 0 || input.elements() % known_elements != 0) {
    return {false, {}, "reshape element count mismatch"};
  }
  std::vector<int64_t> result = dims;
  if (inferred_axis >= 0) result[inferred_axis] = input.elements() / known_elements;
  if (input.elements() != known_elements && inferred_axis < 0) {
    return {false, {}, "reshape element count mismatch"};
  }
  return {true, {result, input.type, input.layout}, {}};
}

ShapeResult infer_reduce(const Shape& input, int axis, bool keep_dimension) {
  if (axis < 0) axis += static_cast<int>(input.dims.size());
  if (axis < 0 || axis >= static_cast<int>(input.dims.size())) {
    return {false, {}, "reduce axis out of range"};
  }
  std::vector<int64_t> result = input.dims;
  if (keep_dimension) result[axis] = 1;
  else result.erase(result.begin() + axis);
  return {true, {result, input.type, input.layout}, {}};
}

bool validate(const Module& module, std::vector<std::string>* errors) {
  bool valid = true;
  auto report = [&](const std::string& message) {
    valid = false;
    if (errors) errors->push_back(message);
  };
  for (const Node& node : module.nodes()) {
    size_t expected_inputs = 0;
    bool arity_checked = true;
    switch (node.op) {
      case Op::Parameter:
      case Op::Constant: expected_inputs = 0; break;
      case Op::Relu:
      case Op::Gelu:
      case Op::Softmax:
      case Op::ReduceSum:
      case Op::Cast:
      case Op::Reshape: expected_inputs = 1; break;
      case Op::MatMul:
      case Op::Add:
      case Op::Mul: expected_inputs = 2; break;
      case Op::LayerNorm: expected_inputs = 3; break;
      default: arity_checked = false; break;
    }
    if (arity_checked && node.inputs.size() != expected_inputs) {
      report("node " + std::to_string(node.id) + " has " +
             std::to_string(node.inputs.size()) + " inputs; expected " +
             std::to_string(expected_inputs));
    }
    for (const int input : node.inputs) {
      if (input < 0 || input >= node.id) {
        report("node " + std::to_string(node.id) + " has invalid input " +
               std::to_string(input));
      }
    }
    if (node.op == Op::MatMul && node.inputs.size() == 2 &&
        node.inputs[0] >= 0 && node.inputs[0] < node.id &&
        node.inputs[1] >= 0 && node.inputs[1] < node.id) {
      const ShapeResult inferred = infer_matmul(
          module.node(node.inputs[0]).output.shape,
          module.node(node.inputs[1]).output.shape);
      if (!inferred.ok) report("matmul node " + std::to_string(node.id) + ": " + inferred.error);
      else if (!(node.output.shape.dims == inferred.shape.dims))
        report("matmul node " + std::to_string(node.id) + ": output shape mismatch");
    }
    if ((node.op == Op::Add || node.op == Op::Mul) && node.inputs.size() == 2 &&
        node.inputs[0] >= 0 && node.inputs[0] < node.id &&
        node.inputs[1] >= 0 && node.inputs[1] < node.id) {
      const ShapeResult inferred = infer_broadcast(
          module.node(node.inputs[0]).output.shape,
          module.node(node.inputs[1]).output.shape);
      if (!inferred.ok) report("elementwise node " + std::to_string(node.id) + ": " + inferred.error);
      else if (node.output.shape.dims != inferred.shape.dims)
        report("elementwise node " + std::to_string(node.id) + ": output shape mismatch");
    }
  }
  return valid;
}

}  // namespace gpuforge::compiler
