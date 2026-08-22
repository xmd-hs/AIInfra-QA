#include "gpuforge/shape.hpp"

#include <algorithm>

namespace gpuforge::compiler {

ShapeResult infer_matmul(const Shape& lhs, const Shape& rhs) {
  if (lhs.dims.size() != 2 || rhs.dims.size() != 2) {
    return {false, {}, "matmul expects rank-2 shapes"};
  }
  if (lhs.dims[1] != rhs.dims[0]) {
    return {false, {}, "matmul inner dimensions mismatch"};
  }
  return {true, {{lhs.dims[0], rhs.dims[1]}, lhs.type, Layout::Tiled16}, {}};
}

ShapeResult infer_broadcast(const Shape& lhs, const Shape& rhs) {
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
  size_t known_elements = 1;
  int inferred_axis = -1;
  for (size_t axis = 0; axis < dims.size(); ++axis) {
    if (dims[axis] < 0) {
      if (inferred_axis >= 0) return {false, {}, "multiple inferred dimensions"};
      inferred_axis = static_cast<int>(axis);
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
