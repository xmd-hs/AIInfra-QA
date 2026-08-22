#pragma once

#include "compiler.hpp"

#include <string>
#include <vector>

namespace gpuforge::compiler {

struct ShapeResult {
  bool ok = false;
  Shape shape;
  std::string error;
};

ShapeResult infer_matmul(const Shape&, const Shape&);
ShapeResult infer_broadcast(const Shape&, const Shape&);
ShapeResult infer_reshape(const Shape&, const std::vector<int64_t>&);
ShapeResult infer_reduce(const Shape&, int axis, bool keepdim);
bool validate(const Module&, std::vector<std::string>* errors = nullptr);

}  // namespace gpuforge::compiler
