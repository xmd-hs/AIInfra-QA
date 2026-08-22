#include "gpuforge/tensor.hpp"

#include <algorithm>
#include <utility>

namespace gpuforge {

Tensor::Tensor(std::vector<size_t> shape, DType dtype)
    : shape_(std::move(shape)), dtype_(dtype) {
  const size_t elements = numel();
  if (elements != 0) {
    storage_ = std::shared_ptr<float[]>(new float[elements]());
  }
}

Tensor::Tensor(std::initializer_list<size_t> shape, DType dtype)
    : Tensor(std::vector<size_t>(shape), dtype) {}

size_t Tensor::numel() const {
  if (shape_.empty()) return 0;
  size_t result = 1;
  for (const size_t dimension : shape_) result *= dimension;
  return result;
}

float* Tensor::data() { return storage_.get(); }
const float* Tensor::data() const { return storage_.get(); }

float& Tensor::at(size_t index) {
  if (index >= numel()) throw std::out_of_range("tensor index");
  return data()[index];
}

const float& Tensor::at(size_t index) const {
  if (index >= numel()) throw std::out_of_range("tensor index");
  return data()[index];
}

void Tensor::fill(float value) { std::fill(data(), data() + numel(), value); }
void Tensor::zeros() { fill(0.0f); }

Tensor Tensor::clone() const {
  Tensor copy(shape_, dtype_);
  std::copy(data(), data() + numel(), copy.data());
  return copy;
}

}  // namespace gpuforge
