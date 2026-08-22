#pragma once

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <vector>

namespace gpuforge {

enum class DType { F32, F16, BF16, I8 };

class Tensor {
 public:
  Tensor() = default;
  explicit Tensor(std::vector<size_t> shape, DType dtype = DType::F32);
  Tensor(std::initializer_list<size_t> shape, DType dtype = DType::F32);

  size_t numel() const;
  size_t ndim() const { return shape_.size(); }
  const std::vector<size_t>& shape() const { return shape_; }
  DType dtype() const { return dtype_; }

  float* data();
  const float* data() const;
  float& at(size_t i);
  const float& at(size_t i) const;

  void fill(float v);
  Tensor clone() const;
  void zeros();

 private:
  std::vector<size_t> shape_;
  DType dtype_{DType::F32};
  std::shared_ptr<float[]> storage_;
};

}  // namespace gpuforge
