#pragma once

#include "tensor.hpp"

#include <mutex>
#include <unordered_map>
#include <vector>

namespace gpuforge {

class KVCache {
 public:
  KVCache(size_t layers, size_t heads, size_t head_dim, size_t capacity);

  void append(size_t layer, size_t token, const Tensor& key, const Tensor& value);
  std::pair<Tensor, Tensor> get(size_t layer, size_t begin, size_t end) const;
  size_t capacity() const { return capacity_; }
  size_t used() const;
  void reset();

 private:
  size_t layers_;
  size_t heads_;
  size_t dim_;
  size_t capacity_;
  mutable std::mutex mu_;
  std::vector<Tensor> keys_;
  std::vector<Tensor> values_;
  std::vector<bool> valid_;
};

}  // namespace gpuforge
