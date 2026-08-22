#include "gpuforge/kv_cache.hpp"

#include <algorithm>
#include <stdexcept>

namespace gpuforge {

KVCache::KVCache(size_t layers, size_t heads, size_t head_dim,
                 size_t capacity)
    : layers_(layers), heads_(heads), dim_(head_dim), capacity_(capacity),
      valid_(layers * capacity, false) {
  keys_.reserve(layers * capacity);
  values_.reserve(layers * capacity);
  for (size_t i = 0; i < layers * capacity; ++i) {
    keys_.emplace_back(Tensor({heads, head_dim}));
    values_.emplace_back(Tensor({heads, head_dim}));
  }
}

void KVCache::append(size_t layer, size_t token, const Tensor& key,
                     const Tensor& value) {
  if (layer >= layers_ || token >= capacity_ ||
      key.numel() != heads_ * dim_ || value.numel() != key.numel()) {
    throw std::out_of_range("KVCache append: invalid layer, token or shape");
  }
  std::lock_guard<std::mutex> lock(mu_);
  const size_t index = layer * capacity_ + token;
  keys_[index] = key.clone();
  values_[index] = value.clone();
  valid_[index] = true;
}

std::pair<Tensor, Tensor> KVCache::get(size_t layer, size_t begin,
                                       size_t end) const {
  if (layer >= layers_ || begin > end || end > capacity_) {
    throw std::out_of_range("KVCache get: invalid range");
  }
  std::lock_guard<std::mutex> lock(mu_);
  const size_t token_count = end - begin;
  Tensor keys({token_count, heads_ * dim_});
  Tensor values(keys.shape());
  for (size_t token = begin; token < end; ++token) {
    const size_t index = layer * capacity_ + token;
    if (!valid_[index]) continue;
    const Tensor& source_key = keys_[index];
    const Tensor& source_value = values_[index];
    for (size_t element = 0; element < source_key.numel(); ++element) {
      const size_t output_index = (token - begin) * source_key.numel() + element;
      keys.at(output_index) = source_key.at(element);
      values.at(output_index) = source_value.at(element);
    }
  }
  return {std::move(keys), std::move(values)};
}

size_t KVCache::used() const {
  std::lock_guard<std::mutex> lock(mu_);
  return static_cast<size_t>(std::count(valid_.begin(), valid_.end(), true));
}

void KVCache::reset() {
  std::lock_guard<std::mutex> lock(mu_);
  std::fill(valid_.begin(), valid_.end(), false);
}

}  // namespace gpuforge
