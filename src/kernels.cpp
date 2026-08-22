#include "gpuforge/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace gpuforge {

namespace {
void check_matrix_multiply(const Tensor& lhs, const Tensor& rhs) {
  if (lhs.ndim() != 2 || rhs.ndim() != 2 || lhs.shape()[1] != rhs.shape()[0]) {
    throw std::invalid_argument("gemm requires compatible rank-2 tensors");
  }
}
}

Tensor gemm(const Tensor& lhs, const Tensor& rhs, const GemmConfig& config) {
  check_matrix_multiply(lhs, rhs);
  const size_t rows = lhs.shape()[0];
  const size_t inner = lhs.shape()[1];
  const size_t columns = rhs.shape()[1];
  const size_t tile = static_cast<size_t>(std::max(1, config.tile));
  Tensor output({rows, columns});
  output.zeros();

  for (size_t row_block = 0; row_block < rows; row_block += tile) {
    for (size_t column_block = 0; column_block < columns; column_block += tile) {
      for (size_t inner_block = 0; inner_block < inner; inner_block += tile) {
        const size_t row_end = std::min(row_block + tile, rows);
        const size_t column_end = std::min(column_block + tile, columns);
        const size_t inner_end = std::min(inner_block + tile, inner);
        for (size_t row = row_block; row < row_end; ++row) {
          for (size_t column = column_block; column < column_end; ++column) {
            float accumulator = output.at(row * columns + column);
            for (size_t index = inner_block; index < inner_end; ++index) {
              accumulator += lhs.at(row * inner + index) *
                             rhs.at(index * columns + column);
            }
            output.at(row * columns + column) = accumulator;
          }
        }
      }
    }
  }
  return output;
}

Tensor softmax(const Tensor& input) {
  if (input.ndim() != 2) throw std::invalid_argument("softmax expects a matrix");
  const size_t rows = input.shape()[0];
  const size_t columns = input.shape()[1];
  if (columns == 0) throw std::invalid_argument("softmax requires non-empty rows");
  Tensor output(input.shape());
  for (size_t row = 0; row < rows; ++row) {
    float maximum = -std::numeric_limits<float>::infinity();
    for (size_t column = 0; column < columns; ++column) {
      maximum = std::max(maximum, input.at(row * columns + column));
    }
    float denominator = 0.0f;
    for (size_t column = 0; column < columns; ++column) {
      const float value = std::exp(input.at(row * columns + column) - maximum);
      output.at(row * columns + column) = value;
      denominator += value;
    }
    for (size_t column = 0; column < columns; ++column) {
      output.at(row * columns + column) /= denominator;
    }
  }
  return output;
}

Tensor attention(const Tensor& query, const Tensor& key, const Tensor& value,
                 const AttentionConfig& config) {
  // Attention contracts are Q=[queries, depth], K=[keys, depth],
  // V=[keys, values].  K is transposed internally for Q*K^T.
  if (query.ndim() != 2 || key.ndim() != 2 || value.ndim() != 2 ||
      query.shape()[1] != key.shape()[1] || key.shape()[0] != value.shape()[0]) {
    throw std::invalid_argument("attention value shape is incompatible");
  }
  Tensor transposed_key({key.shape()[1], key.shape()[0]});
  for (size_t row = 0; row < key.shape()[0]; ++row) {
    for (size_t column = 0; column < key.shape()[1]; ++column) {
      transposed_key.at(column * key.shape()[0] + row) =
          key.at(row * key.shape()[1] + column);
    }
  }
  Tensor scores = gemm(query, transposed_key);
  const float scale = config.scale == 0.0f
                          ? 1.0f / std::sqrt(static_cast<float>(query.shape()[1]))
                          : config.scale;
  for (size_t i = 0; i < scores.numel(); ++i) scores.at(i) *= scale;
  if (config.causal) {
    for (size_t row = 0; row < scores.shape()[0]; ++row) {
      for (size_t column = row + 1; column < scores.shape()[1]; ++column) {
        scores.at(row * scores.shape()[1] + column) = -1e9f;
      }
    }
  }
  return gemm(softmax(scores), value);
}

Tensor layer_norm(const Tensor& input, const Tensor& gamma, const Tensor& beta,
                  float epsilon) {
  if (input.ndim() != 2 || gamma.numel() != input.shape()[1] ||
      beta.numel() != gamma.numel()) {
    throw std::invalid_argument("layer_norm shape mismatch");
  }
  if (!(epsilon > 0.0f) || !std::isfinite(epsilon)) {
    throw std::invalid_argument("layer_norm epsilon must be finite and positive");
  }
  const size_t rows = input.shape()[0];
  const size_t columns = input.shape()[1];
  Tensor output(input.shape());
  for (size_t row = 0; row < rows; ++row) {
    float mean = 0.0f;
    for (size_t column = 0; column < columns; ++column) mean += input.at(row * columns + column);
    mean /= static_cast<float>(columns);
    float variance = 0.0f;
    for (size_t column = 0; column < columns; ++column) {
      const float delta = input.at(row * columns + column) - mean;
      variance += delta * delta;
    }
    variance /= static_cast<float>(columns);
    for (size_t column = 0; column < columns; ++column) {
      const float normalized = (input.at(row * columns + column) - mean) /
                              std::sqrt(variance + epsilon);
      output.at(row * columns + column) = normalized * gamma.at(column) + beta.at(column);
    }
  }
  return output;
}

std::string kernel_config(const GemmConfig& config) {
  return "tile=" + std::to_string(config.tile) +
         ",warps=" + std::to_string(config.warps) +
         (config.use_tensor_core ? ",tc" : "");
}

#ifndef GPUFORGE_CUDA
bool cuda_available() { return false; }
Tensor gemm_cuda(const Tensor& lhs, const Tensor& rhs, const GemmConfig& c) { return gemm(lhs, rhs, c); }
Tensor attention_cuda(const Tensor& q, const Tensor& k, const Tensor& v, const AttentionConfig& c) { return attention(q, k, v, c); }
Tensor gemm_tensorcore(const Tensor& lhs, const Tensor& rhs) { return gemm(lhs, rhs, {16, 4, true}); }
#endif

}  // namespace gpuforge
