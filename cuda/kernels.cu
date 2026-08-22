#include "gpuforge/kernels.hpp"

#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <stdexcept>
#include <vector>

using namespace nvcuda;

namespace gpuforge {

namespace {
constexpr int kGemmTile = 32;
constexpr int kAttentionBlock = 128;

void check_cuda(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

void check_matrix_shapes(const Tensor& lhs, const Tensor& rhs,
                         const char* operation) {
  if (lhs.ndim() != 2 || rhs.ndim() != 2 || lhs.shape()[1] != rhs.shape()[0])
    throw std::invalid_argument(std::string(operation) + " shape");
}

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t elements) {
    check_cuda(cudaMalloc(&ptr_, elements * sizeof(T)));
  }
  ~DeviceBuffer() { if (ptr_) cudaFree(ptr_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  T* get() const { return ptr_; }

 private:
  T* ptr_ = nullptr;
};
}  // namespace

__global__ void gemm_tiled(const float* lhs, const float* rhs, float* output,
                           int rows, int columns, int inner) {
  __shared__ float lhs_tile[kGemmTile][kGemmTile];
  __shared__ float rhs_tile[kGemmTile][kGemmTile];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int row = blockIdx.y * kGemmTile + ty;
  const int column = blockIdx.x * kGemmTile + tx;
  float accumulator = 0.0f;
  for (int block = 0; block < inner; block += kGemmTile) {
    lhs_tile[ty][tx] = row < rows && block + tx < inner
                           ? lhs[row * inner + block + tx] : 0.0f;
    rhs_tile[ty][tx] = block + ty < inner && column < columns
                           ? rhs[(block + ty) * columns + column] : 0.0f;
    __syncthreads();
    for (int k = 0; k < kGemmTile; ++k) accumulator += lhs_tile[ty][k] * rhs_tile[k][tx];
    __syncthreads();
  }
  if (row < rows && column < columns) output[row * columns + column] = accumulator;
}

__global__ void qkv_attention(const float* query, const float* key,
                              const float* value, float* output, int queries,
                              int keys, int depth, int value_dim, float scale,
                              bool causal) {
  const int query_row = blockIdx.x * blockDim.x + threadIdx.x;
  if (query_row >= queries) return;
  float scores[256];
  float maximum = -1e30f;
  for (int key_row = 0; key_row < keys; ++key_row) {
    float score = 0.0f;
    for (int d = 0; d < depth; ++d) score += query[query_row * depth + d] * key[key_row * depth + d];
    scores[key_row] = causal && key_row > query_row ? -1e9f : score * scale;
    maximum = fmaxf(maximum, scores[key_row]);
  }
  float denominator = 0.0f;
  for (int key_row = 0; key_row < keys; ++key_row) {
    scores[key_row] = expf(scores[key_row] - maximum);
    denominator += scores[key_row];
  }
  for (int d = 0; d < value_dim; ++d) {
    float result = 0.0f;
    for (int key_row = 0; key_row < keys; ++key_row)
      result += scores[key_row] / denominator * value[key_row * value_dim + d];
    output[query_row * value_dim + d] = result;
  }
}

bool cuda_available() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

Tensor gemm_cuda(const Tensor& lhs, const Tensor& rhs, const GemmConfig&) {
  check_matrix_shapes(lhs, rhs, "gemm_cuda");
  const int rows = static_cast<int>(lhs.shape()[0]);
  const int inner = static_cast<int>(lhs.shape()[1]);
  const int columns = static_cast<int>(rhs.shape()[1]);
  Tensor output({static_cast<size_t>(rows), static_cast<size_t>(columns)});
  DeviceBuffer<float> device_lhs(lhs.numel()), device_rhs(rhs.numel()), device_output(output.numel());
  const size_t lhs_bytes = lhs.numel() * sizeof(float);
  const size_t rhs_bytes = rhs.numel() * sizeof(float);
  const size_t output_bytes = output.numel() * sizeof(float);
  check_cuda(cudaMemcpy(device_lhs.get(), lhs.data(), lhs_bytes, cudaMemcpyHostToDevice));
  check_cuda(cudaMemcpy(device_rhs.get(), rhs.data(), rhs_bytes, cudaMemcpyHostToDevice));
  dim3 block(kGemmTile, kGemmTile);
  dim3 grid((columns + kGemmTile - 1) / kGemmTile, (rows + kGemmTile - 1) / kGemmTile);
  gemm_tiled<<<grid, block>>>(device_lhs.get(), device_rhs.get(), device_output.get(), rows, columns, inner);
  check_cuda(cudaGetLastError());
  check_cuda(cudaDeviceSynchronize());
  check_cuda(cudaMemcpy(output.data(), device_output.get(), output_bytes, cudaMemcpyDeviceToHost));
  return output;
}

Tensor attention_cuda(const Tensor& query, const Tensor& key, const Tensor& value,
                      const AttentionConfig& config) {
  if (query.ndim() != 2 || key.ndim() != 2 || value.ndim() != 2 ||
      query.shape()[1] != key.shape()[1] || key.shape()[0] != value.shape()[0] ||
      key.shape()[0] > 256) throw std::invalid_argument("attention_cuda shape");
  const int queries = static_cast<int>(query.shape()[0]);
  const int keys = static_cast<int>(key.shape()[0]);
  const int depth = static_cast<int>(query.shape()[1]);
  const int value_dim = static_cast<int>(value.shape()[1]);
  Tensor output({static_cast<size_t>(queries), static_cast<size_t>(value_dim)});
  DeviceBuffer<float> dq(query.numel()), dk(key.numel()), dv(value.numel()), dout(output.numel());
  const size_t qb = query.numel() * sizeof(float), kb = key.numel() * sizeof(float);
  const size_t vb = value.numel() * sizeof(float), ob = output.numel() * sizeof(float);
  check_cuda(cudaMemcpy(dq.get(), query.data(), qb, cudaMemcpyHostToDevice));
  check_cuda(cudaMemcpy(dk.get(), key.data(), kb, cudaMemcpyHostToDevice));
  check_cuda(cudaMemcpy(dv.get(), value.data(), vb, cudaMemcpyHostToDevice));
  const float scale = config.scale == 0.0f ? 1.0f / sqrtf(static_cast<float>(depth)) : config.scale;
  qkv_attention<<<(queries + kAttentionBlock - 1) / kAttentionBlock, kAttentionBlock>>>(
      dq.get(), dk.get(), dv.get(), dout.get(), queries, keys, depth, value_dim, scale, config.causal);
  check_cuda(cudaGetLastError()); check_cuda(cudaDeviceSynchronize());
  check_cuda(cudaMemcpy(output.data(), dout.get(), ob, cudaMemcpyDeviceToHost));
  return output;
}

__global__ void wmma_gemm(const half* lhs, const half* rhs, float* output,
                          int rows, int columns, int inner) {
  const int block_x = blockIdx.x, block_y = blockIdx.y;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> lhs_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> rhs_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
  wmma::fill_fragment(accumulator, 0.0f);
  for (int k = 0; k < inner; k += 16) {
    wmma::load_matrix_sync(lhs_frag, lhs + block_y * 16 * inner + k, inner);
    wmma::load_matrix_sync(rhs_frag, rhs + k * columns + block_x * 16, columns);
    wmma::mma_sync(accumulator, lhs_frag, rhs_frag, accumulator);
  }
  wmma::store_matrix_sync(output + block_y * 16 * columns + block_x * 16,
                          accumulator, columns, wmma::mem_row_major);
}

Tensor gemm_tensorcore(const Tensor& lhs, const Tensor& rhs) {
  check_matrix_shapes(lhs, rhs, "tensorcore");
  if (lhs.shape()[0] % 16 || rhs.shape()[1] % 16 || lhs.shape()[1] % 16)
    throw std::invalid_argument("tensorcore requires multiples of 16");
  const int rows = static_cast<int>(lhs.shape()[0]);
  const int inner = static_cast<int>(lhs.shape()[1]);
  const int columns = static_cast<int>(rhs.shape()[1]);
  Tensor output({static_cast<size_t>(rows), static_cast<size_t>(columns)});
  std::vector<half> host_lhs(lhs.numel()), host_rhs(rhs.numel());
  for (size_t i = 0; i < lhs.numel(); ++i) host_lhs[i] = __float2half(lhs.at(i));
  for (size_t i = 0; i < rhs.numel(); ++i) host_rhs[i] = __float2half(rhs.at(i));
  DeviceBuffer<half> device_lhs(host_lhs.size()), device_rhs(host_rhs.size());
  DeviceBuffer<float> device_output(output.numel());
  check_cuda(cudaMemcpy(device_lhs.get(), host_lhs.data(), host_lhs.size() * sizeof(half), cudaMemcpyHostToDevice));
  check_cuda(cudaMemcpy(device_rhs.get(), host_rhs.data(), host_rhs.size() * sizeof(half), cudaMemcpyHostToDevice));
  wmma_gemm<<<dim3(columns / 16, rows / 16), 32>>>(device_lhs.get(), device_rhs.get(), device_output.get(), rows, columns, inner);
  check_cuda(cudaGetLastError()); check_cuda(cudaDeviceSynchronize());
  check_cuda(cudaMemcpy(output.data(), device_output.get(), output.numel() * sizeof(float), cudaMemcpyDeviceToHost));
  return output;
}

}  // namespace gpuforge
