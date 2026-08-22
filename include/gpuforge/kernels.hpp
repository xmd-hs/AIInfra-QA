#pragma once

#include "tensor.hpp"

#include <string>

namespace gpuforge {

struct GemmConfig {
  int tile = 16;
  int warps = 4;
  bool use_tensor_core = false;
};

struct AttentionConfig {
  int tile = 64;
  float scale = 0.f;
  bool causal = false;
};

Tensor gemm(const Tensor&, const Tensor&, const GemmConfig& = {});
Tensor layer_norm(const Tensor&,
                  const Tensor&,
                  const Tensor&,
                  float eps = 1e-5f);
Tensor softmax(const Tensor&);
Tensor attention(const Tensor& q,
                 const Tensor& k,
                 const Tensor& v,
                 const AttentionConfig& = {});
std::string kernel_config(const GemmConfig&);
bool cuda_available();
Tensor gemm_cuda(const Tensor&, const Tensor&, const GemmConfig& = {});
Tensor attention_cuda(const Tensor&,
                      const Tensor&,
                      const Tensor&,
                      const AttentionConfig& = {});
Tensor gemm_tensorcore(const Tensor&, const Tensor&);

}  // namespace gpuforge
