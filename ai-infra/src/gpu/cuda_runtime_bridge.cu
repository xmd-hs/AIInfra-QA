#include <cstddef>

#include <cuda_runtime.h>

extern "C" void* aiinfra_cuda_malloc(std::size_t bytes) {
  void* ptr = nullptr;
  if (cudaMalloc(&ptr, bytes) != cudaSuccess) {
    return nullptr;
  }
  return ptr;
}

extern "C" void aiinfra_cuda_free(void* ptr) {
  if (ptr != nullptr) {
    cudaFree(ptr);
  }
}
