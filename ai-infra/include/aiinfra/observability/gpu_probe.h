#pragma once

#include <string>

#include "aiinfra/gpu/paged_memory_pool.h"

namespace aiinfra::observability {

struct GpuSnapshot {
  double gpu_utilization{0.0};
  double memory_utilization{0.0};
  double power_watts{0.0};
  double temperature_celsius{0.0};
  double sm_active{0.0};
  std::string backend{"fallback"};
};

class GpuProbe {
 public:
  explicit GpuProbe(const gpu::PagedGpuMemoryPool* pool = nullptr);
  GpuSnapshot sample() const;

 private:
  const gpu::PagedGpuMemoryPool* pool_;
};

}  // namespace aiinfra::observability
