#include "aiinfra/observability/gpu_probe.h"

namespace aiinfra::observability {

GpuProbe::GpuProbe(const gpu::PagedGpuMemoryPool* pool) : pool_(pool) {}

GpuSnapshot GpuProbe::sample() const {
  GpuSnapshot snapshot;
  if (pool_ != nullptr) {
    const auto stats = pool_->stats();
    snapshot.memory_utilization = stats.utilization;
  }
  return snapshot;
}

}  // namespace aiinfra::observability
