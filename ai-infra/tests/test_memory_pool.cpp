#include <cassert>

#include "aiinfra/gpu/paged_memory_pool.h"

int main() {
  aiinfra::gpu::PagedGpuMemoryPool pool(1024, 8, 0.75);

  auto a = pool.allocate(2048, "req:a");
  auto b = pool.allocate(1024, "req:b");
  assert(a.has_value());
  assert(b.has_value());
  assert(pool.stats().used_pages == 3);
  assert(pool.owner_page_counts()["req:a"] == 2);
  assert(pool.page_table_snapshot().size() == 8);

  assert(pool.release(*a));
  const auto fragmented = pool.stats();
  assert(fragmented.free_pages == 7);
  assert(fragmented.fragmentation_ratio >= 0.0);

  const auto released = pool.release_owner("req:b");
  assert(released == 1);
  assert(pool.stats().used_pages == 0);
  assert(pool.owner_page_counts().empty());

  auto big = pool.allocate(8 * 1024, "model:demo");
  assert(big.has_value());
  assert(pool.above_high_watermark());
  return 0;
}
