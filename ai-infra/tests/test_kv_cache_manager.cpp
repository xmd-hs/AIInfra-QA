#include <cassert>

#include "aiinfra/gpu/kv_cache_manager.h"

int main() {
  aiinfra::gpu::PagedGpuMemoryPool pool(1024, 64);
  aiinfra::gpu::KvCacheConfig config;
  config.layers = 1;
  config.kv_heads = 1;
  config.head_dim = 8;
  config.bytes_per_element = 2;
  config.tokens_per_block = 4;

  aiinfra::gpu::KvCacheManager kv(pool, config);
  assert(kv.bytes_per_token() == 32);

  auto seq = kv.allocate_sequence(7, 3);
  assert(seq.has_value());
  assert(seq->token_capacity == 4);
  assert(kv.active_sequences() == 1);

  auto grown = kv.grow_sequence(7, 20);
  assert(grown.has_value());
  assert(grown->token_capacity == 20);
  assert(grown->pages.size() >= seq->pages.size());

  auto table = kv.block_table(7);
  assert(table.has_value());
  assert(table->sequence_id == 7);

  assert(kv.release_sequence(7));
  assert(kv.active_sequences() == 0);
  assert(pool.stats().used_pages == 0);
  return 0;
}
