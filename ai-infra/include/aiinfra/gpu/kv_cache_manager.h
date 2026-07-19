#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "aiinfra/gpu/paged_memory_pool.h"

namespace aiinfra::gpu {

struct KvCacheConfig {
  std::size_t layers{32};
  std::size_t kv_heads{32};
  std::size_t head_dim{128};
  std::size_t bytes_per_element{2};
  std::size_t tokens_per_block{16};
};

struct SequenceBlockTable {
  std::uint64_t sequence_id{0};
  std::size_t token_capacity{0};
  std::size_t bytes_reserved{0};
  std::vector<PageHandle> pages;
};

class KvCacheManager {
 public:
  KvCacheManager(PagedGpuMemoryPool& memory_pool, KvCacheConfig config);

  std::optional<SequenceBlockTable> allocate_sequence(std::uint64_t sequence_id, std::size_t initial_tokens);
  std::optional<SequenceBlockTable> grow_sequence(std::uint64_t sequence_id, std::size_t target_tokens);
  bool release_sequence(std::uint64_t sequence_id);

  std::optional<SequenceBlockTable> block_table(std::uint64_t sequence_id) const;
  std::size_t bytes_per_token() const;
  std::size_t active_sequences() const;

 private:
  std::size_t bytes_for_tokens(std::size_t tokens) const;
  std::size_t round_tokens(std::size_t tokens) const;
  std::string owner(std::uint64_t sequence_id) const;

  PagedGpuMemoryPool& memory_pool_;
  KvCacheConfig config_;
  mutable std::mutex mutex_;
  std::unordered_map<std::uint64_t, SequenceBlockTable> sequences_;
};

}  // namespace aiinfra::gpu
