#include "aiinfra/gpu/kv_cache_manager.h"

#include <algorithm>
#include <mutex>
#include <sstream>
#include <utility>

namespace aiinfra::gpu {

KvCacheManager::KvCacheManager(PagedGpuMemoryPool& memory_pool, KvCacheConfig config)
    : memory_pool_(memory_pool), config_(config) {}

std::optional<SequenceBlockTable> KvCacheManager::allocate_sequence(std::uint64_t sequence_id,
                                                                     std::size_t initial_tokens) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (sequences_.find(sequence_id) != sequences_.end()) {
    return std::nullopt;
  }

  const auto capacity = round_tokens(initial_tokens);
  auto allocation = memory_pool_.allocate(bytes_for_tokens(capacity), owner(sequence_id));
  if (!allocation) {
    return std::nullopt;
  }

  SequenceBlockTable table;
  table.sequence_id = sequence_id;
  table.token_capacity = capacity;
  table.bytes_reserved = allocation->pages.size() * memory_pool_.stats().page_size;
  table.pages = std::move(allocation->pages);
  sequences_[sequence_id] = table;
  return table;
}

std::optional<SequenceBlockTable> KvCacheManager::grow_sequence(std::uint64_t sequence_id,
                                                                std::size_t target_tokens) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = sequences_.find(sequence_id);
  if (it == sequences_.end()) {
    const auto capacity = round_tokens(target_tokens);
    auto allocation = memory_pool_.allocate(bytes_for_tokens(capacity), owner(sequence_id));
    if (!allocation) {
      return std::nullopt;
    }

    SequenceBlockTable table;
    table.sequence_id = sequence_id;
    table.token_capacity = capacity;
    table.bytes_reserved = allocation->pages.size() * memory_pool_.stats().page_size;
    table.pages = std::move(allocation->pages);
    sequences_[sequence_id] = table;
    return table;
  }

  const auto target_capacity = round_tokens(target_tokens);
  if (target_capacity <= it->second.token_capacity) {
    return it->second;
  }

  const auto current_bytes = bytes_for_tokens(it->second.token_capacity);
  const auto target_bytes = bytes_for_tokens(target_capacity);
  auto extra = memory_pool_.allocate(target_bytes - current_bytes, owner(sequence_id));
  if (!extra) {
    return std::nullopt;
  }

  it->second.token_capacity = target_capacity;
  it->second.bytes_reserved += extra->pages.size() * memory_pool_.stats().page_size;
  it->second.pages.insert(it->second.pages.end(), extra->pages.begin(), extra->pages.end());
  return it->second;
}

bool KvCacheManager::release_sequence(std::uint64_t sequence_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = sequences_.find(sequence_id);
  if (it == sequences_.end()) {
    return false;
  }
  memory_pool_.release_owner(owner(sequence_id));
  sequences_.erase(it);
  return true;
}

std::optional<SequenceBlockTable> KvCacheManager::block_table(std::uint64_t sequence_id) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto it = sequences_.find(sequence_id);
  if (it == sequences_.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::size_t KvCacheManager::bytes_per_token() const {
  return config_.layers * config_.kv_heads * config_.head_dim * 2 * config_.bytes_per_element;
}

std::size_t KvCacheManager::active_sequences() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return sequences_.size();
}

std::size_t KvCacheManager::bytes_for_tokens(std::size_t tokens) const {
  return std::max<std::size_t>(1, tokens) * bytes_per_token();
}

std::size_t KvCacheManager::round_tokens(std::size_t tokens) const {
  const auto block = std::max<std::size_t>(1, config_.tokens_per_block);
  return ((std::max<std::size_t>(1, tokens) + block - 1) / block) * block;
}

std::string KvCacheManager::owner(std::uint64_t sequence_id) const {
  std::ostringstream out;
  out << "kv:" << sequence_id;
  return out.str();
}

}  // namespace aiinfra::gpu
