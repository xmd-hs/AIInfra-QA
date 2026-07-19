#include "aiinfra/gpu/paged_memory_pool.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace aiinfra::gpu {

PagedGpuMemoryPool::PagedGpuMemoryPool(std::size_t page_size, std::size_t page_count, double high_watermark)
    : page_size_(page_size), high_watermark_(high_watermark), pages_(page_count) {
  if (page_size == 0 || page_count == 0) {
    throw std::invalid_argument("page size and page count must be positive");
  }
  if (high_watermark <= 0.0 || high_watermark > 1.0) {
    throw std::invalid_argument("high watermark must be in (0, 1]");
  }
  free_list_.reserve(page_count);
  for (std::uint32_t i = 0; i < page_count; ++i) {
    free_list_.push_back(static_cast<std::uint32_t>(page_count - 1 - i));
  }
}

std::optional<PageAllocation> PagedGpuMemoryPool::allocate(std::size_t bytes, const std::string& owner) {
  const std::size_t needed = pages_for_bytes(bytes);
  std::lock_guard<std::mutex> lock(mutex_);

  if (free_list_.size() < needed) {
    return std::nullopt;
  }

  PageAllocation allocation;
  allocation.owner = owner;
  allocation.bytes = bytes;
  allocation.pages.reserve(needed);

  for (std::size_t i = 0; i < needed; ++i) {
    const std::uint32_t page_id = free_list_.back();
    free_list_.pop_back();
    auto& page = pages_[page_id];
    page.free = false;
    page.owner = owner;
    allocation.pages.push_back(PageHandle{page_id, page.generation, page_id * page_size_});
  }
  owner_pages_[owner] += needed;

  return allocation;
}

bool PagedGpuMemoryPool::release(const PageAllocation& allocation) {
  std::lock_guard<std::mutex> lock(mutex_);
  bool all_released = true;
  for (const auto& handle : allocation.pages) {
    if (handle.page_id >= pages_.size()) {
      all_released = false;
      continue;
    }
    auto& page = pages_[handle.page_id];
    if (page.free || page.generation != handle.generation) {
      all_released = false;
      continue;
    }
    auto owner_it = owner_pages_.find(page.owner);
    if (owner_it != owner_pages_.end()) {
      if (owner_it->second > 1) {
        --owner_it->second;
      } else {
        owner_pages_.erase(owner_it);
      }
    }
    page.free = true;
    page.owner.clear();
    ++page.generation;
    free_list_.push_back(handle.page_id);
  }
  return all_released;
}

std::size_t PagedGpuMemoryPool::release_owner(const std::string& owner) {
  std::lock_guard<std::mutex> lock(mutex_);
  std::size_t released = 0;
  for (std::uint32_t page_id = 0; page_id < pages_.size(); ++page_id) {
    auto& page = pages_[page_id];
    if (!page.free && page.owner == owner) {
      page.free = true;
      page.owner.clear();
      ++page.generation;
      free_list_.push_back(page_id);
      ++released;
    }
  }
  owner_pages_.erase(owner);
  return released;
}

MemoryPoolStats PagedGpuMemoryPool::stats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  MemoryPoolStats stats;
  stats.page_size = page_size_;
  stats.total_pages = pages_.size();
  stats.free_pages = free_list_.size();
  stats.used_pages = pages_.size() - stats.free_pages;
  stats.reserved_bytes = stats.used_pages * page_size_;
  stats.largest_free_run = largest_free_run_locked();
  stats.utilization = static_cast<double>(stats.used_pages) / static_cast<double>(stats.total_pages);
  stats.fragmentation_ratio = fragmentation_locked();
  return stats;
}

std::unordered_map<std::string, std::size_t> PagedGpuMemoryPool::owner_page_counts() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return owner_pages_;
}

std::vector<PageTableEntry> PagedGpuMemoryPool::page_table_snapshot() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::vector<PageTableEntry> snapshot;
  snapshot.reserve(pages_.size());
  for (std::uint32_t page_id = 0; page_id < pages_.size(); ++page_id) {
    const auto& page = pages_[page_id];
    snapshot.push_back(PageTableEntry{
        page_id,
        page.generation,
        page_id * page_size_,
        page.owner,
        page.free,
    });
  }
  return snapshot;
}

bool PagedGpuMemoryPool::above_high_watermark() const {
  return stats().utilization >= high_watermark_;
}

std::size_t PagedGpuMemoryPool::pages_for_bytes(std::size_t bytes) const {
  return std::max<std::size_t>(1, (bytes + page_size_ - 1) / page_size_);
}

double PagedGpuMemoryPool::fragmentation_locked() const {
  std::size_t free_pages = 0;
  std::size_t largest_run = 0;
  std::size_t current_run = 0;

  for (const auto& page : pages_) {
    if (page.free) {
      ++free_pages;
      ++current_run;
      largest_run = std::max(largest_run, current_run);
    } else {
      current_run = 0;
    }
  }

  if (free_pages == 0) {
    return 0.0;
  }
  return 1.0 - (static_cast<double>(largest_run) / static_cast<double>(free_pages));
}

std::size_t PagedGpuMemoryPool::largest_free_run_locked() const {
  std::size_t largest_run = 0;
  std::size_t current_run = 0;
  for (const auto& page : pages_) {
    if (page.free) {
      ++current_run;
      largest_run = std::max(largest_run, current_run);
    } else {
      current_run = 0;
    }
  }
  return largest_run;
}

}  // namespace aiinfra::gpu
