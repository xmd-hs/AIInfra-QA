#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace aiinfra::gpu {

struct PageHandle {
  std::uint32_t page_id{0};
  std::uint32_t generation{0};
  std::size_t byte_offset{0};
};

struct PageAllocation {
  std::string owner;
  std::size_t bytes{0};
  std::vector<PageHandle> pages;
};

struct MemoryPoolStats {
  std::size_t page_size{0};
  std::size_t total_pages{0};
  std::size_t free_pages{0};
  std::size_t used_pages{0};
  std::size_t reserved_bytes{0};
  std::size_t largest_free_run{0};
  double utilization{0.0};
  double fragmentation_ratio{0.0};
};

struct PageTableEntry {
  std::uint32_t page_id{0};
  std::uint32_t generation{0};
  std::size_t byte_offset{0};
  std::string owner;
  bool free{true};
};

class PagedGpuMemoryPool {
 public:
  PagedGpuMemoryPool(std::size_t page_size, std::size_t page_count, double high_watermark = 0.92);

  std::optional<PageAllocation> allocate(std::size_t bytes, const std::string& owner);
  bool release(const PageAllocation& allocation);
  std::size_t release_owner(const std::string& owner);

  MemoryPoolStats stats() const;
  std::unordered_map<std::string, std::size_t> owner_page_counts() const;
  std::vector<PageTableEntry> page_table_snapshot() const;
  bool above_high_watermark() const;

 private:
  struct PageMeta {
    bool free{true};
    std::uint32_t generation{0};
    std::string owner;
  };

  std::size_t pages_for_bytes(std::size_t bytes) const;
  double fragmentation_locked() const;
  std::size_t largest_free_run_locked() const;

  const std::size_t page_size_;
  const double high_watermark_;
  mutable std::mutex mutex_;
  std::vector<PageMeta> pages_;
  std::vector<std::uint32_t> free_list_;
  std::unordered_map<std::string, std::size_t> owner_pages_;
};

}  // namespace aiinfra::gpu
