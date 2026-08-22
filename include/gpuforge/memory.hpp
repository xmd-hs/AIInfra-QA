#pragma once

#include <cstddef>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace gpuforge {

class MemoryPool {
 public:
  explicit MemoryPool(size_t bytes);

  void* allocate(size_t bytes, size_t alignment = 256);
  void release(void*);
  size_t capacity() const { return capacity_; }
  size_t used() const;
  double fragmentation() const;

 private:
  size_t capacity_;
  size_t used_bytes_ = 0;
  std::unordered_map<void*, size_t> allocations_;
  mutable std::mutex mu_;
};

class PagedKV {
 public:
  struct Location {
    int page = -1;
    size_t offset = 0;
    bool valid = false;
  };

  PagedKV(size_t layers, size_t page_tokens, size_t page_bytes, size_t pages);

  int acquire();
  void release(int page);
  void release_sequence(size_t sequence);
  void append(size_t sequence, const void* data, size_t bytes);
  size_t read(size_t sequence, size_t offset, void* out, size_t bytes) const;
  Location locate(size_t sequence, size_t byte_offset) const;
  size_t bytes(size_t sequence) const;
  size_t resident(size_t sequence) const;
  size_t pages() const { return pages_.size(); }

 private:
  struct Page {
    std::vector<unsigned char> data;
    bool allocated = false;
    size_t write_offset = 0;
  };

  size_t page_tokens_;
  size_t page_bytes_;
  std::vector<Page> pages_;
  std::vector<std::vector<int>> sequences_;
  std::vector<size_t> sequence_bytes_;
  mutable std::mutex mu_;
};

}  // namespace gpuforge
