#include "gpuforge/memory.hpp"

#include <kama/MemoryPool.h>

#include <cstring>
#include <new>
#include <stdexcept>

namespace gpuforge {

MemoryPool::MemoryPool(size_t n) : capacity_(n) {}

void* MemoryPool::allocate(size_t n, size_t a) {
  if (n == 0) return nullptr;
  std::lock_guard<std::mutex> g(mu_);
  if (n > capacity_ - std::min(capacity_, used_bytes_)) throw std::bad_alloc();
  void* p = a > 8 ? Kama_memoryPool::MemoryPool::allocateAligned(n, a)
                  : Kama_memoryPool::MemoryPool::allocate(n);
  if (!p) throw std::bad_alloc();
  allocations_.emplace(p, n);
  used_bytes_ += n;
  return p;
}

void MemoryPool::release(void* p) {
  if (p) {
    std::lock_guard<std::mutex> g(mu_);
    const auto it = allocations_.find(p);
    if (it == allocations_.end()) return;
    used_bytes_ -= it->second;
    allocations_.erase(it);
    Kama_memoryPool::MemoryPool::deallocate(p);
  }
}

size_t MemoryPool::used() const {
  std::lock_guard<std::mutex> g(mu_);
  return used_bytes_;
}

double MemoryPool::fragmentation() const {
  const auto s = Kama_memoryPool::MemoryPool::stats();
  return s.reservedBytes ? 1.0 - (double)s.cachedPageBytes / s.reservedBytes
                         : 0.0;
}

PagedKV::PagedKV(size_t /*layers*/, size_t t, size_t b, size_t p)
    : page_tokens_(t), page_bytes_(b), pages_(p) {
  for (auto& x : pages_) {
    x.data.resize(b);
  }
}

int PagedKV::acquire() {
  std::lock_guard<std::mutex> g(mu_);
  for (size_t i = 0; i < pages_.size(); ++i) {
    if (!pages_[i].allocated) {
      pages_[i].allocated = true;
      pages_[i].write_offset = 0;
      return (int)i;
    }
  }
  return -1;
}

void PagedKV::release(int i) {
  std::lock_guard<std::mutex> g(mu_);
  if (i >= 0 && (size_t)i < pages_.size()) {
    pages_[i].allocated = false;
    pages_[i].write_offset = 0;
  }
}

void PagedKV::release_sequence(size_t s) {
  std::lock_guard<std::mutex> g(mu_);
  if (s >= sequences_.size()) return;
  for (int id : sequences_[s]) {
    if (id >= 0 && static_cast<size_t>(id) < pages_.size()) {
      pages_[id].allocated = false;
      pages_[id].write_offset = 0;
    }
  }
  sequences_[s].clear();
  if (s < sequence_bytes_.size()) {
    sequence_bytes_[s] = 0;
  }
}

void PagedKV::append(size_t s, const void* d, size_t n) {
  std::lock_guard<std::mutex> g(mu_);
  if (s >= sequences_.size()) {
    sequences_.resize(s + 1);
    sequence_bytes_.resize(s + 1);
  }

  size_t left = n;
  const unsigned char* p = static_cast<const unsigned char*>(d);

  while (left) {
    int id = sequences_[s].empty() ? -1 : sequences_[s].back();
    if (id < 0 || pages_[id].write_offset >= page_bytes_) {
      id = -1;
      for (size_t i = 0; i < pages_.size(); ++i) {
        if (!pages_[i].allocated) {
          id = (int)i;
          pages_[i].allocated = true;
          pages_[i].write_offset = 0;
          sequences_[s].push_back(id);
          break;
        }
      }
    }
    if (id < 0) throw std::bad_alloc();

    const size_t offset = pages_[id].write_offset;
    const size_t c = std::min(left, page_bytes_ - offset);
    std::memcpy(pages_[id].data.data() + offset, p, c);
    pages_[id].write_offset += c;
    p += c;
    left -= c;
    sequence_bytes_[s] += c;
  }
}

PagedKV::Location PagedKV::locate(size_t s, size_t off) const {
  std::lock_guard<std::mutex> g(mu_);
  if (s >= sequence_bytes_.size() || off >= sequence_bytes_[s]) return {};
  const size_t page = off / page_bytes_;
  const size_t inner = off % page_bytes_;
  if (page >= sequences_[s].size()) return {};
  const int id = sequences_[s][page];
  return {id, inner, pages_[id].allocated};
}

size_t PagedKV::read(size_t s, size_t off, void* out, size_t n) const {
  if (!out || n == 0) return 0;
  std::lock_guard<std::mutex> g(mu_);
  if (s >= sequence_bytes_.size() || off >= sequence_bytes_[s]) return 0;
  size_t copied = 0;
  n = std::min(n, sequence_bytes_[s] - off);
  auto* dst = static_cast<unsigned char*>(out);
  while (copied < n) {
    const size_t absolute = off + copied;
    const size_t page_index = absolute / page_bytes_;
    const size_t page_offset = absolute % page_bytes_;
    if (page_index >= sequences_[s].size()) break;
    const int page_id = sequences_[s][page_index];
    const size_t chunk = std::min(n - copied, page_bytes_ - page_offset);
    std::memcpy(dst + copied, pages_[page_id].data.data() + page_offset,
                chunk);
    copied += chunk;
  }
  return copied;
}

size_t PagedKV::bytes(size_t s) const {
  std::lock_guard<std::mutex> g(mu_);
  return s < sequence_bytes_.size() ? sequence_bytes_[s] : 0;
}

size_t PagedKV::resident(size_t s) const {
  std::lock_guard<std::mutex> g(mu_);
  return s < sequences_.size() ? sequences_[s].size() : 0;
}

}  // namespace gpuforge
