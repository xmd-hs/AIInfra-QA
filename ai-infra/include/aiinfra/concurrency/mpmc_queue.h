#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace aiinfra::concurrency {

inline std::size_t next_power_of_two(std::size_t value) {
  if (value < 2) {
    return 2;
  }
  --value;
  for (std::size_t shift = 1; shift < sizeof(std::size_t) * 8; shift <<= 1) {
    value |= value >> shift;
  }
  return value + 1;
}

template <typename T>
class BoundedMpmcQueue {
 public:
  explicit BoundedMpmcQueue(std::size_t capacity)
      : capacity_(next_power_of_two(capacity)),
        mask_(capacity_ - 1),
        buffer_(new Cell[capacity_]) {
    for (std::size_t i = 0; i < capacity_; ++i) {
      buffer_[i].sequence.store(i, std::memory_order_relaxed);
    }
    enqueue_pos_.store(0, std::memory_order_relaxed);
    dequeue_pos_.store(0, std::memory_order_relaxed);
  }

  BoundedMpmcQueue(const BoundedMpmcQueue&) = delete;
  BoundedMpmcQueue& operator=(const BoundedMpmcQueue&) = delete;

  bool enqueue(T value) {
    Cell* cell = nullptr;
    std::size_t pos = enqueue_pos_.load(std::memory_order_relaxed);

    for (;;) {
      cell = &buffer_[pos & mask_];
      const std::size_t seq = cell->sequence.load(std::memory_order_acquire);
      const intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);
      if (diff == 0) {
        if (enqueue_pos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
          break;
        }
      } else if (diff < 0) {
        return false;
      } else {
        pos = enqueue_pos_.load(std::memory_order_relaxed);
      }
    }

    cell->data = std::move(value);
    cell->sequence.store(pos + 1, std::memory_order_release);
    return true;
  }

  bool dequeue(T& value) {
    Cell* cell = nullptr;
    std::size_t pos = dequeue_pos_.load(std::memory_order_relaxed);

    for (;;) {
      cell = &buffer_[pos & mask_];
      const std::size_t seq = cell->sequence.load(std::memory_order_acquire);
      const intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos + 1);
      if (diff == 0) {
        if (dequeue_pos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
          break;
        }
      } else if (diff < 0) {
        return false;
      } else {
        pos = dequeue_pos_.load(std::memory_order_relaxed);
      }
    }

    value = std::move(cell->data);
    cell->sequence.store(pos + capacity_, std::memory_order_release);
    return true;
  }

  std::size_t capacity() const { return capacity_; }

 private:
  struct Cell {
    std::atomic<std::size_t> sequence;
    T data;
  };

  const std::size_t capacity_;
  const std::size_t mask_;
  std::unique_ptr<Cell[]> buffer_;
  alignas(64) std::atomic<std::size_t> enqueue_pos_{0};
  alignas(64) std::atomic<std::size_t> dequeue_pos_{0};
};

}  // namespace aiinfra::concurrency
