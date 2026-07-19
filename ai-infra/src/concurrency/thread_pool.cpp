#include "aiinfra/concurrency/thread_pool.h"

#include <chrono>
#include <stdexcept>

namespace aiinfra::concurrency {

ThreadPool::ThreadPool(std::size_t workers, std::size_t queue_capacity)
    : queue_(queue_capacity) {
  if (workers == 0) {
    throw std::invalid_argument("thread pool needs at least one worker");
  }
  workers_.reserve(workers);
  for (std::size_t i = 0; i < workers; ++i) {
    workers_.emplace_back([this]() { worker_loop(); });
  }
}

ThreadPool::~ThreadPool() { shutdown(); }

void ThreadPool::shutdown() {
  bool expected = true;
  if (!running_.compare_exchange_strong(expected, false)) {
    return;
  }
  cv_.notify_all();
  for (auto& worker : workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
}

void ThreadPool::worker_loop() {
  while (running_.load(std::memory_order_acquire)) {
    std::function<void()> task;
    if (queue_.dequeue(task)) {
      task();
      continue;
    }

    std::unique_lock<std::mutex> lock(wait_mutex_);
    cv_.wait_for(lock, std::chrono::milliseconds(2));
  }

  std::function<void()> task;
  while (queue_.dequeue(task)) {
    task();
  }
}

}  // namespace aiinfra::concurrency
