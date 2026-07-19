#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include "aiinfra/concurrency/mpmc_queue.h"

namespace aiinfra::concurrency {

class ThreadPool {
 public:
  explicit ThreadPool(std::size_t workers, std::size_t queue_capacity = 4096);
  ~ThreadPool();

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  template <typename Fn>
  auto submit(Fn&& fn) -> std::future<decltype(fn())> {
    using Result = decltype(fn());
    if (!running_.load(std::memory_order_acquire)) {
      throw std::runtime_error("thread pool is stopped");
    }

    auto task = std::make_shared<std::packaged_task<Result()>>(std::forward<Fn>(fn));
    std::future<Result> future = task->get_future();

    std::function<void()> wrapper = [task]() { (*task)(); };
    if (!queue_.enqueue(std::move(wrapper))) {
      throw std::runtime_error("thread pool queue is full");
    }
    cv_.notify_one();
    return future;
  }

  void shutdown();

 private:
  void worker_loop();

  std::atomic<bool> running_{true};
  BoundedMpmcQueue<std::function<void()>> queue_;
  std::mutex wait_mutex_;
  std::condition_variable cv_;
  std::vector<std::thread> workers_;
};

}  // namespace aiinfra::concurrency
