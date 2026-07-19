#include <atomic>
#include <cassert>
#include <thread>
#include <vector>

#include "aiinfra/concurrency/mpmc_queue.h"

int main() {
  aiinfra::concurrency::BoundedMpmcQueue<int> queue(1024);
  constexpr int producers = 4;
  constexpr int per_producer = 1000;
  constexpr int total = producers * per_producer;

  std::atomic<int> consumed{0};
  std::atomic<long long> sum{0};
  std::vector<std::thread> threads;

  for (int p = 0; p < producers; ++p) {
    threads.emplace_back([&, p]() {
      for (int i = 0; i < per_producer; ++i) {
        const int value = p * per_producer + i;
        while (!queue.enqueue(value)) {
          std::this_thread::yield();
        }
      }
    });
  }

  for (int c = 0; c < 4; ++c) {
    threads.emplace_back([&]() {
      while (consumed.load() < total) {
        int value = 0;
        if (queue.dequeue(value)) {
          sum.fetch_add(value);
          consumed.fetch_add(1);
        } else {
          std::this_thread::yield();
        }
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  assert(consumed.load() == total);
  assert(sum.load() == (static_cast<long long>(total - 1) * total) / 2);
  return 0;
}
