#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <mutex>

namespace aiinfra::scheduler {

class TokenBucket {
 public:
  TokenBucket(double refill_per_second, double burst)
      : refill_per_second_(refill_per_second),
        burst_(burst),
        tokens_(burst),
        last_refill_(std::chrono::steady_clock::now()) {}

  bool allow(double cost = 1.0) {
    std::lock_guard<std::mutex> lock(mutex_);
    refill_locked();
    if (tokens_ < cost) {
      return false;
    }
    tokens_ -= cost;
    return true;
  }

 private:
  void refill_locked() {
    const auto now = std::chrono::steady_clock::now();
    const auto elapsed = std::chrono::duration<double>(now - last_refill_).count();
    tokens_ = std::min(burst_, tokens_ + elapsed * refill_per_second_);
    last_refill_ = now;
  }

  const double refill_per_second_;
  const double burst_;
  double tokens_;
  std::chrono::steady_clock::time_point last_refill_;
  std::mutex mutex_;
};

class CircuitBreaker {
 public:
  CircuitBreaker(std::size_t failure_threshold, std::chrono::milliseconds open_duration)
      : failure_threshold_(failure_threshold), open_duration_(open_duration) {}

  bool allow() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (state_ != State::Open) {
      return true;
    }
    if (std::chrono::steady_clock::now() - opened_at_ >= open_duration_) {
      state_ = State::HalfOpen;
      return true;
    }
    return false;
  }

  void record_success() {
    std::lock_guard<std::mutex> lock(mutex_);
    failures_ = 0;
    state_ = State::Closed;
  }

  void record_failure() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++failures_;
    if (failures_ >= failure_threshold_) {
      state_ = State::Open;
      opened_at_ = std::chrono::steady_clock::now();
    }
  }

 private:
  enum class State { Closed, Open, HalfOpen };

  const std::size_t failure_threshold_;
  const std::chrono::milliseconds open_duration_;
  std::size_t failures_{0};
  State state_{State::Closed};
  std::chrono::steady_clock::time_point opened_at_{};
  std::mutex mutex_;
};

}  // namespace aiinfra::scheduler
