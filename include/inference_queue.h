#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <mutex>
#include <thread>

#include "game_position.h"
#include "policy_value_network.h"

// Collects evaluation requests from worker threads into batched evaluator calls.
class InferenceQueue {
 public:
  class Client;

  // Accumulated batch evaluation metrics.
  struct Stats {
    double inference_ms = 0.0;
    std::int64_t batches = 0;
    std::int64_t batched_positions = 0;
  };

  InferenceQueue(PolicyValueEvaluator& evaluator, int batch_size,
                 std::chrono::microseconds timeout);
  InferenceQueue(const InferenceQueue&) = delete;
  InferenceQueue& operator=(const InferenceQueue&) = delete;
  ~InferenceQueue();

  // Registers a worker client with the queue.
  Client Connect();

  Stats stats() const;

 private:
  struct Request;

  Evaluation Evaluate(const GamePosition& position);
  void Disconnect();
  void Run();
  bool IsBatchReady() const;

  PolicyValueEvaluator& evaluator_;
  const size_t batch_size_;
  const std::chrono::microseconds timeout_;

  mutable std::mutex mutex_;
  std::condition_variable evaluator_cv_;
  std::condition_variable client_cv_;
  std::deque<Request*> pending_;
  int clients_ = 0;
  bool closing_ = false;
  std::exception_ptr error_;
  Stats stats_;

  std::thread evaluator_thread_;
};

// Worker handle for submitting positions to an InferenceQueue.
class InferenceQueue::Client {
 public:
  Client(Client&& other) noexcept : queue_(other.queue_) {
    other.queue_ = nullptr;
  }

  Client& operator=(Client&&) = delete;

  ~Client() {
    if (queue_) queue_->Disconnect();
  }

  // Submits `position` for batched evaluation and returns its result.
  Evaluation Evaluate(const GamePosition& position) {
    return queue_->Evaluate(position);
  }

 private:
  friend class InferenceQueue;

  explicit Client(InferenceQueue* queue) : queue_(queue) {}

  InferenceQueue* queue_;
};
