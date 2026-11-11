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

// Batches positions from several client threads into EvaluateBatch calls made
// on one evaluator thread. A batch is sent when `batch_size` positions are
// waiting, when every connected client is waiting, or when the oldest request
// has waited `timeout`.
class InferenceQueue {
 public:
  class Client;

  struct Stats {
    double inference_ms = 0.0;  // Time spent in EvaluateBatch.
    std::int64_t batches = 0;
    std::int64_t batched_positions = 0;
  };

  // Starts the evaluator thread.
  InferenceQueue(PolicyValueEvaluator& evaluator, int batch_size,
                 std::chrono::microseconds timeout);

  InferenceQueue(const InferenceQueue&) = delete;
  InferenceQueue& operator=(const InferenceQueue&) = delete;

  // Stops the evaluator thread. Every client must be gone by then.
  ~InferenceQueue();

  // Registers a client. Connect every client before any of them calls
  // Evaluate, so the queue knows how many requests a batch can wait for.
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

// A handle for submitting positions. Destroying it tells the queue that this
// client will send no more requests, so batches stop waiting for it.
class InferenceQueue::Client {
 public:
  Client(Client&& other) noexcept : queue_(other.queue_) {
    other.queue_ = nullptr;
  }

  Client& operator=(Client&&) = delete;

  ~Client() {
    if (queue_) queue_->Disconnect();
  }

  // Blocks until `position` has been evaluated. Rethrows the evaluator's
  // error if any batch has failed.
  Evaluation Evaluate(const GamePosition& position) {
    return queue_->Evaluate(position);
  }

 private:
  friend class InferenceQueue;

  explicit Client(InferenceQueue* queue) : queue_(queue) {}

  InferenceQueue* queue_;
};
