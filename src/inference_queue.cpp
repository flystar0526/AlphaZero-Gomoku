#include "inference_queue.h"

#include <algorithm>
#include <cassert>
#include <vector>

struct InferenceQueue::Request {
  GamePosition position;
  Evaluation result;
  std::chrono::steady_clock::time_point enqueued;
  bool done = false;
};

// ---------------------------------------------------------------------------
// Lifecycle and client requests
// ---------------------------------------------------------------------------

InferenceQueue::InferenceQueue(PolicyValueEvaluator& evaluator, int batch_size,
                               std::chrono::microseconds timeout)
    : evaluator_(evaluator),
      batch_size_(static_cast<size_t>(batch_size)),
      timeout_(timeout),
      evaluator_thread_([this] { Run(); }) {}

InferenceQueue::~InferenceQueue() {
  {
    std::lock_guard lock(mutex_);
    assert(clients_ == 0 && pending_.empty());
    closing_ = true;
  }

  evaluator_cv_.notify_one();
  evaluator_thread_.join();
}

InferenceQueue::Client InferenceQueue::Connect() {
  std::lock_guard lock(mutex_);
  ++clients_;
  return Client(this);
}

InferenceQueue::Stats InferenceQueue::stats() const {
  std::lock_guard lock(mutex_);
  return stats_;
}

Evaluation InferenceQueue::Evaluate(const GamePosition& position) {
  std::unique_lock lock(mutex_);
  if (error_) std::rethrow_exception(error_);

  Request request{position, {}, std::chrono::steady_clock::now()};
  pending_.push_back(&request);
  evaluator_cv_.notify_one();

  client_cv_.wait(lock, [&] { return request.done; });
  if (error_) std::rethrow_exception(error_);

  return request.result;
}

void InferenceQueue::Disconnect() {
  std::lock_guard lock(mutex_);
  --clients_;
  evaluator_cv_.notify_one();
}

// ---------------------------------------------------------------------------
// Batch evaluation loop
// ---------------------------------------------------------------------------

bool InferenceQueue::IsBatchReady() const {
  return pending_.size() >= batch_size_ ||
         static_cast<int>(pending_.size()) >= clients_;
}

void InferenceQueue::Run() {
  std::unique_lock lock(mutex_);

  while (true) {
    // Wait for pending requests or queue shutdown.
    evaluator_cv_.wait(lock, [&] { return !pending_.empty() || closing_; });
    if (pending_.empty()) return;

    evaluator_cv_.wait_until(lock, pending_.front()->enqueued + timeout_,
                             [&] { return IsBatchReady(); });

    // Pop up to `batch_size_` requests for evaluation.
    const size_t count = std::min(batch_size_, pending_.size());
    std::vector<Request*> batch(pending_.begin(), pending_.begin() + count);
    pending_.erase(pending_.begin(), pending_.begin() + count);

    std::vector<GamePosition> positions;
    positions.reserve(batch.size());
    for (const Request* request : batch) {
      positions.push_back(request->position);
    }

    // Run model inference without holding the queue lock.
    lock.unlock();

    std::vector<Evaluation> results;
    std::exception_ptr error;
    const auto start = std::chrono::steady_clock::now();
    try {
      results = evaluator_.EvaluateBatch(positions);
      assert(results.size() == positions.size());
    } catch (...) {
      error = std::current_exception();
    }
    const auto elapsed = std::chrono::steady_clock::now() - start;

    lock.lock();

    // Update statistics and wake waiting clients.
    stats_.inference_ms +=
        std::chrono::duration<double, std::milli>(elapsed).count();
    ++stats_.batches;
    stats_.batched_positions += static_cast<std::int64_t>(positions.size());

    if (error && !error_) error_ = error;

    for (size_t i = 0; i < batch.size(); ++i) {
      if (!error) batch[i]->result = results[i];
      batch[i]->done = true;
    }

    if (error_) {
      for (Request* request : pending_) request->done = true;
      pending_.clear();
    }

    client_cv_.notify_all();
  }
}
