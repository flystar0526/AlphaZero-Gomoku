#include "mcts.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

namespace {

using Clock = std::chrono::steady_clock;

constexpr Action kNoAction = -1;

// Scoped timer that accumulates elapsed milliseconds into a SearchTiming field.
class PhaseTimer {
 public:
  PhaseTimer(SearchTiming* timing, double SearchTiming::*field)
      : elapsed_ms_(timing ? &(timing->*field) : nullptr) {
    if (elapsed_ms_) start_ = Clock::now();
  }

  ~PhaseTimer() {
    if (elapsed_ms_) {
      *elapsed_ms_ +=
          std::chrono::duration<double, std::milli>(Clock::now() - start_)
              .count();
    }
  }

 private:
  double* elapsed_ms_;
  Clock::time_point start_;
};

// Test-and-test-and-set spinlock guarding one tree node.
class SpinLock {
 public:
  SpinLock() = default;

  // Nodes are only copied or moved while no search is running, so the copy
  // simply starts unlocked. This keeps TreeNode storable in std::vector.
  SpinLock(const SpinLock&) noexcept {}
  SpinLock& operator=(const SpinLock&) noexcept { return *this; }

  void lock() noexcept {
    while (locked_.exchange(true, std::memory_order_acquire)) {
      while (locked_.load(std::memory_order_relaxed)) {
        std::this_thread::yield();
      }
    }
  }

  void unlock() noexcept { locked_.store(false, std::memory_order_release); }

 private:
  std::atomic<bool> locked_{false};
};

// Producer-consumer buffer between search workers and the batch evaluator.
// Workers block in Evaluate() until their batch has been run. A batch is sent
// when `batch_size` positions are queued, when every active worker is
// waiting, or when the oldest request has waited `timeout`.
class InferenceQueue {
 public:
  InferenceQueue(PolicyValueEvaluator& evaluator, int batch_size,
                 std::chrono::microseconds timeout, int num_workers)
      : evaluator_(evaluator),
        batch_size_(static_cast<size_t>(batch_size)),
        timeout_(timeout),
        active_workers_(num_workers) {}

  // Queues `position` and blocks until it has been evaluated. Rethrows the
  // evaluator's error if any batch has failed.
  Evaluation Evaluate(const GamePosition& position) {
    std::unique_lock lock(mutex_);
    if (error_) std::rethrow_exception(error_);

    Request request{position, {}, Clock::now(), false};
    pending_.push_back(&request);
    evaluator_cv_.notify_one();
    worker_cv_.wait(lock, [&] { return request.done; });

    if (error_) std::rethrow_exception(error_);
    return request.result;
  }

  // Called once by each worker when it will submit no more requests.
  void WorkerFinished() {
    std::lock_guard lock(mutex_);
    --active_workers_;
    evaluator_cv_.notify_one();
  }

  // Batch evaluator loop. Returns once every worker has finished and the
  // queue is empty.
  void Run() {
    std::unique_lock lock(mutex_);
    while (true) {
      evaluator_cv_.wait(
          lock, [&] { return !pending_.empty() || active_workers_ == 0; });
      if (pending_.empty()) return;

      evaluator_cv_.wait_until(lock, pending_.front()->enqueued + timeout_,
                               [&] { return IsBatchReady(); });

      const size_t count = std::min(batch_size_, pending_.size());
      std::vector<Request*> batch(pending_.begin(), pending_.begin() + count);
      pending_.erase(pending_.begin(), pending_.begin() + count);

      std::vector<GamePosition> positions;
      positions.reserve(batch.size());
      for (const Request* request : batch) {
        positions.push_back(request->position);
      }

      lock.unlock();
      std::vector<Evaluation> results;
      std::exception_ptr error;
      ++batches_;
      batched_positions_ += static_cast<int>(positions.size());
      try {
        PhaseTimer timer(&timing_, &SearchTiming::inference_ms);
        results = evaluator_.EvaluateBatch(positions);
        if (results.size() != positions.size()) {
          throw std::runtime_error(
              "Evaluator returned the wrong number of evaluations");
        }
      } catch (...) {
        error = std::current_exception();
      }
      lock.lock();

      if (error && !error_) error_ = error;
      for (size_t i = 0; i < batch.size(); ++i) {
        if (!error) batch[i]->result = results[i];
        batch[i]->done = true;
      }
      if (error_) {
        // Fail every waiting request; later requests fail immediately.
        for (Request* request : pending_) request->done = true;
        pending_.clear();
      }
      worker_cv_.notify_all();
    }
  }

  // Evaluator-side counters; read only after Run() has returned.
  double inference_ms() const { return timing_.inference_ms; }
  int batches() const { return batches_; }
  int batched_positions() const { return batched_positions_; }

 private:
  struct Request {
    GamePosition position;
    Evaluation result;
    Clock::time_point enqueued;
    bool done;
  };

  // A batch is ready when full or when no other worker can add to it.
  bool IsBatchReady() const {
    return pending_.size() >= batch_size_ ||
           static_cast<int>(pending_.size()) >= active_workers_;
  }

  PolicyValueEvaluator& evaluator_;
  const size_t batch_size_;
  const std::chrono::microseconds timeout_;

  std::mutex mutex_;
  std::condition_variable evaluator_cv_;
  std::condition_variable worker_cv_;
  std::deque<Request*> pending_;
  int active_workers_;
  std::exception_ptr error_;

  // Touched only by the evaluator thread.
  SearchTiming timing_;
  int batches_ = 0;
  int batched_positions_ = 0;
};

}  // namespace

// ---------------------------------------------------------------------------
// MCTS::TreeNode
// ---------------------------------------------------------------------------

// Locking rule: a node's `lock` guards its `children` vector and the
// statistics of each child. The root's own statistics are guarded by the
// root's lock. Workers never hold two locks at once.
struct MCTS::TreeNode {
  TreeNode* parent = nullptr;
  Action action = kNoAction;
  float prior = 0.0f;

  // Edge statistics, guarded by the parent's lock (the root's own lock).
  int visits = 0;
  float value_sum = 0.0f;
  int in_flight = 0;  // Simulations that passed this node but not backed up.

  SpinLock lock;
  std::vector<TreeNode> children;  // Empty until expanded; then immutable.

  TreeNode(TreeNode* parent, Action action, float prior)
      : parent(parent), action(action), prior(prior) {}

  SpinLock& stats_lock() { return parent ? parent->lock : lock; }

  float MeanValue() const {
    return visits > 0 ? value_sum / static_cast<float>(visits) : 0.0f;
  }
};

// ---------------------------------------------------------------------------
// MCTS::Worker
// ---------------------------------------------------------------------------

// One search thread. Runs playouts on the shared tree until the search
// budget is spent or another thread fails.
class MCTS::Worker {
 public:
  Worker(const SearchOptions& options, TreeNode& root,
         const GamePosition& position, InferenceQueue& queue)
      : options_(options), root_(root), position_(position), queue_(queue) {}

  // Runs one simulation: select, evaluate, expand, and back up.
  void Playout() {
    GamePosition position = position_;
    path_.clear();

    // 1. Selection, applying the pre-update to every node on the path.
    TreeNode* node = &root_;
    {
      PhaseTimer timer(&timing_, &SearchTiming::selection_ms);
      float parent_count;
      {
        std::lock_guard guard(node->lock);
        parent_count = ExplorationCount(*node);
        ++node->in_flight;
      }
      path_.push_back(node);

      while (true) {
        TreeNode* child;
        {
          std::lock_guard guard(node->lock);
          if (node->children.empty()) break;
          child = SelectChild(*node, parent_count);
          parent_count = ExplorationCount(*child);
          ++child->in_flight;
        }
        position.PlayMove(child->action);
        node = child;
        path_.push_back(node);
      }
    }

    // 2. Evaluation and expansion.
    float leaf_value = 0.0f;
    if (position.is_terminal()) {
      if (position.winner().has_value()) {
        leaf_value =
            (*position.winner() == position.current_player()) ? 1.0f : -1.0f;
      }
    } else {
      Evaluation evaluation;
      {
        PhaseTimer timer(&timing_, &SearchTiming::evaluation_ms);
        evaluation = queue_.Evaluate(position);
      }
      leaf_value = evaluation.value;

      PhaseTimer timer(&timing_, &SearchTiming::expansion_ms);
      std::vector<TreeNode> children = MakeChildren(node, position, evaluation);
      std::lock_guard guard(node->lock);
      // Another worker may have expanded this leaf while we waited; keep its
      // children and only back up our value.
      if (node->children.empty()) {
        node->children = std::move(children);
      } else {
        ++timing_.collisions;
      }
    }

    // 3. Backup, reverting the pre-update along the same path.
    PhaseTimer timer(&timing_, &SearchTiming::backup_ms);
    float value = -leaf_value;
    for (auto it = path_.rbegin(); it != path_.rend(); ++it) {
      TreeNode& path_node = **it;
      {
        std::lock_guard guard(path_node.stats_lock());
        --path_node.in_flight;
        ++path_node.visits;
        path_node.value_sum += value;
      }
      value = -value;
    }
    ++timing_.playouts;
  }

  const SearchTiming& timing() const { return timing_; }

 private:
  // Visit count used in the exploration term, including in-flight
  // simulations for virtual loss and WU-PUCT.
  float ExplorationCount(const TreeNode& node) const {
    switch (options_.parallel_mode) {
      case ParallelMode::kVirtualLoss:
        return static_cast<float>(node.visits) +
               options_.virtual_loss * static_cast<float>(node.in_flight);
      case ParallelMode::kWuPuct:
        return static_cast<float>(node.visits + node.in_flight);
      case ParallelMode::kNone:
        break;
    }
    return static_cast<float>(node.visits);
  }

  // Mean value used for exploitation. Virtual loss counts each in-flight
  // simulation as `virtual_loss` lost games.
  float ExploitationValue(const TreeNode& node) const {
    if (options_.parallel_mode != ParallelMode::kVirtualLoss) {
      return node.MeanValue();
    }
    const float loss =
        options_.virtual_loss * static_cast<float>(node.in_flight);
    const float visits = static_cast<float>(node.visits) + loss;
    return visits > 0.0f ? (node.value_sum - loss) / visits : 0.0f;
  }

  // Selects the child with the highest PUCT score (Q + U). Caller holds
  // `node.lock`.
  TreeNode* SelectChild(TreeNode& node, float parent_count) const {
    TreeNode* best_child = nullptr;
    float best_score = -std::numeric_limits<float>::infinity();
    const float exploration_factor = options_.c_puct * std::sqrt(parent_count);

    for (TreeNode& child : node.children) {
      const float u_value = exploration_factor * child.prior /
                            (1.0f + ExplorationCount(child));
      const float score = ExploitationValue(child) + u_value;
      if (score > best_score) {
        best_score = score;
        best_child = &child;
      }
    }
    return best_child;
  }

  // Builds children for every legal move with priors renormalized over legal
  // actions (uniform if the policy gives them no mass).
  static std::vector<TreeNode> MakeChildren(TreeNode* node,
                                            const GamePosition& position,
                                            const Evaluation& evaluation) {
    const std::vector<Action> legal_actions = position.LegalActions();
    double policy_sum = 0.0;
    for (Action action : legal_actions) {
      policy_sum += evaluation.policy[action];
    }

    const float uniform_prior = 1.0f / static_cast<float>(legal_actions.size());
    std::vector<TreeNode> children;
    children.reserve(legal_actions.size());
    for (Action action : legal_actions) {
      const float prior =
          (policy_sum > 0.0)
              ? static_cast<float>(evaluation.policy[action] / policy_sum)
              : uniform_prior;
      children.emplace_back(node, action, prior);
    }
    return children;
  }

  const SearchOptions& options_;
  TreeNode& root_;
  const GamePosition& position_;
  InferenceQueue& queue_;
  SearchTiming timing_;
  std::vector<TreeNode*> path_;
};

// ---------------------------------------------------------------------------
// MCTS
// ---------------------------------------------------------------------------

bool SearchOptions::HasValidFields() const {
  return std::isfinite(c_puct) && c_puct > 0.0f && num_playouts > 0 &&
         std::isfinite(temperature) && temperature >= 0.0f &&
         num_sampling_moves >= 0 && num_sampling_moves <= kActionCount &&
         (search_limit == SearchLimit::kPlayouts ||
          search_limit == SearchLimit::kThinkTime) &&
         think_time_ms > 0 && think_time_ms <= kMaxThinkTimeMs &&
         (parallel_mode == ParallelMode::kNone ||
          parallel_mode == ParallelMode::kVirtualLoss ||
          parallel_mode == ParallelMode::kWuPuct) &&
         std::isfinite(virtual_loss) && virtual_loss > 0.0f &&
         num_threads > 0 && num_threads <= kMaxThreads && batch_size > 0 &&
         batch_timeout_us > 0 && batch_timeout_us <= kMaxBatchTimeoutUs;
}

bool SearchOptions::IsValid() const {
  return HasValidFields() && batch_size <= num_threads;
}

MCTS::MCTS(SearchOptions options) : options_(options) {
  if (!options_.IsValid()) {
    throw std::invalid_argument(
        "Search requires valid search and move sampling options");
  }
  Reset();
}

MCTS::~MCTS() = default;

std::vector<MoveVisits> MCTS::Search(const GamePosition& position,
                                     PolicyValueEvaluator& evaluator,
                                     SearchTiming* timing) {
  if (timing) *timing = {};
  if (position.is_terminal()) return {};

  const int num_workers = options_.num_threads;
  InferenceQueue queue(evaluator, options_.batch_size,
                       std::chrono::microseconds(options_.batch_timeout_us),
                       num_workers);

  std::vector<std::unique_ptr<Worker>> workers;
  for (int i = 0; i < num_workers; ++i) {
    workers.push_back(
        std::make_unique<Worker>(options_, *root_, position, queue));
  }

  // Search budget shared by all workers.
  std::atomic<int> dispatched{0};
  std::atomic<bool> stop{false};
  const Clock::time_point deadline =
      Clock::now() + std::chrono::milliseconds(options_.think_time_ms);
  const auto claim_playout = [&] {
    if (stop.load(std::memory_order_relaxed)) return false;
    if (options_.search_limit == SearchLimit::kThinkTime) {
      return Clock::now() < deadline;
    }
    return dispatched.fetch_add(1, std::memory_order_relaxed) <
           options_.num_playouts;
  };

  std::mutex error_mutex;
  std::exception_ptr first_error;
  const auto record_error = [&](std::exception_ptr error) {
    std::lock_guard lock(error_mutex);
    if (!first_error) first_error = error;
    stop.store(true, std::memory_order_relaxed);
  };

  std::vector<std::thread> threads;
  threads.reserve(num_workers + 1);
  int started = 0;
  try {
    threads.emplace_back([&queue] { queue.Run(); });
    for (; started < num_workers; ++started) {
      threads.emplace_back([&, worker = workers[started].get()] {
        try {
          while (claim_playout()) worker->Playout();
        } catch (...) {
          record_error(std::current_exception());
        }
        queue.WorkerFinished();
      });
    }
  } catch (...) {
    // Thread creation failed: release the evaluator from waiting on workers
    // that never started, then join whatever is running.
    record_error(std::current_exception());
    if (threads.empty()) {
      Reset();
      std::rethrow_exception(first_error);
    }
    for (int i = started; i < num_workers; ++i) queue.WorkerFinished();
  }
  for (std::thread& thread : threads) thread.join();

  if (first_error) {
    // Interrupted playouts leave pre-updates behind, so drop the tree.
    Reset();
    std::rethrow_exception(first_error);
  }
  assert(CountInFlight() == 0);

  if (timing) {
    for (const auto& worker : workers) {
      timing->selection_ms += worker->timing().selection_ms;
      timing->expansion_ms += worker->timing().expansion_ms;
      timing->evaluation_ms += worker->timing().evaluation_ms;
      timing->backup_ms += worker->timing().backup_ms;
      timing->playouts += worker->timing().playouts;
      timing->collisions += worker->timing().collisions;
    }
    timing->inference_ms = queue.inference_ms();
    timing->batches = queue.batches();
    timing->batched_positions = queue.batched_positions();
  }

  std::vector<MoveVisits> move_stats;
  move_stats.reserve(root_->children.size());
  for (const TreeNode& child : root_->children) {
    move_stats.push_back(
        {child.action, child.visits, child.prior, child.MeanValue()});
  }
  return move_stats;
}

void MCTS::ObserveMove(Action action) {
  assert(IsValidAction(action));
  const auto it = std::find_if(
      root_->children.begin(), root_->children.end(),
      [action](const TreeNode& child) { return child.action == action; });
  if (it == root_->children.end()) {
    Reset();
    return;
  }

  auto next_root = std::make_unique<TreeNode>(std::move(*it));
  next_root->parent = nullptr;
  for (TreeNode& child : next_root->children) {
    child.parent = next_root.get();
  }
  root_ = std::move(next_root);
}

void MCTS::Reset() {
  root_ = std::make_unique<TreeNode>(nullptr, kNoAction, 1.0f);
}

int MCTS::CountInFlight() const {
  int total = 0;
  std::vector<const TreeNode*> stack = {root_.get()};
  while (!stack.empty()) {
    const TreeNode* node = stack.back();
    stack.pop_back();
    total += node->in_flight;
    for (const TreeNode& child : node->children) stack.push_back(&child);
  }
  return total;
}
