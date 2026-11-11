#include "mcts.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "inference_queue.h"

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

// Joins its threads on destruction, including during stack unwinding.
class ThreadGroup {
 public:
  explicit ThreadGroup(size_t capacity) { threads_.reserve(capacity); }

  ThreadGroup(const ThreadGroup&) = delete;
  ThreadGroup& operator=(const ThreadGroup&) = delete;

  ~ThreadGroup() {
    for (std::thread& thread : threads_) thread.join();
  }

  template <typename Function>
  void Spawn(Function&& function) {
    threads_.emplace_back(std::forward<Function>(function));
  }

 private:
  std::vector<std::thread> threads_;
};

// A validation rule for SearchOptions: the check, the message shown when it
// fails, and the field reset to its default to repair it.
struct OptionRule {
  bool (*holds)(const SearchOptions&);
  const char* message;
  void (*reset)(SearchOptions&);
};

template <auto Field>
void ResetField(SearchOptions& options) {
  options.*Field = SearchOptions{}.*Field;
}

// Cross-field rules come last so Sanitized() checks them after every field
// has its own valid value.
const OptionRule kOptionRules[] = {
    {[](const SearchOptions& o) { return o.num_playouts > 0; },
     "num_playouts must be > 0", ResetField<&SearchOptions::num_playouts>},
    {[](const SearchOptions& o) { return o.think_time_ms > 0; },
     "think_time_ms must be > 0", ResetField<&SearchOptions::think_time_ms>},
    {[](const SearchOptions& o) {
       return std::isfinite(o.c_puct) && o.c_puct > 0.0f;
     },
     "c_puct must be > 0", ResetField<&SearchOptions::c_puct>},
    {[](const SearchOptions& o) {
       return std::isfinite(o.temperature) && o.temperature >= 0.0f;
     },
     "temperature must be >= 0", ResetField<&SearchOptions::temperature>},
    {[](const SearchOptions& o) {
       return o.num_sampling_moves >= 0 &&
              o.num_sampling_moves <= kActionCount;
     },
     "num_sampling_moves must be in 0..225",
     ResetField<&SearchOptions::num_sampling_moves>},
    {[](const SearchOptions& o) {
       return std::isfinite(o.virtual_loss) && o.virtual_loss > 0.0f;
     },
     "virtual_loss must be > 0", ResetField<&SearchOptions::virtual_loss>},
    {[](const SearchOptions& o) {
       return o.num_threads > 0 && o.num_threads <= SearchOptions::kMaxThreads;
     },
     "num_threads must be in 1..64", ResetField<&SearchOptions::num_threads>},
    {[](const SearchOptions& o) { return o.batch_timeout_us > 0; },
     "batch_timeout_us must be > 0",
     ResetField<&SearchOptions::batch_timeout_us>},
    {[](const SearchOptions& o) {
       return o.batch_size > 0 && o.batch_size <= o.num_threads;
     },
     "batch_size must be in 1..num_threads",
     ResetField<&SearchOptions::batch_size>},
};

static_assert(kActionCount == 225 && SearchOptions::kMaxThreads == 64,
              "Update the option rule messages");

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
         const GamePosition& position)
      : options_(options), root_(root), position_(position) {}

  // Runs one simulation: select, evaluate, expand, and back up.
  void Playout(InferenceQueue::Client& queue) {
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
        evaluation = queue.Evaluate(position);
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
  SearchTiming timing_;
  std::vector<TreeNode*> path_;
};

// ---------------------------------------------------------------------------
// MCTS
// ---------------------------------------------------------------------------

std::vector<std::string> SearchOptions::Validate() const {
  std::vector<std::string> violations;
  for (const OptionRule& rule : kOptionRules) {
    if (!rule.holds(*this)) violations.emplace_back(rule.message);
  }
  return violations;
}

SearchOptions SearchOptions::Sanitized() const {
  SearchOptions options = *this;
  for (const OptionRule& rule : kOptionRules) {
    if (!rule.holds(options)) rule.reset(options);
  }
  assert(options.IsValid());
  return options;
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
                       std::chrono::microseconds(options_.batch_timeout_us));

  // Connect every client before any worker starts so the first batch waits
  // for all of them.
  std::vector<Worker> workers;
  std::vector<InferenceQueue::Client> clients;
  workers.reserve(num_workers);
  clients.reserve(num_workers);
  for (int i = 0; i < num_workers; ++i) {
    workers.emplace_back(options_, *root_, position);
    clients.push_back(queue.Connect());
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

  {
    ThreadGroup threads(num_workers);
    for (int i = 0; i < num_workers; ++i) {
      // Each thread owns its client, so the queue stops waiting for a worker
      // as soon as that worker's thread ends.
      threads.Spawn([&, worker = &workers[i],
                     client = std::move(clients[i])]() mutable {
        try {
          while (claim_playout()) worker->Playout(client);
        } catch (...) {
          std::lock_guard lock(error_mutex);
          if (!first_error) first_error = std::current_exception();
          stop.store(true, std::memory_order_relaxed);
        }
      });
    }
  }

  if (first_error) {
    // Interrupted playouts leave pre-updates behind, so drop the tree.
    Reset();
    std::rethrow_exception(first_error);
  }
  assert(CountInFlight() == 0);

  if (timing) {
    for (const Worker& worker : workers) *timing += worker.timing();
    const InferenceQueue::Stats stats = queue.stats();
    timing->inference_ms = stats.inference_ms;
    timing->batches = stats.batches;
    timing->batched_positions = stats.batched_positions;
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
