#pragma once

#include <memory>
#include <vector>

#include "game_position.h"
#include "policy_value_network.h"

// Strategy that steers concurrent search workers onto different paths.
enum class ParallelMode {
  kNone,         // No pre-update; workers read only completed statistics.
  kVirtualLoss,  // N += virtual_loss, W -= virtual_loss on in-flight edges.
  kWuPuct,       // Counts in-flight simulations O in the exploration term.
};

// Budget that ends a search.
enum class SearchLimit {
  kPlayouts,   // Stop after `num_playouts` simulations.
  kThinkTime,  // Stop dispatching simulations after `think_time_ms`.
};

// Hyperparameters for MCTS search and move sampling.
struct SearchOptions {
  static constexpr int kMaxThreads = 64;
  static constexpr int kMaxBatchTimeoutUs = 1'000'000;
  static constexpr int kMaxThinkTimeMs = 60'000;

  float c_puct = 5.0f;
  int num_playouts = 400;
  float temperature = 1.0f;
  int num_sampling_moves = 30;
  SearchLimit search_limit = SearchLimit::kPlayouts;
  int think_time_ms = 1000;
  ParallelMode parallel_mode = ParallelMode::kNone;
  float virtual_loss = 1.0f;
  int num_threads = 1;
  int batch_size = 1;
  int batch_timeout_us = 1000;

  // Checks each field's own range, ignoring cross-field constraints.
  bool HasValidFields() const;

  // Checks every field and requires batch_size <= num_threads.
  bool IsValid() const;
};

// Search statistics for a candidate move at the root.
struct MoveVisits {
  Action action = -1;
  int visits = 0;
  float prior = 0.0f;
  float value = 0.0f;
};

// Accumulated MCTS phase durations in milliseconds for a single search.
struct SearchTiming {
  double selection_ms = 0.0;
  double expansion_ms = 0.0;
  double evaluation_ms = 0.0;
  double backup_ms = 0.0;
};

// Monte Carlo Tree Search engine with subtree reuse.
//
// Each search runs `num_threads` workers over one shared tree (tree
// parallelization with per-node spinlocks). Workers queue leaf positions for
// a batch evaluator thread that sends up to `batch_size` positions to the
// network at once.
class MCTS {
 public:
  explicit MCTS(SearchOptions options = {});
  MCTS(const MCTS&) = delete;
  MCTS& operator=(const MCTS&) = delete;
  ~MCTS();

  // Runs playouts from `position` and returns visit statistics for legal moves.
  // Phase timings are summed over all workers. If the evaluator throws, the
  // tree is reset and the first error is rethrown.
  std::vector<MoveVisits> Search(const GamePosition& position,
                                 PolicyValueEvaluator& evaluator,
                                 SearchTiming* timing = nullptr);

  // Advances the root to the child matching `action`, preserving its subtree.
  void ObserveMove(Action action);

  // Resets the search tree to a fresh root node.
  void Reset();

  // Returns the number of pre-updates (virtual losses or unobserved samples)
  // still applied anywhere in the tree. Always zero between searches.
  int CountInFlight() const;

  const SearchOptions& options() const { return options_; }

 private:
  struct TreeNode;
  class Worker;

  SearchOptions options_;
  std::unique_ptr<TreeNode> root_;
};
