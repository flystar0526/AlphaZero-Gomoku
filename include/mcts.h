#pragma once

#include <cstdint>
#include <memory>
#include <string>
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

// Names of each enum value, indexed by value. Used by both gomoku.cfg and the
// Settings window.
inline constexpr const char* kParallelModeNames[] = {"none", "virtual_loss",
                                                     "wu_puct"};
inline constexpr const char* kSearchLimitNames[] = {"playouts", "think_time"};

// Hyperparameters for MCTS search and move sampling.
struct SearchOptions {
  static constexpr int kMaxThreads = 64;

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

  // Returns one message per violated rule, or nothing when valid.
  std::vector<std::string> Validate() const;

  bool IsValid() const { return Validate().empty(); }

  // Returns a copy with every field that breaks a rule reset to its default.
  SearchOptions Sanitized() const;
};

// Search statistics for a candidate move at the root.
struct MoveVisits {
  Action action = -1;
  int visits = 0;
  float prior = 0.0f;
  float value = 0.0f;
};

// Accumulated MCTS phase durations in milliseconds and counters for a single
// search. Phase durations are summed over all workers, so with several
// threads they measure CPU time and can exceed the wall-clock search time.
struct SearchTiming {
  double selection_ms = 0.0;
  double expansion_ms = 0.0;
  double evaluation_ms = 0.0;  // Waiting for a batch result, incl. queueing.
  double inference_ms = 0.0;   // Batch evaluator time spent in EvaluateBatch.
  double backup_ms = 0.0;
  std::int64_t playouts = 0;           // Completed simulations.
  std::int64_t batches = 0;            // EvaluateBatch calls.
  std::int64_t batched_positions = 0;  // Positions sent over all batches.
  std::int64_t collisions = 0;  // Leaves already expanded by another worker.

  // Visits each duration field with its name (without the `_ms` suffix).
  template <typename Visitor>
  static void ForEachDuration(Visitor&& visit) {
    visit("selection", &SearchTiming::selection_ms);
    visit("expansion", &SearchTiming::expansion_ms);
    visit("evaluation", &SearchTiming::evaluation_ms);
    visit("inference", &SearchTiming::inference_ms);
    visit("backup", &SearchTiming::backup_ms);
  }

  // Visits each counter field with its name.
  template <typename Visitor>
  static void ForEachCounter(Visitor&& visit) {
    visit("playouts", &SearchTiming::playouts);
    visit("batches", &SearchTiming::batches);
    visit("batched_positions", &SearchTiming::batched_positions);
    visit("collisions", &SearchTiming::collisions);
  }

  SearchTiming& operator+=(const SearchTiming& other) {
    const auto add = [&](const char*, auto field) {
      this->*field += other.*field;
    };
    ForEachDuration(add);
    ForEachCounter(add);
    return *this;
  }
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
