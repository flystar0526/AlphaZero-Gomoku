#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "game_position.h"
#include "policy_value_network.h"

// ---------------------------------------------------------------------------
// Search options
// ---------------------------------------------------------------------------

// Strategy for steering concurrent workers onto distinct search paths.
enum class ParallelMode {
  kNone,
  kVirtualLoss,
  kWuPuct,
};

// Search budget mode.
enum class SearchLimit {
  kPlayouts,
  kThinkTime,
};

// Configuration and UI labels for search enums.
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

  // Returns validation error messages, or an empty list if valid.
  std::vector<std::string> Validate() const;

  bool IsValid() const { return Validate().empty(); }

  // Returns a copy with invalid fields reset to their defaults.
  SearchOptions Sanitized() const;
};

// ---------------------------------------------------------------------------
// Search results and metrics
// ---------------------------------------------------------------------------

// Search statistics for a candidate move at the root.
struct MoveVisits {
  Action action = -1;
  int visits = 0;
  float prior = 0.0f;
  float value = 0.0f;
};

// Accumulated MCTS phase durations (ms) and search counters.
struct SearchTiming {
  double selection_ms = 0.0;
  double expansion_ms = 0.0;
  double evaluation_ms = 0.0;
  double inference_ms = 0.0;
  double backup_ms = 0.0;

  std::int64_t playouts = 0;
  std::int64_t batches = 0;
  std::int64_t batched_positions = 0;
  std::int64_t collisions = 0;

  template <typename Visitor>
  static void ForEachDuration(Visitor&& visit) {
    visit("selection", &SearchTiming::selection_ms);
    visit("expansion", &SearchTiming::expansion_ms);
    visit("evaluation", &SearchTiming::evaluation_ms);
    visit("inference", &SearchTiming::inference_ms);
    visit("backup", &SearchTiming::backup_ms);
  }

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

// ---------------------------------------------------------------------------
// MCTS engine
// ---------------------------------------------------------------------------

// Monte Carlo Tree Search engine with subtree reuse and parallel playouts.
class MCTS {
 public:
  explicit MCTS(SearchOptions options = {});
  MCTS(const MCTS&) = delete;
  MCTS& operator=(const MCTS&) = delete;
  ~MCTS();

  // Runs playouts from `position` and returns visit statistics for legal moves.
  std::vector<MoveVisits> Search(const GamePosition& position,
                                 PolicyValueEvaluator& evaluator,
                                 SearchTiming* timing = nullptr);

  // Advances the root to the child matching `action`, preserving its subtree.
  void ObserveMove(Action action);

  // Resets the search tree to a fresh root node.
  void Reset();

  // Returns the number of in-flight simulations across the tree.
  int CountInFlight() const;

  const SearchOptions& options() const { return options_; }

 private:
  struct TreeNode;
  class Worker;

  SearchOptions options_;
  std::unique_ptr<TreeNode> root_;
};
