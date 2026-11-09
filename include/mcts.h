#pragma once

#include <memory>
#include <vector>

#include "game_position.h"
#include "policy_value_network.h"

// Hyperparameters for MCTS search and move sampling.
struct SearchOptions {
  float c_puct = 5.0f;
  int num_playouts = 400;
  float temperature = 1.0f;
  int num_sampling_moves = 30;

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

  const SearchOptions& options() const { return options_; }

 private:
  struct TreeNode;

  void Playout(GamePosition position, PolicyValueEvaluator& evaluator,
               SearchTiming* timing);

  SearchOptions options_;
  std::unique_ptr<TreeNode> root_;
};
