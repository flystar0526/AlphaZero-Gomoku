#pragma once

#include <memory>
#include <vector>

#include "game_position.h"
#include "policy_value_network.h"

// Hyperparameters controlling MCTS search and move sampling.
struct SearchOptions {
  float c_puct = 5.0f;
  int num_playouts = 400;
  float temperature = 1.0f;
  // Total moves by both players; zero disables move sampling.
  int num_sampling_moves = 30;

  // Checks search limits and finite, nonnegative sampling parameters.
  bool IsValid() const;
};

// Search statistics for a candidate move at the root position.
struct MoveVisits {
  Action action = -1;
  int visits = 0;
  float prior = 0.0f;
  float value = 0.0f;  // Mean value (Q) from the current player's perspective.
};

// Phase durations in milliseconds, accumulated over one search's playouts.
struct SearchTiming {
  double selection_ms = 0.0;
  double expansion_ms = 0.0;
  double evaluation_ms = 0.0;
  double backup_ms = 0.0;
};

// Stateful Monte Carlo Tree Search engine with subtree reuse.
// All calls must be serialized; call Reset() after undo or starting a new game.
class MCTS {
 public:
  explicit MCTS(SearchOptions options = {});
  MCTS(const MCTS&) = delete;
  MCTS& operator=(const MCTS&) = delete;
  ~MCTS();

  // Runs simulations and returns statistics for all legal moves. If supplied,
  // timing is reset on entry; omitting it disables phase timing.
  std::vector<MoveVisits> Search(const GamePosition& position,
                                 PolicyValueEvaluator& evaluator,
                                 SearchTiming* timing = nullptr);

  // Advances the root to the child matching `action` to reuse its subtree.
  void ObserveMove(Action action);

  // Discards the current search tree and initializes a fresh root node.
  void Reset();

  const SearchOptions& options() const { return options_; }

 private:
  struct TreeNode;

  // Executes a single selection, expansion, evaluation, and backup pass.
  void Playout(GamePosition position, PolicyValueEvaluator& evaluator,
               SearchTiming* timing);

  SearchOptions options_;
  std::unique_ptr<TreeNode> root_;
};
