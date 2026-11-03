#pragma once

#include <memory>
#include <vector>

#include "game_position.h"
#include "policy_value_network.h"

// Hyperparameters controlling MCTS exploration and simulation budget.
struct SearchOptions {
  float c_puct = 5.0f;
  int num_playouts = 400;

  // Checks whether c_puct is finite and both parameters are positive.
  bool IsValid() const;
};

// Search statistics for a candidate move at the root position.
struct MoveVisits {
  Action action = -1;
  int visits = 0;
  float prior = 0.0f;
  float value = 0.0f;  // Mean value (Q) from the current player's perspective.
};

// Stateful Monte Carlo Tree Search engine with subtree reuse.
// All calls must be serialized; call Reset() after undo or starting a new game.
class MCTS {
 public:
  explicit MCTS(SearchOptions options = {});
  MCTS(const MCTS&) = delete;
  MCTS& operator=(const MCTS&) = delete;
  ~MCTS();

  // Runs simulations from `position` and returns statistics for all legal moves.
  std::vector<MoveVisits> Search(const GamePosition& position,
                                 PolicyValueEvaluator& evaluator);

  // Advances the root to the child matching `action` to reuse its subtree.
  void ObserveMove(Action action);

  // Discards the current search tree and initializes a fresh root node.
  void Reset();

 private:
  struct TreeNode;

  // Executes a single selection, expansion, evaluation, and backup pass.
  void Playout(GamePosition position, PolicyValueEvaluator& evaluator);

  SearchOptions options_;
  std::unique_ptr<TreeNode> root_;
};
