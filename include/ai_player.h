#pragma once

#include <memory>
#include <optional>

#include "mcts.h"

// Coordinates MCTS simulations and selects moves for the AI player.
// Not thread-safe; all calls must be serialized by the caller.
class AIPlayer {
 public:
  explicit AIPlayer(std::shared_ptr<PolicyValueEvaluator> evaluator,
                    SearchOptions options = {});
  AIPlayer(const AIPlayer&) = delete;
  AIPlayer& operator=(const AIPlayer&) = delete;

  // Searches from `position` and returns the most visited move, or std::nullopt
  // if the position is terminal. Does not advance the internal search tree.
  std::optional<Action> PickMove(const GamePosition& position);

  // Synchronizes the search tree after a move is played on the board.
  void ObserveMove(Action action);

  // Clears the search tree (required on new game or undo).
  void Reset();

 private:
  // May be shared between players; evaluation calls must still be serialized.
  std::shared_ptr<PolicyValueEvaluator> evaluator_;
  MCTS mcts_;
};
