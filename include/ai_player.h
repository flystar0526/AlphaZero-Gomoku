#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <random>

#include "mcts.h"

// Coordinates MCTS simulations and selects moves for the AI player.
// Not thread-safe; all calls must be serialized by the caller.
class AIPlayer {
 public:
  explicit AIPlayer(std::shared_ptr<PolicyValueEvaluator> evaluator,
                    SearchOptions options = {},
                    std::optional<std::uint32_t> random_seed = std::nullopt);
  AIPlayer(const AIPlayer&) = delete;
  AIPlayer& operator=(const AIPlayer&) = delete;

  // Samples moves by visit count within the sampling move limit, then selects
  // the most visited move. Returns std::nullopt if terminal. Does not advance
  // the search tree.
  std::optional<Action> PickMove(const GamePosition& position);

  // Synchronizes the search tree after a move is played on the board.
  void ObserveMove(Action action);

  // Clears the search tree (required on new game or undo).
  void Reset();

 private:
  // May be shared between players; evaluation calls must still be serialized.
  std::shared_ptr<PolicyValueEvaluator> evaluator_;
  MCTS mcts_;
  std::mt19937 random_;
};
