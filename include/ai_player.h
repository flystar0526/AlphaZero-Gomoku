#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <random>

#include "mcts.h"

// Selects moves for an AI player using MCTS.
class AIPlayer {
 public:
  explicit AIPlayer(std::shared_ptr<PolicyValueEvaluator> evaluator,
                    SearchOptions options = {},
                    std::optional<std::uint32_t> random_seed = std::nullopt);

  AIPlayer(const AIPlayer&) = delete;
  AIPlayer& operator=(const AIPlayer&) = delete;

  // Runs MCTS on `position` and returns the chosen move, or std::nullopt if
  // the position is terminal.
  std::optional<Action> PickMove(const GamePosition& position,
                                 SearchTiming* timing = nullptr);

  // Advances the search tree after `action` is played on the board.
  void ObserveMove(Action action);

  // Resets the search tree.
  void Reset();

 private:
  std::shared_ptr<PolicyValueEvaluator> evaluator_;
  MCTS mcts_;
  std::mt19937 random_;
};
