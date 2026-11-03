#pragma once

#include <string>

#include "alpha_zero.h"
#include "mcts.h"
#include "mcts_state.h"

// AI player: MCTS + AlphaZero.
// PickMove() is called from a background thread; all other methods from main.
class AIPlayer {
 public:
  explicit AIPlayer(const std::string& model_path, float c_puct = 5.0f,
                    int n_playout = 400);

  // Runs MCTS and returns the best action (flat index row*15+col).
  // Also advances the internal tree root to the chosen move.
  int PickMove(const MCTSState& state);

  // Advances MCTS tree after the opponent's (human) move.
  void NotifyMove(int last_move);

  // Resets MCTS tree; call on new game or after undo.
  void Reset();

 private:
  AlphaZero net_;
  MCTS mcts_;
};
