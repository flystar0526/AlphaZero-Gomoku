#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "board.h"
#include "move_record.h"

enum class PositionResult : std::uint8_t {
  kOngoing,
  kBlackWin,
  kWhiteWin,
  kDraw,
};

// Copyable board state, side to move, and game outcome used by play, replay,
// and MCTS.
class GamePosition {
 public:
  // Rebuilds a position by replaying the first `num_steps` moves, or returns
  // std::nullopt if any move is invalid.
  static std::optional<GamePosition> Rebuild(
      const std::vector<MoveRecord>& moves, int num_steps);

  // Plays `action` for the current player, updates the outcome, and switches
  // turns.
  void PlayMove(Action action);

  bool IsLegalMove(Action action) const;
  std::vector<Action> LegalActions() const;
  std::optional<Player> winner() const;

  const Board& board() const { return board_; }
  Player current_player() const { return current_player_; }
  std::optional<Action> last_move() const { return last_move_; }
  PositionResult result() const { return result_; }
  bool is_terminal() const { return result_ != PositionResult::kOngoing; }

 private:
  Board board_;
  Player current_player_ = Player::kBlack;
  std::optional<Action> last_move_;
  PositionResult result_ = PositionResult::kOngoing;
};
