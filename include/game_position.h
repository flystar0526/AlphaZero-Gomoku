#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "board.h"
#include "move_record.h"

// Outcome of a Gomoku position.
enum class PositionResult : std::uint8_t {
  kOngoing,
  kBlackWin,
  kWhiteWin,
  kDraw,
};

// Lightweight, copyable game rules state shared by live play, replay, and MCTS.
class GamePosition {
 public:
  // Reconstructs a position by replaying the first `num_steps` moves.
  // Returns std::nullopt if move order, bounds, or legality checks fail.
  static std::optional<GamePosition> Rebuild(
      const std::vector<MoveRecord>& moves, int num_steps);

  // Places a stone for the current player, updates the game outcome, and
  // advances the turn. Requires an ongoing game and a legal action.
  void PlayMove(Action action);

  // Checks if the game is ongoing and `action` is an empty cell on the board.
  bool IsLegalMove(Action action) const;

  // Returns all legal actions, or an empty list if the game has ended.
  std::vector<Action> LegalActions() const;

  // Returns the winning player, or std::nullopt if ongoing or drawn.
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
