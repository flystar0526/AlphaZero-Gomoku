#pragma once

#include <chrono>
#include <optional>

#include "configuration.h"
#include "game_position.h"
#include "game_record.h"
#include "move_record.h"

// Non-blocking game session controller for an immediate-mode GUI loop.
class Game {
 public:
  enum class State { kPlaying, kWin, kDraw, kTimeout, kForfeit };

  explicit Game(const Configuration& config);

  // Resets the board, record, and timers for a new game.
  void Start();

  // Checks the turn timer and transitions to kTimeout if time expired.
  void Update();

  // Plays a move at (row, col) if legal and the game is ongoing.
  bool TryPlaceStone(int row, int col);

  // Reverts the last full turn (two plies) so the same player moves again.
  void Undo();

  // Ends the game immediately with the opponent as winner.
  void Forfeit();

  // Seals and returns the game record. Requires the game to have ended.
  GameRecord FinalizeRecord();

  // Returns remaining turn time in seconds, or -1.0f if the timer is disabled.
  float GetSecondsRemaining() const;

  State state() const { return state_; }
  Player current_player() const { return position_.current_player(); }
  const Board& board() const { return position_.board(); }
  const GamePosition& position() const { return position_; }
  int next_move_number() const { return record_.size() + 1; }
  std::optional<Player> winner() const { return winner_; }
  const GameRecord& record() const { return record_; }

 private:
  using Clock = std::chrono::steady_clock;

  Configuration config_;
  GamePosition position_;
  GameRecord record_;

  State state_ = State::kPlaying;
  std::optional<Player> winner_;

  Clock::time_point turn_start_time_;
  Clock::time_point game_start_time_;
};
