#pragma once

#include <chrono>
#include <optional>

#include "configuration.h"
#include "game_position.h"
#include "game_record.h"
#include "move_record.h"

// Manages rules, turn timers, move history, and outcome for a single game.
class Game {
 public:
  enum class State { kPlaying, kWin, kDraw, kTimeout, kForfeit };
  enum class Mode { kPlayerVsPlayer, kPlayerVsAI, kAIvsAI };

  explicit Game(const Configuration& config,
                Mode mode = Mode::kPlayerVsPlayer);

  bool IsAIPlayer(Player player) const;
  Mode mode() const { return mode_; }

  // Lifecycle and move actions.
  void Start();
  void Update();
  bool TryPlaceStone(Action action);
  void Undo();
  void Forfeit();

  // Seals and returns the completed game record.
  GameRecord FinalizeRecord();

  // Returns remaining turn time in seconds, or -1.0f if untimed.
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
  Mode mode_;
  GamePosition position_;
  GameRecord record_;

  State state_ = State::kPlaying;
  std::optional<Player> winner_;

  Clock::time_point turn_start_time_;
  Clock::time_point game_start_time_;
};
