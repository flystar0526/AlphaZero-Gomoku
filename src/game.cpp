#include "game.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

Game::Game(const Configuration& config, Mode mode)
    : config_(config), mode_(mode) {
  Start();
}

bool Game::IsAIPlayer(Player player) const {
  return mode_ == Mode::kAIvsAI ||
         (mode_ == Mode::kPlayerVsAI && player == Player::kWhite);
}

void Game::Start() {
  position_ = GamePosition();
  record_ = GameRecord();
  state_ = State::kPlaying;
  winner_ = std::nullopt;

  const Clock::time_point now = Clock::now();
  turn_start_time_ = now;
  game_start_time_ = now;
}

void Game::Update() {
  if (state_ != State::kPlaying) return;
  if (config_.time_limit_enabled() && !IsAIPlayer(current_player()) &&
      GetSecondsRemaining() <= 0.0f) {
    winner_ = Opponent(current_player());
    state_ = State::kTimeout;
  }
}

float Game::GetSecondsRemaining() const {
  if (!config_.time_limit_enabled() || IsAIPlayer(current_player())) {
    return -1.0f;
  }
  const float elapsed =
      std::chrono::duration<float>(Clock::now() - turn_start_time_).count();
  const float remaining =
      static_cast<float>(config_.time_limit_seconds()) - elapsed;
  return std::max(0.0f, remaining);
}

bool Game::TryPlaceStone(Action action) {
  // Re-check the timer so expired turns cannot slip a move in between frames.
  Update();
  if (state_ != State::kPlaying || !position_.IsLegalMove(action)) return false;

  record_.AddMove({next_move_number(), current_player(), ActionRow(action),
                   ActionCol(action)});
  position_.PlayMove(action);

  if (position_.winner().has_value()) {
    winner_ = position_.winner();
    state_ = State::kWin;
    return true;
  }
  if (position_.result() == PositionResult::kDraw) {
    state_ = State::kDraw;
    return true;
  }

  turn_start_time_ = Clock::now();
  return true;
}

void Game::Undo() {
  if (state_ != State::kPlaying || mode_ == Mode::kAIvsAI ||
      !config_.undo_enabled() ||
      record_.size() < 2) {
    return;
  }

  record_.RemoveLastMove();
  record_.RemoveLastMove();

  // Rebuild from history to keep last-move and turn state consistent.
  std::optional<GamePosition> restored =
      GamePosition::Rebuild(record_.moves(), record_.size());
  assert(restored.has_value());
  position_ = *restored;
  turn_start_time_ = Clock::now();
}

void Game::Forfeit() {
  if (state_ != State::kPlaying) return;
  winner_ = Opponent(current_player());
  state_ = State::kForfeit;
}

GameRecord Game::FinalizeRecord() {
  assert(state_ != State::kPlaying);

  const int duration_seconds = static_cast<int>(
      std::chrono::duration_cast<std::chrono::seconds>(
          Clock::now() - game_start_time_)
          .count());

  const std::time_t now_time = std::time(nullptr);
  std::tm local_time{};
  localtime_r(&now_time, &local_time);

  std::ostringstream date_stream;
  date_stream << std::put_time(&local_time, "%Y-%m-%d");

  GameMetadata metadata;
  metadata.date = date_stream.str();
  metadata.duration_sec = duration_seconds;
  metadata.total_moves = record_.size();
  metadata.winner = winner_;
  metadata.undo_enabled = config_.undo_enabled() && mode_ != Mode::kAIvsAI;
  metadata.time_limit_enabled =
      config_.time_limit_enabled() && mode_ != Mode::kAIvsAI;
  metadata.time_limit_seconds = config_.time_limit_seconds();

  record_.Finalize(metadata);
  return record_;
}
