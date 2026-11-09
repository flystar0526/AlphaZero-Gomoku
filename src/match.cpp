#include "match.h"

#include <cassert>
#include <stdexcept>
#include <utility>

namespace {

size_t Index(Player color) { return static_cast<size_t>(color); }

}  // namespace

Seats DefaultSeats(Game::Mode mode) {
  switch (mode) {
    case Game::Mode::kPlayerVsPlayer:
      return {std::nullopt, std::nullopt};
    case Game::Mode::kPlayerVsAI:
      return {std::nullopt, AIPlayerId::kPlayer1};
    case Game::Mode::kAIvsAI:
      return {AIPlayerId::kPlayer1, AIPlayerId::kPlayer2};
  }
  return {};
}

Match::Match(const Configuration& config, Game::Mode mode, Seats seats,
             std::shared_ptr<PolicyValueEvaluator> evaluator)
    : game_(config, mode), seats_(seats) {
  for (Player color : {Player::kBlack, Player::kWhite}) {
    assert(seats_[Index(color)].has_value() == game_.IsAIPlayer(color));
    if (const auto id = seats_[Index(color)]) {
      // Active AI players share one evaluator instance.
      players_[Index(color)] =
          std::make_unique<AIPlayer>(evaluator, config.ai_options(*id));
    }
  }
}

Match::Match(const Configuration& config, Game::Mode mode,
             std::shared_ptr<PolicyValueEvaluator> evaluator)
    : Match(config, mode, DefaultSeats(mode), std::move(evaluator)) {}

bool Match::Play(Action action) {
  if (!game_.TryPlaceStone(action)) return false;
  for (auto& player : players_) {
    if (player) player->ObserveMove(action);
  }
  return true;
}

std::optional<Action> Match::Think(SearchTiming* timing) {
  assert(IsAITurn());
  return players_[Index(game_.current_player())]->PickMove(game_.position(),
                                                           timing);
}

Action Match::PlayAIMove(SearchTiming* timing) {
  const std::optional<Action> action = Think(timing);
  if (!action || !Play(*action)) {
    throw std::runtime_error("AI returned no legal move");
  }
  return *action;
}

bool Match::IsAITurn() const {
  return players_[Index(game_.current_player())] != nullptr;
}

std::optional<AIPlayerId> Match::seat(Player color) const {
  return seats_[Index(color)];
}

void Match::Undo() {
  game_.Undo();
  ResetAIs();
}

void Match::Restart() {
  game_.Start();
  ResetAIs();
}

void Match::ResetAIs() {
  for (auto& player : players_) {
    if (player) player->Reset();
  }
}
