#include "game_position.h"

#include <cassert>

std::optional<GamePosition> GamePosition::Rebuild(
    const std::vector<MoveRecord>& moves, int num_steps) {
  if (num_steps < 0 || num_steps > static_cast<int>(moves.size())) {
    return std::nullopt;
  }

  GamePosition position;
  for (int i = 0; i < num_steps; ++i) {
    const MoveRecord& move = moves[i];
    if (move.move_number != i + 1 ||
        move.player != position.current_player() ||
        !IsValidCell(move.row, move.col)) {
      return std::nullopt;
    }

    const Action action = ToAction(move.row, move.col);
    if (!position.IsLegalMove(action)) {
      return std::nullopt;
    }
    position.PlayMove(action);
  }
  return position;
}

void GamePosition::PlayMove(Action action) {
  assert(IsLegalMove(action));
  board_.PlaceStone(action, current_player_);
  last_move_ = action;

  if (board_.CheckWin(action, current_player_)) {
    result_ = (current_player_ == Player::kBlack) ? PositionResult::kBlackWin
                                                  : PositionResult::kWhiteWin;
  } else if (board_.IsFull()) {
    result_ = PositionResult::kDraw;
  }

  current_player_ = Opponent(current_player_);
}

bool GamePosition::IsLegalMove(Action action) const {
  return !is_terminal() && board_.IsLegalMove(action);
}

std::vector<Action> GamePosition::LegalActions() const {
  return is_terminal() ? std::vector<Action>{} : board_.LegalActions();
}

std::optional<Player> GamePosition::winner() const {
  switch (result_) {
    case PositionResult::kBlackWin:
      return Player::kBlack;
    case PositionResult::kWhiteWin:
      return Player::kWhite;
    case PositionResult::kOngoing:
    case PositionResult::kDraw:
      return std::nullopt;
  }
  return std::nullopt;
}
