#include "ai_player.h"

#include <algorithm>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

AIPlayer::AIPlayer(std::unique_ptr<PolicyValueEvaluator> evaluator,
                   SearchOptions options)
    : evaluator_(std::move(evaluator)), mcts_(options) {
  if (!evaluator_) {
    throw std::invalid_argument("AI requires a non-null evaluator");
  }
}

std::optional<Action> AIPlayer::PickMove(const GamePosition& position) {
  const std::vector<MoveVisits> candidates = mcts_.Search(position, *evaluator_);
  if (candidates.empty()) return std::nullopt;

  // Select the move with the highest visit count, breaking ties by higher
  // prior probability and then smaller action index for deterministic play.
  const auto best_move = std::max_element(
      candidates.begin(), candidates.end(),
      [](const MoveVisits& a, const MoveVisits& b) {
        return std::make_tuple(a.visits, a.prior, -a.action) <
               std::make_tuple(b.visits, b.prior, -b.action);
      });
  return best_move->action;
}

void AIPlayer::ObserveMove(Action action) { mcts_.ObserveMove(action); }

void AIPlayer::Reset() { mcts_.Reset(); }
