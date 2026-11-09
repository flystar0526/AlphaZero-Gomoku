#include "ai_player.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

AIPlayer::AIPlayer(std::shared_ptr<PolicyValueEvaluator> evaluator,
                   SearchOptions options,
                   std::optional<std::uint32_t> random_seed)
    : evaluator_(std::move(evaluator)),
      mcts_(options),
      random_(random_seed ? *random_seed : std::random_device{}()) {
  if (!evaluator_) {
    throw std::invalid_argument("AI requires a non-null evaluator");
  }
}

std::optional<Action> AIPlayer::PickMove(const GamePosition& position,
                                         SearchTiming* timing) {
  const std::vector<MoveVisits> candidates =
      mcts_.Search(position, *evaluator_, timing);
  if (candidates.empty()) return std::nullopt;

  // Select the move with the highest visit count, breaking ties by higher
  // prior probability and then smaller action index for deterministic play.
  const auto best_move =
      std::max_element(candidates.begin(), candidates.end(),
                       [](const MoveVisits& a, const MoveVisits& b) {
                         return std::make_tuple(a.visits, a.prior, -a.action) <
                                std::make_tuple(b.visits, b.prior, -b.action);
                       });
  const SearchOptions& options = mcts_.options();
  if (options.temperature == 0.0f ||
      position.board().stone_count() >= options.num_sampling_moves ||
      best_move->visits == 0) {
    return best_move->action;
  }

  // Sample a move with probability proportional to visits^(1 / temperature).
  const double exponent = 1.0 / static_cast<double>(options.temperature);
  std::vector<double> weights;
  weights.reserve(candidates.size());
  for (const MoveVisits& move : candidates) {
    weights.push_back(std::pow(static_cast<double>(move.visits) /
                                   static_cast<double>(best_move->visits),
                               exponent));
  }
  std::discrete_distribution<int> distribution(weights.begin(), weights.end());
  return candidates[distribution(random_)].action;
}

void AIPlayer::ObserveMove(Action action) { mcts_.ObserveMove(action); }

void AIPlayer::Reset() { mcts_.Reset(); }
