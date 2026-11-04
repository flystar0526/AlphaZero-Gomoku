#include "mcts.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

constexpr Action kNoAction = -1;

}  // namespace

struct MCTS::TreeNode {
  TreeNode* parent = nullptr;
  Action action = kNoAction;
  std::vector<TreeNode> children;
  int visits = 0;
  float q_value = 0.0f;  // Value from the perspective of the player who took `action`.
  float prior = 0.0f;

  TreeNode(TreeNode* parent, Action action, float prior)
      : parent(parent), action(action), prior(prior) {}

  // Selects the child that maximizes the PUCT score: Q + U.
  TreeNode* Select(float c_puct) {
    TreeNode* best_child = nullptr;
    float best_score = -std::numeric_limits<float>::infinity();
    const float exploration_factor =
        c_puct * std::sqrt(static_cast<float>(visits));

    for (TreeNode& child : children) {
      const float u_value =
          exploration_factor * child.prior / (1.0f + child.visits);
      const float score = child.q_value + u_value;
      if (score > best_score) {
        best_score = score;
        best_child = &child;
      }
    }
    return best_child;
  }

  // Propagates the leaf evaluation up to the root, flipping perspective each ply.
  void Backup(float leaf_value) {
    float value = leaf_value;
    for (TreeNode* node = this; node != nullptr; node = node->parent) {
      ++node->visits;
      node->q_value +=
          (value - node->q_value) / static_cast<float>(node->visits);
      value = -value;
    }
  }
};

bool SearchOptions::IsValid() const {
  return std::isfinite(c_puct) && c_puct > 0.0f && num_playouts > 0 &&
         std::isfinite(temperature) && temperature >= 0.0f &&
         num_sampling_moves >= 0 && num_sampling_moves <= kActionCount;
}

MCTS::MCTS(SearchOptions options) : options_(options) {
  if (!options_.IsValid()) {
    throw std::invalid_argument(
        "Search requires valid search and move sampling options");
  }
  Reset();
}

MCTS::~MCTS() = default;

void MCTS::Playout(GamePosition position, PolicyValueEvaluator& evaluator) {
  // 1. Selection: descend the tree until reaching an unexpanded or terminal leaf.
  TreeNode* node = root_.get();
  while (!node->children.empty()) {
    node = node->Select(options_.c_puct);
    position.PlayMove(node->action);
  }

  // 2. Evaluation & Expansion: evaluate leaf value and expand legal actions.
  float leaf_value = 0.0f;
  if (position.is_terminal()) {
    if (position.winner().has_value()) {
      leaf_value =
          (*position.winner() == position.current_player()) ? 1.0f : -1.0f;
    }
  } else {
    const Evaluation evaluation = evaluator.Evaluate(position);
    if (!std::isfinite(evaluation.value) || evaluation.value < -1.0f ||
        evaluation.value > 1.0f) {
      throw std::runtime_error("Evaluator value must be finite and in [-1, 1]");
    }
    leaf_value = evaluation.value;

    for (float prob : evaluation.policy) {
      if (!std::isfinite(prob) || prob < 0.0f) {
        throw std::runtime_error(
            "Evaluator policy must be finite and nonnegative");
      }
    }

    const std::vector<Action> legal_actions = position.LegalActions();
    double policy_sum = 0.0;
    for (Action action : legal_actions) {
      policy_sum += evaluation.policy[action];
    }

    const float uniform_prior = 1.0f / static_cast<float>(legal_actions.size());
    node->children.reserve(legal_actions.size());
    for (Action action : legal_actions) {
      const float prior =
          (policy_sum > 0.0)
              ? static_cast<float>(evaluation.policy[action] / policy_sum)
              : uniform_prior;
      node->children.emplace_back(node, action, prior);
    }
  }

  // 3. Backup: negate leaf_value so `node` stores value for the player who moved into it.
  node->Backup(-leaf_value);
}

std::vector<MoveVisits> MCTS::Search(const GamePosition& position,
                                     PolicyValueEvaluator& evaluator) {
  if (position.is_terminal()) return {};

  for (int i = 0; i < options_.num_playouts; ++i) {
    Playout(position, evaluator);
  }

  std::vector<MoveVisits> move_stats;
  move_stats.reserve(root_->children.size());
  for (const TreeNode& child : root_->children) {
    move_stats.push_back(
        {child.action, child.visits, child.prior, child.q_value});
  }
  return move_stats;
}

void MCTS::ObserveMove(Action action) {
  assert(IsValidAction(action));
  const auto it = std::find_if(
      root_->children.begin(), root_->children.end(),
      [action](const TreeNode& child) { return child.action == action; });
  if (it == root_->children.end()) {
    Reset();
    return;
  }

  auto next_root = std::make_unique<TreeNode>(std::move(*it));
  next_root->parent = nullptr;
  for (TreeNode& child : next_root->children) {
    child.parent = next_root.get();
  }
  root_ = std::move(next_root);
}

void MCTS::Reset() {
  root_ = std::make_unique<TreeNode>(nullptr, kNoAction, 1.0f);
}
