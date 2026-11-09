#include "mcts.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

namespace {

constexpr Action kNoAction = -1;

// Scoped timer that accumulates elapsed milliseconds into a SearchTiming field.
class PhaseTimer {
 public:
  PhaseTimer(SearchTiming* timing, double SearchTiming::*field)
      : elapsed_ms_(timing ? &(timing->*field) : nullptr) {
    if (elapsed_ms_) start_ = Clock::now();
  }

  ~PhaseTimer() {
    if (elapsed_ms_) {
      *elapsed_ms_ +=
          std::chrono::duration<double, std::milli>(Clock::now() - start_)
              .count();
    }
  }

 private:
  using Clock = std::chrono::steady_clock;
  double* elapsed_ms_;
  Clock::time_point start_;
};

}  // namespace

// ---------------------------------------------------------------------------
// MCTS::TreeNode
// ---------------------------------------------------------------------------

struct MCTS::TreeNode {
  TreeNode* parent = nullptr;
  Action action = kNoAction;
  std::vector<TreeNode> children;
  int visits = 0;
  float q_value = 0.0f;
  float prior = 0.0f;

  TreeNode(TreeNode* parent, Action action, float prior)
      : parent(parent), action(action), prior(prior) {}

  // Selects the child with the highest PUCT score (Q + U).
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

  // Updates visit counts and mean values along the path to the root.
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

// ---------------------------------------------------------------------------
// MCTS
// ---------------------------------------------------------------------------

bool SearchOptions::HasValidFields() const {
  return std::isfinite(c_puct) && c_puct > 0.0f && num_playouts > 0 &&
         std::isfinite(temperature) && temperature >= 0.0f &&
         num_sampling_moves >= 0 && num_sampling_moves <= kActionCount &&
         (search_limit == SearchLimit::kPlayouts ||
          search_limit == SearchLimit::kThinkTime) &&
         think_time_ms > 0 && think_time_ms <= kMaxThinkTimeMs &&
         (parallel_mode == ParallelMode::kNone ||
          parallel_mode == ParallelMode::kVirtualLoss ||
          parallel_mode == ParallelMode::kWuPuct) &&
         std::isfinite(virtual_loss) && virtual_loss > 0.0f &&
         num_threads > 0 && num_threads <= kMaxThreads && batch_size > 0 &&
         batch_timeout_us > 0 && batch_timeout_us <= kMaxBatchTimeoutUs;
}

bool SearchOptions::IsValid() const {
  return HasValidFields() && batch_size <= num_threads;
}

MCTS::MCTS(SearchOptions options) : options_(options) {
  if (!options_.IsValid()) {
    throw std::invalid_argument(
        "Search requires valid search and move sampling options");
  }
  Reset();
}

MCTS::~MCTS() = default;

void MCTS::Playout(GamePosition position, PolicyValueEvaluator& evaluator,
                   SearchTiming* timing) {
  // 1. Selection.
  TreeNode* node = root_.get();
  {
    PhaseTimer timer(timing, &SearchTiming::selection_ms);
    while (!node->children.empty()) {
      node = node->Select(options_.c_puct);
      position.PlayMove(node->action);
    }
  }

  // 2. Evaluation and expansion.
  float leaf_value = 0.0f;
  if (position.is_terminal()) {
    if (position.winner().has_value()) {
      leaf_value =
          (*position.winner() == position.current_player()) ? 1.0f : -1.0f;
    }
  } else {
    Evaluation evaluation;
    {
      PhaseTimer timer(timing, &SearchTiming::evaluation_ms);
      evaluation = evaluator.Evaluate(position);
    }
    leaf_value = evaluation.value;

    PhaseTimer timer(timing, &SearchTiming::expansion_ms);
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

  // 3. Backup.
  PhaseTimer timer(timing, &SearchTiming::backup_ms);
  node->Backup(-leaf_value);
}

std::vector<MoveVisits> MCTS::Search(const GamePosition& position,
                                     PolicyValueEvaluator& evaluator,
                                     SearchTiming* timing) {
  if (timing) *timing = {};
  if (position.is_terminal()) return {};

  for (int i = 0; i < options_.num_playouts; ++i) {
    Playout(position, evaluator, timing);
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
