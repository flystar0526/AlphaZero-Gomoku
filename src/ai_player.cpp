#include "ai_player.h"

AIPlayer::AIPlayer(const std::string& model_path, float c_puct, int n_playout)
    : net_(model_path), mcts_(c_puct, n_playout) {}

int AIPlayer::PickMove(const MCTSState& state) {
  auto move_visits = mcts_.GetMoveVisits(state, net_);

  int best_act = move_visits[0].first;
  int best_cnt = move_visits[0].second;

  for (auto& [act, cnt] : move_visits) {
    if (cnt > best_cnt) {
      best_cnt = cnt;
      best_act = act;
    }
  }

  mcts_.UpdateWithMove(best_act);

  return best_act;
}

void AIPlayer::NotifyMove(int last_move) { mcts_.UpdateWithMove(last_move); }

void AIPlayer::Reset() { mcts_.UpdateWithMove(-1); }
