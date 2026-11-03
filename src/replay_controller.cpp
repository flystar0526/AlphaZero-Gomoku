#include "replay_controller.h"

#include <algorithm>

#include "file_manager.h"

bool ReplayController::Load(const std::string& path) {
  GameRecord tmp;
  if (!FileManager::Load(path, &tmp)) return false;
  // Verify that the entire move sequence obeys game rules before accepting it.
  if (!GamePosition::Rebuild(tmp.moves(), tmp.size())) return false;
  record_ = tmp;
  return true;
}

Board ReplayController::BoardAtStep(int step) const {
  return GamePosition::Rebuild(record_.moves(),
                               std::clamp(step, 0, record_.size()))
      .value()
      .board();
}
