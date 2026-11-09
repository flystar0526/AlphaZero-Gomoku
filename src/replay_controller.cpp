#include "replay_controller.h"

#include <algorithm>
#include <optional>
#include <utility>

#include "file_manager.h"

bool ReplayController::Load(const std::string& path) {
  std::optional<GameRecord> loaded = FileManager::Load(path);
  // Verify that the entire move sequence obeys game rules before accepting it.
  if (!loaded || !GamePosition::Rebuild(loaded->moves(), loaded->size())) {
    return false;
  }
  record_ = std::move(*loaded);
  return true;
}

Board ReplayController::BoardAtStep(int step) const {
  return GamePosition::Rebuild(record_.moves(),
                               std::clamp(step, 0, record_.size()))
      .value()
      .board();
}
