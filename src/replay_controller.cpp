#include "replay_controller.h"

#include <algorithm>
#include <cassert>
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
  step_ = 0;
  position_ = GamePosition();
  return true;
}

void ReplayController::Seek(int step) {
  step_ = std::clamp(step, 0, record_.size());
  // Load() already verified every prefix of the record.
  std::optional<GamePosition> position =
      GamePosition::Rebuild(record_.moves(), step_);
  assert(position.has_value());
  position_ = *position;
}
