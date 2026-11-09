#include "replay_controller.h"

#include <algorithm>
#include <cassert>
#include <optional>
#include <utility>

#include "file_manager.h"

bool ReplayController::Load(const std::string& path) {
  std::optional<GameRecord> loaded = FileManager::Load(path);
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
  std::optional<GamePosition> position =
      GamePosition::Rebuild(record_.moves(), step_);
  assert(position.has_value());
  position_ = *position;
}
