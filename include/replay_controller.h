#pragma once

#include <string>

#include "game_position.h"
#include "game_record.h"

// Loads a .gom game record and steps through its moves.
class ReplayController {
 public:
  bool Load(const std::string& path);

  // Jumps to `step`, clamped to [0, total_moves()].
  void Seek(int step);

  int step() const { return step_; }
  int total_moves() const { return record_.size(); }
  const GamePosition& position() const { return position_; }
  const GameRecord& record() const { return record_; }

 private:
  GameRecord record_;
  int step_ = 0;
  GamePosition position_;
};
