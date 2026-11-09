#pragma once

#include <string>

#include "game_position.h"
#include "game_record.h"

// Loads a .gom file and navigates its moves for interactive replay.
class ReplayController {
 public:
  // Loads and validates the .gom file at path, then moves to step 0.
  // Returns false and keeps the current replay on failure.
  bool Load(const std::string& path);

  // Moves to `step`, clamped to [0, total_moves()]. Step 0 is the empty board;
  // step N is the position after the first N moves.
  void Seek(int step);

  int step() const { return step_; }
  int total_moves() const { return record_.size(); }
  // Position at the current step; last_move() is the stone placed at it.
  const GamePosition& position() const { return position_; }
  const GameRecord& record() const { return record_; }

 private:
  GameRecord record_;
  int step_ = 0;
  GamePosition position_;
};
