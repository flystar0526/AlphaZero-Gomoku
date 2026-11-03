#pragma once

#include <string>

#include "board.h"
#include "game_position.h"
#include "game_record.h"

// Loads a .gom file and reconstructs board states for interactive replay.
// Navigation (cursor, prev/next) is managed by the caller (App).
class ReplayController {
 public:
  // Loads and validates the .gom file at path. Returns false on failure.
  bool Load(const std::string& path);

  // Returns a Board reconstructed by replaying moves[0..step-1].
  // step=0 → empty board, step=total_moves → final position.
  Board BoardAtStep(int step) const;

  int total_moves() const { return record_.size(); }
  const GameRecord& record() const { return record_; }

 private:
  GameRecord record_;
};
