#pragma once

#include <optional>
#include <string>
#include <vector>

#include "move_record.h"

struct GameMetadata {
  std::string date;      // ISO-8601, e.g. "2026-03-31"
  int duration_sec = 0;  // wall-clock seconds for the full game
  int total_moves = 0;
  std::optional<Player> winner;  // nullopt means draw
  bool undo_enabled = false;
  bool time_limit_enabled = false;
  int time_limit_seconds = 60;
};

// Accumulates moves during a live game and stores final metadata on completion.
class GameRecord {
 public:
  void AddMove(const MoveRecord& move);

  // Removes the last move. Used for undo; requires a nonempty record.
  void RemoveLastMove();

  // Seals the record with final metadata after the game ends.
  void Finalize(const GameMetadata& metadata);

  const GameMetadata& metadata() const;
  const std::vector<MoveRecord>& moves() const;
  int size() const;

 private:
  GameMetadata metadata_;
  std::vector<MoveRecord> moves_;
};
