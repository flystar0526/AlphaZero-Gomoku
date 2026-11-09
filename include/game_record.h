#pragma once

#include <optional>
#include <string>
#include <vector>

#include "move_record.h"

struct GameMetadata {
  std::string date;
  int duration_sec = 0;
  int total_moves = 0;
  std::optional<Player> winner;
  bool undo_enabled = false;
  bool time_limit_enabled = false;
  int time_limit_seconds = 60;
};

// Stores the move history and metadata of a game.
class GameRecord {
 public:
  void AddMove(const MoveRecord& move);
  void RemoveLastMove();
  void Finalize(const GameMetadata& metadata);

  const GameMetadata& metadata() const;
  const std::vector<MoveRecord>& moves() const;
  int size() const;

 private:
  GameMetadata metadata_;
  std::vector<MoveRecord> moves_;
};
