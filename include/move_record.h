#pragma once

#include "board.h"

struct MoveRecord {
  int move_number;  // 1-indexed
  Player player;
  int row;  // 0-indexed [0, 14]
  int col;  // 0-indexed [0, 14]
};
