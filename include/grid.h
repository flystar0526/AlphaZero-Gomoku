#pragma once

#include <utility>

#include "board.h"

// ImGui-based board renderer.
// All public methods must be called from within an active ImGui window.
class Grid {
 public:
  // Draws the 15x15 gomoku board at the current ImGui cursor position.
  // Stones are drawn as filled circles (black or white).
  // A semi-transparent hover preview is shown when the cell is empty.
  // last_row/last_col: 0-indexed position of the most recently placed stone
  // (pass -1 to disable the indicator).
  //
  // Returns the (row, col) of the intersection clicked this frame,
  // or {-1, -1} if no click occurred.
  static std::pair<int, int> DrawBoard(const Board& board,
                                       int last_row = -1, int last_col = -1);

  // Same as DrawBoard but read-only (no hover preview, no click detection).
  static void DrawBoardReadOnly(const Board& board,
                                int last_row = -1, int last_col = -1);

  // Board geometry constants (pixels).
  static constexpr float kCellSize = 40.0f;
  static constexpr float kStoneRadius = 17.0f;
  static constexpr float kBoardPadding = 28.0f;

  // Total canvas size for one axis: 14 gaps + 2 padding.
  static constexpr float kCanvasSize =
      kBoardPadding * 2.0f + (kBoardSize - 1) * kCellSize;
};
