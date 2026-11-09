#pragma once

#include <optional>

#include "board.h"

// ImGui-based board renderer.
// All public methods must be called from within an active ImGui window.
class Grid {
 public:
  // Draws the 15x15 gomoku board at the current ImGui cursor position.
  // Stones are drawn as filled circles (black or white).
  // A semi-transparent hover preview is shown when the cell is empty.
  // `last_move` marks the most recently placed stone, if any.
  //
  // Returns the empty intersection clicked this frame, if any.
  static std::optional<Action> DrawBoard(
      const Board& board, std::optional<Action> last_move = std::nullopt);

  // Same as DrawBoard but read-only (no hover preview, no click detection).
  static void DrawBoardReadOnly(const Board& board,
                                std::optional<Action> last_move = std::nullopt);

  // Board geometry constants (pixels).
  static constexpr float kCellSize = 40.0f;
  static constexpr float kStoneRadius = 17.0f;
  static constexpr float kBoardPadding = 28.0f;

  // Total canvas size for one axis: 14 gaps + 2 padding.
  static constexpr float kCanvasSize =
      kBoardPadding * 2.0f + (kBoardSize - 1) * kCellSize;
};
