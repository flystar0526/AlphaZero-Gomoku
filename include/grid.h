#pragma once

#include <optional>

#include "board.h"

// Renders the 15x15 Gomoku board using ImGui.
class Grid {
 public:
  // Draws an interactive board and returns the clicked empty cell, if any.
  static std::optional<Action> DrawBoard(
      const Board& board, std::optional<Action> last_move = std::nullopt);

  // Draws a non-interactive board.
  static void DrawBoardReadOnly(const Board& board,
                                std::optional<Action> last_move = std::nullopt);

  static constexpr float kCellSize = 40.0f;
  static constexpr float kStoneRadius = 17.0f;
  static constexpr float kBoardPadding = 28.0f;
  static constexpr float kCanvasSize =
      kBoardPadding * 2.0f + (kBoardSize - 1) * kCellSize;
};
