#include "grid.h"

#include "imgui.h"

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

namespace {

// Board background colour (warm wood tone).
constexpr ImU32 kBoardBg = IM_COL32(205, 160, 80, 255);
// Grid line colour.
constexpr ImU32 kLineColor = IM_COL32(60, 40, 10, 200);
// Star-point dot colour.
constexpr ImU32 kStarColor = IM_COL32(60, 40, 10, 255);
// Last-move indicator.
constexpr ImU32 kLastMoveDot = IM_COL32(220, 50, 50, 230);

// Star-point positions (0-indexed) for a standard 15×15 board.
constexpr int kStarPoints[][2] = {
    {2, 2},  {2, 7},  {2, 12},
    {7, 2},  {7, 7},  {7, 12},
    {12, 2}, {12, 7}, {12, 12},
};

// Screen position of (row, col) intersection relative to the board origin.
ImVec2 IntersectionPos(ImVec2 origin, int row, int col) {
  return {origin.x + Grid::kBoardPadding + col * Grid::kCellSize,
          origin.y + Grid::kBoardPadding + row * Grid::kCellSize};
}

void DrawStone(ImDrawList* dl, ImVec2 center, Cell cell) {
  if (cell == Cell::kBlack) {
    // Black stone: dark fill + subtle highlight ring.
    dl->AddCircleFilled(center, Grid::kStoneRadius,
                        IM_COL32(20, 20, 20, 255), 40);
    dl->AddCircle(center, Grid::kStoneRadius,
                  IM_COL32(90, 90, 90, 160), 40, 1.2f);
  } else {
    // White stone: ivory fill + thin dark outline + inner shadow.
    dl->AddCircleFilled(center, Grid::kStoneRadius,
                        IM_COL32(245, 243, 228, 255), 40);
    dl->AddCircle(center, Grid::kStoneRadius,
                  IM_COL32(30, 30, 30, 220), 40, 1.5f);
    dl->AddCircle(center, Grid::kStoneRadius * 0.78f,
                  IM_COL32(180, 170, 145, 80), 32, 1.0f);
  }
}

// Core drawing logic. interactive=true enables hover + click.
std::optional<Action> DrawBoardImpl(const Board& board, bool interactive,
                                    std::optional<Action> last_move) {
  ImDrawList* dl = ImGui::GetWindowDrawList();
  ImVec2 origin = ImGui::GetCursorScreenPos();

  const float sz = Grid::kCanvasSize;

  // Background
  dl->AddRectFilled(origin, {origin.x + sz, origin.y + sz}, kBoardBg, 4.0f);

  // Grid lines
  for (int i = 0; i < kBoardSize; ++i) {
    // Vertical line at column i.
    ImVec2 a = IntersectionPos(origin, 0, i);
    ImVec2 b = IntersectionPos(origin, kBoardSize - 1, i);
    dl->AddLine(a, b, kLineColor, 1.0f);
    // Horizontal line at row i.
    ImVec2 c = IntersectionPos(origin, i, 0);
    ImVec2 d = IntersectionPos(origin, i, kBoardSize - 1);
    dl->AddLine(c, d, kLineColor, 1.0f);
  }

  // Star points
  for (auto& sp : kStarPoints) {
    ImVec2 center = IntersectionPos(origin, sp[0], sp[1]);
    dl->AddCircleFilled(center, 4.0f, kStarColor, 12);
  }

  // Hover preview (drawn before stones so it sits behind them)
  std::optional<Action> hover;
  if (interactive && ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows |
                                            ImGuiHoveredFlags_AllowWhenBlockedByActiveItem)) {
    ImVec2 mouse = ImGui::GetMousePos();
    float rel_x = mouse.x - origin.x - Grid::kBoardPadding;
    float rel_y = mouse.y - origin.y - Grid::kBoardPadding;
    // Round to nearest intersection.
    int col = static_cast<int>(rel_x / Grid::kCellSize + 0.5f);
    int row = static_cast<int>(rel_y / Grid::kCellSize + 0.5f);
    if (IsValidCell(row, col)) {
      if (board.GetCell(ToAction(row, col)) == Cell::kEmpty) {
        hover = ToAction(row, col);
        ImVec2 center = IntersectionPos(origin, row, col);
        dl->AddCircleFilled(center, Grid::kStoneRadius,
                            IM_COL32(80, 80, 80, 90), 40);
      }
    }
  }

  // Stones
  for (int r = 0; r < kBoardSize; ++r) {
    for (int c = 0; c < kBoardSize; ++c) {
      Cell cell = board.GetCell(ToAction(r, c));
      if (cell != Cell::kEmpty) {
        ImVec2 center = IntersectionPos(origin, r, c);
        DrawStone(dl, center, cell);
      }
    }
  }

  // Last-move indicator
  if (last_move) {
    ImVec2 center =
        IntersectionPos(origin, ActionRow(*last_move), ActionCol(*last_move));
    dl->AddCircleFilled(center, 4.5f, kLastMoveDot, 12);
  }

  // Reserve layout space
  ImGui::Dummy({sz, sz});

  // Click detection
  if (interactive && ImGui::IsItemClicked(ImGuiMouseButton_Left)) {
    return hover;
  }
  return std::nullopt;
}

}  // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

std::optional<Action> Grid::DrawBoard(const Board& board,
                                      std::optional<Action> last_move) {
  return DrawBoardImpl(board, /*interactive=*/true, last_move);
}

void Grid::DrawBoardReadOnly(const Board& board,
                             std::optional<Action> last_move) {
  DrawBoardImpl(board, /*interactive=*/false, last_move);
}
