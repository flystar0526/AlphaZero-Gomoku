#include "grid.h"

#include "imgui.h"

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

namespace {

constexpr ImU32 kBoardBg = IM_COL32(205, 160, 80, 255);
constexpr ImU32 kLineColor = IM_COL32(60, 40, 10, 200);
constexpr ImU32 kStarColor = IM_COL32(60, 40, 10, 255);
constexpr ImU32 kLastMoveDot = IM_COL32(220, 50, 50, 230);

constexpr int kStarPoints[][2] = {
    {2, 2},  {2, 7},  {2, 12},
    {7, 2},  {7, 7},  {7, 12},
    {12, 2}, {12, 7}, {12, 12},
};

ImVec2 IntersectionPos(ImVec2 origin, int row, int col) {
  return {origin.x + Grid::kBoardPadding + col * Grid::kCellSize,
          origin.y + Grid::kBoardPadding + row * Grid::kCellSize};
}

void DrawStone(ImDrawList* dl, ImVec2 center, Cell cell) {
  if (cell == Cell::kBlack) {
    dl->AddCircleFilled(center, Grid::kStoneRadius,
                        IM_COL32(20, 20, 20, 255), 40);
    dl->AddCircle(center, Grid::kStoneRadius,
                  IM_COL32(90, 90, 90, 160), 40, 1.2f);
  } else {
    dl->AddCircleFilled(center, Grid::kStoneRadius,
                        IM_COL32(245, 243, 228, 255), 40);
    dl->AddCircle(center, Grid::kStoneRadius,
                  IM_COL32(30, 30, 30, 220), 40, 1.5f);
    dl->AddCircle(center, Grid::kStoneRadius * 0.78f,
                  IM_COL32(180, 170, 145, 80), 32, 1.0f);
  }
}

// Renders board background, grid lines, stones, and optional hover/click UI.
std::optional<Action> DrawBoardImpl(const Board& board, bool interactive,
                                    std::optional<Action> last_move) {
  ImDrawList* dl = ImGui::GetWindowDrawList();
  ImVec2 origin = ImGui::GetCursorScreenPos();
  const float sz = Grid::kCanvasSize;

  dl->AddRectFilled(origin, {origin.x + sz, origin.y + sz}, kBoardBg, 4.0f);

  for (int i = 0; i < kBoardSize; ++i) {
    ImVec2 a = IntersectionPos(origin, 0, i);
    ImVec2 b = IntersectionPos(origin, kBoardSize - 1, i);
    dl->AddLine(a, b, kLineColor, 1.0f);

    ImVec2 c = IntersectionPos(origin, i, 0);
    ImVec2 d = IntersectionPos(origin, i, kBoardSize - 1);
    dl->AddLine(c, d, kLineColor, 1.0f);
  }

  for (auto& sp : kStarPoints) {
    ImVec2 center = IntersectionPos(origin, sp[0], sp[1]);
    dl->AddCircleFilled(center, 4.0f, kStarColor, 12);
  }

  // Hover preview on empty cells.
  std::optional<Action> hover;
  if (interactive && ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows |
                                            ImGuiHoveredFlags_AllowWhenBlockedByActiveItem)) {
    ImVec2 mouse = ImGui::GetMousePos();
    float rel_x = mouse.x - origin.x - Grid::kBoardPadding;
    float rel_y = mouse.y - origin.y - Grid::kBoardPadding;
    int col = static_cast<int>(rel_x / Grid::kCellSize + 0.5f);
    int row = static_cast<int>(rel_y / Grid::kCellSize + 0.5f);

    if (IsValidCell(row, col) &&
        board.GetCell(ToAction(row, col)) == Cell::kEmpty) {
      hover = ToAction(row, col);
      ImVec2 center = IntersectionPos(origin, row, col);
      dl->AddCircleFilled(center, Grid::kStoneRadius,
                          IM_COL32(80, 80, 80, 90), 40);
    }
  }

  // Placed stones and last-move marker.
  for (int r = 0; r < kBoardSize; ++r) {
    for (int c = 0; c < kBoardSize; ++c) {
      Cell cell = board.GetCell(ToAction(r, c));
      if (cell != Cell::kEmpty) {
        DrawStone(dl, IntersectionPos(origin, r, c), cell);
      }
    }
  }

  if (last_move) {
    ImVec2 center =
        IntersectionPos(origin, ActionRow(*last_move), ActionCol(*last_move));
    dl->AddCircleFilled(center, 4.5f, kLastMoveDot, 12);
  }

  ImGui::Dummy({sz, sz});

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
