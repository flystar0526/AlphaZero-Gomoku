#include "board.h"

void Board::Reset() {
  cells_.fill(Cell::kEmpty);
  stone_count_ = 0;
}

bool Board::PlaceStone(Action action, Player player) {
  if (!IsLegalMove(action)) return false;
  cells_[action] = ToCell(player);
  ++stone_count_;
  return true;
}

Cell Board::GetCell(Action action) const {
  if (!IsValidAction(action)) return Cell::kEmpty;
  return cells_[action];
}

bool Board::IsLegalMove(Action action) const {
  return IsValidAction(action) && cells_[action] == Cell::kEmpty;
}

std::vector<Action> Board::LegalActions() const {
  std::vector<Action> actions;
  actions.reserve(kActionCount - stone_count_);
  for (Action action = 0; action < kActionCount; ++action) {
    if (cells_[action] == Cell::kEmpty) actions.push_back(action);
  }
  return actions;
}

bool Board::CheckWin(Action action, Player player) const {
  if (!IsValidAction(action) || cells_[action] != ToCell(player)) {
    return false;
  }

  const int row = ActionRow(action);
  const int col = ActionCol(action);

  // Four axes: horizontal, vertical, main diagonal, anti-diagonal.
  static constexpr int kDeltaRow[] = {0, 1, 1, 1};
  static constexpr int kDeltaCol[] = {1, 0, 1, -1};

  for (int i = 0; i < 4; ++i) {
    const int count =
        1 + CountInDirection(row, col, player, kDeltaRow[i], kDeltaCol[i]) +
        CountInDirection(row, col, player, -kDeltaRow[i], -kDeltaCol[i]);
    if (count >= 5) return true;
  }
  return false;
}

bool Board::IsFull() const {
  return stone_count_ == kActionCount;
}

int Board::CountInDirection(int row, int col, Player player, int dr,
                            int dc) const {
  const Cell target = ToCell(player);
  int count = 0;
  int r = row + dr;
  int c = col + dc;
  while (IsValidCell(r, c) && cells_[ToAction(r, c)] == target) {
    ++count;
    r += dr;
    c += dc;
  }
  return count;
}
