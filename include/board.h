#pragma once

#include <array>
#include <cstdint>
#include <vector>

// Flat board position index in [0, kActionCount).
using Action = int;

inline constexpr int kBoardSize = 15;
inline constexpr int kActionCount = kBoardSize * kBoardSize;

enum class Player : std::uint8_t { kBlack, kWhite };
enum class Cell : std::uint8_t { kEmpty = 0, kBlack, kWhite };

constexpr Player Opponent(Player player) {
  return player == Player::kBlack ? Player::kWhite : Player::kBlack;
}

constexpr Cell ToCell(Player player) {
  return player == Player::kBlack ? Cell::kBlack : Cell::kWhite;
}

// Coordinate validation and conversion helpers.
constexpr bool IsValidCell(int row, int col) {
  return row >= 0 && row < kBoardSize && col >= 0 && col < kBoardSize;
}

constexpr bool IsValidAction(Action action) {
  return action >= 0 && action < kActionCount;
}

constexpr Action ToAction(int row, int col) { return row * kBoardSize + col; }
constexpr int ActionRow(Action action) { return action / kBoardSize; }
constexpr int ActionCol(Action action) { return action % kBoardSize; }

// Manages the 15x15 Gomoku board state and five-in-a-row detection.
class Board {
 public:
  Board() = default;

  void Reset();
  void PlaceStone(Action action, Player player);

  Cell GetCell(Action action) const;
  bool IsLegalMove(Action action) const;
  std::vector<Action> LegalActions() const;
  int stone_count() const { return stone_count_; }

  // Returns true if `player` has five or more in a row through `action`.
  bool CheckWin(Action action, Player player) const;
  bool IsFull() const;

 private:
  int CountInDirection(int row, int col, Player player, int dr, int dc) const;

  std::array<Cell, kActionCount> cells_{};
  int stone_count_ = 0;
};
