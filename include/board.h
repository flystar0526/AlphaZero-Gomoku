#pragma once

#include <array>
#include <cstdint>
#include <vector>

// 1D flat board position index in [0, kActionCount).
using Action = int;

inline constexpr int kBoardSize = 15;
inline constexpr int kActionCount = kBoardSize * kBoardSize;

enum class Player : std::uint8_t { kBlack, kWhite };
enum class Cell : std::uint8_t { kEmpty = 0, kBlack, kWhite };

// Returns the opposing player.
constexpr Player Opponent(Player player) {
  return player == Player::kBlack ? Player::kWhite : Player::kBlack;
}

// Converts a player to their corresponding stone type.
constexpr Cell ToCell(Player player) {
  return player == Player::kBlack ? Cell::kBlack : Cell::kWhite;
}

// Checks whether (row, col) is within the board boundaries.
constexpr bool IsValidCell(int row, int col) {
  return row >= 0 && row < kBoardSize && col >= 0 && col < kBoardSize;
}

// Checks whether action is within [0, kActionCount).
constexpr bool IsValidAction(Action action) {
  return action >= 0 && action < kActionCount;
}

// Converts between 2D (row, col) coordinates and 1D Action index.
constexpr Action ToAction(int row, int col) { return row * kBoardSize + col; }
constexpr int ActionRow(Action action) { return action / kBoardSize; }
constexpr int ActionCol(Action action) { return action % kBoardSize; }

// Manages the 15x15 Gomoku board state and win/draw detection.
class Board {
 public:
  Board() = default;

  // Clears the board to its initial empty state.
  void Reset();

  // Places a stone for `player`. Returns false if out of bounds or occupied.
  bool PlaceStone(Action action, Player player);

  // Returns the cell state, or Cell::kEmpty if out of bounds.
  Cell GetCell(Action action) const;

  // Checks if the action is within bounds and the cell is empty.
  bool IsLegalMove(Action action) const;

  // Returns all currently empty board positions.
  std::vector<Action> LegalActions() const;

  // Returns the total number of stones placed on the board.
  int stone_count() const { return stone_count_; }

  // Checks if `player` has five or more stones in a row through `action`.
  bool CheckWin(Action action, Player player) const;

  // Checks if the board has no empty cells left (draw condition).
  bool IsFull() const;

 private:
  // Counts consecutive stones of `player` from (row, col) along (dr, dc).
  int CountInDirection(int row, int col, Player player, int dr, int dc) const;

  std::array<Cell, kActionCount> cells_{};
  int stone_count_ = 0;
};
