#pragma once

// Helpers shared by the test executables.

#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "game_position.h"

#define CHECK(condition)                                                   \
  do {                                                                     \
    if (!(condition))                                                      \
      throw std::runtime_error(std::string(__func__) + ": " + #condition); \
  } while (false)

namespace test {

// Fails unless `function` throws `Exception`.
template <typename Exception, typename Function>
void CheckThrows(Function function) {
  try {
    function();
  } catch (const Exception&) {
    return;
  }
  throw std::runtime_error("Expected exception was not thrown");
}

// A full-board game with no five on any axis: horizontal colors alternate,
// while vertical/diagonal runs are at most two stones long.
inline std::vector<Action> DrawMoves() {
  std::vector<Action> black, white, moves;
  for (int row = 0; row < kBoardSize; ++row)
    for (int col = 0; col < kBoardSize; ++col)
      ((row + 2 * col) % 4 < 2 ? black : white).push_back(ToAction(row, col));
  CHECK(black.size() == white.size() + 1);

  for (size_t i = 0; i < white.size(); ++i) {
    moves.push_back(black[i]);
    moves.push_back(white[i]);
  }
  moves.push_back(black.back());
  return moves;
}

// Unique directory under the system temp dir, removed on destruction.
struct TempDirectory {
  using Clock = std::chrono::steady_clock;
  std::filesystem::path path;

  explicit TempDirectory(const std::string& prefix) {
    const auto stamp = Clock::now().time_since_epoch().count();
    path = std::filesystem::temp_directory_path() /
           (prefix + "-" + std::to_string(stamp));
    std::filesystem::create_directory(path);
  }

  ~TempDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path, error);
  }
};

}  // namespace test
