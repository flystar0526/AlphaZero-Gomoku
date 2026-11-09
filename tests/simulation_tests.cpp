#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ai_player.h"
#include "simulation.h"
#include "test_support.h"

namespace {

using test::CheckThrows;
using test::DrawMoves;

constexpr const char* kTempPrefix = "gomoku-simulation-test";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

// Nine-move game (ten for White) where `winner` gets five in row 7.
std::vector<Action> WinningMoves(Player winner) {
  std::vector<Action> moves;
  if (winner == Player::kWhite) moves.push_back(ToAction(14, 14));
  for (int col = 0; col < 4; ++col) {
    moves.push_back(ToAction(7, col));
    moves.push_back(ToAction(0, col));
  }
  moves.push_back(ToAction(7, 4));
  return moves;
}

// Forces each game to follow a script (cycling through `scripts` per game) by
// putting all policy mass on the scripted move. Records which move numbers
// were evaluated and can throw at (fail_game, fail_move) to simulate failure.
class ScriptedEvaluator : public PolicyValueEvaluator {
 public:
  std::vector<std::vector<Action>> scripts = {WinningMoves(Player::kBlack)};
  std::vector<std::vector<int>> evaluated_moves;
  int fail_game = -1;
  int fail_move = -1;

  Evaluation Evaluate(const GamePosition& position) override {
    CHECK(!position.is_terminal());
    const int move = position.board().stone_count();

    // An empty board marks the start of a new game.
    if (move == 0) evaluated_moves.emplace_back();
    CHECK(!evaluated_moves.empty());
    evaluated_moves.back().push_back(move);
    if (static_cast<int>(evaluated_moves.size()) == fail_game &&
        move == fail_move) {
      throw std::runtime_error("inference failed");
    }

    const auto& script = scripts[(evaluated_moves.size() - 1) % scripts.size()];
    CHECK(move < static_cast<int>(script.size()));
    CHECK(position.IsLegalMove(script[move]));
    Evaluation result;
    result.policy[script[move]] = 1.0f;
    return result;
  }
};

// ---------------------------------------------------------------------------
// Log parsing
// ---------------------------------------------------------------------------

using Table = std::vector<std::vector<std::string>>;

std::string ReadFile(const std::filesystem::path& path) {
  std::ifstream in(path);
  CHECK(in.is_open());
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

// Parses CSV rows after the heading line, from the file start or from a
// summary.txt `[section]`, stopping at a blank line or the next section.
Table ReadTable(const std::string& text, const std::string& section = "") {
  size_t start = 0;
  if (!section.empty()) {
    const std::string marker = '[' + section + "]\n";
    start = text.find(marker);
    CHECK(start != std::string::npos);
    start += marker.size();
  }
  std::istringstream input(text.substr(start));
  std::string line;
  std::getline(input, line);  // Skip the column headings.
  Table rows;
  while (std::getline(input, line) && !line.empty() && line.front() != '[') {
    std::istringstream row(line);
    std::vector<std::string> cells;
    std::string cell;
    while (std::getline(row, cell, ',')) cells.push_back(cell);
    rows.push_back(cells);
  }
  return rows;
}

// Distinct settings per AI so tests can see they follow the AI, not the color.
Configuration TestConfiguration(int player_2_budget = 1) {
  Configuration config;
  config.set_ai_options(AIPlayerId::kPlayer1, {5.0f, 1, 0.0f, 0});
  config.set_ai_options(AIPlayerId::kPlayer2, {2.0f, player_2_budget, 0.0f, 0});
  config.set_time_limit_enabled(true);
  config.set_time_limit_seconds(0);  // AI turns must remain exempt.
  return config;
}

void CheckZeroTiming(const SearchTiming& timing) {
  CHECK(timing.selection_ms == 0.0);
  CHECK(timing.expansion_ms == 0.0);
  CHECK(timing.evaluation_ms == 0.0);
  CHECK(timing.backup_ms == 0.0);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// MCTS phase timing: reset on entry, zero when no work is done, and no effect
// on move choice.
void SearchPhaseTiming() {
  auto evaluator = std::make_shared<ScriptedEvaluator>();
  MCTS search({5.0f, 1, 0.0f, 0});
  SearchTiming timing{1e6, 1e6, 1e6, 1e6};
  search.Search(GamePosition(), *evaluator, &timing);

  // Stale values must be cleared by Search().
  for (double value : {timing.selection_ms, timing.expansion_ms,
                       timing.evaluation_ms, timing.backup_ms}) {
    CHECK(std::isfinite(value) && value >= 0.0 && value < 1e6);
  }

  // Terminal positions do no search work at all.
  GamePosition terminal;
  const auto actions = WinningMoves(Player::kBlack);
  for (Action action : actions) terminal.PlayMove(action);
  search.Search(terminal, *evaluator, &timing);
  CheckZeroTiming(timing);

  // Reusing a terminal child performs selection/backup without inference.
  GamePosition before_win;
  for (size_t i = 0; i + 1 < actions.size(); ++i)
    before_win.PlayMove(actions[i]);

  class WinningEvaluator : public PolicyValueEvaluator {
   public:
    int calls = 0;
    Evaluation Evaluate(const GamePosition&) override {
      ++calls;
      Evaluation output;
      output.policy[ToAction(7, 4)] = 1.0f;
      return output;
    }
  } winning;

  search.Reset();
  search.Search(before_win, winning, &timing);
  timing = {1e6, 1e6, 1e6, 1e6};
  search.Search(before_win, winning, &timing);
  CHECK(winning.calls == 1);
  CHECK(timing.evaluation_ms == 0.0);
  CHECK(timing.expansion_ms == 0.0);

  // Instrumentation must preserve sampling decisions and per-search budgets.
  class UniformEvaluator : public PolicyValueEvaluator {
   public:
    int calls = 0;
    Evaluation Evaluate(const GamePosition&) override {
      ++calls;
      Evaluation output;
      output.policy.fill(1.0f / kActionCount);
      return output;
    }
  };
  auto uniform = std::make_shared<UniformEvaluator>();
  AIPlayer timed(uniform, {5.0f, 8, 1.0f, 30}, 123);
  AIPlayer untimed(uniform, {5.0f, 8, 1.0f, 30}, 123);
  GamePosition position;

  for (int i = 0; i < 6; ++i) {
    timing = {1e6, 1e6, 1e6, 1e6};
    const int previous_calls = uniform->calls;
    const auto action = timed.PickMove(position, &timing);
    CHECK(uniform->calls - previous_calls == 8);
    CHECK(action == untimed.PickMove(position));
    CHECK(timing.evaluation_ms < 1e6);
    position.PlayMove(*action);
    timed.ObserveMove(*action);
    untimed.ObserveMove(*action);
  }

  timed.PickMove(terminal, &timing);
  CheckZeroTiming(timing);
}

// Three games: colors alternate, settings stay with each AI, and the summary
// agrees with values recomputed from moves.csv.
void AlternatingColorsAndLogs() {
  test::TempDirectory temp(kTempPrefix);
  SimulationOptions options;
  options.num_games = 3;
  options.log_root = temp.path;
  auto evaluator = std::make_shared<ScriptedEvaluator>();
  const auto config = TestConfiguration(2);
  std::ostringstream progress;
  const auto directory = RunSimulation(options, config, evaluator, &progress);
  CHECK(directory.parent_path() == temp.path);

  // games.csv: Player 1 is black in odd games and always wins as black.
  const auto games = ReadTable(ReadFile(directory / "games.csv"));
  CHECK(games.size() == 3);
  CHECK(games[0][1] == "1" && games[0][2] == "2" && games[0][3] == "1");
  CHECK(games[1][1] == "2" && games[1][2] == "1" && games[1][3] == "2");
  CHECK(games[2][1] == "1" && games[2][2] == "2" && games[2][3] == "1");

  // Player 2 has two playouts, so it evaluates a second position per move.
  CHECK(evaluator->evaluated_moves.size() == 3);
  CHECK(evaluator->evaluated_moves[0][1] == 1);
  CHECK(evaluator->evaluated_moves[0][2] == 2);
  CHECK(evaluator->evaluated_moves[1][1] == 1);
  CHECK(evaluator->evaluated_moves[1][2] == 1);  // Player 2 kept its budget.

  // moves.csv: replay every move and accumulate expected timing statistics,
  // indexed by [AI][color][metric].
  const auto moves = ReadTable(ReadFile(directory / "moves.csv"));
  CHECK(moves.size() == 27);

  std::array<std::array<int, 2>, 2> counts{};
  std::array<std::array<std::array<double, 5>, 2>, 2> totals{};
  std::array<std::array<std::array<double, 5>, 2>, 2> maxima{};
  for (size_t game = 0; game < games.size(); ++game) {
    GamePosition position;
    CHECK(games[game][4] == "black" && games[game][5] == "9");
    CHECK(std::stod(games[game][6]) >= 0.0);

    for (size_t move = 0; move < 9; ++move) {
      const auto& row = moves[game * 9 + move];
      CHECK(row.size() == 11);
      CHECK(std::stoul(row[0]) == game + 1);
      CHECK(std::stoul(row[1]) == move + 1);
      const size_t color = move % 2;
      const size_t player = color == 0 ? game % 2 : 1 - game % 2;
      CHECK(std::stoul(row[2]) == player + 1);
      CHECK(row[3] == (color == 0 ? "black" : "white"));
      const Action action = ToAction(std::stoi(row[4]), std::stoi(row[5]));
      CHECK(position.IsLegalMove(action));
      position.PlayMove(action);
      ++counts[player][color];

      double phase_sum = 0.0;
      for (size_t i = 0; i < 5; ++i) {
        const double duration = std::stod(row[6 + i]);
        CHECK(std::isfinite(duration) && duration >= 0.0);
        totals[player][color][i] += duration;
        maxima[player][color][i] = std::max(maxima[player][color][i], duration);
        if (i > 0) phase_sum += duration;
      }
      // Phases never overlap, so they cannot exceed total thinking time.
      CHECK(std::stod(row[6]) + 1e-5 >= phase_sum);
    }
    CHECK(position.winner() == Player::kBlack);
  }

  // summary.txt: metadata, win rates, and timing rows (color 2 = "all").
  const auto summary = ReadFile(directory / "summary.txt");
  CHECK(summary.find("status = complete\n") != std::string::npos);
  CHECK(summary.find("completed_games = 3\n") != std::string::npos);
  CHECK(summary.find("ai_player_2_num_playouts = 2\n") != std::string::npos);

  const auto results = ReadTable(summary, "players");
  CHECK(results.size() == 6);
  CHECK(results[0] == std::vector<std::string>(
                          {"1", "black", "2", "2", "0", "0", "100.000000"}));
  CHECK(results[1] == std::vector<std::string>(
                          {"1", "white", "1", "0", "1", "0", "0.000000"}));
  CHECK(results[2][2] == "3" && results[2][3] == "2");
  CHECK(results[2][6] == "66.666667");

  const auto timing = ReadTable(summary, "timing");
  CHECK(timing.size() == 30);
  for (size_t player = 0; player < 2; ++player) {
    for (size_t color = 0; color < 3; ++color) {
      for (size_t metric = 0; metric < 5; ++metric) {
        const auto& row = timing[(player * 3 + color) * 5 + metric];
        const int count = color == 2 ? counts[player][0] + counts[player][1]
                                     : counts[player][color];
        const double total =
            color == 2 ? totals[player][0][metric] + totals[player][1][metric]
                       : totals[player][color][metric];
        const double maximum = color == 2 ? std::max(maxima[player][0][metric],
                                                     maxima[player][1][metric])
                                          : maxima[player][color][metric];
        CHECK(std::stoi(row[2]) == count);
        CHECK(std::abs(std::stod(row[4]) - total) < 2e-5);
        CHECK(std::abs(std::stod(row[5]) - total / count) < 2e-5);
        CHECK(std::abs(std::stod(row[6]) - maximum) < 2e-5);
      }
    }
  }
  CHECK(progress.str().find("Game 3/3:") != std::string::npos);

  // A second run gets its own directory; empty cells report N/A.
  options.num_games = 1;
  evaluator = std::make_shared<ScriptedEvaluator>();
  const auto second = RunSimulation(options, config, evaluator);
  CHECK(second != directory && std::filesystem::exists(directory));
  const auto single = ReadFile(second / "summary.txt");
  CHECK(single.find("1,white,0,0,0,0,N/A\n") != std::string::npos);
  CHECK(single.find("2,black,0,think,0.000000,N/A,N/A\n") != std::string::npos);
}

// White wins and draws are counted per color and per AI.
void WhiteWinsAndDraws() {
  test::TempDirectory temp(kTempPrefix);
  SimulationOptions options;
  options.num_games = 2;
  options.log_root = temp.path;
  auto evaluator = std::make_shared<ScriptedEvaluator>();
  evaluator->scripts = {WinningMoves(Player::kWhite), DrawMoves()};
  const auto directory = RunSimulation(options, TestConfiguration(), evaluator);

  const auto games = ReadTable(ReadFile(directory / "games.csv"));
  CHECK(games[0][3] == "2" && games[0][4] == "white");
  CHECK(games[1][3].empty() && games[1][4] == "draw");
  CHECK(games[1][5] == "225");

  const auto summary = ReadFile(directory / "summary.txt");
  CHECK(summary.find("white,2,1,50.000000\n") != std::string::npos);
  CHECK(summary.find("draws = 1\n") != std::string::npos);
  CHECK(summary.find("2,all,2,1,0,1,50.000000\n") != std::string::npos);
  CHECK(summary.find("1,all,2,0,1,1,0.000000\n") != std::string::npos);
}

// A mid-run failure keeps partial logs and marks the summary failed; invalid
// input is rejected up front.
void FailedSimulation() {
  test::TempDirectory temp(kTempPrefix);
  SimulationOptions options;
  options.num_games = 3;
  options.log_root = temp.path;
  auto evaluator = std::make_shared<ScriptedEvaluator>();
  evaluator->fail_game = 2;
  evaluator->fail_move = 2;

  // Game 2 fails on its second move, after game 1 completed.
  bool failed = false;
  try {
    RunSimulation(options, TestConfiguration(), evaluator);
  } catch (const std::runtime_error& error) {
    CHECK(std::string(error.what()).find("inference failed") !=
          std::string::npos);
    failed = true;
  }
  CHECK(failed);

  const auto directory = std::filesystem::directory_iterator(temp.path)->path();
  const auto summary = ReadFile(directory / "summary.txt");
  CHECK(summary.find("status = failed\n") != std::string::npos);
  CHECK(summary.find("completed_games = 1\n") != std::string::npos);
  CHECK(summary.find("error = inference failed\n") != std::string::npos);

  // The unfinished game leaves move rows but no game row or statistics.
  CHECK(ReadTable(ReadFile(directory / "games.csv")).size() == 1);
  CHECK(ReadTable(ReadFile(directory / "moves.csv")).size() == 11);
  CHECK(summary.find("1,black,5,think,") != std::string::npos);
  CHECK(summary.find("1,white,0,0,0,0,N/A\n") != std::string::npos);

  // Invalid input: non-positive count, missing evaluator, unusable log root.
  for (int count : {0, -1}) {
    options.num_games = count;
    CheckThrows<std::invalid_argument>(
        [&] { RunSimulation(options, TestConfiguration(), evaluator); });
  }

  options.num_games = 1;
  CheckThrows<std::invalid_argument>(
      [&] { RunSimulation(options, TestConfiguration(), nullptr); });

  options.log_root = temp.path / "regular-file";
  std::ofstream(options.log_root) << "not a directory";
  CheckThrows<std::filesystem::filesystem_error>(
      [&] { RunSimulation(options, TestConfiguration(), evaluator); });
}

}  // namespace

int main() {
  const std::pair<const char*, void (*)()> tests[] = {
      {"search phase timing", SearchPhaseTiming},
      {"alternating colors and logs", AlternatingColorsAndLogs},
      {"white wins and draws", WhiteWinsAndDraws},
      {"failed simulation", FailedSimulation},
  };
  int failures = 0;
  for (const auto& [name, test] : tests) {
    try {
      test();
      std::cout << "PASS " << name << '\n';
    } catch (const std::exception& error) {
      ++failures;
      std::cerr << "FAIL " << name << ": " << error.what() << '\n';
    }
  }
  return failures == 0 ? 0 : 1;
}
#include <algorithm>
