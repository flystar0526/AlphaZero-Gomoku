#include "simulation.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <locale>
#include <optional>
#include <sstream>
#include <stdexcept>

#include "ai_player.h"
#include "game.h"

namespace {

using Clock = std::chrono::steady_clock;

// One move's durations in ms, ordered as kTimingNames: the whole PickMove()
// call first, then the four MCTS phases.
using Durations = std::array<double, 5>;
constexpr std::array<const char*, 5> kTimingNames = {
    "think", "selection", "expansion", "evaluation", "backup"};

constexpr std::array<Player, 2> kColors = {Player::kBlack, Player::kWhite};

// AI index (0 = Player 1, 1 = Player 2) seated at each color, indexed by color.
using Seats = std::array<size_t, 2>;

size_t Index(Player color) { return static_cast<size_t>(color); }

const char* ColorName(Player color) {
  return color == Player::kBlack ? "black" : "white";
}

double Milliseconds(Clock::duration elapsed) {
  return std::chrono::duration<double, std::milli>(elapsed).count();
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

// Move count plus per-metric sum and maximum.
struct TimingStatistics {
  std::int64_t moves = 0;
  Durations total{};
  Durations maximum{};

  void Add(const Durations& durations) { Merge({1, durations, durations}); }

  void Merge(const TimingStatistics& other) {
    moves += other.moves;
    for (size_t i = 0; i < total.size(); ++i) {
      total[i] += other.total[i];
      maximum[i] = std::max(maximum[i], other.maximum[i]);
    }
  }
};

// Results of one AI in one color (or merged over several cells).
struct PlayerStatistics {
  int games = 0;
  int wins = 0;
  int losses = 0;
  int draws = 0;
  TimingStatistics timing;

  void Merge(const PlayerStatistics& other) {
    games += other.games;
    wins += other.wins;
    losses += other.losses;
    draws += other.draws;
    timing.Merge(other.timing);
  }
};

// Indexed by AI, then color. Every summary figure is derived from these four
// cells, so they are the single source of truth.
using SimulationStatistics = std::array<std::array<PlayerStatistics, 2>, 2>;

// Both AIs merged for one color. Each game has exactly one black and one white
// seat, so `games` equals the completed game count.
PlayerStatistics ColorTotal(const SimulationStatistics& stats, Player color) {
  PlayerStatistics total = stats[0][Index(color)];
  total.Merge(stats[1][Index(color)]);
  return total;
}

// Both colors merged for one AI.
PlayerStatistics PlayerTotal(const SimulationStatistics& stats, size_t player) {
  PlayerStatistics total = stats[player][Index(Player::kBlack)];
  total.Merge(stats[player][Index(Player::kWhite)]);
  return total;
}

// Outcome of one finished game; `timing` is indexed by AI.
struct GameResult {
  std::optional<Player> winner;  // std::nullopt for a draw.
  int total_moves = 0;
  double duration_ms = 0.0;
  std::array<TimingStatistics, 2> timing;
};

// Credits a finished game to the cell of each AI in the color it played.
void RecordGame(SimulationStatistics& stats, const Seats& seats,
                const GameResult& game) {
  for (Player color : kColors) {
    const size_t player = seats[Index(color)];
    PlayerStatistics& cell = stats[player][Index(color)];

    ++cell.games;
    if (!game.winner) {
      ++cell.draws;
    } else if (*game.winner == color) {
      ++cell.wins;
    } else {
      ++cell.losses;
    }
    cell.timing.Merge(game.timing[player]);
  }
}

// ---------------------------------------------------------------------------
// Log files
// ---------------------------------------------------------------------------

// Creates logs/simulation_<timestamp>_<us>[_N]; never reuses an existing one.
std::filesystem::path CreateLogDirectory(const std::filesystem::path& root) {
  std::filesystem::create_directories(root);

  const auto now = std::chrono::system_clock::now();
  const std::time_t timestamp = std::chrono::system_clock::to_time_t(now);
  std::tm local_time{};
  localtime_r(&timestamp, &local_time);
  const auto microseconds =
      std::chrono::duration_cast<std::chrono::microseconds>(
          now.time_since_epoch())
          .count() %
      1000000;

  std::ostringstream name;
  name << "simulation_" << std::put_time(&local_time, "%Y%m%d_%H%M%S") << '_'
       << std::setfill('0') << std::setw(6) << microseconds;

  // Exclusive directory creation also protects simultaneous runs from overlap.
  for (int suffix = 0; suffix < 1000; ++suffix) {
    const auto directory =
        root / (name.str() + (suffix == 0 ? "" : "_" + std::to_string(suffix)));
    if (std::filesystem::create_directory(directory)) return directory;
  }
  throw std::runtime_error(
      "Could not create a unique simulation log directory");
}

// Opens a log that throws on write errors and formats numbers locale-free.
std::ofstream OpenLog(const std::filesystem::path& path) {
  std::ofstream out;
  out.exceptions(std::ios::failbit | std::ios::badbit);
  out.imbue(std::locale::classic());
  out.open(path);
  out << std::fixed << std::setprecision(6);
  return out;
}

void WriteWinRate(std::ostream& out, int wins, int games) {
  if (games == 0) {
    out << "N/A";
  } else {
    out << 100.0 * wins / games;
  }
}

// One [players] row: player_id,color,games,wins,losses,draws,win_rate_pct
void WritePlayerStatistics(std::ostream& out, size_t player, const char* color,
                           const PlayerStatistics& stats) {
  out << player + 1 << ',' << color << ',' << stats.games << ',' << stats.wins
      << ',' << stats.losses << ',' << stats.draws << ',';
  WriteWinRate(out, stats.wins, stats.games);
  out << '\n';
}

// One [timing] row per metric: player_id,color,moves,metric,total,mean,max
void WriteTimingStatistics(std::ostream& out, size_t player, const char* color,
                           const TimingStatistics& stats) {
  for (size_t i = 0; i < kTimingNames.size(); ++i) {
    out << player + 1 << ',' << color << ',' << stats.moves << ','
        << kTimingNames[i] << ',' << stats.total[i] << ',';
    if (stats.moves == 0) {
      out << "N/A,N/A\n";
    } else {
      out << stats.total[i] / stats.moves << ',' << stats.maximum[i] << '\n';
    }
  }
}

// Rewrites summary.txt from scratch, so it is valid after every game.
void WriteSummary(const std::filesystem::path& directory,
                  const SimulationOptions& options, const Configuration& config,
                  const SimulationStatistics& stats, const char* status,
                  const std::string& error = "") {
  auto out = OpenLog(directory / "summary.txt");
  const PlayerStatistics black = ColorTotal(stats, Player::kBlack);
  const int completed_games = black.games;

  // Run metadata.
  out << "status = " << status << "\nrequested_games = " << options.num_games
      << "\ncompleted_games = " << completed_games
      << "\nmodel_path = " << options.model_path
      << "\ntime_unit = milliseconds\ntiming_scope = completed_games\n"
      << "win_rate = wins / games (draws included)\n";
  if (!error.empty()) out << "error = " << error << '\n';

  // Search settings of each AI.
  out << "\n[configuration]\n";
  for (AIPlayerId player : {AIPlayerId::kPlayer1, AIPlayerId::kPlayer2}) {
    WriteAIOptions(out, player, config.ai_options(player));
  }

  // Win rate of each color regardless of which AI played it.
  out << "\n[overall]\ncolor,games,wins,win_rate_pct\n";
  for (Player color : kColors) {
    const int wins = ColorTotal(stats, color).wins;
    out << ColorName(color) << ',' << completed_games << ',' << wins << ',';
    WriteWinRate(out, wins, completed_games);
    out << '\n';
  }
  // Every game has one black seat, so black-cell draws count each draw once.
  out << "draws = " << black.draws << '\n';

  // Each AI's results by color, then in total.
  out << "\n[players]\nplayer_id,color,games,wins,losses,draws,win_rate_pct\n";
  for (size_t player = 0; player < stats.size(); ++player) {
    for (Player color : kColors) {
      WritePlayerStatistics(out, player, ColorName(color),
                            stats[player][Index(color)]);
    }
    WritePlayerStatistics(out, player, "all", PlayerTotal(stats, player));
  }

  // Each AI's thinking time by color, then in total.
  out << "\n[timing]\nplayer_id,color,moves,metric,total_ms,mean_ms,max_ms\n";
  for (size_t player = 0; player < stats.size(); ++player) {
    for (Player color : kColors) {
      WriteTimingStatistics(out, player, ColorName(color),
                            stats[player][Index(color)].timing);
    }
    WriteTimingStatistics(out, player, "all",
                          PlayerTotal(stats, player).timing);
  }

  out.close();
}

// One games.csv row; a draw leaves the winner id empty.
void WriteGameRow(std::ostream& out, int game_number, const Seats& seats,
                  const GameResult& game) {
  out << game_number << ',' << seats[Index(Player::kBlack)] + 1 << ','
      << seats[Index(Player::kWhite)] + 1 << ',';
  if (game.winner) {
    out << seats[Index(*game.winner)] + 1 << ',' << ColorName(*game.winner);
  } else {
    out << ",draw";
  }
  out << ',' << game.total_moves << ',' << game.duration_ms << '\n';
}

void PrintProgress(std::ostream& out, int game_number, int num_games,
                   const Seats& seats, const GameResult& game) {
  out << "Game " << game_number << '/' << num_games << ": black=Player "
      << seats[Index(Player::kBlack)] + 1 << ", white=Player "
      << seats[Index(Player::kWhite)] + 1 << ", winner=";
  if (game.winner) {
    out << "Player " << seats[Index(*game.winner)] + 1 << " ("
        << ColorName(*game.winner) << ')';
  } else {
    out << "draw";
  }
  out << ", moves=" << game.total_moves << std::endl;
}

// ---------------------------------------------------------------------------
// Game loop
// ---------------------------------------------------------------------------

// Plays one game to the end, appending one moves.csv row per move.
GameResult PlayGame(int game_number, const Seats& seats, Game& game,
                    std::array<AIPlayer, 2>& players, std::ostream& moves) {
  game.Start();
  for (auto& player : players) player.Reset();

  GameResult result;
  const auto game_start = Clock::now();

  while (game.state() == Game::State::kPlaying) {
    const Player color = game.current_player();
    const size_t player = seats[Index(color)];
    const int move_number = game.next_move_number();

    // think_ms wraps the whole PickMove(); phase times come from MCTS.
    SearchTiming timing;
    const auto start = Clock::now();
    const auto action = players[player].PickMove(game.position(), &timing);
    const Durations durations = {Milliseconds(Clock::now() - start),
                                 timing.selection_ms, timing.expansion_ms,
                                 timing.evaluation_ms, timing.backup_ms};

    if (!action || !game.TryPlaceStone(*action)) {
      throw std::runtime_error("AI returned no legal move");
    }
    // Both trees must follow the board, as in the GUI's AI vs AI mode.
    for (auto& ai : players) ai.ObserveMove(*action);
    result.timing[player].Add(durations);

    moves << game_number << ',' << move_number << ',' << player + 1 << ','
          << ColorName(color) << ',' << ActionRow(*action) << ','
          << ActionCol(*action);
    for (double duration : durations) moves << ',' << duration;
    moves << '\n';
  }

  result.duration_ms = Milliseconds(Clock::now() - game_start);
  result.winner = game.winner();
  result.total_moves = game.record().size();
  return result;
}

}  // namespace

std::filesystem::path RunSimulation(
    const SimulationOptions& options, const Configuration& config,
    std::shared_ptr<PolicyValueEvaluator> evaluator, std::ostream* progress) {
  if (options.num_games <= 0 || !evaluator) {
    throw std::invalid_argument(
        "Simulation requires a positive game count and a non-null evaluator");
  }

  const auto directory = CreateLogDirectory(options.log_root);
  SimulationStatistics stats{};

  try {
    if (progress) {
      *progress << "Simulation logs: " << directory.string() << std::endl;
    }
    WriteSummary(directory, options, config, stats, "running");

    auto moves = OpenLog(directory / "moves.csv");
    auto games = OpenLog(directory / "games.csv");
    moves << "game,move,player_id,color,row,col,think_ms,selection_ms,"
             "expansion_ms,evaluation_ms,backup_ms\n";
    games << "game,black_player_id,white_player_id,winner_player_id,"
             "winner_color,total_moves,duration_ms\n";
    moves.flush();
    games.flush();

    // Both AIs share one evaluator, so the model is loaded only once.
    std::array<AIPlayer, 2> players = {
        AIPlayer(evaluator, config.ai_options(AIPlayerId::kPlayer1)),
        AIPlayer(evaluator, config.ai_options(AIPlayerId::kPlayer2))};
    Game game(config, Game::Mode::kAIvsAI);

    for (int index = 0; index < options.num_games; ++index) {
      const int game_number = index + 1;

      // Player 1 is black in odd games; settings follow the AI, not the color.
      const size_t black = static_cast<size_t>(index % 2);
      const Seats seats = {black, 1 - black};

      const GameResult result =
          PlayGame(game_number, seats, game, players, moves);
      WriteGameRow(games, game_number, seats, result);
      moves.flush();
      games.flush();

      // Only completed games reach the summary.
      RecordGame(stats, seats, result);
      WriteSummary(directory, options, config, stats, "running");
      if (progress) {
        PrintProgress(*progress, game_number, options.num_games, seats,
                      result);
      }
    }

    moves.close();
    games.close();
    WriteSummary(directory, options, config, stats, "complete");
  } catch (const std::exception& error) {
    // Keep partial logs and mark the summary failed if it is still writable.
    const std::string message =
        "Simulation failed; logs: " + directory.string() + ": " + error.what();
    try {
      WriteSummary(directory, options, config, stats, "failed", error.what());
    } catch (const std::exception& summary_error) {
      throw std::runtime_error(
          message + "; could not update summary: " + summary_error.what());
    }
    throw std::runtime_error(message);
  }

  return directory;
}
