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

#include "match.h"

namespace {

using Clock = std::chrono::steady_clock;

// Per-move durations in ms: wall-clock think time followed by the MCTS phases
// (summed over workers) and the batch evaluator's inference time.
using Durations = std::array<double, 6>;
constexpr std::array<const char*, 6> kTimingNames = {
    "think", "selection", "expansion", "evaluation", "inference", "backup"};

// Per-move search counters.
struct Counters {
  std::int64_t playouts = 0;
  std::int64_t batches = 0;
  std::int64_t batched_positions = 0;
  std::int64_t collisions = 0;

  void Merge(const Counters& other) {
    playouts += other.playouts;
    batches += other.batches;
    batched_positions += other.batched_positions;
    collisions += other.collisions;
  }
};

constexpr std::array<Player, 2> kColors = {Player::kBlack, Player::kWhite};

size_t Index(Player color) { return static_cast<size_t>(color); }

size_t SeatIndex(const Seats& seats, Player color) {
  return static_cast<size_t>(*seats[Index(color)]);
}

const char* ColorName(Player color) {
  return color == Player::kBlack ? "black" : "white";
}

double Milliseconds(Clock::duration elapsed) {
  return std::chrono::duration<double, std::milli>(elapsed).count();
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

struct TimingStatistics {
  std::int64_t moves = 0;
  Durations total{};
  Durations maximum{};
  Counters counters;

  void Add(const Durations& durations, const Counters& move_counters) {
    Merge({1, durations, durations, move_counters});
  }

  void Merge(const TimingStatistics& other) {
    moves += other.moves;
    for (size_t i = 0; i < total.size(); ++i) {
      total[i] += other.total[i];
      maximum[i] = std::max(maximum[i], other.maximum[i]);
    }
    counters.Merge(other.counters);
  }
};

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

// Aggregated statistics indexed by [AI player][stone color].
using SimulationStatistics = std::array<std::array<PlayerStatistics, 2>, 2>;

PlayerStatistics ColorTotal(const SimulationStatistics& stats, Player color) {
  PlayerStatistics total = stats[0][Index(color)];
  total.Merge(stats[1][Index(color)]);
  return total;
}

PlayerStatistics PlayerTotal(const SimulationStatistics& stats, size_t player) {
  PlayerStatistics total = stats[player][Index(Player::kBlack)];
  total.Merge(stats[player][Index(Player::kWhite)]);
  return total;
}

struct GameResult {
  std::optional<Player> winner;
  int total_moves = 0;
  double duration_ms = 0.0;
  std::array<TimingStatistics, 2> timing;
};

void RecordGame(SimulationStatistics& stats, const Seats& seats,
                const GameResult& game) {
  for (Player color : kColors) {
    const size_t player = SeatIndex(seats, color);
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

// Creates a unique timestamped directory under `root` for simulation logs.
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

  for (int suffix = 0; suffix < 1000; ++suffix) {
    const auto directory =
        root / (name.str() + (suffix == 0 ? "" : "_" + std::to_string(suffix)));
    if (std::filesystem::create_directory(directory)) return directory;
  }

  throw std::runtime_error(
      "Could not create a unique simulation log directory");
}

std::ofstream OpenLog(const std::filesystem::path& path) {
  std::ofstream out;
  out.exceptions(std::ios::failbit | std::ios::badbit);
  out.imbue(std::locale::classic());
  out.open(path);
  out << std::fixed << std::setprecision(6);
  return out;
}

// Writes numerator / denominator, or N/A when the denominator is zero.
void WriteRatio(std::ostream& out, double numerator, double denominator) {
  if (denominator == 0.0) {
    out << "N/A";
  } else {
    out << numerator / denominator;
  }
}

void WriteWinRate(std::ostream& out, int wins, int games) {
  if (games == 0) {
    out << "N/A";
  } else {
    out << 100.0 * wins / games;
  }
}

void WritePlayerStatistics(std::ostream& out, size_t player, const char* color,
                           const PlayerStatistics& stats) {
  out << player + 1 << ',' << color << ',' << stats.games << ',' << stats.wins
      << ',' << stats.losses << ',' << stats.draws << ',';
  WriteWinRate(out, stats.wins, stats.games);
  out << '\n';
}

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

void WriteSearchStatistics(std::ostream& out, size_t player, const char* color,
                           const TimingStatistics& stats) {
  const Counters& counters = stats.counters;
  const double moves = static_cast<double>(stats.moves);
  out << player + 1 << ',' << color << ',' << stats.moves << ','
      << counters.playouts << ',';
  WriteRatio(out, counters.playouts, moves);
  out << ',' << counters.batches << ',';
  WriteRatio(out, counters.batched_positions, counters.batches);
  out << ',' << counters.collisions << ',';
  WriteRatio(out, counters.collisions, moves);
  out << '\n';
}

// Writes summary.txt with run metadata, AI settings, win rates, and timings.
void WriteSummary(const std::filesystem::path& directory,
                  const SimulationOptions& options, const Configuration& config,
                  const SimulationStatistics& stats, const char* status,
                  const std::string& error = "") {
  auto out = OpenLog(directory / "summary.txt");
  const PlayerStatistics black = ColorTotal(stats, Player::kBlack);
  const int completed_games = black.games;

  out << "status = " << status << "\nrequested_games = " << options.num_games
      << "\ncompleted_games = " << completed_games
      << "\nmodel_path = " << options.model_path
      << "\ntime_unit = milliseconds\ntiming_scope = completed_games\n"
      << "win_rate = wins / games (draws included)\n";
  if (!error.empty()) out << "error = " << error << '\n';

  out << "\n[configuration]\n";
  for (AIPlayerId player : {AIPlayerId::kPlayer1, AIPlayerId::kPlayer2}) {
    WriteAIOptions(out, player, config.ai_options(player));
  }

  out << "\n[overall]\ncolor,games,wins,win_rate_pct\n";
  for (Player color : kColors) {
    const int wins = ColorTotal(stats, color).wins;
    out << ColorName(color) << ',' << completed_games << ',' << wins << ',';
    WriteWinRate(out, wins, completed_games);
    out << '\n';
  }
  out << "draws = " << black.draws << '\n';

  out << "\n[players]\nplayer_id,color,games,wins,losses,draws,win_rate_pct\n";
  for (size_t player = 0; player < stats.size(); ++player) {
    for (Player color : kColors) {
      WritePlayerStatistics(out, player, ColorName(color),
                            stats[player][Index(color)]);
    }
    WritePlayerStatistics(out, player, "all", PlayerTotal(stats, player));
  }

  out << "\n[timing]\nplayer_id,color,moves,metric,total_ms,mean_ms,max_ms\n";
  for (size_t player = 0; player < stats.size(); ++player) {
    for (Player color : kColors) {
      WriteTimingStatistics(out, player, ColorName(color),
                            stats[player][Index(color)].timing);
    }
    WriteTimingStatistics(out, player, "all",
                          PlayerTotal(stats, player).timing);
  }

  out << "\n[search]\nplayer_id,color,moves,total_playouts,mean_playouts,"
         "total_batches,mean_batch_size,total_collisions,mean_collisions\n";
  for (size_t player = 0; player < stats.size(); ++player) {
    for (Player color : kColors) {
      WriteSearchStatistics(out, player, ColorName(color),
                            stats[player][Index(color)].timing);
    }
    WriteSearchStatistics(out, player, "all",
                          PlayerTotal(stats, player).timing);
  }

  out.close();
}

void WriteGameRow(std::ostream& out, int game_number, const Seats& seats,
                  const GameResult& game) {
  out << game_number << ',' << SeatIndex(seats, Player::kBlack) + 1 << ','
      << SeatIndex(seats, Player::kWhite) + 1 << ',';
  if (game.winner) {
    out << SeatIndex(seats, *game.winner) + 1 << ','
        << ColorName(*game.winner);
  } else {
    out << ",draw";
  }
  out << ',' << game.total_moves << ',' << game.duration_ms << '\n';
}

void PrintProgress(std::ostream& out, int game_number, int num_games,
                   const Seats& seats, const GameResult& game) {
  out << "Game " << game_number << '/' << num_games << ": black=Player "
      << SeatIndex(seats, Player::kBlack) + 1 << ", white=Player "
      << SeatIndex(seats, Player::kWhite) + 1 << ", winner=";
  if (game.winner) {
    out << "Player " << SeatIndex(seats, *game.winner) + 1 << " ("
        << ColorName(*game.winner) << ')';
  } else {
    out << "draw";
  }
  out << ", moves=" << game.total_moves << std::endl;
}

// ---------------------------------------------------------------------------
// Game loop
// ---------------------------------------------------------------------------

// Plays a single game to completion and logs each move to `moves`.
GameResult PlayGame(int game_number, Match& match, std::ostream& moves) {
  const Game& game = match.game();
  GameResult result;
  const auto game_start = Clock::now();

  while (game.state() == Game::State::kPlaying) {
    const Player color = game.current_player();
    const size_t player = static_cast<size_t>(*match.seat(color));
    const int move_number = game.next_move_number();

    SearchTiming timing;
    const auto start = Clock::now();
    const auto action = match.Think(&timing);
    const Durations durations = {Milliseconds(Clock::now() - start),
                                 timing.selection_ms, timing.expansion_ms,
                                 timing.evaluation_ms, timing.inference_ms,
                                 timing.backup_ms};
    const Counters counters = {timing.playouts, timing.batches,
                               timing.batched_positions, timing.collisions};

    if (!action || !match.Play(*action)) {
      throw std::runtime_error("AI returned no legal move");
    }
    result.timing[player].Add(durations, counters);

    moves << game_number << ',' << move_number << ',' << player + 1 << ','
          << ColorName(color) << ',' << ActionRow(*action) << ','
          << ActionCol(*action);
    for (double duration : durations) moves << ',' << duration;
    moves << ',' << counters.playouts << ',' << counters.batches << ',';
    WriteRatio(moves, counters.batched_positions, counters.batches);
    moves << ',' << counters.collisions << '\n';
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
             "expansion_ms,evaluation_ms,inference_ms,backup_ms,playouts,"
             "batches,mean_batch_size,collisions\n";
    games << "game,black_player_id,white_player_id,winner_player_id,"
             "winner_color,total_moves,duration_ms\n";
    moves.flush();
    games.flush();

    for (int index = 0; index < options.num_games; ++index) {
      const int game_number = index + 1;

      // Alternate colors between AI Player 1 and AI Player 2 each game.
      const bool swapped = index % 2 == 1;
      const Seats seats = {
          swapped ? AIPlayerId::kPlayer2 : AIPlayerId::kPlayer1,
          swapped ? AIPlayerId::kPlayer1 : AIPlayerId::kPlayer2};
      Match match(config, Game::Mode::kAIvsAI, seats, evaluator);

      const GameResult result = PlayGame(game_number, match, moves);
      WriteGameRow(games, game_number, seats, result);
      moves.flush();
      games.flush();

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
