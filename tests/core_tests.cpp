#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "ai_player.h"
#include "file_manager.h"
#include "game.h"
#include "policy_value_network.h"
#include "replay_controller.h"
#include "test_support.h"

namespace {

using test::CheckThrows;
using test::DrawMoves;

bool Near(float a, float b) { return std::abs(a - b) < 1e-5f; }

class FakeEvaluator : public PolicyValueEvaluator {
 public:
  int calls = 0;
  Evaluation output;
  std::function<Evaluation(const GamePosition&)> evaluate;

  FakeEvaluator() { output.policy.fill(1.f / kActionCount); }

  Evaluation Evaluate(const GamePosition& position) override {
    CHECK(!position.is_terminal());
    ++calls;
    return evaluate ? evaluate(position) : output;
  }
};

GamePosition PositionBeforeWin(Player player) {
  GamePosition position;
  if (player == Player::kWhite) position.PlayMove(ToAction(14, 14));
  for (int col = 0; col < 4; ++col) {
    position.PlayMove(ToAction(7, col));
    position.PlayMove(ToAction(0, col));
  }
  CHECK(position.current_player() == player);
  return position;
}

void BoardRules() {
  Board board;
  CHECK(board.stone_count() == 0);
  CHECK(board.LegalActions().size() == kActionCount);
  for (const auto& cell : std::vector<std::pair<int, int>>{
           {-1, 0}, {0, -1}, {kBoardSize, 0}, {0, kBoardSize}})
    CHECK(!IsValidCell(cell.first, cell.second));
  CHECK(!board.IsLegalMove(-1));
  CHECK(!board.IsLegalMove(kActionCount));
  board.PlaceStone(ToAction(0, 1), Player::kBlack);
  CHECK(!board.IsLegalMove(ToAction(0, 1)));
  CHECK(board.stone_count() == 1);
  CHECK(board.GetCell(ToAction(0, 1)) == Cell::kBlack);

  const int directions[][2] = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};
  for (const auto& direction : directions) {
    board.Reset();
    const int start_col = direction[1] < 0 ? kBoardSize - 1 : 0;
    for (int i = 0; i < 6; ++i) {
      const int row = i * direction[0];
      const int col = start_col + i * direction[1];
      const Action action = ToAction(row, col);
      board.PlaceStone(action, Player::kWhite);
      CHECK(board.CheckWin(action, Player::kWhite) == (i >= 4));
      CHECK(!board.CheckWin(action, Player::kBlack));
    }
    // An empty origin must never count as a fifth stone.
    CHECK(!board.CheckWin(
        ToAction(6 * direction[0], start_col + 6 * direction[1]),
        Player::kWhite));
  }
  board.Reset();
  for (Action action = 0; action < kActionCount; ++action)
    board.PlaceStone(action, Player::kBlack);
  CHECK(board.IsFull());
  CHECK(board.LegalActions().empty());
  board.Reset();
  CHECK(!board.IsFull());
  CHECK(board.stone_count() == 0);
}

void PositionRules() {
  GamePosition position;
  CHECK(position.current_player() == Player::kBlack);
  CHECK(!position.last_move());
  CHECK(!position.is_terminal());
  position.PlayMove(ToAction(0, 1));
  CHECK(position.current_player() == Player::kWhite);
  CHECK(position.last_move() == ToAction(0, 1));
  CHECK(!position.IsLegalMove(ToAction(0, 1)));
  auto copy = position;
  copy.PlayMove(ToAction(14, 12));
  CHECK(position.board().GetCell(ToAction(14, 12)) == Cell::kEmpty);
  for (Player player : {Player::kBlack, Player::kWhite}) {
    auto win = PositionBeforeWin(player);
    win.PlayMove(ToAction(7, 4));
    CHECK(win.is_terminal());
    CHECK(win.winner() == player);
    CHECK(win.current_player() == Opponent(player));
    CHECK(win.LegalActions().empty());
    CHECK(!win.IsLegalMove(ToAction(10, 10)));
  }
  GamePosition draw;
  for (Action action : DrawMoves()) {
    CHECK(!draw.is_terminal());
    draw.PlayMove(action);
  }
  CHECK(draw.result() == PositionResult::kDraw);
  CHECK(!draw.winner());
}

void FeatureContract() {
  GamePosition position;
  auto features = PolicyValueNetwork::EncodeFeatures(position);
  CHECK(std::all_of(features.begin(), features.end(), [](float x) { return x == 0.f; }));
  // Asymmetric corners catch vertical flips, transposes and policy remapping.
  position.PlayMove(ToAction(0, 1));
  PolicyValueNetwork::Features expected{};
  expected[kActionCount + ToAction(0, 1)] = 1.f;
  expected[2 * kActionCount + ToAction(0, 1)] = 1.f;
  CHECK(PolicyValueNetwork::EncodeFeatures(position) == expected);

  position.PlayMove(ToAction(14, 12));
  expected.fill(0.f);
  expected[ToAction(0, 1)] = 1.f;
  expected[kActionCount + ToAction(14, 12)] = 1.f;
  expected[2 * kActionCount + ToAction(14, 12)] = 1.f;
  CHECK(PolicyValueNetwork::EncodeFeatures(position) == expected);

  position.PlayMove(ToAction(4, 6));
  expected.fill(0.f);
  expected[ToAction(14, 12)] = 1.f;
  expected[kActionCount + ToAction(0, 1)] = 1.f;
  expected[kActionCount + ToAction(4, 6)] = 1.f;
  expected[2 * kActionCount + ToAction(4, 6)] = 1.f;
  CHECK(PolicyValueNetwork::EncodeFeatures(position) == expected);
}

void GameUndoAndReplay() {
  Configuration config;
  config.set_undo_enabled(true);
  config.set_time_limit_enabled(false);
  Game game(config);
  game.Start();
  CHECK(!game.TryPlaceStone(-1));
  CHECK(!game.TryPlaceStone(kActionCount));
  CHECK(game.record().size() == 0);
  CHECK(game.TryPlaceStone(ToAction(0, 1)));
  CHECK(!game.TryPlaceStone(ToAction(0, 1)));
  CHECK(game.TryPlaceStone(ToAction(14, 12)));
  const auto before = PolicyValueNetwork::EncodeFeatures(game.position());
  CHECK(game.TryPlaceStone(ToAction(4, 6)));
  CHECK(game.TryPlaceStone(ToAction(9, 8)));
  game.Undo();
  CHECK(PolicyValueNetwork::EncodeFeatures(game.position()) == before);
  CHECK(game.current_player() == Player::kBlack);
  CHECK(game.next_move_number() == 3);
  CHECK(game.record().size() == 2);
  CHECK(game.position().last_move() == ToAction(14, 12));
  CHECK(GamePosition::Rebuild(game.record().moves(), 2).has_value());
  game.Undo();
  CHECK(!game.position().last_move());
  CHECK(game.next_move_number() == 1);
  game.Undo();
  CHECK(game.board().stone_count() == 0);

  for (int col = 0; col < 4; ++col) {
    CHECK(game.TryPlaceStone(ToAction(7, col)));
    CHECK(game.TryPlaceStone(ToAction(0, col)));
  }
  CHECK(game.TryPlaceStone(ToAction(7, 4)));
  CHECK(game.state() == Game::State::kWin);
  CHECK(game.winner() == Player::kBlack);
  CHECK(!game.TryPlaceStone(ToAction(10, 10)));
  auto moves = game.record().moves();
  CHECK(GamePosition::Rebuild(moves, static_cast<int>(moves.size()))->winner() == Player::kBlack);
  moves.push_back({10, Player::kWhite, 10, 10});
  CHECK(!GamePosition::Rebuild(moves, static_cast<int>(moves.size())));
  moves = game.record().moves();
  moves[0].player = Player::kWhite;
  CHECK(!GamePosition::Rebuild(moves, 1));
  moves[0] = {1, Player::kBlack, -1, 0};
  CHECK(!GamePosition::Rebuild(moves, 1));
  CHECK(!GamePosition::Rebuild(moves, -1));

  // Verify Forfeit transitions state and assigns the opponent as winner.
  game.Start();
  CHECK(game.TryPlaceStone(ToAction(7, 7)));  // Black plays; now White's turn.
  game.Forfeit();
  CHECK(game.state() == Game::State::kForfeit);
  CHECK(game.winner() == Player::kBlack);
  CHECK(game.FinalizeRecord().metadata().winner == Player::kBlack);

  game.Start();
  CHECK(!game.position().last_move());
  CHECK(!game.winner());
  config.set_time_limit_enabled(true);
  config.set_time_limit_seconds(0);
  Game timeout_game(config);
  CHECK(!timeout_game.TryPlaceStone(ToAction(0, 0)));
  CHECK(timeout_game.state() == Game::State::kTimeout);
  CHECK(timeout_game.winner() == Player::kWhite);
  CHECK(timeout_game.board().stone_count() == 0);
}

// Verifies role queries, human-only timers, and mode-specific undo/metadata.
void GameModesAndTimers() {
  Configuration config;
  config.set_time_limit_enabled(true);
  config.set_time_limit_seconds(60);
  config.set_undo_enabled(true);

  Game human(config);
  CHECK(human.mode() == Game::Mode::kPlayerVsPlayer);
  CHECK(!human.IsAIPlayer(Player::kBlack));
  CHECK(!human.IsAIPlayer(Player::kWhite));
  CHECK(!human.GetAIPlayerId(Player::kBlack));
  CHECK(!human.GetAIPlayerId(Player::kWhite));
  CHECK(human.GetSecondsRemaining() > 0.0f);
  CHECK(human.TryPlaceStone(ToAction(7, 7)));
  CHECK(human.GetSecondsRemaining() > 0.0f);

  Game mixed(config, Game::Mode::kPlayerVsAI);
  CHECK(!mixed.IsAIPlayer(Player::kBlack));
  CHECK(mixed.IsAIPlayer(Player::kWhite));
  CHECK(!mixed.GetAIPlayerId(Player::kBlack));
  CHECK(mixed.GetAIPlayerId(Player::kWhite) == AIPlayerId::kPlayer1);
  CHECK(mixed.GetSecondsRemaining() > 0.0f);
  CHECK(mixed.TryPlaceStone(ToAction(7, 7)));
  CHECK(mixed.GetSecondsRemaining() == -1.0f);
  mixed.Update();  // Disabled timer sentinel on AI turns must not time out.
  CHECK(mixed.state() == Game::State::kPlaying);
  CHECK(mixed.TryPlaceStone(ToAction(7, 8)));
  CHECK(mixed.GetSecondsRemaining() > 59.0f);
  mixed.Undo();
  CHECK(mixed.board().stone_count() == 0);
  CHECK(mixed.GetSecondsRemaining() > 59.0f);
  mixed.Forfeit();
  CHECK(mixed.FinalizeRecord().metadata().time_limit_enabled);

  config.set_time_limit_seconds(0);
  Game zero_time_human(config, Game::Mode::kPlayerVsAI);
  zero_time_human.Update();
  CHECK(zero_time_human.state() == Game::State::kTimeout);
  CHECK(zero_time_human.winner() == Player::kWhite);

  Game self_play(config, Game::Mode::kAIvsAI);
  CHECK(self_play.IsAIPlayer(Player::kBlack));
  CHECK(self_play.IsAIPlayer(Player::kWhite));
  CHECK(self_play.GetAIPlayerId(Player::kBlack) == AIPlayerId::kPlayer1);
  CHECK(self_play.GetAIPlayerId(Player::kWhite) == AIPlayerId::kPlayer2);
  for (int col = 0; col < 4; ++col) {
    CHECK(self_play.GetSecondsRemaining() == -1.0f);
    self_play.Update();
    CHECK(self_play.state() == Game::State::kPlaying);
    CHECK(self_play.TryPlaceStone(ToAction(7, col)));
    CHECK(self_play.TryPlaceStone(ToAction(0, col)));
  }
  self_play.Undo();  // Undo is disabled in AI vs AI.
  CHECK(self_play.record().size() == 8);
  CHECK(self_play.TryPlaceStone(ToAction(7, 4)));
  CHECK(self_play.state() == Game::State::kWin);
  const auto record = self_play.FinalizeRecord();
  CHECK(!record.metadata().time_limit_enabled);
  CHECK(!record.metadata().undo_enabled);
  CHECK(config.time_limit_enabled());
  CHECK(config.undo_enabled());
  self_play.Start();
  CHECK(self_play.mode() == Game::Mode::kAIvsAI);
  CHECK(self_play.GetAIPlayerId(Player::kBlack) == AIPlayerId::kPlayer1);
  CHECK(self_play.GetAIPlayerId(Player::kWhite) == AIPlayerId::kPlayer2);
  CHECK(self_play.GetSecondsRemaining() == -1.0f);
  CHECK(self_play.record().size() == 0);
}

struct ScopedTempDir {
  std::filesystem::path original;
  std::filesystem::path directory;

  explicit ScopedTempDir(const std::string& prefix)
      : original(std::filesystem::current_path()),
        directory(original /
                  (prefix + "-" +
                   std::to_string(std::chrono::steady_clock::now()
                                      .time_since_epoch()
                                      .count()))) {
    std::filesystem::create_directories(directory);
    std::filesystem::current_path(directory);
  }

  ~ScopedTempDir() {
    std::filesystem::current_path(original);
    std::error_code ec;
    std::filesystem::remove_all(directory, ec);
  }
};

// Verifies loading, saving, and validation of per-player AI search options.
void AIConfiguration() {
  ScopedTempDir temp_dir("test-ai-config");

  Configuration defaults;
  for (AIPlayerId player : {AIPlayerId::kPlayer1, AIPlayerId::kPlayer2}) {
    CHECK(defaults.ai_options(player).num_playouts == 400);
    CHECK(Near(defaults.ai_options(player).c_puct, 5.0f));
    CHECK(defaults.ai_options(player).temperature == 1.0f);
    CHECK(defaults.ai_options(player).num_sampling_moves == 30);
  }
  {
    std::ofstream("gomoku.cfg")
        << "undo_enabled = true\ntime_limit_seconds = 30\n"
        << "black_ai_num_playouts = 17\nwhite_ai_c_puct = 2.5\n";
  }
  Configuration partial;
  CHECK(partial.undo_enabled());
  CHECK(partial.time_limit_seconds() == 30);
  CHECK(partial.ai_options(AIPlayerId::kPlayer1).num_playouts == 400);
  CHECK(Near(partial.ai_options(AIPlayerId::kPlayer2).c_puct, 5.0f));
  for (AIPlayerId player : {AIPlayerId::kPlayer1, AIPlayerId::kPlayer2}) {
    CHECK(partial.ai_options(player).temperature == 1.0f);
    CHECK(partial.ai_options(player).num_sampling_moves == 30);
  }

  partial.set_ai_options(AIPlayerId::kPlayer1, {1.2345678f, 7, 0.12345678f, 7});
  partial.set_ai_options(AIPlayerId::kPlayer2,
                         {8.7654321f, 19, 2.5f, kActionCount});
  partial.Save();
  {
    std::ifstream file("gomoku.cfg");
    const std::string saved{std::istreambuf_iterator<char>(file),
                            std::istreambuf_iterator<char>()};
    CHECK(saved.find("ai_player_1_num_playouts = 7\n") != std::string::npos);
    CHECK(saved.find("ai_player_2_num_playouts = 19\n") != std::string::npos);
    CHECK(saved.find("black_ai_") == std::string::npos);
    CHECK(saved.find("white_ai_") == std::string::npos);
  }
  Configuration restored;
  for (AIPlayerId player : {AIPlayerId::kPlayer1, AIPlayerId::kPlayer2}) {
    CHECK(restored.ai_options(player).num_playouts ==
          partial.ai_options(player).num_playouts);
    CHECK(restored.ai_options(player).c_puct ==
          partial.ai_options(player).c_puct);
    CHECK(restored.ai_options(player).temperature ==
          partial.ai_options(player).temperature);
    CHECK(restored.ai_options(player).num_sampling_moves ==
          partial.ai_options(player).num_sampling_moves);
  }
  CHECK(restored.undo_enabled());
  CHECK(restored.time_limit_seconds() == 30);

  for (SearchOptions invalid :
       {SearchOptions{0.0f, 1}, SearchOptions{-1.0f, 1}, SearchOptions{5.0f, 0},
        SearchOptions{5.0f, -1},
        SearchOptions{std::numeric_limits<float>::infinity(), 1},
        SearchOptions{std::numeric_limits<float>::quiet_NaN(), 1},
        SearchOptions{5.0f, 1, -1.0f, 30},
        SearchOptions{5.0f, 1, std::numeric_limits<float>::infinity(), 30},
        SearchOptions{5.0f, 1, std::numeric_limits<float>::quiet_NaN(), 30},
        SearchOptions{5.0f, 1, 1.0f, -1},
        SearchOptions{5.0f, 1, 1.0f, kActionCount + 1}}) {
    CHECK(!invalid.IsValid());
    CheckThrows<std::invalid_argument>([&] {
      restored.set_ai_options(AIPlayerId::kPlayer1, invalid);
    });
    CheckThrows<std::invalid_argument>([&] { MCTS search(invalid); });
    CHECK(restored.ai_options(AIPlayerId::kPlayer1).num_playouts == 7);
    CHECK(restored.ai_options(AIPlayerId::kPlayer1).c_puct == 1.2345678f);
    CHECK(restored.ai_options(AIPlayerId::kPlayer1).temperature == 0.12345678f);
    CHECK(restored.ai_options(AIPlayerId::kPlayer1).num_sampling_moves == 7);
  }

  for (const char* value :
       {"0", "-2", "nan", "inf", "1e9999", "oops", "2junk"}) {
    {
      std::ofstream file("gomoku.cfg");
      file << "ai_player_1_num_playouts = " << value << "\n"
           << "ai_player_2_c_puct = " << value << "\n"
           << "ai_player_1_c_puct = 2.5\nai_player_2_num_playouts = 17\n";
    }
    Configuration invalid;
    CHECK(invalid.ai_options(AIPlayerId::kPlayer1).num_playouts == 400);
    CHECK(invalid.ai_options(AIPlayerId::kPlayer1).c_puct == 2.5f);
    CHECK(invalid.ai_options(AIPlayerId::kPlayer2).num_playouts == 17);
    CHECK(invalid.ai_options(AIPlayerId::kPlayer2).c_puct == 5.0f);
  }
}

// Verifies parsing and boundary validation of move sampling configuration.
void MoveSamplingConfiguration() {
  ScopedTempDir temp_dir("test-sampling-config");

  for (const char* value :
       {"-2", "nan", "inf", "-inf", "1e9999", "1e-100", "oops", "2junk"}) {
    {
      std::ofstream file("gomoku.cfg");
      file << "ai_player_1_temperature = 0.5\n"
           << "ai_player_1_temperature = " << value << "\n"
           << "ai_player_2_num_sampling_moves = 12\n"
           << "ai_player_2_num_sampling_moves = " << value << "\n"
           << "ai_player_2_temperature = 2\n"
           << "ai_player_1_num_sampling_moves = 9\n";
    }
    Configuration config;
    CHECK(config.ai_options(AIPlayerId::kPlayer1).temperature == 1.0f);
    CHECK(config.ai_options(AIPlayerId::kPlayer1).num_sampling_moves == 9);
    CHECK(config.ai_options(AIPlayerId::kPlayer2).temperature == 2.0f);
    CHECK(config.ai_options(AIPlayerId::kPlayer2).num_sampling_moves == 30);
  }
  {
    std::ofstream file("gomoku.cfg");
    file << "ai_player_1_temperature = 0\nai_player_1_num_sampling_moves = 0\n"
         << "ai_player_2_temperature = 226\n"
         << "ai_player_2_num_sampling_moves = 226\n";
  }
  Configuration config;
  CHECK(config.ai_options(AIPlayerId::kPlayer1).IsValid());
  CHECK(config.ai_options(AIPlayerId::kPlayer1).temperature == 0.0f);
  CHECK(config.ai_options(AIPlayerId::kPlayer1).num_sampling_moves == 0);
  CHECK(config.ai_options(AIPlayerId::kPlayer2).temperature == 226.0f);
  CHECK(config.ai_options(AIPlayerId::kPlayer2).num_sampling_moves == 30);
  config.Save();
  Configuration restored;
  CHECK(restored.ai_options(AIPlayerId::kPlayer1).temperature == 0.0f);
  CHECK(restored.ai_options(AIPlayerId::kPlayer1).num_sampling_moves == 0);
  for (float temperature :
       {std::numeric_limits<float>::denorm_min(),
        std::numeric_limits<float>::min(), std::numeric_limits<float>::max()}) {
    restored.set_ai_options(AIPlayerId::kPlayer1, {5.0f, 11, temperature, 30});
    restored.Save();
    Configuration reloaded;
    CHECK(reloaded.ai_options(AIPlayerId::kPlayer1).temperature == temperature);
  }
}

// Verifies saved-game file discovery.
void SavedGameFiles() {
  ScopedTempDir temp_dir("test-saved-games");
  std::vector<std::string> paths{"stale.gom"};
  CHECK(FileManager::ListGameFiles("saves", &paths));
  CHECK(paths.empty());

  std::filesystem::create_directory("saves");
  CHECK(FileManager::ListGameFiles("saves", &paths));
  CHECK(paths.empty());

  GameRecord record;
  record.AddMove({1, Player::kBlack, 7, 7});
  GameMetadata metadata;
  metadata.total_moves = 1;
  record.Finalize(metadata);
  CHECK(FileManager::Save(record, "saves/b.gom"));
  CHECK(FileManager::Save(record, "saves/a game.gom"));
  std::ofstream("saves/notes.txt");
  std::ofstream("saves/upper.GOM");
  std::filesystem::create_directory("saves/nested.gom");
  CHECK(FileManager::Save(record, "saves/nested.gom/hidden.gom"));

  CHECK(FileManager::ListGameFiles("saves", &paths));
  CHECK(paths == std::vector<std::string>({"saves/a game.gom", "saves/b.gom"}));
  ReplayController replay;
  for (const auto& path : paths) {
    CHECK(replay.Load(path));
    CHECK(replay.total_moves() == 1);
    CHECK(replay.BoardAtStep(1).GetCell(ToAction(7, 7)) == Cell::kBlack);
  }

  CHECK(FileManager::Save(record, "saves/c.gom"));
  std::filesystem::remove("saves/b.gom");
  CHECK(FileManager::ListGameFiles("saves", &paths));
  CHECK(paths == std::vector<std::string>({"saves/a game.gom", "saves/c.gom"}));
  CHECK(!FileManager::ListGameFiles("saves/notes.txt", &paths));
  CHECK(paths.empty());
  CHECK(!FileManager::ListGameFiles("saves", nullptr));
}

void FileValidation() {
  // CTest runs this executable in the build directory, keeping test files local.
  const auto path = std::filesystem::current_path() / "test-record.gom";
  struct Cleanup {
    std::filesystem::path path;
    ~Cleanup() { std::error_code ec; std::filesystem::remove(path, ec); }
  } cleanup{path};
  GameRecord record;
  record.AddMove({1, Player::kBlack, 0, 1});
  record.AddMove({2, Player::kWhite, 14, 12});
  GameMetadata metadata;
  metadata.total_moves = 2;
  record.Finalize(metadata);
  CHECK(FileManager::Save(record, path.string()));
  ReplayController replay;
  CHECK(replay.Load(path.string()));
  CHECK(replay.BoardAtStep(0).stone_count() == 0);
  CHECK(replay.BoardAtStep(1).GetCell(ToAction(0, 1)) == Cell::kBlack);
  CHECK(replay.BoardAtStep(2).GetCell(ToAction(14, 12)) == Cell::kWhite);
  CHECK(replay.BoardAtStep(99).stone_count() == 2);
  CHECK(replay.BoardAtStep(-1).stone_count() == 0);

  for (const auto& bad_moves : std::vector<std::vector<MoveRecord>>{
           {{1, Player::kBlack, 0, 0}, {2, Player::kWhite, 0, 0}},
           {{1, Player::kWhite, 0, 0}},
           {{1, Player::kBlack, 15, 0}},
           {{3, Player::kBlack, 0, 0}}}) {
    GameRecord bad;
    for (const auto& move : bad_moves) bad.AddMove(move);
    metadata.total_moves = bad.size();
    bad.Finalize(metadata);
    CHECK(FileManager::Save(bad, path.string()));
    CHECK(!replay.Load(path.string()));
    CHECK(replay.total_moves() == 2);  // A failed load preserves the old replay.
  }
  {
    std::ofstream file(path);
    file << "[metadata]\ntotal_moves = invalid\n[moves]\n";
  }
  CHECK(!replay.Load(path.string()));
}

void SearchPriorsAndOptions() {
  for (const auto& options : std::vector<SearchOptions>{
           {0.f, 1}, {-1.f, 1}, {5.f, 0}, {5.f, -1},
           {std::numeric_limits<float>::infinity(), 1}})
    CheckThrows<std::invalid_argument>([&] { MCTS search(options); });

  GamePosition position;
  position.PlayMove(0);
  FakeEvaluator evaluator;
  evaluator.output.policy.fill(0.f);
  evaluator.output.policy[0] = 0.99f;  // Occupied probability must be discarded.
  evaluator.output.policy[1] = 0.002f;
  evaluator.output.policy[2] = 0.008f;
  MCTS search({5.f, 1});
  auto moves = search.Search(position, evaluator);
  CHECK(moves.size() == kActionCount - 1);
  CHECK(moves.front().action == 1);
  CHECK(Near(moves[0].prior, 0.2f));
  CHECK(Near(moves[1].prior, 0.8f));
  CHECK(Near(std::accumulate(moves.begin(), moves.end(), 0.f,
      [](float sum, const MoveVisits& move) { return sum + move.prior; }), 1.f));

  search.Reset();
  evaluator.output.policy.fill(0.f);
  evaluator.output.policy[0] = 1.f;  // All legal moves have zero probability.
  moves = search.Search(position, evaluator);
  for (const auto& move : moves) CHECK(Near(move.prior, 1.f / 224));

  for (float invalid : {-1.f, std::numeric_limits<float>::quiet_NaN(),
                        std::numeric_limits<float>::infinity()}) {
    evaluator.output.policy[1] = invalid;
    search.Reset();
    CheckThrows<std::runtime_error>([&] { search.Search(position, evaluator); });
  }
  evaluator.output.policy.fill(1.f);
  evaluator.output.value = 2.f;
  CheckThrows<std::runtime_error>([&] { search.Search(position, evaluator); });
}

void SearchValuePerspective() {
  // A terminal winning child must be +1 for the player choosing that move.
  for (Player player : {Player::kBlack, Player::kWhite}) {
    auto position = PositionBeforeWin(player);
    FakeEvaluator evaluator;
    evaluator.output.policy.fill(0.f);
    evaluator.output.policy[ToAction(7, 4)] = 1.f;
    MCTS search({5.f, 2});
    auto moves = search.Search(position, evaluator);
    auto win = std::find_if(moves.begin(), moves.end(), [](const MoveVisits& move) {
      return move.action == ToAction(7, 4);
    });
    CHECK(win != moves.end());
    CHECK(win->visits == 1);
    CHECK(Near(win->value, 1.f));
    CHECK(evaluator.calls == 1);
    position.PlayMove(ToAction(7, 4));
    CHECK(search.Search(position, evaluator).empty());
    CHECK(evaluator.calls == 1);
  }

  // The child's predicted advantage is a disadvantage for the parent player.
  FakeEvaluator evaluator;
  evaluator.output.policy.fill(0.f);
  evaluator.output.policy[17] = 1.f;
  evaluator.output.value = 0.75f;
  MCTS search({5.f, 2});
  auto moves = search.Search(GamePosition(), evaluator);
  CHECK(Near(moves[17].value, -0.75f));

  // A full-board draw does not invoke the model or give either player a win.
  GamePosition draw;
  const auto actions = DrawMoves();
  for (size_t i = 0; i + 1 < actions.size(); ++i) draw.PlayMove(actions[i]);
  evaluator.calls = 0;
  evaluator.output.policy.fill(1.f);
  search.Reset();
  moves = search.Search(draw, evaluator);
  CHECK(moves.size() == 1);
  CHECK(Near(moves[0].value, 0.f));
  CHECK(evaluator.calls == 1);
  draw.PlayMove(actions.back());
  CHECK(search.Search(draw, evaluator).empty());
}

void TreeLifecycleAndAI() {
  auto evaluator = std::make_unique<FakeEvaluator>();
  auto* fake = evaluator.get();
  fake->output.policy.fill(0.f);
  fake->output.policy[19] = 0.8f;
  fake->output.policy[20] = 0.2f;
  AIPlayer player(std::move(evaluator), {5.f, 1});
  GamePosition position;
  CHECK(player.PickMove(position) == 19);  // Prior breaks the zero-visit tie.
  CHECK(player.PickMove(position) == 19);  // Selection did not commit a move.
  CHECK(position.board().stone_count() == 0);
  position.PlayMove(19);
  player.ObserveMove(19);
  auto next = player.PickMove(position);
  CHECK(next.has_value());
  CHECK(position.IsLegalMove(*next));
  position.PlayMove(*next);
  player.ObserveMove(*next);
  CHECK(player.PickMove(position).has_value());
  player.Reset();
  CHECK(player.PickMove(GamePosition()) == 19);

  auto terminal = PositionBeforeWin(Player::kBlack);
  terminal.PlayMove(ToAction(7, 4));
  const int calls = fake->calls;
  CHECK(!player.PickMove(terminal));
  CHECK(fake->calls == calls);

  // Reusing an already-expanded child avoids evaluating it as a fresh root
  // and keeps grandchild->parent pointers valid after moving the new root.
  MCTS search({5.f, 3});
  FakeEvaluator reuse;
  reuse.output.policy.fill(0.f);
  reuse.output.policy[0] = 1.f;
  reuse.output.policy[1] = 0.5f;
  auto stats = search.Search(GamePosition(), reuse);
  CHECK(stats[0].visits == 2);
  search.ObserveMove(0);
  GamePosition after;
  after.PlayMove(0);
  const int previous_calls = reuse.calls;
  stats = search.Search(after, reuse);
  CHECK(reuse.calls == previous_calls + 3);
  CHECK(std::accumulate(stats.begin(), stats.end(), 0,
      [](int sum, const MoveVisits& move) { return sum + move.visits; }) == 4);
  search.Reset();
  stats = search.Search(GamePosition(), reuse);
  CHECK(stats[0].visits == 2);

  // The existing single background job must transfer exceptions to its caller.
  auto throwing = std::make_unique<FakeEvaluator>();
  throwing->evaluate = [](const GamePosition&) -> Evaluation {
    throw std::runtime_error("model failed");
  };
  AIPlayer failing(std::move(throwing), {5.f, 1});
  auto result = std::async(std::launch::async, [&] {
    return failing.PickMove(GamePosition());
  });
  CheckThrows<std::runtime_error>([&] { result.get(); });
  CHECK(!result.valid());
  failing.Reset();
  CheckThrows<std::invalid_argument>([] { AIPlayer no_model(nullptr); });
}

// These priors yield root visits [6, 3, 1] after 11 simulations at value zero.
std::shared_ptr<FakeEvaluator> SamplingEvaluator(const GamePosition& position) {
  const auto actions = position.LegalActions();
  CHECK(actions.size() >= 3);
  auto evaluator = std::make_shared<FakeEvaluator>();
  evaluator->output.policy.fill(0.0f);
  evaluator->output.policy[actions[0]] = 0.6f;
  evaluator->output.policy[actions[1]] = 0.3f;
  evaluator->output.policy[actions[2]] = 0.1f;
  return evaluator;
}

// Verifies that move sampling matches the temperature-weighted visit distribution.
void MoveSamplingDistribution() {
  GamePosition position;
  const auto draw_moves = DrawMoves();
  for (int i = 0; i < kActionCount - 3; ++i) position.PlayMove(draw_moves[i]);
  const auto actions = position.LegalActions();
  auto evaluator = SamplingEvaluator(position);
  const std::array<int, 3> visits = {6, 3, 1};
  MCTS reference({5.0f, 11});
  const auto candidates = reference.Search(position, *evaluator);
  CHECK(candidates.size() == visits.size());
  for (size_t i = 0; i < visits.size(); ++i) {
    CHECK(candidates[i].action == actions[i]);
    CHECK(candidates[i].visits == visits[i]);
  }

  constexpr int kSamples = 10000;
  for (float temperature : {0.5f, 1.0f, 2.0f}) {
    const SearchOptions options{5.0f, 11, temperature, kActionCount};
    AIPlayer player(evaluator, options, 12345);
    AIPlayer identical(evaluator, options, 12345);
    std::array<int, 3> counts{};
    for (int i = 0; i < kSamples; ++i) {
      player.Reset();
      identical.Reset();
      const auto action = player.PickMove(position);
      CHECK(action.has_value());
      CHECK(action == identical.PickMove(position));
      const auto found = std::find(actions.begin(), actions.end(), *action);
      CHECK(found != actions.end());
      ++counts[static_cast<size_t>(found - actions.begin())];
    }
    double total_weight = 0.0;
    for (int count : visits) {
      total_weight += std::pow(count, 1.0 / temperature);
    }
    for (size_t i = 0; i < visits.size(); ++i) {
      const double expected =
          std::pow(visits[i], 1.0 / temperature) / total_weight;
      const double actual = static_cast<double>(counts[i]) / kSamples;
      CHECK(std::abs(actual - expected) < 0.03);
    }
  }
}

// Verifies sampling move count limits, undo restoration, and disabling options.
void MoveSamplingBoundaries() {
  auto evaluator = SamplingEvaluator(GamePosition());
  AIPlayer player(evaluator, {5.0f, 11, 1.0f, 4}, 12345);
  const auto count_alternatives = [&](const GamePosition& position,
                                      AIPlayer& ai) {
    evaluator->output = SamplingEvaluator(position)->output;
    const Action greedy = position.LegalActions().front();
    int alternatives = 0;
    for (int i = 0; i < 64; ++i) {
      ai.Reset();
      const auto action = ai.PickMove(position);
      CHECK(action.has_value());
      CHECK(position.IsLegalMove(*action));
      if (*action != greedy) ++alternatives;
    }
    return alternatives;
  };

  Configuration config;
  config.set_undo_enabled(true);
  config.set_time_limit_enabled(false);
  Game game(config, Game::Mode::kPlayerVsAI);
  CHECK(count_alternatives(game.position(), player) > 0);
  for (int col = 14; col > 11; --col) CHECK(game.TryPlaceStone(ToAction(14, col)));
  CHECK(game.board().stone_count() == 3);
  CHECK(count_alternatives(game.position(), player) > 0);
  CHECK(game.TryPlaceStone(ToAction(14, 11)));
  CHECK(game.board().stone_count() == 4);
  CHECK(count_alternatives(game.position(), player) == 0);

  game.Undo();
  CHECK(game.board().stone_count() == 2);
  CHECK(count_alternatives(game.position(), player) > 0);
  game.Start();
  CHECK(count_alternatives(game.position(), player) > 0);

  AIPlayer zero_temperature(evaluator, {5.0f, 11, 0.0f, 30}, 12345);
  CHECK(count_alternatives(game.position(), zero_temperature) == 0);
  AIPlayer zero_moves(evaluator, {5.0f, 11, 1.0f, 0}, 12345);
  CHECK(count_alternatives(game.position(), zero_moves) == 0);
}

// Verifies fallback and extreme temperature behavior during move sampling.
void MoveSamplingEdgeCases() {
  GamePosition position;
  auto evaluator = SamplingEvaluator(position);
  AIPlayer unvisited(evaluator, {5.0f, 1}, 12345);
  CHECK(unvisited.PickMove(position) == 0);
  evaluator->output.policy[0] = 0.3f;
  CHECK(unvisited.PickMove(position) == 0);
  unvisited.Reset();
  CHECK(unvisited.PickMove(position) == 0);

  evaluator = SamplingEvaluator(position);
  AIPlayer one_visited(evaluator,
                       {5.0f, 2, std::numeric_limits<float>::max(), 30}, 12345);
  AIPlayer cold(evaluator,
                {5.0f, 11, std::numeric_limits<float>::denorm_min(), 30},
                12345);
  for (int i = 0; i < 64; ++i) {
    one_visited.Reset();
    CHECK(one_visited.PickMove(position) == 0);
    cold.Reset();
    CHECK(cold.PickMove(position) == 0);
  }

  const auto moves = DrawMoves();
  for (int i = 0; i < kActionCount - 1; ++i) position.PlayMove(moves[i]);
  AIPlayer last_move(evaluator, {5.0f, 2, 1.0f, kActionCount}, 12345);
  CHECK(last_move.PickMove(position) == moves.back());
  CHECK(position.board().stone_count() == kActionCount - 1);
  position.PlayMove(moves.back());
  last_move.ObserveMove(moves.back());
  const int calls = evaluator->calls;
  CHECK(!last_move.PickMove(position));
  CHECK(evaluator->calls == calls);
}

// Verifies that two AI players share an evaluator while keeping separate trees.
void IndependentAIPlayers() {
  auto evaluator = std::make_shared<FakeEvaluator>();
  std::vector<int> evaluated_stones;
  evaluator->evaluate = [&](const GamePosition& position) {
    evaluated_stones.push_back(position.board().stone_count());
    return evaluator->output;
  };
  AIPlayer black(evaluator, {0.5f, 1});
  AIPlayer white(evaluator, {9.0f, 3});
  GamePosition empty;
  CHECK(black.PickMove(empty).has_value());
  CHECK(evaluator->calls == 1);
  CHECK(white.PickMove(empty).has_value());
  CHECK(evaluator->calls == 4);  // Separate simulation budgets.
  black.Reset();
  evaluated_stones.clear();
  CHECK(white.PickMove(empty).has_value());
  CHECK(evaluated_stones.front() > 0);  // Resetting Black did not reset White.

  Configuration config;
  config.set_time_limit_enabled(true);
  config.set_time_limit_seconds(0);  // Neither AI may lose to this time limit.
  Game game(config, Game::Mode::kAIvsAI);
  for (int round = 0; round < 2; ++round) {
    game.Start();
    black.Reset();
    white.Reset();
    while (game.state() == Game::State::kPlaying) {
      auto& player = game.current_player() == Player::kBlack ? black : white;
      auto result = std::async(std::launch::async, [&] {
        return player.PickMove(game.position());
      });
      const auto action = result.get();
      CHECK(action.has_value());
      CHECK(game.TryPlaceStone(*action));
      black.ObserveMove(*action);
      white.ObserveMove(*action);
      CHECK(game.record().size() <= kActionCount);
    }
    CHECK(game.position().is_terminal());
    CHECK(!black.PickMove(game.position()));
    CHECK(!white.PickMove(game.position()));
    const auto record = game.FinalizeRecord();
    CHECK(!record.metadata().time_limit_enabled);
    CHECK(GamePosition::Rebuild(record.moves(), record.size())->result() ==
          game.position().result());
    const auto path = std::filesystem::current_path() / "test-self-play.gom";
    struct Cleanup {
      std::filesystem::path path;
      ~Cleanup() { std::error_code ec; std::filesystem::remove(path, ec); }
    } cleanup{path};
    CHECK(FileManager::Save(record, path.string()));
    ReplayController replay;
    CHECK(replay.Load(path.string()));
    CHECK(replay.total_moves() == record.size());
    CHECK(replay.BoardAtStep(record.size()).stone_count() == record.size());
  }
}

void FullGameWithUndo() {
  Configuration config;
  config.set_time_limit_enabled(false);
  config.set_undo_enabled(true);
  Game game(config);
  game.Start();
  AIPlayer player(std::make_unique<FakeEvaluator>(), {5.f, 8});
  bool undone = false;
  while (game.state() == Game::State::kPlaying) {
    const auto action = player.PickMove(game.position());
    CHECK(action.has_value());
    CHECK(game.TryPlaceStone(*action));
    player.ObserveMove(*action);
    if (!undone && game.record().size() == 8) {
      game.Undo();
      player.Reset();
      undone = true;
      CHECK(game.record().size() == 6);
    }
    CHECK(game.record().size() <= kActionCount);
  }
  CHECK(undone);
  CHECK(game.position().is_terminal());
  CHECK(!player.PickMove(game.position()));
  const auto record = game.FinalizeRecord();
  CHECK(GamePosition::Rebuild(record.moves(), record.size())->result() == game.position().result());
}

}  // namespace

int main() {
  const std::pair<const char*, void (*)()> tests[] = {
      {"board rules", BoardRules},
      {"position rules", PositionRules},
      {"training feature contract", FeatureContract},
      {"game, undo, replay and timeout", GameUndoAndReplay},
      {"game modes and human-only timers", GameModesAndTimers},
      {"per-player AI configuration", AIConfiguration},
      {"move sampling configuration", MoveSamplingConfiguration},
      {"saved-game discovery", SavedGameFiles},
      {"save/load validation", FileValidation},
      {"search priors and options", SearchPriorsAndOptions},
      {"search value perspective", SearchValuePerspective},
      {"tree lifecycle and AI", TreeLifecycleAndAI},
      {"move sampling distribution", MoveSamplingDistribution},
      {"move sampling boundaries and undo", MoveSamplingBoundaries},
      {"move sampling edge cases", MoveSamplingEdgeCases},
      {"independent AI self-play and replay", IndependentAIPlayers},
      {"full game with undo", FullGameWithUndo},
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
