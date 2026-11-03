#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "ai_player.h"
#include "policy_value_network.h"
#include "file_manager.h"
#include "game.h"
#include "replay_controller.h"

namespace {

#define CHECK(condition)                                                       \
  do {                                                                        \
    if (!(condition))                                                         \
      throw std::runtime_error(std::string(__func__) + ": " + #condition);     \
  } while (false)

bool Near(float a, float b) { return std::abs(a - b) < 1e-5f; }

template <typename Exception, typename Function>
void CheckThrows(Function function) {
  try {
    function();
  } catch (const Exception&) {
    return;
  }
  throw std::runtime_error("Expected exception was not thrown");
}

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

std::vector<Action> DrawMoves() {
  // This full board has no five on any axis: horizontal colors alternate,
  // while vertical/diagonal runs are at most two stones long.
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

void BoardRules() {
  Board board;
  CHECK(board.stone_count() == 0);
  CHECK(board.LegalActions().size() == kActionCount);
  for (const auto& cell : std::vector<std::pair<int, int>>{
           {-1, 0}, {0, -1}, {kBoardSize, 0}, {0, kBoardSize}})
    CHECK(!IsValidCell(cell.first, cell.second));
  CHECK(!board.PlaceStone(-1, Player::kBlack));
  CHECK(!board.PlaceStone(kActionCount, Player::kBlack));
  CHECK(!board.IsLegalMove(-1));
  CHECK(!board.IsLegalMove(kActionCount));
  CHECK(board.PlaceStone(ToAction(0, 1), Player::kBlack));
  CHECK(!board.PlaceStone(ToAction(0, 1), Player::kWhite));
  CHECK(board.stone_count() == 1);
  CHECK(board.GetCell(ToAction(0, 1)) == Cell::kBlack);
  CHECK(board.GetCell(-1) == Cell::kEmpty);
  CHECK(board.GetCell(kActionCount) == Cell::kEmpty);

  const int directions[][2] = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};
  for (const auto& direction : directions) {
    board.Reset();
    const int start_col = direction[1] < 0 ? kBoardSize - 1 : 0;
    for (int i = 0; i < 6; ++i) {
      const int row = i * direction[0];
      const int col = start_col + i * direction[1];
      const Action action = ToAction(row, col);
      CHECK(board.PlaceStone(action, Player::kWhite));
      CHECK(board.CheckWin(action, Player::kWhite) == (i >= 4));
      CHECK(!board.CheckWin(action, Player::kBlack));
    }
    // An empty origin must never count as a fifth stone.
    CHECK(!board.CheckWin(
        ToAction(6 * direction[0], start_col + 6 * direction[1]),
        Player::kWhite));
    CHECK(!board.CheckWin(-1, Player::kWhite));
    CHECK(!board.CheckWin(kActionCount, Player::kWhite));
  }
  board.Reset();
  for (Action action = 0; action < kActionCount; ++action)
    CHECK(board.PlaceStone(action, Player::kBlack));
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
  CHECK(!game.TryPlaceStone(-1, 0));
  CHECK(!game.TryPlaceStone(0, kBoardSize));
  CHECK(game.record().size() == 0);
  CHECK(game.TryPlaceStone(0, 1));
  CHECK(!game.TryPlaceStone(0, 1));
  CHECK(game.TryPlaceStone(14, 12));
  const auto before = PolicyValueNetwork::EncodeFeatures(game.position());
  CHECK(game.TryPlaceStone(4, 6));
  CHECK(game.TryPlaceStone(9, 8));
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
    CHECK(game.TryPlaceStone(7, col));
    CHECK(game.TryPlaceStone(0, col));
  }
  CHECK(game.TryPlaceStone(7, 4));
  CHECK(game.state() == Game::State::kWin);
  CHECK(game.winner() == Player::kBlack);
  CHECK(!game.TryPlaceStone(10, 10));
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
  CHECK(game.TryPlaceStone(7, 7));  // Black plays; now White's turn.
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
  CHECK(!timeout_game.TryPlaceStone(0, 0));
  CHECK(timeout_game.state() == Game::State::kTimeout);
  CHECK(timeout_game.winner() == Player::kWhite);
  CHECK(timeout_game.board().stone_count() == 0);
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
    CHECK(game.TryPlaceStone(ActionRow(*action), ActionCol(*action)));
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
      {"save/load validation", FileValidation},
      {"search priors and options", SearchPriorsAndOptions},
      {"search value perspective", SearchValuePerspective},
      {"tree lifecycle and AI", TreeLifecycleAndAI},
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
