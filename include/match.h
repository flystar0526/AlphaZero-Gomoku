#pragma once

#include <array>
#include <memory>
#include <optional>

#include "ai_player.h"
#include "configuration.h"
#include "game.h"

// Maps each stone color to an AI player slot, or std::nullopt for a human.
using Seats = std::array<std::optional<AIPlayerId>, 2>;

// Returns the default seat assignment for `mode`.
Seats DefaultSeats(Game::Mode mode);

// Coordinates a Game and its seated AI players, keeping AI search trees
// synchronized with board moves, undo, and restart.
class Match {
 public:
  Match(const Configuration& config, Game::Mode mode, Seats seats,
        std::shared_ptr<PolicyValueEvaluator> evaluator);
  Match(const Configuration& config, Game::Mode mode,
        std::shared_ptr<PolicyValueEvaluator> evaluator = nullptr);

  // Plays `action` and updates all AI search trees.
  bool Play(Action action);

  // Computes the current AI player's move without playing it.
  std::optional<Action> Think(SearchTiming* timing = nullptr);

  // Computes and plays the current AI player's move.
  Action PlayAIMove(SearchTiming* timing = nullptr);

  bool IsAITurn() const;
  std::optional<AIPlayerId> seat(Player color) const;

  void Update() { game_.Update(); }
  void Undo();
  void Forfeit() { game_.Forfeit(); }
  void Restart();
  GameRecord FinalizeRecord() { return game_.FinalizeRecord(); }

  const Game& game() const { return game_; }

 private:
  void ResetAIs();

  Game game_;
  Seats seats_;
  std::array<std::unique_ptr<AIPlayer>, 2> players_;
};
