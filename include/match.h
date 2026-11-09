#pragma once

#include <array>
#include <memory>
#include <optional>

#include "ai_player.h"
#include "configuration.h"
#include "game.h"

// AI settings seated at each color, indexed by Player; std::nullopt is a human.
using Seats = std::array<std::optional<AIPlayerId>, 2>;

// Seats for `mode`: Player vs AI seats AI Player 1 as White; AI vs AI seats
// AI Player 1 as Black and AI Player 2 as White.
Seats DefaultSeats(Game::Mode mode);

// A game plus the AI players seated in it. Every move goes through Match so
// each AI's search tree follows the board; undo and restart reset the trees.
// Not thread-safe, except that Think() may run on a worker thread while other
// threads only read game().
class Match {
 public:
  // `seats` must place AIs exactly where `mode` has AI players. Throws
  // std::invalid_argument if an AI is seated and `evaluator` is null.
  Match(const Configuration& config, Game::Mode mode, Seats seats,
        std::shared_ptr<PolicyValueEvaluator> evaluator);
  Match(const Configuration& config, Game::Mode mode,
        std::shared_ptr<PolicyValueEvaluator> evaluator = nullptr);

  // Plays `action` for the side to move and syncs every AI's search tree.
  // Returns false if the move is illegal or the game is not in progress.
  bool Play(Action action);

  // Searches for the move of the AI to play without changing the game.
  // Requires IsAITurn(). Returns std::nullopt if the position is terminal.
  std::optional<Action> Think(SearchTiming* timing = nullptr);

  // Think() then Play(). Throws std::runtime_error if the AI has no legal move.
  Action PlayAIMove(SearchTiming* timing = nullptr);

  // Returns true if the side to move is an AI.
  bool IsAITurn() const;
  // Returns the AI settings seated at `color`, or std::nullopt for a human.
  std::optional<AIPlayerId> seat(Player color) const;

  // Game controls; Undo and Restart also reset every AI's search tree.
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
  // AI player at each color, indexed by Player; null for a human.
  std::array<std::unique_ptr<AIPlayer>, 2> players_;
};
