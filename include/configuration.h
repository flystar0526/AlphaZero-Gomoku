#pragma once

#include <array>
#include <ostream>
#include <string>

#include "mcts.h"

// Identifies an AI player slot in configuration.
enum class AIPlayerId { kPlayer1, kPlayer2 };

// Writes AI search options in gomoku.cfg key-value format.
void WriteAIOptions(std::ostream& out, AIPlayerId player,
                    const SearchOptions& options);

// Loads and saves user-configurable game and AI settings (gomoku.cfg).
class Configuration {
 public:
  static constexpr int kDefaultTimeLimitSeconds = 60;

  Configuration();

  void Save() const;

  bool undo_enabled() const;
  bool time_limit_enabled() const;
  int time_limit_seconds() const;
  const SearchOptions& ai_options(AIPlayerId player) const;

  void set_undo_enabled(bool value);
  void set_time_limit_enabled(bool value);
  void set_time_limit_seconds(int seconds);
  void set_ai_options(AIPlayerId player, SearchOptions options);

 private:
  static constexpr const char* kConfigPath = "gomoku.cfg";

  bool undo_enabled_ = false;
  bool time_limit_enabled_ = false;
  int time_limit_seconds_ = kDefaultTimeLimitSeconds;
  std::array<SearchOptions, 2> ai_options_{};
};
