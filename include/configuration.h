#pragma once

#include <array>
#include <string>

#include "mcts.h"

// Holds all user-configurable game settings.
// Loaded from and persisted to gomoku.cfg on disk.
class Configuration {
 public:
  static constexpr int kDefaultTimeLimitSeconds = 60;

  // Loads settings from kConfigPath if the file exists; otherwise uses
  // defaults.
  Configuration();

  // Writes current settings to kConfigPath.
  void Save() const;

  bool undo_enabled() const;
  bool time_limit_enabled() const;
  int time_limit_seconds() const;
  const SearchOptions& ai_options(Player player) const;

  void set_undo_enabled(bool value);
  void set_time_limit_enabled(bool value);
  void set_time_limit_seconds(int seconds);
  // Sets MCTS options for `player`; throws std::invalid_argument if invalid.
  void set_ai_options(Player player, SearchOptions options);

 private:
  static constexpr const char* kConfigPath = "gomoku.cfg";

  bool undo_enabled_ = false;
  bool time_limit_enabled_ = false;
  int time_limit_seconds_ = kDefaultTimeLimitSeconds;
  std::array<SearchOptions, 2> ai_options_{};
};
