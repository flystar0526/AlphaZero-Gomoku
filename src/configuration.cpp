#include "configuration.h"

#include <cassert>
#include <cerrno>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <limits>
#include <optional>
#include <string>
#include <string_view>

#include "key_value.h"

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

namespace {

constexpr std::string_view ConfigPrefix(AIPlayerId player) {
  return player == AIPlayerId::kPlayer1 ? "ai_player_1_" : "ai_player_2_";
}

// Visits each SearchOptions field with its config key name.
template <typename Visitor>
void ForEachSearchOption(Visitor&& visit) {
  visit("num_playouts", &SearchOptions::num_playouts);
  visit("c_puct", &SearchOptions::c_puct);
  visit("temperature", &SearchOptions::temperature);
  visit("num_sampling_moves", &SearchOptions::num_sampling_moves);
  visit("search_limit", &SearchOptions::search_limit);
  visit("think_time_ms", &SearchOptions::think_time_ms);
  visit("parallel_mode", &SearchOptions::parallel_mode);
  visit("virtual_loss", &SearchOptions::virtual_loss);
  visit("num_threads", &SearchOptions::num_threads);
  visit("batch_size", &SearchOptions::batch_size);
  visit("batch_timeout_us", &SearchOptions::batch_timeout_us);
}

template <typename Enum, size_t N>
std::optional<Enum> ParseEnum(const std::string& text,
                              const char* const (&names)[N]) {
  for (size_t i = 0; i < N; ++i) {
    if (names[i] == text) return static_cast<Enum>(i);
  }
  return std::nullopt;
}

std::optional<SearchLimit> ParseValue(const std::string& text, SearchLimit*) {
  return ParseEnum<SearchLimit>(text, kSearchLimitNames);
}

std::optional<ParallelMode> ParseValue(const std::string& text,
                                       ParallelMode*) {
  return ParseEnum<ParallelMode>(text, kParallelModeNames);
}

std::optional<int> ParseValue(const std::string& text, int*) {
  try {
    size_t parsed = 0;
    const int value = std::stoi(text, &parsed);
    if (parsed == text.size()) return value;
  } catch (const std::exception&) {
  }
  return std::nullopt;
}

std::optional<float> ParseValue(const std::string& text, float*) {
  char* end = nullptr;
  errno = 0;
  const float value = std::strtof(text.c_str(), &end);
  if (end != text.c_str() && end == text.c_str() + text.size() &&
      (errno != ERANGE || value != 0.0f)) {
    return value;
  }
  return std::nullopt;
}

void WriteValue(std::ostream& out, int value) { out << value; }

void WriteValue(std::ostream& out, float value) { out << value; }

void WriteValue(std::ostream& out, SearchLimit value) {
  out << kSearchLimitNames[static_cast<size_t>(value)];
}

void WriteValue(std::ostream& out, ParallelMode value) {
  out << kParallelModeNames[static_cast<size_t>(value)];
}

// Parses a single AI search option, falling back to default on parse failure.
void ParseSearchOption(std::string_view name, const std::string& text,
                       SearchOptions& options) {
  ForEachSearchOption([&](std::string_view option_name, auto member) {
    if (option_name != name) return;
    auto& field = options.*member;
    const auto value = ParseValue(text, &field);
    field = value ? *value : SearchOptions{}.*member;
  });
}

}  // namespace

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

void WriteAIOptions(std::ostream& out, AIPlayerId player,
                    const SearchOptions& options) {
  const std::ios_base::fmtflags flags = out.flags();
  const std::streamsize precision = out.precision();
  out.unsetf(std::ios_base::floatfield);
  out.precision(std::numeric_limits<float>::max_digits10);

  ForEachSearchOption([&](std::string_view name, auto member) {
    out << ConfigPrefix(player) << name << " = ";
    WriteValue(out, options.*member);
    out << '\n';
  });

  out.flags(flags);
  out.precision(precision);
}

Configuration::Configuration() {
  std::ifstream file(kConfigPath);
  if (!file.is_open()) return;

  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') continue;

    std::string key, val;
    if (!SplitKeyValue(line, key, val)) continue;

    if (key == "undo_enabled") {
      undo_enabled_ = (val == "true");
    } else if (key == "time_limit_enabled") {
      time_limit_enabled_ = (val == "true");
    } else if (key == "time_limit_seconds") {
      try {
        time_limit_seconds_ = std::stoi(val);
      } catch (...) {
      }
    } else {
      for (AIPlayerId player : {AIPlayerId::kPlayer1, AIPlayerId::kPlayer2}) {
        const std::string_view prefix = ConfigPrefix(player);
        if (std::string_view(key).substr(0, prefix.size()) != prefix) continue;
        ParseSearchOption(std::string_view(key).substr(prefix.size()), val,
                          ai_options_[static_cast<size_t>(player)]);
      }
    }
  }

  for (SearchOptions& options : ai_options_) options = options.Sanitized();
}

void Configuration::Save() const {
  std::ofstream file(kConfigPath);
  if (!file.is_open()) return;

  file << "# Gomoku configuration\n";
  file << "undo_enabled = " << (undo_enabled_ ? "true" : "false") << "\n";
  file << "time_limit_enabled = " << (time_limit_enabled_ ? "true" : "false")
       << "\n";
  file << "time_limit_seconds = " << time_limit_seconds_ << "\n";

  for (AIPlayerId player : {AIPlayerId::kPlayer1, AIPlayerId::kPlayer2}) {
    WriteAIOptions(file, player, ai_options(player));
  }
}

bool Configuration::undo_enabled() const { return undo_enabled_; }

bool Configuration::time_limit_enabled() const { return time_limit_enabled_; }

int Configuration::time_limit_seconds() const { return time_limit_seconds_; }

const SearchOptions& Configuration::ai_options(AIPlayerId player) const {
  return ai_options_[static_cast<size_t>(player)];
}

void Configuration::set_undo_enabled(bool value) { undo_enabled_ = value; }

void Configuration::set_time_limit_enabled(bool value) {
  time_limit_enabled_ = value;
}

void Configuration::set_time_limit_seconds(int seconds) {
  time_limit_seconds_ = seconds;
}

void Configuration::set_ai_options(AIPlayerId player, SearchOptions options) {
  assert(options.IsValid());
  ai_options_[static_cast<size_t>(player)] = options;
}
