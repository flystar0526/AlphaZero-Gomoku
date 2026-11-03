#include "configuration.h"

#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

Configuration::Configuration() {
  std::ifstream file(kConfigPath);
  if (!file.is_open()) return;

  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') continue;
    auto eq = line.find('=');
    if (eq == std::string::npos) continue;
    std::string key = line.substr(0, eq);
    std::string val = line.substr(eq + 1);
    // Trim whitespace.
    auto trim = [](std::string& s) {
      while (!s.empty() && (s.front() == ' ' || s.front() == '\t'))
        s.erase(s.begin());
      while (!s.empty() &&
             (s.back() == ' ' || s.back() == '\t' || s.back() == '\r'))
        s.pop_back();
    };
    trim(key);
    trim(val);
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
      // Parse per-color AI search options; invalid values fall back to defaults.
      for (Player player : {Player::kBlack, Player::kWhite}) {
        const std::string prefix =
            player == Player::kBlack ? "black_ai_" : "white_ai_";
        auto& options = ai_options_[static_cast<size_t>(player)];
        try {
          size_t parsed = 0;
          if (key == prefix + "num_playouts") {
            options.num_playouts = SearchOptions{}.num_playouts;
            const int value = std::stoi(val, &parsed);
            if (parsed == val.size() && value > 0) {
              options.num_playouts = value;
            }
          } else if (key == prefix + "c_puct") {
            options.c_puct = SearchOptions{}.c_puct;
            const float value = std::stof(val, &parsed);
            SearchOptions candidate = options;
            candidate.c_puct = value;
            if (parsed == val.size() && candidate.IsValid()) {
              options.c_puct = value;
            }
          }
        } catch (const std::exception&) {
        }
      }
    }
  }
}

void Configuration::Save() const {
  std::ofstream file(kConfigPath);
  if (!file.is_open()) return;
  file << "# Gomoku configuration\n";
  file << "undo_enabled = " << (undo_enabled_ ? "true" : "false") << "\n";
  file << "time_limit_enabled = " << (time_limit_enabled_ ? "true" : "false")
       << "\n";
  file << "time_limit_seconds = " << time_limit_seconds_ << "\n";
  file << std::setprecision(std::numeric_limits<float>::max_digits10);
  for (Player player : {Player::kBlack, Player::kWhite}) {
    const char* prefix = player == Player::kBlack ? "black_ai_" : "white_ai_";
    const auto& options = ai_options(player);
    file << prefix << "num_playouts = " << options.num_playouts << "\n";
    file << prefix << "c_puct = " << options.c_puct << "\n";
  }
}

bool Configuration::undo_enabled() const { return undo_enabled_; }
bool Configuration::time_limit_enabled() const { return time_limit_enabled_; }
int Configuration::time_limit_seconds() const { return time_limit_seconds_; }
const SearchOptions& Configuration::ai_options(Player player) const {
  return ai_options_[static_cast<size_t>(player)];
}
void Configuration::set_undo_enabled(bool value) { undo_enabled_ = value; }
void Configuration::set_time_limit_enabled(bool value) {
  time_limit_enabled_ = value;
}
void Configuration::set_time_limit_seconds(int seconds) {
  time_limit_seconds_ = seconds;
}
void Configuration::set_ai_options(Player player, SearchOptions options) {
  if (!options.IsValid()) {
    throw std::invalid_argument(
        "AI settings require positive c_puct and num_playouts");
  }
  ai_options_[static_cast<size_t>(player)] = options;
}
