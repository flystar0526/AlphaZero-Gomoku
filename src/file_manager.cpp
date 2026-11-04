#include "file_manager.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

namespace {

std::string PlayerToString(Player p) {
  return (p == Player::kBlack) ? "black" : "white";
}

std::optional<Player> ParsePlayer(const std::string& s) {
  if (s == "black") return Player::kBlack;
  if (s == "white") return Player::kWhite;
  return std::nullopt;
}

void TrimLine(std::string& s) {
  while (!s.empty() && (s.back() == '\r' || s.back() == '\n' ||
                        s.back() == ' ' || s.back() == '\t'))
    s.pop_back();
}

// Splits "key = value" into key and value strings.
bool SplitKeyValue(const std::string& line, std::string& key,
                   std::string& value) {
  auto eq = line.find('=');
  if (eq == std::string::npos) return false;
  key = line.substr(0, eq);
  value = line.substr(eq + 1);
  auto trim = [](std::string& s) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t'))
      s.erase(s.begin());
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) s.pop_back();
  };
  trim(key);
  trim(value);
  return true;
}

}  // namespace

bool FileManager::ListGameFiles(const std::string& directory,
                                std::vector<std::string>* paths) {
  if (!paths) return false;
  paths->clear();

  std::error_code error;
  std::filesystem::directory_iterator entry(directory, error);
  if (error) return error == std::errc::no_such_file_or_directory;

  std::vector<std::string> files;
  const std::filesystem::directory_iterator end;
  while (!error && entry != end) {
    if (entry->is_regular_file(error) && entry->path().extension() == ".gom") {
      files.push_back(entry->path().generic_string());
    }
    if (!error) entry.increment(error);
  }
  if (error) return false;

  std::sort(files.begin(), files.end());
  *paths = std::move(files);
  return true;
}

// ---------------------------------------------------------------------------
// FileManager::Save
// ---------------------------------------------------------------------------

bool FileManager::Save(const GameRecord& record, const std::string& path) {
  std::ofstream out(path);
  if (!out.is_open()) return false;

  const GameMetadata& meta = record.metadata();

  out << "# Gomoku Game Record\n";
  out << "# Format version 1\n\n";

  out << "[metadata]\n";
  out << "date = " << meta.date << "\n";
  out << "duration_sec = " << meta.duration_sec << "\n";
  out << "total_moves = " << meta.total_moves << "\n";
  out << "winner = "
      << (meta.winner.has_value() ? PlayerToString(*meta.winner) : "draw")
      << "\n";
  out << "undo_enabled = " << (meta.undo_enabled ? "true" : "false") << "\n";
  out << "time_limit_enabled = " << (meta.time_limit_enabled ? "true" : "false")
      << "\n";
  out << "time_limit_seconds = " << meta.time_limit_seconds << "\n";

  out << "\n[moves]\n";
  out << "# num  player  row  col\n";
  for (const auto& m : record.moves()) {
    out << m.move_number << "      " << PlayerToString(m.player) << "   "
        << m.row << "    " << m.col << "\n";
  }

  return out.good();
}

// ---------------------------------------------------------------------------
// FileManager::Load
// ---------------------------------------------------------------------------

bool FileManager::Load(const std::string& path, GameRecord* record) try {
  if (!record) return false;
  std::ifstream in(path);
  if (!in.is_open()) return false;

  enum class Section { kNone, kMetadata, kMoves };
  Section section = Section::kNone;

  GameMetadata meta;
  std::vector<MoveRecord> moves;
  bool metadata_complete = false;

  std::string line;
  while (std::getline(in, line)) {
    TrimLine(line);
    if (line.empty() || line[0] == '#') continue;

    if (line == "[metadata]") {
      section = Section::kMetadata;
      continue;
    }
    if (line == "[moves]") {
      section = Section::kMoves;
      metadata_complete = true;
      continue;
    }

    if (section == Section::kMetadata) {
      std::string key, value;
      if (!SplitKeyValue(line, key, value)) return false;
      if (key == "date") {
        meta.date = value;
      } else if (key == "duration_sec") {
        meta.duration_sec = std::stoi(value);
      } else if (key == "total_moves") {
        meta.total_moves = std::stoi(value);
      } else if (key == "winner") {
        if (value == "draw") {
          meta.winner = std::nullopt;
        } else {
          meta.winner = ParsePlayer(value);
          if (!meta.winner.has_value() && value != "draw") return false;
        }
      } else if (key == "undo_enabled") {
        meta.undo_enabled = (value == "true");
      } else if (key == "time_limit_enabled") {
        meta.time_limit_enabled = (value == "true");
      } else if (key == "time_limit_seconds") {
        meta.time_limit_seconds = std::stoi(value);
      }
    } else if (section == Section::kMoves) {
      std::istringstream ss(line);
      MoveRecord m;
      std::string player_str;
      if (!(ss >> m.move_number >> player_str >> m.row >> m.col)) return false;
      auto p = ParsePlayer(player_str);
      if (!p.has_value()) return false;
      m.player = *p;
      moves.push_back(m);
    }
  }

  if (!metadata_complete || meta.total_moves != static_cast<int>(moves.size()))
    return false;

  GameRecord loaded;
  loaded.Finalize(meta);
  for (const auto& m : moves) loaded.AddMove(m);
  *record = std::move(loaded);
  return true;
} catch (const std::invalid_argument&) {
  return false;
} catch (const std::out_of_range&) {
  return false;
}
