#pragma once

#include <optional>
#include <string>
#include <vector>

#include "game_record.h"

// Saves, loads, and lists .gom game record files.
class FileManager {
 public:
  FileManager() = delete;

  static bool Save(const GameRecord& record, const std::string& path);

  static std::optional<GameRecord> Load(const std::string& path);

  // Lists .gom files in `directory` sorted by path, or std::nullopt on error.
  static std::optional<std::vector<std::string>> ListGameFiles(
      const std::string& directory);
};
