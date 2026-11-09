#pragma once

#include <optional>
#include <string>
#include <vector>

#include "game_record.h"

// Static utility class for .gom record storage and discovery. Not instantiable.
class FileManager {
 public:
  FileManager() = delete;

  // Writes record to path. Returns true on success.
  static bool Save(const GameRecord& record, const std::string& path);

  // Reads the record at path. Returns std::nullopt on failure.
  static std::optional<GameRecord> Load(const std::string& path);

  // Lists .gom files in directory (non-recursive), sorted by filename.
  // A missing directory yields an empty list; returns std::nullopt on error.
  static std::optional<std::vector<std::string>> ListGameFiles(
      const std::string& directory);
};
