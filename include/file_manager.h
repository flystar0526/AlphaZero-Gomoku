#pragma once

#include <string>
#include <vector>

#include "game_record.h"

// Static utility class for .gom record storage and discovery. Not instantiable.
class FileManager {
 public:
  FileManager() = delete;

  // Writes record to path. Returns true on success.
  static bool Save(const GameRecord& record, const std::string& path);

  // Reads path into *record. Returns true on success.
  static bool Load(const std::string& path, GameRecord* record);

  // Lists .gom files in directory (non-recursive), sorted by filename.
  static bool ListGameFiles(const std::string& directory,
                            std::vector<std::string>* paths);
};
