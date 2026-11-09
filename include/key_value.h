#pragma once

#include <string>

// Helpers for the line-based "key = value" text formats (gomoku.cfg, .gom).

// Strips leading and trailing spaces, tabs, and line endings in place.
inline void TrimWhitespace(std::string& text) {
  constexpr const char* kWhitespace = " \t\r\n";
  const auto first = text.find_first_not_of(kWhitespace);
  if (first == std::string::npos) {
    text.clear();
    return;
  }
  const auto last = text.find_last_not_of(kWhitespace);
  text = text.substr(first, last - first + 1);
}

// Splits "key = value" into trimmed key and value. Returns false without '='.
inline bool SplitKeyValue(const std::string& line, std::string& key,
                          std::string& value) {
  const auto eq = line.find('=');
  if (eq == std::string::npos) return false;
  key = line.substr(0, eq);
  value = line.substr(eq + 1);
  TrimWhitespace(key);
  TrimWhitespace(value);
  return true;
}
