#pragma once

#include <filesystem>
#include <memory>
#include <ostream>
#include <string>

#include "configuration.h"
#include "policy_value_network.h"

// Settings for a headless AI vs AI run. AI search settings come from
// Configuration, not from here.
struct SimulationOptions {
  int num_games = 1;
  std::filesystem::path log_root = "logs";  // Parent of each run directory.
  std::string model_path = "model/alpha-zero.pt";  // Recorded in summary.txt.
};

// Plays num_games synchronously, alternating colors while retaining each AI's
// settings. Returns the new log directory; throws on invalid input or failure.
// Failed runs preserve logs and mark the summary as failed when writable.
std::filesystem::path RunSimulation(
    const SimulationOptions& options, const Configuration& config,
    std::shared_ptr<PolicyValueEvaluator> evaluator,
    std::ostream* progress = nullptr);
