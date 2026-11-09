#pragma once

#include <filesystem>
#include <memory>
#include <ostream>
#include <string>

#include "configuration.h"
#include "policy_value_network.h"

// Options for a headless AI vs AI simulation run.
struct SimulationOptions {
  int num_games = 1;
  std::filesystem::path log_root = "logs";
  std::string model_path = "model/alpha-zero.pt";
};

// Runs `options.num_games` AI vs AI games, alternating colors and writing logs
// to a new subdirectory under `options.log_root`.
std::filesystem::path RunSimulation(
    const SimulationOptions& options, const Configuration& config,
    std::shared_ptr<PolicyValueEvaluator> evaluator,
    std::ostream* progress = nullptr);
