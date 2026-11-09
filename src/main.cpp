#include <charconv>
#include <exception>
#include <iostream>
#include <memory>
#include <optional>
#include <string_view>
#include <system_error>

#include "app.h"
#include "configuration.h"
#include "policy_value_network.h"
#include "simulation.h"

namespace {

void PrintUsage(std::ostream& out) {
  out << "Usage: gomoku [--simulation <n>]\n"
         "  --simulation <n>  Play n AI games without a GUI, alternating "
         "colors.\n"
         "                    n must be a positive integer. Logs: logs/\n";
}

// Parses a positive integer game count.
std::optional<int> ParseGameCount(std::string_view text) {
  const char* end = text.data() + text.size();
  int value = 0;
  const auto [ptr, error] = std::from_chars(text.data(), end, value);
  if (error != std::errc{} || ptr != end || value <= 0) return std::nullopt;
  return value;
}

// Runs headless AI vs AI simulation using settings from gomoku.cfg.
int RunHeadless(int num_games) {
  try {
    SimulationOptions options;
    options.num_games = num_games;
    Configuration config;

    auto evaluator = std::make_shared<PolicyValueNetwork>(options.model_path);
    const auto directory =
        RunSimulation(options, config, evaluator, &std::cout);

    std::cout << "Simulation completed. Logs: " << directory.string() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 1) {
    App app;
    app.Run();
    return 0;
  }

  if (argc == 2 && std::string_view(argv[1]) == "--help") {
    PrintUsage(std::cout);
    return 0;
  }

  std::optional<int> num_games;
  if (argc == 3 && std::string_view(argv[1]) == "--simulation") {
    num_games = ParseGameCount(argv[2]);
  }
  if (!num_games) {
    PrintUsage(std::cerr);
    return 1;
  }

  return RunHeadless(*num_games);
}
