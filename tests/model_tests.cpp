#include <ATen/Parallel.h>
#include <torch/cuda.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "policy_value_network.h"
#include "simulation.h"
#include "test_support.h"

namespace {

void Validate(const Evaluation& result) {
  float sum = 0.f;
  for (float probability : result.policy) {
    if (!std::isfinite(probability) || probability < 0.f) {
      throw std::runtime_error("Invalid policy probability");
    }
    sum += probability;
  }

  if (std::abs(sum - 1.f) > 1e-4f || !std::isfinite(result.value) ||
      std::abs(result.value) > 1.f) {
    throw std::runtime_error("Invalid policy sum or value");
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) return 1;

  const bool cuda = argc > 2 && std::string(argv[2]) == "--cuda";
  const bool simulation = argc > 2 && std::string(argv[2]) == "--simulation";

  if (cuda && !torch::cuda::is_available()) {
    try {
      PolicyValueNetwork network(argv[1]);
      std::cerr << "CUDA-only network unexpectedly initialized\n";
      return 1;
    } catch (const std::runtime_error& error) {
      if (std::string(error.what()).find("CUDA") == std::string::npos) return 1;
      std::cout << error.what() << "\nSKIP CUDA inference: no available GPU\n";
      return 77;
    }
  }

  try {
    at::set_num_threads(2);

    GamePosition position;
    position.PlayMove(ToAction(0, 1));
    position.PlayMove(ToAction(14, 12));

    const auto device = cuda ? PolicyValueNetwork::Device::kCuda
                             : PolicyValueNetwork::Device::kCpu;
    auto network = std::make_shared<PolicyValueNetwork>(argv[1], device);

    const auto first = network->Evaluate(position);
    const auto second = network->Evaluate(position);
    Validate(first);
    Validate(second);

    for (int i = 0; i < kActionCount; ++i) {
      if (std::abs(first.policy[i] - second.policy[i]) > 1e-5f) {
        throw std::runtime_error("Inference changed between identical inputs");
      }
    }

    // A batched forward pass must match per-position inference row by row.
    // CUDA may pick different kernels per batch size, so allow float noise.
    GamePosition other = position;
    other.PlayMove(ToAction(7, 7));
    const std::vector<GamePosition> batch = {position, GamePosition(), other};
    const auto batched = network->EvaluateBatch(batch);
    if (batched.size() != batch.size()) {
      throw std::runtime_error("Batch inference returned the wrong row count");
    }
    constexpr float kBatchTolerance = 1e-3f;
    for (size_t row = 0; row < batch.size(); ++row) {
      const auto single = network->Evaluate(batch[row]);
      Validate(batched[row]);
      bool matches =
          std::abs(batched[row].value - single.value) <= kBatchTolerance;
      for (int i = 0; i < kActionCount; ++i) {
        matches = matches && std::abs(batched[row].policy[i] -
                                      single.policy[i]) <= kBatchTolerance;
      }
      if (!matches) {
        throw std::runtime_error(
            "Batch inference differs from single inference");
      }
    }
    if (!network->EvaluateBatch({}).empty()) {
      throw std::runtime_error("Empty batch must return no evaluations");
    }

    if (simulation) {
      test::TempDirectory logs("gomoku-model-simulation");
      Configuration config;
      for (AIPlayerId player : {AIPlayerId::kPlayer1, AIPlayerId::kPlayer2}) {
        config.set_ai_options(player, {5.0f, 2, 0.0f, 0});
      }

      SimulationOptions options;
      options.num_games = 2;
      options.log_root = logs.path;
      options.model_path = argv[1];

      RunSimulation(options, config, network, &std::cout);
      std::cout << "CPU model simulation passed\n";
    } else {
      std::cout << (cuda ? "CUDA inference passed\n"
                         : "CPU checkpoint contract passed\n");
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
