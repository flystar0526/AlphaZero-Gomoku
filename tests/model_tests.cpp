#include <ATen/Parallel.h>
#include <torch/cuda.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#include "policy_value_network.h"

namespace {

void Validate(const Evaluation& result) {
  float sum = 0.f;
  for (float probability : result.policy) {
    if (!std::isfinite(probability) || probability < 0.f)
      throw std::runtime_error("Invalid policy probability");
    sum += probability;
  }
  if (std::abs(sum - 1.f) > 1e-4f || !std::isfinite(result.value) ||
      std::abs(result.value) > 1.f)
    throw std::runtime_error("Invalid policy sum or value");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) return 1;
  const bool cuda = argc > 2 && std::string(argv[2]) == "--cuda";
  if (cuda && !torch::cuda::is_available()) {
    // Verify the default CUDA backend reports its GPU requirement cleanly.
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

    // Exercise the exact same PolicyValueNetwork implementation on CPU or CUDA.
    const auto device = cuda ? PolicyValueNetwork::Device::kCuda
                             : PolicyValueNetwork::Device::kCpu;
    PolicyValueNetwork network(argv[1], device);
    const auto first = network.Evaluate(position);
    const auto second = network.Evaluate(position);
    Validate(first);
    Validate(second);
    for (int i = 0; i < kActionCount; ++i)
      if (std::abs(first.policy[i] - second.policy[i]) > 1e-5f)
        throw std::runtime_error("Inference changed between identical inputs");

    std::cout << (cuda ? "CUDA inference passed\n" : "CPU checkpoint contract passed\n");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
