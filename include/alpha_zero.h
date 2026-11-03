#pragma once

#include <string>
#include <utility>
#include <vector>

#include <torch/script.h>
#include <torch/torch.h>

class AlphaZero {
 public:
  static constexpr int kHeight   = 15;
  static constexpr int kWidth    = 15;
  static constexpr int kChannels = 9;

  // Loads the TorchScript model on CUDA.
  // Throws std::runtime_error if CUDA is unavailable or loading fails.
  explicit AlphaZero(const std::string& model_path);

  // Input: flat vector of size kChannels * kHeight * kWidth = 2025 floats.
  // Output: { action_probs (kHeight*kWidth), value_scalar }
  std::pair<std::vector<float>, float> Infer(
      const std::vector<float>& state) const;

 private:
  mutable torch::jit::script::Module module_;
  torch::Device device_;
};
