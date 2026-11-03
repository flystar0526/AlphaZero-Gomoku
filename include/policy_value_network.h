#pragma once

#include <array>
#include <memory>
#include <string>

#include "game_position.h"

// Prior move probabilities and position value returned by the evaluator.
struct Evaluation {
  std::array<float, kActionCount> policy{};
  float value = 0.f;  // Expected outcome for the side to move, in [-1, 1].
};

// Abstract interface for evaluating board positions (enables mocking in tests).
class PolicyValueEvaluator {
 public:
  PolicyValueEvaluator() = default;
  PolicyValueEvaluator(const PolicyValueEvaluator&) = delete;
  PolicyValueEvaluator& operator=(const PolicyValueEvaluator&) = delete;
  virtual ~PolicyValueEvaluator() = default;

  virtual Evaluation Evaluate(const GamePosition& position) = 0;
};

// Evaluates game positions using a TorchScript policy-value neural network.
class PolicyValueNetwork final : public PolicyValueEvaluator {
 public:
  enum class Device { kCuda, kCpu };

  static constexpr int kOwnChannel = 0;
  static constexpr int kOpponentChannel = 1;
  static constexpr int kLastMoveChannel = 2;
  static constexpr int kChannels = 3;

  using Features = std::array<float, kChannels * kActionCount>;

  // Encodes the position into a 3-channel CHW tensor (own, opponent, last move).
  static Features EncodeFeatures(const GamePosition& position) {
    Features features{};
    const Cell own_cell = ToCell(position.current_player());
    for (Action action = 0; action < kActionCount; ++action) {
      const Cell cell = position.board().GetCell(action);
      if (cell != Cell::kEmpty) {
        const int channel = (cell == own_cell) ? kOwnChannel : kOpponentChannel;
        features[channel * kActionCount + action] = 1.f;
      }
    }
    if (position.last_move().has_value()) {
      features[kLastMoveChannel * kActionCount + *position.last_move()] = 1.f;
    }
    return features;
  }

  // Loads and validates the TorchScript model on the specified device.
  explicit PolicyValueNetwork(const std::string& model_path,
                              Device device = Device::kCuda);
  ~PolicyValueNetwork() override;

  // Runs neural network inference to predict move probabilities and value.
  Evaluation Evaluate(const GamePosition& position) override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
