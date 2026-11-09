#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "game_position.h"

// Move priors and position value returned by an evaluator.
struct Evaluation {
  std::array<float, kActionCount> policy{};
  float value = 0.f;
};

// Interface for evaluating board positions.
class PolicyValueEvaluator {
 public:
  PolicyValueEvaluator() = default;
  PolicyValueEvaluator(const PolicyValueEvaluator&) = delete;
  PolicyValueEvaluator& operator=(const PolicyValueEvaluator&) = delete;
  virtual ~PolicyValueEvaluator() = default;

  virtual Evaluation Evaluate(const GamePosition& position) = 0;

  // Evaluates `positions` in order. The default evaluates them one at a time.
  virtual std::vector<Evaluation> EvaluateBatch(
      const std::vector<GamePosition>& positions) {
    std::vector<Evaluation> evaluations;
    evaluations.reserve(positions.size());
    for (const GamePosition& position : positions) {
      evaluations.push_back(Evaluate(position));
    }
    return evaluations;
  }
};

// TorchScript neural network evaluator for policy and value prediction.
class PolicyValueNetwork final : public PolicyValueEvaluator {
 public:
  enum class Device { kCuda, kCpu };

  static constexpr int kOwnChannel = 0;
  static constexpr int kOpponentChannel = 1;
  static constexpr int kLastMoveChannel = 2;
  static constexpr int kChannels = 3;

  using Features = std::array<float, kChannels * kActionCount>;

  // Encodes `position` into a 3-channel feature tensor (own, opponent, last
  // move).
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

  explicit PolicyValueNetwork(const std::string& model_path,
                              Device device = Device::kCuda);
  ~PolicyValueNetwork() override;

  Evaluation Evaluate(const GamePosition& position) override;

  // Runs a single forward pass over all `positions` as one [B,3,15,15] batch.
  std::vector<Evaluation> EvaluateBatch(
      const std::vector<GamePosition>& positions) override;

 private:
  struct Impl;

  void ValidateModel();

  std::unique_ptr<Impl> impl_;
};
