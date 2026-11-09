#include "policy_value_network.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <c10/core/InferenceMode.h>
#include <torch/cuda.h>
#include <torch/script.h>

namespace {

constexpr float kPolicySumTolerance = 1e-3f;

bool IsValidEvaluation(const Evaluation& evaluation) {
  float policy_sum = 0.f;
  for (float prob : evaluation.policy) {
    if (!std::isfinite(prob) || prob < 0.f) return false;
    policy_sum += prob;
  }

  return std::isfinite(policy_sum) &&
         std::abs(policy_sum - 1.f) <= kPolicySumTolerance &&
         std::isfinite(evaluation.value) && std::abs(evaluation.value) <= 1.f;
}

}  // namespace

struct PolicyValueNetwork::Impl {
  torch::jit::Module module;
  torch::Device device{torch::kCUDA};

  // Encodes `positions` into one [B,3,15,15] tensor and runs the model.
  c10::IValue Forward(const std::vector<GamePosition>& positions) {
    const int64_t batch_size = static_cast<int64_t>(positions.size());
    std::vector<float> features(positions.size() * kChannels * kActionCount);
    for (size_t i = 0; i < positions.size(); ++i) {
      const Features encoded = EncodeFeatures(positions[i]);
      std::copy(encoded.begin(), encoded.end(),
                features.begin() + i * encoded.size());
    }

    const torch::Tensor input =
        torch::from_blob(features.data(),
                         {batch_size, kChannels, kBoardSize, kBoardSize},
                         torch::TensorOptions().dtype(torch::kFloat32))
            .to(device);
    return module.forward({input});
  }

  // Converts raw (log_policy, value) tensors into one Evaluation per row.
  static std::vector<Evaluation> ToEvaluations(const c10::IValue& output) {
    const auto& elements = output.toTuple()->elements();
    const torch::Tensor policy_cpu =
        elements[0].toTensor().exp().to(torch::kCPU).contiguous();
    const torch::Tensor value_cpu =
        elements[1].toTensor().to(torch::kCPU).contiguous();

    const int64_t batch_size = policy_cpu.size(0);
    const float* policy = policy_cpu.data_ptr<float>();
    const float* value = value_cpu.data_ptr<float>();

    std::vector<Evaluation> evaluations(static_cast<size_t>(batch_size));
    for (int64_t i = 0; i < batch_size; ++i) {
      std::copy_n(policy + i * kActionCount, kActionCount,
                  evaluations[i].policy.begin());
      evaluations[i].value = value[i];
    }
    return evaluations;
  }
};

PolicyValueNetwork::PolicyValueNetwork(const std::string& model_path,
                                       Device device)
    : impl_(std::make_unique<Impl>()) {
  if (device == Device::kCuda) {
    if (!torch::cuda::is_available()) {
      throw std::runtime_error(
          "AI requires a CUDA-capable NVIDIA GPU and a compatible driver");
    }
    impl_->device = torch::Device(torch::kCUDA);
  } else {
    impl_->device = torch::Device(torch::kCPU);
  }

  try {
    impl_->module = torch::jit::load(model_path, impl_->device);
    impl_->module.eval();
    ValidateModel();
  } catch (const std::exception& e) {
    throw std::runtime_error("Failed to initialize AI model '" + model_path +
                             "': " + e.what());
  }
}

PolicyValueNetwork::~PolicyValueNetwork() = default;

// Runs a warm-up pass and verifies model output shapes and value ranges.
void PolicyValueNetwork::ValidateModel() {
  c10::InferenceMode inference_mode;

  const c10::IValue output = impl_->Forward({GamePosition()});
  if (!output.isTuple() || output.toTuple()->elements().size() != 2) {
    throw std::runtime_error("AI model must return (log_policy, value)");
  }

  const auto& elements = output.toTuple()->elements();
  if (!elements[0].isTensor() || !elements[1].isTensor()) {
    throw std::runtime_error("AI model outputs must be tensors");
  }

  const torch::Tensor log_policy = elements[0].toTensor();
  const torch::Tensor value = elements[1].toTensor();
  if (log_policy.sizes() != c10::IntArrayRef({1, kActionCount}) ||
      value.sizes() != c10::IntArrayRef({1, 1}) ||
      log_policy.scalar_type() != torch::kFloat32 ||
      value.scalar_type() != torch::kFloat32) {
    throw std::runtime_error(
        "AI model requires float32 outputs shaped [1,225] and [1,1]");
  }

  if (!IsValidEvaluation(Impl::ToEvaluations(output).front())) {
    throw std::runtime_error(
        "AI model returned invalid probabilities or value");
  }
}

Evaluation PolicyValueNetwork::Evaluate(const GamePosition& position) {
  return EvaluateBatch({position}).front();
}

std::vector<Evaluation> PolicyValueNetwork::EvaluateBatch(
    const std::vector<GamePosition>& positions) {
  if (positions.empty()) return {};

  c10::InferenceMode inference_mode;
  std::vector<Evaluation> evaluations =
      Impl::ToEvaluations(impl_->Forward(positions));
  assert(evaluations.size() == positions.size());
  assert(std::all_of(evaluations.begin(), evaluations.end(),
                     IsValidEvaluation));
  return evaluations;
}
