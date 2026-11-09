#include "policy_value_network.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

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

  c10::IValue Forward(const GamePosition& position) {
    Features features = EncodeFeatures(position);
    const torch::Tensor input =
        torch::from_blob(features.data(), {1, kChannels, kBoardSize, kBoardSize},
                         torch::TensorOptions().dtype(torch::kFloat32))
            .to(device);
    return module.forward({input});
  }

  // Converts raw (log_policy, value) tensors into an Evaluation.
  static Evaluation ToEvaluation(const c10::IValue& output) {
    const auto& elements = output.toTuple()->elements();
    const torch::Tensor policy_cpu =
        elements[0].toTensor().exp().to(torch::kCPU).contiguous();

    Evaluation evaluation;
    std::copy_n(policy_cpu.data_ptr<float>(), kActionCount,
                evaluation.policy.begin());
    evaluation.value = elements[1].toTensor().item<float>();
    return evaluation;
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

  const c10::IValue output = impl_->Forward(GamePosition());
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

  if (!IsValidEvaluation(Impl::ToEvaluation(output))) {
    throw std::runtime_error(
        "AI model returned invalid probabilities or value");
  }
}

Evaluation PolicyValueNetwork::Evaluate(const GamePosition& position) {
  c10::InferenceMode inference_mode;
  const Evaluation evaluation = Impl::ToEvaluation(impl_->Forward(position));
  assert(IsValidEvaluation(evaluation));
  return evaluation;
}
