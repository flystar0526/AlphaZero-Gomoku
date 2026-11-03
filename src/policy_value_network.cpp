#include "policy_value_network.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <c10/core/InferenceMode.h>
#include <torch/cuda.h>
#include <torch/script.h>

namespace {

constexpr float kPolicySumTolerance = 1e-3f;

}  // namespace

struct PolicyValueNetwork::Impl {
  torch::jit::Module module;
  torch::Device device{torch::kCUDA};
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
    // Run a warm-up evaluation to verify model compatibility at startup.
    Evaluate(GamePosition());
  } catch (const std::exception& e) {
    throw std::runtime_error("Failed to initialize AI model '" + model_path +
                             "': " + e.what());
  }
}

PolicyValueNetwork::~PolicyValueNetwork() = default;

Evaluation PolicyValueNetwork::Evaluate(const GamePosition& position) {
  c10::InferenceMode inference_mode;

  // Prepare input tensor [1, kChannels, kBoardSize, kBoardSize] on target device.
  Features features = EncodeFeatures(position);
  const torch::Tensor input_tensor =
      torch::from_blob(features.data(),
                       {1, kChannels, kBoardSize, kBoardSize},
                       torch::TensorOptions().dtype(torch::kFloat32))
          .to(impl_->device);

  // Run forward pass and validate output tuple structure and shapes.
  const c10::IValue output = impl_->module.forward({input_tensor});
  if (!output.isTuple() || output.toTuple()->elements().size() != 2) {
    throw std::runtime_error("AI model must return (log_policy, value)");
  }
  const auto& elements = output.toTuple()->elements();
  if (!elements[0].isTensor() || !elements[1].isTensor()) {
    throw std::runtime_error("AI model outputs must be tensors");
  }

  const torch::Tensor log_policy_tensor = elements[0].toTensor();
  const torch::Tensor value_tensor = elements[1].toTensor();
  if (log_policy_tensor.sizes() != c10::IntArrayRef({1, kActionCount}) ||
      value_tensor.sizes() != c10::IntArrayRef({1, 1}) ||
      log_policy_tensor.scalar_type() != torch::kFloat32 ||
      value_tensor.scalar_type() != torch::kFloat32) {
    throw std::runtime_error(
        "AI model requires float32 outputs shaped [1,225] and [1,1]");
  }

  // Convert log-probabilities to probabilities and copy results to CPU.
  const torch::Tensor policy_cpu =
      log_policy_tensor.exp().to(torch::kCPU).contiguous();
  Evaluation evaluation;
  std::copy_n(policy_cpu.data_ptr<float>(), kActionCount,
              evaluation.policy.begin());
  evaluation.value = value_tensor.item<float>();

  // Verify that outputs form a valid probability distribution and bounded value.
  float policy_sum = 0.f;
  for (float prob : evaluation.policy) {
    if (!std::isfinite(prob) || prob < 0.f) {
      throw std::runtime_error(
          "AI model returned non-finite or negative policy");
    }
    policy_sum += prob;
  }
  if (!std::isfinite(policy_sum) ||
      std::abs(policy_sum - 1.f) > kPolicySumTolerance ||
      !std::isfinite(evaluation.value) || std::abs(evaluation.value) > 1.f) {
    throw std::runtime_error(
        "AI model returned invalid probabilities or value");
  }

  return evaluation;
}
