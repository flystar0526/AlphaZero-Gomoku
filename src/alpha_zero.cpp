#include "alpha_zero.h"

#include <stdexcept>

AlphaZero::AlphaZero(const std::string& model_path) : device_(torch::kCUDA) {
  if (!torch::cuda::is_available())
    throw std::runtime_error(
        "AlphaZero requires a CUDA-capable NVIDIA GPU and a compatible driver");

  try {
    module_ = torch::jit::load(model_path, device_);
  } catch (const c10::Error& e) {
    throw std::runtime_error("Failed to load AlphaZero model: " +
                             std::string(e.what()));
  }

  module_.to(device_);
  module_.eval();
}

std::pair<std::vector<float>, float> AlphaZero::Infer(
    const std::vector<float>& state) const {
  const int HW = kHeight * kWidth;

  auto opts = torch::TensorOptions().dtype(torch::kFloat32);
  torch::Tensor input = torch::from_blob(const_cast<float*>(state.data()),
                                         {1, kChannels, kHeight, kWidth}, opts)
                            .clone()
                            .to(device_);

  torch::NoGradGuard no_grad;
  auto output = module_.forward({input});

  auto list = output.toList();
  auto log_probs = list.get(0).toTensor().squeeze(0).cpu();  // (H*W)
  auto val_tensor = list.get(1).toTensor().squeeze().cpu();  // scalar

  auto probs_tensor = log_probs.exp();
  std::vector<float> probs(probs_tensor.data_ptr<float>(),
                           probs_tensor.data_ptr<float>() + HW);
  float value = val_tensor.item<float>();

  return {probs, value};
}
