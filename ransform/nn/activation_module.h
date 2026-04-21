#pragma once

#include "module.h"

namespace MNNL::nn {

class SigmoidModule : public Module {
public:
    std::shared_ptr<Tensor<float>> forward(const Tensor<float>& x) override { return x.sigmoid(); }
};

class ReLUModule : public Module {
public:
    std::shared_ptr<Tensor<float>> forward(const Tensor<float>& x) override { return x.relu(); }
};
}  // namespace MNNL::nn