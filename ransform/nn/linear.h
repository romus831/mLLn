#pragma once

#include <cstdlib>
#include <stdexcept>
#include "module.h"
#include <random>

namespace MNNL::nn {

class Linear : public Module {
public:
    Linear(size_t in_features, size_t out_features, unsigned seed = 42)
        : weight_({in_features, out_features}),
          bias_({1, out_features}),
          in_features_(in_features),
          out_features_(out_features) {
        if (in_features == 0 || out_features == 0) {
            throw std::invalid_argument("Linear: in_features and out_features must be positive");
        }
        std::srand(seed);
        he_uniform_init_(weight_, in_features);
        random_fill_(weight_);
        bias_.zero();
        weight_.set_requires_grad(true);
        bias_.set_requires_grad(true);
    }

    Tensor<float>& weight() { return weight_; }
    Tensor<float>& bias() { return bias_; }
    const Tensor<float>& weight() const { return weight_; }
    const Tensor<float>& bias() const { return bias_; }

    std::shared_ptr<Tensor<float>> forward(const Tensor<float>& x) override {
        if (x.ndim() != 2 || x.shape()[1] != in_features_) {
            throw std::invalid_argument("Linear::forward: expected x shape (batch, in_features)");
        }
        auto z = x.matmul(weight_);
        return *z + bias_;
    }

private:
    Tensor<float> weight_;
    Tensor<float> bias_;
    size_t in_features_;
    size_t out_features_;

    static void random_fill_(Tensor<float>& t) {
        float* p = t.data();
        for (size_t i = 0; i < t.size(); ++i) {
            p[i] = (static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX)) - 0.5f;
        }
    }
    static void he_uniform_init_(Tensor<float>& t, size_t fan_in) {
        float bound = std::sqrt(6.0f / fan_in);
        float* p = t.data();
        for (size_t i = 0; i < t.size(); ++i) {
            p[i] = (static_cast<float>(std::rand()) / RAND_MAX) * 2.0f * bound - bound;
        }
    }
};

}  // namespace MNNL::nn
