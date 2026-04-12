#pragma once

#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>
#include "module.h"

namespace MNNL::nn {

class Sequential : public Module {
public:
    void add(std::unique_ptr<Module> layer) {
        if (!layer) {
            throw std::invalid_argument("Sequential::add: null layer");
        }
        layers_.push_back(std::move(layer));
    }

    template <typename T, typename... Args>
    void emplace_back(Args&&... args) {
        layers_.push_back(std::make_unique<T>(std::forward<Args>(args)...));
    }

    std::shared_ptr<Tensor<float>> forward(const Tensor<float>& x) override {
        if (layers_.empty()) {
            throw std::runtime_error("Sequential::forward: empty module list");
        }
        std::shared_ptr<Tensor<float>> y = layers_[0]->forward(x);
        for (size_t i = 1; i < layers_.size(); ++i) {
            y = layers_[i]->forward(*y);
        }
        return y;
    }

    size_t size() const noexcept { return layers_.size(); }

private:
    std::vector<std::unique_ptr<Module>> layers_;
};

}  // namespace MNNL::nn
