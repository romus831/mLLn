#pragma once

#include <memory>
#include "tensor.h"

namespace MNNL::nn {

class Module {
public:
    virtual ~Module() = default;
    virtual std::shared_ptr<Tensor<float>> forward(const Tensor<float>& x) = 0;
};

}  // namespace MNNL::nn
