#include <algorithm>
#include <iostream>
#include <iomanip>
#include "nn/mnnl_nn.h"
#include "tensor.h"

using namespace MNNL;

int main() {
    std::cout << "nn_demo: MLP 2-4-1 через Sequential (Linear + ReLUModule) x2\n";

    nn::Sequential net;
    net.emplace_back<nn::Linear>(2, 4, 11u);
    net.emplace_back<nn::ReLUModule>();
    net.emplace_back<nn::Linear>(4, 1, 13u);
    net.emplace_back<nn::ReLUModule>();

    Tensor<float> X({4, 2});
    const float xv[] = {0, 0, 0, 1, 1, 0, 1, 1};
    std::copy(std::begin(xv), std::end(xv), X.data());

    auto y = net.forward(X);
    y->to_cpu();
    std::cout << std::fixed << std::setprecision(4);
    for (size_t i = 0; i < 4; ++i) {
        std::cout << "sample " << i << " -> " << y->data()[i] << "\n";
    }
    return 0;
}
