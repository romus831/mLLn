#include <gtest/gtest.h>
#include <memory>
#include <cuda_runtime.h>
#include "nn/mnnl_nn.h"
#include "tensor.h"

using MNNL::Tensor;

static bool cuda_ok() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

TEST(NnLinear, ForwardShape) {
    ASSERT_TRUE(cuda_ok());
    MNNL::nn::Linear layer(3, 5, 99u);
    Tensor<float> x({2, 3});
    for (size_t i = 0; i < x.size(); ++i) {
        x.data()[i] = static_cast<float>(i) * 0.1f;
    }
    auto y = layer.forward(x);
    ASSERT_EQ(y->shape().size(), 2u);
    EXPECT_EQ(y->shape()[0], 2u);
    EXPECT_EQ(y->shape()[1], 5u);
}

TEST(NnSequential, ChainMatchesIndividual) {
    ASSERT_TRUE(cuda_ok());
    Tensor<float> x({1, 2});
    x.data()[0] = 0.2f;
    x.data()[1] = -0.3f;

    MNNL::nn::Linear L1(2, 3, 7u);
    MNNL::nn::SigmoidModule S;
    MNNL::nn::Linear L2(3, 1, 8u);

    auto a = L1.forward(x);
    auto b = S.forward(*a);
    auto c = L2.forward(*b);

    MNNL::nn::Sequential seq;
    seq.add(std::make_unique<MNNL::nn::Linear>(2, 3, 7u));
    seq.emplace_back<MNNL::nn::SigmoidModule>();
    seq.add(std::make_unique<MNNL::nn::Linear>(3, 1, 8u));

    auto y = seq.forward(x);
    y->to_cpu();
    c->to_cpu();
    for (size_t i = 0; i < y->size(); ++i) {
        EXPECT_NEAR(y->data()[i], c->data()[i], 1e-4f) << "i=" << i;
    }
}

TEST(NnSequential, EmptyThrows) {
    MNNL::nn::Sequential seq;
    Tensor<float> x({1, 1});
    x.data()[0] = 1.f;
    EXPECT_THROW(seq.forward(x), std::runtime_error);
}
