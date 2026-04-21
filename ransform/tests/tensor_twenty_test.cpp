#include <gtest/gtest.h>
#include <vector>

#include "nn/mnnl_nn.h"
#include "tensor.h"

using MNNL::Tensor;

TEST(TensorTwenty, InvalidZeroTotalSizeThrows) {
    EXPECT_THROW((Tensor<float>({2, 0})), std::invalid_argument);
}

TEST(TensorTwenty, ZerosAreAllZero) {
    auto t = Tensor<float>::zeros({3, 4});
    ASSERT_EQ(t.size(), 12u);
    for (size_t i = 0; i < t.size(); ++i) {
        EXPECT_EQ(t.data()[i], 0.f);
    }
}

TEST(TensorTwenty, OnesAreAllOne) {
    auto t = Tensor<float>::ones({2, 5});
    ASSERT_EQ(t.size(), 10u);
    for (size_t i = 0; i < t.size(); ++i) {
        EXPECT_EQ(t.data()[i], 1.f);
    }
}

TEST(TensorTwenty, CloneCopiesValues) {
    Tensor<float> a({2, 2});
    float* p = a.data();
    p[0] = 1.f;
    p[1] = -2.f;
    p[2] = 3.5f;
    p[3] = 0.f;
    Tensor<float> b = a.clone();
    for (size_t i = 0; i < a.size(); ++i) {
        EXPECT_EQ(b.data()[i], p[i]);
    }
}

TEST(TensorTwenty, CloneIsIndependentStorage) {
    Tensor<float> a({2});
    a.data()[0] = 7.f;
    a.data()[1] = 8.f;
    Tensor<float> b = a.clone();
    b.data()[0] = 0.f;
    EXPECT_EQ(a.data()[0], 7.f);
}

TEST(TensorTwenty, ReshapeContiguousSixElements) {
    Tensor<float> a({2, 3});
    for (size_t i = 0; i < 6; ++i) {
        a.data()[i] = static_cast<float>(i);
    }
    Tensor<float> b = a.reshape({6});
    EXPECT_EQ(b.shape().size(), 1u);
    EXPECT_EQ(b.shape()[0], 6u);
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_EQ(b.data()[i], static_cast<float>(i));
    }
}

TEST(TensorTwenty, TransposeSwapsAxes) {
    Tensor<float> a({2, 3});
    for (size_t i = 0; i < 6; ++i) {
        a.data()[i] = static_cast<float>(i + 1);
    }
    Tensor<float> t = a.transpose();
    EXPECT_EQ(t.shape()[0], 3u);
    EXPECT_EQ(t.shape()[1], 2u);
    EXPECT_FLOAT_EQ(t(0, 0), 1.f);
    EXPECT_FLOAT_EQ(t(0, 1), 4.f);
    EXPECT_FLOAT_EQ(t(2, 1), 6.f);
}

TEST(TensorTwenty, PermuteThreeDimensions) {
    Tensor<float> a({2, 1, 3});
    for (size_t i = 0; i < a.size(); ++i) {
        a.data()[i] = static_cast<float>(10 + i);
    }
    Tensor<float> p = a.permute({2, 0, 1});
    EXPECT_EQ(p.shape(), (std::vector<size_t>{3, 2, 1}));
    EXPECT_FLOAT_EQ(p(0, 0, 0), 10.f);
    // original linear index 5 -> (i0,i1,i2)=(1,0,2) -> permuted (2,1,0)
    EXPECT_FLOAT_EQ(p(2, 1, 0), 15.f);
}

TEST(TensorTwenty, SliceReturnsViewOfRegion) {
    Tensor<float> a({5});
    for (size_t i = 0; i < 5; ++i) {
        a.data()[i] = static_cast<float>(i * 2);
    }
    Tensor<float> s = a.slice({2}, {2});
    ASSERT_EQ(s.shape(), (std::vector<size_t>{2}));
    EXPECT_FLOAT_EQ(s.data()[0], 4.f);
    EXPECT_FLOAT_EQ(s.data()[1], 6.f);
}

TEST(TensorTwenty, SliceOutOfRangeThrows) {
    Tensor<float> a({3});
    EXPECT_THROW(a.slice({0}, {5}), std::out_of_range);
}

TEST(TensorTwenty, ScalarMultiply) {
    Tensor<float> a({2, 2});
    for (size_t i = 0; i < 4; ++i) {
        a.data()[i] = 1.5f;
    }
    Tensor<float> b = a * 2.f;
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(b.data()[i], 3.f, 1e-6f);
    }
}

TEST(TensorTwenty, AddSameShapeCpu) {
    Tensor<float> a({2});
    Tensor<float> b({2});
    a.data()[0] = 1.f;
    a.data()[1] = 2.f;
    b.data()[0] = 10.f;
    b.data()[1] = -1.f;
    auto c = a + b;
    c->to_cpu();
    EXPECT_NEAR(c->data()[0], 11.f, 1e-6f);
    EXPECT_NEAR(c->data()[1], 1.f, 1e-6f);
}

TEST(TensorTwenty, SubSameShapeCpu) {
    Tensor<float> a({3});
    Tensor<float> b({3});
    for (size_t i = 0; i < 3; ++i) {
        a.data()[i] = static_cast<float>(i);
        b.data()[i] = 1.f;
    }
    auto c = a - b;
    c->to_cpu();
    EXPECT_NEAR(c->data()[0], -1.f, 1e-6f);
    EXPECT_NEAR(c->data()[1], 0.f, 1e-6f);
    EXPECT_NEAR(c->data()[2], 1.f, 1e-6f);
}

TEST(TensorTwenty, MulElementwiseCpu) {
    Tensor<float> a({2});
    Tensor<float> b({2});
    a.data()[0] = 2.f;
    a.data()[1] = -3.f;
    b.data()[0] = 4.f;
    b.data()[1] = 2.f;
    auto c = a * b;
    c->to_cpu();
    EXPECT_NEAR(c->data()[0], 8.f, 1e-6f);
    EXPECT_NEAR(c->data()[1], -6.f, 1e-6f);
}

TEST(TensorTwenty, BroadcastAddRowToMatrix) {
    Tensor<float> a({2, 1});
    Tensor<float> b({1, 3});
    for (size_t i = 0; i < 2; ++i) {
        a.data()[i] = static_cast<float>(i);
    }
    for (size_t j = 0; j < 3; ++j) {
        b.data()[j] = static_cast<float>(j);
    }
    auto c = a + b;
    c->to_cpu();
    ASSERT_EQ(c->shape(), (std::vector<size_t>{2, 3}));
    EXPECT_NEAR(c->data()[0], 0.f, 1e-6f);
    EXPECT_NEAR(c->data()[1], 1.f, 1e-6f);
    EXPECT_NEAR(c->data()[3], 1.f, 1e-6f);
    EXPECT_NEAR(c->data()[5], 3.f, 1e-6f);
}

TEST(TensorTwenty, ZeroMethodClearsBuffer) {
    Tensor<float> a({4});
    for (size_t i = 0; i < 4; ++i) {
        a.data()[i] = 9.f;
    }
    a.zero();
    for (size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(a.data()[i], 0.f);
    }
}

TEST(TensorTwenty, MultiIndexReadWrite) {
    Tensor<float> t = Tensor<float>::zeros({2, 3});
    t(1, 2) = 42.f;
    EXPECT_FLOAT_EQ(t(1, 2), 42.f);
    EXPECT_FLOAT_EQ(t(0, 0), 0.f);
}

TEST(TensorTwenty, ReshapeNonContiguousThrows) {
    Tensor<float> a({2, 3});
    Tensor<float> nt = a.transpose();
    ASSERT_FALSE(nt.is_contiguous());
    EXPECT_THROW(nt.reshape({6}), std::runtime_error);
}

TEST(TensorTwenty, LinearZeroInFeaturesThrows) {
    EXPECT_THROW((MNNL::nn::Linear(0, 4)), std::invalid_argument);
}

TEST(TensorTwenty, LinearForwardWrongBatchWidthThrows) {
    MNNL::nn::Linear layer(4, 2);
    Tensor<float> x({3, 3});
    x.data()[0] = 1.f;
    EXPECT_THROW(layer.forward(x), std::invalid_argument);
}
