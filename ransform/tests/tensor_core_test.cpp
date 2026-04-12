#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

#include "tensor.h"

using MNNL::Tensor;

namespace {

bool cuda_available() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

void matmul_cpu_ref(const Tensor<float>& A, const Tensor<float>& B, float* out_c) {
    const size_t m = A.shape()[0];
    const size_t k = A.shape()[1];
    const size_t n = B.shape()[1];
    const float* pa = A.data();
    const float* pb = B.data();
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float s = 0.f;
            for (size_t t = 0; t < k; ++t) {
                s += pa[i * k + t] * pb[t * n + j];
            }
            out_c[i * n + j] = s;
        }
    }
}

}  // namespace

TEST(Environment, CudaDevicePresent) {
    ASSERT_TRUE(cuda_available()) << "CUDA device required for tensor GPU tests";
}

TEST(Matmul, GoldenSmall_ToCpuMatchesReference) {
    ASSERT_TRUE(cuda_available());

    Tensor<float> A({2, 2});
    Tensor<float> B({2, 2});
    float* a = A.data();
    float* b = B.data();
    a[0] = 1.f;
    a[1] = 2.f;
    a[2] = 3.f;
    a[3] = 4.f;
    b[0] = 1.f;
    b[1] = 0.f;
    b[2] = 0.f;
    b[3] = 1.f;

    auto C = A.matmul(B);
    C->to_cpu();

    float ref[4];
    matmul_cpu_ref(A, B, ref);
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(C->data()[i], ref[i], 1e-3f) << "index " << i;
    }
}

TEST(Matmul, LargerRandom_ToCpuMatchesReference) {
    ASSERT_TRUE(cuda_available());

    const size_t m = 16, k = 24, n = 12;
    Tensor<float> A({m, k});
    Tensor<float> B({k, n});
    for (size_t i = 0; i < A.size(); ++i) {
        A.data()[i] = std::sin(static_cast<float>(i) * 0.13f) * 0.5f;
    }
    for (size_t i = 0; i < B.size(); ++i) {
        B.data()[i] = std::cos(static_cast<float>(i) * 0.11f) * 0.5f;
    }

    auto C = A.matmul(B);
    C->to_cpu();

    std::vector<float> ref(m * n);
    matmul_cpu_ref(A, B, ref.data());
    for (size_t i = 0; i < m * n; ++i) {
        EXPECT_NEAR(C->data()[i], ref[i], 5e-2f) << "i=" << i;
    }
}

// Регрессия: результат matmul только на GPU; bias на CPU. Сложение должно
// давать matmul(A,B)+bias. Без синхронизации CPU-буфера это нарушается.
TEST(MixedDevice, AddBiasAfterMatmul_matchesReference) {
    ASSERT_TRUE(cuda_available());

    Tensor<float> A({3, 2});
    Tensor<float> B({2, 4});
    float* pa = A.data();
    pa[0] = 1.f;
    pa[1] = 2.f;
    pa[2] = 3.f;
    pa[3] = 4.f;
    pa[4] = 5.f;
    pa[5] = 6.f;
    float* pb = B.data();
    for (size_t i = 0; i < B.size(); ++i) {
        pb[i] = static_cast<float>(i % 5) * 0.1f - 0.2f;
    }

    auto C_gpu = A.matmul(B);
    ASSERT_TRUE(C_gpu->is_gpu());

    Tensor<float> bias({3, 4});
    for (size_t i = 0; i < bias.size(); ++i) {
        bias.data()[i] = 0.25f;
    }

    std::vector<float> ref(3 * 4);
    matmul_cpu_ref(A, B, ref.data());
    for (size_t i = 0; i < ref.size(); ++i) {
        ref[i] += bias.data()[i];
    }

    auto S = *C_gpu + bias;
    S->to_cpu();

    for (size_t i = 0; i < ref.size(); ++i) {
        EXPECT_NEAR(S->data()[i], ref[i], 1e-2f) << "i=" << i;
    }
}

TEST(MixedDevice, AddBiasAfterMatmul_correctWhenSyncedFirst) {
    ASSERT_TRUE(cuda_available());

    Tensor<float> A({3, 2});
    Tensor<float> B({2, 4});
    float* pa = A.data();
    for (size_t i = 0; i < A.size(); ++i) {
        pa[i] = static_cast<float>(i + 1) * 0.05f;
    }
    float* pb = B.data();
    for (size_t i = 0; i < B.size(); ++i) {
        pb[i] = static_cast<float>(i) * 0.03f - 0.1f;
    }

    auto C = A.matmul(B);
    C->to_cpu();

    Tensor<float> bias({3, 4});
    for (size_t i = 0; i < bias.size(); ++i) {
        bias.data()[i] = 1.5f;
    }

    auto S = *C + bias;
    S->to_cpu();

    std::vector<float> ref(3 * 4);
    matmul_cpu_ref(A, B, ref.data());
    for (size_t i = 0; i < ref.size(); ++i) {
        ref[i] += bias.data()[i];
    }

    for (size_t i = 0; i < ref.size(); ++i) {
        EXPECT_NEAR(S->data()[i], ref[i], 1e-2f) << "i=" << i;
    }
}

TEST(Elementwise, AddOnGpuBothOperands) {
    ASSERT_TRUE(cuda_available());

    Tensor<float> U({4});
    Tensor<float> V({4});
    for (int i = 0; i < 4; ++i) {
        U.data()[i] = static_cast<float>(i);
        V.data()[i] = static_cast<float>(10 - i);
    }
    U.to_gpu();
    V.to_gpu();

    auto W = U + V;
    W->to_cpu();
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(W->data()[i], 10.f, 1e-5f);
    }
}
