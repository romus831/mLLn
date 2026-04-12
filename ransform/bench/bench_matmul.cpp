#include <chrono>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <vector>
#include "tensor.h"

using MNNL::Tensor;

int main() {
    int devs = 0;
    if (cudaGetDeviceCount(&devs) != cudaSuccess || devs == 0) {
        std::cerr << "bench_matmul: CUDA device required\n";
        return 1;
    }

    struct Case {
        size_t m, k, n;
    };
    const std::vector<Case> cases = {
        {32, 32, 32},
        {128, 128, 128},
        {512, 512, 512},
        {1024, 1024, 1024},
        {2048, 512, 512},
    };

    const int warmup = 5;
    const int iters = 20;

    std::cout << "bench_matmul: float GEMM via Tensor::matmul (cuBLAS), ms per call\n";
    std::cout << std::setw(6) << "m" << std::setw(8) << "k" << std::setw(8) << "n" << std::setw(14) << "ms/call"
              << std::setw(14) << "GFLOPS\n";

    for (const Case& c : cases) {
        Tensor<float> A({c.m, c.k});
        Tensor<float> B({c.k, c.n});
        for (size_t i = 0; i < A.size(); ++i) {
            A.data()[i] = static_cast<float>(i % 17) * 0.01f;
        }
        for (size_t i = 0; i < B.size(); ++i) {
            B.data()[i] = static_cast<float>(i % 19) * 0.01f;
        }

        for (int w = 0; w < warmup; ++w) {
            auto C = A.matmul(B);
            (void)C;
        }
        cudaDeviceSynchronize();

        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
            auto C = A.matmul(B);
            (void)C;
        }
        cudaDeviceSynchronize();
        const auto t1 = std::chrono::steady_clock::now();

        const double sec = std::chrono::duration<double>(t1 - t0).count();
        const double ms = (sec / static_cast<double>(iters)) * 1000.0;
        const double flops = 2.0 * static_cast<double>(c.m) * static_cast<double>(c.n) * static_cast<double>(c.k);
        const double gflops = (flops / 1e9) / (sec / static_cast<double>(iters));

        std::cout << std::setw(6) << c.m << std::setw(8) << c.k << std::setw(8) << c.n << std::fixed
                  << std::setprecision(4) << std::setw(14) << ms << std::setw(14) << gflops << "\n";
    }

    return 0;
}
