#include "cuda_kernels.h"
#include <cuda_runtime.h>

__global__ void relu_kernel(float* data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

__global__ void leaky_relu_kernel(float* data, size_t n, float slope) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = data[idx];
        data[idx] = x > 0.0f ? x : slope * x;
    }
}

__global__ void sigmoid_kernel(float* data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = data[idx];
        data[idx] = 1.0f / (1.0f + expf(-x));
    }
}

__global__ void tanh_kernel(float* data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = data[idx];
        data[idx] = tanhf(x);
    }
}

__global__ void fused_binary_activation_kernel(
    const float* a, const float* b, float* c,
    size_t n,
    int binary_op,
    int act_op,
    float leaky_slope
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t offset = idx * 4;

    if (offset + 3 >= n) {
        // Обработка остатка по одному элементу
        for (size_t i = offset; i < n; ++i) {
            float x;
            switch (binary_op) {
            case 0: x = a[i] + b[i]; break;
            case 1: x = a[i] - b[i]; break;
            case 2: x = a[i] * b[i]; break;
            case 3: x = a[i] / b[i]; break;
            default: x = 0.0f;
            }
            switch (act_op) {
            case 0: break;
            case 1: x = fmaxf(0.0f, x); break;
            case 2: x = x > 0.0f ? x : leaky_slope * x; break;
            case 3: x = 1.0f / (1.0f + expf(-x)); break;
            case 4: x = tanhf(x); break;
            }
            c[i] = x;
        }
        return;
    }

    float4 a4 = reinterpret_cast<const float4*>(a)[idx];
    float4 b4 = reinterpret_cast<const float4*>(b)[idx];
    float4 c4;

    switch (binary_op) {
    case 0: // ADD
        c4.x = a4.x + b4.x;
        c4.y = a4.y + b4.y;
        c4.z = a4.z + b4.z;
        c4.w = a4.w + b4.w;
        break;
    case 1: // SUB
        c4.x = a4.x - b4.x;
        c4.y = a4.y - b4.y;
        c4.z = a4.z - b4.z;
        c4.w = a4.w - b4.w;
        break;
    case 2: // MUL
        c4.x = a4.x * b4.x;
        c4.y = a4.y * b4.y;
        c4.z = a4.z * b4.z;
        c4.w = a4.w * b4.w;
        break;
    case 3: // DIV
        c4.x = a4.x / b4.x;
        c4.y = a4.y / b4.y;
        c4.z = a4.z / b4.z;
        c4.w = a4.w / b4.w;
        break;
    default:
        c4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    switch (act_op) {
    case 0: break;
    case 1: // ReLU
        c4.x = fmaxf(0.0f, c4.x);
        c4.y = fmaxf(0.0f, c4.y);
        c4.z = fmaxf(0.0f, c4.z);
        c4.w = fmaxf(0.0f, c4.w);
        break;
    case 2: // Leaky ReLU
        c4.x = c4.x > 0.0f ? c4.x : leaky_slope * c4.x;
        c4.y = c4.y > 0.0f ? c4.y : leaky_slope * c4.y;
        c4.z = c4.z > 0.0f ? c4.z : leaky_slope * c4.z;
        c4.w = c4.w > 0.0f ? c4.w : leaky_slope * c4.w;
        break;
    case 3: // Sigmoid
        c4.x = 1.0f / (1.0f + expf(-c4.x));
        c4.y = 1.0f / (1.0f + expf(-c4.y));
        c4.z = 1.0f / (1.0f + expf(-c4.z));
        c4.w = 1.0f / (1.0f + expf(-c4.w));
        break;
    case 4: // Tanh
        c4.x = tanhf(c4.x);
        c4.y = tanhf(c4.y);
        c4.z = tanhf(c4.z);
        c4.w = tanhf(c4.w);
        break;
    }

    reinterpret_cast<float4*>(c)[idx] = c4;
}

static inline void launch_fused_kernel(
    const float* a, const float* b, float* c, size_t n,
    int binary_op, int act_op, float leaky_slope
) {
    if (n == 0) return;
    int blockSize = 256;
    int gridSize = (n + blockSize * 4 - 1) / (blockSize * 4);
    fused_binary_activation_kernel << <gridSize, blockSize >> > (
        a, b, c, n, binary_op, act_op, leaky_slope
        );
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());
}

void add_gpu_impl(const float* a, const float* b, float* c, size_t n) {
    launch_fused_kernel(a, b, c, n, 0, 0, 0.0f);
}

void sub_gpu_impl(const float* a, const float* b, float* c, size_t n) {
    launch_fused_kernel(a, b, c, n, 1, 0, 0.0f);
}

void mul_gpu_impl(const float* a, const float* b, float* c, size_t n) {
    launch_fused_kernel(a, b, c, n, 2, 0, 0.0f);
}

void div_gpu_impl(const float* a, const float* b, float* c, size_t n) {
    launch_fused_kernel(a, b, c, n, 3, 0, 0.0f);
}

void relu_gpu_impl(float* data, size_t n) {
    if (n == 0) return;
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    relu_kernel << <gridSize, blockSize >> > (data, n);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());
}
void leaky_relu_gpu_impl(float* data, size_t n, float negative_slope) {
    if (n == 0) return;
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    leaky_relu_kernel << <gridSize, blockSize >> > (data, n, negative_slope);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());
}

void sigmoid_gpu_impl(float* data, size_t n) {
    if (n == 0) return;
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    sigmoid_kernel << <gridSize, blockSize >> > (data, n);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());
}

void tanh_gpu_impl(float* data, size_t n) {
    if (n == 0) return;
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    tanh_kernel << <gridSize, blockSize >> > (data, n);
    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaDeviceSynchronize());
}

void fused_binary_activation_gpu_impl(
    const float* a, const float* b, float* c, size_t n,
    int binary_op, int act_op, float leaky_slope
) {
    launch_fused_kernel(a, b, c, n, binary_op, act_op, leaky_slope);
}