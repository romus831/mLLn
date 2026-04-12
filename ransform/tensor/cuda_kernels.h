#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H
#include <cuda_runtime.h>

enum class FusedOp {
    ADD_RELU,
    ADD_SIGMOID,
    MUL_LEAKY_RELU,
};

#ifdef __CUDACC__

__global__ void relu_kernel(float* data, size_t n);
__global__ void leaky_relu_kernel(float* data, size_t n, float slope);
__global__ void sigmoid_kernel(float* data, size_t n);
__global__ void tanh_kernel(float* data, size_t n);
__global__ void fused_binary_activation_kernel(const float* a, const float* b, float* c, size_t n, int binary_op, int act_op, float leaky_slope);
#endif

void add_gpu_impl(const float* a, const float* b, float* c, size_t n);
void sub_gpu_impl(const float* a, const float* b, float* c, size_t n);
void mul_gpu_impl(const float* a, const float* b, float* c, size_t n);
void div_gpu_impl(const float* a, const float* b, float* c, size_t n);
void relu_gpu_impl(float* data, size_t n);
void leaky_relu_gpu_impl(float* data, size_t n, float negative_slope);
void sigmoid_gpu_impl(float* data, size_t n);
void tanh_gpu_impl(float* data, size_t n);
void fused_binary_activation_gpu_impl(
    const float* a, const float* b, float* c, size_t n,
    int binary_op, int act_op, float leaky_slope = 0.01f
);
#endif