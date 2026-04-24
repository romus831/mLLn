#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

#define CUDA_KERNEL_CHECK()  CUDA_CHECK(cudaGetLastError())
#define CUDA_KERNEL_CHECK_SYNC() \
    CUDA_KERNEL_CHECK();         \
    CUDA_CHECK(cudaDeviceSynchronize())

    template<typename T>
struct GPUMemory {
    T* ptr = nullptr;
    size_t count = 0;
    
    void alloc(size_t n) {
        free();
        CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
        count = n;
    }
    void free() {
        if (ptr) { cudaFree(ptr); ptr = nullptr; count = 0; }
    }
    ~GPUMemory() { free(); }
    operator T*() { return ptr; }
    operator const T*() const { return ptr; }
};

enum class BinaryOp : int {
    ADD = 0,
    SUB = 1,
    MUL = 2,
    DIV = 3
};

enum class ActOp : int {
    NONE = 0,
    RELU = 1,
    LEAKY_RELU = 2,
    SIGMOID = 3,
    TANH = 4
};

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