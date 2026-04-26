#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <string>
#include <sstream>

namespace cuda_util {
    inline void check_(cudaError_t err, const char* file, int line) {
        if (err != cudaSuccess) {
            std::ostringstream oss;
            oss << "CUDA error at " << file << ":" << line
                << " (" << cudaGetErrorName(err) << ") "
                << cudaGetErrorString(err);
            throw std::runtime_error(oss.str());
        }
    }

    inline void check_(cublasStatus_t err, const char* file, int line) {
        if (err != CUBLAS_STATUS_SUCCESS) {
            std::ostringstream oss;
            oss << "cuBLAS error at " << file << ":" << line << " (code " << err << ")";
            throw std::runtime_error(oss.str());
        }
    }
}

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        cuda_util::check_(err, __FILE__, __LINE__); \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t err = (call); \
        cuda_util::check_(err, __FILE__, __LINE__); \
    } while(0)

#define CUDA_KERNEL_CHECK() \
    CUDA_CHECK(cudaGetLastError())

#define CUDA_SYNC_CHECK() \
    CUDA_CHECK(cudaDeviceSynchronize())

#define CUDA_LAUNCH_AND_SYNC(kernel_launch) \
    do { \
        kernel_launch; \
        CUDA_KERNEL_CHECK(); \
        CUDA_SYNC_CHECK(); \
    } while(0)