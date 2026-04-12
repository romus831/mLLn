#ifndef MNNL_MATH_GEMM_H
#define MNNL_MATH_GEMM_H

#include <cublas_v2.h>

namespace MNNL {
    namespace math {

        cublasStatus_t gemm_float(
            cublasHandle_t handle,
            cublasOperation_t transA,
            cublasOperation_t transB,
            int m, int n, int k,
            const float* alpha,
            const float* A, int lda,
            const float* B, int ldb,
            const float* beta,
            float* C, int ldc
        );

        cublasStatus_t gemm_double(
            cublasHandle_t handle,
            cublasOperation_t transA,
            cublasOperation_t transB,
            int m, int n, int k,
            const double* alpha,
            const double* A, int lda,
            const double* B, int ldb,
            const double* beta,
            double* C, int ldc
        );

        cublasStatus_t matmul_float(
            cublasHandle_t handle,
            int m, int n, int k,
            const float* A,
            const float* B,
            float* C
        );
    }
}
#endif