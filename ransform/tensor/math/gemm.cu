#include "gemm.h"
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
        ) {
            return cublasSgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        }

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
        ) {
            return cublasDgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        }

        cublasStatus_t matmul_float(
            cublasHandle_t handle,
            int m, int n, int k,
            const float* A,
            const float* B,
            float* C
        ) {
            const float alpha = 1.0f;
            const float beta = 0.0f;
            // Для row‑major матриц A (m×k) и B (k×n) результат C = A·B
            // Используем формулу C^T = B^T * A^T
            return cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                n, m, k, &alpha,
                B, n,   // B^T, ldb = n
                A, k,   // A^T, lda = k
                &beta, C, n);
        }

        cublasStatus_t matmul_double(
            cublasHandle_t handle,
            int m, int n, int k,
            const double* A,
            const double* B,
            double* C
        ) {
            const double alpha = 1.0;
            const double beta = 0.0;
            return cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                n, m, k, &alpha,
                B, n,
                A, k,
                &beta, C, n);
        }

    }
}