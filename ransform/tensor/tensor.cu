#include "cuda_util.h"
#include "tensor.h"
#include "cuda_kernels.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <utility>
#include "core/autograd.h"
#include "core/concepts.h"
#undef DEBUG_MODE

#ifdef DEBUG_MODE
#include <iostream>
#define DEBUG_LOG(msg) std::cout << msg << std::endl
#else
#define DEBUG_LOG(msg) ((void)0)
#endif

namespace MNNL {

    inline std::unique_ptr<float[], CudaDeleter> allocate_gpu_memory(size_t count) {
        float* raw_ptr = nullptr;
        CUDA_CHECK(cudaMalloc(&raw_ptr, count * sizeof(float)));
        return std::unique_ptr<float[], CudaDeleter>(raw_ptr, CudaDeleter{});
    }

    template<ArithmeticType T>
    Tensor<T>::Tensor(Tensor&& other) noexcept
        : shape_(std::move(other.shape_)),
        strides_(std::move(other.strides_)),
        data_(std::move(other.data_)),
        offset_(other.offset_),
        total_size_(other.total_size_),
        gpu_data_(std::move(other.gpu_data_)),
        on_gpu_(other.on_gpu_) {
        other.gpu_data_ = nullptr;
        other.on_gpu_ = false;
        other.offset_ = 0;
        other.total_size_ = 0;
    }

    template<ArithmeticType T>
    Tensor<T>::Tensor(const Tensor& other)
        : shape_(other.shape_),
        strides_(other.strides_),
        data_(other.data_),
        offset_(other.offset_),
        total_size_(other.total_size_),
        gpu_data_(nullptr),
        on_gpu_(false),
        requires_grad_(other.requires_grad_),
        grad_(nullptr)
    {
        if (other.on_gpu_ && other.gpu_data_) {
            auto gpu_ptr = allocate_gpu_memory(total_size_);
            if (!gpu_ptr) throw std::runtime_error("gpu_ptr is null");
            CUDA_CHECK(cudaMemcpy(gpu_ptr.get(), other.gpu_data_.get(),
                total_size_ * sizeof(T), cudaMemcpyDeviceToDevice));
            gpu_data_ = std::move(gpu_ptr);
            on_gpu_ = true;
        }
    }

    template<ArithmeticType T>
    Tensor<T>& Tensor<T>::operator=(Tensor&& other) noexcept {
        if (this != &other) {
            gpu_data_.reset();

            shape_ = std::move(other.shape_);
            strides_ = std::move(other.strides_);
            data_ = std::move(other.data_);
            offset_ = other.offset_;
            total_size_ = other.total_size_;
            gpu_data_ = std::move(other.gpu_data_);
            on_gpu_ = other.on_gpu_;
            other.offset_ = 0;
            other.total_size_ = 0;
            other.on_gpu_ = false;
        }
        return *this;
    }

    template<ArithmeticType T>
    void Tensor<T>::to_gpu() {
        if (on_gpu_) {
            DEBUG_LOG("[DEBUG to_gpu] Уже на GPU, ничего не делаем.");
            return;
        }
        if (size() == 0) return;

        DEBUG_LOG("[DEBUG to_gpu] Переводим тензор на GPU. Размер = " << size());
        if (!gpu_data_) {
            auto gpu_ptr = allocate_gpu_memory(total_size_);
            DEBUG_LOG("[DEBUG to_gpu] Выделили новую GPU память.");

            if (data_) {
                CUDA_CHECK(cudaMemcpy(gpu_ptr.get(), this->data(),
                    total_size_ * sizeof(T), cudaMemcpyHostToDevice));
                DEBUG_LOG("[DEBUG to_gpu] Данные скопированы CPU → GPU.");
            }

            // Перемещаем владение в член класса
            gpu_data_ = std::move(gpu_ptr);
        }
        // Сценарий 2: Буфер уже существует — используем его без перевыделения
        else {
            DEBUG_LOG("[DEBUG to_gpu] GPU буфер уже существует, используем его.");
            if (data_) {
                CUDA_CHECK(cudaMemcpy(gpu_data_.get(), this->data(),
                    total_size_ * sizeof(T), cudaMemcpyHostToDevice));
                DEBUG_LOG("[DEBUG to_gpu] Данные скопированы CPU → GPU (повторное использование буфера).");
            }
        }

        on_gpu_ = true;
        DEBUG_LOG("[DEBUG to_gpu] Успешно. on_gpu_ = true");
    }

    template<ArithmeticType T>
    void Tensor<T>::to_cpu() {
        if (!on_gpu_) return;
        if (size() == 0) return;

        if (!data_) {
            void* ptr = nullptr;
#ifdef _WIN32
            ptr = _aligned_malloc(total_size_ * sizeof(T), 64);
            if (!ptr) throw std::bad_alloc();
#else
            if (posix_memalign(&ptr, 64, total_size_ * sizeof(T)) != 0)
                throw std::bad_alloc();
#endif
            data_ = std::shared_ptr<T[]>(static_cast<T*>(ptr), AlignedDeleter{});
            ComputeContiguousStrides();
            offset_ = 0;
        }
        CUDA_CHECK(cudaMemcpy(this->data(), gpu_data_.get(),
            total_size_ * sizeof(T), cudaMemcpyDeviceToHost));
        on_gpu_ = false;
    }

    template<ArithmeticType T>
    void Tensor<T>::backward() {
        autograd::backward(*this);
    }

    template<ArithmeticType T>
    Tensor<T>::~Tensor() noexcept {
    }

    template class Tensor<float>;
}