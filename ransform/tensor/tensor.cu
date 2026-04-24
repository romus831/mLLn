#include "tensor.h"
#include "cuda_kernels.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <utility>
#include "core/autograd.h"
#include "concepts.h"
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
        cudaError_t err = cudaMalloc(&raw_ptr, count * sizeof(float));
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed: " + std::string(cudaGetErrorString(err)));
        }
        return std::unique_ptr<float[], CudaDeleter>(raw_ptr, CudaDeleter{});
    }
    /*
        template<typename T>
        Tensor<T>::Tensor(const Tensor& other)
            : shape_(other.shape_),
            strides_(other.strides_),
            data_(other.data_),
            offset_(other.offset_),
            total_size_(other.total_size_),
            gpu_data_(nullptr),
            on_gpu_(false),
            requires_grad_(other.requires_grad_),   // добавить
            grad_(nullptr) {

            DEBUG_LOG("[DEBUG Copy Ctor] Копируем тензор. Исходный on_gpu = "
                << (other.on_gpu_ ? "true" : "false")
                << ", размер = " << other.size());

            if (other.on_gpu_ && other.gpu_data_) {
                float* raw = nullptr;
                cudaError_t err = cudaMalloc(&raw, total_size_ * sizeof(float));
                if (err != cudaSuccess) {
                    throw std::runtime_error("cudaMalloc in copy ctor: "
                        + std::string(cudaGetErrorString(err)));
                }
                gpu_data_.reset(raw);

                err = cudaMemcpy(gpu_data_.get(), other.gpu_data_.get(),
                    total_size_ * sizeof(float), cudaMemcpyDeviceToDevice);
                if (err != cudaSuccess) {
                    throw std::runtime_error("cudaMemcpy D->D in copy ctor: "
                        + std::string(cudaGetErrorString(err)));
                }

                on_gpu_ = true;
                DEBUG_LOG("[DEBUG Copy Ctor] Успешно скопировали на GPU. on_gpu_ = true");
            }
            else {
                DEBUG_LOG("[DEBUG Copy Ctor] Тензор на CPU, GPU память не копируем.");
            }
        }
        */

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
            cudaError_t err = cudaMemcpy(gpu_ptr.get(), other.gpu_data_.get(),
                total_size_ * sizeof(float), cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) {
                throw std::runtime_error("cudaMemcpy D->D in copy ctor: "
                    + std::string(cudaGetErrorString(err)));
            }
            gpu_data_ = std::move(gpu_ptr);
            on_gpu_ = true;
        }
    }

    template<ArithmeticType T>
    Tensor<T>& Tensor<T>::operator=(Tensor&& other) noexcept {
        if (this != &other) {
            gpu_data_.reset();  // освобождает старую GPU память

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

        // Сценарий 1: GPU-буфера ещё нет — выделяем новый безопасным способом
        if (!gpu_data_) {
            auto gpu_ptr = allocate_gpu_memory(total_size_);  // unique_ptr с кастомным deleter
            DEBUG_LOG("[DEBUG to_gpu] Выделили новую GPU память.");

            if (data_) {
                cudaError_t err = cudaMemcpy(gpu_ptr.get(), this->data(),
                    total_size_ * sizeof(float), cudaMemcpyHostToDevice);
                if (err != cudaSuccess) {
                    // Исключение: gpu_ptr автоматически освободит память, состояние объекта не меняется
                    throw std::runtime_error("cudaMemcpy H->D failed: " + std::string(cudaGetErrorString(err)));
                }
                DEBUG_LOG("[DEBUG to_gpu] Данные скопированы CPU → GPU.");
            }

            // Перемещаем владение в член класса
            gpu_data_ = std::move(gpu_ptr);
        }
        // Сценарий 2: Буфер уже существует — используем его без перевыделения
        else {
            DEBUG_LOG("[DEBUG to_gpu] GPU буфер уже существует, используем его.");
            if (data_) {
                cudaError_t err = cudaMemcpy(gpu_data_.get(), this->data(),
                    total_size_ * sizeof(float), cudaMemcpyHostToDevice);
                if (err != cudaSuccess) {
                    throw std::runtime_error("cudaMemcpy H->D failed: " + std::string(cudaGetErrorString(err)));
                }
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

        cudaError_t err = cudaMemcpy(this->data(), gpu_data_.get(),
            total_size_ * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMemcpy D->H failed: " + std::string(cudaGetErrorString(err)));
        }

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