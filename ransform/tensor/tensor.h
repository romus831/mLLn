#pragma once
#include <numeric>
#include <algorithm>
#include <functional>
#include <immintrin.h>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "cuda_kernels.h"
#include <cublas_v2.h>
#include "math/gemm.h"
#include "core/autograd.h"
#include "concepts.h"

namespace MNNL {
    struct AlignedDeleter {
        void operator()(void* ptr) const noexcept {
#ifdef _WIN32
            _aligned_free(ptr);
#else
            ::free(ptr);
#endif
        }
    };

    struct CudaDeleter {
        void operator()(float* ptr) const noexcept {
            if (ptr) cudaFree(ptr);
        }
    };

template<ArithmeticType T>
class Tensor : public std::enable_shared_from_this<Tensor<T>> {
        //static_assert(std::is_same_v<T, float>, "SIMD operations are currently supported only for float.");
    public:

        Tensor() noexcept
            : shape_(), strides_(), data_(nullptr), offset_(0), total_size_(0) {
        }

        explicit Tensor(const std::vector<size_t>& shape, size_t alignment = 64)
            : shape_(shape), offset_(0) {
            total_size_ = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies{});
            if (total_size_ == 0) throw std::invalid_argument("Tensor: shape contains zero dimension");
            void* ptr = nullptr;
#ifdef _WIN32
            ptr = _aligned_malloc(total_size_ * sizeof(T), alignment);
            if (!ptr) throw std::bad_alloc();
#else
            if (posix_memalign(&ptr, alignment, total_size_ * sizeof(T)) != 0)
                throw std::bad_alloc();
#endif
            data_ = std::shared_ptr<T[]>(static_cast<T*>(ptr), AlignedDeleter{});
            ComputeContiguousStrides();
        }

        Tensor(const std::vector<size_t>& shape,
            const std::vector<size_t>& strides,
            std::shared_ptr<T[]> data,
            size_t offset)
            : shape_(shape), strides_(strides), data_(std::move(data)), offset_(offset) {
            if (shape_.size() != strides_.size())
                throw std::invalid_argument("shape and strides must have the same dimensionality");
            total_size_ = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies{});
        }

        Tensor(const Tensor& other);
        Tensor(Tensor&& other) noexcept;
        Tensor& operator=(Tensor&& other) noexcept;
        Tensor& operator=(const Tensor&) = delete;

        ~Tensor() noexcept;

        template <typename... Idx>
        T& operator()(Idx... indices) {
            if (sizeof...(Idx) != shape_.size())
                throw std::invalid_argument("Number of indices does not match tensor dimensions");
            size_t linear_idx = offset_;
            size_t dim = 0;
            ((linear_idx += static_cast<size_t>(indices) * strides_[dim++]), ...);
            return data_[linear_idx];
        }

        template <typename... Idx>
        const T& operator()(Idx... indices) const {
            if (sizeof...(Idx) != shape_.size())
                throw std::invalid_argument("Number of indices does not match tensor dimensions");
            size_t linear_idx = offset_;
            size_t dim = 0;
            ((linear_idx += static_cast<size_t>(indices) * strides_[dim++]), ...);
            return data_[linear_idx];
        }

        Tensor slice(const std::vector<size_t>& starts, const std::vector<size_t>& sizes) const {
            if (starts.size() != shape_.size() || sizes.size() != shape_.size())
                throw std::invalid_argument("slice: dimensions do not match");
            std::vector<size_t> new_shape = sizes;
            std::vector<size_t> new_strides = strides_;
            size_t new_offset = offset_;
            for (size_t i = 0; i < starts.size(); ++i) {
                if (starts[i] + sizes[i] > shape_[i])
                    throw std::out_of_range("slice: out of bounds");
                new_offset += starts[i] * strides_[i];
            }
            return static_cast<Tensor>(Tensor(new_shape, new_strides, data_, new_offset));
        }

        Tensor reshape(const std::vector<size_t>& new_shape) const {
            size_t new_total = std::accumulate(new_shape.begin(), new_shape.end(), size_t(1), std::multiplies{});
            if (new_total != total_size_)
                throw std::invalid_argument("reshape: total number of elements does not match");
            if (!is_contiguous())
                throw std::runtime_error("reshape: tensor is not contiguous → call clone() first");
            std::vector<size_t> new_strides(new_shape.size());
            size_t stride = 1;
            for (int i = static_cast<int>(new_shape.size()) - 1; i >= 0; --i) {
                new_strides[i] = stride;
                stride *= new_shape[i];
            }
            return static_cast<Tensor>(Tensor(Tensor(new_shape, new_strides, data_, offset_)));
        }

        Tensor contiguous() const {
            if (is_contiguous()) {
                return static_cast<Tensor>(*this);
            }
            return clone();
        }

        Tensor permute(const std::vector<size_t>& order) const {
            if (order.size() != shape_.size())
                throw std::invalid_argument("permute: order rank mismatch");
            std::vector<size_t> new_shape(order.size());
            std::vector<size_t> new_strides(order.size());
            for (size_t i = 0; i < order.size(); ++i) {
                size_t src = order[i];
                if (src >= shape_.size())
                    throw std::out_of_range("permute: axis index out of range");
                new_shape[i] = shape_[src];
                new_strides[i] = strides_[src];
            }
            return static_cast<Tensor>(Tensor(new_shape, new_strides, data_, offset_));
        }

        Tensor transpose() const {
            if (ndim() != 2)
                throw std::invalid_argument("transpose: expected 2D tensor");
            return permute({ 1, 0 });
        }

        [[nodiscard]] Tensor clone() const {
            Tensor copy(shape_);
            if (is_contiguous()) {
                std::copy_n(data_.get() + offset_, total_size_, copy.data_.get());
            }
            else {
                T* dest_ptr = copy.data_.get();
                MultiDimCopy(dest_ptr);
            }
            return copy;
        }

        const std::vector<size_t>& shape() const noexcept { return shape_; }
        const std::vector<size_t>& strides() const noexcept { return strides_; }
        size_t size() const noexcept {
            if (!data_) return 0;
            return total_size_;
        }
        size_t ndim() const noexcept { return shape_.size(); }
        size_t bytes() const noexcept { return size() * sizeof(T); }
        T* data() {
            if (!data_) {
                throw std::runtime_error("Tensor is default-constructed (empty)");
            }
            return data_.get() + offset_;
        }
        const T* data() const {
            if (!data_) {
                throw std::runtime_error("Tensor is default-constructed (empty)");
            }
            return data_.get() + offset_;
        }

        bool is_contiguous() const noexcept {
            size_t expected = 1;
            for (size_t i = shape_.size(); i-- > 0;) {
                if (strides_[i] != expected) return false;
                expected *= shape_[i];
            }
            return true;
        }
        std::vector<size_t> broadcast_shapes(const std::vector<size_t>& a, const std::vector<size_t>& b) const {
            std::vector<size_t> res;
            size_t na = a.size(), nb = b.size();
            size_t max_dim = std::max(na, nb);
            res.resize(max_dim);
            for (size_t i = 0; i < max_dim; i++) {
                size_t da = (i < na) ? a[na - 1 - i] : 1;
                size_t db = (i < nb) ? b[nb - 1 - i] : 1;
                if (da != db && da != 1 && db != 1) {
                    throw std::invalid_argument("Shapes are not broadcastable");
                }
                res[max_dim - 1 - i] = std::max(da, db);
            }
            return res;
        }

        Tensor broadcast_to(const std::vector<size_t>& target_shape) const {
            if (shape_ == target_shape) {
                return static_cast<Tensor>(*this);
            }
            std::vector<size_t> new_strides(target_shape.size(), 0);
            int src_dim = static_cast<int>(shape_.size()) - 1;
            int tgt_dim = static_cast<int>(target_shape.size()) - 1;

            while (src_dim >= 0 && tgt_dim >= 0) {
                size_t src_size = shape_[src_dim];
                size_t tgt_size = target_shape[tgt_dim];
                if (src_size == tgt_size) {
                    new_strides[tgt_dim] = strides_[src_dim];
                }
                else if (src_size == 1) {
                    new_strides[tgt_dim] = 0;
                }
                else {
                    throw std::invalid_argument("Cannot broadcast dim " + std::to_string(src_dim) + ": " + std::to_string(src_size) + " -> " + std::to_string(tgt_size));
                }
                --src_dim;
                --tgt_dim;
            }

            while (tgt_dim >= 0) {
                new_strides[tgt_dim] = 0;
                --tgt_dim;
            }

            while (src_dim >= 0) {
                if (shape_[src_dim] != 1) {
                    throw std::invalid_argument("Excess leading dims must be 1");
                }
                --src_dim;
            }

            Tensor view(target_shape, new_strides, data_, offset_);
            if (!view.is_contiguous()) {
                return view.clone();
            }
            return static_cast<Tensor>(std::move(view));
        }

       float* data_gpu() { return gpu_data_.get(); }
        const float* data_gpu() const { return gpu_data_.get(); }

        static Tensor zeros(const std::vector<size_t>& shape) {
            Tensor t(shape);
            std::fill_n(t.data(), t.size(), T(0));
            return t;
        }

        static Tensor ones(const std::vector<size_t>& shape) {
            Tensor t(shape);
            std::fill_n(t.data(), t.size(), T(1));
            return t;
        }

        std::shared_ptr<Tensor<T>> operator+(const Tensor& other) const {
            auto result_shape = broadcast_shapes(shape_, other.shape_);
            auto result = std::make_shared<Tensor<T>>(result_shape);
            if (on_gpu_ && other.on_gpu_) {
                *result = ElementwiseOp(other, GPUOp::kADD);
            }
            else {
                Tensor a = this->contiguous();
                Tensor b = other.contiguous();
                if (a.on_gpu_) a.to_cpu();
                if (b.on_gpu_) b.to_cpu();
                *result = a.ElementwiseBinary(b, std::plus<T>{});
            }
            if (autograd::is_grad_enabled() && (requires_grad_ || other.requires_grad_)) {
                result->set_requires_grad(true);
                autograd::OpRecord rec = {};
                rec.type = autograd::OpType::ADD;
                rec.a = const_cast<Tensor<T>*>(this);
                rec.b = const_cast<Tensor<T>*>(&other);
                rec.out = result.get();
                autograd::push_record(std::move(rec));
            }
            return result;
        }

        [[nodiscard]] std::shared_ptr<Tensor<float>> operator-(const Tensor& other) const {
            auto result_shape = broadcast_shapes(shape_, other.shape_);
            auto result = std::make_shared<Tensor<float>>(result_shape);
            if (on_gpu_ && other.on_gpu_) {
                *result = ElementwiseOp(other, GPUOp::kSUB);
            }
            else {
                Tensor a = this->contiguous();
                Tensor b = other.contiguous();
                if (a.on_gpu_) a.to_cpu();
                if (b.on_gpu_) b.to_cpu();
                *result = a.ElementwiseBinary(b, std::minus<T>{});
            }
            if (autograd::is_grad_enabled() && (requires_grad_ || other.requires_grad_)) {
                result->set_requires_grad(true);
                autograd::OpRecord rec = {};
                rec.type = autograd::OpType::SUB;
                rec.a = const_cast<Tensor<float>*>(this);
                rec.b = const_cast<Tensor<float>*>(&other);
                rec.out = result.get();
                autograd::push_record(std::move(rec));
            }
            return result;
        }

        [[nodiscard]] std::shared_ptr<Tensor<float>> operator*(const Tensor& other) const {
            auto result_shape = broadcast_shapes(shape_, other.shape_);
            auto result = std::make_shared<Tensor<float>>(result_shape);
            if (on_gpu_ && other.on_gpu_) {
                *result = ElementwiseOp(other, GPUOp::kMUL);
            }
            else {
                Tensor a = this->contiguous();
                Tensor b = other.contiguous();
                if (a.on_gpu_) a.to_cpu();
                if (b.on_gpu_) b.to_cpu();
                *result = a.ElementwiseBinary(b, std::multiplies<T>{});
            }
            if (autograd::is_grad_enabled() && (requires_grad_ || other.requires_grad_)) {
                result->set_requires_grad(true);
                autograd::OpRecord rec = {};
                rec.type = autograd::OpType::MUL;
                rec.a = const_cast<Tensor<float>*>(this);
                rec.b = const_cast<Tensor<float>*>(&other);
                rec.out = result.get();
                autograd::push_record(std::move(rec));
            }
            return result;
        }


        [[nodiscard]] Tensor operator*(float scalar) const {
            Tensor result(shape_);
            const T* src = data();
            T* dst = result.data();
            for (size_t i = 0; i < size(); ++i) dst[i] = src[i] * scalar;
            return result;
        }

        [[nodiscard]] friend Tensor operator*(float scalar, const Tensor& t) {
            return t * scalar;
        }

        [[nodiscard]] std::shared_ptr<Tensor<float>> operator/(const Tensor& other) const {
            auto result_shape = broadcast_shapes(shape_, other.shape_);
            auto result = std::make_shared<Tensor<float>>(result_shape);
            if (on_gpu_ && other.on_gpu_) {
                *result = ElementwiseOp(other, GPUOp::kDIV);
            }
            else {
                Tensor a = this->contiguous();
                Tensor b = other.contiguous();
                if (a.on_gpu_) a.to_cpu();
                if (b.on_gpu_) b.to_cpu();
                *result = a.ElementwiseBinary(b, divides_with_check{});
            }
            if (autograd::is_grad_enabled() && (requires_grad_ || other.requires_grad_)) {
                result->set_requires_grad(true);
                autograd::OpRecord rec = {};
                rec.type = autograd::OpType::DIV;
                rec.a = const_cast<Tensor<float>*>(this);
                rec.b = const_cast<Tensor<float>*>(&other);
                rec.out = result.get();
                autograd::push_record(std::move(rec));
            }
            return result;
        }

        [[nodiscard]] Tensor& operator+=(const Tensor& other) {
            add_(other);
            return *this;
        }

        Tensor operator>(float threshold) const noexcept {
            Tensor result(shape_);
            const T* src = data();
            T* dst = result.data();
            for (size_t i = 0; i < size(); ++i) dst[i] = (src[i] > threshold) ? 1.0f : 0.0f;
            return result;
        }
        Tensor operator<=(float threshold) const noexcept {
            Tensor result(shape_);
            const T* src = data();
            T* dst = result.data();
            for (size_t i = 0; i < size(); ++i) dst[i] = (src[i] <= threshold) ? 1.0f : 0.0f;
            return result;
        }

        Tensor& add_(const Tensor& other) {
            auto common = broadcast_shapes(shape_, other.shape_);
            if (common != shape_) {
                throw std::runtime_error("In-place add requires left side to already have broadcast shape");
            }
            Tensor b = other.broadcast_to(shape_);
            T* this_ptr = data();
            const T* b_ptr = b.data();
            for (size_t i = 0; i < size(); ++i) {
                this_ptr[i] += b_ptr[i];
            }
            return *this;
        }

        void zero() { std::fill_n(data(), size(), T(0)); }

        static Tensor from_vector(const std::vector<T>& vec, const std::vector<size_t>& shape) {
            Tensor t(shape);
            if (vec.size() != t.size()) {
                throw std::invalid_argument("from_vector: size mismatch");
            }
            std::copy(vec.begin(), vec.end(), t.data());
            return t;
        }

        std::shared_ptr<Tensor<float>> relu() const {
            std::shared_ptr<Tensor<float>> result;
            if (on_gpu_) {
                Tensor work(*this);
                result = std::make_shared<Tensor<float>>(std::move(work));
                relu_gpu_impl(result->data_gpu(), result->size());
            }
            else {
                Tensor work = contiguous().clone();
                T* p = work.data();
                const size_t n = work.size();
                const size_t vec_width = 8;
                const size_t vec_loops = n / vec_width;
#pragma omp parallel for if (n > OMP_THRESHOLD)
                for (int i = 0; i < static_cast<int>(vec_loops); ++i) {
                    size_t idx = i * vec_width;
                    __m256 v = _mm256_loadu_ps(p + idx);
                    __m256 z = _mm256_setzero_ps();
                    __m256 m = _mm256_cmp_ps(v, z, _CMP_GT_OQ);
                    v = _mm256_and_ps(v, m);
                    _mm256_storeu_ps(p + idx, v);
                }
                for (size_t i = vec_loops * vec_width; i < n; ++i) {
                    if (p[i] < T(0)) p[i] = T(0);
                }
                result = std::make_shared<Tensor<float>>(std::move(work));
            }
            if (autograd::is_grad_enabled()) {
                bool needs_grad = requires_grad_ || (this->has_grad());
                if (needs_grad || true) {
                    result->set_requires_grad(true);
                    autograd::OpRecord rec = {};
                    rec.type = autograd::OpType::RELU;
                    rec.a = const_cast<Tensor<float>*>(this);
                    rec.out = result.get();

                    autograd::push_record(std::move(rec));
                }
            
            }
            return result;
        }

        std::shared_ptr<Tensor<float>> sigmoid() const {
            Tensor result_cpu = this->contiguous().clone();

            if (on_gpu_) {
                result_cpu.to_gpu();
                sigmoid_gpu_impl(result_cpu.data_gpu(), result_cpu.size());
                auto result = std::make_shared<Tensor<float>>(std::move(result_cpu));
                result->on_gpu_ = true;
                if (autograd::is_grad_enabled()) {
                    result->set_requires_grad(true);
                    autograd::OpRecord rec{};
                    rec.type = autograd::OpType::SIGMOID;
                    rec.a = const_cast<Tensor<float>*>(this);
                    rec.out = result.get();
                    autograd::push_record(std::move(rec));
                }
                return result;
            }
            else {
                float* p = result_cpu.data();
                const size_t n = result_cpu.size();
                for (size_t i = 0; i < n; ++i) {
                    p[i] = 1.0f / (1.0f + std::exp(-p[i]));
                }
                auto result = std::make_shared<Tensor<float>>(std::move(result_cpu));
                if (autograd::is_grad_enabled()) {
                    result->set_requires_grad(true);
                    autograd::OpRecord rec{};
                    rec.type = autograd::OpType::SIGMOID;
                    rec.a = const_cast<Tensor<float>*>(this);
                    rec.out = result.get();
                    autograd::push_record(std::move(rec));
                }
                return result;
            }
        }

        std::shared_ptr<Tensor<float>> tanh() const {
            Tensor result_cpu = this->contiguous().clone();

            if (on_gpu_) {
                result_cpu.to_gpu();
                tanh_gpu_impl(result_cpu.data_gpu(), result_cpu.size());
                auto result = std::make_shared<Tensor<float>>(std::move(result_cpu));
                result->on_gpu_ = true;
                if (autograd::is_grad_enabled()) {
                    result->set_requires_grad(true);
                    autograd::OpRecord rec{};
                    rec.type = autograd::OpType::TANH;
                    rec.a = const_cast<Tensor<float>*>(this);
                    rec.out = result.get();
                    autograd::push_record(std::move(rec));
                }
                return result;
            }
            else {
                float* p = result_cpu.data();
                const size_t n = result_cpu.size();
                for (size_t i = 0; i < n; ++i) {
                    p[i] = std::tanh(p[i]);
                }
                auto result = std::make_shared<Tensor<float>>(std::move(result_cpu));
                if (autograd::is_grad_enabled()) {
                    result->set_requires_grad(true);
                    autograd::OpRecord rec{};
                    rec.type = autograd::OpType::TANH;
                    rec.a = const_cast<Tensor<float>*>(this);
                    rec.out = result.get();
                    autograd::push_record(std::move(rec));
                }
                return result;
            }
        }

        std::shared_ptr<Tensor<float>> leaky_relu(float negative_slope = 0.01f) const {
            std::shared_ptr<Tensor<float>> result;
            if (on_gpu_) {
                Tensor work(*this);
                result = std::make_shared<Tensor<float>>(std::move(work));
                leaky_relu_gpu_impl(result->data_gpu(), result->size(), negative_slope);
            }
            else {
                Tensor work = contiguous().clone();
                T* p = work.data();
                const size_t n = work.size();
#pragma omp parallel for if (n > OMP_THRESHOLD)
                for (int i = 0; i < static_cast<int>(n); ++i) {
                    float x = p[i];
                    p[i] = x > 0.0f ? x : negative_slope * x;
                }
                result = std::make_shared<Tensor<float>>(std::move(work));
            }
            if (autograd::is_grad_enabled()) {
                bool needs_grad = requires_grad_ || (this->has_grad());
                if (needs_grad || true) {
                    result->set_requires_grad(true);
                    autograd::OpRecord rec = {};
                    rec.type = autograd::OpType::LEAKY_RELU;
                    rec.a = const_cast<Tensor<float>*>(this);
                    rec.out = result.get();
                    if (rec.type == autograd::OpType::LEAKY_RELU) rec.leaky_slope = negative_slope;
                    autograd::push_record(std::move(rec));
                }
            }
            return result;
        }

        std::shared_ptr<Tensor<float>> add_relu(const Tensor& other) const {
            auto common = broadcast_shapes(shape_, other.shape_);
            Tensor a = this->contiguous().broadcast_to(common);
            Tensor b = other.contiguous().broadcast_to(common);
            auto result = std::make_shared<Tensor<float>>(common);

            a.to_gpu();
            b.to_gpu();
            result->to_gpu();

            fused_binary_activation_gpu_impl(
                a.data_gpu(), b.data_gpu(), result->data_gpu(), result->size(),
                0, 1, 0.0f
            );
            result->on_gpu_ = true;
            return result;
        }

        std::shared_ptr<Tensor<float>> sum() const {


            auto result = std::make_shared<Tensor<float>>(std::vector<size_t>{1});
            T* dst = result->data();
            const T* src = data();
            T sum_val = 0;
            for (size_t i = 0; i < size(); ++i) {
                sum_val += src[i];
            }
            dst[0] = sum_val;

            if (autograd::is_grad_enabled() && requires_grad_) {
                result->set_requires_grad(true);

                autograd::OpRecord rec = {};
                rec.type = autograd::OpType::SUM;
                rec.a = const_cast<Tensor<float>*>(this);
                rec.out = result.get();
                autograd::push_record(std::move(rec));
            }
            return result;
        }

        std::shared_ptr<Tensor<float>> matmul(const Tensor& other) const {
            if (ndim() != 2 || other.ndim() != 2) {
                throw std::invalid_argument("matmul: both tensors must be 2D");
            }
            size_t m = shape_[0];
            size_t k = shape_[1];
            size_t n = other.shape_[1];
            if (k != other.shape_[0]) {
                throw std::invalid_argument("matmul: inner dimensions mismatch");
            }

            auto result = std::make_shared<Tensor<float>>(std::vector<size_t>{m, n});

            Tensor a = this->contiguous();
            Tensor b = other.contiguous();
            a.to_gpu();
            b.to_gpu();
            result->to_gpu();

            cublasHandle_t handle;
            cublasCreate(&handle);
            cublasStatus_t status = math::matmul_float(
                handle,
                static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                a.data_gpu(),
                b.data_gpu(),
                result->data_gpu()
            );
            if (status != CUBLAS_STATUS_SUCCESS) {
                cublasDestroy(handle);
                throw std::runtime_error("matmul: cublas failed");
            }
            cudaError_t sync_err = cudaDeviceSynchronize();
            cublasDestroy(handle);
            if (sync_err != cudaSuccess) {
                throw std::runtime_error(
                    std::string("matmul: cudaDeviceSynchronize failed: ") + cudaGetErrorString(sync_err));
            }

            result->on_gpu_ = true;
            if (autograd::is_grad_enabled()) {
                bool needs_grad = requires_grad_ || (this->has_grad());

                if (needs_grad || true) {
                    result->set_requires_grad(true);
                    autograd::OpRecord rec{};
                    rec.type = autograd::OpType::MATMUL;
                    rec.a = const_cast<Tensor<float>*>(this);
                    rec.b = const_cast<Tensor<float>*>(&other);
                    rec.out = result.get();
                    autograd::push_record(std::move(rec));
                }
            }
            return result;
        }

        void set_requires_grad(bool req) { requires_grad_ = req; }
        bool requires_grad() const { return requires_grad_; }

        bool has_grad() const noexcept {
            return grad_ != nullptr;
        }

        void ensure_grad() {
            if (shape_.empty()) {
            }
            if (!has_grad()) {
                grad_ = std::make_shared<Tensor<float>>(shape_);
                grad_->on_gpu_ = false;
                grad_->gpu_data_.reset();
                grad_->zero();
            }
            else {
                grad_->to_cpu();
            }
        }

        Tensor& grad() {
            ensure_grad();
            return *grad_;
        }

        const Tensor& grad() const {
            if (!has_grad()) {
                throw std::runtime_error("grad() called on tensor that does not have gradient");
            }
            return *grad_;
        }

        void zero_grad() {
            if (has_grad()) {
                grad_->zero();
            }
        }
        
        void backward();
        void to_gpu();
        void to_cpu();
        bool is_gpu() const { return on_gpu_; }

    private:
        std::vector<size_t> shape_;
        std::vector<size_t> strides_;
        std::shared_ptr<T[]> data_;
        size_t offset_;
        size_t total_size_;
        std::unique_ptr<float[], CudaDeleter> gpu_data_;
        bool on_gpu_ = false;
        bool requires_grad_ = false;
        std::shared_ptr<Tensor<float>> grad_;
        
        void ComputeContiguousStrides() {
            strides_.resize(shape_.size());
            size_t stride = 1;
            for (size_t i = shape_.size(); i-- > 0;) {
                strides_[i] = stride;
                stride *= shape_[i];
            }
        }

        struct divides_with_check {
            T operator()(T x, T y) const {
                if (y == T(0)) throw std::runtime_error("Division by zero");
                return x / y;
            }
        };

        template <typename BinaryOp>
        Tensor ElementwiseBinary(const Tensor& other, BinaryOp op) const {
            auto common = broadcast_shapes(shape_, other.shape_);
            Tensor a = this->broadcast_to(common);
            Tensor b = other.broadcast_to(common);
            Tensor result(common);
            if (!a.is_contiguous() || !b.is_contiguous())
                throw std::runtime_error("SIMD requires contiguous tensors");
            const size_t vec_width = 8;
            const size_t n = result.size();
            const size_t vec_loops = n / vec_width;
            T* res_ptr = result.data();
            const T* a_ptr = a.data();
            const T* b_ptr = b.data();

            PerformElementwise(res_ptr, a_ptr, b_ptr, n, vec_loops, op);
            return result;
        }

        template <typename BinaryOp>
        void PerformElementwise(T* res, const T* a, const T* b, size_t n, size_t vec_loops, BinaryOp op) const {
#pragma omp parallel for if (n > OMP_THRESHOLD)
            for (int i = 0; i < static_cast<int>(vec_loops); ++i) {
                size_t idx = i * 8;
                if constexpr (std::is_same_v<BinaryOp, std::plus<T>>) {
                    __m256 av = _mm256_loadu_ps(a + idx);
                    __m256 bv = _mm256_loadu_ps(b + idx);
                    _mm256_storeu_ps(res + idx, _mm256_add_ps(av, bv));
                }
                else if constexpr (std::is_same_v<BinaryOp, std::minus<T>>) {
                    __m256 av = _mm256_loadu_ps(a + idx);
                    __m256 bv = _mm256_loadu_ps(b + idx);
                    _mm256_storeu_ps(res + idx, _mm256_sub_ps(av, bv));
                }
                else if constexpr (std::is_same_v<BinaryOp, std::multiplies<T>>) {
                    __m256 av = _mm256_loadu_ps(a + idx);
                    __m256 bv = _mm256_loadu_ps(b + idx);
                    _mm256_storeu_ps(res + idx, _mm256_mul_ps(av, bv));
                }
                else if constexpr (std::is_same_v<BinaryOp, divides_with_check>) {
                    __m256 av = _mm256_loadu_ps(a + idx);
                    __m256 bv = _mm256_loadu_ps(b + idx);
                    __m256 zm = _mm256_cmp_ps(bv, _mm256_setzero_ps(), _CMP_EQ_OQ);
                    if (!_mm256_testz_ps(zm, zm))
                        throw std::runtime_error("Division by zero in vector");
                    _mm256_storeu_ps(res + idx, _mm256_div_ps(av, bv));
                }
                else {
                    for (size_t j = 0; j < 8; ++j)
                        res[idx + j] = op(a[idx + j], b[idx + j]);
                }
            }
            for (size_t i = vec_loops * 8; i < n; ++i)
                res[i] = op(a[i], b[i]);
        }

        void MultiDimCopy(T* dest) const {
            std::vector<size_t> idx(ndim(), 0);
            for (size_t flat = 0; flat < total_size_; ++flat) {
                size_t src_idx = offset_;
                for (size_t d = 0; d < ndim(); ++d)
                    src_idx += idx[d] * strides_[d];
                *dest++ = data_.get()[src_idx];
                for (size_t d = ndim() - 1; d != static_cast<size_t>(-1); --d) {
                    if (++idx[d] < shape_[d]) break;
                    idx[d] = 0;
                }
            }
        }

        enum class GPUOp { kADD, kSUB, kMUL, kDIV };

        Tensor ElementwiseOp(const Tensor& other, GPUOp op) const {
            auto common = broadcast_shapes(shape_, other.shape_);
            if (on_gpu_ && other.on_gpu_) {
                Tensor a = this->contiguous().broadcast_to(common);
                Tensor b = other.contiguous().broadcast_to(common);
                a.to_gpu();
                b.to_gpu();
                Tensor result(common);
                result.to_gpu();
                switch (op) {
                    case GPUOp::kADD:
                        add_gpu_impl(a.data_gpu(), b.data_gpu(), result.data_gpu(), result.size());
                        break;
                    case GPUOp::kSUB:
                        sub_gpu_impl(a.data_gpu(), b.data_gpu(), result.data_gpu(), result.size());
                        break;
                    case GPUOp::kMUL:
                        mul_gpu_impl(a.data_gpu(), b.data_gpu(), result.data_gpu(), result.size());
                        break;
                    case GPUOp::kDIV:
                        div_gpu_impl(a.data_gpu(), b.data_gpu(), result.data_gpu(), result.size());
                        break;
                }
                result.on_gpu_ = true;
                return result;
            }
            switch (op) {
                case GPUOp::kADD: return ElementwiseBinary(other, std::plus<T>{});
                case GPUOp::kSUB: return ElementwiseBinary(other, std::minus<T>{});
                case GPUOp::kMUL: return ElementwiseBinary(other, std::multiplies<T>{});
                case GPUOp::kDIV: return ElementwiseBinary(other, divides_with_check{});
            }
            return Tensor{};
        }
    };
}