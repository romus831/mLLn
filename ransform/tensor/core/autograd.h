#pragma once

#include <functional>
#include <vector>
#include <memory>
#include "concepts.h"

namespace MNNL {

    template<ArithmeticType T>
    class Tensor;
}

namespace MNNL {
    namespace autograd {
        enum class OpType { ADD, SUB, MUL, DIV, RELU, LEAKY_RELU, SIGMOID, TANH, MATMUL, SUM };

        struct OpRecord {
            OpType type;
            std::weak_ptr<Tensor<float>> a;
            std::weak_ptr<Tensor<float>> b;
            std::weak_ptr<Tensor<float>> out;
            float leaky_slope = 0.01f;
        };
        template<typename T>
        static std::shared_ptr<Tensor<T>>lock_week(const std::weak_ptr<Tensor<T>> wp) {
            if (auto sp = wp.lock())return sp;
            return nullptr;
        }

        extern thread_local std::vector<OpRecord> tape;
        extern thread_local bool grad_enabled;

        inline bool is_grad_enabled() { return grad_enabled; }
        inline void set_grad_enabled(bool enabled) { grad_enabled = enabled; }

        extern void push_record(const OpRecord& rec);
        void clear_tape();

        void backward_add(const OpRecord& rec);
        void backward_sub(const OpRecord& rec);
        void backward_mul(const OpRecord& rec);
        void backward_div(const OpRecord& rec);
        void backward_relu(const OpRecord& rec);
        void backward_leaky_relu(const OpRecord& rec);
        void backward_sigmoid(const OpRecord& rec);
        void backward_tanh(const OpRecord& rec);
        void backward_matmul(const OpRecord& rec);
        void backward_sum(const OpRecord& rec);
        extern void backward(Tensor<float>& loss);

#define LOCK_WEAK(var, weakPtr) auto var = (weakPtr).lock()
#define LOCK_WEAK_OR_RETURN(var, weakPtr) \
    auto var = (weakPtr).lock(); \
    if (!(var)) return

#define LOCK_WEAK_OR_CONTINUE(var, weakPtr) \
    auto var = (weakPtr).lock(); \
    if (!(var)) continue
    }
}