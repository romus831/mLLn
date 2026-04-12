#pragma once

#include <functional>
#include <vector>
#include <memory>

namespace MNNL {
    template <typename T> class Tensor;
}

namespace MNNL {
    namespace autograd {
        enum class OpType { ADD, SUB, MUL, DIV, RELU, LEAKY_RELU, SIGMOID, TANH, MATMUL, SUM };

        struct OpRecord {
            OpType type;
            Tensor<float>* a = nullptr;
            Tensor<float>* b = nullptr;
            Tensor<float>* out = nullptr;
            float leaky_slope = 0.01f;
        };

        // ТОЛЬКО ОБЪЯВЛЕНИЯ:
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
    }
}