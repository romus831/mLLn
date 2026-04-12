#include "autograd.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "tensor.h"
#include <stdexcept>

#define DEBUG_BACKWARD 0

#if DEBUG_BACKWARD
#define BACKWARD_LOG(msg) std::cerr << "[BACKWARD] " << msg << std::endl
#define BACKWARD_LOG_VAL(msg, val) std::cerr << "[BACKWARD] " << msg << " = " << val << std::endl
#else
#define BACKWARD_LOG(msg)
#define BACKWARD_LOG_VAL(msg, val)
#endif

namespace MNNL {
    namespace autograd {

        thread_local std::vector<OpRecord> tape;
        thread_local bool grad_enabled = true;

        void clear_tape() {
            tape.clear();
        }

        void push_record(const OpRecord& rec) {
            if (grad_enabled) {
                tape.push_back(rec);
            }
        }

        inline void prepare_backward_inputs(const OpRecord& rec) {
            if (rec.out) {
                rec.out->to_cpu();
                rec.out->grad().to_cpu();
            }
            if (rec.a) rec.a->to_cpu();
            if (rec.b) rec.b->to_cpu();
            if (rec.a && rec.a->requires_grad()) {
                rec.a->ensure_grad();
                rec.a->grad().to_cpu();
            }
            if (rec.b && rec.b->requires_grad()) {
                rec.b->ensure_grad();
                rec.b->grad().to_cpu();
            }
        }

        Tensor<float> reduce_grad(const Tensor<float>& grad_out, const std::vector<size_t>& target_shape) {
            if (grad_out.shape() == target_shape) {
                return grad_out.clone();
            }
            std::vector<size_t> reduce_dims;
            size_t grad_ndim = grad_out.ndim();
            size_t target_ndim = target_shape.size();
            int offset = static_cast<int>(grad_ndim) - static_cast<int>(target_ndim);

            for (size_t i = 0; i < target_ndim; ++i) {
                size_t grad_dim = (offset >= 0) ? i + offset : i;
                if (target_shape[i] == 1 && grad_out.shape()[grad_dim] > 1) {
                    reduce_dims.push_back(grad_dim);
                }
            }
            if (reduce_dims.empty()) {
                return grad_out.clone();
            }
            Tensor<float> result(target_shape);
            result.zero();
            std::vector<size_t> idx_grad(grad_ndim, 0);
            std::vector<size_t> idx_res(target_ndim, 0);
            size_t total = grad_out.size();

            for (size_t flat = 0; flat < total; ++flat) {
                size_t tmp = flat;
                for (size_t d = grad_ndim; d-- > 0; ) {
                    idx_grad[d] = tmp % grad_out.shape()[d];
                    tmp /= grad_out.shape()[d];
                }
                for (size_t d = 0; d < target_ndim; ++d) {
                    size_t grad_dim = (offset >= 0) ? d + offset : d;
                    if (std::find(reduce_dims.begin(), reduce_dims.end(), grad_dim) != reduce_dims.end()) {
                        idx_res[d] = 0;
                    }
                    else {
                        idx_res[d] = idx_grad[grad_dim];
                    }
                }
                size_t res_flat = 0;
                size_t stride = 1;
                for (size_t d = target_ndim; d-- > 0; ) {
                    res_flat += idx_res[d] * stride;
                    stride *= target_shape[d];
                }
                result.data()[res_flat] += grad_out.data()[flat];
            }
            return result;
        }

        void backward_add(const OpRecord& rec) {
            if (!rec.out) return;
            prepare_backward_inputs(rec);
            Tensor<float> grad_out_cpu = rec.out->grad().contiguous();

            if (rec.a && rec.a->requires_grad()) {
                Tensor<float> grad_a = reduce_grad(grad_out_cpu, rec.a->shape());
                rec.a->grad() += grad_a;
            }
            if (rec.b && rec.b->requires_grad()) {
                Tensor<float> grad_b = reduce_grad(grad_out_cpu, rec.b->shape());
                rec.b->grad() += grad_b;
            }
        }

        void backward_sub(const OpRecord& rec) {
            if (!rec.out) return;
            prepare_backward_inputs(rec);
            Tensor<float> grad_out_cpu = rec.out->grad().contiguous();

            if (rec.a && rec.a->requires_grad()) {
                Tensor<float> grad_a = reduce_grad(grad_out_cpu, rec.a->shape());
                rec.a->grad() += grad_a;
            }
            if (rec.b && rec.b->requires_grad()) {
                Tensor<float> grad_b_full = grad_out_cpu * (-1.0f);
                Tensor<float> grad_b = reduce_grad(grad_b_full, rec.b->shape());
                rec.b->grad() += grad_b;
            }
        }

        void backward_matmul(const OpRecord& rec) { 
            if (!rec.out) return; prepare_backward_inputs(rec);
            const Tensor<float>& grad_out = rec.out->grad();
            Tensor<float> grad_contig = grad_out.contiguous();
            Tensor<float> A_contig = rec.a->contiguous();
            Tensor<float> B_contig = rec.b->contiguous(); 
            if (rec.a && rec.a->requires_grad()) { 
                size_t m = A_contig.shape()[0]; 
                size_t k = A_contig.shape()[1]; 
                size_t n = B_contig.shape()[1]; 
                Tensor<float> B_T = B_contig.transpose(); 
                Tensor<float> B_T_contig = B_T.contiguous();
                Tensor<float> dA({ m, k }); dA.zero();
                const float* grad_data = grad_contig.data();
                const float* B_T_data = B_T_contig.data();
                float* dA_data = dA.data(); 
                for (size_t i = 0; i < m; ++i) { 
                    for (size_t j = 0; j < k; ++j) { 
                        float sum = 0.0f; 
                        for (size_t p = 0; p < n; ++p) {
                            sum += grad_data[i * n + p] * B_T_data[p * k + j];
                        } 
                        dA_data[i * k + j] = sum;
                    }
                } 
                rec.a->grad() += dA; 
            } 
            if (rec.b && rec.b->requires_grad()) {
                size_t m = A_contig.shape()[0]; 
                size_t k = A_contig.shape()[1]; 
                size_t n = B_contig.shape()[1]; 
                Tensor<float> A_T = A_contig.transpose();
                Tensor<float> A_T_contig = A_T.contiguous();
                Tensor<float> dB({ k, n }); dB.zero(); 
                const float* A_T_data = A_T_contig.data(); 
                const float* grad_data = grad_contig.data(); 
                float* dB_data = dB.data(); 
                for (size_t i = 0; i < k; ++i) { 
                    for (size_t j = 0; j < n; ++j) { 
                        float sum = 0.0f;
                        for (size_t p = 0; p < m; ++p) {
                            sum += A_contig.data()[p * k + i] * grad_contig.data()[p * n + j];
                        } 
                        dB_data[i * n + j] = sum; 
                    } 
                } 
                rec.b->grad() += dB;
            } 
        }

        void backward_div(const OpRecord& rec) {
            if (!rec.out) return;
            prepare_backward_inputs(rec);
            const Tensor<float> grad_out = rec.out->grad().contiguous();
            if (rec.a && rec.a->requires_grad()) {
                auto grad_a_sp = grad_out / (*rec.b);
                rec.a->grad() += reduce_grad(*grad_a_sp, rec.a->shape());
            }
            if (rec.b && rec.b->requires_grad()) {
                const Tensor<float> a = *rec.a;
                const Tensor<float> b = *rec.b;
                auto b_squared = b * b;
                auto numerator = grad_out * a;
                auto fraction = *numerator / *b_squared;
                Tensor<float> grad_b = *fraction * (-1.0f);
                rec.b->grad() += reduce_grad(grad_b, rec.b->shape());
            }
        }

        void backward_relu(const OpRecord& rec) {
            if (!rec.out || !rec.a) return;
            prepare_backward_inputs(rec);
            Tensor<float> mask = (*rec.a) > 0.0f;
            Tensor<float> grad_out_cpu = rec.out->grad().contiguous();
            auto grad = grad_out_cpu * mask;
            rec.a->grad() += *grad;
        }

        void backward_leaky_relu(const OpRecord& rec) {
            if (!rec.out || !rec.a) return;
            prepare_backward_inputs(rec);
            Tensor<float> mask_pos = (*rec.a) > 0.0f;
            Tensor<float> mask_neg = (*rec.a) <= 0.0f;
            Tensor<float> coeff = *(mask_pos + (rec.leaky_slope * mask_neg));
            Tensor<float> grad_out_cpu = rec.out->grad().contiguous();
            auto grad_a = grad_out_cpu * coeff;
            rec.a->grad() += *grad_a;
            
        }

        void backward_sigmoid(const OpRecord& rec) {
            if (!rec.out || !rec.a) return;
            prepare_backward_inputs(rec);

            Tensor<float> grad_out_cpu = rec.out->grad().contiguous();

            const Tensor<float>& out = *rec.out;
            const Tensor<float>& g_out = rec.out->grad();

            Tensor<float> grad_input(out.shape());
            grad_input.zero();

            const float* o = out.data();
            const float* g = grad_out_cpu.data();
            float* gi = grad_input.data();
            size_t n = out.size();

            for (size_t i = 0; i < n; ++i) {
                float s = o[i];
                gi[i] = g[i] * s * (1.0f - s);
            }

            rec.a->grad() += grad_input;
        }

        void backward_tanh(const OpRecord& rec) {
            if (!rec.out || !rec.a) return;
            prepare_backward_inputs(rec);
            Tensor<float> grad_out_cpu = rec.out->grad().contiguous();
            const Tensor<float>& output = *rec.out;
            Tensor<float> grad_input(output.shape());
            grad_input.zero();
            const float* out_data = output.data();
            const float* gout_data = grad_out_cpu.data();
            float* gin_data = grad_input.data();
            size_t n = output.size();
            for (size_t i = 0; i < n; ++i) {
                float t = out_data[i];
                gin_data[i] = gout_data[i] * (1.0f - t * t);
            }
            rec.a->grad() += grad_input;
        }

        void backward_mul(const OpRecord& rec) {
            if (!rec.out) return;
            prepare_backward_inputs(rec);
            Tensor<float> grad_out_cpu = rec.out->grad().contiguous();
            if (rec.a && rec.a->requires_grad()) {
                if (!rec.b) {
                    std::cerr << "ERROR: rec.b is null in mul backward" << std::endl;
                    return;
                }
                auto grad_a_ptr = grad_out_cpu * (*rec.b);
                rec.a->grad() += *grad_a_ptr;
            }
            if (rec.b && rec.b->requires_grad()) {
                auto grad_b_ptr = grad_out_cpu * (*rec.a);
                rec.b->grad() += *grad_b_ptr;
            }
        }

        void backward_sum(const OpRecord& rec) {
            if (!rec.a || !rec.out) return;
            prepare_backward_inputs(rec);
            Tensor<float> grad_out_cpu = rec.out->grad().contiguous();
            if (grad_out_cpu.size() == 0) return;
            float grad_val = grad_out_cpu.data()[0];
            Tensor<float> ones = Tensor<float>::ones(rec.a->shape());
            Tensor<float> scaled = grad_val * ones;
            rec.a->grad() += scaled;
        }

        void backward(Tensor<float>& loss) {

            bool old_state = grad_enabled;
            grad_enabled = false;

            BACKWARD_LOG("=== Starting backward ===");
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                BACKWARD_LOG("CUDA error before op: " << cudaGetErrorString(err));
            }
            if (!loss.requires_grad()) {
                throw std::runtime_error("backward called on tensor that does not require grad");
            }

            std::string shape_str;
            for (size_t s : loss.shape()) shape_str += std::to_string(s) + " ";
            BACKWARD_LOG("Loss shape: " << shape_str);
            BACKWARD_LOG_VAL("Tape size", tape.size());

            loss.to_cpu();
            BACKWARD_LOG("Loss moved to CPU");
            loss.ensure_grad();
            loss.grad().zero();
            BACKWARD_LOG("Loss grad zeroed");

            Tensor<float> ones = Tensor<float>::ones(loss.shape());
            loss.grad() += ones;
            BACKWARD_LOG("Initial grad set to ones");

            auto loss_shape = loss.shape();
            if (loss_shape.size() != 1 || loss_shape[0] != 1) {
                std::cerr << "WARNING: loss shape is not scalar! size=" << loss_shape.size()
                    << ", dim0=" << (loss_shape.size() > 0 ? loss_shape[0] : 0) << std::endl;
            }

            for (auto it = tape.rbegin(); it != tape.rend(); ++it) {
                const OpRecord& rec = *it;

                if (static_cast<int>(rec.type) == 0 && !rec.out) {
                    std::cout << "    Skipping garbage record (type "
                        << static_cast<int>(rec.type) << ")" << std::endl;
                    continue;
                }

                BACKWARD_LOG("--- Processing op type " << static_cast<int>(rec.type) << " ---");
                BACKWARD_LOG("  a=" << (rec.a ? "non-null" : "NULL") << ", requires_grad=" << (rec.a ? rec.a->requires_grad() : false));
                BACKWARD_LOG("  b=" << (rec.b ? "non-null" : "NULL") << ", requires_grad=" << (rec.b ? rec.b->requires_grad() : false));
                BACKWARD_LOG("  out=" << (rec.out ? "non-null" : "NULL"));

                if (rec.out) {

                    std::string os;
                    for (size_t s : rec.out->shape()) os += std::to_string(s) + " ";
                    BACKWARD_LOG("out shape: " << os);

                    std::string so;
                    for (size_t s : rec.out->grad().shape()) so += std::to_string(s) + " ";
                    BACKWARD_LOG("out grad shape: " << so);
                }

                try {
                    switch (rec.type) {
                    case OpType::ADD:      backward_add(rec); break;
                    case OpType::SUB:      backward_sub(rec); break;
                    case OpType::MUL:      backward_mul(rec); break;
                    case OpType::DIV:      backward_div(rec); break;
                    case OpType::RELU:     backward_relu(rec); break;
                    case OpType::LEAKY_RELU: backward_leaky_relu(rec); break;
                    case OpType::SIGMOID:  backward_sigmoid(rec); break;
                    case OpType::TANH:     backward_tanh(rec); break;
                    case OpType::MATMUL:   backward_matmul(rec); break;
                    case OpType::SUM:      backward_sum(rec); break;
                    default:
                        BACKWARD_LOG("Unknown op type " << static_cast<int>(rec.type));
                        break;
                    }
                }
                catch (const std::exception& e) {
                    BACKWARD_LOG("Exception in backward op: " << e.what());
                    throw;
                }
                catch (...) {
                    BACKWARD_LOG("Unknown exception in backward op");
                    throw;
                }
                BACKWARD_LOG("--- Op finished ---");
            }

            grad_enabled = old_state;
            clear_tape();
            BACKWARD_LOG("=== Backward completed successfully ===");
        }
    }
}