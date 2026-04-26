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
            LOCK_WEAK(out, rec.out);
            if (out) {
                out->to_cpu();
                out->grad().to_cpu();
            }
            LOCK_WEAK(a, rec.a);
            if (a) a->to_cpu();
            LOCK_WEAK(b, rec.b);
            if (b) b->to_cpu();
            if (a && a->requires_grad()) {
                a->ensure_grad();
                a->grad().to_cpu();
            }
            if (b && b->requires_grad()) {
                b->ensure_grad();
                b->grad().to_cpu();
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
            LOCK_WEAK_OR_RETURN(out, rec.out);
            prepare_backward_inputs(rec);;
            Tensor<float> grad_out_cpu = out->grad().contiguous();
            LOCK_WEAK(a, rec.a);
            if (a && a->requires_grad()) {
                Tensor<float> grad_a = reduce_grad(grad_out_cpu, a->shape());
                a->grad() += grad_a;
            }
            LOCK_WEAK(b, rec.b);
            if (b && b->requires_grad()) {
                Tensor<float> grad_b = reduce_grad(grad_out_cpu, b->shape());
                b->grad() += grad_b;
            }
        }

        void backward_sub(const OpRecord& rec) {
            LOCK_WEAK_OR_RETURN(out, rec.out);
            prepare_backward_inputs(rec);
            Tensor<float> grad_out_cpu = out->grad().contiguous();
            LOCK_WEAK(a, rec.a);
            if (a && a->requires_grad()) {
                Tensor<float> grad_a = reduce_grad(grad_out_cpu, a->shape());
                a->grad() += grad_a;
            }
            LOCK_WEAK(b, rec.b);
            if (b && b->requires_grad()) {
                Tensor<float> grad_b_full = grad_out_cpu * (-1.0f);
                Tensor<float> grad_b = reduce_grad(grad_b_full, b->shape());
                b->grad() += grad_b;
            }
        }

        void backward_matmul(const OpRecord& rec) { 
            LOCK_WEAK_OR_RETURN(out, rec.out);
            prepare_backward_inputs(rec);
            const Tensor<float>& grad_out = out->grad();
            Tensor<float> grad_contig = grad_out.contiguous();
            LOCK_WEAK(a, rec.a);
            Tensor<float> A_contig = a->contiguous();
            LOCK_WEAK(b, rec.b);
            Tensor<float> B_contig = b->contiguous(); 
            if (a && a->requires_grad()) { 
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
                a->grad() += dA; 
            } 
            if (b && b->requires_grad()) {
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
                b->grad() += dB;
            } 
        }

        void backward_div(const OpRecord& rec) {
            LOCK_WEAK_OR_RETURN(out, rec.out);
            prepare_backward_inputs(rec);
            const Tensor<float> grad_out = out->grad().contiguous();
            LOCK_WEAK(a, rec.a);
            LOCK_WEAK(b, rec.b);
            if (a && a->requires_grad()) {
                auto grad_a_sp = grad_out / (*b);
                a->grad() += reduce_grad(*grad_a_sp, a->shape());
            }
            if (b && b->requires_grad()) {
                const Tensor<float> aa = *a;
                const Tensor<float> bb = *b;
                auto b_squared = bb * bb;
                auto numerator = grad_out * aa;
                auto fraction = *numerator / *b_squared;
                Tensor<float> grad_b = *fraction * (-1.0f);
                b->grad() += reduce_grad(grad_b, b->shape());
            }
        }

        void backward_relu(const OpRecord& rec) {
            LOCK_WEAK(out, rec.out);
            LOCK_WEAK(a, rec.a);
            LOCK_WEAK(b, rec.b);
            if (!out || !a) return;
            prepare_backward_inputs(rec);
            Tensor<float> mask = (*a) > 0.0f;
            Tensor<float> grad_out_cpu = out->grad().contiguous();
            auto grad = grad_out_cpu * mask;
            a->grad() += *grad;
        }

        void backward_leaky_relu(const OpRecord& rec) {
            LOCK_WEAK(out, rec.out);
            LOCK_WEAK(a, rec.a);
            LOCK_WEAK(b, rec.b);
            if (!out || !a) return;
            prepare_backward_inputs(rec);
            Tensor<float> mask_pos = (*a) > 0.0f;
            Tensor<float> mask_neg = (*a) <= 0.0f;
            Tensor<float> coeff = *(mask_pos + (rec.leaky_slope * mask_neg));
            Tensor<float> grad_out_cpu = out->grad().contiguous();
            auto grad_a = grad_out_cpu * coeff;
            a->grad() += *grad_a;
            
        }

        void backward_sigmoid(const OpRecord& rec) {
            LOCK_WEAK(out, rec.out);
            LOCK_WEAK(a, rec.a);
            if (!out || !a) return;
            prepare_backward_inputs(rec);

            Tensor<float> grad_out_cpu = out->grad().contiguous();

            const Tensor<float>& oout = *out;
            const Tensor<float>& g_out = out->grad();

            Tensor<float> grad_input(oout.shape());
            grad_input.zero();

            const float* o = oout.data();
            const float* g = grad_out_cpu.data();
            float* gi = grad_input.data();
            size_t n = oout.size();

            for (size_t i = 0; i < n; ++i) {
                float s = o[i];
                gi[i] = g[i] * s * (1.0f - s);
            }

            a->grad() += grad_input;
        }

        void backward_tanh(const OpRecord& rec) {
            LOCK_WEAK(out, rec.out);
            LOCK_WEAK(a, rec.a);
            LOCK_WEAK(b, rec.b);
            if (!out || !a) return;
            prepare_backward_inputs(rec);
            Tensor<float> grad_out_cpu = out->grad().contiguous();
            const Tensor<float>& output = *out;
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
            a->grad() += grad_input;
        }

        void backward_mul(const OpRecord& rec) {
            LOCK_WEAK(a, rec.a);
            LOCK_WEAK(b, rec.b);
            LOCK_WEAK_OR_RETURN(out, rec.out);
            prepare_backward_inputs(rec);
            Tensor<float> grad_out_cpu = out->grad().contiguous();
            if (a && a->requires_grad()) {
                if (!b) {
                    std::cerr << "ERROR: rec.b is null in mul backward" << std::endl;
                    return;
                }
                auto grad_a_ptr = grad_out_cpu * (*b);
                a->grad() += *grad_a_ptr;
            }
            if (b && b->requires_grad()) {
                auto grad_b_ptr = grad_out_cpu * (*a);
                b->grad() += *grad_b_ptr;
            }

        }

        void backward_sum(const OpRecord& rec) {
            LOCK_WEAK(out, rec.out);
            LOCK_WEAK(a, rec.a);
            LOCK_WEAK(b, rec.b);
            if (!a || !out) return;
            prepare_backward_inputs(rec);
            Tensor<float> grad_out_cpu = out->grad().contiguous();
            if (grad_out_cpu.size() == 0) return;
            float grad_val = grad_out_cpu.data()[0];
            Tensor<float> ones = Tensor<float>::ones(a->shape());
            Tensor<float> scaled = grad_val * ones;
            a->grad() += scaled;
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

                if (static_cast<int>(rec.type) == 0 && rec.out.expired()) {
                    std::cout << "    Skipping garbage record (type "
                        << static_cast<int>(rec.type) << ")" << std::endl;
                    continue;
                }

                auto a_check = rec.a.lock();
                auto b_check = rec.b.lock();
                auto out_check = rec.out.lock();

                BACKWARD_LOG("  a=" << (a_check ? "non-null" : "NULL") << ", requires_grad=" << (a_check ? a_check->requires_grad() : false));
                BACKWARD_LOG("  b=" << (b_check ? "non-null" : "NULL") << ", requires_grad=" << (b_check ? b_check->requires_grad() : false));
                BACKWARD_LOG("  out=" << (out_check ? "non-null" : "NULL"));

                if (out_check) {
                    std::string os;
                    for (size_t s : out_check->shape()) os += std::to_string(s) + " ";
                    BACKWARD_LOG("out shape: " << os);

                    std::string so;
                    for (size_t s : out_check->grad().shape()) so += std::to_string(s) + " ";
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
                    //clear_tape();
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