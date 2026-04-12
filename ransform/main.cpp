#include "tensor/tensor.h"
#include "math/gemm.h"        // добавляем
#include <iostream>
#include <iomanip>
#include <windows.h>
#include <cublas_v2.h>        // для cublasHandle_t

int main() {
    SetConsoleOutputCP(CP_UTF8);
    try {
        std::cout << "=== Тест активаций на GPU ===\n" << std::endl;

        // Создаём тензор с отрицательными и положительными значениями
        std::vector<size_t> shape = { 8 };
        MNNL::Tensor<float> x(shape);

        float* data = x.data();
        for (size_t i = 0; i < x.size(); ++i) {
            data[i] = static_cast<float>(i) - 3.5f;  // от -3.5 до 4.5
        }

        std::cout << "Исходный тензор:" << std::endl;
        for (size_t i = 0; i < x.size(); ++i) {
            std::cout << std::fixed << std::setprecision(2) << data[i] << " ";
        }
        std::cout << "\n\n";

        // === Переводим на GPU ===
        std::cout << "--- Переводим на GPU ---" << std::endl;
        x.to_gpu();
        std::cout << "Тензор успешно перемещён на GPU. on_gpu = "
            << (x.is_gpu() ? "true" : "false") << "\n\n";

        // === ReLU на GPU ===
        std::cout << "--- ReLU на GPU ---" << std::endl;
        auto relu_result = x.relu();
        relu_result->to_cpu();

        std::cout << "ReLU результат: ";
        const float* r = relu_result->data();
        for (size_t i = 0; i < relu_result->size(); ++i) {
            std::cout << std::fixed << std::setprecision(2) << r[i] << " ";
        }
        std::cout << "\n\n";

        // === Sigmoid на GPU ===
        std::cout << "--- Sigmoid на GPU ---" << std::endl;
        auto sig_result = x.sigmoid();
        sig_result->to_cpu();

        std::cout << "Sigmoid результат: ";
        const float* s = sig_result->data();
        for (size_t i = 0; i < sig_result->size(); ++i) {
            std::cout << std::fixed << std::setprecision(4) << s[i] << " ";
        }
        std::cout << "\n\n";

        // === Tanh на GPU ===
        std::cout << "--- Tanh на GPU ---" << std::endl;
        auto tanh_result = x.tanh();
        tanh_result->to_cpu();

        std::cout << "Tanh результат: ";
        const float* t = tanh_result->data();
        for (size_t i = 0; i < tanh_result->size(); ++i) {
            std::cout << std::fixed << std::setprecision(4) << t[i] << " ";
        }
        std::cout << "\n\n";

        std::cout << "--- relu_leaky на GPU ---" << std::endl;
        auto lr_result = x.leaky_relu();
        lr_result->to_cpu();

        std::cout << "relu_leaky результат: ";
        const float* lr = lr_result->data();
        for (size_t i = 0; i < lr_result->size(); ++i) {
            std::cout << std::fixed << std::setprecision(4) << lr[i] << " ";
        }
        std::cout << "\n\n";

        // === Дополнительный тест: ReLU на отрицательных значениях ===
        std::cout << "--- Дополнительный тест (ReLU должен занулить отрицательные) ---" << std::endl;
        MNNL::Tensor<float> neg(shape);
        for (size_t i = 0; i < neg.size(); ++i) {
            neg.data()[i] = -static_cast<float>(i + 1);  // все отрицательные
        }

        neg.to_gpu();
        auto relu_neg = neg.relu();
        relu_neg->to_cpu();

        std::cout << "ReLU(отрицательные): ";
        for (size_t i = 0; i < relu_neg->size(); ++i) {
            std::cout << relu_neg->data()[i] << " ";
        }
        std::cout << "\n\n";

        std::cout << "Все тесты активаций завершены успешно!" << std::endl;


        std::cout << "=== Тест бинарных операций на GPU ===\n";
        MNNL::Tensor<float> A({ 8 });
        MNNL::Tensor<float> B({ 8 });
        for (size_t i = 0; i < 8; ++i) {
            A.data()[i] = i + 1.0f;
            B.data()[i] = i + 1.0f;
        }
        A.to_gpu();
        B.to_gpu();

        auto C_add = A + B;
        C_add->to_cpu();
        std::cout << "A + B: ";
        for (size_t i = 0; i < C_add->size(); ++i) std::cout << C_add->data()[i] << " ";
        std::cout << "\n";

        auto C_sub = A - B;
        C_sub->to_cpu();
        std::cout << "A - B: ";
        for (size_t i = 0; i < C_sub->size(); ++i) std::cout << C_sub->data()[i] << " ";
        std::cout << "\n";

        auto C_mul = A * B;
        C_mul->to_cpu();
        std::cout << "A * B: ";
        for (size_t i = 0; i < C_mul->size(); ++i) std::cout << C_mul->data()[i] << " ";
        std::cout << "\n";

        auto C_div = A / B;
        C_div->to_cpu();
        std::cout << "A / B: ";
        for (size_t i = 0; i < C_div->size(); ++i) std::cout << C_div->data()[i] << " ";
        std::cout << "\n";

        std::cout << "=== Тест add_relu (fused) ===\n";
        auto C_add_relu = A.add_relu(B);
        C_add_relu->to_cpu();
        std::cout << "A + B then ReLU: ";
        for (size_t i = 0; i < C_add_relu->size(); ++i) std::cout << C_add_relu->data()[i] << " ";
        std::cout << "\n";
        std::cout << "\n=== Тест матричного умножения (GEMM) ===\n";

        // Размеры: A: 3x2, B: 2x4, результат C: 3x4
        const size_t m = 3, k = 2, n = 4;
        std::vector<size_t> shape_A = { m, k };
        std::vector<size_t> shape_B = { k, n };
        std::vector<size_t> shape_C = { m, n };

        MNNL::Tensor<float> A_mat(shape_A);
        MNNL::Tensor<float> B_mat(shape_B);
        MNNL::Tensor<float> C_mat(shape_C);

        // Заполняем A и B тестовыми значениями (row-major)
        // A = [1 2; 3 4; 5 6]
        float* a_data = A_mat.data();
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < k; ++j) {
                a_data[i * k + j] = static_cast<float>(i * k + j + 1);
            }
        }

        // B = [1 0 0 0; 0 1 0 0]  (единичная матрица 2x4, но только первые два столбца единичные)
        float* b_data = B_mat.data();
        for (size_t i = 0; i < k; ++i) {
            for (size_t j = 0; j < n; ++j) {
                b_data[i * n + j] = (i == j) ? 1.0f : 0.0f;
            }
        }

        // Вывод исходных матриц (CPU)
        std::cout << "Матрица A (" << m << "x" << k << "):\n";
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < k; ++j) {
                std::cout << std::setw(4) << a_data[i * k + j] << " ";
            }
            std::cout << "\n";
        }

        std::cout << "\nМатрица B (" << k << "x" << n << "):\n";
        for (size_t i = 0; i < k; ++i) {
            for (size_t j = 0; j < n; ++j) {
                std::cout << std::setw(4) << b_data[i * n + j] << " ";
            }
            std::cout << "\n";
        }

        // Переводим на GPU
        A_mat.to_gpu();
        B_mat.to_gpu();
        C_mat.to_gpu();

        // Создаём дескриптор cuBLAS
        cublasHandle_t handle;
        cublasCreate(&handle);

        // Вычисляем C = A * B
        cublasStatus_t status = MNNL::math::matmul_float(
            handle,
            static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
            A_mat.data_gpu(),
            B_mat.data_gpu(),
            C_mat.data_gpu()
        );

        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("matmul_float failed");
        }

        // Синхронизируем и копируем результат на CPU
        cudaDeviceSynchronize();
        C_mat.to_cpu();

        std::cout << "\nРезультат C = A * B (" << m << "x" << n << "):\n";
        const float* c_data = C_mat.data();
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n; ++j) {
                std::cout << std::setw(6) << std::fixed << std::setprecision(2) << c_data[i * n + j] << " ";
            }
            std::cout << "\n";
        }

        // Освобождаем дескриптор
        cublasDestroy(handle);

        std::cout << "\nВсе тесты (включая GEMM) успешно завершены!" << std::endl;

        std::cout << "\n=== Тест Tensor::matmul ===\n";
        MNNL::Tensor<float> A_matmul({ 3, 2 });
        MNNL::Tensor<float> B_matmul({ 2, 4 });
        // заполнение A и B, как раньше
        A_matmul.data()[0] = 1; A_matmul.data()[1] = 2;
        A_matmul.data()[2] = 3; A_matmul.data()[3] = 4;
        A_matmul.data()[4] = 5; A_matmul.data()[5] = 6;

        B_matmul.data()[0] = 1; B_matmul.data()[1] = 0; B_matmul.data()[2] = 0; B_matmul.data()[3] = 0;
        B_matmul.data()[4] = 0; B_matmul.data()[5] = 1; B_matmul.data()[6] = 0; B_matmul.data()[7] = 0;

        auto C_matmul = A_matmul.matmul(B_matmul);
        C_matmul->to_cpu();

        std::cout << "Результат через Tensor::matmul:\n";
        for (size_t i = 0; i < C_matmul->shape()[0]; ++i) {
            for (size_t j = 0; j < C_matmul->shape()[1]; ++j) {
                std::cout << std::setw(6) << std::fixed << std::setprecision(2)
                    << C_matmul->data()[i * C_matmul->shape()[1] + j] << " ";
            }
            std::cout << "\n";
        }

        std::cout << "\n=== Тест autograd (прямой и обратный проход) ===\n";

        auto a_grad = std::make_shared<MNNL::Tensor<float>>(std::vector<size_t>{2, 2});
        auto b_grad = std::make_shared<MNNL::Tensor<float>>(std::vector<size_t>{2, 2});
        a_grad->set_requires_grad(true);
        b_grad->set_requires_grad(true);

        float* a_grad_data = a_grad->data();
        float* b_grad_data = b_grad->data();
        for (size_t i = 0; i < a_grad->size(); ++i) {
            a_grad_data[i] = static_cast<float>(i) + 1.0f;
            b_grad_data[i] = static_cast<float>(i) * 2.0f;
        }

        std::cout << "a:\n";
        for (size_t i = 0; i < a_grad->shape()[0]; ++i) {
            for (size_t j = 0; j < a_grad->shape()[1]; ++j) {
                std::cout << a_grad_data[i * a_grad->shape()[1] + j] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "b:\n";
        for (size_t i = 0; i < b_grad->shape()[0]; ++i) {
            for (size_t j = 0; j < b_grad->shape()[1]; ++j) {
                std::cout << b_grad_data[i * b_grad->shape()[1] + j] << " ";
            }
            std::cout << "\n";
        }

        auto c_grad = *a_grad + *b_grad;
        auto d_grad = *c_grad * *c_grad;
        auto loss_grad = d_grad->sum();

        std::cout << "loss = " << loss_grad->data()[0] << "\n";

        loss_grad->backward();

        std::cout << "Градиент по a:\n";
        const float* grad_a = a_grad->grad().data();
        for (size_t i = 0; i < a_grad->shape()[0]; ++i) {
            for (size_t j = 0; j < a_grad->shape()[1]; ++j) {
                std::cout << grad_a[i * a_grad->shape()[1] + j] << " ";
            }
            std::cout << "\n";
        }

        std::cout << "Градиент по b:\n";
        const float* grad_b = b_grad->grad().data();
        for (size_t i = 0; i < b_grad->shape()[0]; ++i) {
            for (size_t j = 0; j < b_grad->shape()[1]; ++j) {
                std::cout << grad_b[i * b_grad->shape()[1] + j] << " ";
            }
            std::cout << "\n";
        }

        std::cout << "\n=== Расширенные тесты autograd ===\n";

        // Вспомогательные лямбды (без изменений)
        auto print_tensor = [](const MNNL::Tensor<float>& t, const std::string& name) {
            std::cout << name << ":\n";
            const auto& shape = t.shape();
            if (shape.size() == 2) {
                for (size_t i = 0; i < shape[0]; ++i) {
                    for (size_t j = 0; j < shape[1]; ++j)
                        std::cout << std::setw(8) << std::fixed << std::setprecision(4) << t.data()[i * shape[1] + j] << " ";
                    std::cout << "\n";
                }
            }
            else {
                for (size_t i = 0; i < t.size(); ++i)
                    std::cout << t.data()[i] << " ";
                std::cout << "\n";
            }
            };

        auto check_grad = [](const MNNL::Tensor<float>& grad, const MNNL::Tensor<float>& expected, float tol = 1e-5) {
            bool ok = true;
            for (size_t i = 0; i < grad.size(); ++i) {
                if (std::abs(grad.data()[i] - expected.data()[i]) > tol) {
                    std::cerr << "  mismatch at " << i << ": got " << grad.data()[i] << ", expected " << expected.data()[i] << "\n";
                    ok = false;
                }
            }
            if (ok) std::cout << "  ✓ Gradient check passed\n";
            else std::cout << "  ✗ Gradient check FAILED\n";
            };

        // ------------------------------------------------------------
        // Тест 2: matmul + сумма (градиенты аналитически) – имена изменены
        // ------------------------------------------------------------
        std::cout << "\n--- Тест 2: matmul + sum ---\n";
        auto A_mat_test = std::make_shared<MNNL::Tensor<float>>(std::vector<size_t>{2, 3});
        auto B_mat_test = std::make_shared<MNNL::Tensor<float>>(std::vector<size_t>{3, 4});
        A_mat_test->set_requires_grad(true);
        B_mat_test->set_requires_grad(true);

        for (size_t i = 0; i < A_mat_test->size(); ++i) A_mat_test->data()[i] = static_cast<float>(i + 1);
        for (size_t i = 0; i < B_mat_test->size(); ++i) B_mat_test->data()[i] = static_cast<float>(i + 1) * 0.5f;

        print_tensor(*A_mat_test, "A");
        print_tensor(*B_mat_test, "B");

        auto C_mat_test = A_mat_test->matmul(*B_mat_test);
        auto loss_mat_test = C_mat_test->sum();
        loss_mat_test->backward();

        // Аналитические градиенты
        MNNL::Tensor<float> grad_C_expected({ 2,4 });
        for (size_t i = 0; i < grad_C_expected.size(); ++i) grad_C_expected.data()[i] = 1.0f;

        MNNL::Tensor<float> B_T_test = B_mat_test->transpose();
        MNNL::Tensor<float> dA_expected = grad_C_expected.matmul(B_T_test)->contiguous();
        MNNL::Tensor<float> A_T_test = A_mat_test->transpose();
        MNNL::Tensor<float> dB_expected = A_T_test.matmul(grad_C_expected)->contiguous();

        dA_expected.to_cpu();
        dB_expected.to_cpu();

        std::cout << "Градиент по A (ожидаемый):\n";
        print_tensor(dA_expected, "");
        std::cout << "Градиент по B (ожидаемый):\n";
        print_tensor(dB_expected, "");

        check_grad(A_mat_test->grad(), dA_expected);
        check_grad(B_mat_test->grad(), dB_expected);

        // ------------------------------------------------------------
        // Тест 3: ReLU + сумма
        // ------------------------------------------------------------
        std::cout << "\n--- Тест 3: ReLU + sum ---\n";
        auto x_relu = std::make_shared<MNNL::Tensor<float>>(std::vector<size_t>{2, 3});
        x_relu->set_requires_grad(true);
        for (size_t i = 0; i < x_relu->size(); ++i) {
            x_relu->data()[i] = static_cast<float>(i) - 2.5f;
        }
        print_tensor(*x_relu, "x");

        auto y_relu = x_relu->relu();
        auto loss_relu = y_relu->sum();
        loss_relu->backward();

        MNNL::Tensor<float> expected_grad_relu(x_relu->shape());
        for (size_t i = 0; i < x_relu->size(); ++i) {
            expected_grad_relu.data()[i] = (x_relu->data()[i] > 0.0f) ? 1.0f : 0.0f;
        }

        check_grad(x_relu->grad(), expected_grad_relu);

        // ------------------------------------------------------------
        // Тест 4: Sigmoid + сумма
        // ------------------------------------------------------------
        std::cout << "\n--- Тест 4: Sigmoid + sum ---\n";
        auto x_sig = std::make_shared<MNNL::Tensor<float>>(std::vector<size_t>{5});
        x_sig->set_requires_grad(true);
        for (size_t i = 0; i < x_sig->size(); ++i) {
            x_sig->data()[i] = static_cast<float>(i) - 2.0f;
        }
        print_tensor(*x_sig, "x");

        auto y_sig = x_sig->sigmoid();
        auto loss_sig = y_sig->sum();
        loss_sig->backward();

        MNNL::Tensor<float> expected_grad_sig(x_sig->shape());
        for (size_t i = 0; i < x_sig->size(); ++i) {
            float s = 1.0f / (1.0f + std::exp(-x_sig->data()[i]));
            expected_grad_sig.data()[i] = s * (1.0f - s);
        }

        std::cout << "x_sig values: ";
        for (size_t i = 0; i < x_sig->size(); ++i)
            std::cout << x_sig->data()[i] << " ";
        std::cout << "\n";

        std::cout << "Recalculated expected: ";
        for (size_t i = 0; i < x_sig->size(); ++i) {
            float s = 1.0f / (1.0f + std::exp(-x_sig->data()[i]));
            std::cout << (s * (1.0f - s)) << " ";
        }
        std::cout << "\n";

        check_grad(x_sig->grad(), expected_grad_sig);

        // ------------------------------------------------------------
        // Тест 5: Tanh + сумма
        // ------------------------------------------------------------
        std::cout << "\n--- Тест 5: Tanh + sum ---\n";
        auto x_tanh = std::make_shared<MNNL::Tensor<float>>(std::vector<size_t>{5});
        x_tanh->set_requires_grad(true);
        for (size_t i = 0; i < x_tanh->size(); ++i) {
            x_tanh->data()[i] = static_cast<float>(i) - 2.0f;
        }
        auto y_tanh = x_tanh->tanh();
        auto loss_tanh = y_tanh->sum();
        loss_tanh->backward();

        MNNL::Tensor<float> expected_grad_tanh(x_tanh->shape());
        for (size_t i = 0; i < x_tanh->size(); ++i) {
            float t = std::tanh(x_tanh->data()[i]);
            expected_grad_tanh.data()[i] = 1.0f - t * t;
        }

        std::cout << "x_sig values: ";
        for (size_t i = 0; i < x_sig->size(); ++i)
            std::cout << x_sig->data()[i] << " ";
        std::cout << "\n";

        std::cout << "Recalculated expected: ";
        for (size_t i = 0; i < x_sig->size(); ++i) {
            float s = 1.0f / (1.0f + std::exp(-x_sig->data()[i]));
            std::cout << (s * (1.0f - s)) << " ";
        }
        std::cout << "\n";

        check_grad(x_tanh->grad(), expected_grad_tanh);

        // ------------------------------------------------------------
        // Тест 6: Деление и вычитание
        // ------------------------------------------------------------
        std::cout << "\n--- Тест 6: Деление и вычитание ---\n";
        auto a_div = std::make_shared<MNNL::Tensor<float>>(std::vector<size_t>{2, 2});
        auto b_div = std::make_shared<MNNL::Tensor<float>>(std::vector<size_t>{2, 2});
        a_div->set_requires_grad(true);
        b_div->set_requires_grad(true);
        a_div->data()[0] = 4; a_div->data()[1] = 9; a_div->data()[2] = 16; a_div->data()[3] = 25;
        b_div->data()[0] = 2; b_div->data()[1] = 3; b_div->data()[2] = 4; b_div->data()[3] = 5;

        auto c_div = *a_div / *b_div;
        auto d_div = *c_div - *b_div;
        auto loss_div = d_div->sum();
        loss_div->backward();

        std::cout << "Градиент по a (деление): ";
        for (size_t i = 0; i < a_div->grad().size(); ++i)
            std::cout << a_div->grad().data()[i] << " ";
        std::cout << "\nГрадиент по b (деление): ";
        for (size_t i = 0; i < b_div->grad().size(); ++i)
            std::cout << b_div->grad().data()[i] << " ";
        std::cout << "\n(Проверка аналитически: dL/da = 1/b, dL/db = -a/b^2 - 1)\n";

        // ------------------------------------------------------------
        // Тест 7: Цепочка активаций + matmul
        // ------------------------------------------------------------
        std::cout << "\n--- Тест 7: Цепочка: matmul -> sigmoid -> tanh -> sum ---\n";
        auto X_test = std::make_shared<MNNL::Tensor<float>>(std::vector<size_t>{2, 3});
        auto W_test = std::make_shared<MNNL::Tensor<float>>(std::vector<size_t>{3, 2});
        X_test->set_requires_grad(true);
        W_test->set_requires_grad(true);
        for (size_t i = 0; i < X_test->size(); ++i) X_test->data()[i] = static_cast<float>(i + 1) * 0.5f;
        for (size_t i = 0; i < W_test->size(); ++i) W_test->data()[i] = static_cast<float>(i + 1) * 0.2f;

        auto Z_test = X_test->matmul(*W_test);
        auto A_act = Z_test->sigmoid();
        auto B_act = A_act->tanh();
        auto loss_chain = B_act->sum();
        loss_chain->backward();

        std::cout << "Градиент по X:\n";
        print_tensor(X_test->grad(), "");
        std::cout << "Градиент по W:\n";
        print_tensor(W_test->grad(), "");
        std::cout << "(Численная проверка не проводилась, но градиенты должны быть ненулевыми)\n";

        std::cout << "\n=== Все тесты autograd завершены ===\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Ошибка: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}