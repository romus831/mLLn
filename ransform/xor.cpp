#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "tensor.h"

using namespace MNNL;

// Вспомогательная функция для случайной инициализации тензора значениями из [-0.5, 0.5]
void random_init(Tensor<float>& t) {
    float* data = t.data();
    for (size_t i = 0; i < t.size(); ++i) {
        data[i] = ((float)rand() / RAND_MAX) - 0.5f;
    }
}

int main() {
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // Данные XOR: 4 примера (батч 4x2)
    std::vector<float> x_vals = {0,0, 0,1, 1,0, 1,1};
    std::vector<float> y_vals = {0, 1, 1, 0};

    Tensor<float> X({4, 2});
    Tensor<float> Y({4, 1});
    std::copy(x_vals.begin(), x_vals.end(), X.data());
    std::copy(y_vals.begin(), y_vals.end(), Y.data());

    // Параметры модели
    const size_t input_size = 2;
    const size_t hidden_size = 4;
    const size_t output_size = 1;

    // Веса и смещения (requires_grad = true)
    Tensor<float> W1({input_size, hidden_size});
    Tensor<float> b1({1, hidden_size});
    Tensor<float> W2({hidden_size, output_size});
    Tensor<float> b2({1, output_size});

    random_init(W1);
    random_init(b1);
    random_init(W2);
    random_init(b2);

    W1.set_requires_grad(true);
    b1.set_requires_grad(true);
    W2.set_requires_grad(true);
    b2.set_requires_grad(true);

    // Гиперпараметры
    const float lr = 0.5f;
    const int epochs = 2000;
    const float N = 4.0f;  // размер батча

    std::cout << "Training XOR model with MLP (2-4-1, sigmoid, MSE)\n";
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // ---------- Forward pass ----------
        auto h = X.matmul(W1);          // (4,2) * (2,4) = (4,4)
        auto h_bias = *h + b1;          // broadcast: (4,4) + (1,4) -> (4,4)
        auto h_act = h_bias->sigmoid(); // (4,4)

        auto y_pred = h_act->matmul(W2); // (4,4) * (4,1) = (4,1)
        auto y_pred_bias = *y_pred + b2; // (4,1) + (1,1) -> (4,1)
        auto y_act = y_pred_bias->sigmoid(); // (4,1)

        // ---------- Loss: MSE ----------
        auto diff = *y_act - Y;
        auto diff_sq = *diff * *diff;
        auto loss_sum = diff_sq->sum();  // shared_ptr
        Tensor<float> scale({ 1 });
        scale.data()[0] = 1.0f / N;
        auto loss = *loss_sum * scale;   // shared_ptr
        loss->backward();

        // ---------- SGD update ----------
        // Обновляем параметры: param = param - lr * grad
        auto update_param = [lr](Tensor<float>& param) {
            float* p = param.data();
            const float* g = param.grad().data();
            for (size_t i = 0; i < param.size(); ++i) {
                p[i] -= lr * g[i];
            }
        };
        update_param(W1);
        update_param(b1);
        update_param(W2);
        update_param(b2);

        // Обнуляем градиенты для следующей итерации
        W1.zero_grad();
        b1.zero_grad();
        W2.zero_grad();
        b2.zero_grad();

        // Печатаем прогресс
        if (epoch % 200 == 0) {
            std::cout << "Epoch " << epoch << ", loss = " << loss->data()[0] << std::endl;
        }
    }

    // ---------- Тестирование обученной модели ----------
    std::cout << "\nTrained model predictions:\n";
    auto h = X.matmul(W1);
    auto h_bias = *h + b1;
    auto h_act = h_bias->sigmoid();
    auto y_pred = h_act->matmul(W2);
    auto y_pred_bias = *y_pred + b2;
    auto y_act = y_pred_bias->sigmoid();

    // Переводим результат на CPU (если был на GPU) и выводим
    y_act->to_cpu();
    const float* out = y_act->data();
    for (int i = 0; i < 4; ++i) {
        int in0 = (i == 2 || i == 3) ? 1 : 0;
        int in1 = (i == 1 || i == 3) ? 1 : 0;
        std::cout << "Input: (" << in0 << "," << in1 << ") -> prediction: "
                  << out[i] << " (expected: " << y_vals[i] << ")\n";
    }

    return 0;
}