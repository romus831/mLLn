#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "tensor.h"
#include "saver.h"

using namespace MNNL;

//#define TRAIN 0

// Случайная инициализация в диапазоне [-0.5, 0.5]
void random_init(Tensor<float>& t) {
    float* data = t.data();
    for (size_t i = 0; i < t.size(); ++i) {
        data[i] = ((float)rand() / RAND_MAX) - 0.5f;
    }
}

// Для отладки градиентов (используется только при обучении)
auto print_grad = [](const Tensor<float>& t, const std::string& name) {
    if (!t.has_grad()) { std::cout << name << " grad: none\n"; return; }
    const float* g = t.grad().data();
    double sum = 0;
    for (size_t i = 0; i < t.size(); ++i) sum += std::fabs(g[i]);
    std::cout << name << " avg |grad| = " << sum / t.size() << "\n";
    };

int main() {
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // Данные XOR
    std::vector<float> x_vals = { 0,0, 0,1, 1,0, 1,1 };
    std::vector<float> y_vals = { 0, 1, 1, 0 };

    Tensor<float> X({ 4, 2 });
    Tensor<float> Y({ 4, 1 });
    std::copy(x_vals.begin(), x_vals.end(), X.data());
    std::copy(y_vals.begin(), y_vals.end(), Y.data());

    // Архитектура
    const size_t input_size = 2;
    const size_t hidden_size = 10;
    const size_t output_size = 1;

    // Путь к файлу модели (относительно рабочей директории)
    const std::string model_path = "../../xor_model.bin";

#ifdef TRAIN
    // ================== РЕЖИМ ОБУЧЕНИЯ ==================
    std::cout << "Training XOR model with MLP (2-10-1, ReLU hidden, Sigmoid output, MSE)\n";

    Tensor<float> W1({ input_size, hidden_size });
    Tensor<float> b1({ 1, hidden_size });
    Tensor<float> W2({ hidden_size, output_size });
    Tensor<float> b2({ 1, output_size });

    random_init(W1);
    random_init(b1);
    random_init(W2);
    random_init(b2);

    W1.set_requires_grad(true);
    b1.set_requires_grad(true);
    W2.set_requires_grad(true);
    b2.set_requires_grad(true);

    const float lr = 0.5f;
    const int epochs = 2000;
    const float N = 4.0f;  // размер батча

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Forward pass
        auto h = X.matmul(W1);
        auto h_bias = *h + b1;
        auto h_act = h_bias->relu();

        auto y_pred = h_act->matmul(W2);
        auto y_pred_bias = *y_pred + b2;
        auto y_act = y_pred_bias->sigmoid();

        // MSE loss
        auto diff = *y_act - Y;
        auto diff_sq = *diff * *diff;
        auto loss_sum = diff_sq->sum();
        Tensor<float> scale({ 1 });
        scale.data()[0] = 1.0f / N;
        auto loss = *loss_sum * scale;
        loss->backward();

        // Отладка градиентов (можно закомментировать)
        print_grad(W1, "W1");
        print_grad(b1, "b1");
        print_grad(W2, "W2");
        print_grad(b2, "b2");

        // SGD update
        auto update_param = [lr](Tensor<float>& param) {
            if (!param.has_grad()) {
                std::cerr << "Warning: param has no grad\n";
                return;
            }
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

        W1.zero_grad();
        b1.zero_grad();
        W2.zero_grad();
        b2.zero_grad();

        if (epoch % 200 == 0) {
            std::cout << "Epoch " << epoch << ", loss = " << loss->data()[0] << std::endl;
        }
    }

    // Сохраняем обученную модель
    model_io::save_model(model_path, W1, b1, W2, b2);
    std::cout << "Model saved to " << model_path << std::endl;

#else
    // ================== РЕЖИМ ИНФЕРЕНСА ==================
    std::cout << "Loading trained model from " << model_path << "\n";

    Tensor<float> W1, b1, W2, b2;
    model_io::load_model(model_path, W1, b1, W2, b2);

    // Тестирование на всех примерах XOR
    std::cout << "\nTrained model predictions:\n";
    auto h = X.matmul(W1);
    auto h_bias = *h + b1;
    auto h_act = h_bias->relu();          // обязательно ReLU, как при обучении
    auto y_pred = h_act->matmul(W2);
    auto y_pred_bias = *y_pred + b2;
    auto y_act = y_pred_bias->sigmoid();

    y_act->to_cpu();
    const float* out = y_act->data();
    for (int i = 0; i < 4; ++i) {
        int in0 = (i == 2 || i == 3) ? 1 : 0;
        int in1 = (i == 1 || i == 3) ? 1 : 0;
        std::cout << "Input: (" << in0 << "," << in1 << ") -> prediction: "
            << out[i] << " (expected: " << y_vals[i] << ")\n";
    }
#endif

    return 0;
}