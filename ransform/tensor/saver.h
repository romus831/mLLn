#ifndef SAVER_H
#define SAVER_H

#include <fstream>
#include <vector>
#include <string>
#include "tensor.h"

namespace model_io {

    inline void save_tensor(std::ofstream& file, const MNNL::Tensor<float>& t) {
        const auto& shape = t.shape();
        size_t ndim = shape.size();
        file.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
        for (size_t d : shape) {
            file.write(reinterpret_cast<const char*>(&d), sizeof(d));
        }
        MNNL::Tensor<float> cpu_copy = t;
        cpu_copy.to_cpu();
        file.write(reinterpret_cast<const char*>(cpu_copy.data()), cpu_copy.size() * sizeof(float));
    }

    inline void load_tensor(std::ifstream& file, MNNL::Tensor<float>& t) {
        size_t ndim;
        file.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
        std::vector<size_t> shape(ndim);
        for (size_t i = 0; i < ndim; ++i) {
            file.read(reinterpret_cast<char*>(&shape[i]), sizeof(size_t));
        }
        t = MNNL::Tensor<float>(shape);
        t.to_cpu();
        file.read(reinterpret_cast<char*>(t.data()), t.size() * sizeof(float));
    }

    inline void save_model(const std::string& filename,
        const MNNL::Tensor<float>& W1,
        const MNNL::Tensor<float>& b1,
        const MNNL::Tensor<float>& W2,
        const MNNL::Tensor<float>& b2) {
        std::ofstream file(filename, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open file for writing: " + filename);
        save_tensor(file, W1);
        save_tensor(file, b1);
        save_tensor(file, W2);
        save_tensor(file, b2);
    }

    inline void load_model(const std::string& filename,
        MNNL::Tensor<float>& W1,
        MNNL::Tensor<float>& b1,
        MNNL::Tensor<float>& W2,
        MNNL::Tensor<float>& b2) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open file for reading: " + filename);
        load_tensor(file, W1);
        load_tensor(file, b1);
        load_tensor(file, W2);
        load_tensor(file, b2);
    }

}

#endif 