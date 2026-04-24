#pragma once
#include <concepts>
#include <type_traits>

namespace MNNL {
    template<typename T>
    concept ArithmeticType = std::is_arithmetic_v<T>;
}