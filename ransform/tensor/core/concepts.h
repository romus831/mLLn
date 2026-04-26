#pragma once
#include <concepts>
#include <type_traits>

namespace MNNL {

    template <typename T>
    concept ArithmeticType = std::is_arithmetic_v<T>;

    template <typename T>
    concept Addable = requires(T a, T b) {
        { a + b } -> std::same_as<T>;
    };

    template <typename T>
    concept Numeric = ArithmeticType<T> &&
        requires(T a, T b) {
            { a + b } -> std::same_as<T>;
            { a - b } -> std::same_as<T>;
            { a* b } -> std::same_as<T>;
            { a / b } -> std::same_as<T>;
    };
}