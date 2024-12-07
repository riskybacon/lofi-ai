#pragma once
#include <iomanip>
#include <iostream>
#include <memory>
#include <ranges>

#include <lofi/engine.hpp>
#include <lofi/storage.hpp>
#include <lofi/tensor.hpp>

size_t num_tests = 0;
size_t num_passed = 0;
size_t num_failed = 0;

template <typename T> void is_equal_helper(const T &a, const T &b, const char *file, const int line) {
    if (a != b) {
        std::cerr << "[FAIL] (" << file << ":" << line << "): " << a << " != " << b << std::endl;
        num_failed++;
    } else {
        num_passed++;
    }
    num_tests++;
}

void is_equal_helper(const std::string &a, const char *b, const char *file, const int line) {
    if (a != b) {
        std::cerr << "[FAIL] (" << file << ":" << line << "): " << a << " != " << b << std::endl;
        num_failed++;
    } else {
        num_passed++;
    }
    num_tests++;
}

void is_equal_helper(const size_t &a, const size_t &b, const char *file, const int line) {
    if (a != b) {
        std::cerr << "[FAIL] (" << file << ":" << line << "): " << a << " != " << b << std::endl;
        num_failed++;
    } else {
        num_passed++;
    }
    num_tests++;
}

template <typename Iterable0, typename Iterable1>
    requires std::ranges::range<Iterable0> && std::is_arithmetic_v<std::ranges::range_value_t<Iterable0>> &&
             std::ranges::range<Iterable1> && std::is_arithmetic_v<std::ranges::range_value_t<Iterable1>>
void is_equal_helper(Iterable0 &&iterable0, Iterable1 &&iterable1, const char *file, const int line) {
    num_tests++;

    if (std::ranges::size(iterable0) != std::ranges::size(iterable1)) {
        std::cerr << "[FAIL] (" << file << ":" << line << "): size mismatch: " << std::ranges::size(iterable0) << " != "
                  << std::ranges::size(iterable1) << std::endl;
        num_failed++;
        return;
    }

    for (const auto &[val0, val1] : zip(iterable0, iterable1)) {
        if (val0 != val1) {
            std::cerr << "[FAIL] (" << file << ":" << line << "): " << val0 << " != " << val1 << std::endl;
            num_failed++;
            return;
        }
    }
    num_passed++;
}

template <typename T> void not_equal_helper(const T &a, const T &b, const char *file, const int line) {
    if (a == b) {
        std::cerr << "[FAIL] (" << file << ":" << line << "): " << a << " == " << b << std::endl;
        num_failed++;
    } else {
        num_passed++;
    }
    num_tests++;
}

bool values_close(float a, float b) {
    float epsilon = 1e-6;
    float hi = b + epsilon;
    float lo = b - epsilon;

    return a >= lo && a <= hi;
}

bool values_close(size_t a, size_t b) { return a == b; }

void is_close_helper(float a, float b, const char *file, const int line) {
    if (!values_close(a, b)) {
        std::cerr << "[FAIL] (" << file << ":" << line << "): " << std::fixed << std::setprecision(10) << a
                  << " != " << b << std::endl;
        num_failed++;
    } else {
        num_passed++;
    }
    num_tests++;
}

template <typename T>
void is_close_helper(const std::shared_ptr<Context<T>> &a, const std::shared_ptr<Context<T>> &b, const char *file,
                     const int line) {
    is_close_helper(a->data, b->data, file, line);
    is_close_helper(a->grad, b->grad, file, line);
    is_equal_helper(a->label, b->label, file, line);
    is_equal_helper(a->op, b->op, file, line);
}

template <typename T>
void is_close_helper(const MatrixStorage<T> &a, const MatrixStorage<T> &b, const char *file, const int line) {
    const auto [condition, reason] = is_close(a, b);
    if (!condition) {
        std::cerr << "[FAIL] (" << file << ":" << line << "): " << reason << std::endl;
        num_failed++;
    } else {
        num_passed++;
    }
    num_tests++;
}

template <typename T>
void is_close_helper(const std::vector<T> &a, const std::vector<T> &b, const char *file, const int line) {
    bool pass = true;
    std::string reason;
    if (a.size() != b.size()) {
        std::stringstream ss;
        ss << "size mismatch: a.size()=" << a.size() << ", b.size()=" << b.size();
        reason = ss.str();
        pass = false;
    } else {
        for (size_t i = 0; i < a.size() && pass; i++) {
            if (!values_close(a[i], b[i])) {
                pass = false;
                std::stringstream ss;
                ss << "a[" << i << "]=" << a[i] << " != b[" << i << "]=" << b[i];
                reason = ss.str();
            }
        }
    }

    if (pass) {
        num_passed++;
    } else {
        std::cerr << "[FAIL] (" << file << ":" << line << "): " << reason << std::endl;
        num_failed++;
    }
    num_tests++;
}

template <typename T>
void is_close_helper(const Tensor<T> &a, const Tensor<T> &b, const char *file, const int line) {
    num_tests++;
    const auto [condition, reason] = is_close(a, b);
    if (!condition) {
        std::cerr << "[FAIL] (" << file << ":" << line << "): " << reason << std::endl;
        num_failed++;
    } else {
        num_passed++;
    }
}

template <typename T>
void in_range_helper(T k, const T &v, const T &min_val, const T &max_val, const char *file, const int line) {
    if (v < min_val || v > max_val) {
        std::cerr << "[FAIL] (" << file << ":" << line << "): value " << k << ", " << v << " not in range [" << min_val << "," << max_val
                  << "]" << std::endl;
        num_failed++;
    } else {
        num_passed++;
    }
    num_tests++;
}

#define is_close(a, b) is_close_helper(a, b, __FILE__, __LINE__)
#define is_equal(a, b) is_equal_helper(a, b, __FILE__, __LINE__)
#define not_equal(a, b) not_equal_helper(a, b, __FILE__, __LINE__)
#define in_range(k, v, min_val, max_val) in_range_helper(k, v, min_val, max_val, __FILE__, __LINE__)

#define throws_exception(exc_type, call)                                                                               \
    try {                                                                                                              \
        call;                                                                                                          \
        std::cerr << "[FAIL] (" << __FILE__ << ":" << __LINE__ << "): exception not thrown" << std::endl;              \
        num_failed++;                                                                                                  \
    } catch (exc_type & e) {                                                                                           \
        num_passed++;                                                                                                  \
    }                                                                                                                  \
    num_tests++;

// Fill a matrix with some values, must be deterministic between platforms
template <typename T> void fill_mat(MatrixStorage<T> &mat, const T offset = static_cast<T>(0)) {
    T val(offset);
    for (size_t r = 0; r < mat.shape[0]; r++) {
        for (size_t c = 0; c < mat.shape[1]; c++) {
            mat[{r, c}] = val;
            val += static_cast<T>(1);
        }
    }
}
