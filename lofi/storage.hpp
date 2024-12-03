// TODO: Maybe MatrixStorage should be broken into multiple classes.
// * The matrix class that has functions that perform the matrix operations, specifically,
//   the non-autograd style functions.
// * Storage management, which handles memory management and element access.
#pragma once
#include <algorithm>
#include <array>
#include <cblas.h>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <ostream>
#include <random>
#include <ranges>
#include <sstream>
#include <vector>

#include <lofi/generator.hpp>

#ifdef BOUNDS_CHECK
constexpr bool _BOUNDS_CHECK = true;
#else
constexpr bool _BOUNDS_CHECK = false;
#endif

using shape_type = std::array<size_t, 2>;
using std::begin;
using std::end;
using std::cbegin;
using std::cend;
using std::ranges::views::zip;
using std::views::iota;
using std::views::transform;

template <typename T> void assign_op(T &lhs, const T &rhs) { lhs = rhs; }

template <typename T> void accumulate_op(T &lhs, const T &rhs) { lhs += rhs; }

shape_type identity(shape_type &&idx) { return idx; }

shape_type swap_idx(shape_type &&idx) { return shape_type({idx[1], idx[0]}); }

shape_type bcast0(shape_type &&idx) { return shape_type({0, idx[1]}); }

shape_type bcast1(shape_type &&idx) { return shape_type({idx[0], 0}); }

shape_type bcast_all(shape_type &&idx) { return shape_type({0, 0}); }

auto choose_bcast(const shape_type &shape) {
    if (shape[0] == 1 && shape[1] == 1) {
        return bcast_all;
    } else if (shape[0] == 1) {
        return bcast0;
    } else if (shape[1] == 1) {
        return bcast1;
    }
    return identity;
}

shape_type max_shape(const shape_type &a, const shape_type &b) {
    shape_type out;
    for (size_t i = 0; i < out.size(); i++) {
        out[i] = std::max(a[i], b[i]);
    }
    return out;
}

template <size_t DIM> std::ostream &operator<<(std::ostream &out, const std::array<size_t, DIM> &shape) {
    out << "(";
    for (size_t i = 0; i < DIM; i++) {
        out << shape[i];
        if (i < DIM - 1) {
            out << ",";
        }
    }
    out << ")";
    return out;
}

void throw_unexpected_shape(const shape_type &actual, const shape_type &expected, const char *file, const int line) {
    std::stringstream ss;
    ss << "[" << file << ":" << line << "]: Expected shape=" << expected << ", got " << actual;
    throw std::invalid_argument(ss.str());
}

void _assert_expected_shape(const shape_type &actual, const shape_type &expected, const char *file, const int line) {
    if (actual != expected) {
        throw_unexpected_shape(actual, expected, file, line);
    }
}

void _assert_axis_lte(size_t axis, size_t max_axis, const char *file, const int line) {
    if (axis > max_axis) {
        std::stringstream ss;
        ss << "[" << file << ":" << line << "]: axis " << axis << " is greater than max axis " << max_axis;
        throw std::invalid_argument(ss.str());
    }
}

void _assert_shape_not_zero(const shape_type &shape, const char *file, const int line) {
    bool not_zero = false;
    for (size_t i = 0; !not_zero && i < shape.size(); i++) {
        if (shape[i] > 0) {
            not_zero = true;
        }
    }
    if (!not_zero) {
        std::stringstream ss;
        ss << "[" << file << ":" << line << "]: shape " << shape << " must not have zero dimensions";
        throw std::invalid_argument(ss.str());
    }
}

#define assert_expected_shape(a, b) _assert_expected_shape(a, b, __FILE__, __LINE__)
#define assert_axis_lte(a, b) _assert_axis_lte(a, b, __FILE__, __LINE__)
#define assert_shape_not_zero(a) _assert_shape_not_zero(a, __FILE__, __LINE__)

template <typename T> struct MatrixStorage {
    using value_type = T;
    using shape_type = std::array<size_t, 2>;

    shape_type shape = {0, 0};
    shape_type strides = {0, 0};
    std::vector<T> data;

    MatrixStorage() {}

    // Call .clone() explicitly
    MatrixStorage(const MatrixStorage &) = delete;
    MatrixStorage &operator=(const MatrixStorage &rhs) = delete;

    MatrixStorage(const shape_type &shape) : shape(shape), strides({shape[1], 1}), data(shape[0] * shape[1]) {}
    MatrixStorage(const shape_type &shape, value_type val)
        : shape(shape), strides({shape[1], 1}), data(shape[0] * shape[1]) {
        fill_value<value_type>(*this, val);
    }

    MatrixStorage(MatrixStorage &&rhs)
        : shape(std::move(rhs.shape)), strides({shape[1], 1}), data(std::move(rhs.data)) {
        rhs.shape = {0, 0};
    }

    MatrixStorage &operator=(MatrixStorage &&rhs) {
        if (this != &rhs) {
            shape = std::move(rhs.shape);
            strides = std::move(rhs.strides);
            data = std::move(rhs.data);
            for (size_t i = 0; i < shape.size(); i++) {
                rhs.shape[i] = 0;
            }
        }
        return *this;
    }

    MatrixStorage clone() const {
        MatrixStorage out;
        out.shape = shape;
        out.strides = strides;
        out.data = data;
        return out;
    }

    auto slice(size_t axis, size_t index) {
        if (axis == 0) {
            return data | std::views::drop(index * strides[1]) | std::views::stride(strides[0]) | std::views::take(shape[0]);
        }
        return data | std::views::drop(index * strides[0]) | std::views::stride(strides[1]) | std::views::take(shape[1]);
    }

    auto slice(size_t axis, size_t index) const {
        if (axis == 0) {
            // column slice
            return data | std::views::drop(index * strides[1]) | std::views::stride(strides[0]) | std::views::take(shape[0]);
        }
        // row slice
        return data | std::views::drop(index * strides[0]) | std::views::stride(strides[1]) | std::views::take(shape[1]);
    }

    auto slices(size_t outer_axis, size_t inner_axis) {
        return iota(static_cast<size_t>(0), shape[outer_axis]) |
               transform([inner_axis, this](size_t i) { return slice(inner_axis, i); });
    }

    const auto slices(size_t outer_axis, size_t inner_axis) const {
        return iota(static_cast<size_t>(0), shape[outer_axis]) |
               transform([inner_axis, this](size_t i) { return slice(inner_axis, i); });
    }

    /**
     * @brief returns an iterator that ranges from 0 to shape[axis], increasing by 1
     */
    auto axis_iota(size_t axis) const {
        return std::views::iota(static_cast<size_t>(0), shape[axis]);
    }

    MatrixStorage &operator=(const value_type val) {
        fill_value(*this, val);
        return *this;
    }

    MatrixStorage operator+(const MatrixStorage &rhs) {
        MatrixStorage out(shape);
        add(out, *this, rhs);
        return out;
    }

    MatrixStorage operator+(const value_type &rhs) {
        MatrixStorage out(shape);
        add(out, *this, rhs);
        return out;
    }

    MatrixStorage &operator+=(const MatrixStorage &rhs) {
        add(*this, *this, rhs);
        return *this;
    }

    MatrixStorage &operator+=(const value_type &rhs) {
        add(*this, *this, rhs);
        return *this;
    }

    MatrixStorage operator*(const MatrixStorage &rhs) const {
        MatrixStorage out(shape);
        multiply(out, *this, rhs);
        return out;
    }

    MatrixStorage operator*(const value_type &rhs) const {
        MatrixStorage out(shape);
        multiply(out, *this, rhs);
        return out;
    }

    MatrixStorage &operator*=(const MatrixStorage &rhs) {
        multiply(*this, *this, rhs);
        return *this;
    }

    MatrixStorage &operator-() {
        multiply(*this, static_cast<value_type>(-1), *this);
        return *this;
    }

    MatrixStorage operator-(const MatrixStorage &rhs) {
        MatrixStorage out(shape);
        subtract(out, *this, rhs);
        return out;
    }

    MatrixStorage operator-(const value_type &rhs) {
        MatrixStorage out(shape);
        subtract(out, *this, -rhs);
        return out;
    }

    MatrixStorage &operator-=(const MatrixStorage &rhs) {
        subtract(*this, *this, rhs);
        return *this;
    }

    MatrixStorage &operator-=(const value_type &rhs) {
        subtract(*this, *this, rhs);
        return *this;
    }

    MatrixStorage operator/(const MatrixStorage &rhs) {
        MatrixStorage out(shape);
        divide(out, *this, rhs);
        return out;
    }

    MatrixStorage operator/(const value_type &rhs) {
        MatrixStorage out(shape);
        divide(out, *this, rhs);
        return out;
    }

    MatrixStorage &operator/=(const MatrixStorage &rhs) {
        divide(*this, *this, rhs);
        return *this;
    }

    MatrixStorage &operator/=(const value_type &rhs) {
        divide(*this, *this, rhs);
        return *this;
    }

    MatrixStorage exp() const {
        MatrixStorage out(shape);
        ::exp(out, *this);
        return out;
    }

    MatrixStorage log() const {
        MatrixStorage out(shape);
        ::log(out, *this);
        return out;
    }

    MatrixStorage tanh() const {
        MatrixStorage out(shape);
        ::tanh(out, *this);
        return out;
    }

    MatrixStorage pow(const value_type &x) const {
        MatrixStorage out(shape);
        ::pow(out, *this, x);
        return out;
    }

    MatrixStorage sum(size_t axis) const {
        shape_type out_shape = shape;
        out_shape[axis] = 1;
        MatrixStorage out(out_shape);
        sum(out, *this, axis);
        return out;
    }

    MatrixStorage max(size_t axis) const {
        shape_type out_shape = shape;
        out_shape[axis] = 1;
        MatrixStorage out(out_shape);
        max(out, *this, axis);
        return out;
    }

    MatrixStorage mean(size_t axis) const {
        shape_type out_shape = shape;
        out_shape[axis] = 1;
        MatrixStorage out(out_shape);
        mean(out, *this, axis);
        return out;
    }

    MatrixStorage stddev(size_t axis) const {
        shape_type out_shape = shape;
        out_shape[axis] = 1;
        MatrixStorage out(out_shape);
        stddev(out, *this, axis);
        return out;
    }

    value_type &operator[](size_t idx) {
        if constexpr (_BOUNDS_CHECK) {
            return data.at(idx);
        } else {
            return data[idx];
        }
    }

    const value_type &operator[](size_t idx) const {
        if constexpr (_BOUNDS_CHECK) {
            return data.at(idx);
        } else {
            return data[idx];
        }
    }

    const size_t offset(const shape_type &idx) const { return idx[0] * shape[1] + idx[1]; }

    value_type &operator[](const shape_type &idx) { return operator[](offset(idx)); }

    const value_type &operator[](const shape_type &idx) const { return operator[](offset(idx)); }

    template <typename U> MatrixStorage operator[](MatrixStorage<U> &idx) {
        MatrixStorage out({idx.shape[0], 1});
        select_rows_and_cols(out, *this, idx);
        return out;
    }

    auto id() const { return reinterpret_cast<std::uintptr_t>(data.data()); }
};

/**
 * @brief Creates a matrix with elements drawn from a normal distribution.
 *
 * This function generates a matrix of the specified dimensions
 * (`rows` x `cols`), where each element is sampled from a normal (Gaussian)
 * distribution with the given mean and variance. The distribution is
 * generated using the provided random number generator `gen`.
 *
 * @param mat An output matrix with dimensions (`rows` x `cols`), where each
 *            element is drawn from N(mean, variance).
 * @param gen  A Mersenne Twister random number generator used for drawing
 *             random samples.
 * @param mean The mean of the normal distribution. Default value is 0.0.
 * @param variance The variance of the normal distribution. Default value is
 *                 1.0.
 */
template <typename T> void fill_randn(MatrixStorage<T> &mat, std::mt19937 &gen, T mean = 0.0, T variance = 1.0) {
    std::normal_distribution<T> dist(mean, std::sqrt(variance));
    for (auto row_slice : mat.slices(0, 0)) {
        for (auto &elem : row_slice) {
            elem = dist(gen);
        }
    }
}

template <typename T> void fill_randn(MatrixStorage<T> &mat, T mean = 0.0, T variance = 1.0) {
    auto gen = generator();
    return fill_randn(mat, gen, mean, variance);
}

template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
void fill_randint(MatrixStorage<T> &mat, T low, T high, std::mt19937 &gen) {
    std::uniform_int_distribution<T> dist(low, high - 1);

    for (auto row_slice : mat.slices(0, 0)) {
        for (auto &elem : row_slice) {
            elem = dist(gen);
        }
    }
}

template <typename T> void fill_value(MatrixStorage<T> &mat, const T &v) {
    std::fill(std::begin(mat.data), std::end(mat.data), v);
}

template <typename T> void broadcast(MatrixStorage<T> &lhs, const MatrixStorage<T> &rhs) {
    auto throw_incompatible_lhs_rhs = [&]() {
        std::stringstream ss;
        ss << "Cannot broadcast rhs.shape=" << rhs.shape << " into lhs.shape=" << lhs.shape;
        throw std::invalid_argument(ss.str());
    };

    if (rhs.shape[0] == 1 && lhs.shape[1] != rhs.shape[1]) {
        throw_incompatible_lhs_rhs();
    }

    if (rhs.shape[1] == 1 && lhs.shape[0] != rhs.shape[0]) {
        throw_incompatible_lhs_rhs();
    }

    const size_t axis = rhs.shape[0] == 1 ? 0 : 1;
    const size_t outer_axis = axis == 0 ? 1 : 0;
    auto rhs_slice = rhs.slice(outer_axis, 0);
    auto lhs_slices = lhs.slices(axis, outer_axis);

#pragma omp parallel for
    for (auto lhs_slice : lhs_slices) {
        std::copy(std::begin(rhs_slice), std::end(rhs_slice), std::begin(lhs_slice));
    }
}

/**
 * @brief Converts a vector of indices into a one-hot encoded matrix.
 *
 * This function takes a vector of index values and returns a one-hot encoded matrix with
 * dimensions (`xs.size()` x `num_classes`). Each row in the matrix corresponds to an element
 * in the input vector `xs`, with a 1 placed at the index specified by the value in `xs`,
 * and 0s elsewhere.
 *
 * For example, given an input vector `xs = {1, 0, 2}` and `num_classes = 3`, the resulting matrix
 * will be:
 *
 * \code
 * 0 1 0
 * 1 0 0
 * 0 0 1
 * \endcode
 *
 * @param out The destination output matrix, shape [xs.size(), num_classes]
 * @param xs A vector of integers representing the indices to be one-hot encoded.
 *           Each element in `xs` must be in the range [0, `num_classes` - 1].
 *
 * @return A matrix (vector of vectors of floats) with dimensions (`xs.size()` x `num_classes`),
 *         where each row contains a one-hot encoding of the corresponding element in `xs`.
 */
template <typename T, typename U> void one_hot(MatrixStorage<T> &out, const std::vector<U> &xs) {
    fill_value(out, static_cast<T>(0));
    for (size_t r = 0; r < xs.size(); r++) {
        const size_t c = static_cast<size_t>(xs[r]);
        out[{r, c}] = static_cast<T>(1);
    }
}

/**
 * @brief Computes the transpose of this matrix.
 *
 * This function takes this matrix and returns its transpose. The transpose
 * of a matrix is formed by swapping its rows and columns. For a matrix
 * `mat` with dimensions (m x n), the resulting matrix will have dimensions
 * (n x m), where each element at position (i, j) in the original
 * matrix is placed at position (j, i) in the transposed matrix.
 *
 * @return A matrix representing the transpose of `mat`. The resulting
 *         matrix will have dimensions (n x m), where `m` and `n` are the
 *         number of rows and columns of this matrix, respectively.
 */
template <typename T> void transpose(MatrixStorage<T> &out, const MatrixStorage<T> &in) {
    const shape_type expected_shape = {in.shape[1], in.shape[0]};
    if (out.shape != expected_shape) {
        std::stringstream ss;
        ss << "Bad dimensions for transpose: out.shape=" << out.shape << ", expected " << expected_shape;
        throw std::invalid_argument(ss.str());
    }

    auto out_col_slices = out.slices(1, 0);
    auto in_row_slices = in.slices(0, 1);

#pragma omp parallel for
    for (auto [out_col_slice, in_row_slice] : zip(out_col_slices, in_row_slices)) {
        std::copy(cbegin(in_row_slice), cend(in_row_slice), std::begin(out_col_slice));
    }
}

template <typename T, typename Func>
void eltwise_unary_func(MatrixStorage<T> &out, const MatrixStorage<T> &in, Func func) {
    if (out.shape != in.shape) {
        std::stringstream ss;
        ss << "out.shape=" << out.shape << " != in.shape=" << in.shape;
        throw std::invalid_argument(ss.str());
    }

    auto out_row_slices = out.slices(0, 1);
    auto in_row_slices = in.slices(0, 1);

#pragma omp parallel for
    for (auto [out_row_slice, in_row_slice] : zip(out_row_slices, in_row_slices)) {
        std::transform(cbegin(in_row_slice), cend(in_row_slice), std::begin(out_row_slice), func);
    }
}

/**
 * @brief Element-wise combination of two matrices
 *
 * Performs an element-wise binary function on two matrices. It is possible
 * that either `lhs` or `rhs` will need to be broadcast along one axis if
 * they have 1 axis that has a dimension of 1.
 *
 * The resulting matrix will have dimensions equal to the number of rows and
 * columns in `lhs` and `rhs`
 *
 * Possible inputs shapes:
 * lhs: (m x n), rhs: (m x n) - no broadcasting
 * lhs: (m x n), rhs: (1 x n) - broadcast rhs along axis 0
 * lhs: (1 x n), rhs: (m x n) - broadcast lhs along axis 0
 * lhs: (m x n), rhs: (m x 1) - broadcast rhs along axis 1
 * lhs: (m x 1), rhs: (m x n) - broadcast lhs along axis 1
 *
 * Only one of `lhs` or `rhs` can have a dimension of 1. If both have a
 * dimension of 1, then an exception will be thrown
 *
 * @param out The output matrix. Must have shape m x n, where m is the number
 *            of rows and n is the number of columns. This matrix will
 *            contain the element-wise composition of `lhs` and `rhs` as
 *            defined by func.
 * @param lhs The left-hand side matrix
 * @param rhs The right-hand side matrix
 * @param func A function that takes two elements from `lhs` and `rhs` and
 *             returns a new value to be stored in `out`
 */
template <typename T, typename Func>
void eltwise_binary_func(MatrixStorage<T> &out, const MatrixStorage<T> &lhs, const MatrixStorage<T> &rhs, Func func) {
    const shape_type out_shape = max_shape(lhs.shape, rhs.shape);
    assert_expected_shape(out.shape, out_shape);
    if (out.shape != out_shape) {
        std::stringstream ss;
        ss << "Expected out.shape=" << out_shape << ", got " << out.shape;
        throw std::invalid_argument(ss.str());
    }

    auto throw_incompatible_lhs_rhs = [&]() {
        std::stringstream ss;
        ss << "lhs.shape=" << lhs.shape << " and rhs.shape=" << rhs.shape << " are incompatible";
        throw std::invalid_argument(ss.str());
    };

    const size_t rows = out.shape[0];
    const size_t cols = out.shape[1];

    std::function<shape_type(shape_type)> o_idx = identity;
    std::function<shape_type(shape_type)> l_idx = identity;
    std::function<shape_type(shape_type)> r_idx = identity;

    bool bcast_rhs = false;
    bool bcast_lhs = false;

    if (out.shape != rhs.shape) {
        bcast_rhs = true;
        if (rhs.shape[0] == 1 && rhs.shape[1] == 1) {
            r_idx = bcast_all;
        } else if (rhs.shape[0] == 1) {
            r_idx = bcast0;
        } else if (rhs.shape[1] == 1) {
            r_idx = bcast1;
        } else {
            throw_incompatible_lhs_rhs();
        }
    }

    if (out.shape != lhs.shape) {
        bcast_lhs = true;
        if (lhs.shape[0] == 1 && lhs.shape[1] == 1) {
            l_idx = bcast_all;
        } else if (lhs.shape[0] == 1) {
            l_idx = bcast0;
        } else if (lhs.shape[1] == 1) {
            l_idx = bcast1;
        } else {
            throw_incompatible_lhs_rhs();
        }
    }

    if (bcast_lhs && bcast_rhs) {
        throw_incompatible_lhs_rhs();
    }

#pragma omp parallel for
    for (size_t r = 0; r < rows; r++) {
        for (size_t c = 0; c < cols; c++) {
            out[o_idx({r, c})] = func(lhs[l_idx({r, c})], rhs[r_idx({r, c})]);
        }
    }
}

template <typename T, typename Func>
void apply_binary_func(MatrixStorage<T> &out, const MatrixStorage<T> &lhs, const MatrixStorage<T> &rhs, Func func) {
    auto throw_exception = [&]() {
        std::stringstream ss;
        ss << "Shapes incompatible in apply_binary_func: out.shape=" << out.shape << ", lhs.shape=" << lhs.shape
           << ", rhs.shape=" << rhs.shape;
        throw std::invalid_argument(ss.str());
    };

    // TODO: this proabably doesn't belong here, but when I try to put it somewhere else,
    // I get code duplication. This is here for the backward pass
    bool broadcast_out = false;
    size_t axis = 0;
    const shape_type out_shape = max_shape(lhs.shape, rhs.shape);
    if (out.shape != out_shape) {
        if (out.shape[0] == 1 && out.shape[1] == lhs.shape[1]) {
            axis = 0;
        } else if (out.shape[1] == 1 && out.shape[0] == lhs.shape[0]) {
            axis = 1;
        } else {
            throw_exception();
        }
        broadcast_out = true;
    }

    MatrixStorage<T> *out_tmp = broadcast_out ? new MatrixStorage<T>(out_shape) : &out;
    eltwise_binary_func(*out_tmp, lhs, rhs, func);

    // TODO: summing across an axis for output broadcast seems like hidden behavior
    // and belongs elsewhere, but I'm not sure how best to architect that.
    // This is used in the backward pass
    if (broadcast_out) {
        sum(out, *out_tmp, axis);
        delete out_tmp;
    }
}

/**
 * @brief Element-wise addition of two matrices
 *
 * Performs an element-wise addition of `rhs` and `lhs`. If one of the input
 * matrices is a row-vector (1xn) or column-vector (mx1), that matrix will be
 * broadcast appropriately. Only one matrix is allowed to be a row or column
 * vector.
 *
 * Allowed input shapes:
 *
 * (m x n) + (m x 1)
 * (m x n) + (1 x n)
 * (m x 1) + (m x n)
 * (1 x n) + (m x n)
 *
 * The resulting matrix will have shape (m x n)
 *
 * @param lhs The left-hand side matrix
 * @param rhs The right-hand side matrix
 *
 * @return A matrix that represents the element-wise addition of `lhs` and
 *         `rhs`. The result will have shape (m x n) where `m` is the number
 *         of rows and `n` is the number of columns
 */
template <typename T> void add(MatrixStorage<T> &out, const MatrixStorage<T> &lhs, const MatrixStorage<T> &rhs) {
    apply_binary_func(out, lhs, rhs, [](const T &a, const T &b) { return a + b; });
}

template <typename T> void add(MatrixStorage<T> &out, const MatrixStorage<T> &lhs, const T &rhs) {
    eltwise_unary_func(out, lhs, [&rhs](const T &a) { return a + rhs; });
}

template <typename T> void add(MatrixStorage<T> &out, const T &lhs, const MatrixStorage<T> &rhs) {
    eltwise_unary_func(out, rhs, [&lhs](const T &a) { return lhs + a; });
}

template <typename T> MatrixStorage<T> operator+(const T &rhs, const MatrixStorage<T> &lhs) {
    MatrixStorage<T> out(lhs.shape);
    add(out, rhs, lhs);
    return out;
}

template <typename T> void subtract(MatrixStorage<T> &out, const MatrixStorage<T> &lhs, const MatrixStorage<T> &rhs) {
    apply_binary_func(out, lhs, rhs, [](const T &a, const T &b) { return a - b; });
}

template <typename T> void subtract(MatrixStorage<T> &out, const MatrixStorage<T> &lhs, const T &rhs) {
    eltwise_unary_func(out, lhs, [&rhs](const T &a) { return a - rhs; });
}

template <typename T> void subtract(MatrixStorage<T> &out, const T &lhs, const MatrixStorage<T> &rhs) {
    eltwise_unary_func(out, rhs, [&lhs](const T &a) { return lhs - a; });
}

template <typename T> MatrixStorage<T> operator-(const T &rhs, const MatrixStorage<T> &lhs) {
    MatrixStorage<T> out(lhs.shape);
    subtract(out, rhs, lhs);
    return out;
}

/**
 * @brief Element-wise multiplication of two matrices
 *
 * Performs an element-wise multiplication of `rhs` and `lhs`. If one of the
 * input matrices is a row-vector (1xn) or column-vector (mx1), that matrix
 * will be broadcast appropriately. Only one matrix is allowed to be a row or
 * column vector.
 *
 * Allowed input shapes:
 *
 * (m x n) + (m x 1)
 * (m x n) + (1 x n)
 * (m x 1) + (m x n)
 * (1 x n) + (m x n)
 *
 * The resulting matrix will have shape (m x n)
 *
 * @param lhs The left-hand side matrix
 * @param rhs The right-hand side matrix
 *
 * @return A matrix that represents the element-wise multiplication of `lhs`
 *         and `rhs`. The result will have shape (m x n) where `m` is the
 *         number of rows and `n` is the number of columns.
 */
template <typename T> void multiply(MatrixStorage<T> &out, const MatrixStorage<T> &rhs, const MatrixStorage<T> &lhs) {
    apply_binary_func(out, rhs, lhs, [](const T &a, const T &b) { return a * b; });
}

template <typename T> void multiply(MatrixStorage<T> &out, const MatrixStorage<T> &lhs, const T &rhs) {
    eltwise_unary_func(out, lhs, [&rhs](const T &a) { return a * rhs; });
}

template <typename T> void multiply(MatrixStorage<T> &out, const T &lhs, const MatrixStorage<T> &rhs) {
    eltwise_unary_func(out, rhs, [&lhs](const T &a) { return lhs * a; });
}

template <typename T> MatrixStorage<T> operator*(const T &rhs, const MatrixStorage<T> &lhs) {
    MatrixStorage<T> out(lhs.shape);
    multiply(out, rhs, lhs);
    return out;
}

/**
 * @brief Element-wise out = x * y + z
 */
template <typename T>
void fma(MatrixStorage<T> &out, const MatrixStorage<T> &x, const MatrixStorage<T> &y, const MatrixStorage<T> &z) {
    assert_expected_shape(out.shape, x.shape);
    assert_expected_shape(out.shape, y.shape);
    assert_expected_shape(out.shape, z.shape);

    auto out_slices = out.slices(0, 1);
    auto x_slices = x.slices(0, 1);
    auto y_slices = y.slices(0, 1);
    auto z_slices = z.slices(0, 1);

    for (auto [out_slice, x_slice, y_slice, z_slice] : zip(out_slices, x_slices, y_slices, z_slices)) {
        for (auto [out_elem, x_elem, y_elem, z_elem] : zip(out_slice, x_slice, y_slice, z_slice)) {
            out_elem = x_elem * y_elem + z_elem;
        }
    }
}

/**
 * @brief Element-wise out = a * b + c
 */
template <typename T>
void fma(MatrixStorage<T> &out, const T &a_elem, const MatrixStorage<T> &b, const MatrixStorage<T> &c) {
    assert_expected_shape(out.shape, b.shape);
    assert_expected_shape(out.shape, c.shape);

    auto out_slices = out.slices(0, 1);
    auto b_slices = b.slices(0, 1);
    auto c_slices = c.slices(0, 1);

    for (auto [out_slice, b_slice, c_slice] : zip(out_slices, b_slices, c_slices)) {
        for (auto [out_elem, b_elem, c_elem] : zip(out_slice, b_slice, c_slice)) {
            out_elem = a_elem * b_elem + c_elem;
        }
    }
}

template <typename T>
void multiply_bwd(MatrixStorage<T> &lhs_grad, const T &rhs_data, const MatrixStorage<T> &out_grad) {
    MatrixStorage<T> tmp(lhs_grad.shape);
    multiply(tmp, rhs_data, out_grad);
    add(lhs_grad, lhs_grad, tmp);
}

template <typename T>
void multiply_bwd(MatrixStorage<T> &dx, MatrixStorage<T> &dy, const MatrixStorage<T> &x, const MatrixStorage<T> &y, const MatrixStorage<T> &dz) {
    if (x.shape == y.shape) {
        fma(dx, dz, y, dx);
        fma(dy, dz, x, dy);
    } else if (x.shape[0] == 1 and y.shape == dz.shape) {
        for (size_t c = 0; c < dz.shape[1]; c++) {
            T acc = 0;
            const T x_val = x[{0, c}];
            for (size_t r = 0; r < dz.shape[0]; r++) {
                acc += dz[{r, c}] * y[{r, c}];
                dy[{r, c}] += dz[{r, c}] * x_val;
            }
            dx[{0, c}] += acc;
        }
    } else if (x.shape[1] == 1 and y.shape == dz.shape) {
        for (size_t r = 0; r < dz.shape[0]; r++) {
            const T x_val = x[{r, 0}];
            T acc = 0;
            for (size_t c = 0; c < dz.shape[1]; c++) {
                acc  += dz[{r, c}] * y[{r, c}];
                dy[{r, c}] += dz[{r, c}] * x_val;
            }
            dx += acc;
        }
    } else if (y.shape[0] == 1 and x.shape == dz.shape) {
        for (size_t c = 0; c < dx.shape[1]; c++) {
            const T y_val = y[{0, c}];
            T acc = 0;
            for (size_t r = 0; r < dx.shape[0]; r++) {
                dx[{r, c}] += dz[{r, c}] * y_val;
                acc += dz[{r, c}] * x[{r, c}];
            }
            dy[{0, c}] += acc;
        }
     } else if (y.shape[1] == 1 and x.shape == dz.shape) {
        for (size_t r = 0; r < dx.shape[0]; r++) {
            const T y_val = y[{r, 0}];
            T acc = 0;
            for (size_t c = 0; c < dx.shape[1]; c++) {
                dx[{r, c}] += dz[{r, c}] * y_val;
                acc += dz[{r, c}] * x[{r, c}];
            }
            dy[{r, 0}] += acc;
        }
    } else {
        throw std::invalid_argument("Shapes are incompatible for multiply_bwd");
    }
}

template <typename T> void divide(MatrixStorage<T> &out, const MatrixStorage<T> &rhs, const MatrixStorage<T> &lhs) {
    apply_binary_func(out, rhs, lhs, [](const T &a, const T &b) { return a / b; });
}

template <typename T> void divide(MatrixStorage<T> &out, const MatrixStorage<T> &lhs, const T &rhs) {
    eltwise_unary_func(out, lhs, [&rhs](const T &a) { return a / rhs; });
}

template <typename T> void divide(MatrixStorage<T> &out, const T &lhs, const MatrixStorage<T> &rhs) {
    eltwise_unary_func(out, rhs, [&lhs](const T &a) { return lhs / a; });
}

template <typename T> MatrixStorage<T> operator/(const T &rhs, const MatrixStorage<T> &lhs) {
    MatrixStorage<T> out(lhs.shape);
    divide(out, rhs, lhs);
    return out;
}

template <typename T> void divide_bwd(MatrixStorage<T> &lhs_grad, MatrixStorage<T> &rhs_grad, const MatrixStorage<T> &lhs_data, const MatrixStorage<T> &rhs_data, const MatrixStorage<T> &out_grad) {
    // Gradient with respect to lhs (x): ∂(x/y)/∂x = 1/y
    // Gradient with respect to rhs (y): ∂(x/y)/∂y = -x/y^2
    for (size_t i = 0; i < lhs_grad.data.size(); i++) {
        T g = out_grad[i];
        T x = lhs_data[i];
        T y = static_cast<T>(1) / rhs_data[i];
        lhs_grad[i] += g * y;
        rhs_grad[i] -= x * g * y * y;
    }
}

/**
 * @brief Multiplies two matrices and returns the resulting matrix.
 *
 * This function performs matrix multiplication on two input matrices `lhs` and `rhs`.
 * The number of columns in the left-hand side matrix (`lhs`) must match the number of
 * rows in the right-hand side matrix (`rhs`) for the multiplication to be valid.
 *
 * The resulting matrix will have dimensions equal to the number of rows in `lhs`
 * and the number of columns in `rhs`.
 *
 * @param lhs The left-hand side matrix (vector of vectors of floats).
 *            Must have dimensions (m x n), where `m` is the number of rows and `n` is the number of columns.
 * @param rhs The right-hand side matrix (vector of vectors of floats).
 *            Must have dimensions (n x p), where `n` is the number of rows (matching `lhs` columns),
 *            and `p` is the number of columns.
 *
 * @return A matrix (vector of vectors of floats) that represents the product of `lhs` and `rhs`.
 *         The result will have dimensions (m x p), where `m` is the number of rows in `lhs`
 *         and `p` is the number of columns in `rhs`.
 *
 * @throws std::invalid_argument if the number of columns in `lhs` does not match the number of rows in `rhs`.
 */
template <typename T> void matmul_ref(MatrixStorage<T> &out, const MatrixStorage<T> &lhs, const MatrixStorage<T> &rhs) {
    const size_t lhs_rows = lhs.shape[0];
    const size_t lhs_cols = lhs.shape[1];
    const size_t rhs_rows = rhs.shape[0];
    const size_t rhs_cols = rhs.shape[1];

    if (out.id() == lhs.id()) {
        throw std::invalid_argument("out and lhs cannot point to the same data");
    }

    if (out.id() == rhs.id()) {
        throw std::invalid_argument("out and rhs cannot point to the same data");
    }

    if (lhs_cols != rhs_rows) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }

    if (lhs_rows != out.shape[0]) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }

    if (rhs_cols != out.shape[1]) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }

    for (size_t i = 0; i < lhs_rows; ++i) {
        for (size_t j = 0; j < rhs_cols; ++j) {
            for (size_t k = 0; k < lhs_cols; ++k) {
                out[{i, j}] += lhs[{i, k}] * rhs[{k, j}];
            }
        }
    }
}

template <typename T>
void gemm(const enum CBLAS_TRANSPOSE trans_a, const enum CBLAS_TRANSPOSE trans_b, MatrixStorage<T> &mat_c,
          const MatrixStorage<T> &mat_a, const MatrixStorage<T> &mat_b, T alpha, T beta) {
    size_t m = trans_a == CblasNoTrans ? mat_a.shape[0] : mat_a.shape[1];
    size_t n = trans_b == CblasNoTrans ? mat_b.shape[1] : mat_b.shape[0];
    size_t k = trans_a == CblasNoTrans ? mat_a.shape[1] : mat_a.shape[0];

    size_t lda = mat_a.shape[1];
    size_t ldb = mat_b.shape[1];
    size_t ldc = mat_c.shape[1];

    const T *a = mat_a.data.data();
    const T *b = mat_b.data.data();
    T *c = mat_c.data.data();

    if constexpr (std::is_same<T, float>::value) {
        cblas_sgemm(CblasRowMajor, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    } else if constexpr (std::is_same<T, double>::value) {
        cblas_dgemm(CblasRowMajor, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    } else {
        throw std::invalid_argument("Unsupported data type for CBLAS matmul.");
    }
}

template <typename T> void matmul(MatrixStorage<T> &out, const MatrixStorage<T> &lhs, const MatrixStorage<T> &rhs) {
    const size_t lhs_rows = lhs.shape[0];
    const size_t lhs_cols = lhs.shape[1];
    const size_t rhs_rows = rhs.shape[0];
    const size_t rhs_cols = rhs.shape[1];

    if (out.id() == lhs.id()) {
        throw std::invalid_argument("out and lhs cannot point to the same data");
    }

    if (out.id() == rhs.id()) {
        throw std::invalid_argument("out and rhs cannot point to the same data");
    }

    if (lhs_cols != rhs_rows) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }

    if (lhs_rows != out.shape[0]) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }

    if (rhs_cols != out.shape[1]) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }

    gemm(CblasNoTrans, CblasNoTrans, out, lhs, rhs, static_cast<T>(1), static_cast<T>(0));
}

template <typename T> void log(MatrixStorage<T> &y, const MatrixStorage<T> &x) {
    eltwise_unary_func(y, x, [](const T &x_i) { return std::log(x_i); });
}

template <typename T>
void log_bwd(MatrixStorage<T> &dw, const MatrixStorage<T> &x, const MatrixStorage<T> &dy) {
    // https://youtu.be/q8SA3rM6ckI?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&t=1201
    for (size_t i = 0; i < dw.data.size(); i++) {
        dw[i] += dy[i] * std::pow(x[i], static_cast<T>(-1));
    }
}

template <typename T> void tanh(MatrixStorage<T> &y, const MatrixStorage<T> &x) {
    return eltwise_unary_func(y, x, [](const T &x_i) { return std::tanh(x_i); });
}

template <typename T>
void tanh_bwd(MatrixStorage<T> &dw, const MatrixStorage<T> &y, const MatrixStorage<T> &dy) {
    for (size_t i = 0; i < dw.data.size(); i++) {
        dw.data[i] += (1 - y.data[i] * y.data[i]) * dy.data[i];
    }
}

template <typename T> void exp(MatrixStorage<T> &y, const MatrixStorage<T> &x) {
    return eltwise_unary_func(y, x, [](const T &x_i) { return std::exp(x_i); });
}

template <typename T>
void exp_bwd(MatrixStorage<T> &dw, const MatrixStorage<T> &y, const MatrixStorage<T> &dy) {
    for (size_t i = 0; i < dw.data.size(); i++) {
        dw.data[i] += y.data[i] * dy.data[i];
    }
}

/**
 * @brief Element-wise y = x ** p
 *
 * @param y The output matrix
 * @param x The input matrix
 * @param p The exponent
 */
template <typename T> void pow(MatrixStorage<T> &y, const MatrixStorage<T> &x, const T &p) {
    eltwise_unary_func(y, x, [&p](const T &x_i) { return std::pow(x_i, p); });
}

/**
 * @brief Backward pass for y = pow(x, p)
 *
 * @param dw The gradient of the output w.r.t. the input
 * @param x The input to pow(x, p
 * @param p The input to pow(x, p
 * @param dy The gradient of the output
 */
template <typename T>
void pow_bwd(MatrixStorage<T> &dw, const MatrixStorage<T> &x, const T &p, const MatrixStorage<T> &dy) {
    assert_expected_shape(dw.shape, x.shape);
    assert_expected_shape(x.shape, dy.shape);

    for (auto [dw_i, x_i, dy_i] : zip(dw.data, x.data, dy.data)) {
        dw_i += p * std::pow(x_i, p - 1) * dy_i;
    }
}

template <typename T> void sum(MatrixStorage<T> &out, const MatrixStorage<T> &in, const size_t axis) {
    if (axis > 1) {
        std::stringstream ss;
        ss << "Unknown axis=" << axis;
        throw std::invalid_argument(ss.str());
    }

    shape_type expected_shape = in.shape;
    expected_shape[axis] = 1;

    if (out.shape != expected_shape) {
        std::stringstream ss;
        ss << "Expected out.shape()=" << expected_shape << ", got " << out.shape;
        throw std::invalid_argument(ss.str());
    }

    const size_t outer_axis = axis == 0 ? 1 : 0;
    auto out_vals = out.slice(outer_axis, 0);
    auto in_slices = in.slices(outer_axis, axis);

#pragma omp parallel for
    for (auto [out_val, in_slice] : zip(out_vals, in_slices)) {
        out_val = std::accumulate(begin(in_slice), end(in_slice), static_cast<T>(0));
    }
}

template <typename T> void sum_bwd(MatrixStorage<T> &dw, const MatrixStorage<T> &dy) {
    // https://youtu.be/q8SA3rM6ckI?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&t=1746
    // lhs->grad += ones_like(lhs->data) * out->grad
    MatrixStorage<T> tmp(dw.shape, 1);
    multiply(tmp, tmp, dy);
    add(dw, dw, tmp);
}

template <typename T> void mean(MatrixStorage<T> &out, const MatrixStorage<T> &in, const size_t axis) {
    const T divisor = static_cast<T>(1) / static_cast<T>(in.shape[axis]);
    sum(out, in, axis);
    multiply(out, out, divisor);
}

template <typename T> void mean_bwd(MatrixStorage<T> &dw, const MatrixStorage<T> &dy, const size_t axis) {
    // https://youtu.be/q8SA3rM6ckI?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&t=884
    const T divisor = static_cast<T>(1) / static_cast<T>(dw.shape[axis]);
    MatrixStorage<T> tmp(dy.shape);
    multiply(tmp, dy, divisor);
    add(dw, dw, tmp);
}

template <typename T> void stddev(MatrixStorage<T> &out, const MatrixStorage<T> &in, const size_t axis) {
    const size_t off_axis = axis == 0 ? 1 : 0;
    shape_type expected_shape = in.shape;
    expected_shape[axis] = 1;

    assert_shape_not_zero(in.shape);
    assert_axis_lte(axis, 1);
    assert_expected_shape(out.shape, expected_shape);

    const T divisor = static_cast<T>(1) / static_cast<T>(in.shape[axis]);

#if 0
    auto o_itr = inner_itr(out.shape, 0, off_axis).begin();

    for (auto outer : outer_itr(in.shape, off_axis)) {
        T mu = 0;
        for (auto i_offset : inner_itr(in.shape, outer, axis)) {
            mu += in[i_offset];
        }
        mu *= divisor;

        T variance = 0;
        for (auto i_offset : inner_itr(in.shape, outer, axis)) {
            const T val = in[i_offset] - mu;
            variance += val * val;
        }
        out[*o_itr] = variance * divisor;
        ++o_itr;
    }
#else
    auto out_idx = axis == 0 ? bcast0 : bcast1;
    auto in_idx = axis == 0 ? swap_idx : identity;

#pragma omp parallel for
    for (size_t i = 0; i < in.shape[off_axis]; i++) {
        T mu = 0;
        for (size_t j = 0; j < in.shape[axis]; j++) {
            mu += in[in_idx({i, j})];
        }
        mu *= divisor;

        T variance = 0;
        for (size_t j = 0; j < in.shape[axis]; j++) {
            const T val = in[in_idx({i, j})] - mu;
            variance += val * val;
        }
        out[out_idx({i, 0})] = variance * divisor;
    }
#endif
}

template <typename T>
void stddev_bwd(MatrixStorage<T> &lhs_grad, const MatrixStorage<T> &lhs_data, const MatrixStorage<T> &prev_grad,
                const MatrixStorage<T> &out_data, size_t axis) {
    MatrixStorage<T> mu(out_data.shape);
    mean(mu, lhs_data, axis);

    const size_t off_axis = axis == 0 ? 1 : 0;
    const T two_over_n = static_cast<T>(2) / static_cast<T>(lhs_data.shape[axis]);

    auto b_idx = axis == 0 ? bcast0 : bcast1;
    auto idx = axis == 0 ? swap_idx : identity;

    // \frac{\partial \sigma}{\partial x_i} = \frac{1}{2\sigma} \cdot \frac{\partial \text{var}(x)}{\partial x_i} ]
    // where \frac{\partial \text{var}(x)}{\partial x_i} = \frac{2}{N} (x_i - \mu)

    // dsigma / dx_i = (1 / 2 * sigma) * dvar(x) / dx_i
    // dvar(x) / dx_i = (2/n) * (x[i] - mean)
    for (size_t i = 0; i < lhs_grad.shape[off_axis]; i++) {
        const T half_sigma = static_cast<T>(1) / (static_cast<T>(2) * sqrt(out_data[b_idx({i, 0})]));
        const T mu_i = mu[b_idx({i, 0})];
        const T g_i = prev_grad[b_idx({i, 0})];
        for (size_t j = 0; j < lhs_grad.shape[axis]; j++) {
            const T xi = lhs_data[idx({i, j})];
            const T dvar_dxi = two_over_n * (xi - mu_i);
            const T dsigma_dxi = half_sigma * dvar_dxi;
            lhs_grad[idx({i, j})] += dsigma_dxi * g_i;
        }
    }
}

template <typename T> void max(MatrixStorage<T> &out, std::vector<size_t> &indices, const MatrixStorage<T> &in, const size_t axis) {
    if (axis > 1) {
        std::stringstream ss;
        ss << "Unknown axis=" << axis;
        throw std::invalid_argument(ss.str());
    }

    shape_type expected_shape = in.shape;
    expected_shape[axis] = 1;
    if (out.shape != expected_shape) {
        std::stringstream ss;
        ss << "Expected out.shape()=" << expected_shape << ", got " << out.shape;
        throw std::invalid_argument(ss.str());
    }

    const size_t outer_axis = axis == 0 ? 1 : 0;
    indices.resize(in.shape[outer_axis]);

    auto max_vals = out.slice(outer_axis, 0);
    auto in_slices = in.slices(outer_axis, axis).begin();

#pragma omp parallel for
    for (size_t i = 0; i < in.shape[outer_axis]; i++) {
        auto in_slice = in.slice(axis, i);
        const auto max_elem = std::max_element(cbegin(in_slice), cend(in_slice));
        indices[i] = std::distance(cbegin(in_slice), max_elem);
        max_vals[i] = *max_elem;
    }
}

template <typename T>
void max_bwd(MatrixStorage<T> &dw, const MatrixStorage<T> &dy, const std::vector<size_t> &indices) {
    // route the gradient from out to the correct column in lhs
    MatrixStorage<T> one_h(dw.shape);
    one_hot(one_h, indices);
    multiply(one_h, one_h, dy);
    add(dw, dw, one_h);
}

template <typename T> void sqrt(MatrixStorage<T> &y, const MatrixStorage<T> &x) {
    eltwise_unary_func(y, x, [](const T &x_i) { return std::sqrt(x_i); });
}

template <typename T, typename U>
void select_rows_and_cols(MatrixStorage<T> &lhs, MatrixStorage<T> &rhs, const MatrixStorage<U> &idx) {
    if (idx.shape[1] != 2) {
        std::stringstream ss;
        ss << "Expected idx.shape=(n,2), got " << idx.shape;
        throw std::invalid_argument(ss.str());
    }

    if (lhs.shape[1] != 1) {
        std::stringstream ss;
        ss << "Expected lhs.shape=(n,1), got " << lhs.shape;
        throw std::invalid_argument(ss.str());
    }

    const shape_type expected_lhs_shape = {idx.shape[0], 1};
    assert_expected_shape(lhs.shape, expected_lhs_shape);

    const shape_type expected_rhs_shape = {idx.shape[0], rhs.shape[1]};
    assert_expected_shape(rhs.shape, expected_rhs_shape);

    const size_t rows = static_cast<size_t>(idx.shape[0]);
    for (size_t i = 0; i < rows; i++) {
        lhs[{i, 0}] = rhs[{idx[{i, 0}], idx[{i, 1}]}];
    }
}

template <typename T, typename U>
void select_rows_and_cols_bwd(MatrixStorage<T> &dw, const MatrixStorage<T> &dy,
                              const MatrixStorage<U> &idx) {
    const size_t rows = idx.shape[0];
    for (size_t i = 0; i < rows; i++) {
        const size_t r = idx[{i, 0}];
        const size_t c = idx[{i, 1}];
        dw[{r, c}] += dy[{i, 0}];
    }
};

/**
 * @brief Broadcast rows from the source matrix and assigns them to the destination matrix.
 *
 * This function broadcasts rows from the source matrix `src` based on the indices provided
 * in the `idx` matrix and assigns them to the destination matrix `dst`. The `idx` matrix
 * should contain row indices that specify which rows to select from `src`.
 *
 * @tparam T The data type of the elements in the source and destination matrices.
 * @tparam U The data type of the elements in the index matrix.
 * @param dst The destination matrix where the broadcast rows will be stored. It should have
 *            the same number of columns as `src` and the same number of rows as `idx`.
 * @param src The source matrix from which rows will be selected. It should have the same
 *            number of columns as `dst`.
 * @param idx The index matrix containing the row indices to select from `src`. It should
 *            have one column and the same number of rows as `dst`.
 * @param func A function that takes the src element and applies it to the dst element.
 *             For example, if `assign_op` is used, then lhs = rhs. If `accumulate_op` is
 *             used, then lhs += rhs.
 *
 * @throws std::invalid_argument If the shape of `idx` is not (n, 1).
 * @throws std::invalid_argument If the number of columns in `dst` does not match the number
 *                               of columns in `src`.
 */
template <typename T, typename U, typename Func, typename std::enable_if<std::is_integral<U>::value, int>::type = 0>
void broadcast_rows(MatrixStorage<T> &dst, MatrixStorage<T> &src, const MatrixStorage<U> &idx, Func func) {
    if (idx.shape[1] != 1) {
        std::stringstream ss;
        ss << "Expected idx.shape=(n,1), got " << idx.shape;
        throw std::invalid_argument(ss.str());
    }

    if (dst.shape[1] != src.shape[1]) {
        std::stringstream ss;
        ss << "Expected dst.shape=(n," << src.shape[1] << "), got " << dst.shape;
        throw std::invalid_argument(ss.str());
    }

#pragma omp parallel for
    for (auto [dst_idx, src_idx] : zip(dst.axis_iota(0), idx.slice(0, 0))) {
        auto src_row = src.slice(1, src_idx);
        auto dst_row = dst.slice(1, dst_idx);
        for (auto [d, s] : zip(dst_row, src_row)) {
            func(d, s);
        }
    }
}

shape_type select_embeddings_shape(const shape_type &a, const shape_type &b) { return shape_type({b[0], b[1] * a[1]}); }

/**
 * @brief selects embeddings from the embeddding table defined by x
 *
 * The embedding table takes a set of `m` tokens and embeds them into a lower
 * `n` dimensionsal space.
 *
 * Thus, the embeddings table, `emb` is an `m` x `n` matrix. Each row
 * corresponds to a token, and the `n` columns correspond to the
 * dimensionality of each token's embedding. Each element in the row is where
 * that token is in that dimension.
 *
 * `x` is a `i` x `j` matrix where each row corresponds to a context block, or
 * a list of `j` tokens. Each element corresponds to a row index in the
 * embeddings table.
 *
 * The output matrix will have shape `i` x (`j` * `n`):
 * * Same number of rows as x, which is one row for each context block
 * * `j` x `n` columns, which is the number of dimensions for each token x
 *   the number of tokens in a context block
 *
 * @param selected  the embeddings selected from the `emb` matrix by `x`. This
 *                  matrix must have shape (`i` x (`j` * `n`))
 * @param emb the embeddings table. This contains the lower dimensional
 *            representation of the tokens in a `m` x `n` matrix. There are
 *            `m` tokens and `n` dimensions
 * @param x the embeddings to select from emb. This is an `i` by `j` matrix,
 *          where each row corresponds to a context block, and `j` is the
 *          number of tokens in a single context block
 */
template <typename T>
void select_embeddings(MatrixStorage<T> &selected, const MatrixStorage<T> &emb, const MatrixStorage<size_t> &x) {
    // Dimensionality / number of values in a single embedding
    size_t dims = emb.shape[1]; // Number of dimensions for each embedding
    size_t x_rows = x.shape[0]; // Number of context blocks
    size_t x_cols = x.shape[1]; // Size of each context

    assert_expected_shape(selected.shape, shape_type({x_rows, x_cols * dims}));

    for (size_t x_row = 0; x_row < x_rows; x_row++) {
        for (size_t x_col = 0; x_col < x_cols; x_col++) {
            const size_t token = x[{x_row, x_col}];

            if (token >= emb.shape[0]) {
                std::stringstream ss;
                ss << "Token out of bounds: {token=" << token << "}, >= emb.shape[0]=" << emb.shape[0];
                throw std::invalid_argument(ss.str());
            }
            // Fill in the return embedding matrix, flattening out
            // the last 2 dimensions into a single row
            int offset = x_col * dims;
            for (size_t i = 0; i < dims; i++) {
                selected[{x_row, offset + i}] = emb[{token, i}];
            }
        }
    }
}

/**
 * @brief backwards pass for select_embeddings
 *
 * @param dw  output: the gradient to be applied to emb. Must be an `m` x `n`
 *              matrix
 * @param dy  dL/dy, this matrix must have shape
 *            (`i` x (`j` * `n`))
 * @param x the embeddings to select from emb. This is an `i` by `j` matrix,
 *          where each row corresponds to a context block, and `j` is the
 *          number of tokens in a single context block
 */
template <typename T>
void select_embeddings_bwd(MatrixStorage<T> &dw, const MatrixStorage<T> &dy, const MatrixStorage<size_t> &x) {
    // Dimensionality / number of values in a single embedding
    size_t dims = dw.shape[1];
    fill_value(dw, static_cast<T>(0));

    // Route the gradient from dy into dw
    for (size_t r = 0; r < x.shape[0]; r++) {
        for (size_t c = 0; c < x.shape[1]; c++) {
            const auto token = x[{r, c}];
            const auto offset = c * dims;
            for (size_t d = 0; d < dims; d++) {
                dw[{token, d}] += dy[{r, d + offset}];
            }
        }
    }
}

template <typename T>
void batchnorm_1d(MatrixStorage<T> &y, MatrixStorage<T> &mean_running, MatrixStorage<T> &std_running,
                  const MatrixStorage<T> &x, const MatrixStorage<T> &gamma, const MatrixStorage<T> &beta,
                  const T momentum, const T eps) {
    const shape_type gamma_beta_shape = {1, x.shape[1]};
    assert_expected_shape(y.shape, x.shape);
    assert_expected_shape(gamma.shape, gamma_beta_shape);
    assert_expected_shape(beta.shape, gamma_beta_shape);
    assert_expected_shape(mean_running.shape, gamma_beta_shape);
    assert_expected_shape(std_running.shape, gamma_beta_shape);

    auto x_slices = x.slices(1, 0);
    auto y_slices = y.slices(1, 0);
    auto gamma_slice = gamma.slice(1, 0);
    auto beta_slice = beta.slice(1, 0);
    auto mean_running_slice = mean_running.slice(1, 0);
    auto std_running_slice = std_running.slice(1, 0);
    const T one_minus_momentum = static_cast<T>(1) - momentum;
    const T n = static_cast<T>(x.shape[0]);

#pragma omp parallel for
    for (auto [x_slice, y_slice, gamma_i, beta_i, mean_running_i, std_running_i] :
         zip(x_slices, y_slices, gamma_slice, beta_slice, mean_running_slice, std_running_slice)) {
        const T sum = std::accumulate(begin(x_slice), end(x_slice), static_cast<T>(0));
        const T mean_i = sum / n;

        std::vector<T> diff(x.shape[0]);

        T var_i = 0;
        for (const auto [x_i, diff_i] : zip(x_slice, diff)) {
            diff_i = x_i - mean_i;
            var_i += diff_i * diff_i;
        }

        // Note: this should be var /= (n - 1) with bessel's correction. However, the batchnorm
        // paper does not use n - 1 when calculating the variance, so we don't either, which is
        // is also in line with the pytorch implementation.
        var_i /= n;

        const T stddev_i = std::sqrt(var_i + eps);
        const T std_inv_i = static_cast<T>(1) / stddev_i;

        for (auto [y_i, diff_i] : zip(y_slice, diff)) {
            y_i = gamma_i * (diff_i * std_inv_i) + beta_i;
        }

        // track the running mean and std
        mean_running_i = mean_running_i * one_minus_momentum + mean_i * momentum;
        std_running_i = std_running_i * one_minus_momentum + stddev_i * momentum;
    }
}

template <typename T>
void batchnorm_1d_bwd(MatrixStorage<T> &dx, const MatrixStorage<T> &dy, const MatrixStorage<T> &x,
                      const MatrixStorage<T> &gamma, const T eps) {
    // https://youtu.be/q8SA3rM6ckI?t=6463
    const shape_type gamma_shape = {1, dx.shape[1]};
    assert_expected_shape(dy.shape, dx.shape);
    assert_expected_shape(gamma.shape, gamma_shape);

    auto x_slices = x.slices(1, 0);
    auto dx_slices = dx.slices(1, 0);
    auto dy_slices = dy.slices(1, 0);
    auto gamma_slice = gamma.slice(1, 0);

    const T n = static_cast<T>(x.shape[0]);
    const T n1 = n / (n - 1);

#pragma omp parallel for
    for (auto [x_slice, dx_slice, dy_slice, gamma_i] : zip(x_slices, dx_slices, dy_slices, gamma_slice)) {
        const T sum = std::accumulate(begin(x_slice), end(x_slice), static_cast<T>(0));
        const T mean_i = sum / n;

        T var_i = 0;
        for (const auto x_i : x_slice) {
            const T diff_i = x_i - mean_i;
            var_i += diff_i * diff_i;
        }

        var_i /= (n - 1);

        const T std_inv_i = 1 / std::sqrt(var_i + eps);

        std::vector<T> x_hat(x.shape[0]);

        for (auto [x_i, x_hat_i] : zip(x_slice, x_hat)) {
            x_hat_i = (x_i - mean_i) * std_inv_i;
        }

        T dy_sum_i = 0;
        T dy_x_hat_sum_i = 0;
        for (const auto [dy_j, x_hat_j] : zip(dy_slice, x_hat)) {
            dy_sum_i += dy_j;
            dy_x_hat_sum_i += dy_j * x_hat_j;
        }

        const T scale = gamma_i * std_inv_i / n;

        for (auto [dx_i, dy_i, x_hat_i] : zip(dx_slice, dy_slice, x_hat)) {
            dx_i = scale * (n * dy_i - dy_sum_i - n1 * x_hat_i * dy_x_hat_sum_i);
        }
    }
}

template <typename T> std::ostream &operator<<(std::ostream &out, const MatrixStorage<T> &mat) {
    for (size_t r = 0; r < mat.shape[0]; r++) {
        for (size_t c = 0; c < mat.shape[1]; c++) {
            out << mat[{r, c}] << " ";
        }
        if (r < mat.shape[0] - 1) {
            out << std::endl;
        }
    }
    return out;
}

template <typename T> bool is_close(const T &a, const T &b) {
    const T epsilon = 1e-4;
    return std::abs(a - b) <= epsilon;
}

template <typename T> std::tuple<bool, std::string> is_close(const MatrixStorage<T> &a, const MatrixStorage<T> &b) {
    if (a.shape != b.shape) {
        std::stringstream ss;
        ss << "a.shape (" << a.shape << ") != b.shape (" << b.shape << ")";
        return {false, ss.str()};
    }

    bool close = true;
    T max_diff = 0;
    T max_a_val = 0;
    T max_b_val = 0;
    size_t max_r = 0;
    size_t max_c = 0;

    for (size_t r = 0; r < a.shape[0]; r++) {
        for (size_t c = 0; c < a.shape[1]; c++) {
            const T &a_val = a[{r, c}];
            const T &b_val = b[{r, c}];
            if (!is_close(a_val, b_val)) {
                const T diff = std::abs(a_val - b_val);
                if (close || diff > max_diff) {
                    max_diff = diff;
                    max_r = r;
                    max_c = c;
                    max_a_val = a_val;
                    max_b_val = b_val;
                    close = false;
                }
            }
        }
    }

    std::stringstream ss;

    if (!close) {
        std::stringstream coord;
        coord << "{" << max_r << "," << max_c << "}";
        ss << std::fixed << std::setprecision(16);
        ss << "a[" << coord.str() << "](" << max_a_val << ") != b[" << coord.str() << "](" << max_b_val << ")";
    }

    return {close, ss.str()};
}