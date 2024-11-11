// TODO: Maybe MatrixStorage should be broken into multiple classes.
// * The matrix class that has functions that perform the matrix operations, specifically,
//   the non-autograd style functions.
// * Storage management, which handles memory management and element access.
#pragma once
#include <array>
#include <cstdint>
#include <iomanip>
#include <ostream>
#include <random>
#include <sstream>
#include <vector>

#include <lofi/generator.hpp>

using shape_type = std::array<size_t, 2>;

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
    ss << "[" << file << ":" << line << "] : Expected shape=" << expected << ", got " << actual;
}

void _assert_expected_shape(const shape_type &actual, const shape_type &expected, const char *file, const int line) {
    if (actual != expected) {
        throw_unexpected_shape(actual, expected, file, line);
    }
}

#define assert_expected_shape(a, b) _assert_expected_shape(a, b, __FILE__, __LINE__)

template <typename T> struct MatrixStorage {
    using value_type = T;
    using shape_type = std::array<size_t, 2>;

    shape_type shape = {0, 0};
    std::vector<T> data;

    MatrixStorage() {}

    // Call .clone() explicitly
    MatrixStorage(const MatrixStorage &) = delete;
    MatrixStorage &operator=(const MatrixStorage &rhs) = delete;

    MatrixStorage(const shape_type &shape) : shape(shape), data(shape[0] * shape[1]) {}
    MatrixStorage(const shape_type &shape, value_type val) : shape(shape), data(shape[0] * shape[1]) {
        fill_value<value_type>(*this, val);
    }

    MatrixStorage(MatrixStorage &&rhs) : shape(std::move(rhs.shape)), data(std::move(rhs.data)) { rhs.shape = {0, 0}; }

    MatrixStorage &operator=(MatrixStorage &&rhs) {
        if (this != &rhs) {
            shape = std::move(rhs.shape);
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
        out.data = data;
        return out;
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

    MatrixStorage operator*(const MatrixStorage &rhs) {
        MatrixStorage out(shape);
        multiply(out, *this, rhs);
        return out;
    }

    MatrixStorage operator*(const value_type &rhs) {
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

    const size_t offset(const shape_type &idx) const { return idx[0] * shape[1] + idx[1]; }

    value_type &operator[](const shape_type &idx) { return data[offset(idx)]; }

    const value_type &operator[](const shape_type &idx) const { return data[offset(idx)]; }

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
    for (size_t i = 0; i < mat.shape[0]; ++i) {
        for (size_t j = 0; j < mat.shape[1]; ++j) {
            mat[{i, j}] = dist(gen);
        }
    }
}

template <typename T> void fill_randn(MatrixStorage<T> &mat, T mean = 0.0, T variance = 1.0) {    auto gen = generator();
    return fill_randn(mat, gen, mean, variance);
}

template <typename T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
void fill_randint(MatrixStorage<T> &mat, T low, T high, std::mt19937 &gen) {
    std::uniform_int_distribution<T> dist(low, high - 1);

    for (size_t i = 0; i < mat.shape[0]; ++i) {
        for (size_t j = 0; j < mat.shape[1]; ++j) {
            mat[{i, j}] = dist(gen);
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

    auto idx_func = choose_bcast(rhs.shape);

    for (size_t r = 0; r < lhs.shape[0]; r++) {
        for (size_t c = 0; c < lhs.shape[1]; c++) {
            lhs[{r, c}] = rhs[idx_func({r, c})];
        }
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

    for (size_t i = 0; i < in.shape[0]; i++) {
        for (size_t j = 0; j < in.shape[1]; j++) {
            out[{j, i}] = in[{i, j}];
        }
    }
}

template <typename T, typename Func>
void eltwise_unary_func(MatrixStorage<T> &out, const MatrixStorage<T> &in, Func func) {
    if (out.shape != in.shape) {
        std::stringstream ss;
        ss << "out.shape=" << out.shape << " != in.shape=" << in.shape;
        throw std::invalid_argument(ss.str());
    }

    for (size_t r = 0; r < out.shape[0]; r++) {
        for (size_t c = 0; c < out.shape[1]; c++) {
            out[{r, c}] = func(in[{r, c}]);
        }
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

    for (size_t i = 0; i < lhs_rows; ++i) {
        for (size_t j = 0; j < rhs_cols; ++j) {
            for (size_t k = 0; k < lhs_cols; ++k) {
                out[{i, j}] += lhs[{i, k}] * rhs[{k, j}];
            }
        }
    }
}

template <typename T> void log(MatrixStorage<T> &out, const MatrixStorage<T> &in) {
    eltwise_unary_func(out, in, [](const T &x) { return std::log(x); });
}

template <typename T> void tanh(MatrixStorage<T> &out, const MatrixStorage<T> &in) {
    return eltwise_unary_func(out, in, [](const T &x) { return std::tanh(x); });
}

template <typename T> void exp(MatrixStorage<T> &out, const MatrixStorage<T> &in) {
    return eltwise_unary_func(out, in, [](const T &x) { return std::exp(x); });
}

template <typename T> void pow(MatrixStorage<T> &out, const MatrixStorage<T> &in, const T &y) {
    eltwise_unary_func(out, in, [&y](const T &x) { return std::pow(x, y); });
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

    auto out_idx = axis == 0 ? bcast0 : bcast1;
    fill_value(out, static_cast<T>(0));

    for (size_t r = 0; r < in.shape[0]; r++) {
        for (size_t c = 0; c < in.shape[1]; c++) {
            out[out_idx({r, c})] += in[{r, c}];
        }
    }
}

template <typename T> void mean(MatrixStorage<T> &out, MatrixStorage<T> &in, const size_t axis) {
    const T divisor = static_cast<T>(1) / static_cast<T>(in.shape[axis]);
    sum(out, in, axis);
    multiply(out, out, divisor);
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

    auto out_idx = axis == 0 ? bcast0 : bcast1;
    auto in_idx = axis == 0 ? swap_idx : identity;
    size_t off_axis = axis == 0 ? 1 : 0;
    indices.resize(in.shape[off_axis]);

    for (size_t i = 0; i < in.shape[off_axis]; i++) {
        T max_val = in[in_idx({i, 0})];
        size_t max_idx = 0;
        for (size_t j = 1; j < in.shape[axis]; j++) {
            const T in_val = in[in_idx({i, j})];
            if (in_val > max_val) {
                max_val = in_val;
                max_idx = j;
            }
        }

        out[out_idx({i, 0})] = max_val;
        indices[i] = max_idx;
    }
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

    const size_t rows = static_cast<size_t>(idx.shape[0]);
    for (size_t i = 0; i < rows; i++) {
        lhs[{i, 0}] = rhs[{idx[{i, 0}], idx[{i, 1}]}];
    }
}

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

    for (size_t dst_r = 0; dst_r < dst.shape[0]; dst_r++) {
        const size_t src_r = idx[{dst_r, 0}];
        for (size_t c = 0; c < src.shape[1]; c++) {
            func(dst[{dst_r, c}], src[{src_r, c}]);
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
 * @param demb  output: the gradient to be applied to emb. Must be an `m` x `n`
 *              matrix
 * @param dselected  dL/dselected, this matrix must have shape
 *            (`i` x (`j` * `n`))
 * @param x the embeddings to select from emb. This is an `i` by `j` matrix,
 *          where each row corresponds to a context block, and `j` is the
 *          number of tokens in a single context block
 */
template <typename T>
void select_embeddings_bwd(MatrixStorage<T> &demb, const MatrixStorage<T> &dselected, const MatrixStorage<size_t> &x) {
    // Dimensionality / number of values in a single embedding
    size_t dims = demb.shape[1];
    fill_value(demb, static_cast<T>(0));

    // Route the gradient from dselected into demb
    for (size_t r = 0; r < x.shape[0]; r++) {
        for (size_t c = 0; c < x.shape[1]; c++) {
            const auto token = x[{r, c}];
            const auto offset = c * dims;
            for (size_t d = 0; d < dims; d++) {
                demb[{token, d}] += dselected[{r, d + offset}];
            }
        }
    }
}

template <typename T> std::ostream &operator<<(std::ostream &out, const MatrixStorage<T> &mat) {
    for (size_t r = 0; r < mat.shape[0]; r++) {
        for (size_t c = 0; c < mat.shape[1]; c++) {
            out << mat[{r, c}] << " ";
        }
        out << std::endl;
    }
    return out;
}

template <typename T> bool is_close(const T &a, const T &b) {
    const T epsilon = 1e-6;
    return std::abs(a - b) <= epsilon;
}

template <typename T> std::tuple<bool, std::string> is_close(const MatrixStorage<T> &a, const MatrixStorage<T> &b) {
    if (a.shape != b.shape) {
        std::stringstream ss;
        ss << "a.shape (" << a.shape << ") != b.shape (" << b.shape << ")";
        return {false, ss.str()};
    }

    for (size_t r = 0; r < a.shape[0]; r++) {
        for (size_t c = 0; c < a.shape[1]; c++) {
            const T &a_val = a[{r, c}];
            const T &b_val = b[{r, c}];
            if (!is_close(a_val, b_val)) {
                std::stringstream coord;
                coord << "{" << r << "," << c << "}";
                std::stringstream ss;
                ss << std::fixed << std::setprecision(16);
                ss << "a[" << coord.str() << "](" << a_val << ") != b[" << coord.str() << "](" << b_val << ")";
                return {false, ss.str()};
            }
        }
    }

    return {true, ""};
}