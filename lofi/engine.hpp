#pragma once

#include <array>
#include <iostream>
#include <memory>
#include <ostream>
#include <random>
#include <vector>

#include <lofi/context.hpp>
#include <lofi/generator.hpp>

auto range(size_t num) {
    std::vector<size_t> out(num);
    std::iota(begin(out), end(out), 0);
    return out;
}

template <typename T> struct Matrix {
    using value_type = T;
    using context_type = Context<value_type>;
    using context_ptr_type = std::shared_ptr<context_type>;
    using shape_type = std::array<size_t, 2>;

    context_ptr_type ctx_;

    Matrix() {}
    Matrix(const shape_type &shape) : ctx_(std::make_shared<context_type>(shape)) {}
    Matrix(const shape_type &shape, const std::string &label) : ctx_(std::make_shared<context_type>(shape, label)) {}
    Matrix(context_ptr_type &&ctx) : ctx_(ctx) {}

    static Matrix zeros(const shape_type &shape) {
        Matrix out;
        out.ctx_ = std::make_shared<context_type>(shape);
        out.ctx_->fill_zeros();
        return out;
    }

    template <typename U> static Matrix zeros_like(const Matrix<U> &other) {
        return Matrix<value_type>::zeros_like(other.ctx_->data.shape);
    }

    static Matrix ones(const shape_type &shape) {
        Matrix out;
        out.ctx_ = std::make_shared<context_type>(shape);
        out.ctx_->fill_ones();
        return out;
    }

    template <typename U> static Matrix ones_like(const Matrix<U> &other) {
        return Matrix<value_type>::ones(other.ctx_->data.shape);
    }

    static Matrix one_hot(const std::vector<size_t> &xs, const size_t num_classes) {
        const shape_type shape = {xs.size(), num_classes};
        Matrix mat(shape);
        ::one_hot(mat.ctx_->data, xs);
        return mat;
    }

    Matrix(const std::vector<size_t> &rows, std::vector<size_t> &cols)
    : ctx_(std::make_shared<context_type>(shape_type({rows.size(), 2}))) {
        if (rows.size() != cols.size()) {
            throw std::invalid_argument("rows and cols must have the same size");
        }

        for (size_t i = 0; i < rows.size(); i++) {
            ctx_->data[{i, 0}] = rows[i];
            ctx_->data[{i, 1}] = cols[i];
        }
    }
    /**
     * @brief Creates a matrix with elements drawn from a normal distribution.
     *
     * This function generates a matrix of the specified dimensions (`rows` x `cols`),
     * where each element is sampled from a normal (Gaussian) distribution with the given
     * mean and variance. The distribution is generated using the provided random number
     * generator `gen`.
     *
     * @param shape A 2-D array containing (rows, columns)
     * @param cols The number of columns in the resulting matrix.
     * @param gen  A Mersenne Twister random number generator used for drawing random samples.
     * @param mean The mean of the normal distribution. Default value is 0.0.
     * @param variance The variance of the normal distribution. Default value is 1.0.
     *
     * @return A matrix (vector of vectors of floats) with dimensions (`rows` x `cols`),
     *         where each element is drawn from N(mean, variance).
     */
    static Matrix randn(const shape_type &shape, std::mt19937 &gen, float mean = 0.0, float variance = 1.0) {
        Matrix out;
        out.ctx_ = std::make_shared<context_type>(shape);
        fill_randn(out.ctx_->data, gen, mean, variance);
        return out;
    }

    Matrix operator+(Matrix &rhs) {
        Matrix out(shape());
        add(out.ctx_, ctx_, rhs.ctx_);
        return out;
    }

    Matrix operator+(Matrix &&rhs) { return *this + rhs; }

    Matrix operator+(const value_type &rhs) {
        Matrix out(shape(), std::to_string(rhs));
        add(out.ctx_, ctx_, rhs);
        return out;
    }

    Matrix operator+(const value_type &&rhs) { return *this + rhs; }

    Matrix &operator+=(Matrix &rhs) {
        Matrix out(shape());
        add(out.ctx_, ctx_, rhs.ctx_);
        ctx_ = out.ctx_;
        return *this;
    }

    Matrix &operator+=(Matrix &&rhs) {
        *this += rhs;
        return *this;
    }

    Matrix &operator+=(value_type &rhs) {
        Matrix out(shape());
        add(out.ctx_, ctx_, rhs);
        ctx_ = out.ctx_;
        return *this;
    }

    Matrix &operator+=(value_type &&rhs) {
        *this += rhs;
        return *this;
    }

    Matrix operator*(Matrix &rhs) {
        Matrix out(shape());
        multiply(out.ctx_, ctx_, rhs.ctx_);
        return out;
    }

    Matrix operator*(Matrix &&rhs) { return *this * rhs; }

    Matrix operator*(const value_type &rhs) {
        Matrix out(shape());
        multiply(out.ctx_, ctx_, rhs);
        return out;
    }

    Matrix operator*(const value_type &&rhs) { return *this * rhs; }

    Matrix &operator*=(Matrix &rhs) {
        Matrix out(shape());
        multiply(out.ctx_, ctx_, rhs.ctx_);
        ctx_ = out.ctx_;
        return *this;
    }

    Matrix &operator*=(Matrix &&rhs) { return *this *= rhs; }

    Matrix operator/(Matrix &rhs) {
        Matrix out(shape());
        divide(out.ctx_, ctx_, rhs.ctx_);
        return out;
    }

    Matrix operator/(Matrix &&rhs) { return *this / rhs; }

    Matrix operator/(const value_type &rhs) {
        Matrix out(shape());
        divide(out.ctx_, ctx_, rhs);
        return out;
    }

    Matrix operator/(const value_type &&rhs) { return *this / rhs; }

    Matrix &operator/=(Matrix &rhs) {
        Matrix out(shape());
        divide(out.ctx_, ctx_, rhs.ctx_);
        ctx_ = out.ctx_;
        return *this;
    }

    Matrix &operator/=(Matrix &&rhs) { return *this /= rhs; }

    Matrix &operator/=(const value_type &rhs) {
        Matrix out(shape());
        divide(out.ctx_, ctx_, rhs);
        ctx_ = out.ctx_;
        return *this;
    }

    Matrix &operator/=(value_type &&rhs) { return *this /= rhs; }

    Matrix operator-() {
        Matrix out(shape());
        Matrix negate({1, 1}, "-1");
        negate.data()[{0, 0}] = static_cast<value_type>(-1);
        ::multiply(out.ctx_, ctx_, negate.ctx_);
        return out;
    }

    Matrix exp() {
        Matrix out(shape());
        ::exp(out.ctx_, ctx_);
        return out;
    }

    Matrix tanh() {
        Matrix out(shape());
        ::tanh(out.ctx_, ctx_);
        return out;
    }

    Matrix log() {
        Matrix out(shape());
        ::log(out.ctx_, ctx_);
        return out;
    }

    Matrix pow(const T &x) {
        Matrix out(shape());
        ::pow(out.ctx_, ctx_, x);
        return out;
    }

    Matrix sum(size_t axis) {
        shape_type out_shape = shape();
        out_shape[axis] = 1;
        Matrix out(out_shape);
        ::sum(out.ctx_, ctx_, axis);
        return out;
    }

    Matrix max(size_t axis) {
        shape_type out_shape = shape();
        out_shape[axis] = 1;
        Matrix out(out_shape);
        ::max(out.ctx_, ctx_, axis);
        return out;
    }

    void backward() { ::backward(ctx_); }
    void zero_grad() { ::zero_grad(ctx_); }

    template <typename U> Matrix operator[](const Matrix<U> &idx) {
        Matrix out({idx.shape()[0], 1});
        ::select_rows_and_cols(out.ctx_, ctx_, idx.ctx_);
        return out;
    }

    Matrix mean(size_t axis) {
        shape_type out_shape = shape();
        out_shape[axis] = 1;
        Matrix out(out_shape);
        ::mean(out.ctx_, ctx_, axis);
        return out;
    }

    // Convenience methods to make updating context easy with similar api as original micrograd
    MatrixStorage<T> &data() { return ctx_->data; }
    const MatrixStorage<T> &data() const { return ctx_->data; }
    MatrixStorage<T> &grad() { return ctx_->grad; }
    const MatrixStorage<T> &grad() const { return ctx_->grad; }
    std::string &label() { return ctx_->label; }
    const std::string &label() const { return ctx_->label; }
    std::string &op() { return ctx_->op; }
    const std::string &op() const { return ctx_->op; }
    const shape_type &shape() const { return ctx_->data.shape; }
};

template <typename T> std::tuple<bool, std::string> is_close(const Matrix<T> &a, const Matrix<T> &b) {
    return is_close(a.ctx_, b.ctx_);
}

template <typename T> Matrix<T> matmul(Matrix<T> &lhs, Matrix<T> &rhs) {
    Matrix<T> out({lhs.shape()[0], rhs.shape()[1]});
    matmul(out.ctx_, lhs.ctx_, rhs.ctx_);
    return out;
}

template <typename T> Matrix<T> operator+(Matrix<T> &lhs, Matrix<T> &rhs) {
    const auto shape = max_shape(lhs.shape(), rhs.shape());
    Matrix<T> out(shape);
    add(out.ctx_, lhs.ctx_, rhs.ctx_);
    return out;
}

template <typename T> Matrix<T> operator+(Matrix<T> &lhs, const T &rhs) {
    Matrix<T> out(lhs.shape);
    add(out.ctx_, lhs.ctx_, rhs);
    return out;
}

template <typename T> Matrix<T> operator+(const T &lhs, Matrix<T> &rhs) {
    Matrix<T> out(rhs.shape);
    add(out.ctx_, lhs, rhs.ctx_);
    return out;
}

template <typename T> Matrix<T> operator*(Matrix<T> &lhs, Matrix<T> &rhs) {
    const auto shape = max_shape(lhs.shape(), rhs.shape());
    Matrix<T> out(shape);
    multiply(out.ctx_, lhs.ctx_, rhs.ctx_);
    return out;
}

template <typename T> Matrix<T> operator*(Matrix<T> &lhs, const T &rhs) {
    Matrix<T> out(lhs.shape());
    multiply(out.ctx_, lhs.ctx_, rhs);
    return out;
}

template <typename T> Matrix<T> operator*(const T &lhs, const Matrix<T> &rhs) {
    Matrix<T> out(rhs.shape());
    multiply(out.ctx_, lhs, rhs.ctx_);
    return out;
}