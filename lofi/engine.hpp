#pragma once

#include <array>
#include <iostream>
#include <memory>
#include <ostream>
#include <random>
#include <vector>

#include <lofi/context.hpp>
#include <lofi/generator.hpp>
#include <lofi/util.hpp>

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
        Matrix out(shape);
        fill_value(out.ctx_->data, value_type(0));
        return out;
    }

    template <typename U> static Matrix zeros_like(const Matrix<U> &other) {
        return Matrix<value_type>::zeros_like(other.ctx_->data.shape);
    }

    static Matrix ones(const shape_type &shape) {
        Matrix out(shape);
        fill_value(out.ctx_->data, value_type(1));
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

    Matrix(const std::vector<size_t> &rows, const std::vector<size_t> &cols)
        : ctx_(std::make_shared<context_type>(shape_type({rows.size(), 2}))) {
        if (rows.size() != cols.size()) {
            throw std::invalid_argument("rows and cols must have the same size");
        }

        for (size_t i = 0; i < rows.size(); i++) {
            ctx_->data[{i, 0}] = rows[i];
            ctx_->data[{i, 1}] = cols[i];
        }
    }

    Matrix(const std::vector<size_t> &rows, const Matrix<size_t> &cols)
        : ctx_(std::make_shared<context_type>(shape_type({rows.size(), 2}))) {
        if (rows.size() != cols.shape()[0]) {
            throw std::invalid_argument("rows and cols must have the same size");
        }

        if (cols.shape()[1] != 1) {
            throw std::invalid_argument("cols must be a m x 1 matrix");
        }

        for (size_t i = 0; i < rows.size(); i++) {
            ctx_->data[{i, 0}] = rows[i];
            ctx_->data[{i, 1}] = cols[{i, 0}];
        }
    }

    Matrix(Matrix<size_t> &rows, const Matrix<size_t> &cols)
        : ctx_(std::make_shared<context_type>(shape_type({rows.shape()[0], 2}))) {
        if (rows.shape()[0] != cols.shape()[0]) {
            throw std::invalid_argument("rows and cols must have the same size");
        }

        if (rows.shape()[1] != 1 || cols.shape()[1] != 1) {
            throw std::invalid_argument("rows and cols must be m x 1 matrices");
        }

        for (size_t i = 0; i < rows.shape()[0]; i++) {
            ctx_->data[{i, 0}] = rows[{i, 0}];
            ctx_->data[{i, 1}] = cols[{i, 0}];
        }
    }

    static Matrix from_file(const std::string &filename, const shape_type &shape, const std::string &label = "") {
        const auto data = read_numpy_array<value_type>(filename);
        if (data.size() != shape[0] * shape[1]) {
            std::stringstream ss;
            ss << "File `" << filename << "`: data size " << data.size() << " does not match shape " << shape;
            throw std::invalid_argument(ss.str());
        }

        Matrix out(shape, label);
        size_t i = 0;
        for (size_t r = 0; r < shape[0]; r++) {
            for (size_t c = 0; c < shape[1]; c++) {
                out.ctx_->data[{r, c}] = data[i++];
            }
        }
        return out;
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
        Matrix out(shape);
        fill_randn(out.ctx_->data, gen, mean, variance);
        return out;
    }

    template <typename U, typename std::enable_if<std::is_integral<U>::value, int>::type = 0>
    static Matrix<U> randint(U low, U high, const shape_type &shape, std::mt19937 &gen) {
        Matrix<U> out(shape);
        fill_randint(out.ctx_->data, low, high, gen);
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
        Matrix out(max_shape(shape(), rhs.shape()));
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

    Matrix operator-(Matrix &rhs) {
        Matrix out(max_shape(shape(), rhs.shape()));
        subtract(out.ctx_, ctx_, rhs.ctx_);
        return out;
    }

    Matrix operator-(Matrix &&rhs) { return *this - rhs; }

    Matrix operator-(const value_type &rhs) {
        Matrix out(shape(), std::to_string(rhs));
        subtract(out.ctx_, ctx_, rhs);
        return out;
    }

    Matrix operator-(const value_type &&rhs) { return *this - rhs; }

    Matrix operator*(Matrix &rhs) {
        Matrix out(max_shape(shape(), rhs.shape()));
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
        Matrix out(max_shape(shape(), rhs.shape()));
        multiply(out.ctx_, ctx_, rhs.ctx_);
        ctx_ = out.ctx_;
        return *this;
    }

    Matrix &operator*=(Matrix &&rhs) { return *this *= rhs; }

    Matrix &operator*=(const value_type &rhs) {
        Matrix out(shape());
        multiply(out.ctx_, ctx_, rhs);
        ctx_ = out.ctx_;
        return *this;
    }

    Matrix operator/(Matrix &rhs) {
        Matrix out(max_shape(shape(), rhs.shape()));
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
        Matrix out(max_shape(shape(), rhs.shape()));
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

    void backward(bool first_grad_to_one = true) { ::backward(ctx_, first_grad_to_one); }
    void zero_grad() { ::zero_grad(ctx_); }

    template <typename U> Matrix operator[](Matrix<U> &idx) {
        if (idx.shape()[1] == 2) {
            Matrix out({idx.shape()[0], 1});
            ::select_rows_and_cols(out.ctx_, ctx_, idx.ctx_);
            return out;
        } else if (idx.shape()[0] >= 1 && idx.shape()[1] == 1) {
            Matrix out({idx.shape()[0], shape()[1]});
            ::broadcast_rows(out.ctx_, ctx_, idx.ctx_);
            return out;
        }
        std::stringstream ss;
        ss << "Unsupported shape " << idx.shape();
        throw std::invalid_argument(ss.str());
    }

    Matrix mean(size_t axis) {
        shape_type out_shape = shape();
        out_shape[axis] = 1;
        Matrix out(out_shape);
        ::mean(out.ctx_, ctx_, axis);
        return out;
    }

    Matrix stddev(size_t axis) {
        shape_type out_shape = shape();
        out_shape[axis] = 1;
        Matrix out(out_shape);
        ::stddev(out.ctx_, ctx_, axis);
        return out;
    }

    value_type &operator[](const shape_type &idx) { return data()[idx]; }

    const value_type &operator[](const shape_type &idx) const { return data()[idx]; }

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

template <typename T, typename U> Matrix<T> select_embeddings(Matrix<T> &emb, Matrix<U> &x) {
    const shape_type shape = select_embeddings_shape(emb.shape(), x.shape());
    Matrix<T> selected(shape);
    ::select_embeddings(selected.ctx_, emb.ctx_, x.ctx_);
    return selected;
}

template <typename T> void normalize_probabilities(std::vector<T> &probs) {
    const T sum_probs = std::accumulate(probs.begin(), probs.end(), 0.0f);
    if (sum_probs < 0) {
        throw std::invalid_argument("Sum of probabilities must be greater than or equal to 0.");
    }
    const float divisor = sum_probs == 0 ? 0 : static_cast<T>(1) / sum_probs;

    // Normalize the probabilities to ensure they sum to 1
    std::vector<T> normalized_probs(probs.size());
    for (size_t i = 0; i < probs.size(); ++i) {
        if (probs[i] < 0) {
            throw std::invalid_argument("Probabilities must be non-negative.");
        }
        probs[i] *= divisor;
    }
}

/**
 * @brief Sample indexes from a 1xn row matrix using a multinomial distribution
 */
template <typename T>
std::vector<size_t> multinomial(const Matrix<T> &row_matrix_probs, int num_samples, bool replacement,
                                std::mt19937 &gen) {
    if (row_matrix_probs.shape()[0] != 1) {
        throw std::invalid_argument("row_matrix_probs must be a row vector");
    }

    // Make a clone of probabilities so that
    // * They can be modified without changing the input matrix
    // * They can be converted to a std::vector for use in std::discrete_distribution

    // TODO: create iterators for rows & columns in a matrix so that
    // matrices can be used directly in std::discrete_distribution
    // This will allow us to remove the normalization function and
    // use lofi's sum function + multiplication function to normalize
    std::vector<T> probs(row_matrix_probs.shape()[1]);
    for (size_t i = 0; i < row_matrix_probs.shape()[1]; i++) {
        probs[i] = row_matrix_probs.data()[{0, i}];
    }

    normalize_probabilities(probs);

    std::vector<size_t> samples;
    std::discrete_distribution<> dist(probs.begin(), probs.end());

    for (int i = 0; i < num_samples; ++i) {
        size_t sampled = dist(gen);
        samples.push_back(sampled);

        if (!replacement) {
            probs[sampled] = 0;
            normalize_probabilities(probs);
            dist = std::discrete_distribution<>(probs.begin(), probs.end());
        }
    }

    return samples;
}

template <typename T> struct BatchNorm1D {
    // Batchnorm paper: https://arxiv.org/abs/1502.03167
    const T eps;
    const T momentum;
    const T one_minus_momentum;
    Matrix<T> mean_running;
    Matrix<T> std_running;
    Matrix<T> gamma;
    Matrix<T> beta;
    bool training_ = true;

    BatchNorm1D(const size_t features, const T momentum = 0.1, const T eps = 1e-5) : eps(eps), momentum(momentum), one_minus_momentum(static_cast<T>(1) - momentum) {
        mean_running = Matrix<T>::zeros({1, features});
        std_running = Matrix<T>::ones({1, features});
        gamma = Matrix<T>::ones({1, features});
        beta = Matrix<T>::zeros({1, features});
    }

    Matrix<T> forward(Matrix<T> &x) {
        if (training_) {
            Matrix<T> out(x.shape());
            batchnorm_1d(out.ctx_, mean_running.ctx_, std_running.ctx_, x.ctx_, gamma.ctx_, beta.ctx_, momentum, eps);
            return out;
        } else {
            return forward_inference(x);
        }
    }

    Matrix<T> forward_inference(Matrix<T> &x) {
        auto std_running_inv = std_running.pow(-1.0f);
        auto xhat = gamma * (x - mean_running) * std_running_inv + beta;
        return xhat;
    }

    void training(bool tr) { training_ = tr; }

    std::vector<Matrix<T>> parameters() { return {gamma, beta}; }
};

template <typename T> struct CrossEntropyLoss {
    template<typename U>
    auto forward(Matrix<T> &logits, Matrix<U> &labels) {
        Matrix<T> out({1, 1});
        cross_entropy_loss(out.ctx_, logits.ctx_, labels.ctx_, 1);
        return out;
    }
};