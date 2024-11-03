/**
 * @brief computational graph for automatic differentiation
 */
#pragma once

#include <functional>
#include <iostream>
#include <memory>
#include <queue>
#include <string>
#include <unordered_set>
#include <vector>

#include <lofi/storage.hpp>

/**
 * @brief A node in a graph
 *
 * Holds the "context" for a node in the graph. The context contains the output
 * from the operation and the gradient from the backward pass.
 */
template <typename T> struct Context {
    using value_type = T;
    using storage_type = MatrixStorage<T>;
    using shape_type = std::array<size_t, 2>;

    storage_type data;
    storage_type grad;

    std::string label;
    std::string op;
    std::function<void()> backward = []() {};
    std::vector<std::shared_ptr<Context>> prev;
    ssize_t degrees = 0;

    // Used for backward pass for max()
    std::vector<size_t> indices;

    Context(const shape_type &shape) : data(shape), grad(shape) {}
    Context(const shape_type &shape, const std::string &label) : data(shape), grad(shape), label(label) {}
    const shape_type &shape() const { return data.shape; }
};

/**
 * @brief element-wise addition of two matrices
 */
template <typename T>
void add(std::shared_ptr<Context<T>> &out, std::shared_ptr<Context<T>> &lhs, std::shared_ptr<Context<T>> &rhs) {
    add(out->data, lhs->data, rhs->data);
    out->prev = {lhs, rhs};
    lhs->degrees++;
    rhs->degrees++;
    out->op = "+";
    out->backward = [out, lhs, rhs]() {
        add(lhs->grad, lhs->grad, out->grad);
        add(rhs->grad, rhs->grad, out->grad);
    };
}

/**
 * @brief element-wise subtraction of two matrices
 */
template <typename T>
void subtract(std::shared_ptr<Context<T>> &out, std::shared_ptr<Context<T>> &lhs, std::shared_ptr<Context<T>> &rhs) {
    subtract(out->data, lhs->data, rhs->data);
    out->prev = {lhs, rhs};
    lhs->degrees++;
    rhs->degrees++;
    out->op = "-";

    out->backward = [out, lhs, rhs]() {
        add(lhs->grad, lhs->grad, out->grad);
        add(rhs->grad, rhs->grad, out->grad);
    };
}

/**
 * @brief element-wise multiplication of two matrices
 */
template <typename T>
void multiply(std::shared_ptr<Context<T>> &out, std::shared_ptr<Context<T>> &lhs, std::shared_ptr<Context<T>> &rhs) {
    multiply(out->data, lhs->data, rhs->data);
    out->prev = {lhs, rhs};
    lhs->degrees++;
    rhs->degrees++;
    out->op = "*";
    out->backward = [out, lhs, rhs]() {
        // (m x n) = (m x n) * (m x n)
        MatrixStorage<T> lhs_grad(lhs->grad.shape);
        MatrixStorage<T> rhs_grad(rhs->grad.shape);
        multiply(lhs_grad, rhs->data, out->grad);
        multiply(rhs_grad, lhs->data, out->grad);
        add(lhs->grad, lhs->grad, lhs_grad);
        add(rhs->grad, rhs->grad, rhs_grad);
    };
}

/**
 * @brief element-wise division of two matrices
 */
template <typename T>
void divide(std::shared_ptr<Context<T>> &out, std::shared_ptr<Context<T>> &lhs, std::shared_ptr<Context<T>> &rhs) {
    divide(out->data, lhs->data, rhs->data);
    out->prev = {lhs, rhs};
    lhs->degrees++;
    rhs->degrees++;
    out->op = "/";
    out->backward = [out, lhs, rhs]() {
        // (m x n) = (m x n) * (m x n)
        MatrixStorage<T> lhs_grad(lhs->grad.shape);
        MatrixStorage<T> rhs_grad(rhs->grad.shape);
        divide(rhs_grad, lhs->data, out->grad);
        divide(lhs_grad, rhs->data, out->grad);
        add(lhs->grad, lhs->grad, lhs_grad);
        add(rhs->grad, rhs->grad, rhs_grad);
    };
}

/**
 * @brief Matrix multiplication
 */
template <typename T>
void matmul(std::shared_ptr<Context<T>> &out, std::shared_ptr<Context<T>> &lhs, std::shared_ptr<Context<T>> &rhs) {
    matmul(out->data, lhs->data, rhs->data);
    out->prev = {lhs, rhs};
    lhs->degrees++;
    rhs->degrees++;
    out->op = "@";
    out->backward = [out, lhs, rhs]() {
        const auto &l_shape = lhs->data.shape;
        const auto &r_shape = rhs->data.shape;

        MatrixStorage<T> rhs_t({r_shape[1], r_shape[0]});
        MatrixStorage<T> lhs_t({l_shape[1], l_shape[0]});
        MatrixStorage<T> lhs_grad(l_shape);
        MatrixStorage<T> rhs_grad(r_shape);

        // lhs->grad += out->grad @ rhs_t->data
        // rhs->grad += lhs_t->data @ out->grad

        transpose(rhs_t, rhs->data);
        transpose(lhs_t, lhs->data);
        matmul(lhs_grad, out->grad, rhs_t);
        matmul(rhs_grad, lhs_t, out->grad);
        add(lhs->grad, lhs->grad, lhs_grad);
        add(rhs->grad, rhs->grad, rhs_grad);
    };
}

/**
 * @brief element-wise tanh of a matrix
 */
template <typename T> void tanh(std::shared_ptr<Context<T>> &out, std::shared_ptr<Context<T>> &lhs) {
    tanh(out->data, lhs->data);
    out->prev = {lhs};
    lhs->degrees++;
    out->op = "tanh";
    out->backward = [out, lhs]() {
        // lhs->grad += (1 - out->data * out->data) * out->grad;
        MatrixStorage<T> out2(out->data.shape);
        MatrixStorage<T> ones({out2.shape, static_cast<T>(1)});
        multiply(out2, out->data, out->data);
        subtract(out2, 1, out2);
        multiply(out2, out2, out->grad);
        add(lhs->grad, lhs->grad, out2);
    };
}

/**
 * @brief element-wise exp of a matrix
 */
template <typename T> void exp(std::shared_ptr<Context<T>> &out, std::shared_ptr<Context<T>> &lhs) {
    exp(out->data, lhs->data);
    out->prev = {lhs};
    lhs->degrees++;
    out->op = "exp";
    out->backward = [out, lhs]() {
        // lhs->grad += out->data * out->grad;
        MatrixStorage<T> tmp(out->data.shape);
        multiply(lhs->grad, out->data, out->grad);
        add(lhs->grad, lhs->grad, tmp);
    };
}

/**
 * @brief element-wise log base 2 of a matrix
 */
template <typename T> void log(std::shared_ptr<Context<T>> &out, std::shared_ptr<Context<T>> &lhs) {
    log(out->data, lhs->data);
    out->prev = {lhs};
    lhs->degrees++;
    out->op = "log";
    out->backward = [out, lhs]() {
        // https://youtu.be/q8SA3rM6ckI?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&t=1201
        // lhs->grad += (1 / lhs->data) * out->grad;
        // or lhs->grad += (out->grad / lhs->data)
        MatrixStorage<T> out2(out->data.shape);
        divide(out2, out->grad, lhs->data);
        add(lhs->grad, lhs->grad, out2);
    };
}

/**
 * @brief element-wise pow (lhs ** rhs) of a matrix
 */
template <typename T> void pow(std::shared_ptr<Context<T>> &out, std::shared_ptr<Context<T>> &lhs, const T &rhs) {
    pow(out->data, lhs->data, rhs);
    out->prev = {lhs};
    lhs->degrees++;
    std::stringstream ss;
    ss << "pow(" << rhs << ")";
    out->op = ss.str();
    out->backward = [out, lhs, rhs]() {
        // lhs->grad += rhs * std::pow(lhs->data, rhs - 1) * out->grad;
        MatrixStorage<float> tmp({lhs->shape()});
        pow(tmp, lhs->data, rhs - static_cast<T>(1));
        multiply(tmp, rhs, tmp);
        multiply(tmp, tmp, out->grad);
        add(lhs->grad, lhs->grad, tmp);
    };
}

/**
 * @brief select rows and columns from a matrix
 */
template <typename T, typename U>
void select_rows_and_cols(std::shared_ptr<Context<T>> &out, std::shared_ptr<Context<T>> &lhs,
                          const std::shared_ptr<Context<U>> &idx) {
    select_rows_and_cols(out->data, lhs->data, idx->data);
    out->prev = {lhs};
    lhs->degrees++;
    out->op = "select";
    out->backward = [out, lhs, idx]() {
        const size_t rows = idx->shape()[0];
        const MatrixStorage<U> &idx_d = idx->data;

        for (size_t i = 0; i < rows; i++) {
            const size_t r = idx_d[{i, 0}];
            const size_t c = idx_d[{i, 1}];
            lhs->grad[{r, c}] += out->grad[{i, 0}];
        }
    };
}

/**
 * @brief sum a matrix along an axis
 */
template <typename T> void sum(std::shared_ptr<Context<T>> &out, std::shared_ptr<Context<T>> &lhs, const size_t axis) {
    sum(out->data, lhs->data, axis);
    out->prev = {lhs};
    lhs->degrees++;
    std::stringstream ss;
    ss << "sum(" << axis << ")";
    out->op = ss.str();

    out->backward = [out, lhs]() {
        // https://youtu.be/q8SA3rM6ckI?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&t=1746
        // lhs->grad += ones_like(lhs->data) * out->grad
        MatrixStorage<T> tmp(lhs->shape(), 1);
        multiply(tmp, tmp, out->grad);
        add(lhs->grad, lhs->grad, tmp);
    };
}

template <typename T> void mean(std::shared_ptr<Context<T>> &out, std::shared_ptr<Context<T>> &lhs, const size_t axis) {
    mean(out->data, lhs->data, axis);
    out->prev = {lhs};
    lhs->degrees++;
    std::stringstream ss;
    ss << "mean(" << axis << ")";
    out->op = ss.str();

    out->backward = [out, lhs, axis]() {
        // https://youtu.be/q8SA3rM6ckI?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&t=884
        const T divisor = static_cast<T>(1) / static_cast<T>(lhs->shape()[axis]);
        MatrixStorage<T> tmp({out->grad.shape});
        multiply(tmp, out->grad, divisor);
        add(lhs->grad, lhs->grad, tmp);
    };
}

template <typename T>
void max(std::shared_ptr<Context<T>> &out, std::shared_ptr<Context<T>> &lhs, const size_t axis) {
    max(out->data, out->indices, lhs->data, axis);
    out->prev = {lhs};
    lhs->degrees++;
    std::stringstream ss;
    ss << "max(axis=" << axis << ")";
    out->op = ss.str();
    out->backward = [out, lhs, axis]() {
        // route the gradient from out to the correct column in lhs
        MatrixStorage<T> one_h(lhs->shape);
        one_hot(one_h, out->indices);
        multiply(one_h, one_h, out->grad);
        add(lhs->grad, lhs->grad, one_h);
    };
}

template <typename T, typename Func> void bfs(std::shared_ptr<Context<T>> &root, Func func) {
    using ptr_t = std::shared_ptr<Context<T>>;
    std::unordered_set<ptr_t> visited;
    std::queue<ptr_t> q;

    root->degrees = 0;
    q.push(root);
    visited.insert(root);

    while (q.size() > 0) {
        auto ctx = q.front();
        q.pop();
        func(ctx);

        for (auto &child : ctx->prev) {
            if (!visited.contains(child)) {
                q.push(child);
                visited.insert(child);
            }
        }
    }
}

template <typename T, typename Func> void topo(std::shared_ptr<Context<T>> &root, Func func) {
    using ptr_t = std::shared_ptr<Context<T>>;
    std::unordered_set<ptr_t> visited;
    std::queue<ptr_t> q;

    root->degrees = 0;
    q.push(root);
    visited.insert(root);

    while (q.size() > 0) {
        auto ctx = q.front();
        q.pop();
        func(ctx);

        for (auto &child : ctx->prev) {
            child->degrees--;
            if (child->degrees == 0 && !visited.contains(child)) {
                q.push(child);
                visited.insert(child);
            }
        }
    }
}
template <typename T> void backward(std::shared_ptr<Context<T>> &root) {
    topo(root, [](auto &ctx) { ctx->backward(); });
}

template <typename T> void zero_grad(std::shared_ptr<Context<T>> &root) {
    bfs(root, [](auto &ctx) { ctx->grad = static_cast<T>(0); });
}

template <typename T>
std::tuple<bool, std::string> is_close(const std::shared_ptr<Context<T>> &a, const std::shared_ptr<Context<T>> &b) {
    const auto [data_close, data_close_str] = is_close(a->data, b->data);
    const auto [grad_close, grad_close_str] = is_close(a->grad, b->grad);

    if (!data_close || !grad_close) {
        std::stringstream ss;
        ss << "data: `" << data_close_str << "`, grad: `" << grad_close_str << "`";
        return {false, ss.str()};
    }

    return {true, ""};
}