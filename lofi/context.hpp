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

static size_t ref_count = 0;

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

    Context() = delete;
    Context(const Context &) = delete;
    Context &operator=(const Context &) = delete;

    Context(const shape_type &shape) : data(shape), grad(shape) { ref_count++; }
    Context(const shape_type &shape, const std::string &label) : data(shape), grad(shape), label(label) { ref_count++; }
    const shape_type &shape() const { return data.shape; }
    ~Context() { ref_count--; }
};

/**
 * @brief Convert a single shared pointer into a weak pointer
 */
template <typename T>
auto make_one_weak(const std::shared_ptr<T>& sp) {
    return std::weak_ptr<T>(sp);
}

/**
 * @brief Convert N shared pointers into N weak pointers
 */
template <typename... Args>
auto make_weak(Args... shared_ptrs) {
    return std::make_tuple(make_one_weak(shared_ptrs)...);
}

/**
 * @brief Convert a single weak pointer into a shared pointer
 */
template <typename T>
auto lock_one_weak(const std::weak_ptr<T>& wp) {
    if (auto sp = wp.lock()) {
        return sp;
    } else {
        throw std::runtime_error("weak_ptr is expired");
    }
}

/**
 * @brief Convert N weak pointers into N shared pointers
 */
template <typename... Ts>
auto lock_weak(const std::tuple<std::weak_ptr<Ts>...>& weak_tuple) {
    return std::apply([](const auto&... weak_ptrs) {
        return std::make_tuple(lock_one_weak(weak_ptrs)...);
    }, weak_tuple);
}

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
    auto weak = make_weak(out, lhs, rhs);
    out->backward = [weak]() {
        auto [out, lhs, rhs] = lock_weak(weak);
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

    auto weak = make_weak(out, lhs, rhs);
    out->backward = [weak]() {
        auto [out, lhs, rhs] = lock_weak(weak);
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
    auto weak = make_weak(out, lhs, rhs);
    out->backward = [weak]() {
        // (m x n) = (m x n) * (m x n)
        auto [out, lhs, rhs] = lock_weak(weak);
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
    auto weak = make_weak(out, lhs, rhs);
    out->backward = [weak]() {
        // (m x n) = (m x n) * (m x n)
        auto [out, lhs, rhs] = lock_weak(weak);

        // Gradient with respect to lhs (x): ∂(x/y)/∂x = 1/y
        // Gradient with respect to rhs (y): ∂(x/y)/∂y = -x/y^2
        for (size_t i = 0; i < lhs->grad.data.size(); i++) {
            T g = out->grad.data[i];
            T x = lhs->data.data[i];
            T y = static_cast<T>(1) / rhs->data.data[i];
            lhs->grad.data[i] += g * y;
            rhs->grad.data[i] -= x * g * y * y;
        }
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
    auto weak = make_weak(out, lhs, rhs);
    out->backward = [weak]() {
        auto [out, lhs, rhs] = lock_weak(weak);
        const T one(1);
        gemm(CblasNoTrans, CblasTrans, lhs->grad, out->grad, rhs->data, one, one);
        gemm(CblasTrans, CblasNoTrans, rhs->grad, lhs->data, out->grad, one, one);
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
    auto weak = make_weak(out, lhs);
    out->backward = [weak]() {
        auto [out, lhs] = lock_weak(weak);
        for (size_t i = 0; i < lhs->grad.data.size(); i++) {
            lhs->grad.data[i] += (1 - out->data.data[i] * out->data.data[i]) * out->grad.data[i];
        }
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
    auto weak = make_weak(out, lhs);
    out->backward = [weak]() {
        // lhs->grad += out->data * out->grad;
        auto [out, lhs] = lock_weak(weak);
        for (size_t i = 0; i < lhs->grad.data.size(); i++) {
            lhs->grad.data[i] += out->data.data[i] * out->grad.data[i];
        }
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
    auto weak = make_weak(out, lhs);
    out->backward = [weak]() {
        // https://youtu.be/q8SA3rM6ckI?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&t=1201
        // lhs->grad += (1 / lhs->data) * out->grad;
        // or lhs->grad += (out->grad / lhs->data)
        auto [out, lhs] = lock_weak(weak);
        for (size_t i = 0; i < lhs->grad.data.size(); i++) {
            lhs->grad.data[i] += out->grad.data[i] * std::pow(lhs->data.data[i], static_cast<T>(-1));
        }
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
    auto weak = make_weak(out, lhs);
    out->backward = [weak, rhs]() {
        // lhs->grad += rhs * std::pow(lhs->data, rhs - 1) * out->grad;
        auto [out, lhs] = lock_weak(weak);
        for (size_t i = 0; i < lhs->grad.data.size(); i++) {
            lhs->grad.data[i] += rhs * std::pow(lhs->data.data[i], rhs - 1) * out->grad.data[i];
        }
    };
}

/**
 * @brief select rows and columns from a matrix
 */
template <typename T, typename U>
void select_rows_and_cols(std::shared_ptr<Context<T>> &out, std::shared_ptr<Context<T>> &lhs,
                          std::shared_ptr<Context<U>> &idx) {
    select_rows_and_cols(out->data, lhs->data, idx->data);
    out->prev = {lhs};
    lhs->degrees++;
    out->op = "select";
    auto weak = make_weak(out, lhs);
    out->backward = [weak, idx]() {
        auto [out, lhs] = lock_weak(weak);
        const size_t rows = idx->shape()[0];
        const MatrixStorage<U> &idx_d = idx->data;

        for (size_t i = 0; i < rows; i++) {
            const size_t r = idx_d[{i, 0}];
            const size_t c = idx_d[{i, 1}];
            lhs->grad[{r, c}] += out->grad[{i, 0}];
        }
    };
}

template <typename T, typename U>
void select_embeddings(std::shared_ptr<Context<T>> &out, std::shared_ptr<Context<T>> &emb,
                       std::shared_ptr<Context<U>> &idx) {
    select_embeddings(out->data, emb->data, idx->data);
    out->prev = {emb};
    emb->degrees++;
    out->op = "select";
    auto weak = make_weak(out, emb);
    out->backward = [weak, idx]() {
        auto [out, emb] = lock_weak(weak);
        select_embeddings_bwd(emb->grad, out->grad, idx->data);
    };
}

/**
 * @brief broads rows from one matrix to another matrix
 */
template <typename T, typename U>
void broadcast_rows(std::shared_ptr<Context<T>> &out, std::shared_ptr<Context<T>> &src,
                    std::shared_ptr<Context<U>> &idx) {
    broadcast_rows(out->data, src->data, idx->data, assign_op<T>);
    out->prev = {src};
    src->degrees++;
    out->op = "broadcast";
    auto weak = make_weak(out, src);
    out->backward = [weak, idx]() {
        auto [out, src] = lock_weak(weak);
        broadcast_rows(src->grad, out->grad, idx->data, accumulate_op<T>);
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

    auto weak = make_weak(out, lhs);
    out->backward = [weak]() {
        // https://youtu.be/q8SA3rM6ckI?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&t=1746
        // lhs->grad += ones_like(lhs->data) * out->grad
        auto [out, lhs] = lock_weak(weak);
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

    auto weak = make_weak(out, lhs);
    out->backward = [weak, axis]() {
        // https://youtu.be/q8SA3rM6ckI?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&t=884
        auto [out, lhs] = lock_weak(weak);
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
    auto weak = make_weak(out, lhs);
    out->backward = [weak, axis]() {
        // route the gradient from out to the correct column in lhs
        auto [out, lhs] = lock_weak(weak);
        MatrixStorage<T> one_h(lhs->shape());
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

template <typename T> void backward(std::shared_ptr<Context<T>> &root, bool first_grad_to_one = true) {
    if (first_grad_to_one) {
        root->grad = static_cast<T>(1);
    }
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