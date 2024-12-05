#pragma once
#include <lofi/engine.hpp>
#include <random>
#include <vector>

using std::begin;
using std::end;


template <typename T> struct Module {
    bool training_ = true;

    virtual Matrix<T> forward(Matrix<T> &input) { throw std::runtime_error("Not implemented"); };
    virtual std::vector<Matrix<T>> parameters() { return {}; }
    virtual void training(bool tr) { training_ = tr; }
};

template <typename T> struct Sequential : Module<T> {
    using module_ptr_t = std::shared_ptr<Module<T>>;
    std::vector<module_ptr_t> modules;

    Sequential(std::vector<module_ptr_t> &modules) : modules(modules) {}

    Matrix<T> forward(Matrix<T> &input) {
        auto out = input;
        for (auto &module : modules) {
            out = module->forward(out);
        }
        return out;
    }

    std::vector<Matrix<T>> parameters() {
        std::vector<Matrix<T>> params;
        for (auto &module : modules) {
            auto module_params = module->parameters();
            params.insert(end(params), begin(module_params), end(module_params));
        }
        return params;
    }

    void training(bool tr) {
        Module<T>::training(tr);
        for (auto &module : modules) {
            module->training(tr);
        }
    }
};

template <typename T> struct Linear : Module<T> {
    Matrix<T> weight;
    Matrix<T> bias;
    Matrix<T> out;
    bool use_bias = true;

    Linear(size_t in_features, size_t out_features, bool use_bias, std::mt19937 g) : use_bias(use_bias) {
        weight = Matrix<T>::randn({in_features, out_features}, g);
        bias = Matrix<T>::randn({1, out_features}, g);
    }

    Matrix<T> forward(Matrix<T> &input) {
        if (use_bias) {
            out = matmul(input, weight) + bias;
        } else {
            out = matmul(input, weight);
        }
        return out;
    }

    std::vector<Matrix<T>> parameters() {
        if (use_bias) {
            return {weight, bias};
        }
        return {weight};
    }
};

template <typename T> struct Tanh : Module<T> {
    Matrix<T> forward(Matrix<T> &input) { return input.tanh(); }
    std::vector<Matrix<T>> parameters() { return {}; }
};

template <typename T> struct BatchNorm1D : Module<T> {
    // Batchnorm paper: https://arxiv.org/abs/1502.03167
    const T eps;
    const T momentum;
    const T one_minus_momentum;
    Matrix<T> mean_running;
    Matrix<T> std_running;
    Matrix<T> gamma;
    Matrix<T> beta;

    BatchNorm1D(const size_t features, const T momentum = 0.1, const T eps = 1e-5)
        : eps(eps), momentum(momentum), one_minus_momentum(static_cast<T>(1) - momentum) {
        mean_running = Matrix<T>::zeros({1, features});
        std_running = Matrix<T>::ones({1, features});
        gamma = Matrix<T>::ones({1, features});
        beta = Matrix<T>::zeros({1, features});
    }

    Matrix<T> forward(Matrix<T> &x) {
        if (Module<T>::training_) {
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

    std::vector<Matrix<T>> parameters() { return {gamma, beta}; }
};

template <typename T> struct SoftMax : Module<T> {
    auto forward(Matrix<T> &logits) const {
        auto logit_maxes = logits.max(1);
        auto norm_logits = logits - logit_maxes;
        auto counts = norm_logits.exp();
        auto counts_sum = counts.sum(1);
        auto counts_sum_inv = counts_sum.pow(-1.0f);
        auto probs = counts * counts_sum_inv;
        return probs;
    }
};

template <typename T> struct NegativeLogLikelihood {
    auto forward(Matrix<T> &probs, Matrix<size_t> &y) const {
        auto selector = Matrix<size_t>(range(y.shape()[0]), y);
        auto logprobs = probs.log();
        auto loss = -logprobs[selector].mean(0);
        return loss;
    }
};

template <typename T> struct CrossEntropyLoss {
    template <typename U> Matrix<T> forward(Matrix<T> &logits, Matrix<U> &labels) {
        Matrix<T> out({1, 1});
        cross_entropy_loss(out.ctx_, logits.ctx_, labels.ctx_, 1);
        return out;
    }
    std::vector<Matrix<T>> parameters() { return {}; }
};

template <typename T> struct Embeddings {
    Matrix<T> weight;
    bool training_ = true;

    Embeddings(size_t num_embeddings, size_t embedding_dim, std::mt19937 g) {
        weight = Matrix<T>::randn({num_embeddings, embedding_dim}, g);
    }

    template <typename U> Matrix<T> forward(Matrix<U> &input) { return select_embeddings(weight, input); }

    std::vector<Matrix<T>> parameters() { return {weight}; }

    void training(bool tr) { training_ = tr; }
};