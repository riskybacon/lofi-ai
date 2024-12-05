#include <fstream>
#include <iostream>
#include <ranges>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "data/shapes.hpp"

#include <lofi/engine.hpp>
#include <lofi/graphviz.hpp>
#include <lofi/test_helper.hpp>

template <typename T> void test_mlp_bn() {
    auto Xtr = Matrix<size_t>::from_file("data_bn/Xtr.bin", shapes["Xtr"], "Xtr");
    auto Ytr = Matrix<size_t>::from_file("data_bn/Ytr.bin", shapes["Ytr"], "Ytr");
    auto C = Matrix<T>::from_file("data_bn/C.bin", shapes["C"], "C");
    auto W1 = Matrix<T>::from_file("data_bn/W1.bin", shapes["W1"], "W1");
    auto W2 = Matrix<T>::from_file("data_bn/W2.bin", shapes["W2"], "W2");
    auto b2 = Matrix<T>::from_file("data_bn/b2.bin", shapes["b2"], "b2");
    auto bn = BatchNorm1D<T>(W1.shape()[1], 0.001, 1e-5);
    auto ce = CrossEntropyLoss<float>();

    auto emb = select_embeddings(C, Xtr);
    auto hprebn = matmul(emb, W1);
    auto hpreact = bn.forward(hprebn);
    auto h = hpreact.tanh();
    auto h_w2 = matmul(h, W2);
    auto logits = h_w2 + b2;
    auto loss = ce.forward(logits, Ytr);
    // auto logit_maxes = logits.max(1);
    // auto norm_logits = logits - logit_maxes;
    // auto counts = norm_logits.exp();
    // auto counts_sum = counts.sum(1);
    // auto counts_sum_inv = counts_sum.pow(-1.0f);
    // auto probs = counts * counts_sum_inv;
    // auto logprobs = probs.log();
    // auto selector = Matrix<size_t>(range(Ytr.shape()[0]), Ytr);
    // auto loss = -logprobs[selector].mean(0);

    std::vector<std::tuple<Matrix<T>, std::string>> nodes = {
        {emb, "emb_view"},
        {hprebn, "hprebn"},
        {hpreact, "hpreact"},
        {h, "h"},
        {h_w2, "h_w2"},
        {logits, "logits"},
        // {logit_maxes, "logit_maxes"},
        // {norm_logits, "norm_logits"},
        // {counts, "counts"},
        // {counts_sum, "counts_sum"},
        // {counts_sum_inv, "counts_sum_inv"},
        // {probs, "probs"},
        // {logprobs, "logprobs"},
        {loss, "loss"},
    };

    for (auto &[node, label] : nodes) {
        node.label() = label;
    }

    draw_dot(loss, "test_bn.dot", "TB");
    // Fails if graphviz is not installed, but won't stop execution
    generate_svg_from_dot("test_bn.dot", "test_bn.svg");

    // Check forward pass
    for (const auto [node, label] : nodes) {
        std::stringstream fn;
        fn << "data_bn/" << label << ".bin";
        auto expected = Matrix<float>::from_file(fn.str(), shapes[label]);
        std::cout << "Checking " << label << ".data()" << std::endl;
        is_close(node.data(), expected.data());
    }

    std::cout << "Running backward pass" << std::endl;
    loss.backward();

    // Check backward pass
    for (const auto [node, label] : std::views::reverse(nodes)) {
        std::stringstream fn;
        fn << "data_bn/" << label << ".grad.bin";
        auto expected = Matrix<float>::from_file(fn.str(), shapes[label]);
        std::cout << "Checking " << label << ".grad()" << std::endl;
        is_close(node.grad(), expected.data());
    }
}


int main() {
    test_mlp_bn<float>();
    const size_t num_failed = num_tests - num_passed;
    if (num_failed > 0) {
        return 1;
    }
    return 0;
}