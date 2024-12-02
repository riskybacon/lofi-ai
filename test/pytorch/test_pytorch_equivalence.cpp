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

template <typename T> void copy(MatrixStorage<T> &lhs, const MatrixStorage<T> &rhs) {
    for (size_t r = 0; r < rhs.shape[0]; r++) {
        for (size_t c = 0; c < rhs.shape[1]; c++) {
            lhs[{r, c}] = rhs[{r, c}];
        }
    }
}

template <typename T> void test_mlp() {
    auto Xtr = Matrix<size_t>::from_file("data/Xtr.bin", shapes["Xtr"], "Xtr");
    auto Ytr = Matrix<size_t>::from_file("data/Ytr.bin", shapes["Ytr"], "Ytr");
    auto C = Matrix<float>::from_file("data/C.bin", shapes["C"], "C");
    auto W1 = Matrix<float>::from_file("data/W1.bin", shapes["W1"], "W1");
    auto b1 = Matrix<float>::from_file("data/b1.bin", shapes["b1"], "b1");
    auto W2 = Matrix<float>::from_file("data/W2.bin", shapes["W2"], "W2");
    auto b2 = Matrix<float>::from_file("data/b2.bin", shapes["b2"], "b2");
    auto bngain = Matrix<float>::from_file("data/bngain.bin", shapes["bngain"], "bngain");
    auto bnbias = Matrix<float>::from_file("data/bnbias.bin", shapes["bnbias"], "bnbias");

    const size_t hidden_size = W1.shape()[1];
    const size_t batch_size = Xtr.shape()[0];
    auto emb = select_embeddings(C, Xtr);
    auto emb_w1 = matmul(emb, W1);
    auto hprebn = emb_w1 + b1;

    auto bnmeani = hprebn.mean(0);
    auto bndiff = hprebn - bnmeani;
    auto bndiff2 = bndiff.pow(2);
    auto bndiff2_div_n = bndiff2 * (1.0f / batch_size);
    auto bnvar = bndiff2_div_n.sum(0);
    auto bnvar_inv = (bnvar + 1e-5).pow(-0.5);
    auto bnraw = bndiff * bnvar_inv;
    auto hpreact = bngain * bnraw + bnbias;
    auto h = hpreact.tanh();
    auto h_w2 = matmul(h, W2);
    auto logits = h_w2 + b2;

    auto logit_maxes = logits.max(1);
    auto norm_logits = logits - logit_maxes;
    auto counts = norm_logits.exp();
    auto counts_sum = counts.sum(1);
    auto counts_sum_inv = counts_sum.pow(-1.0f);
    auto probs = counts * counts_sum_inv;
    auto logprobs = probs.log();
    auto selector = Matrix<size_t>(range(Ytr.shape()[0]), Ytr);
    auto loss = -logprobs[selector].mean(0);

    std::vector<std::tuple<Matrix<float>, std::string>> nodes = {
        {emb, "emb_view"},
        {emb_w1, "emb_w1"},
        {hprebn, "hprebn"},
        {bnmeani, "bnmeani"},
        {bndiff, "bndiff"},
        {bndiff2, "bndiff2"},
        {bndiff2_div_n, "bndiff2_div_n"},
        {bnvar, "bnvar"},
        {bnvar_inv, "bnvar_inv"},
        {bnraw, "bnraw"},
        {hpreact, "hpreact"},
        {h, "h"},
        {h_w2, "h_w2"},
        {logits, "logits"},
        {logit_maxes, "logit_maxes"},
        {norm_logits, "norm_logits"},
        {counts, "counts"},
        {counts_sum, "counts_sum"},
        {counts_sum_inv, "counts_sum_inv"},
        {probs, "probs"},
        {logprobs, "logprobs"},
        {loss, "loss"},
    };

    for (auto& [node, label] : nodes) {
        node.label() = label;
    }

    draw_dot(loss, "test.dot", "TB");
    // Fails if graphviz is not installed, but won't stop execution
    generate_svg_from_dot("test.dot", "test.svg");

    // Check forward pass
    for (const auto [node, label] : nodes) {
        std::stringstream fn;
        fn << "data/" << label << ".bin";
        auto expected = Matrix<float>::from_file(fn.str(), shapes[label]);
        std::cout << "Checking " << label << ".data()" << std::endl;
        is_close(node.data(), expected.data());
    }

    std::cout << "Running backward pass" << std::endl;
    loss.backward();

    // Check backward pass
    for (const auto [node, label] : std::views::reverse(nodes)) {
        std::stringstream fn;
        fn << "data/" << label << ".grad.bin";
        auto expected = Matrix<float>::from_file(fn.str(), shapes[label]);
        std::cout << "Checking " << label << ".grad()" << std::endl;
        is_close(node.grad(), expected.data());
    }
};

int main() {
    test_mlp<float>();
    const size_t num_failed = num_tests - num_passed;
    if (num_failed > 0) {
        return 1;
    }
    return 0;
}