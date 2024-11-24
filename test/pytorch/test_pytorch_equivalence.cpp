#include <fstream>
#include <iostream>
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

    auto C_grad_expected = Matrix<float>::from_file("data/C.grad.bin", shapes["C"]);
    auto W1_grad_expected = Matrix<float>::from_file("data/W1.grad.bin", shapes["W1"]);
    auto b1_grad_expected = Matrix<float>::from_file("data/b1.grad.bin", shapes["b1"]);
    auto W2_grad_expected = Matrix<float>::from_file("data/W2.grad.bin", shapes["W2"]);
    auto b2_grad_expected = Matrix<float>::from_file("data/b2.grad.bin", shapes["b2"]);

    auto emb_expected = Matrix<float>::from_file("data/emb_view.bin", shapes["emb_view"]);
    auto emb_grad_expected = Matrix<float>::from_file("data/emb_view.grad.bin", shapes["emb_view"]);
    auto logits_expected = Matrix<float>::from_file("data/logits.bin", shapes["logits"]);
    auto logits_grad_expected = Matrix<float>::from_file("data/logits.grad.bin", shapes["logits"]);
    auto emb_w1_expected = Matrix<float>::from_file("data/emb_w1.bin", shapes["emb_w1"]);
    auto emb_w1_grad_expected = Matrix<float>::from_file("data/emb_w1.grad.bin", shapes["emb_w1"]);
    auto emb_w1_b1_expected = Matrix<float>::from_file("data/emb_w1_b1.bin", shapes["emb_w1_b1"]);
    auto emb_w1_b1_grad_expected = Matrix<float>::from_file("data/emb_w1_b1.grad.bin", shapes["emb_w1_b1"]);
    auto h_expected = Matrix<float>::from_file("data/h.bin", shapes["h"]);
    auto h_grad_expected = Matrix<float>::from_file("data/h.grad.bin", shapes["h"]);
    auto h_w2_expected = Matrix<float>::from_file("data/h_w2.bin", shapes["h_w2"]);
    auto h_w2_grad_expected = Matrix<float>::from_file("data/h_w2.grad.bin", shapes["h_w2"]);
    auto logit_maxes_expected = Matrix<float>::from_file("data/logit_maxes.bin", shapes["logit_maxes"]);
    auto logit_maxes_grad_expected = Matrix<float>::from_file("data/logit_maxes.grad.bin", shapes["logit_maxes"]);
    auto norm_logits_expected = Matrix<float>::from_file("data/norm_logits.bin", shapes["norm_logits"]);
    auto norm_logits_grad_expected = Matrix<float>::from_file("data/norm_logits.grad.bin", shapes["norm_logits"]);
    auto counts_expected = Matrix<float>::from_file("data/counts.bin", shapes["counts"]);
    auto counts_grad_expected = Matrix<float>::from_file("data/counts.grad.bin", shapes["counts"]);
    auto counts_sum_expected = Matrix<float>::from_file("data/counts_sum.bin", shapes["counts_sum"]);
    auto counts_sum_grad_expected = Matrix<float>::from_file("data/counts_sum.grad.bin", shapes["counts_sum"]);
    auto counts_sum_inv_expected = Matrix<float>::from_file("data/counts_sum_inv.bin", shapes["counts_sum_inv"]);
    auto counts_sum_inv_grad_expected =
        Matrix<float>::from_file("data/counts_sum_inv.grad.bin", shapes["counts_sum_inv"]);
    auto probs_expected = Matrix<float>::from_file("data/probs.bin", shapes["probs"]);
    auto probs_grad_expected = Matrix<float>::from_file("data/probs.grad.bin", shapes["probs"]);
    auto logprobs_expected = Matrix<float>::from_file("data/logprobs.bin", shapes["logprobs"]);
    auto logprobs_grad_expected = Matrix<float>::from_file("data/logprobs.grad.bin", shapes["logprobs"]);

    auto loss_expected = Matrix<float>::from_file("data/loss.bin", shapes["loss"]);
    auto loss_grad_expected = Matrix<float>::from_file("data/loss.grad.bin", shapes["loss"]);

    std::cout << "Checking select_embeddings" << std::endl;
    auto emb = select_embeddings(C, Xtr);
    emb.label() = "emb";
    is_close(emb.data(), emb_expected.data());

    std::cout << "Checking emb_w1" << std::endl;
    auto emb_w1 = matmul(emb, W1);
    emb_w1.label() = "emb_w1";
    is_close(emb_w1.data(), emb_w1_expected.data());

    std::cout << "Checking emb_w1_b1" << std::endl;
    auto emb_w1_b1 = emb_w1 + b1;
    emb_w1_b1.label() = "emb_w1_b1";
    is_close(emb_w1_b1.data(), emb_w1_b1_expected.data());

    std::cout << "Checking tanh" << std::endl;
    auto h = emb_w1_b1.tanh();
    h.label() = "h";
    is_close(h.data(), h_expected.data());

    std::cout << "Checking h_w2" << std::endl;
    auto h_w2 = matmul(h, W2);
    h_w2.label() = "h_w2";
    is_close(h_w2.data(), h_w2_expected.data());

    std::cout << "Checking logits" << std::endl;
    auto logits = h_w2 + b2;
    logits.label() = "logits";
    is_close(logits.data(), logits_expected.data());

    std::cout << "Checking logit_maxess" << std::endl;
    auto logit_maxes = logits.max(1);
    logit_maxes.label() = "logit_maxes";
    is_close(logit_maxes.data(), logit_maxes_expected.data());

    std::cout << "Checking norm_logits" << std::endl;
    auto norm_logits = logits - logit_maxes;
    norm_logits.label() = "norm_logits";
    is_close(norm_logits.data(), norm_logits_expected.data());

    std::cout << "Checking counts" << std::endl;
    auto counts = norm_logits.exp();
    counts.label() = "counts";
    is_close(counts.data(), counts_expected.data());

    auto counts_sum = counts.sum(1);
    counts_sum.label() = "counts_sum";
    std::cout << "Checking counts_sum" << std::endl;
    is_close(counts_sum.data(), counts_sum_expected.data());

    std::cout << "Checking counts_sum_inv" << std::endl;
    auto counts_sum_inv = counts_sum.pow(-1.0f);
    counts_sum_inv.label() = "counts_sum_inv";
    is_close(counts_sum_inv.data(), counts_sum_inv_expected.data());

    std::cout << "Checking probs" << std::endl;
    auto probs = counts * counts_sum_inv;
    probs.label() = "probs";
    is_close(probs.data(), probs_expected.data());

    std::cout << "Checking logprobs" << std::endl;
    auto logprobs = probs.log();
    logprobs.label() = "logprobs";
    is_close(logprobs.data(), logprobs_expected.data());

    std::cout << "Checking loss" << std::endl;
    auto selector = Matrix<size_t>(range(Ytr.shape()[0]), Ytr);
    auto loss = -logprobs[selector].mean(0);
    loss.label() = "loss";
    is_close(loss.data(), loss_expected.data());

    draw_dot(loss, "test.dot", "LR");
    // Fails if graphviz is not installed, but won't stop execution
    generate_svg_from_dot("test.dot", "test.svg");

    std::cout << "loss=" << loss.data();
    std::cout << "loss_expected=" << loss_expected.data() << std::endl;

    std::cout << "Running backward pass" << std::endl;
    loss.backward();

    std::cout << "Checking loss.grad" << std::endl;
    is_close(loss.grad(), loss_grad_expected.data());

    std::cout << "Checking logprobs.grad" << std::endl;
    is_close(logprobs.grad(), logprobs_grad_expected.data());

    std::cout << "Checking probs.grad" << std::endl;
    is_close(probs.grad(), probs_grad_expected.data());

    std::cout << "Checking counts_sum_inv.grad" << std::endl;
    is_close(counts_sum_inv.grad(), counts_sum_inv_grad_expected.data());

    std::cout << "Checking counts_sum.grad" << std::endl;
    is_close(counts_sum.grad(), counts_sum_grad_expected.data());

    std::cout << "Checking counts.grad" << std::endl;
    is_close(counts.grad(), counts_grad_expected.data());

    std::cout << "Checking norm_logits.grad" << std::endl;
    is_close(norm_logits.grad(), norm_logits_grad_expected.data());

    std::cout << "Checking logit_maxes.grad" << std::endl;
    is_close(logit_maxes.grad(), logit_maxes_grad_expected.data());

    std::cout << "Checking logits.grad" << std::endl;
    is_close(logits.grad(), logits_grad_expected.data());

    std::cout << "Checking emb_w1_b1.grad" << std::endl;
    is_close(emb_w1_b1.grad(), emb_w1_b1_grad_expected.data());

    std::cout << "Checking emb_w1.grad" << std::endl;
    is_close(emb_w1.grad(), emb_w1_grad_expected.data());

    std::cout << "Checking h_w2.grad" << std::endl;
    is_close(h_w2.grad(), h_w2_grad_expected.data());

    std::cout << "Checking h.grad" << std::endl;
    is_close(h.grad(), h_grad_expected.data());

    std::cout << "Checking W2.grad" << std::endl;
    is_close(W2.grad(), W2_grad_expected.data());

    std::cout << "Checking emb_w1_b1.grad" << std::endl;
    is_close(emb_w1_b1.grad(), emb_w1_b1_grad_expected.data());

    std::cout << "Checking emb.grad" << std::endl;
    is_close(emb.grad(), emb_grad_expected.data());

    std::cout << "Checking C.grad" << std::endl;
    is_close(C.grad(), C_grad_expected.data());
};

int main() {
    test_mlp<void>();
    const size_t num_failed = num_tests - num_passed;
    if (num_failed > 0) {
        return 1;
    }
    return 0;
}