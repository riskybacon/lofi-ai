/**
 * @file bag_of_words.cpp
 * @brief C++ implementation of the single layer NN in makemore
 */
#include <fstream>
#include <iostream>
#include <random>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <lofi/engine.hpp>
#include <lofi/graphviz.hpp>

/**
 * @brief Reads the content of a file and returns various data structures based on its contents.
 *
 * This function reads the specified file and processes its contents to generate four outputs:
 * 1. A vector of strings, where each string represents a line/word in the file.
 * 2. An unordered map (char to int) representing the mapping from characters to indices.
 * 3. An unordered map (int to char) representing the mapping from indices to characters.
 * 4. A vector of characters, representing unique characters in the file. Each element's index
 *    corresponds to the indices in the two unordered_maps
 *
 * @param filename The name of the file to read.
 *
 * @return A tuple containing:
 *         - words
 *         - map of characters to indices
 *         - map of indices to characters
 *         - A vector of characters
 */
std::tuple<std::vector<std::string>, std::unordered_map<char, int>, std::unordered_map<int, char>, std::vector<char>>
read_file(const std::string &filename) {
    std::string word;
    std::ifstream file(filename);
    std::vector<std::string> words;
    std::unordered_set<char> chars_set;

    while (!file.eof()) {
        file >> word;
        words.push_back(word);
        for (const auto &ch : word) {
            chars_set.insert(ch);
        }
    }

    std::vector<char> chars = std::vector<char>(begin(chars_set), end(chars_set));
    std::sort(begin(chars), end(chars));
    chars.insert(begin(chars), '.');

    // build the vocabulary of characters and mappings to/from integers
    std::unordered_map<char, int> stoi;
    std::unordered_map<int, char> itos;

    for (int i = 0; i < chars.size(); i++) {
        stoi[chars[i]] = i;
        itos[i] = chars[i];
    }

    return {words, stoi, itos, chars};
}

std::tuple<std::vector<size_t>, std::vector<size_t>> build_dataset(const std::vector<std::string> &words,
                                                                   const std::unordered_map<char, int> &stoi) {
    std::vector<size_t> xs;
    std::vector<size_t> ys;

    for (const auto &w : words) {
        std::vector<char> chs;
        chs.push_back('.');
        for (const auto &ch : w) {
            chs.push_back(ch);
        }
        chs.push_back('.');

        for (size_t i = 0; i < chs.size() - 1; i++) {
            const auto ch1 = chs[i];
            const auto ch2 = chs[i + 1];
            const auto ix1 = stoi.at(ch1);
            const auto ix2 = stoi.at(ch2);
            xs.push_back(ix1);
            ys.push_back(ix2);
        }
    }

    return {xs, ys};
}

int main(void) {
    std::string filename = "names.txt";
    auto [words, stoi, itos, chars] = read_file(filename);
    auto [xs, ys] = build_dataset(words, stoi);
    const size_t num = xs.size();
    const auto vocab_size = itos.size();
    std::cout << "vocab_size=" << vocab_size << std::endl;
    std::cout << "words=" << words.size() << std::endl;
    std::cout << "xs.size()=" << xs.size() << ", ys.size()=" << ys.size() << std::endl;

    std::random_device rd;
    std::mt19937 g(rd());
    g.seed(INT_MAX);

    float lr = 10;
    size_t num_steps = 1000;

    auto W = Matrix<float>::randn({vocab_size, vocab_size}, g);
    auto xenc = Matrix<float>::one_hot(xs, vocab_size);
    auto selector = Matrix<size_t>(range(num), ys);

    // Training
    for (size_t k = 0; k < num_steps; k++) {
        // Forward
        auto logits = matmul(xenc, W);
        auto counts = logits.exp();
        auto probs = counts * counts.sum(1).pow(-1.0f);

        // Loss
        auto logprobs = probs.log();
        auto loss = -logprobs[selector].mean(0);

        std::cout << "step: " << k << "/" << num_steps - 1 << ": lr=" << lr << ", loss=" << loss.data();

        // Visualize conpute graph on first iteration
        if (k == 0) {
            W.label() = "W";
            xenc.label() = "xenc";
            selector.label() = "selector";
            xenc.label() = "xenc";
            logits.label() = "logits";
            counts.label() = "counts";
            probs.label() = "probs";
            logprobs.label() = "logprobs";
            loss.label() = "loss";

            draw_dot(loss, "one_layer.dot", "LR");
            // Fails if graphviz is not installed, but won't stop execution
            generate_svg_from_dot("one_layer.dot", "one_layer.svg");
        }

        // Backward
        loss.zero_grad();
        loss.grad() = static_cast<float>(1);
        loss.backward();

        // Weight update
        W.data() += -lr * W.grad();
    }

    return 0;
}