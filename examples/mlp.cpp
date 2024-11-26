#include <chrono>
#include <climits>
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
    std::cout << "Reading names from `" << filename << "`" << std::endl;
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

    for (size_t i = 0; i < chars.size(); i++) {
        stoi[chars[i]] = i;
        itos[i] = chars[i];
    }

    return {words, stoi, itos, chars};
}

auto build_dataset(std::vector<std::string>::iterator &w_begin, std::vector<std::string>::iterator &w_end,
                   const size_t block_size, std::unordered_map<char, int> &stoi) {
    std::vector<std::vector<size_t>> x_vec;
    std::vector<size_t> y_vec;
    for (; w_begin != w_end; w_begin++) {
        const auto w = *w_begin;
        std::deque<int> context;
        for (size_t i = 0; i < block_size; i++) {
            context.push_back(0);
        }

        for (auto ch : w + '.') {
            size_t ix = stoi[ch];
            x_vec.push_back(std::vector<size_t>(begin(context), end(context)));
            y_vec.push_back({ix});
            // crop context and append
            context.pop_front();
            context.push_back(ix);
        }
    }

    Matrix<size_t> X({x_vec.size(), block_size});
    Matrix<size_t> Y({y_vec.size(), 1});

    for (size_t i = 0; i < x_vec.size(); i++) {
        for (size_t j = 0; j < block_size; j++) {
            X.data()[{i, j}] = x_vec[i][j];
        }
        Y.data()[{i, 0}] = y_vec[i];
    }

    return std::make_tuple(X, Y);
}

template <typename T> struct Model {
    Matrix<T> C;
    Matrix<T> W1;
    Matrix<T> b1;
    Matrix<T> W2;
    Matrix<T> b2;
    std::vector<Matrix<T>> parameters;

    Model(size_t context_size, size_t embedding_dim, size_t hidden, size_t num_tokens, std::mt19937 g) {
        // TODO: implement Kaming initialization
        // https://youtu.be/P6sfmUTpUmc?t=1860
        // "Kaiming init" paper: https://arxiv.org/abs/1502.01852
        // Batchnorm paper: https://arxiv.org/abs/1502.03167
        T tanh_gain = static_cast<T>(5) / static_cast<T>(3);
        T W1_scale = tanh_gain / static_cast<T>(sqrt(context_size * embedding_dim));
        T W2_scale = tanh_gain / static_cast<T>(sqrt(hidden));
        C = Matrix<T>::randn({num_tokens, embedding_dim}, g);
        W1 = Matrix<T>::randn({context_size * C.shape()[1], hidden}, g) * W1_scale;
        b1 = Matrix<T>::randn({1, hidden}, g) * 0.01;
        W2 = Matrix<T>::randn({hidden, num_tokens}, g) * W2_scale;
        b2 = Matrix<T>::randn({1, num_tokens}, g) * 0;

        parameters = {C, W1, b1, W2, b2};
    }

    size_t num_parameters() const {
        size_t n = 0;
        for (const auto &p : parameters) {
            n += p.shape()[0] * p.shape()[1];
        }
        return n;
    }

    auto forward(Matrix<size_t> &x) {
        auto emb = select_embeddings(C, x);
        auto emb_w1_b1 = matmul(emb, W1) + b1;
        auto h = emb_w1_b1.tanh();
        auto logits = matmul(h, W2) + b2;
        return logits;
    }

    void weight_update(float lr) {
        for (auto &p : parameters) {
            p.data() += -lr * p.grad();
        }
    }
};

template <typename T> struct SoftMax {
    auto operator()(Matrix<T> &logits) const {
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
    auto operator()(Matrix<T> &probs, Matrix<size_t> &y) const {
        auto selector = Matrix<size_t>(range(y.shape()[0]), y);
        auto logprobs = probs.log();
        auto loss = -logprobs[selector].mean(0);
        return loss;
    }
};

void run(void) {
    const size_t context_size = 5;
    const size_t embedding_dim = 10;
    const size_t hidden = 200;

    std::string filename = std::string(PROJECT_SOURCE_DIR) + "/examples/names.txt";
    auto [words, stoi, itos, chars] = read_file(filename);

    const auto num_tokens = itos.size();
    std::cout << "num_tokens: " << num_tokens << std::endl;
    std::cout << "words: " << words.size() << std::endl;

    auto g = generator();
    g.seed(INT_MAX);

    shuffle(begin(words), end(words), g);

    // Define data splits for training, dev, and test
    auto w0 = words.begin();
    auto w1 = w0 + 0.8 * words.size();
    auto w2 = w0 + 0.9 * words.size();
    auto w3 = words.end();

    // Split data
    auto [Xtr, Ytr] = build_dataset(w0, w1, context_size, stoi);
    auto [Xdev, Ydev] = build_dataset(w1, w2, context_size, stoi);
    auto [Xte, Yte] = build_dataset(w2, w3, context_size, stoi);

    std::cout << "Training set size: " << Xtr.shape()[0] << std::endl;
    std::cout << "Dev set size: " << Xdev.shape()[0] << std::endl;
    std::cout << "Test set size: " << Xte.shape()[0] << std::endl;

    Model<float> model(context_size, embedding_dim, hidden, num_tokens, g);
    SoftMax<float> softmax;
    NegativeLogLikelihood<float> loss_fn;

    std::cout << "num_parameters: " << model.num_parameters() << std::endl;

    size_t num_steps = 100000;
    size_t decay_step = 50000;
    size_t decay_step1 = 75000;
    size_t eval_steps = 5000;
    float lr = 0.1;

    const size_t batch_size = 64;
    const size_t zero = 0;

    std::chrono::duration<double> duration(0);

    auto start = std::chrono::high_resolution_clock::now();
    // Training loop
    for (size_t k = 0; k < num_steps; k++) {

        if (k == decay_step || k == decay_step1) {
            lr *= 0.1;
        }

        Matrix<size_t> x_batch;
        Matrix<size_t> y_batch;
        std::string prefix;

        bool training = true;

        if (k == num_steps - 1) {
            prefix = "***TEST*** ";
            x_batch = Xte;
            y_batch = Yte;
            training = false;
        } else if (k > 0 && k % eval_steps == 0) {
            prefix = "***EVAL*** ";
            x_batch = Xdev;
            y_batch = Ydev;
            training = false;
        } else {
            auto ix = Matrix<size_t>::randint(zero, Xtr.shape()[0], {batch_size, 1}, g);
            x_batch = Xtr[ix];
            y_batch = Ytr[ix];
            training = true;
        }

        // Forward
        auto logits = model.forward(x_batch);
        auto probs = softmax(logits);
        auto loss = loss_fn(probs, y_batch);

        if (k % 500 == 0 || k == num_steps - 1) {
            const auto end = std::chrono::high_resolution_clock::now();
            duration = (end - start) / 500;
            std::cout << prefix << "step: " << k + 1 << "/" << num_steps << ": lr=" << lr << ", loss=" << loss.data()
                      << ", duration=" << duration.count() << " s" << std::endl;
            start = std::chrono::high_resolution_clock::now();
        }

        if (training) {
            loss.zero_grad();
            loss.backward();
            model.weight_update(lr);
        }
    }

    // Sample from model
    Matrix<size_t> context({1, context_size});
    for (size_t i = 0; i < 20; i++) {
        context.data() = 0;
        std::vector<char> out;

        while (out.size() < 100) {
            auto logits = model.forward(context);
            auto probs = softmax(logits);
            auto ix = multinomial(probs, 1, true, g).front();
            out.push_back(itos[ix]);

            if (ix == 0) {
                break;
            }

            for (size_t i = 0; i < context_size - 1; i++) {
                context.data()[{0, i}] = context.data()[{0, i + 1}];
            }
            context.data()[{0, context_size - 1}] = ix;
        }

        for (auto ch : out) {
            std::cout << ch;
        }
        std::cout << std::endl;
    }
}

int main() {
    run();

    std::cout << "ref_count=" << ref_count << std::endl;
    return 0;
}