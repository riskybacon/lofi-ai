#include <lofi/test_helper.hpp>
#include <unordered_map>

// TODO: write tests, this is a placeholder

template <typename T> void test1() {
    using matrix_t = Matrix<T>;

    size_t batch_size = 2;
    size_t block_size = 3;
    size_t n_embd = 2;
    size_t n_hidden = 2;

    auto gen = generator();
    gen.seed(42);

    auto emb = matrix_t::randn({batch_size, n_embd * block_size}, gen);
    auto W1 = matrix_t::randn({n_embd * block_size, n_hidden}, gen);
    auto b1 = matrix_t::randn({1, n_hidden}, gen);

    auto out = matmul(emb, W1) + b1;
    std::cout << "embd=" << std::endl << emb.data() << std::endl;
    std::cout << "W1=" << std::endl << W1.data() << std::endl;
    std::cout << "b1=" << std::endl << b1.data() << std::endl;
    std::cout << out.data() << std::endl << std::endl;
    ;

    out.grad() = static_cast<T>(1);
    out.backward();

    std::cout << "embd=" << std::endl << emb.grad() << std::endl;
    std::cout << "W1=" << std::endl << W1.grad() << std::endl;
    std::cout << "b1=" << std::endl << b1.grad() << std::endl;
}

template<typename T>
void test_multinomial() {
    const size_t num_cols = 10;
    const size_t num_samples = 10000;
    const T expected_counts = static_cast<T>(num_samples) / static_cast<T>(num_cols);

    const size_t min_counts = static_cast<size_t>(expected_counts * 0.90);
    const size_t max_counts = static_cast<size_t>(expected_counts * 1.10);

    Matrix<T> probs({1, num_cols});
    probs.data() = static_cast<T>(1);

    auto s = probs.sum(1);
    T divisor = static_cast<T>(1) / s.data()[{0, 0}];
    probs.data() = probs.data() * divisor;

    auto g = generator();
    g.seed(42);
    auto samples = multinomial(probs, num_samples, true, g);

    std::unordered_map<size_t, size_t> freq;
    for (auto s : samples) {
        freq[s]++;
    }

    for (auto [k, v] : freq) {
        in_range(k, v, min_counts, max_counts);
    }
}

int main(int argc, char **argv) {
    test1<float>();
    test_multinomial<float>();
    std::cout << argv[0] << ": " << num_passed << " passed / " << num_tests << " total" << std::endl;
    return 0;
}
