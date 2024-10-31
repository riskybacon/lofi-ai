#include <lofi/test_helper.hpp>

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

int main(int argc, char **argv) {
    test1<float>();
    std::cout << argv[0] << ": " << num_passed << " passed / " << num_tests << " total" << std::endl;
    return 0;
}
