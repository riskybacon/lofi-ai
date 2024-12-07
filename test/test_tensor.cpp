#include <iostream>
#include <memory>

#include <lofi/tensor.hpp>
#include <lofi/test_helper.hpp>

// Fill a tensor with some values, must be deterministic between platforms
template <typename T> void fill_tensor(Tensor<T> &tensor, const T offset = static_cast<T>(0)) {
    T val(offset);

    for (auto &t_i : tensor) {
        t_i = val;
        val += static_cast<T>(1);
    }
}

template <typename T> void test_iterator_traits() {
    using elements_t = Tensor<T>;
    using iterator_t = typename elements_t::Iterator;
    static_assert(std::input_or_output_iterator<iterator_t>);
    static_assert(std::ranges::input_range<elements_t>);
}

template <typename T> void test_add_scalar() {
    Tensor<T> a({2, 4, 4});
    auto c = Tensor<T>::empty_like(a);
    auto expected_c = Tensor<T>::empty_like(a);
    const T scalar{93.456};

    fill_tensor(a);

    for (auto [c_i, a_i] : zip(expected_c, a)) {
        c_i = a_i + scalar;
    }

    add(c, a, scalar);
    is_close(c, expected_c);
}

template <typename T> void test_subtract_scalar() {
    Tensor<T> a({2, 4, 4});
    auto c = Tensor<T>::empty_like(a);
    auto expected_c = Tensor<T>::empty_like(a);
    const T scalar{7.432};

    fill_tensor(a);

    for (auto [c_i, a_i] : zip(expected_c, a)) {
        c_i = a_i - scalar;
    }

    subtract(c, a, scalar);
    is_close(c, expected_c);
}

template <typename T> void test_multiply_scalar() {
    Tensor<T> a({2, 4, 4});
    auto c = Tensor<T>::empty_like(a);
    auto expected_c = Tensor<T>::empty_like(a);
    const T scalar{2.3};

    fill_tensor(a);

    for (auto [c_i, a_i] : zip(expected_c, a)) {
        c_i = a_i * scalar;
    }

    multiply(c, a, scalar);
    is_close(c, expected_c);
}

template <typename T> void test_divide_scalar() {
    Tensor<T> a({2, 4, 4});
    auto c = Tensor<T>::empty_like(a);
    auto expected_c = Tensor<T>::empty_like(a);
    const T scalar{3.14159};

    fill_tensor(a);

    for (auto [c_i, a_i] : zip(expected_c, a)) {
        c_i = a_i / scalar;
    }

    divide(c, a, scalar);
    is_close(c, expected_c);
}

template <typename T> void test_add_tensor() {
    Tensor<T> a({10, 4, 30});
    auto b = Tensor<T>::empty_like(a);
    auto c = Tensor<T>::empty_like(a);
    auto expected_c = Tensor<T>::empty_like(a);

    fill_tensor(a);
    fill_tensor(b);

    for (auto [c_i, a_i, b_i] : zip(expected_c, a, b)) {
        c_i = a_i + b_i;
    }
    add(c, a, b);
    is_close(c, expected_c);

    // f = d + e
    Tensor<T> d({3, 4, 5});
    fill_tensor(d);

    auto e = Tensor<T>::ones({d.extents(2)});
    fill_tensor(e);

    auto f_extents = broadcast_extents(d.extents(), e.extents());
    Tensor<T> f(f_extents);
    auto f_expected = Tensor<T>::empty_like(f);

    for (auto [f_outer, d_outer] : zip(f_expected.slices(), d.slices())) {
        for (auto [f_inner, d_inner] : zip(f_outer.slices(), d_outer.slices())) {
            for (auto [f_i, d_i, e_i] : zip(f_inner, d_inner, e)) {
                f_i = d_i + e_i;
            }
        }
    }

    add(f, d, e);
    is_close(f, f_expected);

    f.fill_value(static_cast<T>(0));
    add(f, e, d);
    is_close(f, f_expected);

    f.fill_value(static_cast<T>(0));
    add(f.view({12, 5}), d.view({12, 5}), e);
    is_close(f, f_expected);

    f.fill_value(static_cast<T>(0));
    add(f.view({12, 5}), e, d.view({12, 5}));
    is_close(f, f_expected);

    f = 0.f;
    f_expected = 0.f;

    Tensor<T> g({d.extents(0), 1, 1});
    fill_tensor(g);

    for (auto [f_outer, d_outer, g_outer] : zip(f_expected.slices(), d.slices(), g.slices())) {
        auto g_inner = g_outer.slice({{0, 1}}).squeeze({0});
        for (auto [f_inner, d_inner] : zip(f_outer.slices(), d_outer.slices())) {
            const T g_i = g_inner.item();
            for (auto [f_i, d_i] : zip(f_inner, d_inner)) {
                f_i = d_i + g_i;
            }
        }
    }

    add(f, d, g);
    is_close(f, f_expected);

    // TODO: does this belong here?
    throws_exception(std::invalid_argument, add(f.view({60}), d.view({60}), e));
}

int main(int argc, char **argv) {
    test_iterator_traits<float>();
    test_add_scalar<float>();
    test_subtract_scalar<float>();
    test_multiply_scalar<float>();
    test_divide_scalar<float>();
    test_add_tensor<float>();

    std::cout << argv[0] << ": " << num_passed << " passed / " << num_tests << " total" << std::endl;
    return 0;
}