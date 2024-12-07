#include <iostream>
#include <memory>
#include <numeric>
#include <ostream>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <lofi/layout.hpp>
#include <lofi/test_helper.hpp>

using std::views::iota;

void test_iterator_traits() {
    // Layout::Iterator:
    // * needs to be default constructible.
    // * std::iter_difference_t<Layout::Iterator> must be valid, and
    // * the pre-increment operator must return a reference.
    // * Layout::Iterator must be a std::sentinel_for<Layout::Iterator, Layout::Iterator>
    // * by taking const references in  operators == and !=
    // * Must satisfy std::input_iterator, which requires std::iter_value_t to be valid.
    static_assert(std::input_or_output_iterator<Layout::Iterator>);
    static_assert(std::ranges::input_range<Layout>);
}

void test_eq() {
    Layout a({2, 3, 4});
    Layout b({2, 3, 4});
    Layout c({1, 2});

    is_equal(a, b);
    not_equal(a, c);
}

void test_make_strides_row_major() {
    const std::vector<size_t> a_extents{2, 3, 4};
    const std::vector<ssize_t> a_expected_strides{12, 4, 1};
    const auto a_strides = make_strides_row_major(a_extents);
    is_equal(a_strides, a_expected_strides);

    const std::vector<size_t> b_extents{3, 1, 1};
    const std::vector<ssize_t> b_expected_strides{1, 0, 0};
    const auto b_strides = make_strides_row_major(b_extents);
    is_equal(b_strides, b_expected_strides);

    const std::vector<size_t> c_extents{4, 3, 1, 1};
    const std::vector<ssize_t> c_expected_strides{3, 1, 0, 0};
    const auto c_strides = make_strides_row_major(c_extents);
    is_equal(c_strides, c_expected_strides);

    const std::vector<size_t> d_extents{1, 2, 3};
    const std::vector<ssize_t> d_expected_strides{0, 3, 1};
    const auto d_strides = make_strides_row_major(d_extents);
    is_equal(d_strides, d_expected_strides);
}

void test_construct() {
    const Layout a({3, 4, 5, 6});
    const Layout b({3, 4, 5, 6}, {120, 30, 6, 1}, 0);
    is_equal(a, b);
    is_equal(a.rank(), 4);
    is_equal(a.size(), 3 * 4 * 5 * 6);
    is_equal(a.offset(), 0);
    is_equal(a.mod(), a.size());
    is_equal(a, iota(0u, a.size()));
}

void test_view() {
    const Layout a({2, 3, 4});
    const auto b = a.view({2, 3, 4});
    is_equal(a, b);

    const auto c = a.view({6, 4});
    const Layout c_expected{{6u, 4u}, {4, 1}, 0};
    is_equal(c, c_expected);

    const auto d = Layout({6, 4});
    const auto e = d.view({2, 3, 4});
    const Layout e_expected{{2u, 3u, 4u}, {12, 4, 1}, 0};
    is_equal(e, e_expected);

    throws_exception(std::invalid_argument, a.view({999999}));
}

void test_slice() {
    Layout a({2, 3, 4});

    auto a00 = a.slice({{0u, 1u}, {0u, 3u}, {0u, 4u}});
    is_equal(a00.rank(), 3);
    is_equal(a00.size(), 3 * 4);
    is_equal(a00.offset(), 0);
    is_equal(a00.extents(), std::vector<size_t>({1, 3, 4}));
    is_equal(a00.strides(), std::vector<ssize_t>({12, 4, 1}));
    is_equal(a00, iota(0u, a.size() / 2));

    auto a00_view = a00.squeeze({0});
    is_equal(a00_view.rank(), 2);
    is_equal(a00_view.size(), 3 * 4);
    is_equal(a00_view.offset(), 0);
    is_equal(a00_view.extents(), std::vector<size_t>({3, 4}));
    is_equal(a00_view.strides(), std::vector<ssize_t>({4, 1}));
    is_equal(a00_view, iota(0u, a.size() / 2));

    auto a01 = a.slice({{1u, 2u}, {0u, 3u}, {0u, 4u}});
    is_equal(a01.rank(), 3);
    is_equal(a01.size(), 3 * 4);
    is_equal(a01.offset(), 12);
    is_equal(a01.extents(), std::vector<size_t>({1, 3, 4}));
    is_equal(a01.strides(), std::vector<ssize_t>({12, 4, 1}));
    is_equal(a01, iota(a.size() / 2, a.size()));

    auto a01_view = a01.squeeze();

    is_equal(a01_view.rank(), 2);
    is_equal(a01_view.size(), 3 * 4);
    is_equal(a01_view.offset(), 12);
    is_equal(a01_view.extents(), std::vector<size_t>({3, 4}));
    is_equal(a01_view.strides(), std::vector<ssize_t>({4, 1}));
    is_equal(a01_view, iota(a.size() / 2, a.size()));

    Layout b({10, 2, 3});

    auto b10 = b.slice({{0u, 10u}, {0u, 1u}, {0u, 3u}});
    is_equal(b10.rank(), 3);
    is_equal(b10.size(), 10 * 3);
    is_equal(b10.offset(), 0);
    is_equal(b10.extents(), std::vector<size_t>({10, 1, 3}));
    is_equal(b10.strides(), std::vector<ssize_t>({6, 3, 1}));
    std::vector<size_t> expected_indexes_b10(30);
    for (size_t i = 0; i < 10; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            expected_indexes_b10[i * 3 + j] = i * 6 + j;
        }
    }

    is_equal(b10, expected_indexes_b10);

    auto b10_view = b10.squeeze();
    is_equal(b10_view.rank(), 2);
    is_equal(b10_view.size(), 10 * 3);
    is_equal(b10_view.offset(), 0);
    is_equal(b10_view.extents(), std::vector<size_t>({10, 3}));
    is_equal(b10_view.strides(), std::vector<ssize_t>({6, 1}));
    is_equal(b10_view, expected_indexes_b10);

    auto b11 = b.slice({{0u, 10u}, {1u, 2u}, {0u, 3u}});
    is_equal(b11.rank(), 3);
    is_equal(b11.size(), 10 * 3);
    is_equal(b11.offset(), 3);
    is_equal(b11.extents(), std::vector<size_t>({10, 1, 3}));
    is_equal(b11.strides(), std::vector<ssize_t>({6, 3, 1}));
    std::vector<size_t> expected_indexes_b11(30);
    for (size_t i = 0; i < 10; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            expected_indexes_b11[i * 3 + j] = 3 + i * 6 + j;
        }
    }

    is_equal(b11, expected_indexes_b11);

    auto b11_view = b11.squeeze();
    is_equal(b11_view.rank(), 2);
    is_equal(b11_view.size(), 10 * 3);
    is_equal(b11_view.offset(), 3);
    is_equal(b11_view.extents(), std::vector<size_t>({10, 3}));
    is_equal(b11_view.strides(), std::vector<ssize_t>({6, 1}));
    is_equal(b11_view, expected_indexes_b11);

    auto b20 = b.slice({{0u, 10u}, {0u, 2u}, {0u, 1u}});
    is_equal(b20.rank(), 3);
    is_equal(b20.size(), 10 * 2);
    is_equal(b20.offset(), 0);
    is_equal(b20.extents(), std::vector<size_t>({10, 2, 1}));
    is_equal(b20.strides(), std::vector<ssize_t>({6, 3, 1}));
    std::vector<size_t> expected_indexes_b20;
    for (size_t i = 0; i < 10; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            expected_indexes_b20.push_back(i * 6 + j * 3);
        }
    }

    is_equal(b20, expected_indexes_b20);

    auto b20_view = b20.squeeze({2});
    is_equal(b20_view.rank(), 2);
    is_equal(b20_view.size(), 10 * 2);
    is_equal(b20_view.offset(), 0);
    is_equal(b20_view.extents(), std::vector<size_t>({10, 2}));
    is_equal(b20_view.strides(), std::vector<ssize_t>({6, 3}));
    is_equal(b20_view, expected_indexes_b20);

    auto b21 = b.slice({{0u, 10u}, {0u, 2u}, {1u, 2u}});
    is_equal(b21.rank(), 3);
    is_equal(b21.size(), 10 * 2);
    is_equal(b21.offset(), 1);
    is_equal(b21.extents(), std::vector<size_t>({10, 2, 1}));
    is_equal(b21.strides(), std::vector<ssize_t>({6, 3, 1}));

    std::vector<size_t> expected_indexes_b21; // (30);
    for (size_t i = 0; i < 10; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            expected_indexes_b21.push_back(i * 6 + j * 3 + 1);
        }
    }

    auto b21_view = b21.squeeze({2});
    is_equal(b21_view.rank(), 2);
    is_equal(b21_view.size(), 10 * 2);
    is_equal(b21_view.offset(), 1);
    is_equal(b21_view.extents(), std::vector<size_t>({10, 2}));
    is_equal(b21_view.strides(), std::vector<ssize_t>({6, 3}));
    is_equal(b21_view, expected_indexes_b21);

    auto b22 = b.slice({{4u, 10u}, {0u, 2u}, {1u, 3u}});
    is_equal(b22.rank(), 3);
    is_equal(b22.size(), 6 * 2 * 2);
    is_equal(b22.offset(), 4 * 6 + 1);
    is_equal(b22.extents(), std::vector<size_t>({6, 2, 2}));
    is_equal(b22.strides(), std::vector<ssize_t>({6, 3, 1}));

    std::vector<size_t> expected_indexes_b22;

    size_t b22_idx = 25;
    for (size_t i = 0; i < 12; i++) {
        for (size_t j = 0; j < 2; j++) {
            expected_indexes_b22.push_back(b22_idx);
            b22_idx += 1;
        }
        b22_idx += 1;
    }
    is_equal(b22, expected_indexes_b22);

    auto b23_view = b.slice({{1u, 2u}}).squeeze({0});
    is_equal(b23_view.rank(), 2);
    is_equal(b23_view.size(), 2 * 3);
    is_equal(b23_view.offset(), 6);
    is_equal(b23_view.extents(), std::vector<size_t>({2, 3}));
    is_equal(b23_view.strides(), std::vector<ssize_t>({3, 1}));
    is_equal(b23_view, iota(6u, 6u * 2u));

    throws_exception(std::invalid_argument, b.slice({{0u, 10u}, {0u, 2u}, {0u, 1u}, {0u, 1u}}));
}

void test_lambda_slices() {
    // Inside of the slices() method, there is a lambda. That lambda
    // needs to capture the layout correctly, and if captured incorrectly,
    // it will refer to a layout that is no longer valid.
    //
    // This test makes sure that if slices receives an rvalue for *this,
    // that layout is captured.
    Layout f({3, 4, 5});
    std::vector<size_t> f_expected_idxs(std::begin(f), std::end(f));

    std::vector<size_t> f_idxs;
    for (const auto f_outer : f.slices(0)) {
        for (const auto f_inner : f_outer.slices(0)) {
            for (const auto f_i : f_inner) {
                f_idxs.push_back(f_i);
            }
        }
    }

    is_equal(f_idxs, f_expected_idxs);
}

void test_slices() {
    Layout a({3, 1, 1});

    for (auto [expected_offset, a_outer] : zip(iota(0u, a.extents(0)), a.slices(0))) {
        is_equal(a_outer.rank(), 2);
        is_equal(a_outer.size(), 1);
        is_equal(a_outer.offset(), expected_offset);
        is_equal(a_outer.extents(), std::vector<size_t>({1, 1}));
        is_equal(a_outer.strides(), std::vector<ssize_t>({0, 0}));

        for (const auto a_inner : a_outer.slices(0)) {
            is_equal(a_inner.rank(), 1);
            is_equal(a_inner.size(), 1);
            is_equal(a_inner.offset(), expected_offset);
            is_equal(a_inner.extents(), std::vector<size_t>({1}));
            is_equal(a_inner.strides(), std::vector<ssize_t>({0}));

            for (const auto a_i : a_inner) {
                is_equal(a_i, expected_offset);
            }
        }
    }

    for (auto [expected_offset, a_outer] : zip(iota(0u, a.extents(0)), a.slices(0, KeepDim::True))) {
        is_equal(a_outer.rank(), 3);
        is_equal(a_outer.size(), 1);
        is_equal(a_outer.offset(), expected_offset);
        is_equal(a_outer.extents(), std::vector<size_t>({1, 1, 1}));
        is_equal(a_outer.strides(), std::vector<ssize_t>({1, 0, 0}));
    }
}

void test_broadcast() {
    Layout a({1});
    auto ab = a.broadcast({10});
    std::vector<size_t> expected;
    expected.resize(10);
    std::fill(std::begin(expected), std::end(expected), 0);
    is_equal(ab.rank(), 1);
    is_equal(ab.size(), 10);
    is_equal(ab.extents(), std::vector<size_t>({10}));
    is_equal(ab.strides(), std::vector<ssize_t>({0}));
    is_equal(ab.mod(), 1);
    is_equal(ab, expected);

    auto ac = a.broadcast({10, 1});
    is_equal(ac.rank(), 2);
    is_equal(ac.size(), 10);
    is_equal(ac.extents(), std::vector<size_t>({10, 1}));
    is_equal(ac.strides(), std::vector<ssize_t>({0, 0}));
    is_equal(ac.mod(), 1);
    is_equal(ac, expected);

    Layout b({3});
    auto bb = b.broadcast({4, 3});
    is_equal(bb.rank(), 2);
    is_equal(bb.size(), 12);
    is_equal(bb.extents(), std::vector<size_t>({4, 3}));
    is_equal(bb.strides(), std::vector<ssize_t>({0, 1}));
    is_equal(bb.mod(), 3);

    expected.clear();
    for (size_t i = 0; i < bb.extents(0); ++i) {
        for (size_t j = 0; j < bb.extents(1); ++j) {
            expected.push_back(j);
        }
    }

    is_equal(bb, expected);

    Layout c({3, 1, 1});
    auto cb = c.broadcast({3, 4, 5});
    is_equal(cb.rank(), 3);
    is_equal(cb.size(), 3 * 4 * 5);
    is_equal(cb.extents(), std::vector<size_t>({3, 4, 5}));
    is_equal(cb.strides(), std::vector<ssize_t>({1, 0, 0}));
    is_equal(cb.mod(), 3);

    expected.clear();
    for (size_t i = 0; i < cb.extents(0); ++i) {
        for (size_t j = 0; j < cb.extents(1); ++j) {
            for (size_t k = 0; k < cb.extents(2); ++k) {
                expected.push_back(i);
            }
        }
    }

    is_equal(cb, expected);

    Layout d({3, 4, 5});
    throws_exception(std::invalid_argument, d.broadcast({4, 5}));
    throws_exception(std::invalid_argument, d.broadcast({12, 5}));
    throws_exception(std::invalid_argument, d.broadcast({3, 4, 5, 6}));
    throws_exception(std::invalid_argument, d.broadcast({3, 4, 5, 1}));

    auto db = d.broadcast({1, 3, 4, 5});
    is_equal(db.rank(), 4);
    is_equal(db.size(), 3 * 4 * 5);
    is_equal(db.extents(), std::vector<size_t>({1, 3, 4, 5}));
    is_equal(db.strides(), std::vector<ssize_t>({0, 20, 5, 1}));
    is_equal(db.mod(), 3 * 4 * 5);

    expected.clear();
    for (size_t i = 0; i < db.extents(0); ++i) {
        size_t idx = 0;
        for (size_t j = 0; j < db.extents(1); ++j) {
            for (size_t k = 0; k < db.extents(2); ++k) {
                for (size_t l = 0; l < db.extents(3); ++l) {
                    expected.push_back(idx);
                    idx++;
                }
            }
        }
    }

    is_equal(db, expected);

    auto dc = d.broadcast({2, 3, 4, 5});
    is_equal(dc.rank(), 4);
    is_equal(dc.size(), 2 * 3 * 4 * 5);
    is_equal(dc.extents(), std::vector<size_t>({2, 3, 4, 5}));
    is_equal(dc.strides(), std::vector<ssize_t>({0, 20, 5, 1}));
    is_equal(dc.mod(), 3 * 4 * 5);

    expected.clear();
    for (size_t i = 0; i < dc.extents(0); ++i) {
        size_t idx = 0;
        for (size_t j = 0; j < dc.extents(1); ++j) {
            for (size_t k = 0; k < dc.extents(2); ++k) {
                for (size_t l = 0; l < dc.extents(3); ++l) {
                    expected.push_back(idx);
                    idx++;
                }
            }
        }
    }

    is_equal(dc, expected);

    auto dd = d.broadcast(d.extents());
    is_equal(dd.rank(), d.rank());
    is_equal(dd.size(), d.size());
    is_equal(dd.extents(), d.extents());
    is_equal(dd.strides(), d.strides());
    is_equal(dd.mod(), d.mod());
    is_equal(dd, d);
}

int main(int argc, char **argv) {
    test_iterator_traits();
    test_make_strides_row_major();
    test_eq();
    test_construct();
    test_view();
    test_slice();
    test_lambda_slices();
    test_slices();
    test_broadcast();
    std::cout << argv[0] << ": " << num_passed << " passed / " << num_tests << " total" << std::endl;
    return 0;
}