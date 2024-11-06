#include <array>
#include <iomanip>
#include <iostream>

#include <lofi/test_helper.hpp>

template <typename T> void test_constructor() {
    const shape_type shape0 = {0, 0};
    const shape_type shape1 = {120, 30};

    auto mat0 = MatrixStorage<T>();
    is_equal(mat0.shape, shape0);
    is_equal(mat0.data.size(), shape0[0]);

    auto mat1 = MatrixStorage<T>(shape1);
    is_equal(mat1.shape, shape1);

    T *expected_data = mat1.data.data();
    MatrixStorage<T> mat2(std::move(mat1));
    is_equal(mat2.shape, shape1);
    is_equal(mat2.data.data(), expected_data);
    is_equal(mat1.shape, shape0);
    is_equal(mat1.data.data(), static_cast<T *>(nullptr));
}

template <typename T> void test_move_assign() {
    const shape_type shape0 = {0, 0};
    const shape_type shape1 = {120, 30};

    auto mat0 = MatrixStorage<T>();
    is_equal(mat0.shape, shape0);
    is_equal(mat0.data.size(), shape0[0]);

    auto mat1 = MatrixStorage<T>(shape1);
    is_equal(mat1.shape, shape1);

    T *expected_data = mat1.data.data();
    mat0 = std::move(mat1);

    is_equal(mat0.shape, shape1);
    is_equal(mat0.data.data(), expected_data);

    is_equal(mat1.shape, shape0);
    is_equal(mat1.data.data(), static_cast<T *>(nullptr));
}

template <typename T> void fill_value_local(MatrixStorage<T> &mat, T val) {
    for (size_t r = 0; r < mat.shape[0]; r++) {
        for (size_t c = 0; c < mat.shape[1]; c++) {
            mat[{r, c}] = val;
        }
    }
}

template <typename T> void test_fill_value() {
    const T val0(-99);
    const T val1(101.101);
    const shape_type shape = {120, 30};
    auto mat0 = MatrixStorage<T>(shape, val0);
    fill_value(mat0, val1);

    auto mat1 = MatrixStorage<T>(shape);
    fill_value_local(mat1, val1);
    is_close(mat0, mat1);
}

template <typename T> void test_assign_value() {
    const shape_type shape = {34, 21};
    MatrixStorage<T> mat0(shape, static_cast<T>(0));
    MatrixStorage<T> mat1(shape, static_cast<T>(10));
    mat0 = static_cast<T>(10);
    is_close(mat0, mat1);
}

template <typename T> void test_clone() {
    const shape_type shape = {95, 21};
    MatrixStorage<T> mat0(shape);
    auto g = generator();
    fill_randn(mat0, g);

    auto mat1 = mat0.clone();
    is_close(mat0, mat1);
}

template <typename T> void test_add() {
    const shape_type shape = {10, 10};
    const T val0 = 10;
    const T val1 = 20;

    MatrixStorage<float> mat0(shape, val0);
    MatrixStorage<float> mat1(shape, val1);
    MatrixStorage<float> mat2(shape);
    add(mat2, mat0, mat1);

    MatrixStorage<float> mat3(shape, val0 + val1);
    is_close(mat2, mat3);

    // Test addition where out and lhs point to the same data
    add(mat0, mat0, mat1);
    is_close(mat0, mat3);
}

template <typename T> void test_subtract() {
    const shape_type shape = {3, 4};
    const T val0 = 70;
    const T val1 = 40;

    MatrixStorage<float> mat0(shape, val0);
    MatrixStorage<float> mat1(shape, val1);
    MatrixStorage<float> out(shape);
    MatrixStorage<float> expected(shape, val0 - val1);

    subtract(out, mat0, mat1);
    is_close(out, expected);
}

template <typename T> void test_multiply() {
    const shape_type shape = {101, 99};
    const T val0 = 9.f;
    const T val1 = -2001.f;

    MatrixStorage<float> mat0(shape, val0);
    MatrixStorage<float> mat1(shape, val1);
    MatrixStorage<float> mat2(shape);
    MatrixStorage<float> mat3(shape, val0 * val1);

    multiply(mat2, mat0, mat1);
    is_close(mat2, mat3);

    // Test multiplication where out and lhs point to the same data
    multiply(mat0, mat0, mat1);
    is_close(mat0, mat3);
}

template <typename T> void test_divide() {
    const shape_type shape = {5, 6};
    const T val0 = 5;
    const T val1 = 3;

    MatrixStorage<float> mat0(shape, val0);
    MatrixStorage<float> mat1(shape, val1);
    MatrixStorage<float> out(shape);
    MatrixStorage<float> expected(shape, val0 / val1);

    divide(out, mat0, mat1);
    is_close(out, expected);
}

template <typename T> void test_matmul() {
    const shape_type shape = {19, 19};
    MatrixStorage<float> mat0(shape);
    MatrixStorage<float> mat1(shape);

    fill_randn(mat0);

    for (size_t r = 0; r < mat1.shape[0]; r++) {
        for (size_t c = 0; c < mat1.shape[1]; c++) {
            mat1[{r, c}] = r == c ? static_cast<T>(1) : static_cast<T>(0);
        }
    }

    MatrixStorage<float> mat2(shape);
    matmul(mat2, mat0, mat1);
    is_close(mat2, mat0);

    throws_exception(std::invalid_argument, matmul(mat0, mat0, mat1));
    throws_exception(std::invalid_argument, matmul(mat1, mat0, mat1));
}

template <typename T> void test_sum(size_t axis) {
    const shape_type shape = {10, 8};
    shape_type bcast_shape = shape;
    shape_type bcast_fail_shape = shape;
    bcast_shape[axis] = 1;
    bcast_fail_shape[axis] = 2;
    MatrixStorage<T> mat(shape);
    MatrixStorage<T> out(bcast_shape, 0);
    MatrixStorage<T> out_fail(bcast_fail_shape, 0);
    MatrixStorage<T> expected(bcast_shape, 0);

    fill_mat(mat);

    auto out_idx = axis == 0 ? bcast0 : bcast1;

    for (size_t r = 0; r < shape[0]; r++) {
        for (size_t c = 0; c < shape[1]; c++) {
            expected[out_idx({r, c})] += mat[{r, c}];
        }
    }

    sum(out, mat, axis);
    is_close(out, expected);

    throws_exception(std::invalid_argument, sum(out, mat, 34));
    throws_exception(std::invalid_argument, sum(out_fail, mat, axis));
}

template <typename T> auto make_max_axis(size_t axis) {
    shape_type shape;
    size_t long_dim = 14;
    size_t short_dim = 3;

    if (axis == 0) {
        shape = {short_dim, long_dim};
    } else if (axis == 1) {
        shape = {long_dim, short_dim};
    } else {
        std::stringstream ss;
        ss << "Unknown axis " << axis;
        throw std::invalid_argument(ss.str());
    }

    shape_type reduced_shape = shape;
    reduced_shape[axis] = 1;
    auto idxr = axis == 0 ? bcast0 : bcast1;

    MatrixStorage<T> src(shape);
    MatrixStorage<T> expected_values(reduced_shape);
    std::vector<size_t> expected_indices(long_dim);

    auto idx_func = axis == 0 ? identity : swap_idx;

    T val(0);
    size_t q = 0;

    for (size_t j = 0; j < long_dim; j++) {
        for (size_t i = 0; i < short_dim; i++) {
            src[idx_func({q, j})] = val;
            if (i == short_dim - 1) {
                expected_values[idxr({j, 0})] = val;
                expected_indices[j] = q;
            }
            val += static_cast<T>(1);
            q = (q + 1) % short_dim;
        }
        // Stagger the starting index for each row/column
        // so that the max value appears at different places
        q = (q + 1) % short_dim;
    }
    return std::make_tuple(std::move(shape), std::move(src), std::move(expected_values), std::move(expected_indices));
}

template <typename T> void test_max(size_t axis) {
    auto [shape, src, expected_values, expected_indices] = make_max_axis<T>(axis);
    shape[axis] = 1;

    MatrixStorage<T> out(shape);
    std::vector<size_t> indices(shape[axis]);

    max(out, indices, src, axis);

    is_close(out, expected_values);
    is_close(indices, expected_indices);
}

template <typename T> void test_log() {
    const shape_type shape = {11, 5};
    MatrixStorage<T> src(shape);
    MatrixStorage<T> dst(shape);
    MatrixStorage<T> expected(shape);

    T min_val = static_cast<T>(shape[0]) - static_cast<T>(shape[0]) / static_cast<T>(2);

    for (size_t r = 0; r < shape[0]; r++) {
        const T v = min_val + r;
        for (size_t c = 0; c < shape[1]; c++) {
            src[{r, c}] = v;
            expected[{r, c}] = std::log(v);
        }
    }

    log(dst, src);
    is_close(dst, expected);
}

template <typename T> void test_tanh() {
    const shape_type shape = {11, 5};
    MatrixStorage<T> src(shape);
    MatrixStorage<T> dst(shape);
    MatrixStorage<T> expected(shape);

    T min_val = static_cast<T>(shape[0]) - static_cast<T>(shape[0]) / static_cast<T>(2);

    for (size_t r = 0; r < shape[0]; r++) {
        const T v = min_val + r;
        for (size_t c = 0; c < shape[1]; c++) {
            src[{r, c}] = v;
            expected[{r, c}] = std::tanh(v);
        }
    }

    log(dst, src);
    is_close(dst, expected);
}

template <typename T> void test_exp() {
    const shape_type shape = {11, 5};
    MatrixStorage<T> src(shape);
    MatrixStorage<T> dst(shape);
    MatrixStorage<T> expected(shape);

    T min_val = static_cast<T>(shape[0]) - static_cast<T>(shape[0]) / static_cast<T>(2);

    for (size_t r = 0; r < shape[0]; r++) {
        const T v = min_val + r;
        for (size_t c = 0; c < shape[1]; c++) {
            src[{r, c}] = v;
            expected[{r, c}] = std::exp(v);
        }
    }

    exp(dst, src);
    is_close(dst, expected);
}

template <typename T> void test_pow(const T &x) {
    const shape_type shape = {11, 5};
    MatrixStorage<T> src(shape);
    MatrixStorage<T> dst(shape);
    MatrixStorage<T> expected(shape);

    T min_val = static_cast<T>(shape[0]) - static_cast<T>(shape[0]) / static_cast<T>(2);

    for (size_t r = 0; r < shape[0]; r++) {
        const T v = min_val + r;
        for (size_t c = 0; c < shape[1]; c++) {
            src[{r, c}] = v;
            expected[{r, c}] = std::pow(v, x);
        }
    }

    pow(dst, src, x);
    is_close(dst, expected);
}

template <typename T> void test_one_hot() {
    std::vector<int> xs = {1, 2, 3, 4, 5};
    MatrixStorage<T> out({xs.size(), xs.size() + 1});

    one_hot(out, xs);

    MatrixStorage<T> expected({xs.size(), xs.size() + 1}, 0);
    for (size_t r = 0; r < xs.size(); r++) {
        const size_t c = static_cast<size_t>(xs[r]);
        expected[{r, c}] = static_cast<T>(1);
    }

    is_close(out, expected);
}

template <typename T> void test_transpose() {
    shape_type shape = {4, 5};
    shape_type shape_tp = {shape[1], shape[0]};
    MatrixStorage<T> src({shape});
    MatrixStorage<T> dst({shape_tp});
    auto expected = src.clone();

    fill_mat(src);
    transpose(dst, src);
    transpose(src, dst);
    is_close(src, expected);

    throws_exception(std::invalid_argument, transpose(src, src));
}

template <typename T> void test_eltwise_binary_func_bcast(size_t axis) {
    shape_type full_shape = {13, 19};
    shape_type bcast_shape = full_shape;
    shape_type bcast_fail_shape = full_shape;
    bcast_shape[axis] = 1;
    bcast_fail_shape[axis] = 2;

    MatrixStorage<T> full(full_shape);
    MatrixStorage<T> bcast(bcast_shape);
    MatrixStorage<T> bcast_fail(bcast_fail_shape);
    MatrixStorage<T> out(full_shape);
    MatrixStorage<T> expected(full_shape, 0);

    auto bcast_idx = axis == 0 ? bcast0 : bcast1;

    fill_mat(full);
    fill_mat(bcast);

    for (size_t r = 0; r < out.shape[0]; r++) {
        for (size_t c = 0; c < out.shape[1]; c++) {
            expected[{r, c}] = full[{r, c}] + bcast[bcast_idx({r, c})];
        }
    }

    eltwise_binary_func(out, full, bcast, [](const T &a, const T &b) { return a + b; });
    is_close(out, expected);

    eltwise_binary_func(out, bcast, full, [](const T &a, const T &b) { return a + b; });
    is_close(out, expected);

    throws_exception(std::invalid_argument,
                     eltwise_binary_func(out, bcast, bcast, [](const T &a, const T &b) { return a + b; }));
    throws_exception(std::invalid_argument,
                     eltwise_binary_func(out, full, bcast_fail, [](const T &a, const T &b) { return a + b; }));
    throws_exception(std::invalid_argument,
                     eltwise_binary_func(out, bcast_fail, full, [](const T &a, const T &b) { return a + b; }));
}

template <typename T> void test_broadcast(size_t axis) {
    // shape_type shape = {17, 23};
    shape_type shape = {4, 5};
    shape_type shape_bcast = shape;
    shape_bcast[axis] = 1;
    size_t off_axis = axis == 0 ? 1 : 0;

    // Ugh, special case for axis == 2, should size_t be used?
    // Should there be some special axis type? Or use ssize_t and -1?
    if (axis == 2) {
        shape_bcast[off_axis] = 1;
    }

    MatrixStorage<T> lhs(shape, 0);
    MatrixStorage<T> rhs(shape_bcast);

    MatrixStorage<T> out(shape);
    MatrixStorage<T> expected(shape);

    fill_mat(rhs);
    broadcast(out, rhs);

    if (axis == 0) {
        for (size_t r = 0; r < out.shape[0]; r++) {
            for (size_t c = 0; c < out.shape[1]; c++) {
                expected[{r, c}] = rhs[{0, c}];
            }
        }
    } else if (axis == 1) {
        for (size_t r = 0; r < out.shape[0]; r++) {
            for (size_t c = 0; c < out.shape[1]; c++) {
                expected[{r, c}] = rhs[{r, 0}];
            }
        }
    } else if (axis == 2) {
        for (size_t r = 0; r < out.shape[0]; r++) {
            for (size_t c = 0; c < out.shape[1]; c++) {
                expected[{r, c}] = rhs[{0, 0}];
            }
        }
    } else {
        std::stringstream ss;
        ss << "Unknown axis " << axis;
        throw std::invalid_argument(ss.str());
    }

    is_close(out, expected);

    if (axis == 2) {
        return;
    }

    // Verify that correct exceptions are thrown
    // Test axis == 0 throws
    // mxn op 1xn
    shape_type shape_bcast_fail = shape;
    shape_bcast_fail[axis] = 1;
    shape_bcast_fail[off_axis] ++;
    MatrixStorage<T> lhs_fail(shape);
    MatrixStorage<T> rhs_fail(shape_bcast_fail);
    throws_exception(std::invalid_argument, broadcast(lhs_fail, rhs_fail));
}

template <typename T> void test_mean(size_t axis) {
    shape_type shape = {3, 4};
    shape_type bcast_shape = shape;
    bcast_shape[axis] = 1;

    MatrixStorage<T> mat(shape);
    MatrixStorage<T> out(bcast_shape, 0);
    MatrixStorage<T> expected(bcast_shape, 0);
    fill_mat(mat);

    mean(out, mat, axis);

    const T divisor = static_cast<T>(1) / static_cast<T>(shape[axis]);
    sum(expected, mat, axis);
    multiply(expected, expected, divisor);

    is_close(out, expected);
}

template <typename T> void test_select_rows_and_cols() {
    const shape_type shape = {5, 5};
    const shape_type idx_shape = {5, 2};

    MatrixStorage<T> rhs(shape);
    MatrixStorage<size_t> idx(idx_shape);
    MatrixStorage<T> out({shape[0], 1});
    MatrixStorage<T> expected(out.shape);

    fill_mat(rhs);

    size_t c = 0;
    for (size_t r = 0; r < idx_shape[0]; r++) {
        idx[{r, 0}] = r;
        idx[{r, 1}] = c;
        c = (c + 2) % shape[1];
    }

    for (size_t r = 0; r < shape[0]; r++) {
        const size_t row = idx[{r, 0}];
        const size_t col = idx[{r, 1}];
        expected[{r, 0}] = rhs[{row, col}];
    }

    select_rows_and_cols(out, rhs, idx);
    is_close(out, expected);
}

template <typename T> void test_broadcast_rows() {
    MatrixStorage<T> src({3, 3});

    src[{0, 0}] = 1;
    src[{0, 1}] = 2;
    src[{0, 2}] = 3;
    src[{1, 0}] = 4;
    src[{1, 1}] = 5;
    src[{1, 2}] = 6;
    src[{2, 0}] = 7;
    src[{2, 1}] = 8;
    src[{2, 2}] = 9;

    MatrixStorage<size_t> idx({2, 1});
    idx[{0, 0}] = 2;
    idx[{1, 0}] = 0;

    MatrixStorage<float> dst({2, 3});

    broadcast_rows(dst, src, idx, assign_op<T>);

    MatrixStorage<T> expected({2, 3});
    expected[{0, 0}] = 7;
    expected[{0, 1}] = 8;
    expected[{0, 2}] = 9;
    expected[{1, 0}] = 1;
    expected[{1, 1}] = 2;
    expected[{1, 2}] = 3;

    is_close(dst, expected);

    dst = static_cast<T>(1);
    expected += static_cast<T>(1);
    broadcast_rows(dst, src, idx, accumulate_op<T>);
    is_close(dst, expected);
}

template <typename T> void test_bcast_all(size_t axis) {
    const shape_type base_shape = {5, 5};

    shape_type shape = base_shape;
    shape[axis] = 1;
    const shape_type bcast_shape = {1, 1};

    MatrixStorage<T> lhs(shape);
    MatrixStorage<T> rhs(bcast_shape);
    MatrixStorage<T> out(shape);
    MatrixStorage<T> expected(shape);
    const T val = static_cast<T>(0.1);

    fill_mat(lhs);
    rhs[{0, 0}] = val;

    for (size_t r = 0; r < shape[0]; r++) {
        for (size_t c = 0; c < shape[1]; c++) {
            expected[{r, c}] = lhs[{r, c}] * rhs[{0, 0}];
        }
    }

    multiply(out, lhs, rhs);
    is_close(out, expected);
}

template <typename T> void test_select_embeddings() {
    // Create an embedding matrix, vocab size 10, embedding dim 3
    MatrixStorage<T> emb({10, 3});

    T emb_val(0);
    for (size_t row = 0; row < emb.shape[0]; row++) {
        for (size_t col = 0; col < emb.shape[1]; col++) {
            emb[{row, col}] = emb_val;
            emb_val += static_cast<T>(1);
        }
    }

    // Create a selector - context block size 2
    MatrixStorage<size_t> x({5, 2});
    size_t select_val = 0;
    for (size_t row = 0; row < x.shape[0]; row++) {
        for (size_t col = 0; col < x.shape[1]; col++) {
            x[{row, col}] = select_val;
            select_val = (select_val + 1) % emb.shape[0];
        }
    }

    // Select embeddings
    const auto selected_shape = select_embeddings_shape(emb.shape, x.shape);
    MatrixStorage<T> selected(selected_shape);

    // Construct expected matrix
    MatrixStorage<T> expected(selected_shape);
    T expected_val = 0;
    for (size_t row = 0; row < expected.shape[0]; row++) {
        for (size_t col = 0; col < expected.shape[1]; col++) {
            expected[{row, col}] = expected_val;
            expected_val += 1;
        }
    }

    // Validate selection
    select_embeddings(selected, emb, x);

    is_close(selected, expected);
}


template<typename T>
void test_select_embeddings_bwd() {
    // Create an embedding matrix, vocab size 10, embedding dim 3
    MatrixStorage<T> demb({10, 3});
    // Create a selector - context block size 2
    MatrixStorage<size_t> x({10, 2});
    MatrixStorage<T> dselected({x.shape[0], x.shape[1] * demb.shape[1]});

    size_t select_val = 0;
    for (size_t row = 0; row < x.shape[0]; row++) {
        for (size_t col = 0; col < x.shape[1]; col++) {
            x[{row, col}] = select_val;
            select_val = (select_val + 1) % demb.shape[0];
        }
    }

    for (size_t row = 0; row < dselected.shape[0]; row++) {
        for (size_t col = 0; col < dselected.shape[1]; col++) {
            dselected[{row, col}] = static_cast<T>(1);
        }
    }

    MatrixStorage<T> expected({10, 3});
    for (size_t row = 0; row < expected.shape[0]; row++) {
        for (size_t col = 0; col < expected.shape[1]; col++) {
            expected[{row,col}] = static_cast<T>(2);
        }
    }

    select_embeddings_bwd(demb, dselected, x);
    is_close(demb, expected);
}

int main(int argc, char **argv) {
    test_constructor<float>();
    test_move_assign<float>();
    test_assign_value<float>();
    test_clone<float>();
    test_fill_value<float>();
    test_add<float>();
    test_subtract<float>();
    test_multiply<float>();
    test_divide<float>();
    test_matmul<float>();
    test_sum<float>(0);
    test_sum<float>(1);
    test_max<float>(0);
    test_max<float>(1);
    test_log<float>();
    test_tanh<float>();
    test_exp<float>();
    for (float y = -10; y <= 10; y += 0.1) {
        test_pow<float>(y);
    }

    test_one_hot<float>();
    test_transpose<float>();
    test_eltwise_binary_func_bcast<float>(0);
    test_eltwise_binary_func_bcast<float>(1);
    test_broadcast<float>(0);
    test_broadcast<float>(1);
    test_broadcast<float>(2);
    test_mean<float>(0);
    test_mean<float>(1);
    test_select_rows_and_cols<float>();
    test_broadcast_rows<float>();
    test_bcast_all<float>(0);
    test_bcast_all<float>(1);

    test_select_embeddings<float>();
    test_select_embeddings_bwd<float>();

    std::cout << argv[0] << ": " << num_passed << " passed / " << num_tests << " total" << std::endl;
    return 0;
}
