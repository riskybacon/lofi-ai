#include <lofi/test_helper.hpp>

template <typename T> auto make_context_ptr(const shape_type &shape) { return std::make_shared<Context<T>>(shape); }

template <typename T> void test_constructor() {
    const shape_type shape = {51, 17};
    Context<T> ctx0(shape);

    is_equal(ctx0.data.shape, shape);
    is_equal(ctx0.grad.shape, shape);
    is_equal(ctx0.label, std::string());

    Context<T> ctx1(shape, "label a");
    is_equal(ctx1.data.shape, shape);
    is_equal(ctx1.grad.shape, shape);
    is_equal(ctx1.label, std::string("label a"));
}

template <typename T> void test_assign() {
    const shape_type shape = {24, 43};
    auto prev0 = make_context_ptr<T>(shape);
    auto prev1 = make_context_ptr<T>(shape);
    auto out0 = make_context_ptr<T>(shape);

    out0->prev = {prev0, prev1};
    out0->op = "+";

    is_equal(out0->data.shape, shape);
    is_equal(out0->prev.size(), 2);
    is_equal(out0->prev[0]->data.id(), prev0->data.id());
    is_equal(out0->prev[1]->data.id(), prev1->data.id());
    is_equal(out0->op, "+");
}

template <typename T> void test_add() {
    const shape_type shape = {4, 3};
    auto lhs = make_context_ptr<T>(shape);
    auto rhs = make_context_ptr<T>(shape);
    auto out = make_context_ptr<T>(shape);

    lhs->data = 2;
    rhs->data = -1;

    MatrixStorage<T> expected_data(shape, 1);
    add(out, lhs, rhs);
    is_close(out->data, expected_data);

    MatrixStorage<T> expected_grad(shape);
    out->grad = 1;

    // Unzero the lhs and rhs grads to ensure that
    // the backward pass accumulates properly
    lhs->grad = 10;
    rhs->grad = 11;

    backward(out);

    expected_grad = 11;
    is_close(lhs->grad, expected_grad);
    expected_grad = 12;
    is_close(rhs->grad, expected_grad);
}

template <typename T> void test_subtract() {
    const shape_type shape = {3,4};
    T val0 = 50;
    T val1 = -10;

    auto lhs = make_context_ptr<T>(shape);
    auto rhs = make_context_ptr<T>(shape);
    auto out = make_context_ptr<T>(shape);
    auto expected = make_context_ptr<T>(shape);

    lhs->data = val0;
    rhs->data = val1;
    expected->data = val0 - val1;

    out->grad = 1;
    lhs->grad = 1;
    rhs->grad = 1;

    subtract(out, lhs, rhs);
    backward(out);

    // Use existing add functionality to valid forward and backward for subtraction
    auto lhs_add = make_context_ptr<T>(shape);
    auto rhs_add = make_context_ptr<T>(shape);
    auto out_add = make_context_ptr<T>(shape);

    lhs_add->data = val0;
    rhs_add->data = -val1;

    out_add->grad = 1;
    lhs_add->grad = 1;
    rhs_add->grad = 1;

    add(out_add, lhs_add, rhs_add);
    backward(out_add);

    is_close(out->data, expected->data);
    is_close(lhs->grad, lhs_add->grad);
    is_close(rhs->grad, rhs_add->grad);
}

template <typename T> void test_add_bcast_0() {
    const shape_type shape0 = {4, 3};
    const shape_type shape1 = {1, shape0[1]};

    auto mat = make_context_ptr<float>(shape0);
    auto col = make_context_ptr<float>(shape1);
    auto out0 = make_context_ptr<float>(shape0);
    auto out1 = make_context_ptr<float>(shape0);

    mat->data = 2;
    col->data = -1;

    add(out0, mat, col);

    out0->grad = 1;
    mat->grad = 10;
    col->grad = 11;
    backward(out0);

    MatrixStorage<float> mat_grad_expected(shape0, 11);
    MatrixStorage<float> col_grad_expected(shape1, 48);

    is_close(mat->grad, mat_grad_expected);
    is_close(col->grad, col_grad_expected);

    add(out1, col, mat);

    out1->grad = 1;
    mat->grad = 10;
    col->grad = 11;
    backward(out1);

    is_close(mat->grad, mat_grad_expected);
    is_close(col->grad, col_grad_expected);
}

template <typename T> void test_add_bcast_1() {
    const shape_type shape0 = {4, 3};
    const shape_type shape1 = {shape0[0], 1};

    auto mat = make_context_ptr<float>(shape0);
    auto col = make_context_ptr<float>(shape1);
    auto out0 = make_context_ptr<float>(shape0);
    auto out1 = make_context_ptr<float>(shape0);

    mat->data = 2;
    col->data = -1;

    add(out0, mat, col);

    out0->grad = 1;
    mat->grad = 10;
    col->grad = 11;

    backward(out0);

    MatrixStorage<float> mat_grad_expected(shape0, 11);
    MatrixStorage<float> col_grad_expected(shape1, 36);

    is_close(mat->grad, mat_grad_expected);
    is_close(col->grad, col_grad_expected);

    add(out1, col, mat);

    out1->grad = 1;
    mat->grad = 10;
    col->grad = 11;
    backward(out1);

    is_close(mat->grad, mat_grad_expected);
    is_close(col->grad, col_grad_expected);
}

template <typename T> void test_multiply() {
    const shape_type shape = {5, 4};
    auto lhs = make_context_ptr<float>(shape);
    auto rhs = make_context_ptr<float>(shape);
    auto out = make_context_ptr<float>(shape);

    fill_mat(lhs->data);
    fill_mat(rhs->data, static_cast<T>(10));

    auto expected = make_context_ptr<float>(shape);
    for (size_t r = 0; r < shape[0]; r++) {
        for (size_t c = 0; c < shape[1]; c++) {
            expected->data[{r, c}] = lhs->data[{r, c}] * rhs->data[{r, c}];
        }
    }

    multiply(out, lhs, rhs);
    is_close(out->data, expected->data);

    out->grad = static_cast<T>(1);
    lhs->grad = 1;
    rhs->grad = 1;

    backward(out);

    // Use ?hs->data as an expected value. Increase by 1
    // to confirm that gradient accumulation is working
    add(lhs->data, lhs->data, static_cast<T>(1));
    add(rhs->data, rhs->data, static_cast<T>(1));

    is_close(rhs->grad, lhs->data);
    is_close(lhs->grad, rhs->data);
}

template <typename T> void test_divide() {
    shape_type shape = {5, 4};
    auto lhs = make_context_ptr<float>(shape);
    auto rhs = make_context_ptr<float>(shape);
    auto out = make_context_ptr<float>(shape);

    fill_mat(lhs->data);
    fill_mat(rhs->data, static_cast<T>(10));

    lhs->grad = 1;
    rhs->grad = 1;
    out->grad = 1;
    divide(out, lhs, rhs);
    backward(out, false);

    auto lhs_mul = make_context_ptr<float>(shape);
    auto rhs_mul = make_context_ptr<float>(shape);
    auto out_mul = make_context_ptr<float>(shape);

    fill_mat(lhs_mul->data);
    fill_mat(rhs_mul->data, static_cast<T>(10));
    eltwise_unary_func(rhs_mul->data, rhs_mul->data, [](const T &x) { return 1 / x; });

    lhs_mul->grad = 1;
    rhs_mul->grad = 1;
    out_mul->grad = 1;

    multiply(out_mul, lhs_mul, rhs_mul);
    backward(out_mul, false);

    is_close(lhs->grad, lhs_mul->grad);
    is_close(rhs->grad, rhs_mul->grad);
    is_close(out->data, out_mul->data);
}

template <typename T> void test_matmul() {
    const shape_type shape0 = {5, 6};
    const shape_type shape1 = {6, 5};

    auto lhs = make_context_ptr<float>(shape0);
    auto rhs = make_context_ptr<float>(shape1);
    auto out = make_context_ptr<float>({shape0[0], shape1[1]});

    fill_mat(lhs->data, static_cast<T>(-2));
    fill_mat(rhs->data, static_cast<T>(10));

    matmul(out, lhs, rhs);

    // Make identity matrix
    for (size_t r = 0; r < out->shape()[0]; r++) {
        for (size_t c = 0; c < out->shape()[1]; c++) {
            out->grad[{r, c}] = r == c ? static_cast<T>(1) : static_cast<T>(0);
        }
    }

    lhs->grad = 1;
    rhs->grad = 1;

    backward(out, false);

    MatrixStorage<float> lhs_expected(shape0);
    MatrixStorage<float> rhs_expected(shape1);

    transpose(lhs_expected, rhs->data);
    transpose(rhs_expected, lhs->data);
    add(lhs_expected, lhs_expected, static_cast<T>(1));
    add(rhs_expected, rhs_expected, static_cast<T>(1));

    is_close(lhs->grad, lhs_expected);
    is_close(rhs->grad, rhs_expected);
}

template <typename T> void test_mean(size_t axis) {
    const shape_type shape = {3, 4};
    shape_type bcast_shape = shape;
    bcast_shape[axis] = 1;
    const T divisor = static_cast<T>(1) / static_cast<T>(shape[axis]);

    auto mat = make_context_ptr<T>(shape);
    auto out = make_context_ptr<T>(bcast_shape);

    fill_mat(mat->data);
    mean(out, mat, axis);

    MatrixStorage<T> expected(bcast_shape);
    sum(expected, mat->data, axis);
    multiply(expected, expected, divisor);
    is_close(out->data, expected);

    out->grad = 1;
    mat->grad = 1;
    T expected_val = mat->grad[{0, 0}] + out->grad[{0, 0}] * divisor;

    backward(out);

    MatrixStorage<T> expected_grad(shape);
    expected_grad = expected_val;
    is_close(mat->grad, expected_grad);
}

template <typename T> void test_select_rows_and_cols() {
    const shape_type shape = {5, 5};
    const shape_type idx_shape = {5, 2};

    auto rhs = make_context_ptr<T>(shape);
    auto out = make_context_ptr<T>({shape[0], 1});
    auto idx = make_context_ptr<size_t>(idx_shape);
    auto expected = make_context_ptr<T>(shape);

    fill_mat(rhs->data);

    size_t c = 0;
    for (size_t r = 0; r < idx_shape[0]; r++) {
        idx->data[{r, 0}] = r;
        idx->data[{r, 1}] = c;
        c = (c + 1) % shape[1];
    }

    select_rows_and_cols(out, rhs, idx);

    out->grad = .5;
    rhs->grad = 1;

    MatrixStorage<T> expected_grad(shape);

    for (size_t r = 0; r < shape[0]; r++) {
        const auto selected_idx = idx->data[{r, 1}];
        for (size_t c = 0; c < shape[1]; c++) {
            T val = c == selected_idx ? 1 + out->grad[{r, 0}] : 1;
            expected_grad[{r, c}] = val;
        }
    }

    backward(out, false);
    is_close(rhs->grad, expected_grad);
}

int main(int argc, char **argv) {
    test_constructor<float>();
    test_assign<float>();
    test_add<float>();
    test_subtract<float>();
    test_add_bcast_0<float>();
    test_add_bcast_1<float>();
    test_multiply<float>();
    test_divide<float>();  // failing, need to revisit
    test_matmul<float>();
    test_mean<float>(0);
    test_mean<float>(1);
    test_select_rows_and_cols<float>();

    std::cout << argv[0] << ": " << num_passed << " passed / " << num_tests << " total" << std::endl;
    return 0;
}
