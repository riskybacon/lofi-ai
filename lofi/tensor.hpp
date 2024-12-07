#pragma once
#include <iostream>
#include <unordered_set>

#include <lofi/layout.hpp>
#include <lofi/util.hpp>

template <typename T> class Tensor {
    Layout layout_;
    std::shared_ptr<T[]> data_;

  public:
    Tensor() {}
    Tensor(const std::vector<size_t> &extents) : layout_(extents), data_(std::make_shared<T[]>(layout_.size())) {}

    Tensor(const Layout &layout, std::shared_ptr<T[]> data) : layout_(layout), data_(data) {}

    static Tensor zeros(const std::vector<size_t> &extents) {
        Tensor t(extents);
        t.fill_value(static_cast<T>(0));
        return t;
    }

    static Tensor zeros_like(const Tensor &other) { return zeros(other.extents()); }

    static Tensor ones(const std::vector<size_t> &extents) {
        Tensor t(extents);
        t.fill_value(static_cast<T>(1));
        return t;
    }

    static Tensor ones_like(const Tensor &other) { return ones(other.extents()); }

    static Tensor empty_like(const Tensor &t) { return Tensor(t.extents()); }

    void fill_value(const T &value) { std::fill(data_.get(), data_.get() + size(), value); }

    Tensor &operator=(const T &value) {
        fill_value(value);
        return *this;
    }

    size_t rank() const { return layout_.rank(); }
    size_t size() const { return layout_.size(); }

    const std::vector<size_t> &extents() const { return layout_.extents(); }
    const std::vector<ssize_t> &strides() const { return layout_.strides(); }
    const Layout &layout() const { return layout_; }

    size_t extents(size_t axis) const { return layout_.extents(axis); }

    auto broadcast(const std::vector<size_t> &extents) const { return Tensor(layout_.broadcast(extents), data_); }

    auto slice(const std::vector<std::tuple<size_t, size_t>> &params) const {
        return Tensor(layout_.slice(params), data_);
    }

    auto slices(const size_t axis, KeepDim keepdim = KeepDim::False) const {
        if (axis >= rank()) {
            std::stringstream ss;
            ss << "axis `" << axis << "` out of bounds, rank=" << rank();
            throw std::invalid_argument(ss.str());
        }

        Layout layout(layout_);
        auto data = data_;

        return layout.slices(axis, keepdim) |
            transform([this, layout, data](const Layout &l) { return Tensor(l, data); });
    }

    auto slices(KeepDim keepdim = KeepDim::False) const {
        return slices(0u, keepdim);
    }

    auto view(const std::vector<size_t> &extents) const { return Tensor(layout_.view(extents), data_); }

    auto squeeze() const { return Tensor(layout_.squeeze(), data_); }

    auto squeeze(const std::unordered_set<size_t> &axes) const { return Tensor(layout_.squeeze(axes), data_); }

    class Iterator {
        std::shared_ptr<T[]> data;
        Layout::Iterator offsets;

      public:
        using difference_type = int;
        using value_type = T;

        Iterator() = default;

        Iterator(const std::shared_ptr<T[]> &data, Layout::Iterator offsets) : data(data), offsets(offsets) {}

        bool operator==(Iterator const &other) const { return offsets == other.offsets; }

        bool operator!=(Iterator const &other) const { return !(*this == other); }

        Iterator &operator++() {
            ++offsets;
            return *this;
        }

        Iterator operator++(int) {
            Iterator tmp = *this;
            ++offsets;
            return tmp;
        }

        value_type &operator*() const { return data[*offsets]; }
    };

    auto begin() const { return Iterator{data_, layout_.begin()}; }

    auto end() const { return Iterator{data_, layout_.end()}; }

    auto cbegin() const { return Iterator{data_, layout_.begin()}; }

    auto cend() const { return Iterator{data_, layout_.end()}; }

    const T& item() const {
        if (size() != 1) {
            throw std::runtime_error("item() requires a single element tensor");
        }

        return data_[layout().item()];
    }

    T& item() {
        if (size() != 1) {
            throw std::runtime_error("item() requires a single element tensor");
        }

        return data_[layout().item()];
    }

};

template <typename T> void print(std::ostream &os, const Tensor<T> &t, const size_t indent) {
    const size_t n = t.extents(0);

    os << "[";
    os.flush();
    if (t.rank() == 1) {
        for (const auto [i, t_i] : zip(iota(static_cast<size_t>(0), n), t)) {
            os << std::fixed << std::setw(11) << std::setprecision(6) << t_i;
            os.flush();
            if (i < n - 1) {
                os << ", ";
                os.flush();
            }
        }
        os << "]";
        os.flush();
    } else {
        for (size_t i = 0; i < n; ++i) {
            const auto slice_i = t.slice({{i, i + 1}}).squeeze({0});
            print(os, slice_i, indent + 1);
            if (i < n - 1) {
                os << ",\n" << std::string(indent + 1, ' ');
                os.flush();
            }
        }
        os << "]";
        os.flush();
    }
}

template <typename T> std::ostream &operator<<(std::ostream &os, Tensor<T> &t) {
    print(os, t, 0);
    return os;
}

template <typename T, typename Func> void eltwise_unary_func(Tensor<T> &z, const Tensor<T> &x, Func func) {
    const auto x_bcast = x.broadcast(z.extents());
    for (auto [z_i, x_i] : zip(z, x_bcast)) {
        z_i = func(x_i);
    }
}

template <typename T, typename Func> void eltwise_unary_func(Tensor<T> &&z, const Tensor<T> &x, Func func) {
    eltwise_unary_func(z, x, func);
}

template <typename T, typename Func> void eltwise_unary_func(Tensor<T> &z, const Tensor<T> &&x, Func func) {
    eltwise_unary_func(z, x, func);
}

template <typename T, typename Func> void eltwise_unary_func(Tensor<T> &&z, const Tensor<T> &&x, Func func) {
    eltwise_unary_func(z, x, func);
}

template <typename T, typename Func>
void eltwise_binary_func(Tensor<T> &z, const Tensor<T> &x, const Tensor<T> &y, Func func) {
    auto z_expected_extents = broadcast_extents(x.extents(), y.extents());
    assert_equal_extents(z.extents(), z_expected_extents);

    const auto x_bcast = x.broadcast(z.extents());
    const auto y_bcast = y.broadcast(z.extents());

    for (auto [z_i, x_i, y_i] : zip(z, x_bcast, y_bcast)) {
        z_i = func(x_i, y_i);
    }
}

template <typename T, typename Func>
void eltwise_binary_func(Tensor<T> &&z, const Tensor<T> &x, const Tensor<T> &y, Func func) {
    eltwise_binary_func(z, x, y, func);
}

template <typename T, typename Func>
void eltwise_binary_func(Tensor<T> &z, const Tensor<T> &&x, const Tensor<T> &y, Func func) {
    eltwise_binary_func(z, x, y, func);
}

template <typename T, typename Func>
void eltwise_binary_func(Tensor<T> &z, const Tensor<T> &x, const Tensor<T> &&y, Func func) {
    eltwise_binary_func(z, x, y, func);
}

template <typename T, typename Func>
void eltwise_binary_func(Tensor<T> &&z, const Tensor<T> &&x, const Tensor<T> &y, Func func) {
    eltwise_binary_func(z, x, y, func);
}

template <typename T, typename Func>
void eltwise_binary_func(Tensor<T> &z, const Tensor<T> &&x, const Tensor<T> &&y, Func func) {
    eltwise_binary_func(z, x, y, func);
}

template <typename T, typename Func>
void eltwise_binary_func(Tensor<T> &&z, const Tensor<T> &x, const Tensor<T> &&y, Func func) {
    eltwise_binary_func(z, x, y, func);
}

template <typename T, typename Func>
void eltwise_binary_func(Tensor<T> &&z, const Tensor<T> &&x, const Tensor<T> &&y, Func func) {
    eltwise_binary_func(z, x, y, func);
}

template <typename T> void add(Tensor<T> &z, const Tensor<T> &x, const Tensor<T> &y) {
    eltwise_binary_func(z, x, y, [](const T &x_i, const T &y_i) { return x_i + y_i; });
}

template <typename T> void add(Tensor<T> &&z, const Tensor<T> &x, const Tensor<T> &y) { add(z, x, y); }

template <typename T> void add(Tensor<T> &z, const Tensor<T> &&x, const Tensor<T> &y) { add(z, x, y); }

template <typename T> void add(Tensor<T> &z, const Tensor<T> &x, const Tensor<T> &&y) { add(z, x, y); }

template <typename T> void add(Tensor<T> &&z, const Tensor<T> &&x, const Tensor<T> &y) { add(z, x, y); }

template <typename T> void add(Tensor<T> &z, const Tensor<T> &&x, const Tensor<T> &&y) { add(z, x, y); }

template <typename T> void add(Tensor<T> &&z, const Tensor<T> &x, const Tensor<T> &&y) { add(z, x, y); }

template <typename T> void add(Tensor<T> &&z, const Tensor<T> &&x, const Tensor<T> &&y) { add(z, x, y); }

template <typename T> void add(Tensor<T> &z, const Tensor<T> &x, const T &y) {
    eltwise_unary_func(z, x, [&y](const T &x_i) { return x_i + y; });
}

template <typename T> void add(Tensor<T> &&z, const Tensor<T> &x, const T &y) { add(z, x, y); }

template <typename T> void add(Tensor<T> &z, const Tensor<T> &&x, const T &y) { add(z, x, y); }

template <typename T> void add(Tensor<T> &z, const Tensor<T> &x, const T &&y) { add(z, x, y); }

template <typename T> void add(Tensor<T> &&z, const Tensor<T> &&x, const T &y) { add(z, x, y); }

template <typename T> void add(Tensor<T> &z, const Tensor<T> &&x, const T &&y) { add(z, x, y); }

template <typename T> void add(Tensor<T> &&z, const Tensor<T> &x, const T &&y) { add(z, x, y); }

template <typename T> void add(Tensor<T> &&z, const Tensor<T> &&x, const T &&y) { add(z, x, y); }

template <typename T> void subtract(Tensor<T> &z, const Tensor<T> &x, const T &y) {
    eltwise_unary_func(z, x, [&y](const T &x_i) { return x_i - y; });
}

template <typename T> void subtract(Tensor<T> &z, const Tensor<T> &x, const Tensor<T> &y) {
    eltwise_binary_func(z, x, y, [](const T &x_i, const T &y_i) { return x_i - y_i; });
}

template <typename T> void multiply(Tensor<T> &z, const Tensor<T> &x, const T &y) {
    eltwise_unary_func(z, x, [&y](const T &x_i) { return x_i * y; });
}

template <typename T> void multiply(Tensor<T> &z, const Tensor<T> &x, const Tensor<T> &y) {
    eltwise_binary_func(z, x, y, [](const T &x_i, const T &y_i) { return x_i * y_i; });
}

template <typename T> void divide(Tensor<T> &z, const Tensor<T> &x, const T &y) {
    eltwise_unary_func(z, x, [&y](const T &x_i) { return x_i / y; });
}

template <typename T> void divide(Tensor<T> &z, const Tensor<T> &x, const Tensor<T> &y) {
    eltwise_binary_func(z, x, y, [](const T &x_i, const T &y_i) { return x_i / y_i; });
}

template <typename T> std::tuple<bool, std::string> is_close(const Tensor<T> &a, const Tensor<T> &b) {
    if (a.extents() != b.extents()) {
        std::stringstream ss;
        ss << "a.extents (" << a.extents() << ") != b.extents (" << b.extents() << ")";
        return {false, ss.str()};
    }

    bool close = true;
    T max_diff = 0;
    T max_a_i = 0;
    T max_b_i = 0;
    T max_i = 0;
    const size_t n = a.size();

    for (auto [i, a_i, b_i] : zip(iota(0u, n), a, b)) {
        if (!is_close(a_i, b_i)) {
            const T diff = std::abs(a_i - b_i);
            if (close || diff > max_diff) {
                max_diff = diff;
                max_a_i = a_i;
                max_b_i = b_i;
                max_i = i;
                close = false;
            }
        }
    }

    std::stringstream ss;

    if (!close) {
        std::vector<size_t> coord = unravel_index(max_i, a.extents());
        ss << std::fixed << std::setprecision(16);
        ss << "a" << coord << "(" << max_a_i << ") != b" << coord << "(" << max_b_i << ")";
    }

    return {close, ss.str()};
}