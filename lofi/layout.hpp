#pragma once
#include <iomanip>
#include <memory>
#include <numeric>
#include <ostream>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include <lofi/util.hpp>

using std::ranges::views::zip;
using std::views::iota;
using std::views::join;
using std::views::transform;

using extent_type = std::vector<size_t>;
using stride_type = std::vector<ssize_t>;

enum class KeepDim { True, False };

// Broadcast two shapes
std::vector<size_t> broadcast_extents(const extent_type &extent_a, const extent_type &extent_b) {
    size_t n = std::max(extent_a.size(), extent_b.size());
    extent_type bcast_exts(n, 1);

    for (size_t i = 0; i < n; ++i) {
        size_t dim_a = (i < n - extent_a.size()) ? 1 : extent_a[i - (n - extent_a.size())];
        size_t dim_b = (i < n - extent_b.size()) ? 1 : extent_b[i - (n - extent_b.size())];
        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            std::stringstream ss;
            ss << "extents " << extent_a << " and " << extent_b << " cannot be broadcast together.";
            throw std::invalid_argument("extents cannot be broadcast together.");
        }
        bcast_exts[i] = std::max(dim_a, dim_b);
    }

    return bcast_exts;
}

void throw_unequal_extents(const extent_type &actual, const extent_type &expected, const char *file, const int line) {
    std::stringstream ss;
    ss << "[" << file << ":" << line << "]: Expected extents=" << expected << ", got " << actual;
    throw std::invalid_argument(ss.str());
}

void _assert_equal_extents(const extent_type &actual, const extent_type &expected, const char *file, const int line) {
    if (actual != expected) {
        throw_unequal_extents(actual, expected, file, line);
    }
}

#define assert_equal_extents(a, b) _assert_equal_extents(a, b, __FILE__, __LINE__)

std::vector<size_t> unravel_index(size_t flat_idx, const std::vector<size_t> &extents) {
    std::vector<size_t> idx(extents.size(), 0);
    for (size_t i = 0; i < extents.size(); ++i) {
        size_t stride = 1;
        for (size_t j = i + 1; j < extents.size(); ++j) {
            stride *= extents[j];
        }
        idx[i] = (flat_idx / stride) % extents[i];
    }
    return idx;
}

std::vector<ssize_t> make_strides_row_major(const std::vector<size_t> &extents) {
    const size_t n = extents.size();
    std::vector<ssize_t> strides(n);
    size_t next_stride = 1;
    for (size_t j = 0; j < n; ++j) {
        const size_t i = n - 1 - j;
        if (extents[i] == 1) {
            strides[i] = 0;
        } else {
            strides[i] = next_stride;
            next_stride *= extents[i];
        }
    }

    return strides;
}

size_t num_elements(const std::vector<size_t> &extents) {
    return std::accumulate(std::cbegin(extents), std::cend(extents), 1, std::multiplies<>());
}

class Layout {
    const std::vector<size_t> extents_ = {};
    const std::vector<ssize_t> strides_ = {};
    const ssize_t offset_ = 0;
    const size_t size_ = 0;
    const size_t mod_ = 0;

  public:
    Layout() = default;

    Layout(const std::vector<size_t> &extents, const std::vector<ssize_t> &strides, const ssize_t offset)
        : extents_(extents), strides_(strides), offset_(offset), size_(num_elements(extents)), mod_(size_) {}

    Layout(const std::vector<size_t> &extents, const std::vector<ssize_t> &strides, const ssize_t offset,
           const size_t size, const size_t mod)
        : extents_(extents), strides_(strides), offset_(offset), size_(size), mod_(mod) {}

    Layout(std::vector<size_t> &&extents, std::vector<ssize_t> &&strides, const ssize_t offset, const size_t size,
           const size_t mod)
        : extents_(std::move(extents)), strides_(std::move(strides)), offset_(offset), size_(size), mod_(mod) {}

    Layout(const std::vector<size_t> &extents)
        : extents_(extents), strides_(make_strides_row_major(extents)), offset_(0), size_(num_elements(extents)),
          mod_(size_) {}

    Layout(const Layout &other)
        : extents_(other.extents_), strides_(other.strides_), offset_(other.offset_), size_(other.size_),
          mod_(other.mod_) {}

    size_t rank() const { return extents_.size(); }
    size_t size() const { return size_; }
    size_t extents(const size_t axis) const { return extents_.at(axis); }
    ssize_t strides(const size_t axis) const { return strides_.at(axis); }
    ssize_t offset() const { return offset_; }
    size_t mod() const { return mod_; }

    const std::vector<size_t> &extents() const { return extents_; }
    const std::vector<ssize_t> &strides() const { return strides_; }

    bool operator==(const Layout &other) const {
        return extents() == other.extents() && strides() == other.strides() && offset() == other.offset();
    }

    bool operator!=(const Layout &other) const { return !(*this == other); }

    /**
     * Broadcast this layout to the given extents. Indexes returned by the iterator will be repeated
     */
    auto broadcast(const std::vector<size_t> &bcast_extents) const {
        const size_t src_size = extents().size();
        const size_t dst_size = bcast_extents.size();

        if (dst_size < src_size) {
            std::stringstream ss;
            ss << "cannot broadast from `" << extents() << "` to `" << bcast_extents << "`";
            throw std::invalid_argument(ss.str());
        }

        // Check that these two extents can be broadcast throws std::invalid_argument on failure
        broadcast_extents(extents(), bcast_extents);

        std::vector<ssize_t> bcast_strides(bcast_extents.size());

        const size_t size_diff = dst_size - src_size;
        for (size_t i = 0; i < size_diff; ++i) {
            bcast_strides[i] = 0;
        }

        for (size_t i = 0, j = size_diff; i < src_size; ++i, ++j) {
            bcast_strides[j] = strides_[i];
        }

        const size_t new_size = num_elements(bcast_extents);

        return Layout(bcast_extents, bcast_strides, offset(), new_size, mod());
    }

    auto slice(const std::vector<std::tuple<size_t, size_t>> &params) const {
        if (params.size() > extents_.size()) {
            std::stringstream ss;
            ss << "too many parameters for tensor of rank " << rank();
            throw std::invalid_argument(ss.str());
        }

        std::vector<size_t> new_extents{extents()};
        std::vector<ssize_t> new_strides{strides()};

        ssize_t new_offset = offset_;

        for (auto [param, extent, stride] : zip(params, new_extents, new_strides)) {
            const auto [start, stop] = param;

            if (stop < start) {
                std::stringstream ss;
                ss << "slice: stop index is less than start index, start=" << start << ", stop=" << stop;
                throw std::invalid_argument(ss.str());
            }

            extent = stop - start;
            new_offset += start * stride;
        }

        const size_t new_size = num_elements(new_extents);
        return Layout(std::move(new_extents), std::move(new_strides), new_offset, new_size, mod());
    }

    auto slices(const size_t axis, KeepDim keepdim = KeepDim::False) const {
        if (axis >= rank()) {
            std::stringstream ss;
            ss << "axis `" << axis << "` out of bounds, rank=" << rank();
            throw std::invalid_argument(ss.str());
        }

        Layout layout(*this);

        return iota(static_cast<size_t>(0), extents(axis)) |
            transform([axis, layout, keepdim](size_t i) {
                auto new_layout = layout.slice({{i, i + 1}});
                if (keepdim == KeepDim::False) {
                    return new_layout.squeeze({0});
                }
                return new_layout;
            });
    }

    auto view(const std::vector<size_t> &extents) const {
        const size_t new_size = std::accumulate(std::cbegin(extents), std::cend(extents), 1, std::multiplies<>());

        if (new_size != size()) {
            std::stringstream ss;
            ss << "extents `" << extents << "` is invalid for input size " << size();
            throw std::invalid_argument(ss.str());
        }

        const size_t cur_dims = extents_.size();
        const size_t new_dims = extents.size();

        std::vector<ssize_t> strides(new_dims);

        const size_t src_offset = cur_dims > new_dims ? cur_dims - new_dims : 0;
        const size_t dst_offset = cur_dims > new_dims ? 0 : new_dims - cur_dims;

        std::copy(strides_.begin() + src_offset, strides_.end(), strides.begin() + dst_offset);

        for (size_t i = dst_offset; i > 0; --i) {
            strides[i - 1] = strides[i] * extents[i];
        }
        return Layout{extents, strides, offset(), num_elements(extents), mod()};
    }

    Layout squeeze() const {
        std::vector<size_t> sq_extents;
        std::vector<ssize_t> sq_strides;
        size_t size = 1;
        for (auto [extent, stride] : zip(extents(), strides())) {
            if (extent != 1) {
                sq_extents.push_back(extent);
                sq_strides.push_back(stride);
                size *= extent;
            }
        }

        return Layout(std::move(sq_extents), std::move(sq_strides), offset(), size, mod());
    }

    Layout squeeze(const std::unordered_set<size_t> &axes) const {
        std::vector<size_t> sq_extents;
        std::vector<ssize_t> sq_strides;

        for (size_t axis = 0; axis < rank(); ++axis) {
            if (axes.contains(axis) && extents(axis) == 1) {
                continue;
            }

            sq_extents.push_back(extents_[axis]);
            sq_strides.push_back(strides_[axis]);
        }

        return Layout(sq_extents, sq_strides, offset(), size(), mod());
    }

    size_t item() const {
        if (size() != 1) {
            std::stringstream ss;
            ss << "item() called on layout with `" << size() << "` elements, expected 1";
            throw std::invalid_argument(ss.str());
        }
        return offset();
    }

    class Iterator {
        size_t pos = 0;
        const std::vector<size_t> *extents = nullptr;
        const std::vector<ssize_t> *strides = nullptr;
        ssize_t offset = 0;
        std::vector<size_t> idxs;
        size_t mod = 0;

      public:
        using difference_type = int;
        using value_type = size_t;

        Iterator() = default;

        Iterator(size_t pos, const std::vector<size_t> *extents, const std::vector<ssize_t> *strides, ssize_t offset,
                 size_t mod)
            : pos{pos}, extents{extents}, strides{strides}, offset{offset}, mod{mod} {
            idxs = unravel_index(pos % mod, *extents);
        }

        bool operator==(Iterator const &other) const { return pos == other.pos; }

        bool operator!=(Iterator const &other) const { return !(*this == other); }

        Iterator &operator++() {
            inc();
            return *this;
        }

        Iterator operator++(int) {
            Iterator tmp = *this;
            inc();
            return tmp;
        }

        void inc() {
            ++pos;
            for (ssize_t i = idxs.size() - 1; i >= 0; i--) {
                size_t &idx = idxs[i];
                const size_t next_offset = offset + (*strides)[i];
                const size_t rewind_offset = offset - (*strides)[i] * ((*extents)[i] - 1);
                const size_t next_idx = idx + 1;
                const size_t rewind_idx = 0;
                const bool is_last = idx == (*extents)[i] - 1;

                idx = is_last ? rewind_idx : next_idx;
                offset = is_last ? rewind_offset : next_offset;

                if (!is_last) {
                    break;
                }

                // if (idxs[i] < (*extents)[i] - 1) {
                //     idxs[i]++;
                //     offset += (*strides)[i];
                //     break;
                // } else {
                //     idxs[i] = 0;
                //     offset -= (*strides)[i] * ((*extents)[i] - 1);
                // }
            }
        }

        value_type operator*() const { return offset % mod; }
    };

    auto begin() const { return Iterator{0, &extents_, &strides_, offset_, mod_}; }
    auto end() const { return Iterator{size(), &extents_, &strides_, offset_, mod_}; }

    auto cbegin() const { return Iterator{0, &extents_, &strides_, offset_, mod_}; }
    auto cend() const { return Iterator{size(), &extents_, &strides_, offset_, mod_}; }
};

void print_indexes(std::ostream &os, const Layout &t, const size_t indent, const size_t width) {
    const size_t n = t.extents(0);

    os << "[";
    if (t.rank() == 1) {
        for (const auto [i, t_i] : zip(iota(static_cast<size_t>(0), n), t)) {
            os << std::setw(width) << t_i;
            if (i < n - 1) {
                os << ", ";
            }
        }
        os << "]";
    } else {
        for (size_t i = 0; i < n; ++i) {
            const auto view_i = t.slice({{i, i + 1}}).squeeze({0});
            print_indexes(os, view_i, indent + 1, width);
            if (i < n - 1) {
                os << ",\n" << std::string(indent + 1, ' ');
            }
        }
        os << "]";
    }
}

void print_indexes(std::ostream &os, const Layout &t) {
    size_t size = t.size();
    size_t width = 0;
    while (size > 0) {
        size /= 10;
        width++;
    }

    print_indexes(os, t, 0, width);
    os << "\n";
}

std::ostream &operator<<(std::ostream &os, const Layout &t) {
    os << "(extents=" << t.extents() << ", strides=" << t.strides() << ", offset=" << t.offset() << ", mod=" << t.mod()
       << ")";
    return os;
}