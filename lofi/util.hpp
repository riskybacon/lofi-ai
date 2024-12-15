#pragma once
#include <vector>
#include <stdexcept>
#include <fstream>

template <typename T>
std::vector<T> read_numpy_array(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    // Determine the file size
    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Read the data into a vector
    std::vector<T> data(size / sizeof(T));
    if (!file.read(reinterpret_cast<char*>(data.data()), size)) {
        throw std::runtime_error("Error reading file: " + filename);
    }

    return data;
}

template <typename T> std::ostream &operator<<(std::ostream &out, const std::vector<T> &vec) {
    out << "[";
    for (size_t i = 0; i < vec.size(); i++) {
        out << vec[i];
        if (i < vec.size() - 1) {
            out << ",";
        }
    }
    out << "]";
    return out;
}

template <typename T> bool is_close(const T &a, const T &b) {
    const T epsilon = 1e-4;
    return std::abs(a - b) <= epsilon;
}
