#pragma once
#include <random>

std::mt19937 generator() {
    std::random_device rd;
    std::mt19937 g(rd());
    return g;
}
