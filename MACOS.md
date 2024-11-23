# Intro

Building and running on MacOS is not a priority for this project.

Out of the box, clang does not support OpenMP or the required c++23 features.

To run, install brew, g++, openblas, and cmake.

# Build

1. Install [Homebrew](https://brew.sh)

2. Install packages

```bash
brew install g++ openblas cmake
```

3. Build

```bash
mkdir build
cd build
cmake -DCMAKE_CXX_COMPILER=$(which g++-14) -DCMAKE_BUILD_TYPE=Release -DOpenBLAS_DIR=/opt/homebrew/Cellar/openblas/0.3.28/lib/cmake/openblas ...
```

# Run

```bash
export OMP_NUM_THREADS=4
./examples/mlp
```

Experiment with the number of threads to find the best performance.
