cmake_minimum_required(VERSION 3.10)
project(lofi-ai VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

set(LOFI_INCLUDE_DIR ${PROJECT_SOURCE_DIR})

set(ARCH ${CMAKE_SYSTEM_PROCESSOR})
set(OpenBLAS_DIR "/usr/lib/${ARCH}-linux-gnu/openblas-pthread/cmake/openblas")
find_package(OpenBLAS REQUIRED)

add_definitions(-DPROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

add_subdirectory(examples)
add_subdirectory(test)