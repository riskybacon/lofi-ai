cmake_minimum_required(VERSION 3.10)
project(lofi-ai VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(LOFI_INCLUDE_DIR ${PROJECT_SOURCE_DIR})

set(ARCH ${CMAKE_SYSTEM_PROCESSOR})
find_package(OpenBLAS REQUIRED)

set(INCLUDE_DIRS ${LOFI_INCLUDE_DIR} ${OpenBLAS_INCLUDE_DIRS})
add_definitions(-DPROJECT_SOURCE_DIR="${PROJECT_SOURCE_DIR}")

enable_testing()

add_subdirectory(examples)
add_subdirectory(test)

option(ENABLE_BOUNDS_CHECK "Enable bounds checking" OFF)

if(ENABLE_BOUNDS_CHECK)
    add_compile_definitions(BOUNDS_CHECK)
endif()