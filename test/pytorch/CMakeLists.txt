add_executable(test test.cpp)
target_include_directories(test PRIVATE ${LOFI_INCLUDE_DIR})
target_link_libraries(test PRIVATE ${OpenBLAS_LIBRARIES})
