add_executable(test test.cpp)
target_include_directories(test PRIVATE ${INCLUDE_DIRS})
target_link_libraries(test PRIVATE ${OpenBLAS_LIBRARIES})
