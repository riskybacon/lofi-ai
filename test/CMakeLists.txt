add_executable(test_storage test_storage.cpp)
target_include_directories(test_storage PRIVATE ${INCLUDE_DIRS})
target_link_libraries(test_storage PRIVATE ${OpenBLAS_LIBRARIES})

add_executable(test_context test_context.cpp)
target_include_directories(test_context PRIVATE ${INCLUDE_DIRS})
target_link_libraries(test_context PRIVATE ${OpenBLAS_LIBRARIES})

add_executable(test_engine test_engine.cpp)
target_include_directories(test_engine PRIVATE ${INCLUDE_DIRS})
target_link_libraries(test_engine PRIVATE ${OpenBLAS_LIBRARIES})

add_executable(test_layout test_layout.cpp)
target_include_directories(test_layout PRIVATE ${INCLUDE_DIRS})
target_link_libraries(test_layout PRIVATE ${OpenBLAS_LIBRARIES})

add_executable(test_tensor test_tensor.cpp)
target_include_directories(test_tensor PRIVATE ${INCLUDE_DIRS})
target_link_libraries(test_tensor PRIVATE ${OpenBLAS_LIBRARIES})

add_subdirectory(pytorch)
