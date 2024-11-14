add_executable(test_storage test_storage.cpp)
target_include_directories(test_storage PRIVATE ${LOFI_INCLUDE_DIR})
target_link_libraries(test_storage PRIVATE ${OpenBLAS_LIBRARIES})

add_executable(test_context test_context.cpp)
target_include_directories(test_context PRIVATE ${LOFI_INCLUDE_DIR})
target_link_libraries(test_context PRIVATE ${OpenBLAS_LIBRARIES})

add_executable(test_engine test_engine.cpp)
target_include_directories(test_engine PRIVATE ${LOFI_INCLUDE_DIR})
target_link_libraries(test_engine PRIVATE ${OpenBLAS_LIBRARIES})

add_subdirectory(pytorch)