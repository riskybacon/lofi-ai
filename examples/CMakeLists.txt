add_executable(mlp mlp.cpp)
target_include_directories(mlp PRIVATE ${LOFI_INCLUDE_DIR})
target_link_libraries(mlp PRIVATE ${OpenBLAS_LIBRARIES})

add_executable(one_layer one_layer.cpp)
target_include_directories(one_layer PRIVATE ${LOFI_INCLUDE_DIR})
target_link_libraries(one_layer PRIVATE ${OpenBLAS_LIBRARIES})
