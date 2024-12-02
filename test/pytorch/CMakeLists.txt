add_executable(test_pytorch_equivalence EXCLUDE_FROM_ALL test_pytorch_equivalence.cpp)
target_include_directories(test_pytorch_equivalence PRIVATE ${INCLUDE_DIRS} ${CMAKE_BINARY_DIR}/test/pytorch)
target_link_libraries(test_pytorch_equivalence PRIVATE ${OpenBLAS_LIBRARIES})

add_test(NAME make_pytorch_fixtures
    COMMAND ${CMAKE_COMMAND} -E env python3 ${CMAKE_SOURCE_DIR}/test/pytorch/make_tensors.py ${CMAKE_SOURCE_DIR}/examples/names.txt ${CMAKE_BINARY_DIR}/test/pytorch/data
)

set_tests_properties(make_pytorch_fixtures PROPERTIES FIXTURES_SETUP pytorch_fixtures)

add_test(NAME build_test_pytorch_equivalence
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target test_pytorch_equivalence
)

set_tests_properties(build_test_pytorch_equivalence
    PROPERTIES FIXTURES_SETUP test_pytorch_equivalence_executable FIXTURES_REQUIRED pytorch_fixtures)

add_test(NAME test_pytorch_equivalence
    COMMAND test_pytorch_equivalence ${CMAKE_BINARY_DIR}/test/pytorch/data
)
set_tests_properties(test_pytorch_equivalence PROPERTIES FIXTURES_REQUIRED test_pytorch_equivalence_executable)

add_executable(test_pytorch_equivalence_bn EXCLUDE_FROM_ALL test_pytorch_equivalence_bn.cpp)
target_include_directories(test_pytorch_equivalence_bn PRIVATE ${INCLUDE_DIRS} ${CMAKE_BINARY_DIR}/test/pytorch)
target_link_libraries(test_pytorch_equivalence_bn PRIVATE ${OpenBLAS_LIBRARIES})

add_test(NAME make_pytorch_fixtures_bn
    COMMAND ${CMAKE_COMMAND} -E env python3 ${CMAKE_SOURCE_DIR}/test/pytorch/make_tensors_bn.py ${CMAKE_SOURCE_DIR}/examples/names.txt ${CMAKE_BINARY_DIR}/test/pytorch/data_bn
)

set_tests_properties(make_pytorch_fixtures_bn PROPERTIES FIXTURES_SETUP pytorch_fixtures)

add_test(NAME build_test_pytorch_equivalence_bn
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target test_pytorch_equivalence_bn
)

set_tests_properties(build_test_pytorch_equivalence_bn
    PROPERTIES FIXTURES_SETUP test_pytorch_equivalence_bn_executable FIXTURES_REQUIRED pytorch_fixtures)

add_test(NAME test_pytorch_equivalence_bn
    COMMAND test_pytorch_equivalence_bn ${CMAKE_BINARY_DIR}/test/pytorch/data
)
set_tests_properties(test_pytorch_equivalence_bn PROPERTIES FIXTURES_REQUIRED test_pytorch_equivalence_bn_executable)
