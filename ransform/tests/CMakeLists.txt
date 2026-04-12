include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

add_executable(tensor_tests
  tensor_core_test.cpp
  nn_module_test.cpp
)
target_link_libraries(tensor_tests PRIVATE tensor GTest::gtest GTest::gtest_main CUDA::cudart CUDA::cublas)
target_include_directories(tensor_tests PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../tensor
  ${CMAKE_CURRENT_SOURCE_DIR}/..)

gtest_discover_tests(tensor_tests
  PROPERTIES LABELS "tensor"
  DISCOVERY_MODE PRE_TEST
)
