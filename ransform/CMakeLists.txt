cmake_minimum_required(VERSION 3.20)
project(MNNL VERSION 0.1.0 LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(CUDAToolkit REQUIRED)
add_subdirectory(tensor)

add_executable(my_lm main.cpp)
target_link_libraries(my_lm PRIVATE tensor CUDA::cudart CUDA::cublas)
target_include_directories(my_lm PRIVATE tensor)

add_executable(xor_demo xor.cpp)
target_link_libraries(xor_demo PRIVATE tensor CUDA::cudart CUDA::cublas)
target_include_directories(xor_demo PRIVATE tensor)

add_executable(nn_demo nn_demo.cpp)
target_link_libraries(nn_demo PRIVATE tensor CUDA::cudart CUDA::cublas)
target_include_directories(nn_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} tensor)

option(BUILD_BENCHMARKS "Build matmul benchmark (bench_matmul)" ON)
if(BUILD_BENCHMARKS)
    add_executable(bench_matmul bench/bench_matmul.cpp)
    target_link_libraries(bench_matmul PRIVATE tensor CUDA::cudart CUDA::cublas)
    target_include_directories(bench_matmul PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tensor)
endif()

option(BUILD_TESTS "Build GoogleTest unit tests" ON)
include(CTest)
if(BUILD_TESTING AND BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

include(GNUInstallDirs)
install(TARGETS tensor my_lm xor_demo nn_demo
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
if(BUILD_BENCHMARKS)
    install(TARGETS bench_matmul RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tensor/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mnnl/tensor
    FILES_MATCHING PATTERN "*.h")
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/nn/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mnnl/nn
    FILES_MATCHING PATTERN "*.h")
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/docs/THESIS_SCOPE.md
    DESTINATION ${CMAKE_INSTALL_DOCDIR}/mnnl)
