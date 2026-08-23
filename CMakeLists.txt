cmake_minimum_required(VERSION 3.20)
project(diffusion_gpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(diffusion_core src/solver.cpp)
target_include_directories(diffusion_core PUBLIC include)

add_executable(diffusion src/main.cpp)
target_link_libraries(diffusion PRIVATE diffusion_core)

add_executable(validate src/validate.cpp)
target_link_libraries(validate PRIVATE diffusion_core)

enable_testing()
add_executable(test_solver tests/test_solver.cpp)
target_link_libraries(test_solver PRIVATE diffusion_core)
add_test(NAME solver_tests COMMAND test_solver)

# CUDA is optional so the project still builds on machines without it.
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)

    add_library(diffusion_cuda src/solver_cuda.cu)
    target_link_libraries(diffusion_cuda PUBLIC diffusion_core)
    set_target_properties(diffusion_cuda PROPERTIES CUDA_ARCHITECTURES native)

    add_executable(benchmark src/benchmark.cpp)
    target_link_libraries(benchmark PRIVATE diffusion_cuda)

    add_executable(test_cuda tests/test_cuda.cpp)
    target_link_libraries(test_cuda PRIVATE diffusion_cuda)
    add_test(NAME cuda_tests COMMAND test_cuda)

    message(STATUS "CUDA found — building GPU targets")
else()
    message(STATUS "CUDA not found — building CPU targets only")
endif()