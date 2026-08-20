cmake_minimum_required(VERSION 3.20)
project(diffusion_gpu LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Core solver library (CPU)
add_library(diffusion_core src/solver.cpp)
target_include_directories(diffusion_core PUBLIC include)

# CLI driver
add_executable(diffusion src/main.cpp)
target_link_libraries(diffusion PRIVATE diffusion_core)

# Test 
enable_testing()
add_executable(test_solver tests/test_solver.cpp)
target_link_libraries(test_solver PRIVATE diffusion_core)
add_test(NAME solver_tests COMMAND test_solver)