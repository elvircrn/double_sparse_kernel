cmake_minimum_required(VERSION 3.25)
project(doublesparse CUDA CXX)

set(CMAKE_CXX_STANDARD 17)

# Find Torch
add_executable(
        doublesparse_exec
        doublesparse_cuda_kernel.cu
)
target_compile_definitions(doublesparse_exec PUBLIC PYBIND_SKIP=1)

set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -arch=native")
set(CMAKE_CUDA_FLAGS_DEBUG "-lineinfo")

set_property(TARGET doublesparse_exec PROPERTY INTERFACE_COMPILE_OPTIONS "")