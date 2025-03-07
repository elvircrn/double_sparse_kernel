cmake_minimum_required(VERSION 3.26)
project(doublesparse CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CUDA_NVCC_FLAGS ${CUDA_FLAGS} -lineinfo)


# Find Torch
add_executable(
        doublesparse_exec
        main.cu
)

