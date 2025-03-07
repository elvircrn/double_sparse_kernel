cmake_minimum_required(VERSION 3.26)
project(doublesparse CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-line-info")

# Find Torch
add_executable(
        doublesparse_exec
        main.cu
)

