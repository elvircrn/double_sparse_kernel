cmake_minimum_required(VERSION 3.26)
project(doublesparse CUDA CXX)

# Find Python Virtual Environment Torch Installation
set(TORCH_PATH "/home/elvircrn/.venv/lib/python3.12/site-packages/torch")

# Add LibTorch
set(Torch_DIR "${TORCH_PATH}/share/cmake/Torch")
find_package(Torch REQUIRED)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-line-info -O3")

# Find Torch
add_executable(
        doublesparse_exec
        main.cu
)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_link_options(-pie)
target_compile_options(doublesparse_exec PRIVATE "-g")
target_link_options(doublesparse_exec PRIVATE "-g")
target_link_libraries(doublesparse_exec "${TORCH_LIBRARIES}")
target_include_directories(doublesparse_exec PRIVATE ${TORCH_INCLUDE_DIRS})
