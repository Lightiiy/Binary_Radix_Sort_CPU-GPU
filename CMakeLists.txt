cmake_minimum_required(VERSION 3.29)
project(Binary_Radix_Sort_CPU_GPU LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build" FORCE)
endif()



add_executable(Binary_Radix_Sort_CPU_GPU
        radix_sort.cpp
        generate_data/generate_data.h
        generate_data/generate_data.cpp
        CUDA_files/radix_sort_gpu.cuh
        CUDA_files/radix_sort_gpu.cu
)
