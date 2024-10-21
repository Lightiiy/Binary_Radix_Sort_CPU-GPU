cmake_minimum_required(VERSION 3.29)
project(Binary_Radix_Sort_CPU_GPU)

set(CMAKE_CXX_STANDARD 17)

add_executable(Binary_Radix_Sort_CPU_GPU
    "generate_data.cpp"
        radix_sort.cpp
        generate_data.h)
