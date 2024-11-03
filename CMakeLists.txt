cmake_minimum_required(VERSION 3.29)
project(Binary_Radix_Sort_CPU_GPU)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build" FORCE)
endif()


set(CMAKE_CXX_STANDARD 17)

add_executable(Binary_Radix_Sort_CPU_GPU
    "generate_data/generate_data.cpp"
        radix_sort.cpp
        generate_data/generate_data.h
        radix_sort_utils/radix_sort_utils.cpp
        radix_sort_utils/radix_sort_utils.h
        radix_sort.cpp)
