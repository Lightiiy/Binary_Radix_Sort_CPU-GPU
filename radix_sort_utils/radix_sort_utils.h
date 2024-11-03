//
// Created by pherb on 03.11.2024.
//

#ifndef RADIX_SORT_UTILS_CPP_H
#define RADIX_SORT_UTILS_CPP_H

#include <vector>
#include <mutex>

using namespace std;

// Bit Extraction Functions
int getBit(int num, int bit_position);

// Core Sorting Function for Specific Bit
vector<int> countSortByBit(vector<int>& data, int bit_position, vector<vector<int>>& result, int thread_id, mutex& mtx);

// // Multithreaded Radix Sort Controller
vector<int> radixSortByBits(vector<int>& data, int num_bits);

#endif //RADIX_SORT_UTILS_CPP_H
