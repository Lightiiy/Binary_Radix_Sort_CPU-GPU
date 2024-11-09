//
// Created by pherb on 03.11.2024.
//

#ifndef RADIX_SORT_UTILS_CPP_H
#define RADIX_SORT_UTILS_CPP_H

#include <vector>
#include <atomic>
#include <mutex>
using namespace std;

extern const int NUM_THREADS;
extern vector<atomic<int>> global_count;

// Bit Extraction Functions
int getBit(int num, int bit_position);
int calculateBitsRequired(int max_value);
void countBits(const vector<int>& array, int bit, int start, int end);
void countPhase(const vector<int>& array, int bit);
void reorderPhase(const vector<int>& array, vector<int>& output, int bit, const vector<int>& position);
// void radixSort(vector<int>& array);


#endif //RADIX_SORT_UTILS_CPP_H
