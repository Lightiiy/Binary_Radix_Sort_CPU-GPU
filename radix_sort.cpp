#include <iostream>
#include <ostream>
#include <thread>
#include <intrin.h>
#include <vector>
#include <optional>
#include <algorithm>
#include <filesystem>
#include "generate_data/generate_data.h"
#include "CUDA_files/radix_sort_gpu.cuh"

//Google shared product that has a CUDA enviorment avaible

using namespace std;
namespace fs = filesystem;

int calculateBitsRequired(int max_value) {
    unsigned long index;
    _BitScanReverse(&index, static_cast<unsigned>(max_value));
    return static_cast<int>(index) + 1;
}

void sortSegment(const vector<int>& array, int start, int end, int shift, vector<int>& local_zero_bucket, vector<int>& local_one_bucket) {
    for (int i = start; i < end; ++i) {
        if ((array[i] & (1 << shift)) == 0) {
            local_zero_bucket.push_back(array[i]);
        } else {
            local_one_bucket.push_back(array[i]);
        }
    }
}

void radixSortCPU(vector<int>& array, int bits_required) {
    int num_threads = thread::hardware_concurrency();

    for (int shift = 0; shift < bits_required; ++shift) {
        vector<int> zero_bucket;
        vector<int> one_bucket;
        vector<thread> threads(num_threads);
        vector<vector<int>> zero_buckets(num_threads);
        vector<vector<int>> one_buckets(num_threads);
        int segment_size = array.size() / num_threads;

        for (int t = 0; t < num_threads; ++t) {
            int start = t * segment_size;
            int end = (t == num_threads - 1) ? array.size() : (t + 1) * segment_size;
            threads[t] = thread(sortSegment, cref(array), start, end, shift, ref(zero_buckets[t]), ref(one_buckets[t]));
        }

        for (auto& t : threads) {
            t.join();
        }

        for (int t = 0; t < num_threads; ++t) {
            zero_bucket.insert(zero_bucket.end(), zero_buckets[t].begin(), zero_buckets[t].end());
            one_bucket.insert(one_bucket.end(), one_buckets[t].begin(), one_buckets[t].end());
        }

        copy(zero_bucket.begin(), zero_bucket.end(), array.begin());
        copy(one_bucket.begin(), one_bucket.end(), array.begin() + zero_bucket.size());
    }
}

int main() {
    const string filename = "basic.txt";
    const int max_value = 1000000;
    const int array_size = 500000;

    vector<int> test_data;

    // Check if file exists
    if (fs::exists(filename)) {
        cout << "File " << filename << " exists. Reading data from the file." << endl;
        optional<vector<int>> opt_data = readDataFromFile(filename);
        if (opt_data) {
            test_data = *opt_data;
        } else {
            cout << "No data found in the file." << endl;
            return 0;
        }
    } else {
        cout << "File " << filename << " does not exist. Generating new data and writing to file." << endl;
        writeRandomDataToFile(filename, max_value, array_size);
        optional<vector<int>> opt_data = readDataFromFile(filename);
        if (opt_data) {
            test_data = *opt_data;
        } else {
            cout << "Failed to read generated data." << endl;
            return 0;
        }
    }

    cout << "Original Data ";
    for (const auto& num : test_data) {
        cout << " " << num;
    }
    cout << endl;
    int bits_required = calculateBitsRequired(max_value);

    radixSortGPU(test_data, bits_required);

    // radixSortCPU(test_data, bits_required);

    string sorted_filename = filename + "_sorted.txt";
    writeSortedDataToFile(sorted_filename, test_data);

    cout << "Sorted Data ";
    for (const auto& num : test_data) {
        cout << " " << num;
    }
    cout << endl;



    return 0;
}


// void radixSort(vector<int>& array, int max_value) {
//     int bits_required = calculateBitsRequired(max_value);
//
//     for (int shift = 0; shift < bits_required; ++shift) {
//         vector<int> zero_bucket;
//         vector<int> one_bucket;
//
//         for (int num : array) {
//             if ((num & (1 << shift)) == 0) {
//                 zero_bucket.push_back(num);
//             } else {
//                 one_bucket.push_back(num);
//             }
//         }
//
//         array.clear();
//         array.insert(array.end(), zero_bucket.begin(), zero_bucket.end());
//         array.insert(array.end(), one_bucket.begin(), one_bucket.end());
//     }
// }
