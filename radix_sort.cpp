#include <iostream>
#include <ostream>
#include <fstream>
#include <thread>
#include <intrin.h>
#include <vector>
#include <optional>
#include <algorithm>
#include <filesystem>
#include <chrono>
#include "generate_data/generate_data.h"
#include "CUDA_files/radix_sort_gpu.cuh"


using namespace std;
namespace fs = filesystem;
chrono::high_resolution_clock::time_point begin_timer, end_timer;

void startTimer() {
    begin_timer = chrono::high_resolution_clock::now();
}

long long stopTimerAndLog(const std::string& task_name = "Task") {
    end_timer = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::nanoseconds>(end_timer - begin_timer).count();
    double duration_seconds = static_cast<double>(duration) / 1e9;
    cout << task_name << " took " << duration << " ns" << "(" << duration_seconds << ")s" << endl;
    return duration;
}

int calculateBitsRequired(int max_value) {
    unsigned long index;
    _BitScanReverse(&index, static_cast<unsigned>(max_value));
    return static_cast<int>(index) + 1;
}

optional<vector<int>> generateAndLoadDataFromFile(string filename, int max_value, int array_size )
{
    if (fs::exists(filename)) {
    cout << "File " << filename << " exists. Reading data from the file." << endl;
    optional<vector<int>> opt_data = readDataFromFile(filename);
    if (opt_data) {
        return *opt_data;
    } else {
        cout << "No data found in the file." << endl;
        return nullopt;
    }
} else {
    cout << "File " << filename << " does not exist. Generating new data and writing to file." << endl;
    writeRandomDataToFile(filename, max_value, array_size);
    optional<vector<int>> opt_data = readDataFromFile(filename);
    if (opt_data) {
        return *opt_data;
    } else {
        cout << "Failed to read generated data." << endl;
        return nullopt;
    }
}
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

        copy(one_bucket.begin(), one_bucket.end(), array.begin());
        copy(zero_bucket.begin(), zero_bucket.end(), array.begin() + one_bucket.size());
    }
}

int main() {
    const string filename = "basic.txt";
    const int max_value = 10000;
    const int array_size = 1000;

    auto import_data = generateDataDirectly((int)max_value, array_size);

    int bits_required = calculateBitsRequired(max_value);

    long long avg_duration = 0;

        vector<int> test_data;
        if(import_data) {
            test_data = *import_data;
        }
        else {
            std::cerr << "Failed to generate data." << std::endl;
        }

        cout << "Original Data ";
        for (const auto& num : test_data) {
            cout << " " << num;
        }
        cout << endl;

        startTimer();

        radixSortGPU(test_data, bits_required);
        // radixSortCPU(test_data, bits_required);

        string log_data = "MaxValue:"+to_string(max_value)+"_ArraySize"+to_string(array_size);
        long long duration = stopTimerAndLog();

    // cout << "Sorted Data ";
    // for (const auto& num : test_data) {
    //     cout << " " << num;
    // }
    // cout << endl;

    string sorted_filename = filename + "_sorted.txt";
    writeSortedDataToFile(sorted_filename, test_data);
    return 0;
}


// test_data = generateAndLoadDataFromFile(filename, max_value, array_size).value_or(vector<int>());




// ofstream outfile("GPU_MAX_VAL" + std::to_string(max_value) + ".txt");

// for ( int i = iterations; i >0; i--) {

// cout << bits_required;
// int iterations = 100;

// outfile << duration << "," << endl;
// avg_duration += duration;
// }
// outfile << avg_duration / iterations << endl;
// outfile.close();

//