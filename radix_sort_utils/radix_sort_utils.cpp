#include <mutex>
#include <vector>
#include <iostream>
#include <thread>
using namespace std;

int getBit(int num, int bit_position) {
    return (num & (1 << bit_position)) >> bit_position;
}


void countSortByBit(vector<int>& data, int bit_position, mutex& mtx) {
    vector<int> zero_bucket;
    vector<int> one_bucket;

    for (int num : data) {
        if (getBit(num, bit_position)) {
            one_bucket.push_back(num);
        } else {
            zero_bucket.push_back(num);
        }
    }

    vector<int> sorted_data;
    sorted_data.reserve(data.size());
    sorted_data.insert(sorted_data.end(), zero_bucket.begin(), zero_bucket.end());
    sorted_data.insert(sorted_data.end(), one_bucket.begin(), one_bucket.end());

    // Update the original data array with sorted data
    {
        lock_guard<mutex> guard(mtx);
        data = sorted_data;
    }
}

vector<int> radixSortByBits(vector<int> data, int num_bits) {
    mutex mtx;
    vector<thread> threads;
    for (int bit_position = 0; bit_position < num_bits; ++bit_position) {
        // Joining threads after each bit position sort to ensure sequential updates
        threads.emplace_back(countSortByBit, ref(data), bit_position, ref(mtx));
        threads[0].join();  // This will make it serial processing for each bit position
        threads.clear();
    }

    return data;
}