#include <atomic>
#include <mutex>
#include <vector>
#include <iostream>
#include <thread>
#include <cmath>
#include <condition_variable>
using namespace std;

const int NUM_THREADS = thread::hardware_concurrency();
vector<atomic<int>> global_count(2);

int getBit(int num, int bit_position) {
    return (num & (1 << bit_position)) >> bit_position;
}

int calculateBitsRequired(int max_value) {
    return static_cast<int>(log2(max_value)) + 1;
}

void countBits(const vector<int>& array, int bit, int start, int end) {
    vector<int> local_count(2, 0);
    for (int i = start; i < end; ++i) {
        local_count[getBit(array[i], bit)]++;
    }
    global_count[0].fetch_add(local_count[0]);
    global_count[1].fetch_add(local_count[1]);
}

void countPhase(const vector<int>& array, int bit) {
    int n = array.size();
    int segments_size = (n + NUM_THREADS - 1) / NUM_THREADS;
    vector<std::thread> threads;
    for (int i = 0; i < NUM_THREADS; ++i) {
        int start = i * segments_size;
        int end = min((i + 1) * segments_size, n);
        if (start < n) {
            threads.emplace_back(countBits, ref(array), bit, start, end);
        }
    }

    for (auto& th : threads) {
        th.join();
    }
}

void reorderPhase(const std::vector<int>& array, std::vector<int>& output, int bit, const std::vector<int>& position) {
    int n = array.size();
    int segment_size = (n + NUM_THREADS - 1) / NUM_THREADS;
    vector<std::thread> threads;
    mutex mu;
    condition_variable cv;
    int completed_threads = 0;

    auto reorderTask = [&](int start, int end, int thread_id) {
       vector<int> local_position = position;
        if (start < n) {
            for (int i = start; i < end; ++i) {
                int bit_value = (array[i] >> bit) & 1;
                {
                    lock_guard<mutex> guard(mu);
                    output[local_position[bit_value]] = array[i];
                    cout << "Placed value " << array[i] << " at position " << local_position[bit_value] << " for bit " << bit_value << endl;
                }
                local_position[bit_value]++;
            }
        }
        else {
            cout << "Thread " << thread_id << " has no work to do." << endl;
        }
        {
            lock_guard<mutex> guard(mu);
            completed_threads++;
            cout << "Thread " << thread_id << " completed for segment [" << start << ", " << end << ") - Completed threads: " << completed_threads << " out of "<< NUM_THREADS << endl;
            if (completed_threads == NUM_THREADS) {
                cv.notify_all();
            }
        }
    };

    for (int i = 0; i < NUM_THREADS; ++i) {
        int start = i * segment_size;
        int end = min((i + 1) * segment_size, n);
        if (start < n) {
            cout << "Starting thread for segment [" << start << ", " << end << ")" << endl;
            threads.emplace_back(reorderTask, start, end, i);
        }
    }

    {
        unique_lock<mutex> lock(mu);
        cv.wait(lock, [&] { return completed_threads == NUM_THREADS; });
        cout << "All threads completed." << endl;
    }

    // Join the threads to ensure they have finished before moving forward
    for (auto& th : threads) {
        if (th.joinable()) {
            th.join();
        }
    }
}

// void countSortByBit(vector<int>& data, int bit_position, mutex& mtx) {
//     vector<int> zero_bucket;
//     vector<int> one_bucket;
//
//     for (int num : data) {
//         if (getBit(num, bit_position)) {
//             one_bucket.push_back(num);
//         } else {
//             zero_bucket.push_back(num);
//         }
//     }
//
//     vector<int> sorted_data;
//     sorted_data.reserve(data.size());
//     sorted_data.insert(sorted_data.end(), zero_bucket.begin(), zero_bucket.end());
//     sorted_data.insert(sorted_data.end(), one_bucket.begin(), one_bucket.end());
//
// }
//
// vector<int> radixSortByBits(vector<int> data, int num_bits) {
//     mutex mtx;
//     vector<thread> threads;
//     for (int bit_position = 0; bit_position < num_bits; ++bit_position) {
//         // Joining threads after each bit position sort to ensure sequential updates
//         threads.emplace_back(countSortByBit, ref(data), bit_position, ref(mtx));
//         threads[0].join();  // This will make it serial processing for each bit position
//         threads.clear();
//     }
//
//     return data;
// }