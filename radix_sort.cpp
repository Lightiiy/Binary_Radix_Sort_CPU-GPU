#include <iostream>
#include <ostream>
#include <thread>
#include <mutex>
#include "generate_data/generate_data.h"
#include "radix_sort_utils/radix_sort_utils.h"
#include <vector>
using namespace std;

mutex mtx;

int main() {
    const string filename = "basic.txt";
    const int max_value = 100;
    const int array_size = 8;

    writeDataToFile(filename, max_value, array_size);

    optional<vector<int>> opt_data = readDataFromFile(filename);
    vector<int> test_data;
    if(opt_data) {
        test_data = *opt_data;
    } else {
        cout << "No data found" << endl;
        return 0;
    }
    vector<int> sorted_data(test_data.size());

    cout << "Test Data: ";
    for (const auto& num : test_data) {
        cout<< " " << num;
    }
    cout << endl;

    int test_bit = 0;

    // Ensure global counts are reset
    global_count[0] = 0;
    global_count[1] = 0;

    countPhase(test_data, test_bit);

    // Print results to verify correctness
    cout << "Global counts for bit position " << test_bit << ": "
         << "0s = " << global_count[0] << ", 1s = " << global_count[1] << endl;

    vector<int> position(2);
    position[0] = 0; // Start position for 0s
    position[1] = global_count[0]; // Start position for 1s

    reorderPhase(test_data, sorted_data, test_bit, position);

    cout << "Output Data: ";
    for (int num : sorted_data) {
        cout << num << " ";
    }
    cout << "\n";

    return 0;
}