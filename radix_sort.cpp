#include <iostream>
#include <ostream>
#include <thread>
#include <mutex>
#include "generate_data/generate_data.h"
#include "radix_sort_utils/radix_sort_utils.h"
#include <vector>
using namespace std;

int main() {
    const string filename = "basic.txt";
    const int max_value = 100;
    const int array_size = 20;

    writeDataToFile(filename, max_value, array_size);

    optional<vector<int>> opt_data = readDataFromFile(filename);
    vector<int> test_data;
    if(opt_data) {
        test_data = *opt_data;
    } else {
        cout << "No data found" << endl;
        return 0;
    }

    const int num_bits = calculateBitsRequired(max_value);
    vector<int> sorted_data = radixSortByBits(test_data, num_bits);

    cout << num_bits << "Sorted array: ";
    for (int num : sorted_data) {
        cout << num << " ";
    }
    cout << endl;


    return 0;
}