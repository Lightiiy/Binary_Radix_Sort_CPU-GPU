
#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include <optional>

using namespace std;

// Function declaration
void writeDataToFile(const string& filename, int max_value, int array_size);

optional<vector<int>> readDataFromFile(const string& filename);

#endif