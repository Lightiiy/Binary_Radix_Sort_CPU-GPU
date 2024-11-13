
#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include <optional>

using namespace std;

// Function declaration
void writeRandomDataToFile(const string& filename, int max_value, int array_size);

optional<vector<int>> readDataFromFile(const string& filename);

void writeSortedDataToFile(const string& filename, const vector<int>& data);

#endif