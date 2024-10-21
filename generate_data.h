
#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include <optional>


// Function declaration
void writeDataToFile(const std::string& filename, int max_value, int array_size);

std::optional<std::vector<int>> readDataFromFile(const std::string& filename);


#endif