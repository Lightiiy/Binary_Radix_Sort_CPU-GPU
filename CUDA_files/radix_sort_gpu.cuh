
#ifndef RADIX_SORT_GPU_CUH
#define RADIX_SORT_GPU_CUH

#include <vector>
#include <optional>

void radixSortGPU(std::vector<int>& h_vector, int bits_required);
optional<vector<int>> generateDataDirectly( int max_value, int array_size);

#endif //RADIX_SORT_GPU_CUH
