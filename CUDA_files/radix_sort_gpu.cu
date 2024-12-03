#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

using namespace std;

__global__ void calculate_flags(int* input_array, int* flag_array, int size, int bit_position) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        // Extract the bit at the specified position
        flag_array[idx] = (input_array[idx] >> bit_position) & 1;
    }
}

void calculatePrefix( thrust::device_vector<int>  &input_array, thrust::device_vector<int> &prefix_array) {    // Use Thrust's exclusive_scan to compute the prefix sum
    thrust::exclusive_scan(thrust::device, input_array.begin(), input_array.end(), prefix_array.begin());
}

__global__ void seperate_values(const int* input_array, const int* flag_array, const int*prefix_sum, int* output_array, int size, int total_sum) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        if (flag_array[idx]) {
            // printf( "\nThread: %d Output array: %d | input array: %d", idx, output_array[prefix_sum[idx]], input_array[idx]);
            output_array[prefix_sum[idx]] = input_array[idx];
        }
        else {
            int position = idx - prefix_sum[idx];
            output_array[total_sum + position] = input_array[idx];
            // printf( "\nThread: %d position: %d, total-sum: %d, output_array: %d, input_array: %d",idx, position, total_sum, output_array[total_sum + position], input_array[idx]);
        }
    }
}

void radixSortGPU(vector<int> &input_array, int defined_bits) {
    int size = input_array.size();
    int num_block = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    thrust::device_vector<int> device_input(input_array.begin(), input_array.end());
    thrust::device_vector<int> array_flags(size);
    thrust::device_vector<int> array_prefix(size);
    thrust::device_vector<int> device_output(size);

    for (int i = 0; i< defined_bits; i++) {
        printf("Loop iteration %d", i);
        calculate_flags<<<num_block, BLOCK_SIZE>>>( raw_pointer_cast(device_input.data()),  raw_pointer_cast(array_flags.data()), size,i);
        cudaDeviceSynchronize();
        vector<int> output_flag(array_flags.begin(), array_flags.end());

        calculatePrefix(array_flags, array_prefix);
        cudaDeviceSynchronize();
        vector<int> output_prefix(array_prefix.begin(), array_prefix.end());

        int total_sum = array_prefix[size - 1] + array_flags[size - 1];
        seperate_values<<<num_block, BLOCK_SIZE>>>(raw_pointer_cast(device_input.data()),
            thrust::raw_pointer_cast(array_flags.data()),
            thrust::raw_pointer_cast(array_prefix.data()),
            thrust::raw_pointer_cast(device_output.data()),
            size,
            total_sum);
        cudaDeviceSynchronize();
        vector<int> output_array(device_output.begin(), device_output.end());

        thrust::copy(device_output.begin(), device_output.end(), device_input.begin());

    }

    thrust::copy(device_input.begin(), device_input.end(), input_array.begin());
}