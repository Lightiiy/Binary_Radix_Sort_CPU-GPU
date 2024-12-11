#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector>
#include <optional>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

using namespace std;

__global__ void generate_random_numbers(int *array, int max_value, int size, unsigned int seed) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // if(idx % 1000 == 0) {
    //     printf("Generating random numbers by thread: %d\n", idx);
    // }
    if (idx < size) {
        // Initialize CURAND
        curandState state;
        curand_init(seed, idx, 0, &state);

        // Generate random number
        array[idx] = curand(&state) % (max_value + 1);
    }
}

optional<vector<int>> generateDataDirectly( int max_value, int array_size) {
    try {
        thrust::host_vector<int> host_vector(array_size);
        thrust::device_vector<int> device_vector(array_size);

        int *raw_device_ptr = thrust::raw_pointer_cast(device_vector.data());

        // Launch kernel
        int threads_per_block = 256;
        int blocks = (array_size + threads_per_block - 1) / threads_per_block;
        generate_random_numbers<<<blocks, threads_per_block>>>(raw_device_ptr, max_value, array_size, static_cast<unsigned int>(std::time(0)));

        // Copy results back to host
        thrust::copy(device_vector.begin(), device_vector.end(), host_vector.begin());

        // Convert to std::vector<int>
        std::vector<int> result(host_vector.begin(), host_vector.end());
        return result;
    } catch (std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return std::nullopt;
    }
}


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
            output_array[prefix_sum[idx]] = input_array[idx];
        }
        else {
            int position = idx - prefix_sum[idx];
            output_array[total_sum + position] = input_array[idx];
        }
    }
}

void radixSortGPU(vector<int> &input_array, int defined_bits) {
    int size = input_array.size();
    int padded_size =  ((size + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    int num_block = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    thrust::device_vector<int> device_input(input_array.begin(), input_array.end());
    device_input.resize(padded_size, 0);
    thrust::device_vector<int> array_flags(padded_size);
    thrust::device_vector<int> array_prefix(padded_size);
    thrust::device_vector<int> device_output(padded_size);

    for (int i = 0; i< defined_bits; i++) {

        calculate_flags<<<num_block, BLOCK_SIZE>>>( raw_pointer_cast(device_input.data()),  raw_pointer_cast(array_flags.data()), padded_size,i);

        calculatePrefix(array_flags, array_prefix);

        int total_sum = array_prefix[padded_size - 1] + array_flags[padded_size - 1];
        seperate_values<<<num_block, BLOCK_SIZE>>>(raw_pointer_cast(device_input.data()),
            thrust::raw_pointer_cast(array_flags.data()),
            thrust::raw_pointer_cast(array_prefix.data()),
            thrust::raw_pointer_cast(device_output.data()),
            padded_size,
            total_sum);
        thrust::copy(device_output.begin(), device_output.end(), device_input.begin());

    }

    thrust::copy(device_input.begin(), device_input.begin() + size, input_array.begin());
}

