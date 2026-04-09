#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "energy_storms.h"


__global__ void prepare_storm(const int* __restrict__ posval_d, int* __restrict__ pos_d, float* __restrict__ energy_d, int storm_size) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < storm_size) {
        pos_d[idx] = __ldg(&posval_d[idx * 2]);
        energy_d[idx] = (float)__ldg(&posval_d[idx * 2 + 1]) * 1000.0f;
    }
}

__global__ void bomb(float* __restrict__ layer_d, int layer_size, const int* __restrict__ pos_d, const float* __restrict__ energy_d, int storm_size) {

    // Grid-stride loop for handling storm > threads
    const int grid_stride = blockDim.x * gridDim.x;
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    const float threshold = THRESHOLD / (float)layer_size;

    // Each thread processes multiple contact points of the layer in a grid-stride loop
    for (int i = thread_id; i < layer_size; i += grid_stride) {

        float temp = layer_d[i]; // Read current value of the layer cell

        // For each contact point i, compute all storm impacts and update the contact point value if affected by it
        for (int j = 0; j < storm_size; j++) {
            // __ldg gives uniform access across the warp so one cache line serves all 32 threads per load
            float energy = __ldg(&energy_d[j]);
            int contact_position = __ldg(&pos_d[j]);
            int distance = abs(contact_position - i) + 1;
            float res = energy / (float)layer_size / sqrtf((float)distance);

            if (fabsf(res) >= threshold) {
                temp += res;
            }
        }
        layer_d[i] = temp; // Write back the updated value
    }
}

__global__ void relax_and_find_max(float* __restrict__ layer_out_d, const float* __restrict__ layer_in_d, int layer_size, float* __restrict__ block_max, int* __restrict__ block_pos) {

    extern __shared__ float shared_data[];
    int* shared_pos = (int*)(shared_data + blockDim.x);

    const int tid = threadIdx.x;
    const int thread_id = blockIdx.x * blockDim.x + tid;
    const int grid_stride = blockDim.x * gridDim.x;

    if (thread_id == 0) {
        layer_out_d[0] = layer_in_d[0];
        layer_out_d[layer_size - 1] = layer_in_d[layer_size - 1];
    }

    float local_max = -FLT_MAX; // Negative float max value
    int local_pos = -1;

    // Each thread scans its stripe, relaxes values and keeps the best local maximum
    for (int i = thread_id + 1; i < layer_size - 1; i += grid_stride) {
        const float center = (layer_in_d[i-1] + layer_in_d[i] + layer_in_d[i+1]) / 3.0f;
        
        layer_out_d[i] = center;

        float left_neighbor;
        if (i == 1) {
            left_neighbor = layer_in_d[0];
        } else {
            left_neighbor = (layer_in_d[i-2] + layer_in_d[i-1] + layer_in_d[i]) / 3.0f;
        }

        float right_neighbor;
        if (i == layer_size - 2) {
            right_neighbor = layer_in_d[layer_size - 1];
        } else {
            right_neighbor = (layer_in_d[i] + layer_in_d[i+1] + layer_in_d[i+2]) / 3.0f;
        }

        if (center > left_neighbor && center > right_neighbor) {
            if (center > local_max) { local_max = center; local_pos = i; }
        }
    }

    shared_data[tid] = local_max;
    shared_pos[tid] = local_pos;
    __syncthreads();

    // Tree reduction which keeps the larger value at each step
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s && shared_data[tid + s] > shared_data[tid]) {
            shared_data[tid] = shared_data[tid + s];
            shared_pos[tid] = shared_pos[tid + s];
        }
        __syncthreads();
    }

    // Write the block's best result to global memory
    if (tid == 0) {
        block_max[blockIdx.x] = shared_data[0];
        block_pos[blockIdx.x] = shared_pos[0];
    }
}

void core(int layer_size, int num_storms, Storm *storms, float *maximum, int *positions) {

    /* 3.1. Obtain grid and block dimensions */
    const int BLOCK = 64; // 64 threads per block
    const int nblocks = (layer_size + BLOCK - 1) / BLOCK; // -1 to round the blocks needed
    dim3 blockDim(BLOCK);
    dim3 gridDim(nblocks);

    /* 3.1.1. Find the largest storm so posval_d can be allocated once */
    int max_storm_size = 0;
    for (int i = 0; i < num_storms; i++) {
        if (storms[i].size > max_storm_size) {
            max_storm_size = storms[i].size;
        }
    }

    /* 3.2. Allocate memory for the layer and initialize to zero */
    float *layer_a_d, *layer_b_d;
    cudaMalloc((void **)&layer_a_d, sizeof(float) * layer_size);
    cudaMalloc((void **)&layer_b_d, sizeof(float) * layer_size);
    cudaMemset(layer_a_d, 0, sizeof(float) * layer_size);
    cudaMemset(layer_b_d, 0, sizeof(float) * layer_size);

    /* 3.2.1 Allocate memory for the posval and initialize to zero */
    int *posval_d;
    cudaMalloc((void **)&posval_d, sizeof(int) * max_storm_size * 2);

    int *pos_d;
    float *energy_d;
    cudaMalloc((void **)&pos_d, sizeof(int) * max_storm_size);
    cudaMalloc((void **)&energy_d, sizeof(float) * max_storm_size);

    /* 3.2.2 Allocate memory for the block max and positions */
    float *block_max_d;
    int *block_pos_d;
    cudaMalloc((void **)&block_max_d, sizeof(float) * nblocks);
    cudaMalloc((void **)&block_pos_d, sizeof(int) * nblocks);

    float *block_max_h = (float *)malloc(sizeof(float) * nblocks);
    int *block_pos_h = (int *)malloc(sizeof(int) * nblocks);

    const size_t smem_max = BLOCK * (sizeof(float) + sizeof(int));
    float *layer_curr_d = layer_a_d;
    float *layer_next_d = layer_b_d;

    /* 4 Simulate for every storm */
    for (int i = 0; i < num_storms; i++) {

        /* 4.1. Transfer the i-th storm's particles */
        cudaMemcpy(posval_d, storms[i].posval, sizeof(int) * storms[i].size * 2, cudaMemcpyHostToDevice);

        /* 4.1.1 Prepare position and energy arrays for better access in the hot loop */
        const int storm_blocks = (storms[i].size + BLOCK - 1) / BLOCK;
        prepare_storm<<<storm_blocks, blockDim>>>(posval_d, pos_d, energy_d, storms[i].size);

        /* 4.2. Simulate impacts energy to layer cells */
        bomb<<<gridDim, blockDim>>>(layer_curr_d, layer_size, pos_d, energy_d, storms[i].size);
        
        /* 4.4 + 4.5 Simulate energy relaxation and find local maxima */
        relax_and_find_max<<<gridDim, blockDim, smem_max>>>(layer_next_d, layer_curr_d, layer_size, block_max_d, block_pos_d);

        float *temp = layer_curr_d;
        layer_curr_d = layer_next_d;
        layer_next_d = temp;

        /* 4.6 Transfer results back to host for final reduction */
        cudaMemcpy(block_max_h, block_max_d, sizeof(float) * nblocks, cudaMemcpyDeviceToHost);
        cudaMemcpy(block_pos_h, block_pos_d, sizeof(int) * nblocks, cudaMemcpyDeviceToHost);

        /* 4.7 Host reduction over the per-block results */
        float global_max = -FLT_MAX; // Negative float max value
        int global_pos = -1;
        for (int b = 0; b < nblocks; b++) {
            if (block_max_h[b] > global_max) {
                global_max = block_max_h[b];
                global_pos = block_pos_h[b];
            }
        }
        // Only update maximum[i] if a valid local max is found
        if (global_max > maximum[i]) {
            maximum[i] = global_max;
            positions[i] = global_pos;
        }
    }

    cudaFree(layer_a_d);
    cudaFree(layer_b_d);
    cudaFree(posval_d);
    cudaFree(pos_d);
    cudaFree(energy_d);
    cudaFree(block_max_d);
    cudaFree(block_pos_d);
    free(block_max_h);
    free(block_pos_h);
}
