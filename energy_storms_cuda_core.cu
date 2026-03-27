#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "energy_storms.h"

/* THIS FUNCTION CAN BE MODIFIED */
/* Function to update a single position of the layer */
__device__ __forceinline__ float update(int affected_position, int layer_size, float energy, int contact_position) {

    /* 1. Compute the absolute value of the distance between the
        impact position and the k-th position of the layer */
    int distance = abs(contact_position - affected_position) + 1;

    /* 2. Square root of the distance */
    /* NOTE: Real world atenuation typically depends on the square of the distance.
       We use here a tailored equation that affects a much wider range of cells */
    float atenuacion = sqrtf((float)distance);

    /* 3. Compute attenuated energy */
    float energy_k = energy / layer_size / atenuacion;

    /* 4. Do not add if its absolute value is lower than the threshold */
    float threshold = THRESHOLD / layer_size;
    return (fabsf(energy_k) >= threshold) ? energy_k : 0.0f; // float abs()
}

__global__ void bomb(float* __restrict__ layer_d, int layer_size, const int* __restrict__ posval_d, int storm_size) {

    // Grid-stride loop for handling storm > threads
    const int grid_stride = blockDim.x * gridDim.x;
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x & 31;

    // Each thread processes multiple contact points of the layer in a grid-stride loop
    for (int i = thread_id; i < layer_size; i += grid_stride) {

        // For each contact point i, compute all storm impacts and update the contact point value if affected by it
        for (int j = 0; j < storm_size; j++) {
            int contact_position = 0;
            float energy = 0.0f;
            if (lane == 0) {
                contact_position = __ldg(&posval_d[j * 2]);
                energy = (float)__ldg(&posval_d[j * 2 + 1]) * 1000.0f;
            }
            contact_position = __shfl_sync(0xffffffff, contact_position, 0);
            energy = __shfl_sync(0xffffffff, energy, 0);
            layer_d[i] += update(i, layer_size, energy, contact_position);
        }
    }
}

__global__ void relax(float* __restrict__ layer_out_d, const float* __restrict__ layer_in_d, int layer_size) {

    // Grid-stride loop for handling layers > threads
    const int grid_stride = blockDim.x * gridDim.x;
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id == 0) {
        layer_out_d[0] = layer_in_d[0];
        layer_out_d[layer_size - 1] = layer_in_d[layer_size - 1];
    }

    // Each thread processes multiple contact points of the layer in a grid-stride loop
    for (int i = thread_id + 1; i < layer_size - 1; i += grid_stride) {
        layer_out_d[i] = (layer_in_d[i-1] + layer_in_d[i] + layer_in_d[i+1]) / 3.0f;
    }
}

__global__ void find_max(const float* __restrict__ layer_d, int layer_size, float* __restrict__ block_max, int* __restrict__ block_pos) {

    extern __shared__ float sdata[];
    int* spos = (int*)(sdata + blockDim.x);

    const int tid = threadIdx.x;
    const int grid_stride = blockDim.x * gridDim.x;
    const int thread_id = blockIdx.x * blockDim.x + tid;

    float lmax = -FLT_MAX; // Negative float max value
    int lpos = -1;

    // Each thread scans its stripe, keeping only the best local maximum
    for (int i = thread_id + 1; i < layer_size - 1; i += grid_stride) {
        if (layer_d[i] > layer_d[i-1] && layer_d[i] > layer_d[i+1]) {
            if (layer_d[i] > lmax) { lmax = layer_d[i]; lpos = i; }
        }
    }

    sdata[tid] = lmax;
    spos[tid] = lpos;
    __syncthreads();

    // Standard tree reduction — keep the larger value at each step
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s && sdata[tid + s] > sdata[tid]) {
            sdata[tid] = sdata[tid + s];
            spos[tid] = spos[tid + s];
        }
        __syncthreads();
    }

    // Write the block's best result to global memory
    if (tid == 0) {
        block_max[blockIdx.x] = sdata[0];
        block_pos[blockIdx.x] = spos[0];
    }
}

void core(int layer_size, int num_storms, Storm *storms, float *maximum, int *positions) {

    /* 3.1. Obtain grid and block dimensions */
    const int BOMB_BLOCK = 512;
    const int REDUCE_BLOCK = 256;
    const int bomb_nblocks = (layer_size + BOMB_BLOCK - 1) / BOMB_BLOCK;
    const int reduce_nblocks = (layer_size + REDUCE_BLOCK - 1) / REDUCE_BLOCK;
    dim3 bombBlockDim(BOMB_BLOCK);
    dim3 bombGridDim(bomb_nblocks);
    dim3 reduceBlockDim(REDUCE_BLOCK);
    dim3 reduceGridDim(reduce_nblocks);

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

    /* 3.2.2 Allocate memory for the block max and positions */
    float *block_max_d;
    int *block_pos_d;
    cudaMalloc((void **)&block_max_d, sizeof(float) * reduce_nblocks);
    cudaMalloc((void **)&block_pos_d, sizeof(int) * reduce_nblocks);

    float *block_max_h = (float *)malloc(sizeof(float) * reduce_nblocks);
    int *block_pos_h = (int *)malloc(sizeof(int) * reduce_nblocks);

    const size_t smem_max = REDUCE_BLOCK * (sizeof(float) + sizeof(int));
    float *layer_curr_d = layer_a_d;
    float *layer_next_d = layer_b_d;

    /* 4 Simulate for every storm */
    for (int i = 0; i < num_storms; i++) {

        /* 4.1. Transfer the i-th storm's particles */
        cudaMemcpy(posval_d, storms[i].posval, sizeof(int) * storms[i].size * 2, cudaMemcpyHostToDevice);

        /* 4.2. Simulate impacts energy to layer cells */
        bomb<<<bombGridDim, bombBlockDim>>>(layer_curr_d, layer_size, posval_d, storms[i].size);
        
        /* 4.4. Simulate energy relaxation */
        relax<<<reduceGridDim, reduceBlockDim>>>(layer_next_d, layer_curr_d, layer_size);

        float *tmp = layer_curr_d;
        layer_curr_d = layer_next_d;
        layer_next_d = tmp;

        /* 4.5 Find maximum energy and its position */
        find_max<<<reduceGridDim, reduceBlockDim, smem_max>>>(layer_curr_d, layer_size, block_max_d, block_pos_d);

        /* 4.6 Transfer results back to host for final reduction */
        cudaMemcpy(block_max_h, block_max_d, sizeof(float) * reduce_nblocks, cudaMemcpyDeviceToHost);
        cudaMemcpy(block_pos_h, block_pos_d, sizeof(int) * reduce_nblocks, cudaMemcpyDeviceToHost);

        /* 4.7 Host reduction over the per-block results */
        float global_max = -FLT_MAX; // Negative float max value
        int global_pos = -1;
        for (int b = 0; b < reduce_nblocks; b++) {
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
    cudaFree(block_max_d);
    cudaFree(block_pos_d);
    free(block_max_h);
    free(block_pos_h);
}