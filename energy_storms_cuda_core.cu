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
    return (fabsf(energy_k) >= threshold) ? energy_k : 0.0f;
}

/* prepare_storm has been removed: the host now uploads pos and energy as
   separate SoA arrays, so no GPU preprocessing kernel is needed. */

__global__ void bomb(float* __restrict__ layer_d, int layer_size, const int* __restrict__ pos_d, const float* __restrict__ energy_d, int storm_size) {

    // Grid-stride loop for handling layer_size > total threads
    const int grid_stride = blockDim.x * gridDim.x;
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread processes one or more layer cells in a grid-stride loop.
    // The inner loop over j is a broadcast: all 32 threads in a warp read
    // the same pos_d[j] / energy_d[j] each iteration, which the hardware
    // satisfies in a single L1 transaction regardless of layout.
    for (int i = thread_id; i < layer_size; i += grid_stride) {
        for (int j = 0; j < storm_size; j++) {
            float energy        = __ldg(&energy_d[j]);
            int contact_position = __ldg(&pos_d[j]);
            layer_d[i] += update(i, layer_size, energy, contact_position);
        }
    }
}

__global__ void relax_and_find_max(float* __restrict__ layer_out_d, const float* __restrict__ layer_in_d, int layer_size, float* __restrict__ block_max, int* __restrict__ block_pos) {

    extern __shared__ float sdata[];
    int* spos = (int*)(sdata + blockDim.x);

    const int tid = threadIdx.x;
    const int grid_stride = blockDim.x * gridDim.x;
    const int thread_id = blockIdx.x * blockDim.x + tid;

    if (thread_id == 0) {
        layer_out_d[0] = layer_in_d[0];
        layer_out_d[layer_size - 1] = layer_in_d[layer_size - 1];
    }

    float lmax = -FLT_MAX;
    int lpos = -1;

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
            if (center > lmax) { lmax = center; lpos = i; }
        }
    }

    sdata[tid] = lmax;
    spos[tid] = lpos;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s && sdata[tid + s] > sdata[tid]) {
            sdata[tid] = sdata[tid + s];
            spos[tid] = spos[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_max[blockIdx.x] = sdata[0];
        block_pos[blockIdx.x] = spos[0];
    }
}

void core(int layer_size, int num_storms, Storm *storms, float *maximum, int *positions) {

    /* 3.1. Obtain grid and block dimensions */
    const int BLOCK = 256;
    const int nblocks = (layer_size + BLOCK - 1) / BLOCK;
    dim3 blockDim(BLOCK);
    dim3 gridDim(nblocks);

    /* 3.1.1. Find the largest storm so device buffers can be allocated once */
    int max_storm_size = 0;
    for (int i = 0; i < num_storms; i++) {
        if (storms[i].size > max_storm_size)
            max_storm_size = storms[i].size;
    }

    /* 3.2. Allocate layer ping-pong buffers on the device */
    float *layer_a_d, *layer_b_d;
    cudaMalloc((void **)&layer_a_d, sizeof(float) * layer_size);
    cudaMalloc((void **)&layer_b_d, sizeof(float) * layer_size);
    cudaMemset(layer_a_d, 0, sizeof(float) * layer_size);
    cudaMemset(layer_b_d, 0, sizeof(float) * layer_size);

    /* 3.3. Allocate SoA device buffers for storm particles.
            pos_d   : contact positions  (int)
            energy_d: pre-scaled energies (float)
       These are sized once for the largest storm and reused every iteration.
       No posval_d (interleaved AoS) buffer is needed anymore. */
    int   *pos_d;
    float *energy_d;
    cudaMalloc((void **)&pos_d,    sizeof(int)   * max_storm_size);
    cudaMalloc((void **)&energy_d, sizeof(float) * max_storm_size);

    /* 3.4. Pinned host staging buffers for the SoA upload.
            Using page-locked memory lets the DMA engine transfer both arrays
            without stalling the CPU, and overlaps with prior-storm GPU work. */
    int   *pos_h;
    float *energy_h;
    cudaMallocHost((void **)&pos_h,    sizeof(int)   * max_storm_size);
    cudaMallocHost((void **)&energy_h, sizeof(float) * max_storm_size);

    /* 3.5. Allocate per-block reduction buffers */
    float *block_max_d;
    int   *block_pos_d;
    cudaMalloc((void **)&block_max_d, sizeof(float) * nblocks);
    cudaMalloc((void **)&block_pos_d, sizeof(int)   * nblocks);

    float *block_max_h = (float *)malloc(sizeof(float) * nblocks);
    int   *block_pos_h = (int   *)malloc(sizeof(int)   * nblocks);

    const size_t smem_max = BLOCK * (sizeof(float) + sizeof(int));
    float *layer_curr_d = layer_a_d;
    float *layer_next_d = layer_b_d;

    /* 4. Simulate for every storm */
    for (int i = 0; i < num_storms; i++) {
        const int ssize = storms[i].size;

        /* 4.1. Split the interleaved AoS posval into two SoA arrays on the host.
                This loop runs in CPU L1/L2 cache and is negligible compared to
                the GPU work that follows.  The coalesced upload below then lets
                the DMA engine issue a single wide burst for each array, whereas
                the old strided posval upload needed 2× the transactions. */
        for (int j = 0; j < ssize; j++) {
            pos_h[j]    = storms[i].posval[j * 2];
            energy_h[j] = (float)storms[i].posval[j * 2 + 1] * 1000.0f;
        }

        /* 4.2. Upload the two compact SoA arrays — one contiguous transfer each */
        cudaMemcpy(pos_d,    pos_h,    sizeof(int)   * ssize, cudaMemcpyHostToDevice);
        cudaMemcpy(energy_d, energy_h, sizeof(float) * ssize, cudaMemcpyHostToDevice);

        /* 4.3. Simulate impacts (no prepare_storm kernel needed) */
        bomb<<<gridDim, blockDim>>>(layer_curr_d, layer_size, pos_d, energy_d, ssize);

        /* 4.4. Relax energy and collect per-block maxima */
        relax_and_find_max<<<gridDim, blockDim, smem_max>>>(layer_next_d, layer_curr_d, layer_size, block_max_d, block_pos_d);

        float *tmp  = layer_curr_d;
        layer_curr_d = layer_next_d;
        layer_next_d = tmp;

        /* 4.5. Transfer per-block results back to host */
        cudaMemcpy(block_max_h, block_max_d, sizeof(float) * nblocks, cudaMemcpyDeviceToHost);
        cudaMemcpy(block_pos_h, block_pos_d, sizeof(int)   * nblocks, cudaMemcpyDeviceToHost);

        /* 4.6. Host reduction over per-block results */
        float global_max = -FLT_MAX;
        int   global_pos = -1;
        for (int b = 0; b < nblocks; b++) {
            if (block_max_h[b] > global_max) {
                global_max = block_max_h[b];
                global_pos = block_pos_h[b];
            }
        }
        if (global_max > maximum[i]) {
            maximum[i]   = global_max;
            positions[i] = global_pos;
        }
    }

    /* 5. Free all resources */
    cudaFree(layer_a_d);
    cudaFree(layer_b_d);
    cudaFree(pos_d);
    cudaFree(energy_d);
    cudaFree(block_max_d);
    cudaFree(block_pos_d);
    cudaFreeHost(pos_h);
    cudaFreeHost(energy_h);
    free(block_max_h);
    free(block_pos_h);
}
