#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "energy_storms.h"

#include <cuda.h>
#include <cuda_runtime.h>

/* Maximum number of particles that fit in constant memory.
   Budget: 64 KB total, split evenly between pos and energy.
   8192 ints × 4 bytes + 8192 floats × 4 bytes = 65536 bytes exactly. */
#define MAX_CONST_PARTICLES 8192

/* Constant memory arrays — broadcast-optimal for the bomb inner loop
   where every thread in a warp reads the same index j each iteration. */
__constant__ int   pos_c[MAX_CONST_PARTICLES];
__constant__ float energy_c[MAX_CONST_PARTICLES];

/* ------------------------------------------------------------------ */
/* Device helper                                                        */
/* ------------------------------------------------------------------ */

__device__ __forceinline__ float update(int affected_position, int layer_size, float energy, int contact_position) {
    int distance = abs(contact_position - affected_position) + 1;
    float atenuacion = sqrtf((float)distance);
    float energy_k = energy / layer_size / atenuacion;
    float threshold = THRESHOLD / layer_size;
    return (fabsf(energy_k) >= threshold) ? energy_k : 0.0f;
}

/* ------------------------------------------------------------------ */
/* Kernels                                                              */
/* ------------------------------------------------------------------ */

/* bomb_const: reads particles from constant memory.
   Called once per chunk of up to MAX_CONST_PARTICLES particles.
   All threads in a warp read the same pos_c[j] / energy_c[j] each
   iteration, so every access is served by the constant cache broadcast
   unit without touching L1/L2 — leaving those caches free for
   layer_d read-modify-writes. */
__global__ void bomb_const(float* __restrict__ layer_d, int layer_size, int chunk_size) {
    const int grid_stride = blockDim.x * gridDim.x;
    const int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = thread_id; i < layer_size; i += grid_stride) {
        float cell_val = layer_d[i];
        for (int j = 0; j < chunk_size; j++) {
            /* No __ldg needed: constant cache handles broadcast natively
               and is a physically separate unit from the L1 read-only cache
               that __ldg uses. */
            cell_val += update(i, layer_size, energy_c[j], pos_c[j]);
        }
        layer_d[i] = cell_val;
    }
}

__global__ void relax_and_find_max(float* __restrict__ layer_out_d, const float* __restrict__ layer_in_d, int layer_size, float* __restrict__ block_max, int* __restrict__ block_pos) {

    extern __shared__ float sdata[];
    int* spos = (int*)(sdata + blockDim.x);

    const int tid         = threadIdx.x;
    const int grid_stride = blockDim.x * gridDim.x;
    const int thread_id   = blockIdx.x * blockDim.x + tid;

    if (thread_id == 0) {
        layer_out_d[0]             = layer_in_d[0];
        layer_out_d[layer_size-1]  = layer_in_d[layer_size-1];
    }

    float lmax = -FLT_MAX;
    int   lpos = -1;

    for (int i = thread_id + 1; i < layer_size - 1; i += grid_stride) {
        const float center = (layer_in_d[i-1] + layer_in_d[i] + layer_in_d[i+1]) / 3.0f;
        layer_out_d[i] = center;

        float left_neighbor  = (i == 1)
            ? layer_in_d[0]
            : (layer_in_d[i-2] + layer_in_d[i-1] + layer_in_d[i]) / 3.0f;

        float right_neighbor = (i == layer_size - 2)
            ? layer_in_d[layer_size-1]
            : (layer_in_d[i] + layer_in_d[i+1] + layer_in_d[i+2]) / 3.0f;

        if (center > left_neighbor && center > right_neighbor && center > lmax) {
            lmax = center;
            lpos = i;
        }
    }

    sdata[tid] = lmax;
    spos[tid]  = lpos;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s && sdata[tid + s] > sdata[tid]) {
            sdata[tid] = sdata[tid + s];
            spos[tid]  = spos[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_max[blockIdx.x] = sdata[0];
        block_pos[blockIdx.x] = spos[0];
    }
}

/* ------------------------------------------------------------------ */
/* Core                                                                 */
/* ------------------------------------------------------------------ */

void core(int layer_size, int num_storms, Storm *storms, float *maximum, int *positions) {

    const int BLOCK   = 256;
    const int nblocks = (layer_size + BLOCK - 1) / BLOCK;
    dim3 blockDim(BLOCK);
    dim3 gridDim(nblocks);

    /* Find the largest storm for staging buffer allocation */
    int max_storm_size = 0;
    for (int i = 0; i < num_storms; i++) {
        if (storms[i].size > max_storm_size)
            max_storm_size = storms[i].size;
    }

    /* Layer ping-pong buffers */
    float *layer_a_d, *layer_b_d;
    cudaMalloc((void **)&layer_a_d, sizeof(float) * layer_size);
    cudaMalloc((void **)&layer_b_d, sizeof(float) * layer_size);
    cudaMemset(layer_a_d, 0, sizeof(float) * layer_size);
    cudaMemset(layer_b_d, 0, sizeof(float) * layer_size);

    /* Pinned host SoA staging buffers — page-locked so the DMA engine can
       transfer each chunk without CPU involvement. cudaMemcpyToSymbol
       also benefits from a page-locked source. */
    int   *pos_h;
    float *energy_h;
    cudaMallocHost((void **)&pos_h,    sizeof(int)   * max_storm_size);
    cudaMallocHost((void **)&energy_h, sizeof(float) * max_storm_size);

    /* Per-block reduction buffers */
    float *block_max_d;
    int   *block_pos_d;
    cudaMalloc((void **)&block_max_d, sizeof(float) * nblocks);
    cudaMalloc((void **)&block_pos_d, sizeof(int)   * nblocks);

    float *block_max_h = (float *)malloc(sizeof(float) * nblocks);
    int   *block_pos_h = (int   *)malloc(sizeof(int)   * nblocks);

    const size_t smem_max = BLOCK * (sizeof(float) + sizeof(int));
    float *layer_curr_d = layer_a_d;
    float *layer_next_d = layer_b_d;

    for (int i = 0; i < num_storms; i++) {
        const int ssize = storms[i].size;

        /* Split interleaved AoS posval into SoA staging arrays.
           Runs in CPU cache, negligible cost. */
        for (int j = 0; j < ssize; j++) {
            pos_h[j]    = storms[i].posval[j * 2];
            energy_h[j] = (float)storms[i].posval[j * 2 + 1] * 1000.0f;
        }

        /* Tile the storm into chunks of MAX_CONST_PARTICLES.
           Each chunk is uploaded to constant memory and processed in one
           bomb_const launch. Contributions are additive and particle-order
           independent, so the final layer_d is identical to a single-pass
           kernel reading all particles from global memory.

           For storms that already fit in one chunk (ssize <= MAX_CONST_PARTICLES)
           this loop executes exactly once — zero overhead vs a non-chunked path. */
        for (int offset = 0; offset < ssize; offset += MAX_CONST_PARTICLES) {
            const int chunk = min(MAX_CONST_PARTICLES, ssize - offset);

            /* Upload this chunk to constant memory.
               cudaMemcpyToSymbol is synchronous by default: the next kernel
               launch is guaranteed to see the updated values. */
            cudaMemcpyToSymbol(pos_c,    pos_h    + offset, sizeof(int)   * chunk);
            cudaMemcpyToSymbol(energy_c, energy_h + offset, sizeof(float) * chunk);

            bomb_const<<<gridDim, blockDim>>>(layer_curr_d, layer_size, chunk);
        }

        /* Relax and collect per-block maxima */
        relax_and_find_max<<<gridDim, blockDim, smem_max>>>(layer_next_d, layer_curr_d, layer_size, block_max_d, block_pos_d);

        float *tmp   = layer_curr_d;
        layer_curr_d = layer_next_d;
        layer_next_d = tmp;

        cudaMemcpy(block_max_h, block_max_d, sizeof(float) * nblocks, cudaMemcpyDeviceToHost);
        cudaMemcpy(block_pos_h, block_pos_d, sizeof(int)   * nblocks, cudaMemcpyDeviceToHost);

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

    cudaFree(layer_a_d);
    cudaFree(layer_b_d);
    cudaFree(block_max_d);
    cudaFree(block_pos_d);
    cudaFreeHost(pos_h);
    cudaFreeHost(energy_h);
    free(block_max_h);
    free(block_pos_h);
}
