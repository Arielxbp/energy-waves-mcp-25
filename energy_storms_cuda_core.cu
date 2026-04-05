#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "energy_storms.h"

/* -----------------------------------------------------------------------
 * update()
 *
 * Arithmetic is IDENTICAL to the sequential reference:
 *   energy / layer_size / sqrtf(distance)
 * This produces bit-exact results because the two IEEE-754 division
 * steps happen in the same order as the CPU code.
 *
 * Why not use the inv_sqrt lookup table?
 *   1.0f/sqrtf(d) introduces an extra rounding step that doesn't exist
 *   in the original chain, shifting the last 1-2 bits of energy_k in
 *   some cells and causing the final maxima to differ slightly.
 *
 * Why not use rsqrtf()?
 *   rsqrtf() is only accurate to ~1.5 ULP (hardware approximation),
 *   which produces the same kind of drift.
 *
 * --fmad=true won't fuse divisions, so there is no FP-order side-effect
 * from that flag on this expression.
 * ----------------------------------------------------------------------- */
__device__ __forceinline__
float update(int   affected_position,
             int   layer_size,
             float energy,
             int   contact_position) {

    int   distance   = abs(contact_position - affected_position) + 1;
    float atenuacion = sqrtf((float)distance);
    float energy_k   = energy / layer_size / atenuacion;
    float threshold  = THRESHOLD / layer_size;
    return (fabsf(energy_k) >= threshold) ? energy_k : 0.0f;
}

/* -----------------------------------------------------------------------
 * bomb
 *   __ldg for posval_d: all 32 threads in a warp read the same index j
 *   simultaneously, so one read-only cache line serves the whole warp.
 * ----------------------------------------------------------------------- */
__global__ void bomb(      float* __restrict__ layer_d,
                     int                       layer_size,
                     const int*   __restrict__ posval_d,
                     int                       storm_size) {

    const int grid_stride = blockDim.x * gridDim.x;
    const int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = thread_id; i < layer_size; i += grid_stride) {
        for (int j = 0; j < storm_size; j++) {
            float energy = (float)__ldg(&posval_d[j * 2 + 1]) * 1000.0f;
            int   pos    = __ldg(&posval_d[j * 2]);
            layer_d[i]  += update(i, layer_size, energy, pos);
        }
    }
}

/* -----------------------------------------------------------------------
 * relax
 *   Starts at index 1 and stops before layer_size-1, matching the
 *   sequential reference exactly.
 * ----------------------------------------------------------------------- */
__global__ void relax(      float* __restrict__ layer_d,
                      const float* __restrict__ layer_copy_d,
                      int                       layer_size) {

    const int grid_stride = blockDim.x * gridDim.x;
    const int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = thread_id + 1; i < layer_size - 1; i += grid_stride) {
        layer_d[i] = (layer_copy_d[i-1] + layer_copy_d[i] + layer_copy_d[i+1]) / 3.0f;
    }
}

/* -----------------------------------------------------------------------
 * find_max
 *   Full GPU reduction: only nblocks (float, int) pairs cross PCIe,
 *   not the entire layer.  Critical for test_08 (100 M positions).
 * ----------------------------------------------------------------------- */
__global__ void find_max(const float* __restrict__ layer_d,
                         int                        layer_size,
                         float*       __restrict__  block_max,
                         int*         __restrict__  block_pos) {

    extern __shared__ float sdata[];
    int* spos = (int*)(sdata + blockDim.x);

    const int tid         = threadIdx.x;
    const int grid_stride = blockDim.x * gridDim.x;
    const int thread_id   = blockIdx.x * blockDim.x + tid;

    float lmax = -FLT_MAX;
    int   lpos = -1;

    for (int i = thread_id + 1; i < layer_size - 1; i += grid_stride) {
        if (layer_d[i] > layer_d[i-1] && layer_d[i] > layer_d[i+1]) {
            if (layer_d[i] > lmax) { lmax = layer_d[i]; lpos = i; }
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

/* -----------------------------------------------------------------------
 * core
 * ----------------------------------------------------------------------- */
void core(int layer_size, int num_storms, Storm *storms,
          float *maximum, int *positions) {

    const int BLOCK   = 256;
    const int nblocks = (layer_size + BLOCK - 1) / BLOCK;
    dim3 blockDim(BLOCK);
    dim3 gridDim(nblocks);

    /* ---- Largest storm size (posval_d capacity) ----------------------- */
    int max_storm_size = 0;
    for (int i = 0; i < num_storms; i++)
        if (storms[i].size > max_storm_size) max_storm_size = storms[i].size;

    /* ---- All device allocations outside the storm loop ---------------- */
    float *layer_d, *layer_copy_d;
    cudaMalloc((void **)&layer_d,      sizeof(float) * layer_size);
    cudaMalloc((void **)&layer_copy_d, sizeof(float) * layer_size);
    cudaMemset(layer_d, 0, sizeof(float) * layer_size);

    int *posval_d;
    cudaMalloc((void **)&posval_d, sizeof(int) * max_storm_size * 2);

    float *block_max_d;  int *block_pos_d;
    cudaMalloc((void **)&block_max_d, sizeof(float) * nblocks);
    cudaMalloc((void **)&block_pos_d, sizeof(int)   * nblocks);

    float *block_max_h = (float *)malloc(sizeof(float) * nblocks);
    int   *block_pos_h = (int   *)malloc(sizeof(int)   * nblocks);

    const size_t smem_max = BLOCK * (sizeof(float) + sizeof(int));

    for (int i = 0; i < num_storms; i++) {

        cudaMemcpy(posval_d, storms[i].posval,
                   sizeof(int) * storms[i].size * 2, cudaMemcpyHostToDevice);

        bomb<<<gridDim, blockDim>>>(layer_d, layer_size,
                                    posval_d, storms[i].size);

        cudaMemcpy(layer_copy_d, layer_d,
                   sizeof(float) * layer_size, cudaMemcpyDeviceToDevice);

        relax<<<gridDim, blockDim>>>(layer_d, layer_copy_d, layer_size);

        find_max<<<gridDim, blockDim, smem_max>>>(layer_d, layer_size,
                                                   block_max_d, block_pos_d);

        cudaMemcpy(block_max_h, block_max_d, sizeof(float) * nblocks,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(block_pos_h, block_pos_d, sizeof(int)   * nblocks,
                   cudaMemcpyDeviceToHost);

        float gmax = -FLT_MAX;
        int   gpos = -1;
        for (int b = 0; b < nblocks; b++) {
            if (block_max_h[b] > gmax) { gmax = block_max_h[b]; gpos = block_pos_h[b]; }
        }
        if (gmax > maximum[i]) {
            maximum[i]   = gmax;
            positions[i] = gpos;
        }
    }

    cudaFree(layer_d);
    cudaFree(layer_copy_d);
    cudaFree(posval_d);
    cudaFree(block_max_d);
    cudaFree(block_pos_d);
    free(block_max_h);
    free(block_pos_h);
}
