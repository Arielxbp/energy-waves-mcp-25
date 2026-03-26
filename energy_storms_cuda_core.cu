#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "energy_storms.h"

/* -----------------------------------------------------------------------
 * update() — identical to original, forced inline
 * ----------------------------------------------------------------------- */
__device__
float update(int affected_position, int layer_size,
             float energy, int contact_position) {
    int   distance   = abs(contact_position - affected_position) + 1;
    float atenuacion = sqrtf((float)distance);
    float energy_k   = energy / layer_size / atenuacion;
    float threshold  = THRESHOLD / layer_size;
    return (fabsf(energy_k) >= threshold) ? energy_k : 0.0f;
}

/* -----------------------------------------------------------------------
 * bomb
 *   Loop structure is identical to the original so the floating-point
 *   summation order is preserved and results match the sequential reference.
 *
 *   The only change: posval reads use __ldg() to route through the
 *   read-only (texture) cache path.  All threads in a warp read the same
 *   particle index j at the same time, so a single cache line covers the
 *   whole warp — effectively the same bandwidth benefit as shared memory
 *   tiling, with zero __syncthreads() overhead and no FP-order change.
 * ----------------------------------------------------------------------- */
__global__ void bomb(      float* __restrict__ layer_d,
                     int                       layer_size,
                     const int*   __restrict__ posval_d,
                     int                       storm_size) {

    const int grid_stride = blockDim.x * gridDim.x;
    const int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = thread_id; i < layer_size; i += grid_stride) {
        for (int j = 0; j < storm_size; j++) {
            /* __ldg: read-only cache; uniform access across the warp
               means one cache line serves all 32 threads per load.    */
            float energy = (float)__ldg(&posval_d[j * 2 + 1]) * 1000.0f;
            int   pos    = __ldg(&posval_d[j * 2]);
            layer_d[i]  += update(i, layer_size, energy, pos);
        }
    }
}

/* -----------------------------------------------------------------------
 * relax
 *   Bug fix from original: original started at thread_id=0, causing
 *   position 0 to read layer_copy_d[-1] (out-of-bounds).
 *   Sequential reference skips k=0 and k=layer_size-1, so we do the same:
 *   start at thread_id+1, stop before layer_size-1.
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
 *   Replaces the original cudaMemcpy of the full layer to the host.
 *   For test_08 (100 M positions) that copy is ~400 MB per storm — the
 *   dominant cost.  This kernel does the reduction entirely on the GPU
 *   and writes one (value, position) pair per block; only that small
 *   array (~KB) is transferred to the host.
 *
 *   Dynamic shared memory layout:
 *     [0 .. blockDim.x-1]            float  — per-thread max values
 *     [blockDim.x .. 2*blockDim.x-1] int    — corresponding positions
 * ----------------------------------------------------------------------- */
__global__ void find_max(const float* __restrict__ layer_d,
                         int                        layer_size,
                         float*       __restrict__  block_max,
                         int*         __restrict__  block_pos) {

    extern __shared__ float sdata[];           /* blockDim.x floats */
    int* spos = (int*)(sdata + blockDim.x);    /* blockDim.x ints   */

    const int tid         = threadIdx.x;
    const int grid_stride = blockDim.x * gridDim.x;
    const int thread_id   = blockIdx.x * blockDim.x + tid;

    float lmax = -FLT_MAX;
    int   lpos = -1;

    /* Each thread scans its stripe, keeping only the best local maximum */
    for (int i = thread_id + 1; i < layer_size - 1; i += grid_stride) {
        if (layer_d[i] > layer_d[i-1] && layer_d[i] > layer_d[i+1]) {
            if (layer_d[i] > lmax) { lmax = layer_d[i]; lpos = i; }
        }
    }

    sdata[tid] = lmax;
    spos[tid]  = lpos;
    __syncthreads();

    /* Standard tree reduction — keep the larger value at each step */
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

    /* ---- Find the largest storm so posval_d can be allocated once ---- */
    int max_storm_size = 0;
    for (int i = 0; i < num_storms; i++)
        if (storms[i].size > max_storm_size) max_storm_size = storms[i].size;

    /* ---- Allocate all device buffers ONCE outside the storm loop ----
       Original called cudaMalloc/cudaFree inside the loop — that
       forces a device sync on every iteration and is O(num_storms) more
       expensive than necessary.                                          */
    float *layer_d, *layer_copy_d;
    cudaMalloc((void **)&layer_d,      sizeof(float) * layer_size);
    cudaMalloc((void **)&layer_copy_d, sizeof(float) * layer_size);
    cudaMemset(layer_d, 0, sizeof(float) * layer_size);

    int *posval_d;
    cudaMalloc((void **)&posval_d, sizeof(int) * max_storm_size * 2);

    float *block_max_d;
    int   *block_pos_d;
    cudaMalloc((void **)&block_max_d, sizeof(float) * nblocks);
    cudaMalloc((void **)&block_pos_d, sizeof(int)   * nblocks);

    float *block_max_h = (float *)malloc(sizeof(float) * nblocks);
    int   *block_pos_h = (int   *)malloc(sizeof(int)   * nblocks);

    const size_t smem_max = BLOCK * (sizeof(float) + sizeof(int));

    for (int i = 0; i < num_storms; i++) {

        /* 4.1. Upload this storm's particles and run bomb */
        cudaMemcpy(posval_d, storms[i].posval,
                   sizeof(int) * storms[i].size * 2, cudaMemcpyHostToDevice);

        bomb<<<gridDim, blockDim>>>(layer_d, layer_size, posval_d, storms[i].size);

        /* 4.2. Relaxation: snapshot then smooth */
        cudaMemcpy(layer_copy_d, layer_d,
                   sizeof(float) * layer_size, cudaMemcpyDeviceToDevice);

        relax<<<gridDim, blockDim>>>(layer_d, layer_copy_d, layer_size);

        /* 4.3. GPU reduction — only nblocks small results cross the bus */
        find_max<<<gridDim, blockDim, smem_max>>>(layer_d, layer_size,
                                                   block_max_d, block_pos_d);

        cudaMemcpy(block_max_h, block_max_d, sizeof(float) * nblocks,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(block_pos_h, block_pos_d, sizeof(int)   * nblocks,
                   cudaMemcpyDeviceToHost);

        /* Final host-side reduction over the per-block results */
        float gmax = -FLT_MAX;
        int   gpos = -1;
        for (int b = 0; b < nblocks; b++) {
            if (block_max_h[b] > gmax) {
                gmax = block_max_h[b];
                gpos = block_pos_h[b];
            }
        }
        /* Only update maximum[i] if a valid (positive) local max was found,
           consistent with the sequential reference initialising to 0.       */
        if (gmax > maximum[i]) {
            maximum[i]   = gmax;
            positions[i] = gpos;
        }
    }

    /* ---- Free everything ---- */
    cudaFree(layer_d);
    cudaFree(layer_copy_d);
    cudaFree(posval_d);
    cudaFree(block_max_d);
    cudaFree(block_pos_d);
    free(block_max_h);
    free(block_pos_h);
}
