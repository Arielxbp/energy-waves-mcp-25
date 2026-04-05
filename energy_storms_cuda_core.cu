#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "energy_storms.h"

/* -----------------------------------------------------------------------
 * update() — bit-identical to the sequential reference.
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
 * bomb — __ldg for uniform warp broadcast on particle data.
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
 * relax  (ping-pong version)
 *
 * Reads from layer_in_d, writes to layer_out_d — caller swaps pointers
 * after each storm so no explicit D2D cudaMemcpy is ever needed.
 *
 * Why the old approach was slow on the cluster
 * ---------------------------------------------
 * Previous: cudaMemcpy(layer_copy_d, layer_d, layer_size, D2D)
 *           relax(layer_d, layer_copy_d)
 * For test_08 (100 M floats) the D2D copy moves 400 MB every storm.
 * Even on a fast GPU that is ~100-200 ms of device-memory bandwidth
 * wasted per storm.  Ping-pong eliminates it with a free pointer swap.
 *
 * Boundary handling — why there is NO separate copy_boundaries kernel
 * --------------------------------------------------------------------
 * The sequential reference never updates positions 0 and layer_size-1
 * during relaxation.  With ping-pong, layer_out_d starts as the stale
 * buffer from the previous storm; those two positions would hold old
 * values unless we copy them.
 *
 * Bad fix: launch a separate kernel for 2 threads.
 *   A kernel launch costs ~5-15 µs of overhead regardless of work.
 *   For 6 storms that is up to 90 µs of pure launch tax — it was the
 *   main reason the previous version was SLOWER than what it replaced.
 *
 * Good fix: thread 0 of THIS kernel copies the two boundary values.
 *   It executes inside the existing launch, adding zero overhead.
 * ----------------------------------------------------------------------- */
__global__ void relax(      float* __restrict__ layer_out_d,
                      const float* __restrict__ layer_in_d,
                      int                       layer_size) {
    const int grid_stride = blockDim.x * gridDim.x;
    const int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;

    /* Thread 0 propagates boundary values into the output buffer.
       No synchronisation needed — these two writes are independent
       of every other thread's work.                                 */
    if (thread_id == 0) {
        layer_out_d[0]             = layer_in_d[0];
        layer_out_d[layer_size-1]  = layer_in_d[layer_size-1];
    }

    /* Interior relaxation: matches the sequential reference exactly */
    for (int i = thread_id + 1; i < layer_size - 1; i += grid_stride)
        layer_out_d[i] = (layer_in_d[i-1] + layer_in_d[i] + layer_in_d[i+1]) / 3.0f;
}

/* -----------------------------------------------------------------------
 * find_max — GPU reduction, only nblocks results cross PCIe.
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
    for (int i = thread_id + 1; i < layer_size - 1; i += grid_stride)
        if (layer_d[i] > layer_d[i-1] && layer_d[i] > layer_d[i+1])
            if (layer_d[i] > lmax) { lmax = layer_d[i]; lpos = i; }

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
    if (tid == 0) { block_max[blockIdx.x] = sdata[0]; block_pos[blockIdx.x] = spos[0]; }
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
    const size_t smem_max = BLOCK * (sizeof(float) + sizeof(int));

    /* ---- Find largest storm for posval_d capacity --------------------- */
    int max_storm_size = 0;
    for (int i = 0; i < num_storms; i++)
        if (storms[i].size > max_storm_size) max_storm_size = storms[i].size;

    /* ---- Device allocations (all outside the storm loop) -------------- */
    float *layer_a_d, *layer_b_d;
    cudaMalloc((void **)&layer_a_d, sizeof(float) * layer_size);
    cudaMalloc((void **)&layer_b_d, sizeof(float) * layer_size);
    cudaMemset(layer_a_d, 0, sizeof(float) * layer_size);
    cudaMemset(layer_b_d, 0, sizeof(float) * layer_size);

    int *posval_d;
    cudaMalloc((void **)&posval_d, sizeof(int) * max_storm_size * 2);

    float *block_max_d;  int *block_pos_d;
    cudaMalloc((void **)&block_max_d, sizeof(float) * nblocks);
    cudaMalloc((void **)&block_pos_d, sizeof(int)   * nblocks);

    /* Pinned host memory: the DMA engine transfers directly without an
       OS bounce buffer, reducing D2H latency on cluster PCIe links.    */
    float *block_max_h;  int *block_pos_h;
    cudaMallocHost((void **)&block_max_h, sizeof(float) * nblocks);
    cudaMallocHost((void **)&block_pos_h, sizeof(int)   * nblocks);

    /* Ping-pong pointers — swapped each storm, no D2D copy ever needed  */
    float *curr_d = layer_a_d;   /* bomb() accumulates here              */
    float *next_d = layer_b_d;   /* relax() writes here                  */

    /* ------------------------------------------------------------------ */
    for (int i = 0; i < num_storms; i++) {

        /* Upload this storm's particles (H2D, only storm_size * 8 bytes) */
        cudaMemcpy(posval_d, storms[i].posval,
                   sizeof(int) * storms[i].size * 2, cudaMemcpyHostToDevice);

        /* Impact: accumulate into curr_d */
        bomb<<<gridDim, blockDim>>>(curr_d, layer_size,
                                    posval_d, storms[i].size);

        /* Relax: read curr_d → write next_d (boundaries handled inside) */
        relax<<<gridDim, blockDim>>>(next_d, curr_d, layer_size);

        /* Swap: next_d (relaxed) becomes the new curr_d — zero cost     */
        float *tmp = curr_d;  curr_d = next_d;  next_d = tmp;

        /* Find maximum in the relaxed layer (now curr_d after swap)     */
        find_max<<<gridDim, blockDim, smem_max>>>(curr_d, layer_size,
                                                   block_max_d, block_pos_d);

        cudaMemcpy(block_max_h, block_max_d, sizeof(float) * nblocks,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(block_pos_h, block_pos_d, sizeof(int)   * nblocks,
                   cudaMemcpyDeviceToHost);

        float gmax = -FLT_MAX;
        int   gpos = -1;
        for (int b = 0; b < nblocks; b++)
            if (block_max_h[b] > gmax) { gmax = block_max_h[b]; gpos = block_pos_h[b]; }

        if (gmax > maximum[i]) { maximum[i] = gmax; positions[i] = gpos; }
    }

    cudaFree(layer_a_d);
    cudaFree(layer_b_d);
    cudaFree(posval_d);
    cudaFree(block_max_d);
    cudaFree(block_pos_d);
    cudaFreeHost(block_max_h);
    cudaFreeHost(block_pos_h);
}
