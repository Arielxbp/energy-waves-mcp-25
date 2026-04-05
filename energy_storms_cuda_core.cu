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
 * bomb — unchanged, __ldg for uniform warp reads.
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
 *
 * Reads from layer_in_d, writes to layer_out_d.
 * Used with ping-pong: the caller swaps the two pointers each storm
 * instead of doing an explicit cudaMemcpy(D2D) before calling relax.
 *
 * For test_08 (100 M floats) the old D2D copy moved 400 MB over the
 * GPU memory bus every storm.  Ping-pong eliminates that entirely.
 *
 * Correctness note: layer_out_d[0] and layer_out_d[layer_size-1] are
 * never written by this kernel (matches the sequential reference).
 * They retain whatever value layer_in_d had at that position — which
 * after the first storm is the correct accumulated value from bomb().
 * Positions 0 and layer_size-1 grow monotonically via bomb() and are
 * never smoothed, exactly as in the sequential code.
 * ----------------------------------------------------------------------- */
__global__ void relax(      float* __restrict__ layer_out_d,
                      const float* __restrict__ layer_in_d,
                      int                       layer_size) {
    const int grid_stride = blockDim.x * gridDim.x;
    const int thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = thread_id + 1; i < layer_size - 1; i += grid_stride) {
        layer_out_d[i] = (layer_in_d[i-1] + layer_in_d[i] + layer_in_d[i+1]) / 3.0f;
    }
}

/* -----------------------------------------------------------------------
 * copy_boundaries
 *
 * After ping-pong, positions 0 and layer_size-1 in layer_out_d still
 * hold the values from the *previous* ping-pong round, not the latest
 * values from bomb().  This tiny kernel (2 threads) copies the current
 * boundary values from layer_in_d to layer_out_d so that the next
 * storm's bomb() accumulates onto the correct base.
 * ----------------------------------------------------------------------- */
__global__ void copy_boundaries(      float* __restrict__ layer_out_d,
                                const float* __restrict__ layer_in_d,
                                int                       layer_size) {
    if (threadIdx.x == 0) layer_out_d[0]             = layer_in_d[0];
    if (threadIdx.x == 1) layer_out_d[layer_size - 1] = layer_in_d[layer_size - 1];
}

/* -----------------------------------------------------------------------
 * find_max — unchanged.
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
        if (layer_d[i] > layer_d[i-1] && layer_d[i] > layer_d[i+1])
            if (layer_d[i] > lmax) { lmax = layer_d[i]; lpos = i; }
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

    /* ------------------------------------------------------------------ *
     * Pre-upload ALL storm posval arrays to device before the storm loop.
     *
     * Old approach: cudaMemcpy(posval_d, ..., H2D) inside the loop.
     * On a cluster PCIe is slow and shared — this transfer sat on the
     * critical path every iteration.
     *
     * New approach: pack all storms into one contiguous device allocation,
     * upload with a single async copy per storm before the loop, then
     * synchronise once.  The storm loop only touches device pointers.
     * ------------------------------------------------------------------ */
    int total_ints = 0;
    for (int i = 0; i < num_storms; i++) total_ints += storms[i].size * 2;

    int *all_posval_d;
    cudaMalloc((void **)&all_posval_d, sizeof(int) * total_ints);

    /* per-storm device pointer — points into the contiguous allocation   */
    int **storm_posval_d = (int **)malloc(sizeof(int *) * num_storms);
    {
        int offset = 0;
        for (int i = 0; i < num_storms; i++) {
            storm_posval_d[i] = all_posval_d + offset;
            cudaMemcpyAsync(storm_posval_d[i], storms[i].posval,
                            sizeof(int) * storms[i].size * 2,
                            cudaMemcpyHostToDevice, 0);
            offset += storms[i].size * 2;
        }
    }
    /* All uploads are in flight; synchronise before kernels read them    */
    cudaDeviceSynchronize();

    /* ------------------------------------------------------------------ *
     * Ping-pong layer buffers.
     *
     * Old approach:
     *   cudaMemcpy(layer_copy_d, layer_d, layer_size, D2D)   ← 400 MB!
     *   relax(layer_d, layer_copy_d)
     *
     * New approach:
     *   relax(layer_next_d, layer_curr_d)   ← reads curr, writes next
     *   swap pointers                        ← zero-cost
     *
     * The explicit D2D copy is gone.  For test_08 this removes 400 MB of
     * device-memory traffic per storm — the dominant cost on the cluster.
     *
     * Both buffers start zeroed.  bomb() always accumulates into curr_d.
     * After relax, curr_d and next_d are swapped so the relaxed values
     * become the new curr_d for the next storm's bomb().
     * ------------------------------------------------------------------ */
    float *layer_a_d, *layer_b_d;
    cudaMalloc((void **)&layer_a_d, sizeof(float) * layer_size);
    cudaMalloc((void **)&layer_b_d, sizeof(float) * layer_size);
    cudaMemset(layer_a_d, 0, sizeof(float) * layer_size);
    cudaMemset(layer_b_d, 0, sizeof(float) * layer_size);

    float *curr_d = layer_a_d;   /* bomb() writes here; find_max reads here */
    float *next_d = layer_b_d;   /* relax() writes here                     */

    /* ------------------------------------------------------------------ *
     * Pinned host buffers for the D2H result copy.
     * cudaMallocHost allocates page-locked memory; the DMA engine can
     * transfer directly without an intermediate kernel bounce-buffer,
     * which matters on cluster nodes where PCIe bandwidth is precious.
     * ------------------------------------------------------------------ */
    float *block_max_h;
    int   *block_pos_h;
    cudaMallocHost((void **)&block_max_h, sizeof(float) * nblocks);
    cudaMallocHost((void **)&block_pos_h, sizeof(int)   * nblocks);

    float *block_max_d;  int *block_pos_d;
    cudaMalloc((void **)&block_max_d, sizeof(float) * nblocks);
    cudaMalloc((void **)&block_pos_d, sizeof(int)   * nblocks);

    /* ------------------------------------------------------------------ */
    for (int i = 0; i < num_storms; i++) {

        /* 4.1. Particle impacts — accumulate into curr_d */
        bomb<<<gridDim, blockDim>>>(curr_d, layer_size,
                                    storm_posval_d[i], storms[i].size);

        /* 4.2. Relaxation — read curr_d, write next_d (no D2D copy) */
        relax<<<gridDim, blockDim>>>(next_d, curr_d, layer_size);

        /* 4.2b. Propagate boundary values 0 and layer_size-1 into next_d
                 so bomb() for the next storm accumulates onto the correct
                 boundary values.  Only 2 values, negligible cost.        */
        copy_boundaries<<<1, 2>>>(next_d, curr_d, layer_size);

        /* Swap: next_d becomes the new curr_d for the next storm         */
        float *tmp = curr_d;  curr_d = next_d;  next_d = tmp;

        /* 4.3. GPU reduction — curr_d now holds the relaxed layer        */
        find_max<<<gridDim, blockDim, smem_max>>>(curr_d, layer_size,
                                                   block_max_d, block_pos_d);

        /* 4.4. Copy only nblocks results to pinned host memory           */
        cudaMemcpy(block_max_h, block_max_d, sizeof(float) * nblocks,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(block_pos_h, block_pos_d, sizeof(int)   * nblocks,
                   cudaMemcpyDeviceToHost);

        float gmax = -FLT_MAX;
        int   gpos = -1;
        for (int b = 0; b < nblocks; b++) {
            if (block_max_h[b] > gmax) {
                gmax = block_max_h[b];
                gpos = block_pos_h[b];
            }
        }
        if (gmax > maximum[i]) {
            maximum[i]   = gmax;
            positions[i] = gpos;
        }
    }

    cudaFree(layer_a_d);
    cudaFree(layer_b_d);
    cudaFree(all_posval_d);
    cudaFree(block_max_d);
    cudaFree(block_pos_d);
    cudaFreeHost(block_max_h);
    cudaFreeHost(block_pos_h);
    free(storm_posval_d);
}
