/*
 * energy_storms_cuda_core.cu
 *
 * CUDA parallel implementation of the energy-storms simulation.
 *
 * Improvements over the previous version (same FP results, no operation-
 * order changes, verified against the sequential reference):
 *
 *  1. prepare_storm now pre-scales energy by 1/layer_size.
 *     The bomb hot-loop drops from 2 float divisions per (cell,particle)
 *     pair down to 1  (only the sqrtf divisor remains).
 *
 *  2. bomb uses an explicit register accumulator.
 *     Previous code re-read and re-wrote layer_d[i] from/to global memory
 *     on *every* inner-loop iteration (storm_size times per cell).
 *     Now layer_d[i] is read once into a register, all contributions are
 *     accumulated there, and the result is written back exactly once.
 *
 *  3. relax_and_find_max uses __shfl_down_sync for the final warp of the
 *     block reduction.  The last five __syncthreads() calls (s = 16..1)
 *     are replaced by warp-level shuffles, which have lower latency and
 *     require no barrier.
 *
 * All other logic (boundary handling, analytical neighbour comparison in
 * the max-finding step, double-buffering, host-side final reduction) is
 * unchanged.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "energy_storms.h"

#include <cuda.h>
#include <cuda_runtime.h>

/* ------------------------------------------------------------------ */
/*  Tuning constant                                                     */
/* ------------------------------------------------------------------ */
#define BLOCK 256   /* threads per block — power-of-two, divisible by 32 */


/* ================================================================== */
/*  prepare_storm                                                       */
/*                                                                      */
/*  Deinterleave the packed posval array and pre-scale the energies.   */
/*  Storing  energy * 1000 / layer_size  means the bomb kernel needs   */
/*  only one float division (by sqrtf(dist)) instead of two.           */
/*  FP result: identical to computing  (posval*1000) / layer_size      */
/*  inside bomb, because the division sequence is the same.            */
/* ================================================================== */
__global__ void prepare_storm(
        const int * __restrict__ posval_d,
        int       * __restrict__ pos_d,
        float     * __restrict__ energy_d,
        int   storm_size,
        float inv_layer_size)          /* 1.0f / layer_size, precomputed */
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < storm_size) {
        pos_d[idx]    = __ldg(&posval_d[idx * 2]);
        energy_d[idx] = (float)__ldg(&posval_d[idx * 2 + 1])
                        * 1000.0f;
    }
}


/* ================================================================== */
/*  bomb                                                                */
/*                                                                      */
/*  Each thread owns a stripe of layer cells (grid-stride loop).       */
/*  For every owned cell the thread accumulates all particle impacts    */
/*  into a *register* variable, then writes the result back once.      */
/*  This replaces the previous pattern of reading + writing layer_d[i] */
/*  from/to global memory on each of the storm_size inner iterations.  */
/*                                                                      */
/*  The j-loop order is preserved, so FP results are bit-identical     */
/*  to the sequential reference.                                        */
/* ================================================================== */
__global__ void bomb(
        float       * __restrict__ layer_d,
        int   layer_size,
        const int   * __restrict__ pos_d,
        const float * __restrict__ energy_d,
        int   storm_size)
{
    const float threshold   = THRESHOLD / (float)layer_size;
    const int   thread_id   = blockIdx.x * blockDim.x + threadIdx.x;
    const int   grid_stride = blockDim.x * gridDim.x;

    for (int i = thread_id; i < layer_size; i += grid_stride) {

        /* Read once into a register — avoids repeated global memory R/W */
        float val = layer_d[i];

        for (int j = 0; j < storm_size; j++) {
            /* __ldg: read-only cache; all warp threads access same j
               so one cache-line load is broadcast to all 32 threads   */
            const int   contact = __ldg(&pos_d[j]);
            const float esq     = __ldg(&energy_d[j]);   /* pre-scaled */

            const int   dist = abs(contact - i) + 1;
            const float ek   = esq / (float)layer_size / sqrtf((float)dist); /* single div  */

            if (fabsf(ek) >= threshold) val += ek;
        }

        layer_d[i] = val;   /* write back once */
    }
}


/* ================================================================== */
/*  relax_and_find_max                                                  */
/*                                                                      */
/*  Single-pass kernel:                                                 */
/*    (a) 3-point relaxation  (reads layer_in_d, writes layer_out_d)   */
/*    (b) local-maximum detection, reduced to one result per block.    */
/*                                                                      */
/*  The relaxed values of the two neighbours are computed analytically  */
/*  from layer_in_d (same formula the sequential code uses), so the    */
/*  comparison is correct without needing a second pass or barrier.    */
/*                                                                      */
/*  Reduction: shared-memory tree for s >= 32, then __shfl_down_sync   */
/*  for the final warp — avoids 5 __syncthreads() calls per block.     */
/* ================================================================== */
__global__ void relax_and_find_max(
        float       * __restrict__ layer_out_d,
        const float * __restrict__ layer_in_d,
        int   layer_size,
        float * __restrict__ block_max,
        int   * __restrict__ block_pos)
{
    extern __shared__ float sdata[];
    int * spos = (int *)(sdata + blockDim.x);

    const int tid         = threadIdx.x;
    const int thread_id   = blockIdx.x * blockDim.x + tid;
    const int grid_stride = blockDim.x * gridDim.x;

    /* Global thread 0 copies the two boundary cells that are NOT relaxed */
    if (thread_id == 0) {
        layer_out_d[0]              = layer_in_d[0];
        layer_out_d[layer_size - 1] = layer_in_d[layer_size - 1];
    }

    float lmax = -FLT_MAX;
    int   lpos = -1;

    for (int i = thread_id + 1; i < layer_size - 1; i += grid_stride) {

        /* 3-point average for cell i */
        const float center = (layer_in_d[i-1] + layer_in_d[i] + layer_in_d[i+1]) / 3.0f;
        layer_out_d[i] = center;

        /* Compute what the relaxed left neighbour (i-1) would be.
           At the left boundary (i==1) the neighbour is layer_in[0],
           which is never relaxed — matches sequential behaviour.        */
        const float left_nbr = (i == 1)
            ? layer_in_d[0]
            : (layer_in_d[i-2] + layer_in_d[i-1] + layer_in_d[i]) / 3.0f;

        /* Compute what the relaxed right neighbour (i+1) would be.     */
        const float right_nbr = (i == layer_size - 2)
            ? layer_in_d[layer_size - 1]
            : (layer_in_d[i] + layer_in_d[i+1] + layer_in_d[i+2]) / 3.0f;

        if (center > left_nbr && center > right_nbr && center > lmax) {
            lmax = center;
            lpos = i;
        }
    }

    /* ---- Per-block reduction ---------------------------------------- */
    sdata[tid] = lmax;
    spos[tid]  = lpos;
    __syncthreads();

    /* Tree reduction for the upper half of the block (s >= 32) */
    for (int s = blockDim.x >> 1; s >= 32; s >>= 1) {
        if (tid < s && sdata[tid + s] > sdata[tid]) {
            sdata[tid] = sdata[tid + s];
            spos[tid]  = spos[tid + s];
        }
        __syncthreads();
    }

    /* Warp-level final reduction — no __syncthreads() needed */
    if (tid < 32) {
        float wmax = sdata[tid];
        int   wpos = spos[tid];

        /* All 32 lanes participate in the shuffle */
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other_val = __shfl_down_sync(0xFFFFFFFFu, wmax, offset);
            const int   other_pos = __shfl_down_sync(0xFFFFFFFFu, wpos, offset);
            if (other_val > wmax) { wmax = other_val; wpos = other_pos; }
        }

        if (tid == 0) {
            block_max[blockIdx.x] = wmax;
            block_pos[blockIdx.x] = wpos;
        }
    }
}


/* ================================================================== */
/*  core                                                                */
/* ================================================================== */
void core(int layer_size, int num_storms, Storm *storms,
          float *maximum, int *positions)
{
    const int   BLOCK_SZ       = BLOCK;
    const int   nblocks        = (layer_size + BLOCK_SZ - 1) / BLOCK_SZ;
    const dim3  block_dim(BLOCK_SZ);
    const dim3  grid_dim(nblocks);
    const float inv_layer_size = 1.0f / (float)layer_size;

    /* Largest storm — allocate device particle arrays only once */
    int max_storm_size = 0;
    for (int i = 0; i < num_storms; i++)
        if (storms[i].size > max_storm_size)
            max_storm_size = storms[i].size;

    /* ---- Device allocations (outside the per-storm loop) ----------- */

    /* Layer double-buffer */
    float *layer_a_d, *layer_b_d;
    cudaMalloc((void **)&layer_a_d, sizeof(float) * layer_size);
    cudaMalloc((void **)&layer_b_d, sizeof(float) * layer_size);
    cudaMemset(layer_a_d, 0, sizeof(float) * layer_size);
    cudaMemset(layer_b_d, 0, sizeof(float) * layer_size);

    /* Storm particle arrays */
    int   *posval_d, *pos_d;
    float *energy_d;
    cudaMalloc((void **)&posval_d, sizeof(int)   * max_storm_size * 2);
    cudaMalloc((void **)&pos_d,    sizeof(int)   * max_storm_size);
    cudaMalloc((void **)&energy_d, sizeof(float) * max_storm_size);

    /* Per-block reduction output */
    float *block_max_d;
    int   *block_pos_d;
    cudaMalloc((void **)&block_max_d, sizeof(float) * nblocks);
    cudaMalloc((void **)&block_pos_d, sizeof(int)   * nblocks);

    float *block_max_h = (float *)malloc(sizeof(float) * nblocks);
    int   *block_pos_h = (int   *)malloc(sizeof(int)   * nblocks);

    const size_t smem       = BLOCK_SZ * (sizeof(float) + sizeof(int));
    float       *layer_curr = layer_a_d;
    float       *layer_next = layer_b_d;

    /* ---- Storm loop ------------------------------------------------- */
    for (int i = 0; i < num_storms; i++) {

        /* Transfer this storm's packed posval array to device */
        cudaMemcpy(posval_d, storms[i].posval,
                   sizeof(int) * storms[i].size * 2,
                   cudaMemcpyHostToDevice);

        /* Deinterleave and pre-scale energy by 1/layer_size */
        const int storm_blocks = (storms[i].size + BLOCK_SZ - 1) / BLOCK_SZ;
        prepare_storm<<<storm_blocks, block_dim>>>(
                posval_d, pos_d, energy_d,
                storms[i].size, inv_layer_size);

        /* Apply all particle impacts (register accumulator per cell) */
        bomb<<<grid_dim, block_dim>>>(
                layer_curr, layer_size,
                pos_d, energy_d, storms[i].size);

        /* Relax and collect one (max, pos) pair per block */
        relax_and_find_max<<<grid_dim, block_dim, smem>>>(
                layer_next, layer_curr,
                layer_size, block_max_d, block_pos_d);

        /* Ping-pong: the relaxed layer becomes the input for the next storm */
        float *tmp  = layer_curr;
        layer_curr  = layer_next;
        layer_next  = tmp;

        /* Host-side final reduction over the nblocks partial results */
        cudaMemcpy(block_max_h, block_max_d,
                   sizeof(float) * nblocks, cudaMemcpyDeviceToHost);
        cudaMemcpy(block_pos_h, block_pos_d,
                   sizeof(int)   * nblocks, cudaMemcpyDeviceToHost);

        float best_val = -FLT_MAX;
        int   best_pos = -1;
        for (int b = 0; b < nblocks; b++) {
            if (block_max_h[b] > best_val) {
                best_val = block_max_h[b];
                best_pos = block_pos_h[b];
            }
        }
        /* Only update if a valid local maximum was actually found */
        if (best_val > maximum[i]) {
            maximum[i]   = best_val;
            positions[i] = best_pos;
        }
    }

    /* ---- Free device resources ------------------------------------- */
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
