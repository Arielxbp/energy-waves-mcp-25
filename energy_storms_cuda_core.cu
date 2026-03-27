#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "energy_storms.h"

/* ------------------------------------------------------------------
 * DEVICE HELPER
 * ------------------------------------------------------------------ */
__device__ __forceinline__ float update(int affected, int layer_size,
                                         float energy, int contact) {
    int distance = abs(contact - affected) + 1;
    float energy_k = energy / layer_size / sqrtf((float)distance);
    float threshold = THRESHOLD / layer_size;
    return (fabsf(energy_k) >= threshold) ? energy_k : 0.0f;
}

/* ------------------------------------------------------------------
 * KERNEL 1 – BOMB
 * Each thread owns one or more layer cells (grid-stride) and loops
 * over all storm particles.  posval_d is read via __ldg so the
 * read-only cache broadcasts each particle pair to the whole warp.
 * ------------------------------------------------------------------ */
__global__ void bomb(float * __restrict__ layer_d,
                     int layer_size,
                     const int * __restrict__ posval_d,
                     int storm_size) {
    const int grid_stride = blockDim.x * gridDim.x;
    const int tid         = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = tid; i < layer_size; i += grid_stride) {
        float val = layer_d[i];
        for (int j = 0; j < storm_size; j++) {
            float energy   = (float)__ldg(&posval_d[j * 2 + 1]) * 1000.0f;
            int   contact  = __ldg(&posval_d[j * 2]);
            val += update(i, layer_size, energy, contact);
        }
        layer_d[i] = val;
    }
}

/* ------------------------------------------------------------------
 * KERNEL 2 – RELAX (ping-pong, no separate D2D copy needed)
 * Reads from src (the "old" buffer), writes into dst (the "new" one).
 * The caller swaps the pointers between storms.
 * ------------------------------------------------------------------ */
__global__ void relax(float * __restrict__       dst,
                      const float * __restrict__ src,
                      int layer_size) {
    const int grid_stride = blockDim.x * gridDim.x;
    const int tid         = blockIdx.x * blockDim.x + threadIdx.x;

    /* Boundaries (index 0 and layer_size-1) are NOT updated per spec */
    for (int i = tid + 1; i < layer_size - 1; i += grid_stride) {
        dst[i] = (src[i-1] + src[i] + src[i+1]) / 3.0f;
    }
    /* Copy the boundary cells unchanged so the next storm sees them */
    if (tid == 0)                   dst[0]             = src[0];
    if (tid == layer_size - 1 ||
        (tid < layer_size && tid + grid_stride >= layer_size))
        dst[layer_size - 1] = src[layer_size - 1];
}

/* ------------------------------------------------------------------
 * KERNEL 3 – PER-BLOCK REDUCTION (first pass)
 * Each block produces one (max_val, max_pos) pair into global arrays.
 * ------------------------------------------------------------------ */
__global__ void reduce_pass1(const float * __restrict__ layer_d,
                              int layer_size,
                              float * __restrict__ block_max,
                              int   * __restrict__ block_pos) {
    extern __shared__ float sdata[];
    int *spos = (int *)(sdata + blockDim.x);

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
    spos [tid] = lpos;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s && sdata[tid + s] > sdata[tid]) {
            sdata[tid] = sdata[tid + s];
            spos [tid] = spos [tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_max[blockIdx.x] = sdata[0];
        block_pos[blockIdx.x] = spos [0];
    }
}

/* ------------------------------------------------------------------
 * KERNEL 4 – SINGLE-BLOCK REDUCTION (second pass)
 * Launched with exactly one block of nblocks threads (nblocks <= 256).
 * Produces a single (max_val, max_pos) in global scalars.
 * ------------------------------------------------------------------ */
__global__ void reduce_pass2(const float * __restrict__ block_max,
                              const int   * __restrict__ block_pos,
                              int nblocks,
                              float * __restrict__ g_max,
                              int   * __restrict__ g_pos) {
    extern __shared__ float sdata[];
    int *spos = (int *)(sdata + blockDim.x);

    const int tid = threadIdx.x;

    float lmax = (tid < nblocks) ? block_max[tid] : -FLT_MAX;
    int   lpos = (tid < nblocks) ? block_pos[tid] : -1;

    sdata[tid] = lmax;
    spos [tid] = lpos;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s && sdata[tid + s] > sdata[tid]) {
            sdata[tid] = sdata[tid + s];
            spos [tid] = spos [tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *g_max = sdata[0];
        *g_pos = spos [0];
    }
}

/* ------------------------------------------------------------------
 * CORE
 * ------------------------------------------------------------------ */
void core(int layer_size, int num_storms, Storm *storms,
          float *maximum, int *positions) {

    /* ---- Grid / block geometry ---- */
    const int BLOCK   = 256;
    const int nblocks = (layer_size + BLOCK - 1) / BLOCK;
    /* pass2 needs a power-of-two >= nblocks, capped at 256 */
    int p2 = 1;
    while (p2 < nblocks) p2 <<= 1;
    const int PASS2_THREADS = (p2 <= 256) ? p2 : 256; /* nblocks always <= 256 for sane sizes */

    /* ---- Find the largest storm so posval can be pre-allocated once ---- */
    int max_storm_size = 0;
    for (int i = 0; i < num_storms; i++)
        if (storms[i].size > max_storm_size) max_storm_size = storms[i].size;

    /* ---- Device buffers ---- */
    float *ping_d, *pong_d;   /* ping-pong pair for layer / layer_copy */
    cudaMalloc((void **)&ping_d, sizeof(float) * layer_size);
    cudaMalloc((void **)&pong_d, sizeof(float) * layer_size);
    cudaMemset(ping_d, 0, sizeof(float) * layer_size);
    cudaMemset(pong_d, 0, sizeof(float) * layer_size);

    /* Reduction temporaries */
    float *block_max_d;  int *block_pos_d;
    float *g_max_d;      int *g_pos_d;
    cudaMalloc((void **)&block_max_d, sizeof(float) * nblocks);
    cudaMalloc((void **)&block_pos_d, sizeof(int)   * nblocks);
    cudaMalloc((void **)&g_max_d,     sizeof(float));
    cudaMalloc((void **)&g_pos_d,     sizeof(int));

    /* ---------------------------------------------------------------
     * OPTIMISATION 1 – pre-upload ALL storms' posval before the loop.
     * This moves every H2D PCIe transfer out of the hot path.
     * ------------------------------------------------------------- */
    int **posval_d_arr = (int **)malloc(sizeof(int *) * num_storms);

    /* Use pinned (page-locked) host memory for the result scalars so
     * the final D2H transfer is as fast as possible.               */
    float *h_max_pinned;
    int   *h_pos_pinned;
    cudaMallocHost((void **)&h_max_pinned, sizeof(float));
    cudaMallocHost((void **)&h_pos_pinned, sizeof(int));

    for (int i = 0; i < num_storms; i++) {
        size_t bytes = sizeof(int) * storms[i].size * 2;
        cudaMalloc((void **)&posval_d_arr[i], bytes);
        /* Async upload – overlaps with host work */
        cudaMemcpyAsync(posval_d_arr[i], storms[i].posval, bytes,
                        cudaMemcpyHostToDevice, 0);
    }
    cudaDeviceSynchronize(); /* ensure all uploads are done before the loop */

    const size_t smem_pass1 = BLOCK        * (sizeof(float) + sizeof(int));
    const size_t smem_pass2 = PASS2_THREADS * (sizeof(float) + sizeof(int));

    /* ---- Storm loop ---- */
    float *src = ping_d;   /* current layer  */
    float *dst = pong_d;   /* relaxation target */

    for (int i = 0; i < num_storms; i++) {

        /* 4.1  Impacts – reads src in-place, writes src in-place */
        bomb<<<nblocks, BLOCK>>>(src, layer_size,
                                  posval_d_arr[i], storms[i].size);

        /* ---------------------------------------------------------------
         * OPTIMISATION 2 – ping-pong relaxation.
         * relax() reads from src and writes into dst; no extra D2D copy.
         * ------------------------------------------------------------- */
        relax<<<nblocks, BLOCK>>>(dst, src, layer_size);

        /* Swap buffers: dst becomes the new "current" layer */
        float *tmp = src; src = dst; dst = tmp;

        /* 4.3  Find maximum – two-pass entirely on GPU */
        reduce_pass1<<<nblocks, BLOCK, smem_pass1>>>(
            src, layer_size, block_max_d, block_pos_d);

        /* ---------------------------------------------------------------
         * OPTIMISATION 3 – second reduction pass on GPU.
         * Only 2 scalars (float + int) transferred to the host instead
         * of nblocks * (float + int).
         * ------------------------------------------------------------- */
        reduce_pass2<<<1, PASS2_THREADS, smem_pass2>>>(
            block_max_d, block_pos_d, nblocks, g_max_d, g_pos_d);

        /* Copy only 2 values – using pinned memory for low latency */
        cudaMemcpy(h_max_pinned, g_max_d, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_pos_pinned, g_pos_d, sizeof(int),   cudaMemcpyDeviceToHost);

        if (*h_max_pinned > maximum[i]) {
            maximum[i]   = *h_max_pinned;
            positions[i] = *h_pos_pinned;
        }
    }

    /* ---- Free device resources ---- */
    for (int i = 0; i < num_storms; i++) cudaFree(posval_d_arr[i]);
    free(posval_d_arr);

    cudaFree(ping_d);      cudaFree(pong_d);
    cudaFree(block_max_d); cudaFree(block_pos_d);
    cudaFree(g_max_d);     cudaFree(g_pos_d);
    cudaFreeHost(h_max_pinned);
    cudaFreeHost(h_pos_pinned);
}
