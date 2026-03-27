#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "energy_storms.h"

/*
 * Block-size constants.
 *
 * BLOCK_BOMB = 128
 *   layer_size=20000 → 157 blocks on the 72-SM Quadro RTX 6000
 *   → ~2.2 blocks/SM.  Turing can schedule up to 8 blocks of 128 threads
 *   per SM (8 × 128 = 1024 threads = 32 warps = full warp occupancy),
 *   so the SM has 16+ warps to hide every memory latency stall.
 *   With BLOCK=256 (the original) you only get 79 blocks → 1.1 blocks/SM
 *   → at most 2 warps/SM can hide latency → poor throughput on the Quadro.
 *
 * BLOCK_OTHER = 256
 *   relax and find_max are memory-bound.  Larger blocks give better
 *   coalescing; 256 is the usual sweet-spot for bandwidth-limited kernels.
 */
#define BLOCK_BOMB    128
#define BLOCK_OTHER   256

/* -----------------------------------------------------------------------
 * Particle tile stored in shared memory.
 *
 * energy_div_size = (raw * 1000.0f) / (float)layer_size
 *   Uses explicit FP divide — NOT multiplication by 1.0f/layer_size.
 *   The sequential reference computes:
 *       energy_k = energy / layer_size / atenuacion;   (two successive divides)
 *   a * (1.0f / n) gives different IEEE-754 rounding than a / n and
 *   produces results that differ from the reference in the last ULP.
 * ----------------------------------------------------------------------- */
struct __align__(8) Particle {
    float energy_div_size;   /* (raw*1000.f) / layer_size  via FP divide */
    int   pos;
};

/* -----------------------------------------------------------------------
 * update_fast() — arithmetic identical to the sequential reference.
 * energy_div_size already carries the /layer_size factor, so the only
 * remaining operation is a single /sqrtf, matching:
 *   energy_k = energy / layer_size / atenuacion
 *            = energy_div_size     / sqrtf(distance)
 * ----------------------------------------------------------------------- */
__device__ __forceinline__
float update_fast(int i, float energy_div_size,
                  float threshold_div_size, int pos) {
    int   distance = abs(pos - i) + 1;
    float energy_k = energy_div_size / sqrtf((float)distance);
    return (fabsf(energy_k) >= threshold_div_size) ? energy_k : 0.0f;
}

/* -----------------------------------------------------------------------
 * bomb — shared-memory tiling, one thread per layer cell.
 *
 * One thread owns one fixed layer index i (no grid-stride outer loop).
 * Both __syncthreads() calls are therefore always hit by every thread in
 * the block, including out-of-bounds threads (which skip only the
 * arithmetic) — no divergent synchronisation.
 *
 * Particles are processed in j = 0 … storm_size-1 order, matching the
 * sequential reference, so the FP summation sequence is identical.
 *
 * Shared memory: BLOCK_BOMB × sizeof(Particle) = 1 KB/block.
 * At 8 concurrent blocks/SM on Turing → 8 KB/SM (well under the 48 KB limit).
 *
 * Launch: bomb<<<nblocks_bomb, BLOCK_BOMB, BLOCK_BOMB*sizeof(Particle)>>>
 * ----------------------------------------------------------------------- */
__global__ __launch_bounds__(BLOCK_BOMB, 8)
void bomb(      float* __restrict__ layer_d,
          const int*   __restrict__ posval_d,
          int                       layer_size,
          int                       storm_size,
          float                     threshold_div_size) {

    extern __shared__ Particle tile[];

    const int i   = blockIdx.x * blockDim.x + threadIdx.x;
    const bool ok = (i < layer_size);

    float acc          = ok ? layer_d[i] : 0.0f;
    const float lsf    = (float)layer_size;   /* for the explicit FP divide */

    for (int t = 0; t < storm_size; t += blockDim.x) {

        /* --- cooperative load: each thread loads one particle ----------- */
        int load = t + threadIdx.x;
        if (load < storm_size) {
            /*
             * Explicit FP divide — preserves the IEEE-754 rounding of the
             * sequential reference.  Do NOT replace with * (1.0f / lsf).
             */
            tile[threadIdx.x].energy_div_size =
                (float)__ldg(&posval_d[load * 2 + 1]) * 1000.0f / lsf;
            tile[threadIdx.x].pos = __ldg(&posval_d[load * 2]);
        }
        __syncthreads();   /* uniform — every thread always reaches this */

        /* --- each thread computes its fixed layer cell against the tile - */
        if (ok) {
            int tile_size = min((int)blockDim.x, storm_size - t);
            int j = 0;
            /* Unroll ×4 to expose independent FP operations to scheduler */
            for (; j <= tile_size - 4; j += 4) {
                acc += update_fast(i, tile[j  ].energy_div_size, threshold_div_size, tile[j  ].pos);
                acc += update_fast(i, tile[j+1].energy_div_size, threshold_div_size, tile[j+1].pos);
                acc += update_fast(i, tile[j+2].energy_div_size, threshold_div_size, tile[j+2].pos);
                acc += update_fast(i, tile[j+3].energy_div_size, threshold_div_size, tile[j+3].pos);
            }
            for (; j < tile_size; j++)
                acc += update_fast(i, tile[j].energy_div_size, threshold_div_size, tile[j].pos);
        }
        __syncthreads();   /* guard tile before the next cooperative load */
    }

    if (ok) layer_d[i] = acc;
}

/* -----------------------------------------------------------------------
 * relax
 *   Sequential reference: for(k=1; k<layer_size-1; k++) — skip endpoints.
 *   Start at tid+1, stop before layer_size-1.
 * ----------------------------------------------------------------------- */
__global__ __launch_bounds__(BLOCK_OTHER)
void relax(      float* __restrict__ layer_d,
           const float* __restrict__ layer_copy_d,
           int                       layer_size) {

    const int grid_stride = blockDim.x * gridDim.x;
    const int tid         = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = tid + 1; i < layer_size - 1; i += grid_stride)
        layer_d[i] = (layer_copy_d[i-1] + layer_copy_d[i] + layer_copy_d[i+1]) / 3.0f;
}

/* -----------------------------------------------------------------------
 * find_max
 *   In-kernel conditional reduction.  Each block emits one (value, pos)
 *   pair.  Only nblocks_other small results cross the PCIe bus, not the
 *   full layer (critical for test_08 with 100 M positions = ~400 MB/storm
 *   with the original approach).
 *
 *   Dynamic shared memory: blockDim.x floats + blockDim.x ints = 2 KB/block.
 * ----------------------------------------------------------------------- */
__global__ __launch_bounds__(BLOCK_OTHER)
void find_max(const float* __restrict__ layer_d,
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

    const int nblocks_bomb  = (layer_size + BLOCK_BOMB  - 1) / BLOCK_BOMB;
    const int nblocks_other = (layer_size + BLOCK_OTHER - 1) / BLOCK_OTHER;

    const float threshold_div_size = THRESHOLD / (float)layer_size;

    /* ---- Largest storm → posval_d capacity ---- */
    int max_storm_size = 0;
    for (int i = 0; i < num_storms; i++)
        if (storms[i].size > max_storm_size) max_storm_size = storms[i].size;

    /* ---- Device buffers — allocated once outside the storm loop ----
     *  Original called cudaMalloc/cudaFree inside the loop, forcing a
     *  device sync on every storm iteration.
     * ---------------------------------------------------------------- */
    float *layer_d, *layer_copy_d;
    cudaMalloc((void **)&layer_d,      sizeof(float) * layer_size);
    cudaMalloc((void **)&layer_copy_d, sizeof(float) * layer_size);
    cudaMemset(layer_d, 0, sizeof(float) * layer_size);

    int *posval_d;
    cudaMalloc((void **)&posval_d, sizeof(int) * max_storm_size * 2);

    float *block_max_d; int *block_pos_d;
    cudaMalloc((void **)&block_max_d, sizeof(float) * nblocks_other);
    cudaMalloc((void **)&block_pos_d, sizeof(int)   * nblocks_other);

    /* ---- Plain malloc for the small host-side reduction arrays --------
     *  cudaMallocHost (pinned memory) is restricted on many HPC clusters.
     *  These arrays are only nblocks_other elements (~78 floats for a 20k
     *  layer) so the DMA overhead of pageable memory is negligible.
     * ---------------------------------------------------------------- */
    float *block_max_h = (float *)malloc(sizeof(float) * nblocks_other);
    int   *block_pos_h = (int   *)malloc(sizeof(int)   * nblocks_other);
    if (!block_max_h || !block_pos_h) {
        fprintf(stderr, "Error: malloc failed for reduction buffers\n");
        exit(EXIT_FAILURE);
    }

    const size_t smem_bomb = BLOCK_BOMB  * sizeof(Particle);
    const size_t smem_max  = BLOCK_OTHER * (sizeof(float) + sizeof(int));

    for (int i = 0; i < num_storms; i++) {

        /* 4.1. Upload storm i's particles (synchronous — portable everywhere) */
        cudaMemcpy(posval_d, storms[i].posval,
                   sizeof(int) * storms[i].size * 2,
                   cudaMemcpyHostToDevice);

        /* 4.2. Bomb */
        bomb<<<nblocks_bomb, BLOCK_BOMB, smem_bomb>>>(
            layer_d, posval_d,
            layer_size, storms[i].size,
            threshold_div_size);

        /* 4.3. Relaxation: snapshot then smooth */
        cudaMemcpy(layer_copy_d, layer_d,
                   sizeof(float) * layer_size,
                   cudaMemcpyDeviceToDevice);

        relax<<<nblocks_other, BLOCK_OTHER>>>(
            layer_d, layer_copy_d, layer_size);

        /* 4.4. GPU reduction */
        find_max<<<nblocks_other, BLOCK_OTHER, smem_max>>>(
            layer_d, layer_size, block_max_d, block_pos_d);

        /* Transfer only the small per-block results to the host */
        cudaMemcpy(block_max_h, block_max_d,
                   sizeof(float) * nblocks_other,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(block_pos_h, block_pos_d,
                   sizeof(int) * nblocks_other,
                   cudaMemcpyDeviceToHost);

        /* 4.5. Host-side reduction over per-block results */
        float gmax = -FLT_MAX;
        int   gpos = -1;
        for (int b = 0; b < nblocks_other; b++) {
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

    /* ---- Free everything ---- */
    cudaFree(layer_d);
    cudaFree(layer_copy_d);
    cudaFree(posval_d);
    cudaFree(block_max_d);
    cudaFree(block_pos_d);
    free(block_max_h);
    free(block_pos_h);
}
