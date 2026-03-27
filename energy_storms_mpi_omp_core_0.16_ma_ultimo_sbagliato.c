#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include "energy_storms.h"
#include <omp.h>
#include <mpi.h>

void core(int layer_size, int num_storms, Storm *storms, float *maximum, int *positions) {
    int i, j, k, size, rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int chunk = layer_size / size;
    int rem   = layer_size % size;
    int lsize = chunk + (rank < rem ? 1 : 0);
    int start = rank * chunk + (rank < rem ? rank : rem);

    float *layer_base      = (float *)calloc(lsize + 2, sizeof(float));
    float *layer_copy_base = (float *)calloc(lsize + 2, sizeof(float));
    if (!layer_base || !layer_copy_base) {
        fprintf(stderr, "Error: Allocating the layer memory\n");
        exit(EXIT_FAILURE);
    }
    float *layer      = layer_base + 1;
    float *layer_copy = layer_copy_base + 1;

    int left  = (rank > 0)        ? rank - 1 : MPI_PROC_NULL;
    int right = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    float inv_ls    = 1.0f / (float)layer_size;
    float threshold = THRESHOLD * inv_ls;

    /* ---------------------------------------------------------------
     * KEY FIX: Precompute sqrt LUT once for all distances 1..layer_size.
     * dist = |pos - gk| + 1, so dist ∈ [1, layer_size].
     * Replaces sqrtf() (≈20 cycles, not vectorisable) with
     * a L1-cached array read (≈4 cycles, vectorisable).
     * --------------------------------------------------------------- */
    float *sqrt_lut = (float *)malloc((size_t)(layer_size + 1) * sizeof(float));
    if (!sqrt_lut) { fprintf(stderr, "Error: sqrt_lut alloc\n"); exit(EXIT_FAILURE); }
    for (k = 1; k <= layer_size; k++)
        sqrt_lut[k] = sqrtf((float)k);

    int max_ssize = 0;
    for (i = 0; i < num_storms; i++)
        if (storms[i].size > max_ssize) max_ssize = storms[i].size;
    float *s_energy = (float *)malloc(max_ssize * sizeof(float));
    int   *s_pos    = (int   *)malloc(max_ssize * sizeof(int));
    if (!s_energy || !s_pos) {
        fprintf(stderr, "Error: Allocating storm scratch buffers\n");
        exit(EXIT_FAILURE);
    }

    struct { float val; int rank; } lmax_s, gmax_s;

    for (i = 0; i < num_storms; i++) {
        int ssize = storms[i].size;

        for (j = 0; j < ssize; j++) {
            s_pos[j]    = storms[i].posval[j * 2];
            s_energy[j] = (float)storms[i].posval[j * 2 + 1] * 1000.0f;
        }

        /* ---------------------------------------------------------------
         * IMPACT: k-outer / j-inner, OMP on k.
         * - sqrt_lut[dist] replaces sqrtf(dist)
         * - cell_val is a register-local accumulator: eliminates repeated
         *   loads/stores to layer[k] across the j loop (the compiler may
         *   already do this under -O3, but explicit is safer with OpenMP)
         * - fabsf(ek) >= threshold is a single-compare form the compiler
         *   can vectorise; the original (ek>=t || ek<=-t) cannot
         * --------------------------------------------------------------- */
        #pragma omp parallel for schedule(static) private(j)
        for (k = 0; k < lsize; k++) {
            int   gk       = k + start;
            float cell_val = layer[k];          /* hoist load out of j loop */
            for (j = 0; j < ssize; j++) {
                int   dist = abs(s_pos[j] - gk) + 1;
                float ek   = s_energy[j] * inv_ls / sqrt_lut[dist];
                if (fabsf(ek) >= threshold)
                    cell_val += ek;
            }
            layer[k] = cell_val;                /* single store after j loop */
        }

        /* Ghost exchange #1 — pre-relax */
        MPI_Sendrecv(&layer[lsize - 1], 1, MPI_FLOAT, right, 0,
                     &layer[-1],        1, MPI_FLOAT, left,  0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&layer[0],    1, MPI_FLOAT, left,  1,
                     &layer[lsize], 1, MPI_FLOAT, right, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Copy — include ghost cells so border relaxation is correct */
        layer_copy[-1]    = layer[-1];
        layer_copy[lsize] = layer[lsize];
        #pragma omp parallel for schedule(static)
        for (k = 0; k < lsize; k++)
            layer_copy[k] = layer[k];

        /* Relax */
        #pragma omp parallel for schedule(static)
        for (k = 0; k < lsize; k++) {
            int gk = k + start;
            if (gk == 0 || gk == layer_size - 1) continue;
            layer[k] = (layer_copy[k - 1] + layer_copy[k] + layer_copy[k + 1]) / 3.0f;
        }

        /* Ghost exchange #2 — post-relax, needed for correct border max-finding */
        MPI_Sendrecv(&layer[lsize - 1], 1, MPI_FLOAT, right, 2,
                     &layer[-1],        1, MPI_FLOAT, left,  2,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&layer[0],    1, MPI_FLOAT, left,  3,
                     &layer[lsize], 1, MPI_FLOAT, right, 3,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Max find — thread-local, then single MPI_Allreduce */
        float lmax = -FLT_MAX;
        int   lpos = 0;

        #pragma omp parallel
        {
            float tmax = -FLT_MAX;
            int   tpos = 0;
            #pragma omp for schedule(static) nowait
            for (k = 0; k < lsize; k++) {
                int gk = k + start;
                if (gk == 0 || gk == layer_size - 1) continue;
                if (layer[k] > layer[k - 1] && layer[k] > layer[k + 1] && layer[k] > tmax) {
                    tmax = layer[k];
                    tpos = gk;
                }
            }
            #pragma omp critical
            { if (tmax > lmax) { lmax = tmax; lpos = tpos; } }
        }

        lmax_s.val  = lmax;
        lmax_s.rank = rank;
        MPI_Allreduce(&lmax_s, &gmax_s, 1, MPI_FLOAT_INT, MPI_MAXLOC, MPI_COMM_WORLD);

        int gpos = (rank == gmax_s.rank) ? lpos : 0;
        MPI_Bcast(&gpos, 1, MPI_INT, gmax_s.rank, MPI_COMM_WORLD);

        if (rank == 0) {
            maximum[i]   = gmax_s.val;
            positions[i] = gpos;
        }
    }

    free(sqrt_lut);
    free(s_energy);
    free(s_pos);
    free(layer_base);
    free(layer_copy_base);
}