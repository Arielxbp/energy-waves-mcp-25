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

    /* Allocate with one ghost cell on each side */
    float *layer_base      = (float *)calloc(lsize + 2, sizeof(float));
    float *layer_copy_base = (float *)calloc(lsize + 2, sizeof(float));
    if (!layer_base || !layer_copy_base) {
        fprintf(stderr, "Error: Allocating the layer memory\n");
        exit(EXIT_FAILURE);
    }
    float *layer      = layer_base + 1;   /* layer[-1] and layer[lsize] are valid */
    float *layer_copy = layer_copy_base + 1;

    int left  = (rank > 0)        ? rank - 1 : MPI_PROC_NULL;
    int right = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    /* Keep sequential arithmetic order for better numerical match */
    float layer_size_f  = (float)layer_size;
    float threshold_div = THRESHOLD / layer_size_f;

    /* Flatten storm data once, outside storm loop, for cache friendliness */
    int max_ssize = 0;
    for (i = 0; i < num_storms; i++)
        if (storms[i].size > max_ssize) max_ssize = storms[i].size;
    float *s_energy = (float *)malloc(max_ssize * sizeof(float));
    int   *s_pos    = (int   *)malloc(max_ssize * sizeof(int));
    float *atten_lut = (float *)malloc((size_t)(layer_size + 1) * sizeof(float));
    if (!s_energy || !s_pos || !atten_lut) {
        fprintf(stderr, "Error: Allocating storm scratch buffers\n");
        exit(EXIT_FAILURE);
    }
    for (k = 1; k <= layer_size; k++) atten_lut[k] = sqrtf((float)k);

    struct { float val; int pos; } lmax_s, gmax_s;

    for (i = 0; i < num_storms; i++) {
        int ssize = storms[i].size;

        for (j = 0; j < ssize; j++) {
            s_pos[j]    = storms[i].posval[j * 2];
            s_energy[j] = ((float)storms[i].posval[j * 2 + 1] * 1000.0f) / layer_size_f;
        }

        /* ---------------------------------------------------------------
         * IMPACT: k-outer / j-inner, OMP on k.
         *
         * FIX 1: use direct  layer[k] += ek  (not a cell_energy accumulator).
         *   Sequential adds to layer[k] once per particle in j-order.
         *   CUDA does the same per thread.  Using a cell_energy variable
         *   changes the FP addition order and causes the value divergence.
         *   With direct +=, the j-order within each cell matches both.
         * --------------------------------------------------------------- */
        #pragma omp parallel for schedule(static) private(j)
        for (k = 0; k < lsize; k++) {
            int gk = k + start;
            for (j = 0; j < ssize; j++) {
                int   dist  = abs(s_pos[j] - gk) + 1;
                float ek    = s_energy[j] / atten_lut[dist];
                if (ek >= threshold_div || ek <= -threshold_div)
                    layer[k] += ek;
            }
        }

        /* ---------------------------------------------------------------
         * GHOST EXCHANGE #1 — before relaxation (pre-relax values).
         *   FIX (original code): the send/recv directions were swapped.
         *   Correct rule:
         *     send last element  → right neighbour's left ghost  layer[-1]
         *     send first element → left  neighbour's right ghost layer[lsize]
         * --------------------------------------------------------------- */
        MPI_Sendrecv(&layer[lsize - 1], 1, MPI_FLOAT, right, 0,
                     &layer[-1],        1, MPI_FLOAT, left,  0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&layer[0],    1, MPI_FLOAT, left,  1,
                     &layer[lsize], 1, MPI_FLOAT, right, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* ---------------------------------------------------------------
         * COPY — include ghost cells.
         *
         * FIX 2: original copy only covered k=0..lsize-1.
         *   The relaxation of the first/last local cell accesses
         *   layer_copy[-1] / layer_copy[lsize].  Without copying them
         *   those stay 0 (calloc), giving wrong averaged values at
         *   rank borders.  Error accumulates across storms.
         * --------------------------------------------------------------- */
        layer_copy[-1]    = layer[-1];
        memcpy(layer_copy, layer, (size_t)lsize * sizeof(float));
        layer_copy[lsize] = layer[lsize];

        /* RELAX — skip global positions 0 and layer_size-1 (matching sequential) */
        #pragma omp parallel for schedule(static)
        for (k = 0; k < lsize; k++) {
            int gk = k + start;
            if (gk == 0 || gk == layer_size - 1) continue;
            layer[k] = (layer_copy[k - 1] + layer_copy[k] + layer_copy[k + 1]) / 3.0f;
        }

        /* ---------------------------------------------------------------
         * GHOST EXCHANGE #2 — after relaxation (post-relax values).
         *
         * FIX 3: max-finding at a rank border compares layer[k] with
         *   layer[-1] / layer[lsize].  Without a second exchange those
         *   ghosts still hold pre-relax values → wrong local-max
         *   decisions at borders (critical for test_03..test_06).
         * --------------------------------------------------------------- */
        MPI_Sendrecv(&layer[lsize - 1], 1, MPI_FLOAT, right, 2,
                     &layer[-1],        1, MPI_FLOAT, left,  2,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&layer[0],    1, MPI_FLOAT, left,  3,
                     &layer[lsize], 1, MPI_FLOAT, right, 3,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* MAX FIND — thread-local then omp critical, single MPI_Allreduce */
        float lmax = 0.0f;
        int   lpos = 0;

        #pragma omp parallel
        {
            float tmax = 0.0f;
            int   tpos = 0;

            #pragma omp for schedule(static) nowait
            for (k = 0; k < lsize; k++) {
                int gk = k + start;
                if (gk == 0 || gk == layer_size - 1) continue;
                if (layer[k] > layer[k - 1] && layer[k] > layer[k + 1]) {
                    float v = layer[k];
                    if (v > tmax || (v == tmax && gk < tpos)) {
                        tmax = v;
                        tpos = gk;
                    }
                }
            }
            #pragma omp critical
            {
                if (tmax > lmax || (tmax == lmax && tpos < lpos)) {
                    lmax = tmax;
                    lpos = tpos;
                }
            }
        }

        /* Single collective (value + global position), deterministic tie-break */
        lmax_s.val  = lmax;
        lmax_s.pos  = lpos;
        MPI_Allreduce(&lmax_s, &gmax_s, 1, MPI_FLOAT_INT, MPI_MAXLOC, MPI_COMM_WORLD);

        if (rank == 0) {
            maximum[i]   = gmax_s.val;
            positions[i] = gmax_s.pos;
        }
    }

    free(s_energy);
    free(s_pos);
    free(atten_lut);
    free(layer_base);
    free(layer_copy_base);
}