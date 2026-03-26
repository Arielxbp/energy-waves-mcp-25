#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
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

    /* Allocate with ghost cells at [-1] and [lsize] */
    float *layer      = (float *)calloc(lsize + 2, sizeof(float)) + 1;
    float *layer_copy = (float *)calloc(lsize + 2, sizeof(float)) + 1;
    if (!layer || !layer_copy) {
        fprintf(stderr, "Error: Allocating the layer memory\n");
        exit(EXIT_FAILURE);
    }

    int left  = (rank > 0)          ? rank - 1 : MPI_PROC_NULL;
    int right = (rank < size - 1)   ? rank + 1 : MPI_PROC_NULL;

    /* Struct for MPI_MAXLOC reduction */
    struct { float max; int rank; } global_max, local_max;

    /* Pre-allocated local posval buffers to avoid pointer chasing */
    int max_storm_size = 0;
    for (i = 0; i < num_storms; i++)
        if (storms[i].size > max_storm_size)
            max_storm_size = storms[i].size;

    float *energies   = (float *)malloc(max_storm_size * sizeof(float));
    int   *positions_ = (int   *)malloc(max_storm_size * sizeof(int));

    for (i = 0; i < num_storms; i++) {

        int ssize = storms[i].size;

        /* --- Cache-friendly copies of storm data --- */
        for (j = 0; j < ssize; j++) {
            positions_[j] = storms[i].posval[j * 2];
            energies[j]   = (float)storms[i].posval[j * 2 + 1] * 1000.0f;
        }

        /* -----------------------------------------------------------
         * FIX 1: Swap loops — k is OUTER (parallelisable), j INNER.
         *         Each thread owns its own k → zero race conditions.
         * ----------------------------------------------------------- */
        float threshold = THRESHOLD / layer_size;

        #pragma omp parallel for schedule(static) private(j)
        for (k = 0; k < lsize; k++) {
            float cell_energy = 0.0f;
            int   gk = k + start;
            for (j = 0; j < ssize; j++) {
                int   distance  = abs(positions_[j] - gk) + 1;
                float atten     = sqrtf((float)distance);
                float ek        = energies[j] / layer_size / atten;
                if (ek >= threshold || ek <= -threshold)
                    cell_energy += ek;
            }
            layer[k] += cell_energy;
        }

        /* --- Ghost cell exchange (unchanged, already correct) --- */
        MPI_Sendrecv(&layer[lsize - 1], 1, MPI_FLOAT, right, 0,
                     &layer[-1],        1, MPI_FLOAT, left,  0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&layer[0],    1, MPI_FLOAT, left,  1,
                     &layer[lsize], 1, MPI_FLOAT, right, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* -----------------------------------------------------------
         * FIX 2: Parallelize the copy and relax passes.
         * ----------------------------------------------------------- */
        #pragma omp parallel for schedule(static)
        for (k = 0; k < lsize; k++)
            layer_copy[k] = layer[k];

        #pragma omp parallel for schedule(static)
        for (k = 0; k < lsize; k++) {
            int gk = k + start;
            /* Preserve first and last global positions unchanged */
            if (gk == 0 || gk == layer_size - 1) continue;
            layer[k] = (layer_copy[k - 1] + layer_copy[k] + layer_copy[k + 1]) / 3.0f;
        }

        /* -----------------------------------------------------------
         * FIX 2 (cont.): Thread-local max, merged with omp critical.
         *         Avoids the overhead of a reduction clause on structs.
         * ----------------------------------------------------------- */
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
            {
                if (tmax > lmax) { lmax = tmax; lpos = tpos; }
            }
        }

        /* -----------------------------------------------------------
         * FIX 3: Single MPI_Allreduce with MPI_MAXLOC, then one Bcast
         *         for the position.  Eliminates the extra Reduce + Bcast
         *         round-trip that rank 0 was doing.
         * ----------------------------------------------------------- */
        local_max.max  = lmax;
        local_max.rank = rank;

        MPI_Allreduce(&local_max, &global_max, 1, MPI_FLOAT_INT,
                      MPI_MAXLOC, MPI_COMM_WORLD);

        /* Only the owner of the global max knows the exact position */
        int global_pos = (rank == global_max.rank) ? lpos : 0;
        MPI_Bcast(&global_pos, 1, MPI_INT, global_max.rank, MPI_COMM_WORLD);

        /* Every rank can now write — but only rank 0 output is read */
        if (rank == 0) {
            maximum[i]   = global_max.max;
            positions[i] = global_pos;
        }
    }

    free(energies);
    free(positions_);
    free(layer - 1);
    free(layer_copy - 1);
}