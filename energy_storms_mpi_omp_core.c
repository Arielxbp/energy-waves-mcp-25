#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "energy_storms.h"
#include <omp.h>
#include <mpi.h>

void core(int layer_size, int num_storms, Storm *storms, float *maximum, int *positions) {
    int i, j, k, size, rank;
    /* 3. Allocate memory for the layer and initialize to zero */

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int int_chunk_size = layer_size / size;
    int rem = layer_size % size;
    int chunk_size = int_chunk_size + (rank < rem ? 1 : 0);
    int start = rank*int_chunk_size + (rank < rem ? rank : rem);

    float *layer_base      = (float *)malloc((chunk_size + 2) * sizeof(float));
    float *layer_copy_base = (float *)malloc((chunk_size + 2) * sizeof(float));
    if (!layer_base || !layer_copy_base) {
        fprintf(stderr, "Error: Allocating the layer memory\n");
        exit(EXIT_FAILURE);
    }

    #pragma omp parallel for schedule(static)
    for (k = 0; k < chunk_size + 2; k++) {
        layer_base[k] = 0.0f;
        layer_copy_base[k] = 0.0f;
    }

    float *layer      = layer_base + 1;
    float *layer_copy = layer_copy_base + 1;

    struct {
        float max;
        int rank;
    } global_max, local_max;

    float *sqrt_distances = (float *)malloc((size_t)(layer_size*2)*sizeof(float));
    for (k = 1; k <= layer_size; k++) sqrt_distances[k] = 1.0f/sqrtf((float)k);

    int max_storm_size = 0;
    for (i = 0; i < num_storms; i++) {
        if (storms[i].size > max_storm_size) {
            max_storm_size = storms[i].size;
        }
    }

    float *precalc_energy = (float *)malloc(max_storm_size*sizeof(float));
    int *precalc_positions = (int*)malloc(max_storm_size*sizeof(int));

    float layer_size_f = (float) layer_size;
    float threshold = THRESHOLD / layer_size_f;

    //MPI_Scatterv(layer, sizes, displs, MPI_FLOAT, local_layer, sizes[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
    /* 4. Storms simulation */
    for( i=0; i<num_storms; i++) {
        /* 4.1. Add impacts energies to layer cells */
        /* For each particle */
        
        for( j=0; j<storms[i].size; j++ ){
            precalc_energy[j] = ((float)storms[i].posval[j*2+1] * 1000)/layer_size_f;
            precalc_positions[j] = storms[i].posval[j*2]; 
        }
        

        #pragma omp parallel private(j, k)
        {
            int threads = omp_get_num_threads();
            int tid = omp_get_thread_num();
            int int_tchunk_size = chunk_size / threads;
            int trem = chunk_size % threads;
            int tchunk_size = int_tchunk_size + (tid < trem ? 1 : 0);
            int tstart = tid*int_tchunk_size + (tid < trem ? tid : trem);
            int global_start = start + tstart; 

            float cell_vals[tchunk_size];
            for (k = 0; k < tchunk_size; k++) {
                cell_vals[k] = layer[tstart + k];
            }

            
            for (j = 0; j < storms[i].size; j++) {
                for (k = 0; k<tchunk_size; k++){
                    int distance = abs(precalc_positions[j] - k - global_start) + 1;

                    float energy_k = precalc_energy[j] * sqrt_distances[distance];
                    cell_vals[k] += (fabsf(energy_k) >= threshold) ? energy_k : 0.0f;
                }
            }

            for (k = 0; k < tchunk_size; k++) {
                layer[tstart + k] = cell_vals[k];
            }

        }
        
        /*printf("Rank %d: Storm %d layer after impacts:\n", rank, i);
        printf("[");
        for (int i = 0; i < chunk_size-1; i++) {
            printf("%f,", layer[i]);
        }
        printf("%f]\n", layer[chunk_size-1]);
        */
        /* 4.2. Energy relaxation between storms */

        /* 4.2.1. Copy current layer to ancillary array */
        
        /* 4.2.2. Update layer using the ancillary values.
                  Skip updating the first and last positions */
        /* Ghost cell exchange — corrected direction */
        int left  = (rank > 0)        ? rank-1 : MPI_PROC_NULL;
        int right = (rank < size - 1) ? rank+1 : MPI_PROC_NULL;

        /* Send rightmost element right, receive into right ghost cell */
        MPI_Sendrecv(&layer[chunk_size-1], 1, MPI_FLOAT, right, 0,
                    &layer[-1],           1, MPI_FLOAT, left,  0,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        /* Send leftmost element left, receive into left ghost cell */
        MPI_Sendrecv(&layer[0],          1, MPI_FLOAT, left,  1,
                    &layer[chunk_size], 1, MPI_FLOAT, right, 1,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        #pragma omp parallel for schedule(static) private(k)
        for (k = 0; k < chunk_size; k++)
            layer_copy[k] = layer[k];

        #pragma omp parallel for schedule(static) private(k)
        for (k = 0; k < chunk_size; k++) {
            int gk = k + start;
            if (gk > 0 && gk < layer_size - 1)   /* correct boundary: || not && */
                layer[k] = (layer_copy[k-1] + layer_copy[k] + layer_copy[k+1]) / 3.0f;
        }

        float lmax = -__FLT_MAX__;
        int lpos = 0;

        for (k = 0; k < chunk_size; k++) {
            int gk = k + start;
            if (gk > 0 && gk < layer_size - 1) {
                if (layer[k] > layer[k-1] && layer[k] > layer[k+1] && layer[k] > lmax) {
                    lmax = layer[k]; lpos = gk;
                }
            }
        }


        local_max.max = lmax;
        local_max.rank = rank;
        global_max.max = 0.0f;
        global_max.rank = 0;
        //printf("Rank %d: Storm %d max: %f at position %d\n", rank, i, lmax, lpos);

        /* 4.4. Reduce to get the global maximum and its position */
        MPI_Allreduce(&local_max, &global_max, 1, MPI_FLOAT_INT, MPI_MAXLOC, MPI_COMM_WORLD);
        int global_max_pos = (rank == global_max.rank) ? lpos : 0;
        MPI_Bcast(&global_max_pos, 1, MPI_INT, global_max.rank, MPI_COMM_WORLD);
        if (rank == 0)
        {
            maximum[i] = global_max.max;
            positions[i] = global_max_pos;
        }
    }
    //MPI_Gatherv(local_layer, sizes[rank], MPI_FLOAT, layer, sizes, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
    free(layer_base);
    free(layer_copy_base);
    free(sqrt_distances);
    free(precalc_energy);
    free(precalc_positions);
}