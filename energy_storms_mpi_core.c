#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "energy_storms.h"
#include <omp.h>
#include <mpi.h>

void core(int layer_size, int num_storms, Storm *storms, float *maximum, int *positions) {
    int i, j, k, size, rank;
    
    /*layer redistribution between ranks*/
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int int_chunk_size = layer_size / size;
    int rem = layer_size % size;
    int chunk_size = int_chunk_size + (rank < rem ? 1 : 0);
    int start = rank*int_chunk_size + (rank < rem ? rank : rem);

    /*local layer allocation with border cells*/
    float *layer_base = (float *)malloc((chunk_size+2)*sizeof(float));
    float *layer_copy_base = (float *)malloc((chunk_size+2)*sizeof(float));
    if (!layer_base || !layer_copy_base) {
        fprintf(stderr, "Error: Allocating the layer memory\n");
        exit(EXIT_FAILURE);
    }

    for (k = 0; k < chunk_size+2; k++) {
        layer_base[k] = 0.0f;
        layer_copy_base[k] = 0.0f;
    }

    float *layer = layer_base + 1;
    float *layer_copy = layer_copy_base + 1;

    struct {
        float max;
        int rank;
    } global_max, local_max;

    /*precalculation of distances square roots*/
    float *sqrt_distances = (float *)malloc((size_t)(layer_size*2)*sizeof(float));
    for (k = 1; k <= layer_size; k++) sqrt_distances[k] = sqrtf((float)k);

    int max_storm_size = 0;
    for (i = 0; i < num_storms; i++) {
        if (storms[i].size > max_storm_size) {
            max_storm_size = storms[i].size;
        }
    }

    float *precalc_energy = (float *)malloc(max_storm_size*sizeof(float));
    int *precalc_positions = (int*)malloc(max_storm_size*sizeof(int));

    float layer_size_f = (float) layer_size;
    float threshold = THRESHOLD/layer_size_f;

    //MPI_Scatterv(layer, sizes, displs, MPI_FLOAT, local_layer, sizes[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
    /*storms simulation*/
    for( i=0; i<num_storms; i++) {

        /*precalculation of energies and positions of particles*/
        for( j=0; j<storms[i].size; j++ ){
            precalc_energy[j] = ((float)storms[i].posval[j*2+1] * 1000)/layer_size_f;
            precalc_positions[j] = storms[i].posval[j*2]; 
        }

        /*bombardment phase: inlining of update function*/
        for (k = 0; k<chunk_size; k++) {
            int global_k = k+start;
            float cell_val = layer[k];
            for (j = 0; j < storms[i].size; j++) {
                int distance = abs(precalc_positions[j] - k - start) + 1;

                float energy_k = precalc_energy[j] / sqrt_distances[distance];
                cell_val += (fabsf(energy_k) >= threshold) ? energy_k : 0.0f;
            }
            layer[k] = cell_val;
        }
        
        /*printf("[");
        for (int i = 0; i < chunk_size-1; i++) {
            printf("%f,", layer[i]);
        }
        printf("%f]\n", layer[chunk_size-1]);
        */

        /*border cell exchange*/
        int left  = (rank > 0) ? rank-1 : MPI_PROC_NULL;
        int right = (rank < size - 1) ? rank+1 : MPI_PROC_NULL;
        MPI_Sendrecv(&layer[chunk_size-1], 1, MPI_FLOAT, right, 0, &layer[-1], 1, MPI_FLOAT, left,  0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&layer[0],1, MPI_FLOAT, left, 1, &layer[chunk_size], 1, MPI_FLOAT, right, 1,MPI_COMM_WORLD, MPI_STATUS_IGNORE);


        /*relaxation phase*/
        for (k = 0; k < chunk_size; k++)
            layer_copy[k] = layer[k];

        for (k = 0; k < chunk_size; k++) {
            int gk = k + start;
            if (gk > 0 && gk < layer_size - 1)  
                layer[k] = (layer_copy[k-1] + layer_copy[k] + layer_copy[k+1]) / 3.0f;
        }

        /*maximum search*/

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

        /*reduction to get global maximum and position*/
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