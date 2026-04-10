#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "energy_storms.h"
#include <omp.h>
#include <mpi.h>

void core(int layer_size, int num_storms, Storm *storms, float *maximum, int *positions) {
    int i, j, k, size, rank;

    float *layer = (float *)malloc((layer_size) * sizeof(float));
    float *layer_copy= (float *)malloc((layer_size) * sizeof(float));
    if (!layer|| !layer_copy) {
        fprintf(stderr, "Error: Allocating the layer memory\n");
        exit(EXIT_FAILURE);
    }

    #pragma omp parallel for schedule(static)
    for (k = 0; k < layer_size; k++) {
        layer[k] = 0.0f;
        layer_copy[k] = 0.0f;
    }

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

    /*storms simulation*/
    for( i=0; i<num_storms; i++) {

        /*precalculation of energies and positions of particles*/
        for( j=0; j<storms[i].size; j++ ){
            precalc_energy[j] = ((float)storms[i].posval[j*2+1] * 1000)/layer_size_f;
            precalc_positions[j] = storms[i].posval[j*2]; 
        }
        
        /*bombardment phase: inlining of update function + inner loop parallelization*/
        #pragma omp parallel for schedule(static) private(j, k) collapse(2)
        for (k = 0; k<layer_size; k++) {
            for (j = 0; j < storms[i].size; j++){
                int distance = abs(precalc_positions[j] - k) + 1;
                float energy_k = precalc_energy[j] / sqrt_distances[distance];
                layer[k] += (fabsf(energy_k) >= threshold) ? energy_k : 0.0f;
            }
        }
        
        /*printf("[");
        for (int i = 0; i < chunk_size-1; i++) {
            printf("%f,", layer[i]);
        }
        printf("%f]\n", layer[chunk_size-1]);
        */

        /*relaxation phase*/
        #pragma omp parallel for schedule(static) private(k)
        for (k = 0; k < layer_size; k++) {
            layer_copy[k] = layer[k];
        }

        #pragma omp parallel for schedule(static) private(k)
        for (k = 1; k < layer_size - 1; k++) {
            layer[k] = (layer_copy[k-1] + layer_copy[k] + layer_copy[k+1]) / 3.0f;
        }

        /*maximum search*/
        float lmax = -__FLT_MAX__;
        int lpos = 0;

        for (k = 1; k < layer_size - 1; k++) {
            if (layer[k] > layer[k-1] && layer[k] > layer[k+1] && layer[k] > lmax) {
                    lmax = layer[k]; lpos = k;
                }
            }
        //printf("Rank %d: Storm %d max: %f at position %d\n", rank, i, lmax, lpos);

            maximum[i] = lmax;
            positions[i] = lpos;
    }
    
    free(layer);
    free(layer_copy);
    free(sqrt_distances);
    free(precalc_energy);
    free(precalc_positions);
}