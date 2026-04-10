#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "energy_storms.h"
#include <omp.h>
#include <sys/time.h>

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
        #pragma omp parallel for schedule(static) private(j, k)
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

void core(int layer_size, int num_storms, Storm *storms, float *maximum, int *positions);

/*
 * MAIN PROGRAM
 */
int main(int argc, char *argv[]) {
    int i;

    /* 1.1. Read arguments */
    if (argc<3) {
        fprintf(stderr,"Usage: %s <size> <storm_1_file> [ <storm_i_file> ] ... \n", argv[0] );
        exit( EXIT_FAILURE );
    }

    int layer_size = atoi( argv[1] );
    int num_storms = argc-2;
    Storm* storms = (Storm*) malloc(sizeof(Storm)*num_storms);

    /* 1.2. Read storms information */
    for( i=2; i<argc; i++ ) 
        storms[i-2] = read_storm_file( argv[i] );

    /* 1.3. Intialize maximum levels to zero */
    float maximum[ num_storms ];
    int positions[ num_storms ];
    for (i=0; i<num_storms; i++) {
        maximum[i] = 0.0f;
        positions[i] = 0;
    }

    /* 2. Begin time measurement */
    double ttotal = cp_Wtime();

    /* START: Do NOT optimize/parallelize the code of the main program above this point */
    core(layer_size, num_storms, storms, maximum, positions);
    /* END: Do NOT optimize/parallelize the code below this point */

    /* 5. End time measurement */
    ttotal = cp_Wtime() - ttotal;

    /* 6. DEBUG: Plot the result (only for layers up to 35 points) */
    #ifdef DEBUG
    debug_print( layer_size, layer, positions, maximum, num_storms );
    #endif

    /* 7. Results output, used by the Tablon online judge software */
    printf("\n");
    /* 7.1. Total computation time */
    printf("Time: %lf\n", ttotal );
    /* 7.2. Print the maximum levels */
    printf("Result:");
    for (i=0; i<num_storms; i++)
    printf(" %d %f", positions[i], maximum[i] );
    printf("\n");

    /* 8. Free resources */    
    for( i=0; i<argc-2; i++ )
        free( storms[i].posval );
    free(storms);
    /* 9. Program ended successfully */
    return 0;
}
