#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "energy_storms.h"

void core(int layer_size, int num_storms, Storm *storms, float *maximum, int *positions, float* layer, float* layer_copy) {
    int i, j, k;

    float *sqrt_distances = (float *)malloc((size_t)(layer_size * 5) * sizeof(float));
    if (!sqrt_distances) {
        fprintf(stderr, "Error: Allocating the sqrt_distances memory\n");
        exit(EXIT_FAILURE);
    }
    for (k = 1; k <= layer_size; k++) sqrt_distances[k] = sqrtf((float)k);

    int max_storm_size = 0;
    for (i = 0; i < num_storms; i++) {
        if (storms[i].size > max_storm_size)
            max_storm_size = storms[i].size;
    }

    float *precalc_energy = (float*)malloc(max_storm_size * sizeof(float));
    int   *precalc_positions = (int*)malloc(max_storm_size * sizeof(int));
    if (!precalc_energy || !precalc_positions) {
        fprintf(stderr, "Error: Allocating the precalculation memory\n");
        exit(EXIT_FAILURE);
    }

    float layer_size_f = (float)layer_size;
    float threshold = THRESHOLD / layer_size_f;

    for (i = 0; i < num_storms; i++) {

        for (j = 0; j < storms[i].size; j++) {
            precalc_energy[j] = ((float)storms[i].posval[j * 2 + 1] * 1000) / layer_size_f;
            precalc_positions[j] = storms[i].posval[j * 2];
        }

        for (j = 0; j < storms[i].size; j++) {
            for (k = 0; k < layer_size; k++) {
                int distance = abs(precalc_positions[j] - k) + 1;
                float energy_k = precalc_energy[j] / sqrt_distances[distance];
                layer[k] += (fabsf(energy_k) >= threshold) ? energy_k : 0.0f;
            }
        }

        for (k = 0; k < layer_size; k++)
            layer_copy[k] = layer[k];

        for (k = 1; k < layer_size - 1; k++)
            layer[k] = (layer_copy[k-1] + layer_copy[k] + layer_copy[k+1]) / 3.0f;

        for (k = 1; k < layer_size - 1; k++) {
            if (layer[k] > layer[k-1] && layer[k] > layer[k+1]) {
                if (layer[k] > maximum[i]) {
                    maximum[i]  = layer[k];
                    positions[i] = k;
                }
            }
        }
    }

    free(sqrt_distances);
    free(precalc_energy);
    free(precalc_positions);
}