#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "energy_storms.h"

/* THIS FUNCTION CAN BE MODIFIED */
/* Function to update a single position of the layer */
__device__ float update( int affected_position, int layer_size, float energy, int contact_position ) {

    /* 1. Compute the absolute value of the distance between the
        impact position and the k-th position of the layer */
    int distance = abs(contact_position - affected_position) + 1;

    /* 2. Square root of the distance */
    /* NOTE: Real world atenuation typically depends on the square of the distance.
       We use here a tailored equation that affects a much wider range of cells */
    float atenuacion = sqrtf( (float)distance );

    /* 3. Compute attenuated energy */
    float energy_k = energy / layer_size / atenuacion;

    /* 4. Do not add if its absolute value is lower than the threshold */
    float threshold = THRESHOLD / layer_size;
    return (fabsf(energy_k) >= threshold) ? energy_k : 0.0f; // float abs()
}

__global__ void bomb(float* layer_d, int layer_size, int* posval_d, int storm_size) {

    // Grid-stride loop for handling storm > threads
    int grid_stride = blockDim.x * gridDim.x;
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    float energy;
    int contact_position;

    // Each thread processes multiple contact points of the layer in a grid-stride loop
    for (int i = thread_id; i < layer_size; i += grid_stride) {

        // // Variable that holds all particle energy impacts
        // float sum_energy = 0.0f;

        // For each contact point i, compute all storm impacts and update the contact point value if affected by it
        for (int j = 0; j < storm_size; j++) {
            energy = (float)posval_d[j*2+1] * 1000;
            contact_position = posval_d[j*2];
            // Maybe I can modify it so that only if affected then do the update
            // Or save all storm impacts before and update after all calculations (Not sure if better)
            // sum_energy += update(i, layer_size, energy, contact_position);
            layer_d[i] += update(i, layer_size, energy, contact_position);
        }

        // After computing all impacts that affect contact point i, update his value
        // layer_d[i] += sum_energy;

    }
}

__global__ void relax(float* layer_d, const float* layer_copy_d, int layer_size) {

    // Grid-stride loop for handling layers > threads
    int grid_stride = blockDim.x * gridDim.x;
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread processes multiple contact points of the layer in a grid-stride loop
    for (int i = thread_id; i < layer_size - 1; i += grid_stride) {
        layer_d[i] = (layer_copy_d[i-1] + layer_copy_d[i] + layer_copy_d[i+1]) / 3.0f;
    }
}

void core(int layer_size, int num_storms, Storm *storms, float *maximum, int *positions) {

    int i, k;

    /* 3.1. Obtain grid and block dimensions */
    dim3 blockDim(256, 1, 1); // 256 threads per block
    dim3 gridDim((layer_size + blockDim.x - 1) / blockDim.x, 1, 1); // -1 to round the blocks needed

    /* 3.2. Allocate memory for the layer and initialize to zero */
    float *layer = (float *) malloc( sizeof(float) * layer_size );
    if ( layer == NULL ) {
        fprintf(stderr,"Error: Allocating the layer memory\n");
        exit( EXIT_FAILURE );
    }
    for( k=0; k<layer_size; k++ ) layer[k] = 0.0f;
    float *layer_d;
    cudaMalloc((void **)&layer_d, sizeof(float) * layer_size);
    cudaMemcpy(layer_d, layer, sizeof(float) * layer_size, cudaMemcpyHostToDevice);

    /* 4 Simulate for every storm */
    for ( i=0; i<num_storms; i++ ) {

        /* 4.1. Allocate memory for the posval and initialize to zero*/
        int *posval_d;
        cudaMalloc((void **)&posval_d, sizeof(int) * storms[i].size * 2);
        cudaMemcpy(posval_d, storms[i].posval, sizeof(int) * storms[i].size * 2, cudaMemcpyHostToDevice);

        /* 4.2. Simulate impacts energy to layer cells */
        bomb<<<gridDim, blockDim>>>(layer_d, layer_size, posval_d, storms[i].size);

        /* 4.3. Allocate memory for the layer_copy */
        float *layer_copy_d;
        cudaMalloc((void **)&layer_copy_d, sizeof(float) * layer_size);
        cudaMemcpy(layer_copy_d, layer_d, sizeof(float) * layer_size, cudaMemcpyDeviceToDevice);

        /* 4.4. Simulate energy relaxation */
        relax<<<gridDim, blockDim>>>(layer_d, layer_copy_d, layer_size);

        /* 4.5 Find maximum energy and its position */
        cudaMemcpy(layer, layer_d, sizeof(float) * layer_size, cudaMemcpyDeviceToHost);
        for( k=1; k<layer_size-1; k++ ) {
            /* Check it only if it is a local maximum */
            if ( layer[k] > layer[k-1] && layer[k] > layer[k+1] ) {
                if ( layer[k] > maximum[i] ) {
                    maximum[i] = layer[k];
                    positions[i] = k;
                }
            }
        }
        cudaFree(posval_d);
        cudaFree(layer_copy_d);
    }
    cudaFree(layer_d);
}