#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "energy_storms.h"
#include <omp.h>
#include <mpi.h>

/* THIS FUNCTION CAN BE MODIFIED */
/* Function to update a single position of the layer */
static float update(int layer_size, int k, int pos, float energy ) {
    /* 1. Compute the absolute value of the distance between the
        impact position and the k-th position of the layer */
    int distance = pos - k;
    if ( distance < 0 ) distance = - distance;

    /* 2. Impact cell has a distance value of 1 */
    distance = distance + 1;

    /* 3. Square root of the distance */
    /* NOTE: Real world atenuation typically depends on the square of the distance.
       We use here a tailored equation that affects a much wider range of cells */
    float atenuacion = sqrtf( (float)distance );

    /* 4. Compute attenuated energy */
    float energy_k = energy / layer_size / atenuacion;

    /* 5. Do not add if its absolute value is lower than the threshold */
    return ( energy_k >= THRESHOLD / layer_size || energy_k <= -THRESHOLD / layer_size ) ? energy_k : 0.0f;
    
}


void core(int layer_size, int num_storms, Storm *storms, float *maximum, int *positions) {
    int i, j, k, size, rank;
    /* 3. Allocate memory for the layer and initialize to zero */
    float *layer = (float *)malloc( sizeof(float) * layer_size );

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int int_chunk_size = layer_size / size;
    int rem = layer_size % size;
    int sizes[size];
    int displs[size];
    displs[0] = 0;

    for (int i = 0; i < size; i++) {
        sizes[i] = int_chunk_size;
        if (rem > 0) {
            sizes[i]++;
            rem--;
        }
        if (i > 0) {
            displs[i] = displs[i-1] + sizes[i-1];
        }
    }
    int chunk_size = sizes[rank];

    float *local_layer = (float *)malloc( sizeof(float) * (chunk_size) );
    float *local_layer_copy = (float *)malloc( sizeof(float) * (chunk_size) );
    
    if ( layer == NULL) {
        fprintf(stderr,"Error: Allocating the layer memory\n");
        exit( EXIT_FAILURE );
    }

    for( k=0; k<layer_size; k++ ) {        
        layer[k] = 0.0f;
        if (k < chunk_size) {
            local_layer[k] = 0.0f;
            local_layer_copy[k] = 0.0f;
        }   
    }

    struct max_pos {
        float max;
        int pos;
    } global_max, local_max;

    MPI_Scatterv(layer, sizes, displs, MPI_FLOAT, local_layer, sizes[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
    /* 4. Storms simulation */
    for( i=0; i<num_storms; i++) {
        /* 4.1. Add impacts energies to layer cells */
        /* For each particle */
        for( j=0; j<storms[i].size; j++ ) {
            for( k=0; k<chunk_size; k++ ) {
            /* For each cell in the layer */
                /* Update the energy value for the cell */
                local_layer[k] += update(layer_size, k+displs[rank], storms[i].posval[j*2], (float)storms[i].posval[j*2+1] * 1000);
            }
        }
        /*
        printf("Rank %d: Storm %d layer after impacts:\n", rank, i);
        printf("[");
        for (int i = 0; i < chunk_size-1; i++) {
            printf("%f,", local_layer[i]);
        }
        printf("%f]\n", local_layer[chunk_size-1]);
        */
        /* 4.2. Energy relaxation between storms */

        /* 4.2.1. Copy current layer to ancillary array */
        #pragma omp parallel for schedule(static)
        for( k=0; k<chunk_size; k++ )
            local_layer_copy[k] = local_layer[k];

        
        /* 4.2.2. Update layer using the ancillary values.
                  Skip updating the first and last positions */
        MPI_Status status;
        if (rank > 0) {
            if (rank < size-1)
            {
                float left, right;
                MPI_Sendrecv(&local_layer[0], 1, MPI_FLOAT, rank-1, 0, &left, 1, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
                local_layer[0] = ( left + local_layer_copy[0] + local_layer_copy[1] ) / 3;
                
                MPI_Sendrecv(&local_layer[chunk_size-1], 1, MPI_FLOAT, rank+1, 0, &right, 1, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, &status);
                local_layer[chunk_size-1] = ( local_layer_copy[chunk_size-2] + local_layer_copy[chunk_size-1] + right ) / 3;
            } else {
                float left;
                MPI_Sendrecv(&local_layer[0], 1, MPI_FLOAT, rank-1, 0, &left, 1, MPI_FLOAT, rank-1, 0, MPI_COMM_WORLD, &status);
                local_layer[0] = ( left + local_layer_copy[0] + local_layer_copy[1] ) / 3;
            }
        } else {
            if (size > 1) {
                float right;
                MPI_Sendrecv(&local_layer[chunk_size-1], 1, MPI_FLOAT, rank+1, 0, &right, 1, MPI_FLOAT, rank+1, 0, MPI_COMM_WORLD, &status);
                local_layer[chunk_size-1] = ( local_layer_copy[chunk_size-2] + local_layer_copy[chunk_size-1] + right ) / 3;
            }
        }
        /*(
        printf("Rank %d: Storm %d layer after exchange and border updates:\n", rank, i);
        printf("[");
        for (int i = 0; i < chunk_size-1; i++) {
            printf("%f,", local_layer[i]);
        }
        printf("%f]\n", local_layer[chunk_size-1]);
*/
        #pragma omp parallel for schedule(static)
        for( k=1; k<chunk_size-1; k++ )
            local_layer[k] = ( local_layer_copy[k-1] + local_layer_copy[k] + local_layer_copy[k+1] ) / 3;        
        
        
        /*printf("Rank %d: Storm %d layer after relaxation:\n", rank, i);
        printf("[");
        for (int i = 0; i < chunk_size-1; i++) {
            printf("%f,", local_layer[i]);
        }
        printf("%f]\n", local_layer[chunk_size-1]);
        MPI_Barrier(MPI_COMM_WORLD);*/

	local_max.max = local_layer[0];
        local_max.pos = displs[rank];
        /* 4.3. Locate the maximum value in the layer, and its position */
        for( k=0; k<chunk_size; k++ ) {
            /* Check it only if it is a local maximum */
            if (local_layer[k] >= local_max.max) {
                local_max.max = local_layer[k];
                local_max.pos = k + displs[rank];
            }
        }
        //printf("Rank %d: Storm %d before reduction local max %f at position %d\n", rank, i, max, pos);
        

        //printf("Rank %d: Storm %d local max %f at position %d\n", rank, i, max, pos);
        /* 4.4. Reduce to get the global maximum and its position */
        MPI_Reduce(&local_max, &global_max, 1, MPI_FLOAT_INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            maximum[i] = global_max.max;
            positions[i] = global_max.pos;
        }
    }
    MPI_Gatherv(local_layer, sizes[rank], MPI_FLOAT, layer, sizes, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
}
