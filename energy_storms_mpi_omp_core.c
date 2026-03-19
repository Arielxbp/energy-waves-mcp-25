#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "energy_storms.h"
#include <omp.h>
#include <mpi.h>

/* THIS FUNCTION CAN BE MODIFIED */
/* Function to update a single position of the layer */
static float update(int layer_size, int k, int pos, float energy, float threshold_scaled) {
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
    return ( energy_k >= threshold_scaled || energy_k <= -threshold_scaled ) ? energy_k : 0.0f;
    
}


void core(int layer_size, int num_storms, Storm *storms, float *maximum, int *positions) {
    int i, j, k, size, rank;
    /* 3. Allocate memory for the layer and initialize to zero */

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    const float threshold_scaled = THRESHOLD / layer_size;
    
    int int_chunk_size = layer_size / size;
    int rem = layer_size % size;
    int* sizes = (int*)malloc(size * sizeof(int));
    int* displs = (int*)malloc(size * sizeof(int));
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

    float *layer = (float *)malloc( sizeof(float) * (chunk_size) );
    float *layer_copy = (float *)malloc( sizeof(float) * (chunk_size) );
    
    if ( layer == NULL || layer_copy == NULL ) {
        fprintf(stderr,"Error: Allocating the layer memory\n");
        exit( EXIT_FAILURE );
    }

    for( k=0; k<chunk_size; k++ ) {        
        layer[k] = 0.0f;
        layer_copy[k] = 0.0f;
    }

    struct max_pos {
        float max;
        int pos;
    } global_max, local_max;

    //MPI_Scatterv(layer, sizes, displs, MPI_FLOAT, local_layer, sizes[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
    /* 4. Storms simulation */
    for( i=0; i<num_storms; i++) {
        /* 4.1. Add impacts energies to layer cells */
        /* For each particle */
        #pragma omp parallel for schedule(static) collapse(2)
        for( k=0; k<chunk_size; k++ ){
            for( j=0; j<storms[i].size; j++ ){
            /* For each cell in the layer */
                /* Update the energy value for the cell */
                layer[k] += update(layer_size, k+displs[rank], storms[i].posval[j*2], (float)storms[i].posval[j*2+1] * 1000, threshold_scaled);
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
            layer_copy[k] = layer[k];

        
        /* 4.2.2. Update layer using the ancillary values.
                  Skip updating the first and last positions */
        MPI_Status stats[4];
        MPI_Request reqs[4];
        int nreqs = 0;
        float tleft, tright;
         
        if (rank > 0) {
            MPI_Irecv(&tleft, 1, MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD, &reqs[nreqs++]);
        }
        if (rank < size-1) {
            MPI_Irecv(&tright, 1, MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD, &reqs[nreqs++]);
        }
            
        if (rank > 0) {
           MPI_Isend(&layer[0], 1, MPI_FLOAT, rank-1, 1, MPI_COMM_WORLD, &reqs[nreqs++]);
        }
        if (rank < size-1) {
            MPI_Isend(&layer[chunk_size-1], 1, MPI_FLOAT, rank+1, 1, MPI_COMM_WORLD, &reqs[nreqs++]);
        }
        
        /*(
        printf("Rank %d: Storm %d layer after exchange and border updates:\n", rank, i);
        printf("[");
        for (int i = 0; i < chunk_size-1; i++) {
            printf("%f,", local_layer[i]);
        }
        printf("%f]\n", local_layer[chunk_size-1]);
*/
        if (chunk_size > 2) {
            #pragma omp parallel for schedule(static)
            for( k=1; k<chunk_size-1; k++ ) {
                layer[k] = ( layer_copy[k-1] + layer_copy[k] + layer_copy[k+1] ) / 3; 
            }
        }
        MPI_Waitall(nreqs, reqs, stats);
        if (rank > 0) {
            layer[0] = (layer_copy[0] + layer_copy[1] + tleft) / 3;
        }
        if (rank < size-1) {
            layer[chunk_size-1] = (layer_copy[chunk_size-1] + layer_copy[chunk_size-2] + tright) / 3;
        }
        /*
        printf("Rank %d: Storm %d layer after relaxation:\n", rank, i);
        printf("[");
        for (int i = 0; i < chunk_size-1; i++) {
            printf("%f,", layer[i]);
        }
        printf("%f]\n", layer[chunk_size-1]);*/

        nreqs = 0;
         
        if (rank > 0) {
            MPI_Irecv(&tleft, 1, MPI_FLOAT, rank-1, 2, MPI_COMM_WORLD, &reqs[nreqs++]);
        }
        if (rank < size-1) {
            MPI_Irecv(&tright, 1, MPI_FLOAT, rank+1, 2, MPI_COMM_WORLD, &reqs[nreqs++]);
        }
            
        if (rank > 0) {
           MPI_Isend(&layer[0], 1, MPI_FLOAT, rank-1, 2, MPI_COMM_WORLD, &reqs[nreqs++]);
        }
        if (rank < size-1) {
            MPI_Isend(&layer[chunk_size-1], 1, MPI_FLOAT, rank+1, 2, MPI_COMM_WORLD, &reqs[nreqs++]);
        }
        MPI_Waitall(nreqs, reqs, stats);

        /*printf("Rank %d: Storm %d before max:\n", rank, i);
        printf("[");
        for (int i = 0; i < chunk_size-1; i++) {
            printf("%f,", layer[i]);
        }
        printf("%f]\n", layer[chunk_size-1]);*/

	    float lmax = - __FLT_MAX__;
        int lpos = 0; // sensible default

        // border check left
        if (rank > 0 && layer[0] > tleft && layer[0] > layer[1] && layer[0] > lmax) {
            lmax = layer[0];
        }
        // interior
        for (k = 1; k < chunk_size-1; k++) {
            if (layer[k] > layer[k-1] && layer[k] > layer[k+1] && layer[k] > lmax) {
                lmax = layer[k]; lpos = k;
            }
        }
        // border right
        if (rank < size-1 && layer[chunk_size-1] > tright && layer[chunk_size-1] > layer[chunk_size-2]
            && layer[chunk_size-1] > lmax) {
            lmax = layer[chunk_size-1]; lpos = chunk_size-1;
        }


        local_max.max = lmax;
        local_max.pos = lpos+displs[rank];
        //printf("Rank %d: Storm %d max: %f at position %d\n", rank, i, lmax, lpos);

        /* 4.4. Reduce to get the global maximum and its position */
        MPI_Reduce(&local_max, &global_max, 1, MPI_FLOAT_INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            maximum[i] = global_max.max;
            positions[i] = global_max.pos;
        }
    }
    //MPI_Gatherv(local_layer, sizes[rank], MPI_FLOAT, layer, sizes, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
}
