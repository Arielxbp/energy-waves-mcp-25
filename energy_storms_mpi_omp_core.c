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
    int lsize = int_chunk_size + (rank < rem ? 1 : 0);
    int start = rank*int_chunk_size + (rank < rem ? rank : rem);

    float *layer = (float *)calloc(lsize+2, sizeof(float))+1;
    float *layer_copy = (float *)calloc(lsize+2, sizeof(float))+1;
    
    if ( layer == NULL || layer_copy == NULL ) {
        fprintf(stderr,"Error: Allocating the layer memory\n");
        exit( EXIT_FAILURE );
    }

    struct {
        float max;
        int rank;
    } global_max, local_max;

    //MPI_Scatterv(layer, sizes, displs, MPI_FLOAT, local_layer, sizes[rank], MPI_FLOAT, 0, MPI_COMM_WORLD);
    /* 4. Storms simulation */
    for( i=0; i<num_storms; i++) {
        /* 4.1. Add impacts energies to layer cells */
        /* For each particle */
        //#pragma omp parallel for schedule(static) collapse(2)
        for( j=0; j<storms[i].size; j++ ){
            float energy = (float)storms[i].posval[j*2+1] * 1000.0f;
            int pos = storms[i].posval[j*2]; 
            for( k=0; k<lsize; k++ ){
            /* For each cell in the layer */
                /* Update the energy value for the cell */
                int distance = pos - k-start;
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
                layer[k] += (energy_k >= THRESHOLD / layer_size || energy_k <= -THRESHOLD / layer_size) ? energy_k : 0.0f;
                
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
        
        /* 4.2.2. Update layer using the ancillary values.
                  Skip updating the first and last positions */
        int left = (rank > 0) ? rank-1:MPI_PROC_NULL;
        int right = (rank < size - 1) ? rank+1:MPI_PROC_NULL;

        MPI_Sendrecv(&layer[0], 1, MPI_FLOAT, left, 0, &layer[lsize], 1, MPI_FLOAT, right, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&layer[lsize - 1], 1, MPI_FLOAT, left, 1, &layer[-1], 1, MPI_FLOAT, right,  1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        
        for( k=0; k<lsize; k++ )
            layer_copy[k] = layer[k];

        /*(
        printf("Rank %d: Storm %d layer after exchange and border updates:\n", rank, i);
        printf("[");
        for (int i = 0; i < chunk_size-1; i++) {
            printf("%f,", local_layer[i]);
        }
        printf("%f]\n", local_layer[chunk_size-1]);*/    

        for( k=0; k<lsize; k++ ) {
            if ((k+start == 0 && k+start == layer_size-1) == 0) {
                layer[k] = ( layer_copy[k-1] + layer_copy[k] + layer_copy[k+1] ) / 3.0f; 
            }
        }
        /*
        printf("Rank %d: Storm %d layer after relaxation:\n", rank, i);
        printf("[");
        for (int i = 0; i < chunk_size-1; i++) {
            printf("%f,", layer[i]);
        }
        printf("%f]\n", layer[chunk_size-1]);*/
        /*printf("Rank %d: Storm %d before max:\n", rank, i);
        printf("[");
        for (int i = 0; i < chunk_size-1; i++) {
            printf("%f,", layer[i]);
        }
        printf("%f]\n", layer[chunk_size-1]);*/

	    float lmax = - __FLT_MAX__;
        int lpos = 0;

        for( k=0; k<lsize; k++ ) {
            if ((k+start == 0 && k+start == layer_size-1) == 0) {
                if (layer[k] > layer[k-1] && layer[k] > layer[k+1] && layer[k] > lmax) {
                lmax = layer[k]; lpos = k+start;
                }
            }
        }


        local_max.max = lmax;
        local_max.rank = rank;
        global_max.max = 0.0f;
        global_max.rank = 0;
        //printf("Rank %d: Storm %d max: %f at position %d\n", rank, i, lmax, lpos);

        /* 4.4. Reduce to get the global maximum and its position */
        MPI_Reduce(&local_max, &global_max, 1, MPI_FLOAT_INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);

        int max_owner = (rank == 0) ? global_max.rank : 0;
        MPI_Bcast(&max_owner, 1, MPI_INT, 0, MPI_COMM_WORLD);

        int global_max_pos = (rank == max_owner) ? lpos : 0;
        MPI_Bcast(&global_max_pos, 1, MPI_INT, max_owner, MPI_COMM_WORLD);
        if (rank == 0)
        {
            maximum[i] = global_max.max;
            positions[i] = global_max_pos;
        }
    }
    //MPI_Gatherv(local_layer, sizes[rank], MPI_FLOAT, layer, sizes, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
}
