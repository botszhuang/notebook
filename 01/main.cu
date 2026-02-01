#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>

#define myUINT unsigned int
#define A_TYPE int


int main() {

    cudaError_t err ;

    size_t width  = 100;
    size_t height = 10;
    size_t pitch  = 0 ;

    A_TYPE * h_a ;
    A_TYPE * d_a ;

    const myUINT total_elements = width * height ;
    const myUINT total_bytes = width * height  * sizeof ( A_TYPE ) ;
    const myUINT width_in_bytes = width * sizeof ( A_TYPE ) ;

    h_a = ( A_TYPE * )malloc( total_bytes );


    for ( myUINT i = 0; i < total_elements ; i++){ h_a [ i ] = i ; }

    err = cudaMallocPitch( (void**)&d_a, & pitch , width_in_bytes , height ) ;

    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
        cudaFree( d_a );
        free(h_a);
        return EXIT_FAILURE;
    }

    printf("Logical Width: %zu elements (%u bytes)\n", width, width_in_bytes);
    printf("Actual Pitch: %zu bytes\n", pitch);

    cudaMemcpy2D( d_a, pitch, h_a, width_in_bytes, width_in_bytes, height, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    cudaFree( d_a );
    free( h_a );


    return EXIT_SUCCESS ;
}
