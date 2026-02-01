#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>

#define myUINT unsigned int
#define A_TYPE int

__global__ void print_kernel(int* aPtr, int width, int height) {
    // 為了處理寬度 100，我們需要考慮 Block 的索引
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = 0 ;
    if (x < width && y < height) {
        index = y * width + x;
        printf("(%i, %i) -> index[%i] = %i\n", x, y, index, aPtr[index]);
    }
}

int main() {

    cudaError_t err ;
    const dim3 threadsPerBlock( 32, 32, 1 ) ;
    const dim3 block_num      (  1,  1, 1 ) ;

    size_t width  = 100;
    size_t height = 10;

    A_TYPE * h_a ;
    A_TYPE * d_a ;

    const myUINT total_elements = width * height ;
    const myUINT total_bytes = width * height  * sizeof ( A_TYPE ) ;

    h_a = ( A_TYPE * )malloc( total_bytes );

    for ( myUINT i = 0; i < total_elements ; i++){ h_a [ i ] = i ; }

    err = cudaMalloc ( (void**)&d_a, total_bytes ) ;

    err = cudaMemcpy( d_a, h_a, total_bytes, cudaMemcpyHostToDevice );

    print_kernel <<< block_num , threadsPerBlock >>>( d_a, width, height);

    cudaDeviceSynchronize();

    cudaFree( d_a );
    free( h_a );


    return EXIT_SUCCESS ;
}
