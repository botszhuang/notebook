#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>

#define cuPERR(err) {if (err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));}

#define WRAP32  32
#define maskF32 0xFFFFFFFF

__device__ int warp_sum ( int s ) {

    s += __shfl_down_sync( maskF32 , s , 16 ); // 32/2
    s += __shfl_down_sync( maskF32 , s ,  8 );
    s += __shfl_down_sync( maskF32 , s ,  4 );
    s += __shfl_down_sync( maskF32 , s ,  2 );
    s += __shfl_down_sync( maskF32 , s ,  1 );

    return s;

}

__global__ void sum_kernel(int* aPtr, int width, int hieght , int* aPtr_total ) {

    extern __shared__ int shared_data [] ;

    const unsigned int tid = threadIdx.x;
    const unsigned int gid = ( blockIdx.x * blockDim.x ) + threadIdx.x;
    const unsigned int grid_size = gridDim.x * blockDim.x;
    const unsigned int a_element_num = width * hieght;

    // private memory
    int private_sum = 0;
    for ( unsigned int i = gid ; i < a_element_num ; i += grid_size ){
        private_sum += aPtr[i];
    }

    shared_data [ tid ] = private_sum ;

    // shared memory
    for ( unsigned int s = ( blockDim.x >>1 ) ; s > WRAP32 ; s >>= 1 ) {
        if ( tid < s ) {
            shared_data [ tid ] += shared_data[ ( tid + s) ] ;
        }
        __syncthreads();
    }

    // Wrap up
    if ( tid < WRAP32 ) {

        int final_val = shared_data [ tid ] ;
        final_val =  warp_sum( final_val ) ;

        if ( tid == 0 ) {
            printf ("[%i] shard[0] = %i\n", blockIdx.x , final_val ) ;
            atomicAdd( aPtr_total, final_val ) ;
        }
    }


}

int main() {

    cudaError_t err ;
    const dim3 threadsPerBlock( WRAP32 ,  1, 1 ) ;
    const dim3 block_num      (     16 ,  1, 1 ) ;
    const unsigned int shared_bytes = threadsPerBlock.x * sizeof( int ) ;

    size_t width  = 100;
    size_t height = 100;

    int * h_a ;
    int * d_a ;

    const unsigned int a_element_num = width * height ;
    const unsigned int a_bytes = width * height  * sizeof ( int ) ;

    h_a = ( int * )malloc( a_bytes );

    for ( unsigned int i = 0; i < a_element_num ; i++){ h_a [ i ] = 1 ; }


    int * h_aTotal = ( int * ) malloc ( sizeof ( int ) ) ;
    int * d_atotal ;
    const int zero = 0;


    err = cudaMalloc ( (void**)&d_a, a_bytes ) ;                            cuPERR(err) ;
    err = cudaMalloc ( (void**)&d_atotal, sizeof ( int ) ) ;                cuPERR(err) ;

    err = cudaMemcpy( d_a, h_a, a_bytes, cudaMemcpyHostToDevice );          cuPERR(err) ;
    err = cudaMemcpy(d_atotal, &zero, sizeof(int), cudaMemcpyHostToDevice); cuPERR(err) ;

    // launch the kenel
    sum_kernel <<< block_num , threadsPerBlock , shared_bytes >>>( d_a, width, height , d_atotal );
    err = cudaGetLastError();  cuPERR(err) ;

    // get the sum
    err = cudaMemcpy ( h_aTotal , d_atotal , sizeof(int) , cudaMemcpyDeviceToHost ) ;  cuPERR(err) ;

    printf("sum: %d\n", *h_aTotal );

    cudaDeviceSynchronize();

    // clean up memory
    cudaFree( d_a ) ;
    cudaFree( d_atotal ) ;

    free ( h_a ) ;
    free ( h_aTotal ) ;

    return EXIT_SUCCESS ;
}
