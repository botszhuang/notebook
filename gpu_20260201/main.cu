#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>

#define cuPERR(err) {if (err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));}

#define WRAP32  32
#define maskF32 0xFFFFFFFF

template <typename T>
__device__ T warp_sum ( T s ) {

    s += __shfl_down_sync( maskF32 , s , 16 ); // 32/2
    s += __shfl_down_sync( maskF32 , s ,  8 );
    s += __shfl_down_sync( maskF32 , s ,  4 );
    s += __shfl_down_sync( maskF32 , s ,  2 );
    s += __shfl_down_sync( maskF32 , s ,  1 );

    return s;

}

template <typename T>
__global__ void sum_kernel(T* aPtr, int width, int hieght , T* aPtr_total ) {

    extern __shared__ T shared_data [] ;

    const unsigned int tid = threadIdx.x;
    const unsigned int gid = ( blockIdx.x * blockDim.x ) + threadIdx.x;
    const unsigned int grid_size = gridDim.x * blockDim.x;
    const unsigned int a_element_num = width * hieght;

    // private memory
    T private_sum = 0;
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

        T final_val = shared_data [ tid ] ;
        final_val =  warp_sum( final_val ) ;

        if ( tid == 0 ) {
            //printf ("[%i] shard[0] = %i\n", blockIdx.x , final_val ) ;
            atomicAdd( aPtr_total, final_val ) ;
        }
    }


}

typedef int DATA_TYPE ;

int main() {

    cudaError_t err ;
    const dim3 threadsPerBlock( WRAP32 ,  1, 1 ) ;
    const dim3 block_num      (     16 ,  1, 1 ) ;
    const unsigned int shared_bytes = threadsPerBlock.x * sizeof( DATA_TYPE ) ;

    size_t width  = 100;
    size_t height = 100;

    DATA_TYPE * h_a = NULL ;
    DATA_TYPE * d_a = NULL ;

    const unsigned int a_element_num = width * height ;
    const unsigned int a_bytes = width * height  * sizeof ( DATA_TYPE ) ;

    h_a = ( DATA_TYPE * ) malloc ( a_bytes );

    for ( unsigned int i = 0; i < a_element_num ; i++){ h_a [ i ] = 1 ; }


    const unsigned int d_atotal_bytes = sizeof ( DATA_TYPE ) ;
    DATA_TYPE * h_aTotal = ( DATA_TYPE * ) malloc ( d_atotal_bytes ) ;
    DATA_TYPE * d_atotal = NULL ;
    const DATA_TYPE zero = 0 ;


    err = cudaMalloc ( (void**)&d_a, a_bytes ) ;                               cuPERR(err) ;
    err = cudaMalloc ( (void**)&d_atotal, sizeof ( DATA_TYPE ) ) ;             cuPERR(err) ;

    // tranport the data from host to device
    err = cudaMemcpy( d_a, h_a, a_bytes, cudaMemcpyHostToDevice );             cuPERR(err) ;
    err = cudaMemcpy(d_atotal, &zero, sizeof( zero ), cudaMemcpyHostToDevice); cuPERR(err) ;

    // launch the kenel
    sum_kernel <DATA_TYPE> <<< block_num , threadsPerBlock , shared_bytes >>>( d_a, width, height , d_atotal );
    err = cudaGetLastError();  cuPERR(err) ;

    // tranport result from deive to host
    err = cudaMemcpy ( h_aTotal , d_atotal , d_atotal_bytes , cudaMemcpyDeviceToHost ) ;  cuPERR(err) ;

    printf("sum: %lf\n", ( double ) *h_aTotal );

    cudaDeviceSynchronize();

    // clean up memory
    cudaFree( d_a ) ;
    cudaFree( d_atotal ) ;

    free ( h_a ) ;
    free ( h_aTotal ) ;

    return EXIT_SUCCESS ;
}
