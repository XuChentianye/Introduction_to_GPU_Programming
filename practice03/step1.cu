#include<stdio.h>
#define N 136
__global__ void hello_threads()
{
    int tid = threadIdx.x;
    printf("%d  ", tid);
}
 
__global__ void hello_blocks()
{
    int bid = blockIdx.x;
    printf("%d  ", bid);
}
 
int main() {
    printf("Order of threads: \n");
    hello_threads<<<1, N>>>();
    cudaDeviceSynchronize();
    printf("\n\n");
 
    printf("Order of blocks: \n");
    hello_blocks<<<N, 1>>>();
    cudaDeviceSynchronize();
    printf("\n");
    
    cudaDeviceReset();
    return 0;
}