#include<stdio.h>
__global__ void cuda_hello(void)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    printf("bid:[%d], tid:[%d] Hello World from GPU!\n", bid, tid);
}
 
int main() 
{
    int numOfBlocks = 5;
    int numOfThreads = 3;
    cuda_hello<<<numOfBlocks , numOfThreads>>>();
    cudaDeviceReset();
    return 0;
}