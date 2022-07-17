#include<stdio.h>
__global__ void cuda_hello(void)
{
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int idx = bid*blockDim.x+tid;
    printf("idx:[%d], bid:[%d], tid:[%d] Hello World from GPU!\n", idx, bid, tid);
}
 
int main() 
{
    int numOfBlocks = 5;
    int numOfThreads = 3;
    cuda_hello<<<numOfBlocks , numOfThreads>>>();
    cudaDeviceReset();
    return 0;
}
