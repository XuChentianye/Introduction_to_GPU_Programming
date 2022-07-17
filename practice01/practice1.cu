#include<stdio.h>
__global__ void hello_from_gpu()
{
    int tid = threadIdx.x;
    printf("Hello World from thread %d!\n", tid);
}
int main(void)
{
    hello_from_gpu<<<1, 10>>>();
    cudaDeviceReset();
    return 0;
}