#include<stdio.h>
int main() {
    int dCount;
    cudaGetDeviceCount(&dCount);
    for(int i=0; i<dCount; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("CUDA Device#%d\n", i);
        printf("Device name:%s\n", prop.name);
        printf("multiProcessorCount:%d\n", prop.multiProcessorCount);
        printf("maxThreadsPerBlock:%d\n", prop.maxThreadsPerBlock);
        printf("warpSize:%d\n", prop.warpSize);
        printf("maxThreadsDim[3]:%d, %d, %d\n", 
        prop.maxThreadsDim[0], 
        prop.maxThreadsDim[1], 
        prop.maxThreadsDim[2]);
        printf("maxGridSize[3]:%d, %d, %d\n", 
        prop.maxGridSize[0], 
        prop.maxGridSize[1], 
        prop.maxGridSize[2]);
    }
    cudaDeviceReset();
    return 0;
}