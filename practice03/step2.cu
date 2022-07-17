#include<stdio.h> 
 
int func1(int x){
    return 2*x;
}
 
__device__ int func2(int x){
    return 2*x;
}
 
__host__ __device__ int func3(int x){
    return 2*x;
}
 
void __global__ cube_gpu1(){
    int tid = threadIdx.x;
    int r1;
    // r1 = func1(tid);
    // r1 = func2(tid);
    // r1 = func3(tid);
    printf("tid:%d, cube:%d\n", tid, r1);
}
 
__global__ void cube_gpu2(){
    int tid = threadIdx.x;
    int r1;
    r1= func3(tid);
    printf("tid:%d, cube:%d\n", tid, r1);
}
 
int main() 
{ 
    printf("Host and device functions!\n");
    printf("\nResults from device:\n");
    int nBlk = 3;
    int nGrid = 2;
    cube_gpu1<<<nGrid, nBlk>>>();      
    // cube_gpu2<<<nGrid, nBlk>>>();
    cudaDeviceSynchronize();
 
    int r2;
    // r2 = func1(nBlk);
    // r2 = func2(nBlk);
    r2 = func3(nBlk);
    printf("\nResults from host:%d\n", r2);
    r2 = func3(nGrid);
    printf("\nResults from host:%d\n", r2);
 
    cudaDeviceReset(); 
    return 0; 
}

