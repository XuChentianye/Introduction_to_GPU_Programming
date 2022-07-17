#include"error_check.h"
#include<stdio.h>

__global__ void func1(int x)
{
    int tid = threadIdx.x;
    printf("thread: %d, parameter:%d  \n", tid, x);
}

__global__ void func2(int *x)
{
    int tid = threadIdx.x;
    printf("thread: %d, parameter:%d  \n", tid, *x);
}

__global__ void func3(int x[], int n)
{
    int tid = threadIdx.x;
    printf("thread: %d, parameter:%d  \n", tid, x[tid]);
}

int main() {
	int a = 6, temp = 7;
	int *b, *tem = &temp;
	int n = 3;
	int tempc[3] = {0,1,2};
	int *c;

	cudaMalloc((void**)&b, sizeof(int));
	CHECK(cudaGetLastError());
	cudaMemcpy(b, tem, sizeof(int), cudaMemcpyHostToDevice);
	CHECK(cudaGetLastError());
	cudaMalloc((void**)&c, 3*sizeof(int));
	CHECK(cudaGetLastError());
	cudaMemcpy(c, tempc, 3*sizeof(int), cudaMemcpyHostToDevice);
	CHECK(cudaGetLastError());

	func1<<<1, 1>>>(a);
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());
	printf("\n");

	func2<<<1, 1>>>(b);
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());

	printf("\n");
	func3<<<1, 1>>>(c, n);
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceSynchronize());

	printf("\n");
	CHECK(cudaDeviceReset());

    return 0;
}
