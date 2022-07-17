#include<stdio.h>
#include<stdlib.h>
#include"error_check.h"
#include"gpu_timer.h"

#define DTYPE double
#define DTYPE_FORMAT "%lf"
#define BLOCK_SIZE 32


DTYPE partialSum(DTYPE *vector, int n) {
	DTYPE temp = 0;
	for (int i = 0; i < n; i++) {
		temp += vector[i];
	}
	return temp;
}

/*
 * Todo:
 * 1. reduction kernel in which the threads are consecutively mapped to data
 * 2. using atomicAdd to compute the total sum
*/
__device__ DTYPE global_sum[1];

__global__ void kernel_reduction_atomicAdd(DTYPE *input, DTYPE *output, int n) {
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	if(tid == 0 && bid == 0) global_sum[0] = 0;
	__syncthreads();

	int offset = 2 * blockIdx.x *blockDim.x;
	for(int s = blockDim.x; s >= 1; s >>= 1){
		if(tid + s + offset < n && tid < s)
			input[tid + offset] += input[tid + s + offset];
		__syncthreads();
	}
	//using atomicAdd to compute the total sum
	if(tid == 0) atomicAdd(global_sum, input[offset]);
	__syncthreads();
	output[0] = global_sum[0];
}


/*
 * Todo:
 * Wrapper function without the need to sum the reduced results from blocks
*/
DTYPE gpu_reduction(DTYPE *input, int n,
		void (*kernel)(DTYPE *input, DTYPE *output, int n)) {
	dim3 block(1024);
	//one block can process 2048 numbers
	dim3 grid((n-1)/(block.x*2)+1);

	DTYPE *output = (DTYPE*)malloc(sizeof(DTYPE));
	DTYPE res = 0;

	DTYPE *d_input = NULL;
	DTYPE *d_output = NULL;
	CHECK(cudaMalloc((void**)&d_input, n*sizeof(DTYPE)));
	CHECK(cudaMalloc((void**)&d_output, sizeof(DTYPE)));

	CHECK(cudaMemcpy(d_input,input,n*sizeof(DTYPE),cudaMemcpyHostToDevice));

	kernel<<<grid, block>>>(d_input,d_output,n);

	CHECK(cudaMemcpy(output,d_output,sizeof(DTYPE),cudaMemcpyDeviceToHost));

	CHECK(cudaFree(d_input));
	CHECK(cudaFree(d_output));

	res = *output;
	free(output);
	return res;
}


DTYPE* test_data_gen(int n) {
	srand(time(0));
	DTYPE *data = (DTYPE *) malloc(n * sizeof(DTYPE));
	for (int i = 0; i < n; i++) {
		data[i] = 1.0 * (rand() % RAND_MAX) / RAND_MAX;
	}
	return data;
}

void test(int n,
		DTYPE (*reduction)(DTYPE *input, int n,
		                        void (*kernel)(DTYPE *input, DTYPE *output, int n)),
		                        void (*kernel)(DTYPE *input, DTYPE *output, int n))
{
	DTYPE computed_result, computed_result_gpu;
	DTYPE *vector_input;
	vector_input = test_data_gen(n);

	computed_result = partialSum(vector_input, n);

	GpuTimer timer;
	timer.Start();
	computed_result_gpu = reduction(vector_input, n, kernel);
	timer.Stop();
	printf("Time cost:%f ms\n",timer.Elapsed());

	printf("[%d] Computed sum (CPU): ", n);
	printf(DTYPE_FORMAT, computed_result);
	printf("  GPU result:");
	printf(DTYPE_FORMAT, computed_result_gpu);

	if (abs(computed_result_gpu - computed_result) < 1e-3) {
		printf("\n PASSED! \n");
	} else {
		printf("\n FAILED! \n");
	}
	printf("\n");

	free(vector_input);

}

int main(int argc, char **argv) {
	int n_arr[] = {1, 7, 585, 5000, 300001, 1<<20};
	// int n_arr[] = {1, 7};
	for(int i=0; i<sizeof(n_arr)/sizeof(int); i++)
	{
		test(n_arr[i], gpu_reduction, kernel_reduction_atomicAdd);
	}

	return 0;
}
