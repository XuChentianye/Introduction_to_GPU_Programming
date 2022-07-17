#include<stdio.h>
#include<stdlib.h>
#include"error_check.h"
#include"gpu_timer.h"

#define DTYPE double
#define DTYPE_FORMAT "%lf"
#define BLOCK_SIZE 32

/* CPU implementation */
DTYPE partialSum(DTYPE *vector, int n) {
	DTYPE temp = 0;
	for (int i = 0; i < n; i++) {
		temp += vector[i];
	}
	return temp;
}

/*
 * Todo:
 * reduction kernel in which the threads are mapped to data with stride 2
*/
__global__ void kernel_reduction_non_consecutive(DTYPE *input, DTYPE *output, int n) {
	int tid = threadIdx.x;
	int offset = 2 * blockIdx.x *blockDim.x;
	for(int s = 1; s <= blockDim.x; s <<= 1){
		if(tid % s == 0 && tid*2 + s + offset < n){
			input[tid*2 + offset] += input[tid*2 + s + offset];
			__syncthreads();
		}
	}
	//write result of this block to global memory
	if(tid == 0) output[blockIdx.x] = input[offset];
}

/*
 * Todo:
 * reduction kernel in which the threads are consecutively mapped to data
*/
__global__ void kernel_reduction_consecutive(DTYPE *input, DTYPE *output, int n) {
	int tid = threadIdx.x;
	int offset = 2 * blockIdx.x *blockDim.x;
	for(int s = blockDim.x; s >= 1; s >>= 1){
		if(tid + s + offset < n && tid < s)
			input[tid + offset] += input[tid + s + offset];
		__syncthreads();
	}
	//write result of this block to global memory
	if(tid == 0) output[blockIdx.x] = input[offset];
}

/*
 * Todo:
 * Wrapper function that utilizes cpu computation to sum the reduced results from blocks
*/
DTYPE gpu_reduction_cpu(DTYPE *input, int n,
		void (*kernel)(DTYPE *input, DTYPE *output, int n)) {
	dim3 block(1024);

	//one block can process 2048 numbers
	dim3 grid((n-1)/(block.x*2)+1);

	DTYPE *output = (DTYPE*)malloc(grid.x*sizeof(DTYPE));

	DTYPE *d_input = NULL;
	DTYPE *d_output = NULL;
	CHECK(cudaMalloc((void**)&d_input, n*sizeof(DTYPE)));
	CHECK(cudaMalloc((void**)&d_output, grid.x*sizeof(DTYPE)));

	CHECK(cudaMemcpy(d_input,input,n*sizeof(DTYPE),cudaMemcpyHostToDevice));

	kernel<<<grid,block>>>(d_input,d_output,n);

	CHECK(cudaMemcpy(output,d_output,grid.x*sizeof(DTYPE),cudaMemcpyDeviceToHost));

	CHECK(cudaFree(d_input));
	CHECK(cudaFree(d_output));

	DTYPE res = 0;
	for(int i=0; i<grid.x; i++){
		res += output[i];
	}

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
	for(int i=0; i<sizeof(n_arr)/sizeof(int); i++)
	{
		test(n_arr[i], gpu_reduction_cpu, kernel_reduction_non_consecutive);
		test(n_arr[i], gpu_reduction_cpu, kernel_reduction_consecutive);
	}

	return 0;
}
