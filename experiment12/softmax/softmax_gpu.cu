#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "mnist_helper.h"
#include "time_helper.h"
#include "utils.h"
#include "error_check.h"

#define SDATA_SIZE_SOFTMAX_R 10
#define SDATA_SIZE_SOFTMAX_C 100


// 矩阵乘法加速
__global__ void kernel_matrix_multiply_block(float *M, float *N, float *P, int M_rows, int M_cols, int N_rows, int N_cols)
{
    int idx_x = threadIdx.x;
    int idx_y = threadIdx.y;
    int row = blockIdx.y*16 + idx_y;
    int col = blockIdx.x*16 + idx_x;
    float Pvalue = 0.0;
    __shared__ float M_shared[16][16];
    __shared__ float N_shared[16][16];
    for(int i=0; i<(M_cols+1)/16; i++){
        if(i*16 + idx_x < M_cols && row < M_rows){
            M_shared[idx_y][idx_x] = M[row*M_cols + i*16 + idx_x];
        }else{
            M_shared[idx_y][idx_x] = 0.0;
        } 
        if(i*16 + idx_y < N_rows && col < N_cols){
            N_shared[idx_y][idx_x] = N[(i*16 + idx_y)*N_cols + col];
        }else{
            N_shared[idx_y][idx_x] =0.0;
        }
        __syncthreads();
        for(int k=0 ; k < 16 ; k++){
            Pvalue += M_shared[idx_y][k] * N_shared[k][idx_x];
        }
        __syncthreads();
    }
    if(row < M_rows && col < N_cols){
        P[row*N_cols + col] = Pvalue;
    }
}

void gpu_matrix_multiply(float *M, float *N, float *P, int M_rows, int M_cols, int N_rows, int N_cols)
{
    float *d_M, *d_N, *d_P;
    CHECK(cudaMalloc((void **)&d_M, M_rows*M_cols*sizeof(float)));
    CHECK(cudaMalloc((void **)&d_N, N_rows*N_cols*sizeof(float)));
    CHECK(cudaMalloc((void **)&d_P, M_rows*N_cols*sizeof(float)));
    
    CHECK(cudaMemcpy(d_M, M, M_rows*M_cols*sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_N, N, N_rows*N_cols*sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_P, 0, M_rows*N_cols*sizeof(float)));
    
    dim3 block(16, 16);
    dim3 grid((N_cols-1)/block.y+1, (M_rows-1)/block.x+1);

    kernel_matrix_multiply_block<<<grid, block>>>(d_M, d_N, d_P, M_rows, M_cols, N_rows, N_cols);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(P, d_P, M_rows*N_cols*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_M));
    CHECK(cudaFree(d_N));
    CHECK(cudaFree(d_P));
}


// 矩阵带系数的加法加速 （S = coefficientM * M + coefficientN * N）
__global__ void kernel_matrix_add_with_coefficient(float *M, float *N, float *S, float coefficientM, float coefficientN)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    S[idx] = coefficientM * M[idx] + coefficientN * N[idx];
}

void gpu_matrix_add_with_coefficient(float *M, float *N, float *S, float coefficientM, float coefficientN, int rows, int cols)
{
    float *d_M, *d_N, *d_S;
    CHECK(cudaMalloc((void **)&d_M, rows*cols*sizeof(float)));
    CHECK(cudaMalloc((void **)&d_N, rows*cols*sizeof(float)));
    CHECK(cudaMalloc((void **)&d_S, rows*cols*sizeof(float)));
    
    CHECK(cudaMemcpy(d_M, M, rows*cols*sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_N, N, rows*cols*sizeof(float), cudaMemcpyHostToDevice));

    kernel_matrix_add_with_coefficient<<<(rows*cols-1)/1024+1, 1024>>>(d_M, d_N, d_S, coefficientM, coefficientN);

    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(S, d_S, rows*cols*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_M));
    CHECK(cudaFree(d_N));
    CHECK(cudaFree(d_S));
}


// softmax加速 （一次读入一个10*100的小块到shared memory中，减少求每一列的exp函数值的和时的stride式访问）
__global__ void kernel_softmax(float *activations, int number_of_samples)
{
    __shared__ float sdata[SDATA_SIZE_SOFTMAX_R][SDATA_SIZE_SOFTMAX_C];
    __shared__ float sums[SDATA_SIZE_SOFTMAX_C];
    sdata[threadIdx.y][threadIdx.x] = activations[threadIdx.y * number_of_samples + blockIdx.x * SDATA_SIZE_SOFTMAX_C + threadIdx.x];
    __syncthreads();
    if(threadIdx.y == 0){
        float temp = 0;
        for(int i=0; i < SDATA_SIZE_SOFTMAX_R; i++){
            temp += exp(sdata[i][threadIdx.x]);
        }
        sums[threadIdx.x] = temp;
    }
    __syncthreads();
    activations[threadIdx.y * number_of_samples + blockIdx.x * SDATA_SIZE_SOFTMAX_C + threadIdx.x] = exp(sdata[threadIdx.y][threadIdx.x]) / sums[threadIdx.x];
}

void gpu_softmax(float *activations, int number_of_samples)
{
    float *d_activations = NULL;
    CHECK(cudaMalloc((void **)&d_activations, 10 * number_of_samples * sizeof(float)));
    CHECK(cudaMemcpy(d_activations, activations, 10 * number_of_samples * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(100, 10);
    dim3 grid((number_of_samples-1)/100+1);

    kernel_softmax<<<grid, block>>>(d_activations, number_of_samples);

    CHECK(cudaMemcpy(activations, d_activations, 10 * number_of_samples * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_activations));
}


// delta计算加速
__global__ void kernel_delta(float *delta, float *activations, int *labels, int number_of_samples)
{
    int r = threadIdx.x;
    int c = blockIdx.x;
    if(r == labels[c])
        delta[r * number_of_samples + c] = activations[r * number_of_samples + c] - 1;
    else
        delta[r * number_of_samples + c] = activations[r * number_of_samples + c];
    
}

void gpu_compute_delta(float *delta, float *activations, int *labels, int number_of_samples)
{
    float *d_activations = NULL;
    float *d_delta = NULL;
    int *d_labels = NULL;
    CHECK(cudaMalloc((void **)&d_activations, 10 * number_of_samples * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_delta, 10 * number_of_samples * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_labels, number_of_samples * sizeof(int)));
    CHECK(cudaMemcpy(d_activations, activations, 10 * number_of_samples * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_labels, labels, number_of_samples * sizeof(int), cudaMemcpyHostToDevice));

    kernel_delta<<<number_of_samples, 10>>>(d_delta, d_activations, d_labels, number_of_samples);

    CHECK(cudaMemcpy(delta, d_delta, 10 * number_of_samples * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_activations));
    CHECK(cudaFree(d_delta));
    CHECK(cudaFree(d_labels));
}


// 计算accuracy（利用原子操作atomicAdd）
__global__ void kernel_accuracy(float *activations, int *labels, int number_of_samples, int *correct)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int res = 0;
    float max = activations[i];
    for(int k = 0; k < 10; k++){
        if(activations[k * number_of_samples + i] > max)
        {
            max = activations[k * number_of_samples + i];
            res = k;
        }
    }
    if(res == labels[i]) atomicAdd(correct, 1); 
}

float gpu_compute_accuracy(float *activations, int *labels, int number_of_samples)
{
    float *d_activations = NULL;
    int *d_labels = NULL;
    int *correct = NULL;
    int *d_correct = NULL;
    int co = 0;
    correct = (int *)malloc(sizeof(int));

    CHECK(cudaMalloc((void **)&d_correct, sizeof(int)));
    CHECK(cudaMalloc((void **)&d_activations, 10 * number_of_samples * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_labels, number_of_samples * sizeof(int)));
    CHECK(cudaMemcpy(d_activations, activations, 10 * number_of_samples * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_labels, labels, number_of_samples * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_correct, 0, sizeof(int)));

    kernel_accuracy<<<(number_of_samples-1)/1024+1, 1024>>>(d_activations, d_labels, number_of_samples, d_correct);

    CHECK(cudaMemcpy(correct, d_correct, sizeof(int), cudaMemcpyDeviceToHost));
    co = *correct;
    CHECK(cudaFree(d_activations));
    CHECK(cudaFree(d_labels));
    CHECK(cudaFree(d_correct));
    free(correct);
    return co / (float)number_of_samples;
}


// 计算loss（利用原子操作atomicAdd）
__global__ void kernel_loss(float *activations, int *labels, int number_of_samples, float *loss)
{
    int i = blockIdx.x;
    int k = threadIdx.x;
    if(labels[i] == k)
        atomicAdd(loss, log(activations[k * number_of_samples + i]));
}

float gpu_compute_loss(float *activations, int *labels, int number_of_samples)
{
    float *d_activations = NULL;
    int *d_labels = NULL;
    float *loss = NULL;
    float *d_loss = NULL;
    float res = 0;

    loss = (float *)malloc(sizeof(float));
    CHECK(cudaMalloc((void **)&d_activations, 10 * number_of_samples * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_labels, number_of_samples * sizeof(int)));
    CHECK(cudaMalloc((void **)&d_loss, sizeof(float)));

    CHECK(cudaMemset(d_loss, 0, sizeof(float)));
    CHECK(cudaMemcpy(d_activations, activations, 10 * number_of_samples * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_labels, labels, number_of_samples * sizeof(int), cudaMemcpyHostToDevice));

    kernel_loss<<<number_of_samples, 10>>>(d_activations, d_labels, number_of_samples, d_loss);

    CHECK(cudaMemcpy(loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));

    res = *loss / (- number_of_samples);
    CHECK(cudaFree(d_activations));
    CHECK(cudaFree(d_labels));
    CHECK(cudaFree(d_loss));
    free(loss);
    return res;
}


int main(int argc, char *argv[]) {

	const char * train_images_file = "train-images-idx3-ubyte";
	const char * train_labels_file = "train-labels-idx1-ubyte";
	const char * test_images_file = "t10k-images-idx3-ubyte";
	const char * test_labels_file = "t10k-labels-idx1-ubyte";

	float *data_train, *data_test;
	int *labels_train, *labels_test;
	int number_of_samples_train, number_of_samples_test, rows, columns;

    /*
    * * * * Load training data  * * * *
    * data_train: float, 60000x784, each row represents a data sample *
    * labels_train: int, 60000, data labels, [1,2,3,4,5,...] *
    * number_of_samples_train: 60000 * 
    * rows: 28, number of pixel rows in an image; columns: 28, number of pixel columns in an image * 
    */
	get_dataset(train_images_file, train_labels_file, &data_train, &labels_train, &number_of_samples_train, &rows, &columns);
    scale_pixels(data_train, number_of_samples_train * rows * columns);
    printf("Training dataset: [%d %d %d] \n\n", number_of_samples_train, rows, columns);

	/*
    * * * * Load test data  * * * *
    * data_test: float, 10000x784, each row represents a data sample *
    * labels_test: int, 10000, data labels, [1,2,3,4,5,...] *
    * number_of_samples_test: 10000 * 
    * rows: 28, number of pixel rows in an image; columns: 28, number of pixel columns in an image * 
    */
	get_dataset(test_images_file, test_labels_file, &data_test, &labels_test, &number_of_samples_test, &rows, &columns);
	scale_pixels(data_test, number_of_samples_test * rows * columns);
    printf("\n Test dataset: [%d %d %d] \n", number_of_samples_test, rows, columns);
    
	/* 
    * Model initialization *
    * output = softmax(W*input) * 
    * W:10x784, input:784xn, output:10xn*
    */
    int W_rows = 10;
    int W_columns = 784;
    float* W = (float *)malloc(W_rows*W_columns*sizeof(float));
    weight_initialization(W, W_rows, W_columns);
    
    /* 
    * Training data, activation and gradient buffers *
    */
    float* activations_train = (float *)malloc(W_rows * number_of_samples_train * sizeof(float));
    float* data_transposed_train = (float *)malloc(rows * columns * number_of_samples_train * sizeof(float));
    float *delta = (float *)malloc(W_rows*number_of_samples_train*sizeof(float));
    float *W_grad = (float *)malloc(W_rows*W_columns*sizeof(float));   

    /* 
    * Test data and activation buffers *
    */
    float* activations_test = (float *)malloc(W_rows * number_of_samples_test * sizeof(float));
    float* data_transposed_test = (float *)malloc(rows * columns * number_of_samples_test * sizeof(float));
    
    /* 
    * Data sample visualization *
    */
    printf("label: %d\n", labels_train[18]);
    for(int i=0; i<rows; i++)
    {
        for(int j=0; j<columns; j++)
        {
            printf("%s ", *(data_train+i*columns+j + 18*rows*columns)>0?"#":" ");
        }
        printf("\n");
    }
    
    // 把data_train的所有值限制到 0 ~ 1 之间，防止后面部分值超出float型表示范围
    for(int r = 0; r < number_of_samples_train; r++)
    {
        for(int c = 0; c < rows*columns; c++)
        {
            data_train[rows*columns * r + c] /=225;
        }
    }

    // 把data_test的所有值限制到 0 ~ 1 之间，防止后面部分值超出float型表示范围
    for(int r = 0; r < number_of_samples_test; r++)
    {
        for(int c = 0; c < rows*columns; c++)
        {
            data_test[rows*columns * r + c] /=225;
        }
    }

    /* 
    * data: [n,784], one image per row *
    * data_transposed_train | data_transposed_test: [784,n], one image per column* 
    */
    matrix_transpose(data_train, data_transposed_train, number_of_samples_train, rows*columns);
    matrix_transpose(data_test, data_transposed_test, number_of_samples_test, rows*columns);   
    
    /* 
    * Training loop *
    */
    int epoch_num = 1000; 
    float learning_rate = 0.5;
    float loss_train, acc_train, loss_test, acc_test;
    double time_begin;
    for(int epoch=0; epoch<epoch_num; epoch++)
    {
        time_begin = cpuSecond();
        /* 
        * Forward on training set *
        * data: [n,784], one image per row *
        * W:[10,784], data_transposed_train:[784,n], activations_train: [10,n] * 
        */
        // Todo 1
        // activations_train = W * data_transposed_train
        gpu_matrix_multiply(W, data_transposed_train, activations_train, W_rows, W_columns, W_columns, number_of_samples_train);

        /* 
        * softmax normalization on activations *
        * activations: [10,n] *
        */
        // Todo 2
        // softmax(activations_train)
        gpu_softmax(activations_train, number_of_samples_train);

        // Todo 3
        // loss on training set
        loss_train = gpu_compute_loss(activations_train, labels_train, number_of_samples_train);

        // Todo 4
        // accuracy on training set
        acc_train = gpu_compute_accuracy(activations_train, labels_train, number_of_samples_train);

        // Todo 5
        // [Test] Forward on test set
        // activations_test = W * data_transposed_test
        gpu_matrix_multiply(W, data_transposed_test, activations_test, W_rows, W_columns, W_columns, number_of_samples_test);

        // Todo 6
        // [Test] Softmax on activations_test
        gpu_softmax(activations_test, number_of_samples_test);

        // Todo 7
        // [Test] loss on test set
        loss_test = gpu_compute_loss(activations_test, labels_test, number_of_samples_test);

        // Todo 8
        // [Test] accuracy on test set
        acc_test = gpu_compute_accuracy(activations_test, labels_test, number_of_samples_test);

        // Reset gradients
        memset(delta, 0, W_rows*number_of_samples_train*sizeof(float));
        memset(W_grad, 0, W_rows*W_columns*sizeof(float));

        // Todo 9
        // Compute delta
        // delta [10,n] = activations [10,n] - y [10,n]; (y[labels_train_set, :]=1)
        gpu_compute_delta(delta, activations_train, labels_train, number_of_samples_train);

        // Todo 10
        // W_grad [10,784] = delta [10,n] * images_train_set [n,784]
        gpu_matrix_multiply(delta, data_train, W_grad, W_rows, number_of_samples_train, number_of_samples_train, W_columns);

        // Todo 11
        // Update, W = W - alpha * W_grad;
        gpu_matrix_add_with_coefficient(W, W_grad, W, 1.0, - learning_rate, W_rows, W_columns);

        printf("[GPU][Epoch %d]: Train loss:%f, Train accuracy:%f;   Test loss:%f, Test accuracy:%f ; time cost: %lf s\n", epoch, loss_train, acc_train, loss_test, acc_test, cpuSecond()-time_begin);
    }
    
    return 0;
}
