#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include"mnist_helper.h"
#include"time_helper.h"
#include"utils.h"

/* 
* Compile command *
* gcc -o softmax softmax.c -lm *
* OR *
* nvcc -o softmax softmax.c --run *
*/

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
        memset(activations_train, 0, W_rows * number_of_samples_train * sizeof(float));
        for(int r = 0; r < W_rows; r++)
        {
            for(int c = 0; c < number_of_samples_train; c++)
            {
                for(int i = 0; i < rows*columns; i++)
                {
                    activations_train[r * number_of_samples_train + c] += W[W_columns * r + i] * data_transposed_train[number_of_samples_train * i + c];
                }
                
            }
        }

        /* 
        * softmax normalization on activations *
        * activations: [10,n] *
        */
        // Todo 2
        // softmax(activations_train)
        for(int c = 0; c < number_of_samples_train; c++)
        {
            float sum = 0;
            for(int r = 0; r < W_rows; r++)
            {
                sum += exp(activations_train[r * number_of_samples_train + c]);
            }
            for(int r = 0; r < W_rows; r++)
            {
                activations_train[r * number_of_samples_train + c] = exp(activations_train[r * number_of_samples_train + c]) / sum;
            }
        }

        // Todo 3
        // loss on training set
        loss_train = 0.;
        for(int i = 0; i < number_of_samples_train; i++)
        {
            for(int k = 0; k < W_rows; k++)
            {
                if(fabs(labels_train[i] - k) < 1e-6)
                    loss_train += log(activations_train[k * number_of_samples_train + i]);
            }
        }
        loss_train /= (-number_of_samples_train);

        // Todo 4
        // accuracy on training set
        int correct = 0;
        int res = -1;
        float max = 0;
        for(int i = 0; i < number_of_samples_train; i++)
        {
            res = 0;
            max = activations_train[i];
            for(int k = 0; k < W_rows; k++)
            {
                if(activations_train[k * number_of_samples_train + i] > max)
                {
                    max = activations_train[k * number_of_samples_train + i];
                    res = k;
                }
            }
            if(res == labels_train[i]) correct++;
        }
        acc_train = correct / (float)number_of_samples_train;

        // Todo 5
        // [Test] Forward on test set
        // activations_test = W * data_transposed_test
        memset(activations_test, 0, W_rows * number_of_samples_test * sizeof(float));
        for(int r = 0; r < W_rows; r++)
        {
            for(int c = 0; c < number_of_samples_test; c++)
            {
                for(int i = 0; i < rows*columns; i++)
                {
                    activations_test[r * number_of_samples_test + c] += W[W_columns * r + i] * data_transposed_test[number_of_samples_test * i + c];
                }
            }
        }

        // Todo 6
        // [Test] Softmax on activations_test
        for(int c = 0; c < number_of_samples_test; c++)
        {
            float sum = 0;
            for(int r = 0; r < W_rows; r++)
            {
                sum += exp(activations_test[r * number_of_samples_test + c]);
            }
            for(int r = 0; r < W_rows; r++)
            {
                activations_test[r * number_of_samples_test + c] = exp(activations_test[r * number_of_samples_test + c]) / sum;
            }
        }

        // Todo 7
        // [Test] loss on test set
        loss_test = 0.;
        for(int i = 0; i < number_of_samples_test; i++)
        {
            for(int k = 0; k < W_rows; k++)
            {
                if(fabs(labels_test[i] - k) < 1e-6)
                    loss_test += log(activations_test[k * number_of_samples_test + i]);
            }
        }
        loss_test /= (-number_of_samples_test);

        // Todo 8
        // [Test] accuracy on test set
        correct = 0;
        res = -1;
        max = 0;
        for(int i = 0; i < number_of_samples_test; i++)
        {
            res = 0;
            max = activations_test[i];
            for(int k = 0; k < W_rows; k++)
            {
                if(activations_test[k * number_of_samples_test + i] > max)
                {
                    max = activations_test[k * number_of_samples_test + i];
                    res = k;
                }
            }
            if(res == labels_test[i]) correct++;
        }
        acc_test = correct / (float)number_of_samples_test;
        
        // Reset gradients
        memset(delta, 0, W_rows*number_of_samples_train*sizeof(float));
        memset(W_grad, 0, W_rows*W_columns*sizeof(float));

        // Todo 9
        // Compute delta
        // delta [10,n] = activations [10,n] - y [10,n]; (y[labels_train_set, :]=1)
        for(int c = 0; c < number_of_samples_train; c++)
        {
            for(int r = 0; r < W_rows; r++)
            {
                if(r == labels_train[c])
                    delta[r * number_of_samples_train + c] = activations_train[r * number_of_samples_train + c] - 1;
                else
                    delta[r * number_of_samples_train + c] = activations_train[r * number_of_samples_train + c];
            }
        }

        // Todo 10
        // W_grad [10,784] = delta [10,n] * images_train_set [n,784]
        for(int r = 0; r < W_rows; r++)
        {
            for(int c = 0; c < W_columns; c++)
            {
                for(int i = 0; i < number_of_samples_train; i++)
                {
                    W_grad[r * W_columns + c] += delta[number_of_samples_train * r + i] * data_train[W_columns * i + c];
                }
            }
        }

        // Todo 11
        // Update, W = W - alpha * W_grad;
        for(int r = 0; r < W_rows; r++)
        {
            for(int c = 0; c < W_columns; c++)
            {
                W[r * W_columns + c] -= learning_rate * W_grad[r * W_columns + c];
            }
        }

        printf("[CPU][Epoch %d]: Train loss:%f, Train accuracy:%f;   Test loss:%f, Test accuracy:%f ; time cost: %lf s\n", epoch, loss_train, acc_train, loss_test, acc_test, cpuSecond()-time_begin);
    }
    
    return 0;
}
