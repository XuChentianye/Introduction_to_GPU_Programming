#ifndef _UTILS_H_
#define _UTILS_H_


#define RAND_FLOAT() (((float) rand()) / ((float) RAND_MAX))

void weight_initialization(float* W, int W_rows, int W_columns)
{
    int n = W_rows*W_columns;
    for(int i=0; i<n; i++)
    {
        W[i] = RAND_FLOAT();
    }
}

void matrix_transpose(float *input, float *output, int num_rows, int num_cols)
{
	for(int row_idx=0; row_idx<num_rows; row_idx++)
	{
		for(int col_idx=0; col_idx<num_cols; col_idx++)
		{
			output[col_idx*num_rows+row_idx] = input[row_idx*num_cols+col_idx];
		}
	}
}

void matrix_transpose(float *input, float *output, int num_rows, int num_cols);

void matrix_multiply(float *M, float *N, float *P, int M_rows, int M_cols, int N_rows, int N_cols);

void softmax(float* activations, int rows, int cols);

float accuracy(float* activations, int* labels, int rows, int cols);

float cross_entropy_loss(float* activations, int* labels, int rows, int cols);

void delta_compute(float* delta, float* activations, int *labels, int W_rows, int number_of_samples);

void matrix_add(float *input, float *output, int num_rows, int num_cols, float alpha);

#endif