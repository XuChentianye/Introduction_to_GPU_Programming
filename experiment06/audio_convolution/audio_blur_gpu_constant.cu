#include<stdio.h>
#include<math.h>
#include<sndfile.h>

const int GAUSSIAN_SIDE_WIDTH = 10;
const int GAUSSIAN_SIZE = 2*GAUSSIAN_SIDE_WIDTH + 1;
const float PI = 3.14159265358979;

/*
* 1. Using the following command in terminal to install dependencies:
*          sudo apt-get install libsndfile1-dev -y
*
* 2. Using the following command to compile the code
*          nvcc -o audio_blur audio_blur.cu -lsndfile
*/

__constant__ float filter_con[GAUSSIAN_SIZE];

float gaussian(const int x, const float mu, const float sigma)
{
	float factor1 = sigma*sqrt(2*PI);
	float factor2 = pow((x-mu), 2);
	float factor3 = 2*pow(sigma, 2);
	float result = (1/factor1) * exp(-1.0 * (factor2 / factor3));
	return result;
}

void gaussian_filter(const float mu, const float sigma)
{
	
	float *filter = (float *)malloc(GAUSSIAN_SIZE*sizeof(float));
	for(int i=-GAUSSIAN_SIDE_WIDTH; i<=GAUSSIAN_SIDE_WIDTH; i++)
	{
		filter[i+GAUSSIAN_SIDE_WIDTH] = gaussian(i, mu, sigma);
	}

	float total = 0.0;
	for(int i=0; i<GAUSSIAN_SIZE; i++)
	{
		total += filter[i];
	}

	for(int i=0; i<GAUSSIAN_SIZE; i++)
	{
		filter[i] /= total;
	}

	cudaMemcpyToSymbol(filter_con, filter, GAUSSIAN_SIZE*sizeof(float));
	free(filter);
}

__global__ void h_convolution_gpu(float *single_channel_output,
		const float *single_channel_input, const long n_frames,
		const int filter_size)
{
	int threadNum = gridDim.x * blockDim.x;
	int round = n_frames / threadNum +1;
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	for(int n=0; n<=round; n++){
		int i_real = i + n * threadNum;
		if(i_real < filter_size)
		{
			for (int j = 0; j <= i_real; j++)
			{
				single_channel_output[i_real] += single_channel_input[i_real-j]
										* filter_con[j];
			}
		}
		if(i_real >= filter_size && i_real < n_frames)
		{
			for (int j = 0; j < filter_size; j++)
			{
				single_channel_output[i_real] += single_channel_input[i_real-j]
										* filter_con[j];
			}
		}
	}
}

int main(int argc, char *argv[])
{
	if(argc!=3)
	{
		printf("Arguments: <input file name> <output file name> ");
		return -1;
	}
	const char *input_file_name = argv[1]; // "test.wav"; // "example_test.wav";
	const char *output_file_name = argv[2];  // "output.wav"; // "example_test.wav";

	SNDFILE *in_file_handle;
	SF_INFO in_file_info;
	
	int amt_read;
	printf("Reading %s ...\n", input_file_name);
	in_file_handle = sf_open(input_file_name, SFM_READ, &in_file_info);
	if(!in_file_handle)
	{
		printf("Open file failed!");
		exit(1);
	}	
	long n_frames = in_file_info.frames;
	int n_channels = in_file_info.channels;

	size_t data_size = sizeof(float)*n_frames*n_channels;
	float *all_channel_input = (float *)malloc(data_size);
	amt_read = sf_read_float(in_file_handle, all_channel_input, n_frames*n_channels);
	// assert(amt_read == in_file_info.frames*in_file_info.channels);
	if(amt_read != n_frames*n_channels)
	{
		printf("Error occurs during file reading! \n");
	}

	printf("n_frames:%ld, n_channels:%d\n", n_frames, n_channels);
	sf_close(in_file_handle);
	
	
	/* Filters */
	printf("Gaussian filter: \n");
	int mu = 0;
	int sigma = 5;
	// float *filter = gaussian_filter(mu, sigma);
	gaussian_filter(mu, sigma);

	float *filter1 = (float *)malloc(GAUSSIAN_SIZE*sizeof(float));
	cudaMemcpyFromSymbol(filter1, filter_con, GAUSSIAN_SIZE*sizeof(float));
	for(int i=0; i<GAUSSIAN_SIZE; i++){ printf("%f ", filter1[i]); }
	free(filter1);

	printf("\n Convolution ... \n");
	// Split Channels
	float *all_channel_output = (float *)malloc(data_size);
	
	float *single_channel_input = (float *)malloc(n_frames*sizeof(float));
	float *single_channel_output = (float *)malloc(n_frames*sizeof(float));

	for(int ch=0; ch<n_channels; ch++)
	{
		printf("Processing channel %d ... \n", ch);
		for(int i=0; i<n_frames; i++)
		{
			single_channel_input[i] = all_channel_input[(i*n_channels)+ch];
		}
		memset(single_channel_output, 0, n_frames*sizeof(float));

		dim3 block(128,1,1);
		dim3 grid(64,1,1);
		float *d_single_channel_input = NULL;
		float *d_single_channel_output = NULL;
		cudaMalloc((void **)&d_single_channel_input,n_frames*sizeof(float));
		cudaMalloc((void **)&d_single_channel_output,n_frames*sizeof(float));
		cudaMemcpy(d_single_channel_input, single_channel_input,n_frames*sizeof(float),cudaMemcpyHostToDevice);
		cudaMemcpy(d_single_channel_output, single_channel_output,n_frames*sizeof(float),cudaMemcpyHostToDevice);

		cudaEvent_t start, stop;
		float time;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);

		// Convolution using GPU
		h_convolution_gpu<<<grid, block>>>(d_single_channel_output,d_single_channel_input, n_frames, GAUSSIAN_SIZE);

		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time,start,stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);

		cudaMemcpy(single_channel_output,d_single_channel_output,n_frames*sizeof(float),cudaMemcpyDeviceToHost);
		cudaFree(d_single_channel_input);
		cudaFree(d_single_channel_output);

		for(int i=0; i<n_frames; i++)
		{
			all_channel_output[(i*n_channels)+ch] = single_channel_output[i];
		}

		printf("For ch = %d, the time elapsed is %f ms.\n",ch,time);
	}

	// Write to file
	SNDFILE *out_file_handle;
	SF_INFO out_file_info;
	
	out_file_info = in_file_info;
	out_file_handle = sf_open(output_file_name, SFM_WRITE, &out_file_info);
	if(!out_file_handle)
	{
		printf("Output failed!");
		exit(1);
	}
	sf_write_float(out_file_handle, all_channel_output, amt_read);
	sf_close(out_file_handle);

	printf("Results have been saved to %s \n", output_file_name);

	free(all_channel_input);
	free(all_channel_output);
	free(single_channel_input);
	free(single_channel_output);

	return 0;
}
