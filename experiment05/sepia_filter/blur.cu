#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"
#include"error_check.h"
#include"time_helper.h"

// Todo
// Implement the cuda kernel function ***rgb_to_sepia_gpu***
__global__ void blur_gpu(unsigned char *input_image, unsigned char *output_image, int width, int height, int channels, int blur_size)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	if(col<width && row<height)
	{
		int c1Val = 0;
		int c2Val = 0;
		int c3Val = 0;
		int pixels = 0;
		// get the average of the surrounding blur_size * blur_size box
		for(int blurRow = -blur_size; blurRow <= blur_size; ++blurRow)
		{
			for(int blurCol = -blur_size; blurCol <= blur_size; ++blurCol)
			{
				int curRow = row + blurRow;
				int curCol = col + blurCol;
				int offset = (curRow*width + curCol)*channels;
				// verify we have a valid image pixel
				if(curRow > -1 && curRow < height && curCol > -1 && curCol < width)
				{
					c1Val += input_image[offset];
					c2Val += input_image[offset+1];
					c3Val += input_image[offset+2];
					pixels++;
				}
			}
		}
		int offset2 = (row*width + col)*channels;
		// write our new pixel value out
		*(output_image + offset2) = (unsigned char)(c1Val / pixels);
		*(output_image + offset2 + 1) = (unsigned char)(c2Val / pixels);
		*(output_image + offset2 + 2) = (unsigned char)(c3Val / pixels);
	}
}

void blur_cpu(unsigned char *input_image, unsigned char *output_image, int width, int height, int channels, int blur_size)
{
    for(int row=0; row<height; row++)
    {
        for(int col=0; col<width; col++)
        {
        	int c1Val = 0;
        	int c2Val = 0;
        	int c3Val = 0;
        	int pixels = 0;
        	// get the average of the surrounding blur_size * blur_size box
        	for(int blurRow = -blur_size; blurRow <= blur_size; ++blurRow)
        	{
        		for(int blurCol = -blur_size; blurCol <= blur_size; ++blurCol)
        		{
        			int curRow = row + blurRow;
        			int curCol = col + blurCol;
        			int offset = (curRow*width + curCol)*channels;
        			// verify we have a valid image pixel
        			if(curRow > -1 && curRow < height && curCol > -1 && curCol < width)
        			{
        				c1Val += input_image[offset];
        				c2Val += input_image[offset+1];
        				c3Val += input_image[offset+2];
        				pixels++;
        			}
        		}
        	}
        	int offset2 = (row*width + col)*channels;
        	// write our new pixel value out
        	*(output_image + offset2) = (unsigned char)(c1Val / pixels);
        	*(output_image + offset2 + 1) = (unsigned char)(c2Val / pixels);
        	*(output_image + offset2 + 2) = (unsigned char)(c3Val / pixels);
        }
    }
}


int main(int argc, char *argv[])
{
    if(argc<4)
    {
        printf("Usage: command    input-image-name    output-image-name option   option(cpu/gpu)");
        return -1;
    }
    char *input_image_name = argv[1];
    char *output_image_name = argv[2];
    char *option = argv[3];

    int width, height, original_no_channels;

    // deal with the problem that a PNG image may have four channels.
    // Although we only need rgb information here, if we set desired_no_channels as 3,
    // we would find that when we apply this programme on the png image which has 4 channels,
    // the output result is confusing.
    int desired_no_channels;
    unsigned char *temp = stbi_load(input_image_name, &width, &height, &desired_no_channels, 3);
    stbi_image_free(temp);
    // now, desired_no_channels gets the right value!

    unsigned char *stbi_img = stbi_load(input_image_name, &width, &height, &original_no_channels, desired_no_channels);
    if(stbi_img==NULL){ printf("Error in loading the image.\n"); exit(1);}
    printf("Loaded image with a width of %dpx, a height of %dpx. The original image had %d channels, the loaded image has %d channels.\n", width, height, original_no_channels, desired_no_channels);

    int channels = original_no_channels;
    int img_mem_size = width * height * channels * sizeof(char);
    double begin;
    if(strcmp(option, "cpu")==0)
    {
        printf("Processing with CPU!\n");
        unsigned char *sepia_img = (unsigned char *)malloc(img_mem_size);
        if(sepia_img==NULL){  printf("Unable to allocate memory for the sepia image. \n");  exit(1);  }

        
        // Time stamp
		begin = cpuSecond();

		// CPU computation (for reference)
		blur_cpu(stbi_img, sepia_img, width, height, channels, 15);

        // Time stamp
		printf("Time cost [CPU]:%f s\n", cpuSecond()-begin);

        // Save to an image file
        stbi_write_jpg(output_image_name, width, height, channels, sepia_img, 100);

        free(sepia_img);
    }
    else if(strcmp(option, "gpu")==0) 
    {
        printf("Processing with GPU!\n");

        //  Todo: 1. Allocate memory on GPU
        unsigned char *d_sepia_img = NULL;
        unsigned char *d_stbi_img =NULL;
        unsigned char *sepia_img_from_gpu = (unsigned char *)malloc(img_mem_size);
        CHECK(cudaMalloc((void **)&d_sepia_img, img_mem_size));
        CHECK(cudaMalloc((void **)&d_stbi_img, img_mem_size));

        //  Todo: 2. Copy data from host memory to device memory
        CHECK(cudaMemcpy(d_stbi_img, stbi_img, img_mem_size, cudaMemcpyHostToDevice));

        //  Todo: 3. Call kernel function
        //        3.1 Declare block and grid sizes
        dim3 block(16, 16);
        dim3 grid(ceil(width/16.0), ceil(height/16.0));
        //	PS: here we should use 16.0 (float) rather than 16 (int), thus the result
        //	will be in the type of float rather than int.

		//        3.2 Record the time cost of GPU computation
		begin = cpuSecond();

		//  Todo: 3.3 Call the kernel function (Don't forget to call cudaDeviceSynchronize() before time recording)
		blur_gpu<<<grid, block>>>(d_stbi_img, d_sepia_img, width, height, channels, 15);
		CHECK(cudaDeviceSynchronize());
		printf("Time cost [GPU]:%f s\n", cpuSecond()-begin);

		//  Todo:  4. Copy data from device to host
		CHECK(cudaMemcpy(sepia_img_from_gpu, d_sepia_img, img_mem_size, cudaMemcpyDeviceToHost));

		//  Todo:  5. Save results as an image
        stbi_write_jpg(output_image_name, width, height, channels, sepia_img_from_gpu, 100);

        //  Todo:  6. Release host memory and device memory
        free(sepia_img_from_gpu);
        CHECK(cudaFree(d_sepia_img));
        CHECK(cudaFree(d_stbi_img));
    } 
    else
    {
        printf("Unexpected option (please use cpu/gpu) !\n");
    }   

    stbi_image_free(stbi_img);

    return 0;
}



// we have 3 channels corresponding to RGB
// The input image is encoded as unsigned characters [0, 255]
__global__
void colorToGreyscaleConversion(unsigned char * grayImage, unsigned char * rgbImage,
int width, int height) {
int Col = threadIdx.x + blockIdx.x * blockDim.x;
int Row = threadIdx.y + blockIdx.y * blockDim.y;
if (Col < width && Row < height) {
    // get 1D coordinate for the grayscale image
    int greyOffset = Row*width + Col;
    // one can think of the RGB image having
    // THREE times as many columns of the gray scale image
    int rgbOffset = 3 * greyOffset;
    unsigned char r = rgbImage[rgbOffset ]; // red value for pixel
    unsigned char g = rgbImage[rgbOffset + 1]; // green value for pixel
    unsigned char b = rgbImage[rgbOffset + 2]; // blue value for pixel
    // perform the rescaling and store it
    // We multiply by floating point constants
    grayImage[greyOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
    }
}
