#ifndef _MNIST_HELPER_
#define _MNIST_HELPER_

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>


uint32_t map_uint32(uint32_t in)
{
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    return (
        ((in & 0xFF000000) >> 24) |
        ((in & 0x00FF0000) >>  8) |
        ((in & 0x0000FF00) <<  8) |
        ((in & 0x000000FF) << 24)
    );
#else
    return in;
#endif
}

void scale_pixels(float *pixels, int n)
{
	for (int i = 0; i < n; i++) {
		pixels[i] /= 255.0f; 
	}
}

void get_labels(const char *path, int **labels_p, int *number_of_labels)
{
    FILE *stream;
    uint32_t magic_number;
	uint32_t num_of_items;
    uint8_t *label_data;

    stream = fopen(path, "rb");

    if (NULL == stream) {
        fprintf(stderr, "Could not open file: %s\n", path);
    }

    if (1 != fread(&magic_number, sizeof(uint32_t), 1, stream)) {
        fprintf(stderr, "Could not read file header from: %s\n", path);
        fclose(stream);
        return;
    }
	magic_number = map_uint32(magic_number);

	if (1 != fread(&num_of_items, sizeof(uint32_t), 1, stream)) {
        fprintf(stderr, "Could not read file header from: %s\n", path);
        fclose(stream);
        return;
    }
	num_of_items = map_uint32(num_of_items);

    *number_of_labels = (int)num_of_items;
	printf("[Header] magic number: %08X\n", magic_number);
	printf("[Header] number of images: %d\n", num_of_items);
    label_data = (uint8_t * )malloc(*number_of_labels * sizeof(uint8_t));
	*labels_p = (int * )malloc(*number_of_labels * sizeof(int));

    if (label_data == NULL) {
        fprintf(stderr, "Could not allocated memory for %d labels\n", *number_of_labels);
        fclose(stream);
        return;
    }

    if (*number_of_labels != fread(label_data, sizeof(uint8_t), *number_of_labels, stream)) {
        fprintf(stderr, "Could not read %d labels from: %s\n", *number_of_labels, path);
        free(labels_p);
        fclose(stream);
        return;
    }
	for(int i=0; i<*number_of_labels; i++)
	{
		(*labels_p)[i] = (int)label_data[i];
	}

	printf("%s loaded successfully!\n", path);

	free(label_data);
    fclose(stream);
}

void get_images(const char *path, float **images_p, int *number_of_images, int *rows_p, int *columns_p)
{
    FILE *stream;
    uint32_t magic_number;
	uint32_t num_of_items;
	uint32_t rows;
	uint32_t columns;
    uint8_t *image_data;
    int num_of_all_pixles;

    stream = fopen(path, "rb");

    if (NULL == stream) {
        fprintf(stderr, "Could not open file: %s\n", path);
    }
	// magic number
    if (1 != fread(&magic_number, sizeof(uint32_t), 1, stream)) {
        fprintf(stderr, "Could not read file header from: %s\n", path);
        fclose(stream);
        return;
    }
	magic_number = map_uint32(magic_number);

	// number of images
	if (1 != fread(&num_of_items, sizeof(uint32_t), 1, stream)) {
        fprintf(stderr, "Could not read file header from: %s\n", path);
        fclose(stream);
        return;
    }
	num_of_items = map_uint32(num_of_items);

	// rows
	if (1 != fread(&rows, sizeof(uint32_t), 1, stream)) {
        fprintf(stderr, "Could not read file header from: %s\n", path);
        fclose(stream);
        return;
    }
	rows = map_uint32(rows);

	//columns 
	if (1 != fread(&columns, sizeof(uint32_t), 1, stream)) {
        fprintf(stderr, "Could not read file header from: %s\n", path);
        fclose(stream);
        return;
    }
	columns = map_uint32(columns);

	*number_of_images = (int)num_of_items;
	*rows_p = (int)rows;
    *columns_p = (int)columns;
    num_of_all_pixles = num_of_items*rows*columns;

	printf("[Header] magic number: %08X\n", magic_number);
	printf("[Header] number of images: %d\n", num_of_items);
	printf("[Header] rows: %d\n", rows);
	printf("[Header] columns: %d\n", columns);

    image_data = (uint8_t * )malloc(num_of_all_pixles * sizeof(uint8_t));

    if (image_data == NULL) {
        fprintf(stderr, "Could not allocated memory for %d images\n", *number_of_images);
        fclose(stream);
        return;
    }
    
    // read image data
    if (*number_of_images != fread(image_data, rows * columns * sizeof(uint8_t), *number_of_images, stream)) {
        fprintf(stderr, "Could not read %d images from: %s\n", *number_of_images, path);
        free(images_p);
        fclose(stream);
        return;
    }

	*images_p = (float * )malloc(num_of_all_pixles * sizeof(float));
	for(int i=0; i<num_of_all_pixles; i++)
	{
		(*images_p)[i] = (float)image_data[i];
	}

	printf("%s loaded successfully!\n", path);
    printf("[Return] number_of_images: %d\n", *number_of_images);
	free(image_data);
    fclose(stream);
}

void get_dataset(const char *image_path, const char *label_path, float **images_p, int **labels_p, int *number_of_items, int *rows_p, int *columns_p)
{
	int number_of_images;
	get_images(image_path, images_p, &number_of_images, rows_p, columns_p);

    int number_of_labels;
	get_labels(label_path, labels_p, &number_of_labels);

    if(number_of_images!=number_of_labels)
    {   
        fprintf(stderr, "The number of images not equal to the number of labels.\n");
        return;
    }
    *number_of_items = number_of_images;
}


void print_samples(const float *images, const int *labels, int rows, int columns, int true_value)
{
    int offset = rows*columns;
    for(int idx=9000; idx<9001; idx++)
    {
        // printf("label: %d\n", labels[idx]);
        for(int i=0; i<rows; i++)
        {
            for(int j=0; j<columns; j++)
            {
                if(true_value==1){
                    printf("%f ", *(images+i*columns+j + idx*offset));
                }
                else{
                    printf("%s ", (*(images+i*columns+j + idx*offset)>0)?"#":" ");
                }
            }
            printf("\n");
        }
    }
    
}

#endif