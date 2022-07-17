#ifndef __ERRORCHECK_H__
#define __ERRORCHECK_H__

#include<stdio.h>

#define CHECK(call) \
	do{ cudaError_t error_code = call; \
		if(error_code==cudaSuccess) \
		{ \
			printf("\n"); \
			printf("##CUDA API successed!\n"); \
		} \
		else \
		{ \
			printf("\n"); \
			printf("##CUDA Error:\n"); \
			printf("  File: %s\n", __FILE__); \
			printf("  Line: %d\n", __LINE__); \
			printf("  Error info: %s \n", cudaGetErrorString(error_code)); \
		} \
	}while(0);

#endif
