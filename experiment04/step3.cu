#include"error_check.h"
#include <stdio.h>
#define PWD 3

__global__ void entrypt_p(char *in, char *out)
{
    int idx = threadIdx.x;
    out[idx] = in[idx] + PWD;
}

int main()
{
    char A[] = "Hello, world!";
    int memSize = strlen(A)*sizeof(char);
    int strLen = strlen(A);
    char *B = (char *)malloc(memSize);
    printf("Input: \n");
    for(int i=0; i<strLen; i++){printf("%c ", A[i]);}

    //ToDo
    char *d_A=NULL;
    char *d_B=NULL;
    CHECK(cudaMalloc((void **)&d_B, memSize));
    CHECK(cudaMalloc((void **)&d_A, memSize));
    CHECK(cudaMemcpy(d_A, A, memSize, cudaMemcpyHostToDevice));
    entrypt_p<<<1,strlen(A)>>>(d_A, d_B);
    CHECK(cudaMemcpy(B, d_B, memSize, cudaMemcpyDeviceToHost));
    printf("\nOutput: \n");
    for(int i=0; i<strLen; i++){printf("%c ", B[i]);}
    printf("\n");
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    return 0;
}
