#include"error_check.h"
#include<stdio.h>
__global__ void test(){}
int main() {
cudaDeviceProp prop;
CHECK(cudaGetDeviceProperties(&prop, 1));
test<<<1, 1025>>>();
CHECK(cudaGetLastError());
return 0;
}
