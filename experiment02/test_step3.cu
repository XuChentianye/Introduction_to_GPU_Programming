#include<stdio.h>
__global__ void prime_number_gpu(void)
{
    // ToDo
	int num = threadIdx.x;
	int is_prime = 1;
	if(num==0||num==1)
	{
		is_prime = -1;
	}
	for(int i=2; i<num; i++)
	{   
	    if(num%i==0)
	    {
	        is_prime = -1;
	        break;
	    }      
	}
	if(is_prime==1)
	{
	     printf("%d  ", num);
	}
}
 
void prime_number_cpu(int x)
{
    for(int p=2; p<x; p++)
    {
        int is_prime = 1;
        for(int i=2; i<p; i++)
        {   
            if(p%i==0)
            {
                is_prime = -1;
                break;
            }      
        }
        if(is_prime==1)
        {
            printf("%d  ", p);
        }
    }
}
 
int main() 
{
    int numUpperBound = 50;
    printf("CPU version:\n");
    prime_number_cpu(numUpperBound);
    printf("\nGPU version:\n");
    prime_number_gpu<<<1, numUpperBound>>>();
    cudaDeviceReset();
    return 0;
}
