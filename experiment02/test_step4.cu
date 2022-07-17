#include<stdio.h>
__global__ void narcissistic_and_four_leaf_rose_numbers_gpu(int x1, int x2)
{
    // ToDo
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int idx = bid*blockDim.x+tid;
    if(idx>=x1 && idx<x2)
    {
    	int tho, hun, ten, ind;
    	tho = idx/1000;
    	hun = idx/100;
    	ten = (idx-hun*100)/10;
    	ind = idx%10;
    	if(tho>0)
    	{
    	    hun=(idx-tho*1000)/100;
    	    ten=(idx-tho*1000-hun*100)/10;
    	    ind=idx%10;
    	    if(idx==tho*tho*tho*tho + hun*hun*hun*hun + ten*ten*ten*ten + ind*ind*ind*ind)
    	    {
    	    	printf("%d  ", idx);
    	    }
    	}
    	else
    	{
    	   if(idx==hun*hun*hun + ten*ten*ten + ind*ind*ind)
    	    {
    	        printf("%d  ", idx);
    	    }
    	}
    }
}
 
void narcissistic_and_four_leaf_rose_numbers_cpu(int x1, int x2)
{
    int tho, hun, ten, ind;
    for(int i=x1; i<x2; i++)
    {
        tho = i/1000;
        hun = i/100;
        ten = (i-hun*100)/10;
        ind = i%10;
        if(tho>0)
        {
            hun=(i-tho*1000)/100;
            ten=(i-tho*1000-hun*100)/10;
            ind=i%10;
            if(i==tho*tho*tho*tho + hun*hun*hun*hun + ten*ten*ten*ten + ind*ind*ind*ind)
            {
                printf("%d  ", i);
            }
        }
        else
        {
            if(i==hun*hun*hun + ten*ten*ten + ind*ind*ind)
            {
                printf("%d  ", i);
            }
        }
    }
}
 
int main() 
{
    int numLowerBound = 100;
    int numUpperBound = 10000;
    printf("Narcissistic and four-leaf rose numbers from %d to %d (CPU version):\n", numLowerBound, numUpperBound);
    narcissistic_and_four_leaf_rose_numbers_cpu(numLowerBound, numUpperBound);
    printf("\nNarcissistic and four-leaf rose numbers from %d to %d (GPU version):\n", numLowerBound, numUpperBound);
    // ToDo (Tip: call the function narcissistic_and_four_leaf_rose_numbers_gpu)
    narcissistic_and_four_leaf_rose_numbers_gpu<<<10,1024>>>(numLowerBound, numUpperBound);
    cudaDeviceReset();
    return 0;
} 
