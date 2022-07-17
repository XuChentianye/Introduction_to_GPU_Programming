#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define N 10
 
// Implement the function _hostMalloc
void _hostMalloc(void **p, size_t size)
{
	*p = malloc(size);
}
 
int main()
{
    int *numbers = NULL;
    _hostMalloc((void **)&numbers, N*sizeof(int));
    memset(numbers, 0, N*sizeof(int));
    for(int i=0; i<N; i++){ printf("%d ", numbers[i]); }
 
    printf("\n");
 
    char *strings = NULL;
    _hostMalloc((void **)&strings, N*sizeof(char));
    memset(strings, 'A', N*sizeof(char));
    for(int i=0; i<N; i++){ printf("%c ", strings[i]); }
 
    free(numbers);
    free(strings);
 
    return 0;
}