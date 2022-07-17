#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define N 10

int main()
{
    int *numbers = NULL;
    numbers = (int *)malloc(N*sizeof(int));
    printf("\nMemory allocated at address: %p\n", numbers);
    // memset(numbers, 0, N*sizeof(int));

    for(int i=0; i<N; i++){printf("%d ", numbers[i]);}

    printf("\n");

    char *strings1 = NULL;
    char *strings2 = NULL;
    strings1 = (char *)malloc(N*sizeof(char));
    printf("\nMemory allocated at address: %p\n", strings1);

    strings2 = (char *)malloc(N*sizeof(char));
    printf("\nMemory allocated at address: %p\n", strings2);

    // memset(strings1, 'A', N*sizeof(char));
    // memset(strings2, 'I', N*sizeof(char));

    for(int i=0; i<N; i++){printf("%c ", strings1[i]);}
    printf("\n");
    for(int i=0; i<N; i++){printf("%c ", strings2[i]);}

    memcpy(strings2, strings1, N*sizeof(char));

    printf("\nAfter memcpy:\n");
    for(int i=0; i<N; i++){printf("%c ", strings1[i]);}
    printf("\n");
    for(int i=0; i<N; i++){printf("%c ", strings2[i]);}

    
    free(numbers);
    free(strings1);
    free(strings2);

    return 0;
}
