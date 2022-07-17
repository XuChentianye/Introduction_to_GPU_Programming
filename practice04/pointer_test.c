#include <stdio.h>
int main(){
	int a =1;
	int b = 6;
	int *p1 = &a;
	int *p2 = &b;
	int **p3 = &p1;
	*p3 = p2;
	printf("*p1: %d \n",*p1);
	printf("*p2: %d \n",*p2);
	printf("**p3: %d \n",**p3);
	printf("a: %d \n",a);
	printf("b: %d \n",b);
	return 0;
}