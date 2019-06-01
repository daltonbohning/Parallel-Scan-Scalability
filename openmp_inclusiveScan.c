/*
openmp_inclusiveScan.c

Jordan Kremer
Dalton Bohning

Usage:
    gcc -fopenmp -DARRAY_SIZE=someNumber -o openmp_inclusiveScan openmp_inclusiveScan.c
*/


#include <stdlib.h>
#include <stdio.h>


void inclusive_scan(float *X, float *Y)
{

/*
#pragma parallel for 
    //private, shared
    //atomic
    //critical
#pragma reduction()
#pragma parallel region


*/
}


int main(void)
{
    float *X = (float*) malloc(ARRAY_SIZE * sizeof(float));
    float *Y = (float*) malloc(ARRAY_SIZE * sizeof(float));

    for(int i = 0; i < ARRAY_SIZE; ++i)
    {
        X[i] = i; //change
    }

    inclusive_scan(X, Y);

    for(int i = 0; i < ARRAY_SIZE; ++i)
    {
        printf("%.0f ", Y[i]);
        if((i+1) % 10 == 0){
            printf("\n");
        }
    }
    printf("\n");
}
