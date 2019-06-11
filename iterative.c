/*
  iterative.c

  Jordan Kremer
  Dalton Bohning
*/


#include <stdio.h>
#include <time.h>
#include <stdint.h>

#include "common.h"



int main(void)
{
    float *X = (float*) malloc(ARRAY_SIZE * sizeof(float));
    float *Y = (float*) malloc(ARRAY_SIZE * sizeof(float));

    initArray(X, ARRAY_SIZE);


    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
        
    parallelScan_iterative(X, Y, ARRAY_SIZE);

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    float execTime = (double) (end.tv_sec - start.tv_sec) * 1000 + (double) (end.tv_nsec - start.tv_nsec) / 1000000;


    //Make sure the results are correct
#if defined(PRINT_RESULTS)
    printArray(Y, ARRAY_SIZE);
#endif
#if defined(VERIFY_RESULTS)
    if (verify(X, Y, ARRAY_SIZE))
      printf("ALL CORRECT!\n");
    else
      printf("FAIL!\n");
#endif

    printf("Execution (ms): %f\n", execTime);
    
    free(X);
    free(Y);
}

