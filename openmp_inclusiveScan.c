/*
openmp_inclusiveScan.c

Jordan Kremer
Dalton Bohning

Usage:
    gcc -fopenmp -DARRAY_SIZE=1000 -DNUM_THREADS=8 -o openmp_inclusiveScan openmp_inclusiveScan.c
*/


#include <stdlib.h>
#include <stdio.h>
#include "omp.h"
#include <time.h>

#include "common.h"


//See:  https://www.cs.fsu.edu/~engelen/courses/HPC/Synchronous.pdf
void inclusive_scan(float *x, float *z)
{

int nthreads, tid, work, lo, hi, i, j, n, temp;

n = ARRAY_SIZE;
omp_set_num_threads(NUM_THREADS);

#pragma omp parallel shared(n, nthreads, x, z) private(i, j, tid, work, lo, hi, temp)
{
    //printf("\n Num threads: %i \n", omp_get_num_threads());
    #pragma omp single
    	nthreads = omp_get_num_threads(); //assumes nthreads = 2^k
    tid = omp_get_thread_num();
    work = (n + nthreads-1) / nthreads;
    lo = work * tid;
    hi = lo + work;
    if (hi > n)
    {
    	hi = n;
    }

    //printf("\n THREAD: %i   WORK: %i  LOW: %i  HI: %i", tid, work, lo, hi);

    for(i = lo+1; i < hi; i++)
    {
        x[i] = x[i] + x[i-1];
    }
    z[tid] = x[hi-1];
    #pragma omp barrier
    for (j = 1; j <nthreads; j = 2*j)
    {
      if (tid >= j)
        temp = z[tid-j];
      #pragma omp barrier
      if (tid >= j)
	    {
        z[tid] = z[tid] + temp;
	    }
      #pragma omp barrier
    }

    for (i = lo; i < hi; i++)
      x[i] = x[i] + z[tid] - x[hi-1];
}

}



int main(void)
{
    float *x = (float*) malloc(ARRAY_SIZE * sizeof(float));
    float *x_ = (float*) malloc(ARRAY_SIZE * sizeof(float));
    float *z = (float*) malloc(NUM_THREADS * sizeof(float));


    initArray(x, ARRAY_SIZE);
    initArray(x_, ARRAY_SIZE);
    /*
    for(int i = 0; i < ARRAY_SIZE; ++i)
    {
        x[i] = i; //change
        x_[i] = x[i];
    }
    */

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    inclusive_scan(x, z);

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    float execTime = (double) (end.tv_sec - start.tv_sec) * 1000 + (double) (end.tv_nsec - start.tv_nsec) / 1000000;

#if defined(PRINT_RESULTS)
    printArray(x, ARRAY_SIZE);
#endif
#if defined(VERIFY_RESULTS)
    if (verify(x_, x, ARRAY_SIZE))
      printf("ALL CORRECT!\n");
    else
      printf("FAIL!\n");
#endif


    printf("Execution (ms): %f\n", execTime);
    
    free(x);
    free(x_);
    free(z);
}
