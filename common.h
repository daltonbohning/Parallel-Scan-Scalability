/*
  common.c

  Jordan Kremer
  Dalton Bohning

  Contains code shared between brent-kung and openmp.
  This mostly includes testing and a C implementation.
*/

#include <stdio.h>
#include <malloc.h>
#include <stdbool.h>


void parallelScan_iterative(float *X, float *Y, int size) {
  float acc = X[0];
  Y[0] = acc;
  
  for (int i = 1; i < size; ++i) {
    acc += X[i];
    Y[i] = acc;
  }
}


bool verify(float *X, float *Y, int size){
  float *Y_ = (float*) malloc(size * sizeof(float));
  parallelScan_iterative(X, Y_, size);
  for (int i = 0; i < size; ++i){
    if (Y[i] != Y_[i]) {
      printf("Expected %.0f but got %.0f at Y[%d]\n", Y_[i], Y[i], i);
      free(Y_);
      return false;
    }
  }
  free(Y_);
  return true;
}


void printArray(float *A, int size){
  for(int i = 0; i < size; ++i) {
    printf("%.0f ", A[i]);
    if((i+1) % 10 == 0){
      printf("\n");
    }
  }
  printf("\n");
}


/* Every 100th index gets the value of 1.
   All others get the value 0.
   This is done to maintain the precision of floats,
   since the maximum precise integer that can be represented
   is 2^24 = 16,777,216.
   This allows the maximum array size to be 
   2^24 * 100 = 1,677,721,600, while maintaining precision. */
void initArray(float *A, int size) {
  for(int i = 0; i < size; ++i)
    A[i] = (i % 100 == 0) ? 1 : 0;
}

