
/*
brent-kung.cu

Jordan Kremer
Dalton Bohning

Usage:
    Flags:
        -DARRAY_SIZE
        -SECTION_SIZE 

    Ex:
        nvcc -DARRAY_SIZE=2000 -DSECTION_SIZE=2048 -o brent-kung brent-kung.cu

    Note:
        Section size should not exceed 2048 
*/


#include <cuda.h>
#include <stdio.h>


//#define SECTION_SIZE 100
//#define ARRAY_SIZE 100


#define handleError(CUDA_FUNCTION) {\
    cudaError_t THE_ERROR = (cudaError_t) CUDA_FUNCTION;\
    if (THE_ERROR != cudaSuccess) \
    {\
        printf("%s in %s at line %d\n", cudaGetErrorString(THE_ERROR),__FILE__,__LINE__);\
        exit(EXIT_FAILURE);\
    }\
}



//Credit: https://github.com/aramadia/udacity-cs344/blob/master/Unit2%20Code%20Snippets/gputimer.h
struct GpuTimer
{
      cudaEvent_t start;
      cudaEvent_t stop;
 
      GpuTimer()
      {
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
      }
 
      ~GpuTimer()
      {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
      }
 
      void Start()
      {
            cudaEventRecord(start, 0);
      }
 
      void Stop()
      {
            cudaEventRecord(stop, 0);
      }
 
      float Elapsed()
      {
            float elapsed;
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&elapsed, start, stop);
            return elapsed;
      }
};

GpuTimer timer_kernelExecution;
GpuTimer timer_kernelTotal;


//An iterative version of parallel scan addition
__host__
void sequential_scan(float *X, float *Y){
  int i = 1, acc = Y[0] = X[0];
  
  while (i < ARRAY_SIZE) 
    Y[i++] = acc = acc + X[i];
}

//Runs the iterative version and verifies the results
__host__
bool verify(float *X, float *Y){
  float *Y_ = (float*) malloc(ARRAY_SIZE * sizeof(float));
  sequential_scan(X, Y_);
  for (int i = 0; i < ARRAY_SIZE; ++i){
    if (Y[i] != Y_[i]) {
      printf("Expected %.0f but got %.0f at Y[%d]\n", Y_[i], Y[i], i);
      return false;
    }
  }
  free(Y_);
  return true;
}

__global__ 
void Brent_Kung_scan_kernel(float *X, float *Y)
{
    __shared__ float XY[SECTION_SIZE];

    int i = 2*blockIdx.x*blockDim.x + threadIdx.x;
    if(i < ARRAY_SIZE)
    {
        XY[threadIdx.x] = X[i];
    }

    if(i + blockDim.x < ARRAY_SIZE)
    {
        XY[threadIdx.x+blockDim.x] = X[i + blockDim.x];
    }

    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        __syncthreads();
        int index = (threadIdx.x+1) * 2 * stride - 1;
        if(index < SECTION_SIZE)
        {
            XY[index] += XY[index - stride];
        }
    }

    for(int stride = SECTION_SIZE/4; stride > 0; stride /= 2)
    {
        __syncthreads();
        int index = (threadIdx.x+1) * stride * 2 - 1;
        if(index + stride < SECTION_SIZE) 
        {
            XY[index + stride] += XY[index];
        }
    }

    __syncthreads();
    if(i < ARRAY_SIZE)
    {
        Y[i] = XY[threadIdx.x];
    }
    
    if(i + blockDim.x < ARRAY_SIZE)
    {
        Y[i + blockDim.x] = XY[threadIdx.x + blockDim.x];
    }

}


void inclusive_scan(float *host_X, float *host_Y)
{
    float *X, *Y;
    int mallocSize = ARRAY_SIZE * sizeof(float);

    timer_kernelTotal.Start();

    handleError(cudaMalloc((void **)&X, mallocSize));
    handleError(cudaMalloc((void **)&Y, mallocSize));

    handleError(cudaMemcpy(X, host_X, mallocSize, cudaMemcpyHostToDevice));
   
    //Book says SECTION_SIZE/2 OK, but not sure about
    //other dimensions and blocks per grid
    dim3 threadsPerBlock(SECTION_SIZE/2, 1, 1);
    dim3 blocksPerGrid(100,1,1);

    timer_kernelExecution.Start();
    Brent_Kung_scan_kernel<<<blocksPerGrid, threadsPerBlock>>>(X, Y);
    timer_kernelExecution.Stop();

    handleError(cudaMemcpy(host_Y, Y, mallocSize, cudaMemcpyDeviceToHost));
    handleError(cudaFree(X));
    handleError(cudaFree(Y));

    timer_kernelTotal.Stop();
}

void printArray(float *A){
  for(int i = 0; i < ARRAY_SIZE; ++i) {
    printf("%.0f ", A[i]);
    if((i+1) % 10 == 0){
      printf("\n");
    }
  }
  printf("\n");
}

int main(void)
{
    float *host_X = (float*) malloc(ARRAY_SIZE * sizeof(float));
    float *host_Y = (float*) malloc(ARRAY_SIZE * sizeof(float));

    for(int i = 0; i < ARRAY_SIZE; ++i)
    {
        host_X[i] = i + i %4; //change
    }


    inclusive_scan(host_X, host_Y);

    //Make sure the results are correct
    if (1) {
      printArray(host_Y);
      if (verify(host_X, host_Y))
        printf("ALL CORRECT!\n");
      else
        printf("FAIL!\n");
    }

    float kernelExec = timer_kernelExecution.Elapsed();
    float kernelTotal = timer_kernelTotal.Elapsed();
    float kernelMem = kernelTotal - kernelExec;

    printf("Kernel Execution (ms): %f\n", kernelExec);
    printf("Kernel Memory (ms):    %f\n", kernelMem);
    printf("Kernel Total (ms):     %f\n", kernelTotal);

    free(host_X);
    free(host_Y);
}
