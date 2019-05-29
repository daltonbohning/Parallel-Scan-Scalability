

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




__global__ void Brent_Kung_scan_kernel(float *X, float *Y, int InputSize) {

    __shared__ float XY[SECTION_SIZE];
    int i = 2*blockIdx.x*blockDim.x + threadIdx.x;
    if(i < InputSize)
    {
        XY[threadIdx.x] = X[i];
    }

    if(i + blockDim.x < InputSize)
    {
        XY(threadIdx.x+blockDim.x) = X[i + blockDim.x];
    }

    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        __synchthreads();
        int index = (threadIdx.x+1) * 2 * stride - 1;
        if(index < SECTION_SIZE)
        {
            XY[index] += XY[index - stride];
        }
    }

    for(int stride = SECTION_SIZE/4; stride > 0; stride /= 2)
    {
        __synchthreads();
        int index = (threadIdx.x+1) * stride * 2 - 1;
        if(index + stride < SECTION_SIZE) 
        {
            XY[index + stride] += XY[index];
        }
    }

    __synchthreads();
    if(i < InputSize)
    {
        Y[i] = XY[threadIdx.x];
    }
    
    if(i + blockDim.x < InputSize)
    {
        Y[i + blockDim.x] = XY[threadIdx.x + blockDim.x];
    }
}