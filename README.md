# Parallel Scan in CUDA and OpenMP
A scalability study of parallel scan using CUDA and OpenMP.
## Authors
[Dalton Bohning](https://github.com/daltonbohning)  
[Jordan Kremer](https://github.com/JordanKremer)
## How to Run
> The OpenMP version requires OpenMP to be installed and for "openmp" to be on the GCC path.
> The Brent-Kung version requires an NVIDIA CUDA-capable GPU with at least 2GB memory.
- `make benchmarks`
  - Runs benchmarks for all versions
- `make benchmarks VERSION=iterative`
  - Runs benchmarks for the single-threaded iterative version
- `make benchmarks VERSION=openmp_release`
  - Runs benchmakrs for the OpenMP version
- `make benchmarks VERSION=brent_release`
  - Runs benchmarks for the Brent-Kung version
> The output can be found in a local file `benchmarks.csv`
## CUDA (Brent-Kung)
- Tested and works on wino.cs.pdx.edu on array sizes up to 268,435,456 (2^28) for section sizes 1024 and 2048.
  - The GPUs have ~2GB of memory, and an array of 2^28 `float` uses ~1GB. It's safer to not approach 2GB since the GPU is shared with the display device, etc.
- The algorithm is done in-place on the device to conserve memory, and the output is copied to the output array, so a user would be oblivious to this.
  
## OpenMP
- Tested and works on wino.cs.pdx.edu (4 cores hyperthreaded) and babbage.cs.pdx.edu (30 cores) on array sizes up to 268,435,456 (2^28) for 2, 4, 8,16, and 32 threads.
- Modeled after https://www.cs.fsu.edu/~engelen/courses/HPC/Synchronous.pdf

## Testing
`make tests` runs various tests on the CUDA and OMP versions and compares it to the iterative version.

## Scalability Study
`make benchmarks` will produce a `.csv` with various array sizes, sections sizes of 1024 and 2048, and threads of 2, 4, 8, 16, and 32 for the CUDA, OMP, and iterative version.
