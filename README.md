# Parallel Scan in CUDA and OpenMP
## CUDA (Brent-Kung)
Currently tested and works on array sizes up to 134,217,728 (2^27) for section sizes 1024 and 2048.\
These tests work on the machines in the particle lab. The GPUs have ~2GB of memory, and an array of 2^27 `doubles` uses ~1GB. It's safer to not approach 2GB since the GPU is shared with the display device, etc.\
`make tests` will run tests against various sizes and verify that the CUDA algorithm is correct.
  
## OpenMP
Not yet implemented.

## Scalability Study
`make benchmarks` will produce a `.csv` with various array sizes and sections sizes of 1024 and 2048. This will be useful.

## Research Paper review
Not yet done.

## TODO
- Hook testing into OpenMP
- Implement in OpenMP
- Include benchmarks for OpenMP
- Compile results for study
- Read a paper
- Write the report
