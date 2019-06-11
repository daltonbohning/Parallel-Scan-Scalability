# Parallel Scan in CUDA and OpenMP
## CUDA (Brent-Kung)
- Tested and works on the particle lab machines on array sizes up to 134,217,728 (2^27) for section sizes 1024 and 2048.
  - The GPUs have ~2GB of memory, and an array of 2^27 `doubles` uses ~1GB. It's safer to not approach 2GB since the GPU is shared with the display device, etc.
- `make tests` will run tests against various sizes and verify that the CUDA algorithm is correct.
- The algorithm is done in-place on the device to conserve memory, and the output is copied to the output array, so a user would be oblivious to this.
  
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
