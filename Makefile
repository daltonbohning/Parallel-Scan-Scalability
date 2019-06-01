# Makefile for parallel scan
# 
# Jordan Kremer
# Dalton Bohning

ARRAY_SIZE = 2000
SECTION_SIZE = 2048


#NVCC = nvcc -I. -lcuda -lcudart -lm
NVCC = nvcc -I.
NVCC_DEBUG = $(NVCC) -g -G

GCC = gcc -fopenmp -I.
GCC_DEBUG = $(GCC) -g


brent: brent-kung.cu
	$(NVCC_DEBUG) -DARRAY_SIZE=$(ARRAY_SIZE) -DSECTION_SIZE=$(SECTION_SIZE) -o brent-kung brent-kung.cu

brent_release: brent-kung.cu
	$(NVCC) -DARRAY_SIZE=$(ARRAY_SIZE) -DSECTION_SIZE=$(SECTION_SIZE) -o brent-kung brent-kung.cu


openmp: openmp_inclusiveScan.c
	$(GCC_DEBUG) -DARRAY_SIZE=$(ARRAY_SIZE) -o openmp_inclusiveScan openmp_inclusiveScan.c

openmp_release: openmp_inclusiveScan.c
	$(GCC) -DARRAY_SIZE=$(ARRAY_SIZE) -o openmp_inclusiveScan openmp_inclusiveScan.c
