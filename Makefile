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


#Default, compile with debug flags and print and verify the results
brent: brent-kung.cu
	$(NVCC_DEBUG) -DPRINT_RESULTS -DVERIFY_RESULTS -DARRAY_SIZE=$(ARRAY_SIZE) -DSECTION_SIZE=$(SECTION_SIZE) -o brent-kung brent-kung.cu

#No debug flags, don't print, but do verify the results
brent_test: brent-kung.cu
	$(NVCC) -DVERIFY_RESULTS -DARRAY_SIZE=$(ARRAY_SIZE) -DSECTION_SIZE=$(SECTION_SIZE) -o brent-kung brent-kung.cu

#No debug flags, don't print, don't verify. Just give times
brent_release: brent-kung.cu
	$(NVCC) -DARRAY_SIZE=$(ARRAY_SIZE) -DSECTION_SIZE=$(SECTION_SIZE) -o brent-kung brent-kung.cu


openmp: openmp_inclusiveScan.c
	$(GCC_DEBUG) -DARRAY_SIZE=$(ARRAY_SIZE) -o openmp_inclusiveScan openmp_inclusiveScan.c

openmp_release: openmp_inclusiveScan.c
	$(GCC) -DARRAY_SIZE=$(ARRAY_SIZE) -o openmp_inclusiveScan openmp_inclusiveScan.c


benchmarks: runBenchmarks.py
	./runBenchmarks.py

tests: runTests.py
	./runTests.py

clean:
	rm -f brent-kung openmp_inclusiveScan

.PHONY: clean brent brent_release openmp openmp_release benchmarks tests