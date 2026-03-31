###
### For COMS E6998 Spring 2026
### Instructor: Kaoutar El Maghraoui
### Makefile for HW3 (Part-A, Part-B, Part-C)
###

SDK_INSTALL_PATH := /usr/local/cuda
NVCC             := $(SDK_INSTALL_PATH)/bin/nvcc
LIB              := -L$(SDK_INSTALL_PATH)/lib64
OPTIONS          := -O3

TAR_FILE_NAME := YourNameCUDA1.tar

# All executables
EXECS := vecadd00 matmult00 vecadd01 matmult01 \
         q1 q2 q3 \
         c1 c2 c3

all: $(EXECS)

#######################################################################
clean:
	rm -f $(EXECS) *.o q1_results.csv q2_results.csv q3_results.csv

tar:
	tar -cvf $(TAR_FILE_NAME) Makefile *.h *.cu *.cpp *.py *.pdf

#######################################################################
# Shared timer
timer.o : timer.cu timer.h
	${NVCC} $< -c -o $@ $(OPTIONS)

#######################################################################
# Part-A — vecadd00: baseline non-coalescing kernel
vecaddKernel00.o : vecaddKernel00.cu
	${NVCC} $< -c -o $@ $(OPTIONS)

vecadd00 : vecadd.cu vecaddKernel.h vecaddKernel00.o timer.o
	${NVCC} $< vecaddKernel00.o -o $@ $(LIB) timer.o $(OPTIONS)

# Part-A — vecadd01: coalesced kernel (vecaddKernel01.cu)
vecaddKernel01.o : vecaddKernel01.cu
	${NVCC} $< -c -o $@ $(OPTIONS)

vecadd01 : vecadd.cu vecaddKernel.h vecaddKernel01.o timer.o
	${NVCC} $< vecaddKernel01.o -o $@ $(LIB) timer.o $(OPTIONS)

#######################################################################
# Part-A — matmult00: baseline kernel (1 value per thread, FOOTPRINT=16)
matmultKernel00.o : matmultKernel00.cu matmultKernel.h
	${NVCC} $< -c -o $@ $(OPTIONS)

matmult00 : matmult.cu matmultKernel.h matmultKernel00.o timer.o
	${NVCC} $< matmultKernel00.o -o $@ $(LIB) timer.o $(OPTIONS)

# Part-A — matmult01: optimised kernel (4 values per thread, FOOTPRINT=32)
matmultKernel01.o : matmultKernel01.cu matmultKernel.h
	${NVCC} $< -c -o $@ $(OPTIONS) -DFOOTPRINT_SIZE=32

matmult01 : matmult.cu matmultKernel.h matmultKernel01.o timer.o
	${NVCC} $< matmultKernel01.o -o $@ $(LIB) timer.o $(OPTIONS) -DFOOTPRINT_SIZE=32

#######################################################################
# Part-B — q1: CPU-only vector addition (K million floats, K=1,5,10,50,100)
q1 : q1.cpp
	g++ $< -O3 -o $@

# Part-B — q2: GPU vector addition without unified memory, 3 scenarios
q2 : q2.cu
	${NVCC} $< -o $@ $(LIB) $(OPTIONS)

# Part-B — q3: GPU vector addition with CUDA unified memory, 3 scenarios
q3 : q3.cu
	${NVCC} $< -o $@ $(LIB) $(OPTIONS)

#######################################################################
# Part-C — c1: simple CUDA convolution (double, no shared memory)
c1 : c1.cu timer.o
	${NVCC} $< timer.o -o $@ $(LIB) $(OPTIONS)

# Part-C — c2: tiled CUDA convolution (double, shared memory)
c2 : c2.cu timer.o
	${NVCC} $< timer.o -o $@ $(LIB) $(OPTIONS)

# Part-C — c3: cuDNN convolution (double, CUDNN_CONVOLUTION, fastest algo)
c3 : c3.cu timer.o
	${NVCC} $< timer.o -o $@ $(LIB) -lcudnn $(OPTIONS)
