///
/// matmultKernel01.cu
/// For COMS E6998 Spring 2026
/// Instructor: Kaoutar El Maghraoui
///
/// Multiplies two matrices using CUDA: A x B = C
///
/// Optimization: each thread computes FOUR output values (a 2x2 sub-tile)
/// instead of one. Compiled with -DFOOTPRINT_SIZE=32, so each 16x16 thread
/// block is responsible for a 32x32 output tile.
///
/// Shared memory layout:
///   shared_A[FOOTPRINT_SIZE][BLOCK_SIZE]  (32 x 16)
///   shared_B[BLOCK_SIZE][FOOTPRINT_SIZE]  (16 x 32)
///
/// Each thread (thread_row, thread_col) loads:
///   - rows thread_row and thread_row+BLOCK_SIZE from Asub
///   - cols thread_col and thread_col+BLOCK_SIZE from Bsub
/// and accumulates 4 partial products into Cvalue[2][2].
///

#include "matmultKernel.h"

// FOOTPRINT_SIZE is set to 32 at compile time via -DFOOTPRINT_SIZE=32
// BLOCK_SIZE is 16 (from matmultKernel.h)
// Each 16x16 thread block covers a 32x32 output tile; each thread computes 2x2=4 elements.

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Registers for thread indices
    int thread_row = threadIdx.y;
    int thread_col = threadIdx.x;
    int block_row  = blockIdx.y;
    int block_col  = blockIdx.x;

    // Pointer to the top-left of the 32x32 output sub-matrix this block owns
    float *Csub = &C.elements[C.stride * FOOTPRINT_SIZE * block_row
                              + FOOTPRINT_SIZE * block_col];

    // Four accumulators for the 2x2 output sub-tile per thread
    float Cvalue[2][2] = {{0.f, 0.f}, {0.f, 0.f}};

    // Shared memory tiles
    __shared__ float shared_A[FOOTPRINT_SIZE][BLOCK_SIZE];
    __shared__ float shared_B[BLOCK_SIZE][FOOTPRINT_SIZE];

    // Loop over all BLOCK_SIZE-wide column strips of A (and row strips of B)
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {

        // Pointers to the current sub-matrices of A and B
        float *Asub = &A.elements[A.stride * FOOTPRINT_SIZE * block_row
                                  + BLOCK_SIZE * m];
        float *Bsub = &B.elements[B.stride * BLOCK_SIZE * m
                                  + FOOTPRINT_SIZE * block_col];

        // Collaboratively load Asub (FOOTPRINT_SIZE x BLOCK_SIZE) into shared_A
        // Each thread loads 2 rows
        shared_A[thread_row][thread_col] =
            Asub[thread_row * A.stride + thread_col];
        shared_A[thread_row + BLOCK_SIZE][thread_col] =
            Asub[(thread_row + BLOCK_SIZE) * A.stride + thread_col];

        // Collaboratively load Bsub (BLOCK_SIZE x FOOTPRINT_SIZE) into shared_B
        // Each thread loads 2 columns
        shared_B[thread_row][thread_col] =
            Bsub[thread_row * B.stride + thread_col];
        shared_B[thread_row][thread_col + BLOCK_SIZE] =
            Bsub[thread_row * B.stride + thread_col + BLOCK_SIZE];

        __syncthreads();

        // Accumulate dot products for all 4 output positions
#pragma unroll
        for (int e = 0; e < BLOCK_SIZE; ++e) {
            float a0 = shared_A[2 * thread_row][e];
            float a1 = shared_A[2 * thread_row + 1][e];
            float b0 = shared_B[e][2 * thread_col];
            float b1 = shared_B[e][2 * thread_col + 1];

            Cvalue[0][0] += a0 * b0;
            Cvalue[0][1] += a0 * b1;
            Cvalue[1][0] += a1 * b0;
            Cvalue[1][1] += a1 * b1;
        }

        __syncthreads();
    }

    // Write the 4 results back to global memory
    Csub[(2 * thread_row)     * C.stride + (2 * thread_col)]     = Cvalue[0][0];
    Csub[(2 * thread_row)     * C.stride + (2 * thread_col + 1)] = Cvalue[0][1];
    Csub[(2 * thread_row + 1) * C.stride + (2 * thread_col)]     = Cvalue[1][0];
    Csub[(2 * thread_row + 1) * C.stride + (2 * thread_col + 1)] = Cvalue[1][1];
}
