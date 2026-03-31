///
/// vecadd_unified.cu
/// For COMS E6998 Spring 2026 — HW3 Part-B, Q3
///
/// GPU vector addition WITH CUDA Unified Memory (cudaMallocManaged).
/// No explicit cudaMemcpy needed — the runtime migrates pages on demand.
/// Usage: ./vecadd_unified <N>
///

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "timer.h"

__global__ void AddVectorsUnified(const float* A, const float* B, float* C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        C[idx] = A[idx] + B[idx];
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        printf("Usage: %s N\n", argv[0]);
        printf("  N = number of float elements per vector\n");
        return 1;
    }

    int N = atoi(argv[1]);
    printf("Vector size: %d\n", N);
    size_t size = N * sizeof(float);

    // Allocate unified memory (accessible from both CPU and GPU)
    float *A, *B, *C;
    cudaMallocManaged((void**)&A, size);
    cudaMallocManaged((void**)&B, size);
    cudaMallocManaged((void**)&C, size);

    // Initialize on the CPU — pages start on host, migrate to GPU on first kernel access
    for (int i = 0; i < N; i++) {
        A[i] = (float)i;
        B[i] = (float)(N - i);
    }

    // Grid/block configuration
    int blockSize = 256;
    int gridSize  = (N + blockSize - 1) / blockSize;

    // Warm-up
    AddVectorsUnified<<<gridSize, blockSize>>>(A, B, C, N);
    cudaDeviceSynchronize();

    // Timed run (kernel only)
    initialize_timer();
    start_timer();
    AddVectorsUnified<<<gridSize, blockSize>>>(A, B, C, N);
    cudaDeviceSynchronize();
    stop_timer();
    double kernel_time = elapsed_time();  // seconds

    double elapsed_ms = kernel_time * 1e3;
    double bw_GBs = (3.0 * size) / kernel_time * 1e-9;
    printf("Kernel time: %.4f ms, Bandwidth: %.3f GB/s\n", elapsed_ms, bw_GBs);

    // Result is already accessible on the CPU via unified memory — no memcpy needed
    // Verify
    int errors = 0;
    for (int i = 0; i < N; i++)
        if (fabsf(C[i] - (float)N) > 1e-5f) errors++;
    printf("Test %s\n", errors == 0 ? "PASSED" : "FAILED");

    cudaFree(A); cudaFree(B); cudaFree(C);
    cudaDeviceReset();
    return 0;
}
