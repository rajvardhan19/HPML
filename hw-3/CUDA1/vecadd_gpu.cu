///
/// vecadd_gpu.cu
/// For COMS E6998 Spring 2026 — HW3 Part-B, Q2
///
/// GPU vector addition WITHOUT unified memory.
/// Uses explicit cudaMalloc / cudaMemcpy.
/// Usage: ./vecadd_gpu <N>
///

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "timer.h"

__global__ void AddVectorsGPU(const float* A, const float* B, float* C, int N)
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

    // Allocate host memory
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);
    if (!h_A || !h_B || !h_C) { fprintf(stderr, "host malloc failed\n"); return 1; }

    // Initialize
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy host -> device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Grid/block configuration
    int blockSize = 256;
    int gridSize  = (N + blockSize - 1) / blockSize;

    // Warm-up
    AddVectorsGPU<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // Timed run (kernel only)
    initialize_timer();
    start_timer();
    AddVectorsGPU<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    stop_timer();
    double kernel_time = elapsed_time();  // seconds

    double elapsed_ms = kernel_time * 1e3;
    double bw_GBs = (3.0 * size) / kernel_time * 1e-9;
    printf("Kernel time: %.4f ms, Bandwidth: %.3f GB/s\n", elapsed_ms, bw_GBs);

    // Copy result back
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify
    int errors = 0;
    for (int i = 0; i < N; i++)
        if (fabsf(h_C[i] - (float)N) > 1e-5f) errors++;
    printf("Test %s\n", errors == 0 ? "PASSED" : "FAILED");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    cudaDeviceReset();
    return 0;
}
