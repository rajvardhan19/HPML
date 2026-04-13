#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void vecadd(const float* A, const float* B, float* C, int N)
{
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x  * blockDim.x;
    for (int i = idx; i < N; i += stride)
        C[i] = A[i] + B[i];
}

static float time_scenario(float* d_A, float* d_B, float* d_C, int N,
                            int nblocks, int nthreads)
{
    int N_warmup = (N < 1000000) ? N : 1000000;
    vecadd<<<1, 256>>>(d_A, d_B, d_C, N_warmup);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    vecadd<<<nblocks, nthreads>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

int main()
{
    int K_vals[] = {1, 5, 10, 50, 100};
    int nK = 5;

    printf("K_millions,scenario,time_ms\n");

    for (int ki = 0; ki < nK; ki++) {
        int K = K_vals[ki];
        int N = K * 1000000;
        size_t size = (size_t)N * sizeof(float);

        float* h_A = (float*)malloc(size);
        float* h_B = (float*)malloc(size);
        float* h_C = (float*)malloc(size);
        for (int i = 0; i < N; i++) { h_A[i] = 1.0f; h_B[i] = 2.0f; }

        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, size);
        cudaMalloc(&d_B, size);
        cudaMalloc(&d_C, size);
        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

        float t1 = time_scenario(d_A, d_B, d_C, N, 1, 1);
        printf("%d,1block_1thread,%.3f\n", K, t1);

        float t2 = time_scenario(d_A, d_B, d_C, N, 1, 256);
        printf("%d,1block_256threads,%.3f\n", K, t2);

        int nblocks = (N + 255) / 256;
        float t3 = time_scenario(d_A, d_B, d_C, N, nblocks, 256);
        printf("%d,Nblocks_256threads,%.3f\n", K, t3);

        fflush(stdout);

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        free(h_A); free(h_B); free(h_C);
    }

    cudaDeviceReset();
    return 0;
}
