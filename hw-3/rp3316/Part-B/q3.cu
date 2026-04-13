#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void vecadd_unified(const float* A, const float* B, float* C, int N)
{
    int idx    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x  * blockDim.x;
    for (int i = idx; i < N; i += stride)
        C[i] = A[i] + B[i];
}

static float time_scenario(float* A, float* B, float* C, int N,
                            int nblocks, int nthreads)
{
    vecadd_unified<<<nblocks, nthreads>>>(A, B, C, N);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    vecadd_unified<<<nblocks, nthreads>>>(A, B, C, N);
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

        float *A, *B, *C;
        cudaMallocManaged(&A, size);
        cudaMallocManaged(&B, size);
        cudaMallocManaged(&C, size);

        for (int i = 0; i < N; i++) { A[i] = 1.0f; B[i] = 2.0f; }

        float t1 = time_scenario(A, B, C, N, 1, 1);
        printf("%d,1block_1thread,%.3f\n", K, t1);

        float t2 = time_scenario(A, B, C, N, 1, 256);
        printf("%d,1block_256threads,%.3f\n", K, t2);

        int nblocks = (N + 255) / 256;
        float t3 = time_scenario(A, B, C, N, nblocks, 256);
        printf("%d,Nblocks_256threads,%.3f\n", K, t3);

        fflush(stdout);

        cudaFree(A); cudaFree(B); cudaFree(C);
    }

    cudaDeviceReset();
    return 0;
}
