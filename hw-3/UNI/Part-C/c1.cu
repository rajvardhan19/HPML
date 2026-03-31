///
/// c1.cu
/// For COMS E6998 Spring 2026 — HW3 Part-C, C1
///
/// Simple (naive) CUDA convolution — no tiling, no shared memory.
/// One thread per output element (k, h, w).
///
/// All tensors use double precision.
///
/// Input I[c][h][w]     = c * (w + h),   c in [0,C-1], h in [0,H-1], w in [0,W-1]
/// Filter F[k][c][i][j] = (c+k)*(i+j),   stored as [K][C][FH][FW]
/// I0 = I padded with P=1 zeros on each side → dims [C][H+2P][W+2P]
///
/// True convolution (not cross-correlation):
///   O[k][h][w] = sum_c sum_fh sum_fw  F[k][c][FH-1-fh][FW-1-fw] * I0[c][h+fh][w+fw]
///
/// Prints: checksum (sum of all O elements) and kernel execution time in ms.
/// Last stdout line: CSV format  "checksum,time_ms"
///

#include <stdio.h>
#include <stdlib.h>
#include "timer.h"

#define C_DIM   3
#define K_DIM   64
#define H_DIM   1024
#define W_DIM   1024
#define FH_DIM  3
#define FW_DIM  3
#define P_DIM   1

__global__ void conv_c1(
    const double* __restrict__ I0,   // [C][H0][W0]
    const double* __restrict__ F,    // [K][C][FH][FW]
          double* __restrict__ O,    // [K][H][W]
    int C, int K, int H, int W, int FH, int FW)
{
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z;

    if (w >= W || h >= H || k >= K) return;

    const int H0 = H + 2;   // P=1
    const int W0 = W + 2;

    double val = 0.0;
    for (int c = 0; c < C; c++) {
        for (int fh = 0; fh < FH; fh++) {
            for (int fw = 0; fw < FW; fw++) {
                // True convolution: flip both filter dimensions
                double f  = F [k*C*FH*FW + c*FH*FW + (FH-1-fh)*FW + (FW-1-fw)];
                double i0 = I0[c*H0*W0 + (h+fh)*W0 + (w+fw)];
                val += f * i0;
            }
        }
    }
    O[k*H*W + h*W + w] = val;
}

int main()
{
    const int C = C_DIM, K = K_DIM, H = H_DIM, W = W_DIM;
    const int FH = FH_DIM, FW = FW_DIM, P = P_DIM;
    const int H0 = H + 2*P, W0 = W + 2*P;

    size_t sz_I0 = (size_t)C * H0 * W0 * sizeof(double);
    size_t sz_F  = (size_t)K * C * FH * FW * sizeof(double);
    size_t sz_O  = (size_t)K * H * W * sizeof(double);

    double* h_I0 = (double*)calloc((size_t)C * H0 * W0, sizeof(double));
    double* h_F  = (double*)malloc(sz_F);
    double* h_O  = (double*)malloc(sz_O);

    // I[c][h][w] = c*(w+h) — store directly into padded I0
    for (int c = 0; c < C; c++)
        for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++)
                h_I0[c*H0*W0 + (h+P)*W0 + (w+P)] = (double)c * (w + h);

    // F[k][c][fh][fw] = (c+k)*(fh+fw)
    for (int k = 0; k < K; k++)
        for (int c = 0; c < C; c++)
            for (int fh = 0; fh < FH; fh++)
                for (int fw = 0; fw < FW; fw++)
                    h_F[k*C*FH*FW + c*FH*FW + fh*FW + fw] = (double)(c+k) * (fh+fw);

    double *d_I0, *d_F, *d_O;
    cudaMalloc(&d_I0, sz_I0);
    cudaMalloc(&d_F,  sz_F);
    cudaMalloc(&d_O,  sz_O);
    cudaMemcpy(d_I0, h_I0, sz_I0, cudaMemcpyHostToDevice);
    cudaMemcpy(d_F,  h_F,  sz_F,  cudaMemcpyHostToDevice);

    dim3 block(16, 16, 1);
    dim3 grid((W+15)/16, (H+15)/16, K);

    // Warm-up
    conv_c1<<<grid, block>>>(d_I0, d_F, d_O, C, K, H, W, FH, FW);
    cudaDeviceSynchronize();

    // Timed run
    initialize_timer(); start_timer();
    conv_c1<<<grid, block>>>(d_I0, d_F, d_O, C, K, H, W, FH, FW);
    cudaDeviceSynchronize();
    stop_timer();
    double t_ms = elapsed_time() * 1e3;

    cudaMemcpy(h_O, d_O, sz_O, cudaMemcpyDeviceToHost);

    // Checksum: sum of all output elements
    double checksum = 0.0;
    for (size_t i = 0; i < (size_t)K*H*W; i++)
        checksum += h_O[i];

    fprintf(stderr, "C1: checksum=%.6f  time=%.3f ms\n", checksum, t_ms);
    printf("%.6f,%.3f\n", checksum, t_ms);   // CSV line for program_output.csv

    cudaFree(d_I0); cudaFree(d_F); cudaFree(d_O);
    free(h_I0); free(h_F); free(h_O);
    cudaDeviceReset();
    return 0;
}
