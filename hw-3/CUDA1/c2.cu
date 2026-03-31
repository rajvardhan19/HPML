///
/// c2.cu
/// For COMS E6998 Spring 2026 — HW3 Part-C, C2
///
/// Tiled CUDA convolution using shared memory.
///
/// Each 16x16 thread block covers a TILE×TILE output patch for one output channel k.
/// For each input channel c, the (TILE+FH-1)×(TILE+FW-1) input halo is cooperatively
/// loaded into shared memory; the convolution is then computed from smem.
///
/// Same double-precision initialization and true-convolution formula as C1.
/// Checksum should match C1 exactly (to floating-point precision).
///
/// Last stdout line: CSV "checksum,time_ms"
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
#define TILE    16

// Shared-memory halo dimensions (compile-time constants required by CUDA)
#define HALO_H  (TILE + FH_DIM - 1)   // 18
#define HALO_W  (TILE + FW_DIM - 1)   // 18

__global__ void conv_c2(
    const double* __restrict__ I0,   // [C][H0][W0]  (already padded)
    const double* __restrict__ F,    // [K][C][FH][FW]
          double* __restrict__ O,    // [K][H][W]
    int C, int K, int H, int W, int FH, int FW)
{
    __shared__ double smem[HALO_H][HALO_W];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int ow = blockIdx.x * TILE + tx;   // output column
    int oh = blockIdx.y * TILE + ty;   // output row
    int k  = blockIdx.z;               // output channel

    const int H0 = H + 2;
    const int W0 = W + 2;

    double val = 0.0;

    for (int c = 0; c < C; c++) {
        // Top-left corner of the I0 patch this block needs
        int base_h = blockIdx.y * TILE;
        int base_w = blockIdx.x * TILE;

        // Cooperative load of the (HALO_H × HALO_W) I0 patch into smem
        for (int r = ty; r < HALO_H; r += TILE)
            for (int s = tx; s < HALO_W; s += TILE)
                smem[r][s] = I0[c * H0*W0 + (base_h + r)*W0 + (base_w + s)];

        __syncthreads();

        // Compute convolution from smem (true conv: flip filter)
        if (ow < W && oh < H) {
            for (int fh = 0; fh < FH; fh++)
                for (int fw = 0; fw < FW; fw++) {
                    double f = F[k*C*FH*FW + c*FH*FW + (FH-1-fh)*FW + (FW-1-fw)];
                    val += f * smem[ty + fh][tx + fw];
                }
        }

        __syncthreads();
    }

    if (ow < W && oh < H)
        O[k*H*W + oh*W + ow] = val;
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

    // Same initialization as C1
    for (int c = 0; c < C; c++)
        for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++)
                h_I0[c*H0*W0 + (h+P)*W0 + (w+P)] = (double)c * (w + h);

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

    dim3 block(TILE, TILE, 1);
    dim3 grid((W + TILE-1)/TILE, (H + TILE-1)/TILE, K);

    // Warm-up
    conv_c2<<<grid, block>>>(d_I0, d_F, d_O, C, K, H, W, FH, FW);
    cudaDeviceSynchronize();

    // Timed run
    initialize_timer(); start_timer();
    conv_c2<<<grid, block>>>(d_I0, d_F, d_O, C, K, H, W, FH, FW);
    cudaDeviceSynchronize();
    stop_timer();
    double t_ms = elapsed_time() * 1e3;

    cudaMemcpy(h_O, d_O, sz_O, cudaMemcpyDeviceToHost);

    double checksum = 0.0;
    for (size_t i = 0; i < (size_t)K*H*W; i++)
        checksum += h_O[i];

    fprintf(stderr, "C2: checksum=%.6f  time=%.3f ms\n", checksum, t_ms);
    printf("%.6f,%.3f\n", checksum, t_ms);

    cudaFree(d_I0); cudaFree(d_F); cudaFree(d_O);
    free(h_I0); free(h_F); free(h_O);
    cudaDeviceReset();
    return 0;
}
