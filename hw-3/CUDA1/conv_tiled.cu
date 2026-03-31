///
/// conv_tiled.cu
/// For COMS E6998 Spring 2026 — HW3 Part-C, C2
///
/// Tiled 2D convolution using shared memory.
/// Each thread block loads a (TILE+FH-1) x (TILE+FW-1) halo input patch
/// into shared memory for one input channel at a time, then accumulates
/// across all input channels.
///
/// One thread block → TILE x TILE output elements for one output channel.
///
/// Input  : (1, C_IN,  H,   W)    = (1,  3, 1024, 1024)
/// Filter : (C_OUT, C_IN, FH, FW) = (64, 3,    3,    3)
/// Output : (1, C_OUT, H, W)   padding=1, stride=1
///

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "timer.h"

#define C_IN  3
#define C_OUT 64
#define H     1024
#define W     1024
#define FH    3
#define FW    3
#define PAD   1
#define TILE  16   // output tile size per block dimension

// ---------------------------------------------------------------
// Tiled convolution kernel
// ---------------------------------------------------------------
__global__ void conv2d_tiled(
    const float* __restrict__ input,    // [C_IN][H][W]
    const float* __restrict__ filter,   // [C_OUT][C_IN][FH][FW]
          float* __restrict__ output,   // [C_OUT][H][W]
    int c_in, int c_out, int h, int w, int fh, int fw, int pad)
{
    // Shared memory for one input channel tile (with halo)
    // Dimensions: (TILE + FH - 1) x (TILE + FW - 1)
    __shared__ float smem[TILE + FH - 1][TILE + FW - 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int ow = blockIdx.x * TILE + tx;   // output col
    int oh = blockIdx.y * TILE + ty;   // output row
    int oc = blockIdx.z;               // output channel

    float val = 0.0f;

    // Loop over input channels
    for (int ic = 0; ic < c_in; ic++) {

        // Collaboratively load the (TILE+FH-1) x (TILE+FW-1) input tile
        // Each thread loads one or more elements using a loop
        int halo_h = TILE + fh - 1;
        int halo_w = TILE + fw - 1;

        // Base input coords for this tile (top-left corner)
        int base_ih = blockIdx.y * TILE - pad;
        int base_iw = blockIdx.x * TILE - pad;

        // Use the thread to fill the shared memory (some threads do extra loads)
        for (int r = ty; r < halo_h; r += TILE) {
            for (int c = tx; c < halo_w; c += TILE) {
                int ih = base_ih + r;
                int iw = base_iw + c;
                if (ih >= 0 && ih < h && iw >= 0 && iw < w)
                    smem[r][c] = input[ic * h * w + ih * w + iw];
                else
                    smem[r][c] = 0.0f;
            }
        }

        __syncthreads();

        // Compute convolution for this input channel (if thread is within output bounds)
        if (ow < w && oh < h && oc < c_out) {
            for (int fr = 0; fr < fh; fr++) {
                for (int fc = 0; fc < fw; fc++) {
                    val += smem[ty + fr][tx + fc]
                         * filter[oc * c_in * fh * fw
                                  + ic * fh * fw
                                  + fr * fw + fc];
                }
            }
        }

        __syncthreads();
    }

    if (ow < w && oh < h && oc < c_out)
        output[oc * h * w + oh * w + ow] = val;
}

int main()
{
    size_t in_bytes  = (size_t)C_IN  * H * W * sizeof(float);
    size_t flt_bytes = (size_t)C_OUT * C_IN * FH * FW * sizeof(float);
    size_t out_bytes = (size_t)C_OUT * H * W * sizeof(float);

    float* h_in  = (float*)malloc(in_bytes);
    float* h_flt = (float*)malloc(flt_bytes);
    float* h_out = (float*)calloc(C_OUT * H * W, sizeof(float));
    float* h_ref = (float*)calloc(C_OUT * H * W, sizeof(float));

    for (size_t i = 0; i < C_IN  * H * W;          i++) h_in [i] = (float)(i % 7) * 0.1f;
    for (size_t i = 0; i < C_OUT * C_IN * FH * FW; i++) h_flt[i] = (float)(i % 5) * 0.1f;

    // CPU reference for first 4 channels
    printf("Computing CPU reference for first 4 output channels...\n");
    for (int oc = 0; oc < 4; oc++)
    for (int oh = 0; oh < H; oh++)
    for (int ow = 0; ow < W; ow++) {
        float val = 0.f;
        for (int ic = 0; ic < C_IN; ic++)
        for (int fr = 0; fr < FH;   fr++)
        for (int fc = 0; fc < FW;   fc++) {
            int ih = oh + fr - PAD, iw = ow + fc - PAD;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                val += h_in [ic*H*W + ih*W + iw]
                     * h_flt[oc*C_IN*FH*FW + ic*FH*FW + fr*FW + fc];
        }
        h_ref[oc*H*W + oh*W + ow] = val;
    }

    float *d_in, *d_flt, *d_out;
    cudaMalloc((void**)&d_in,  in_bytes);
    cudaMalloc((void**)&d_flt, flt_bytes);
    cudaMalloc((void**)&d_out, out_bytes);

    cudaMemcpy(d_in,  h_in,  in_bytes,  cudaMemcpyHostToDevice);
    cudaMemcpy(d_flt, h_flt, flt_bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, out_bytes);

    dim3 block(TILE, TILE, 1);
    dim3 grid((W + TILE - 1) / TILE, (H + TILE - 1) / TILE, C_OUT);

    // Warm-up
    conv2d_tiled<<<grid, block>>>(d_in, d_flt, d_out, C_IN, C_OUT, H, W, FH, FW, PAD);
    cudaDeviceSynchronize();

    // Timed run
    initialize_timer();
    start_timer();
    conv2d_tiled<<<grid, block>>>(d_in, d_flt, d_out, C_IN, C_OUT, H, W, FH, FW, PAD);
    cudaDeviceSynchronize();
    stop_timer();
    double t = elapsed_time();

    double flops    = 2.0 * C_OUT * H * W * C_IN * FH * FW;
    double gflops_s = flops / t * 1e-9;
    printf("conv_tiled: time=%.4f ms, GFLOPs/s=%.2f\n", t * 1e3, gflops_s);

    cudaMemcpy(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost);

    double max_err = 0.0;
    int    err_cnt = 0;
    for (int oc = 0; oc < 4; oc++)
    for (int i  = 0; i  < H * W; i++) {
        double diff = fabs(h_out[oc*H*W + i] - h_ref[oc*H*W + i]);
        if (diff > 1e-3) { err_cnt++; if (diff > max_err) max_err = diff; }
    }
    if (err_cnt == 0)
        printf("Correctness check PASSED (first 4 channels)\n");
    else
        printf("Correctness check FAILED: %d errors, max_err=%.6f\n", err_cnt, max_err);

    cudaFree(d_in); cudaFree(d_flt); cudaFree(d_out);
    free(h_in); free(h_flt); free(h_out); free(h_ref);
    cudaDeviceReset();
    return 0;
}
