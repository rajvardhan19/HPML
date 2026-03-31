///
/// conv_simple.cu
/// For COMS E6998 Spring 2026 — HW3 Part-C, C1
///
/// Simple (naive) CUDA 2D convolution — no tiling, no shared memory.
/// One thread per output element (c_out, h, w).
///
/// Input  : (1, C_IN,  H,   W)   = (1,  3, 1024, 1024)
/// Filter : (C_OUT, C_IN, FH, FW) = (64, 3,    3,    3)
/// Output : (1, C_OUT, H, W)  [same spatial size, padding=1, stride=1]
///
/// Build:
///   nvcc conv_simple.cu timer.o -O3 -o conv_simple
///

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "timer.h"

// Problem dimensions
#define C_IN  3
#define C_OUT 64
#define H     1024
#define W     1024
#define FH    3
#define FW    3
#define PAD   1

// ---------------------------------------------------------------
// Naive convolution kernel
// Output tensor layout: [C_OUT][H][W]  (row-major, no batch dim)
// Input  tensor layout: [C_IN ][H][W]
// Filter tensor layout: [C_OUT][C_IN][FH][FW]
// ---------------------------------------------------------------
__global__ void conv2d_simple(
    const float* __restrict__ input,    // [C_IN][H][W]
    const float* __restrict__ filter,   // [C_OUT][C_IN][FH][FW]
          float* __restrict__ output,   // [C_OUT][H][W]
    int c_in, int c_out, int h, int w, int fh, int fw, int pad)
{
    // Thread → one output element
    int ow = blockIdx.x * blockDim.x + threadIdx.x;  // output col
    int oh = blockIdx.y * blockDim.y + threadIdx.y;  // output row
    int oc = blockIdx.z;                               // output channel

    if (ow >= w || oh >= h || oc >= c_out) return;

    float val = 0.0f;

    for (int ic = 0; ic < c_in; ic++) {
        for (int frow = 0; frow < fh; frow++) {
            for (int fcol = 0; fcol < fw; fcol++) {
                int ih = oh + frow - pad;
                int iw = ow + fcol - pad;
                if (ih >= 0 && ih < h && iw >= 0 && iw < w) {
                    float in_val  = input [ic * h * w + ih * w + iw];
                    float flt_val = filter[oc * c_in * fh * fw
                                          + ic * fh * fw
                                          + frow * fw + fcol];
                    val += in_val * flt_val;
                }
            }
        }
    }

    output[oc * h * w + oh * w + ow] = val;
}

// CPU reference for correctness checking
void conv2d_cpu(
    const float* input, const float* filter, float* output,
    int c_in, int c_out, int h, int w, int fh, int fw, int pad)
{
    for (int oc = 0; oc < c_out; oc++)
    for (int oh = 0; oh < h;     oh++)
    for (int ow = 0; ow < w;     ow++) {
        float val = 0.f;
        for (int ic  = 0; ic  < c_in; ic++ )
        for (int fr  = 0; fr  < fh;   fr++ )
        for (int fc  = 0; fc  < fw;   fc++ ) {
            int ih = oh + fr - pad;
            int iw = ow + fc - pad;
            if (ih >= 0 && ih < h && iw >= 0 && iw < w)
                val += input [ic*h*w + ih*w + iw]
                     * filter[oc*c_in*fh*fw + ic*fh*fw + fr*fw + fc];
        }
        output[oc*h*w + oh*w + ow] = val;
    }
}

int main()
{
    size_t in_bytes  = (size_t)C_IN  * H * W * sizeof(float);
    size_t flt_bytes = (size_t)C_OUT * C_IN * FH * FW * sizeof(float);
    size_t out_bytes = (size_t)C_OUT * H * W * sizeof(float);

    // Host buffers
    float* h_in  = (float*)malloc(in_bytes);
    float* h_flt = (float*)malloc(flt_bytes);
    float* h_out = (float*)calloc(C_OUT * H * W, sizeof(float));
    float* h_ref = (float*)calloc(C_OUT * H * W, sizeof(float));

    // Initialize with simple values
    for (size_t i = 0; i < C_IN  * H * W;          i++) h_in [i] = (float)(i % 7) * 0.1f;
    for (size_t i = 0; i < C_OUT * C_IN * FH * FW; i++) h_flt[i] = (float)(i % 5) * 0.1f;

    // CPU reference (only for small subset to avoid long wait — check first 4 channels)
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

    // Device buffers
    float *d_in, *d_flt, *d_out;
    cudaMalloc((void**)&d_in,  in_bytes);
    cudaMalloc((void**)&d_flt, flt_bytes);
    cudaMalloc((void**)&d_out, out_bytes);

    cudaMemcpy(d_in,  h_in,  in_bytes,  cudaMemcpyHostToDevice);
    cudaMemcpy(d_flt, h_flt, flt_bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, out_bytes);

    // Grid: (W/16, H/16, C_OUT)
    dim3 block(16, 16, 1);
    dim3 grid((W + 15) / 16, (H + 15) / 16, C_OUT);

    // Warm-up
    conv2d_simple<<<grid, block>>>(d_in, d_flt, d_out, C_IN, C_OUT, H, W, FH, FW, PAD);
    cudaDeviceSynchronize();

    // Timed run
    initialize_timer();
    start_timer();
    conv2d_simple<<<grid, block>>>(d_in, d_flt, d_out, C_IN, C_OUT, H, W, FH, FW, PAD);
    cudaDeviceSynchronize();
    stop_timer();
    double t = elapsed_time();

    // FLOPs: 2 * C_OUT * H * W * C_IN * FH * FW  (multiply-add)
    double flops    = 2.0 * C_OUT * H * W * C_IN * FH * FW;
    double gflops_s = flops / t * 1e-9;
    printf("conv_simple: time=%.4f ms, GFLOPs/s=%.2f\n", t * 1e3, gflops_s);

    // Copy result back and verify
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
