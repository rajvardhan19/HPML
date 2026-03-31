///
/// conv_cudnn.cu
/// For COMS E6998 Spring 2026 — HW3 Part-C, C3
///
/// cuDNN-based 2D convolution (highly optimized).
/// Uses cudnnFindConvolutionForwardAlgorithm to auto-select the best algorithm.
///
/// Input  : (1, C_IN,  H,   W)    = (1,  3, 1024, 1024)
/// Filter : (C_OUT, C_IN, FH, FW) = (64, 3,    3,    3)
/// Output : (1, C_OUT, H, W)   padding=1, stride=1
///
/// Build:
///   nvcc conv_cudnn.cu timer.o -O3 -lcudnn -o conv_cudnn
///

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cudnn.h>
#include "timer.h"

#define C_IN   3
#define C_OUT  64
#define H      1024
#define W      1024
#define FH     3
#define FW     3
#define PAD    1
#define STRIDE 1

// Convenience macro for cuDNN error checking
#define CUDNN_CHECK(call)                                              \
    do {                                                               \
        cudnnStatus_t status = (call);                                 \
        if (status != CUDNN_STATUS_SUCCESS) {                          \
            fprintf(stderr, "cuDNN error at %s:%d — %s\n",            \
                    __FILE__, __LINE__, cudnnGetErrorString(status));   \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

int main()
{
    // ----------------------------------------------------------------
    // Allocate and initialize host buffers
    // ----------------------------------------------------------------
    size_t in_bytes  = (size_t)C_IN  * H * W * sizeof(float);
    size_t flt_bytes = (size_t)C_OUT * C_IN * FH * FW * sizeof(float);

    float* h_in  = (float*)malloc(in_bytes);
    float* h_flt = (float*)malloc(flt_bytes);
    for (size_t i = 0; i < C_IN  * H * W;          i++) h_in [i] = (float)(i % 7) * 0.1f;
    for (size_t i = 0; i < C_OUT * C_IN * FH * FW; i++) h_flt[i] = (float)(i % 5) * 0.1f;

    // ----------------------------------------------------------------
    // Create cuDNN handle
    // ----------------------------------------------------------------
    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));

    // ----------------------------------------------------------------
    // Tensor / filter / convolution descriptors
    // ----------------------------------------------------------------
    cudnnTensorDescriptor_t    input_desc, output_desc;
    cudnnFilterDescriptor_t    filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CUDNN_CHECK(cudnnCreateTensorDescriptor   (&input_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor   (&output_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor   (&filter_desc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

    // Input: NCHW
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, C_IN, H, W));

    // Filter: (C_OUT, C_IN, FH, FW), NCHW layout
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(
        filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, C_OUT, C_IN, FH, FW));

    // Convolution: padding=1, stride=1, dilation=1
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        conv_desc, PAD, PAD, STRIDE, STRIDE, 1, 1,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    // Compute output dimensions
    int out_n, out_c, out_h, out_w;
    CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(
        conv_desc, input_desc, filter_desc,
        &out_n, &out_c, &out_h, &out_w));

    printf("Output dimensions: %d x %d x %d x %d\n", out_n, out_c, out_h, out_w);

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n, out_c, out_h, out_w));

    size_t out_bytes = (size_t)out_n * out_c * out_h * out_w * sizeof(float);

    // ----------------------------------------------------------------
    // Allocate device memory
    // ----------------------------------------------------------------
    float *d_in, *d_flt, *d_out;
    cudaMalloc((void**)&d_in,  in_bytes);
    cudaMalloc((void**)&d_flt, flt_bytes);
    cudaMalloc((void**)&d_out, out_bytes);

    cudaMemcpy(d_in,  h_in,  in_bytes,  cudaMemcpyHostToDevice);
    cudaMemcpy(d_flt, h_flt, flt_bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, out_bytes);

    // ----------------------------------------------------------------
    // Find the best forward algorithm
    // ----------------------------------------------------------------
    int           requested = 8, returned = 0;
    cudnnConvolutionFwdAlgoPerf_t perf_results[8];
    CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithm(
        handle, input_desc, filter_desc, conv_desc, output_desc,
        requested, &returned, perf_results));

    cudnnConvolutionFwdAlgo_t algo = perf_results[0].algo;
    printf("Best algorithm: %d  (time=%.4f ms)\n", (int)algo, perf_results[0].time);

    // ----------------------------------------------------------------
    // Allocate workspace
    // ----------------------------------------------------------------
    size_t workspace_bytes = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        handle, input_desc, filter_desc, conv_desc, output_desc,
        algo, &workspace_bytes));
    printf("Workspace: %.2f MB\n", workspace_bytes / 1048576.0);

    void* d_workspace = nullptr;
    if (workspace_bytes > 0)
        cudaMalloc(&d_workspace, workspace_bytes);

    // ----------------------------------------------------------------
    // Warm-up
    // ----------------------------------------------------------------
    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionForward(
        handle, &alpha,
        input_desc, d_in,
        filter_desc, d_flt,
        conv_desc, algo,
        d_workspace, workspace_bytes,
        &beta, output_desc, d_out));
    cudaDeviceSynchronize();

    // ----------------------------------------------------------------
    // Timed run
    // ----------------------------------------------------------------
    initialize_timer();
    start_timer();
    CUDNN_CHECK(cudnnConvolutionForward(
        handle, &alpha,
        input_desc, d_in,
        filter_desc, d_flt,
        conv_desc, algo,
        d_workspace, workspace_bytes,
        &beta, output_desc, d_out));
    cudaDeviceSynchronize();
    stop_timer();
    double t = elapsed_time();

    double flops    = 2.0 * C_OUT * out_h * out_w * C_IN * FH * FW;
    double gflops_s = flops / t * 1e-9;
    printf("conv_cudnn:  time=%.4f ms, GFLOPs/s=%.2f\n", t * 1e3, gflops_s);

    // ----------------------------------------------------------------
    // Verify against naive conv_simple output (first 4 channels, CPU ref)
    // ----------------------------------------------------------------
    float* h_out = (float*)malloc(out_bytes);
    cudaMemcpy(h_out, d_out, out_bytes, cudaMemcpyDeviceToHost);

    printf("Computing CPU reference for verification...\n");
    float* h_ref = (float*)calloc(out_c * out_h * out_w, sizeof(float));
    for (int oc = 0; oc < 4; oc++)
    for (int oh = 0; oh < out_h; oh++)
    for (int ow = 0; ow < out_w; ow++) {
        float val = 0.f;
        for (int ic = 0; ic < C_IN; ic++)
        for (int fr = 0; fr < FH;   fr++)
        for (int fc = 0; fc < FW;   fc++) {
            int ih = oh + fr - PAD, iw = ow + fc - PAD;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                val += h_in [ic*H*W + ih*W + iw]
                     * h_flt[oc*C_IN*FH*FW + ic*FH*FW + fr*FW + fc];
        }
        h_ref[oc*out_h*out_w + oh*out_w + ow] = val;
    }

    double max_err = 0.0; int err_cnt = 0;
    for (int oc = 0; oc < 4; oc++)
    for (int i  = 0; i  < out_h * out_w; i++) {
        double diff = fabs(h_out[oc*out_h*out_w + i] - h_ref[oc*out_h*out_w + i]);
        if (diff > 1e-3) { err_cnt++; if (diff > max_err) max_err = diff; }
    }
    if (err_cnt == 0)
        printf("Correctness check PASSED (first 4 channels)\n");
    else
        printf("Correctness check FAILED: %d errors, max_err=%.6f\n", err_cnt, max_err);

    // ----------------------------------------------------------------
    // Cleanup
    // ----------------------------------------------------------------
    if (d_workspace) cudaFree(d_workspace);
    cudaFree(d_in); cudaFree(d_flt); cudaFree(d_out);
    free(h_in); free(h_flt); free(h_out); free(h_ref);

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
    CUDNN_CHECK(cudnnDestroy(handle));
    cudaDeviceReset();
    return 0;
}
