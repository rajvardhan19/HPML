#include <stdio.h>
#include <stdlib.h>
#include <cudnn.h>
#include "timer.h"

#define C_DIM   3
#define K_DIM   64
#define H_DIM   1024
#define W_DIM   1024
#define FH_DIM  3
#define FW_DIM  3
#define P_DIM   1

#define CUDNN_CHECK(call)                                                    \
    do {                                                                     \
        cudnnStatus_t _s = (call);                                           \
        if (_s != CUDNN_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "cuDNN error %s:%d  %s\n",                      \
                    __FILE__, __LINE__, cudnnGetErrorString(_s));            \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

int main()
{
    const int C = C_DIM, K = K_DIM, H = H_DIM, W = W_DIM;
    const int FH = FH_DIM, FW = FW_DIM, P = P_DIM;

    size_t sz_I = (size_t)C * H * W * sizeof(double);
    size_t sz_F = (size_t)K * C * FH * FW * sizeof(double);

    double* h_I = (double*)calloc((size_t)C * H * W, sizeof(double));
    double* h_F = (double*)malloc(sz_F);

    for (int c = 0; c < C; c++)
        for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++)
                h_I[c*H*W + h*W + w] = (double)c * (w + h);

    for (int k = 0; k < K; k++)
        for (int c = 0; c < C; c++)
            for (int fh = 0; fh < FH; fh++)
                for (int fw = 0; fw < FW; fw++)
                    h_F[k*C*FH*FW + c*FH*FW + fh*FW + fw] = (double)(c+k) * (fh+fw);

    cudnnHandle_t handle;
    CUDNN_CHECK(cudnnCreate(&handle));

    cudnnTensorDescriptor_t  in_desc, out_desc;
    cudnnFilterDescriptor_t  flt_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&in_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&out_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&flt_desc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(in_desc,
        CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, C, H, W));

    CUDNN_CHECK(cudnnSetFilter4dDescriptor(flt_desc,
        CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, K, C, FH, FW));

    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc,
        P, P, 1, 1, 1, 1,
        CUDNN_CONVOLUTION,    
        CUDNN_DATA_DOUBLE));

    int on, oc, oh, ow;
    CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(
        conv_desc, in_desc, flt_desc, &on, &oc, &oh, &ow));
    fprintf(stderr, "cuDNN output: %dx%dx%dx%d\n", on, oc, oh, ow);

    size_t sz_O = (size_t)on * oc * oh * ow * sizeof(double);

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(out_desc,
        CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, on, oc, oh, ow));

    double *d_I, *d_F, *d_O;
    cudaMalloc(&d_I, sz_I);
    cudaMalloc(&d_F, sz_F);
    cudaMalloc(&d_O, sz_O);
    cudaMemcpy(d_I, h_I, sz_I, cudaMemcpyHostToDevice);
    cudaMemcpy(d_F, h_F, sz_F, cudaMemcpyHostToDevice);
    cudaMemset(d_O, 0, sz_O);

    int            req = 8, ret = 0;
    cudnnConvolutionFwdAlgoPerf_t perf[8];
    CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithm(
        handle, in_desc, flt_desc, conv_desc, out_desc, req, &ret, perf));

    cudnnConvolutionFwdAlgo_t algo = perf[0].algo;
    fprintf(stderr, "Best algorithm: %d  (%.4f ms)\n", (int)algo, perf[0].time);

    size_t ws_bytes = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        handle, in_desc, flt_desc, conv_desc, out_desc, algo, &ws_bytes));
    fprintf(stderr, "Workspace: %.2f MB\n", ws_bytes / 1048576.0);

    void* d_ws = nullptr;
    if (ws_bytes > 0) cudaMalloc(&d_ws, ws_bytes);

    double alpha = 1.0, beta = 0.0;

    CUDNN_CHECK(cudnnConvolutionForward(handle,
        &alpha, in_desc, d_I, flt_desc, d_F,
        conv_desc, algo, d_ws, ws_bytes,
        &beta,  out_desc, d_O));
    cudaDeviceSynchronize();

    initialize_timer(); start_timer();
    CUDNN_CHECK(cudnnConvolutionForward(handle,
        &alpha, in_desc, d_I, flt_desc, d_F,
        conv_desc, algo, d_ws, ws_bytes,
        &beta,  out_desc, d_O));
    cudaDeviceSynchronize();
    stop_timer();
    double t_ms = elapsed_time() * 1e3;

    double* h_O = (double*)malloc(sz_O);
    cudaMemcpy(h_O, d_O, sz_O, cudaMemcpyDeviceToHost);

    double checksum = 0.0;
    for (size_t i = 0; i < (size_t)on*oc*oh*ow; i++)
        checksum += h_O[i];

    fprintf(stderr, "C3: checksum=%.6f  time=%.3f ms\n", checksum, t_ms);
    printf("%.6f,%.3f\n", checksum, t_ms);

    if (d_ws) cudaFree(d_ws);
    cudaFree(d_I); cudaFree(d_F); cudaFree(d_O);
    free(h_I); free(h_F); free(h_O);
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(in_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(out_desc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(flt_desc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
    CUDNN_CHECK(cudnnDestroy(handle));
    cudaDeviceReset();
    return 0;
}
