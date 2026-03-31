///
/// vecadd_host.cpp
/// For COMS E6998 Spring 2026 — HW3 Part-B, Q1
///
/// CPU-only vector addition benchmark.
/// Usage: ./vecadd_host <N>
///   N = total number of float elements per vector
///

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

static double get_time_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
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

    float* A = (float*)malloc(N * sizeof(float));
    float* B = (float*)malloc(N * sizeof(float));
    float* C = (float*)malloc(N * sizeof(float));
    if (!A || !B || !C) { fprintf(stderr, "malloc failed\n"); return 1; }

    // Initialize
    for (int i = 0; i < N; i++) {
        A[i] = (float)i;
        B[i] = (float)(N - i);
    }

    // Warm-up
    for (int i = 0; i < N; i++) C[i] = A[i] + B[i];

    // Timed run
    double t0 = get_time_sec();
    for (int i = 0; i < N; i++) C[i] = A[i] + B[i];
    double t1 = get_time_sec();

    double elapsed_ms = (t1 - t0) * 1e3;
    // 2 reads + 1 write per element
    double bw_GBs = (3.0 * N * sizeof(float)) / (t1 - t0) * 1e-9;

    printf("Time: %.4f ms, Bandwidth: %.3f GB/s\n", elapsed_ms, bw_GBs);

    // Verify
    int errors = 0;
    for (int i = 0; i < N; i++)
        if (fabsf(C[i] - (float)N) > 1e-5f) errors++;
    printf("Test %s\n", errors == 0 ? "PASSED" : "FAILED");

    free(A); free(B); free(C);
    return 0;
}
