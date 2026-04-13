

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

static double now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

int main() {
    int K_vals[] = {1, 5, 10, 50, 100};
    int nK = 5;

    printf("K_millions,time_ms\n");

    for (int ki = 0; ki < nK; ki++) {
        int K = K_vals[ki];
        int N = K * 1000000;

        float* A = (float*)malloc((size_t)N * sizeof(float));
        float* B = (float*)malloc((size_t)N * sizeof(float));
        float* C = (float*)malloc((size_t)N * sizeof(float));
        if (!A || !B || !C) { fprintf(stderr, "malloc failed for K=%d\n", K); exit(1); }

        for (int i = 0; i < N; i++) { A[i] = 1.0f; B[i] = 2.0f; }

        for (int i = 0; i < N; i++) C[i] = A[i] + B[i];

        double t0 = now_ms();
        for (int i = 0; i < N; i++) C[i] = A[i] + B[i];
        double t1 = now_ms();

        printf("%d,%.3f\n", K, t1 - t0);
        fflush(stdout);

        free(A); free(B); free(C);
    }
    return 0;
}
