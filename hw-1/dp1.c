#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float dp(long N, float *pA, float *pB) {
    float R = 0.0;
    int j;
    for (j = 0; j < N; j++)
        R += pA[j] * pB[j];
    return R;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <N> <reps>", argv[0]);
        return 1;
    }

    long N    = atol(argv[1]);
    int  reps = atoi(argv[2]);

    float *pA = (float *)malloc(N * sizeof(float));
    float *pB = (float *)malloc(N * sizeof(float));
    if (!pA || !pB) {
        fprintf(stderr, "Memory allocation not success");
        free(pA); free(pB);
        return 1;
    }

    for (long i = 0; i < N; i++) {
        pA[i] = 1.0f;
        pB[i] = 1.0f;
    }

    double *times = (double *)malloc(reps * sizeof(double));
    if (!times) {
        fprintf(stderr, "Memory allocation not success");
        free(pA); free(pB);
        return 1;
    }

    struct timespec t0, t1;
    for (int r = 0; r < reps; r++) {
        clock_gettime(CLOCK_MONOTONIC, &t0);
        volatile float R = dp(N, pA, pB);
        (void)R;
        clock_gettime(CLOCK_MONOTONIC, &t1);
        times[r] = (t1.tv_sec - t0.tv_sec) +
                   (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    }

    int half  = reps / 2;
    int count = reps - half;
    double sum = 0.0;
    for (int r = half; r < reps; r++)
        sum += times[r];
    double avg_time = sum / count;

    double bytes     = 2.0 * N * sizeof(float);
    double bandwidth = bytes / avg_time / 1e9;
    double flops     = 2.0 * N / avg_time;

    printf("N: %ld   <T>: %f sec  B: %.3f GB/sec  F: %.3f FLOPs/sec \n",
           N, avg_time, bandwidth, flops);

    free(pA); free(pB); free(times);
    return 0;
}
