import sys
import time
import numpy as np


def dp(N, A, B):
    R = 0.0
    for j in range(0, N):
        R += A[j] * B[j]
    return R


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <N> <reps>", file=sys.stderr)
        sys.exit(1)

    N    = int(sys.argv[1])
    reps = int(sys.argv[2])

    A = np.ones(N, dtype=np.float32)
    B = np.ones(N, dtype=np.float32)

    if N >= 100_000_000:
        print(f"WARNING: N={N} with a Python manual loop will be extremely time consuming"
              f"(estimated hours). Running as required...", file=sys.stderr)

    times = []
    for r in range(reps):
        t0 = time.perf_counter()
        R  = dp(N, A, B)
        t1 = time.perf_counter()
        _ = R
        times.append(t1 - t0)

    half     = reps // 2
    second   = times[half:]
    avg_time = sum(second) / len(second)

    bytes_accessed = 2 * N * 4    
    bandwidth      = bytes_accessed / avg_time / 1e9
    flops          = 2 * N / avg_time

    print(f"N: {N}   <T>: {avg_time:.6f} sec  "
          f"B: {bandwidth:.3f} GB/sec  F: {flops:.3f} FLOPs/sec")


if __name__ == "__main__":
    main()
