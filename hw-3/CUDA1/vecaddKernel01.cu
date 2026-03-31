///
/// vecAddKernel01.cu
/// For COMS E6998 Spring 2026
/// Instructor: Kaoutar El Maghraoui
///
/// This Kernel adds two Vectors A and B in C on GPU
/// using coalesced memory access.
///
/// Coalescing strategy:
///   Thread globalIdx handles elements at indices
///   globalIdx, globalIdx + totalThreads, globalIdx + 2*totalThreads, ...
///   so adjacent threads always access adjacent memory locations.
///

__global__ void AddVectors(const float* A, const float* B, float* C, int N)
{
    int totalThreads = gridDim.x * blockDim.x;
    int globalIdx    = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 0; i < N; i++) {
        int idx = globalIdx + i * totalThreads;
        C[idx] = A[idx] + B[idx];
    }
}
