// The ceiling for the matrix products on this card: cuBLAS on the same shapes as layer 0's weight matrices, with float weights (no 8-bit decode).
// Gives what the card does on these shapes when the kernel is NVIDIA's, to read the layer's own kernel against.
//
//   sgemm-ceiling [REPEATS]       build: nvcc -O3 -arch=sm_86 -o sgemm-ceiling sgemm-ceiling.cu -lcublas
//
// Each shape is C[N x R] = X[N x K] * W^T with N = 1000 tokens: the dense gate/up (R 33792, K 7168), the dense down (R 7168, K 33792),
// one q/k/v component (R 12288, K 7168). Modes: plain float, and TF32 tensor cores (10-bit mantissa inputs, float accumulation: NOT the same
// numbers as the float product, shown only as a speed reference).
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)
#define CUB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "%s:%d: cublas error %d\n", __FILE__, __LINE__, (int)s_); exit(1); } } while (0)

__global__ void fill(float *p, size_t n, unsigned seed)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { unsigned x = (unsigned)i * 2654435761u + seed; x ^= x >> 15; x *= 2246822519u; x ^= x >> 13; p[i] = ((x & 0xffff) - 32768) / 32768.0f; }
}

int main(int argc, char **argv)
{
    const int repeats = argc > 1 ? atoi(argv[1]) : 10;
    const int N = 1000;
    struct { const char *name; int R, K; } shapes[] = { { "dense gate or up", 33792, 7168 }, { "dense down", 7168, 33792 }, { "q, k or v component", 12288, 7168 } };
    cudaDeviceProp prop; CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("%s, %d SMs, max clock %.0f MHz, peak float with FMA %.1f TFLOP/s at that clock\n", prop.name, prop.multiProcessorCount, prop.clockRate / 1000.0,
           2.0 * prop.multiProcessorCount * 128 * prop.clockRate * 1e3 / 1e12);
    cublasHandle_t h; CUB(cublasCreate(&h));
    cudaEvent_t a, b; CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
    for (int s = 0; s < 3; s++) {
        const int R = shapes[s].R, K = shapes[s].K;
        float *W, *X, *C;
        CHECK(cudaMalloc(&W, (size_t)R * K * 4)); CHECK(cudaMalloc(&X, (size_t)N * K * 4)); CHECK(cudaMalloc(&C, (size_t)N * R * 4));
        fill<<<(unsigned)(((size_t)R * K + 255) / 256), 256>>>(W, (size_t)R * K, 1); fill<<<(unsigned)(((size_t)N * K + 255) / 256), 256>>>(X, (size_t)N * K, 2);
        CHECK(cudaDeviceSynchronize());
        for (int mode = 0; mode < 2; mode++) {
            CUB(cublasSetMathMode(h, mode ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH));
            const float one = 1.0f, zero = 0.0f;
            // column-major view: C^T (R x N) = W (K x R, transposed) * X (K x N)
            CUB(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, R, N, K, &one, W, K, X, K, &zero, C, R));
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaEventRecord(a));
            for (int i = 0; i < repeats; i++) CUB(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, R, N, K, &one, W, K, X, K, &zero, C, R));
            CHECK(cudaEventRecord(b)); CHECK(cudaEventSynchronize(b));
            float ms; CHECK(cudaEventElapsedTime(&ms, a, b)); ms /= repeats;
            printf("%-20s R=%5d K=%5d  %-5s %8.2f ms  %6.2f TFLOP/s\n", shapes[s].name, R, K, mode ? "TF32" : "float", ms, 2.0 * N * R * K / (ms * 1e-3) / 1e12);
        }
        cudaFree(W); cudaFree(X); cudaFree(C);
    }
    return 0;
}
