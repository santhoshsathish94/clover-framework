// Track B: the same product on the integer tensor cores. The weights are exact 8-bit integers. Each token's activations are scaled by the token's
// largest value and cut into P signed 8-bit digits (P = 3 gives 23 bits, P = 2 gives 15 bits); one integer product per digit (exact, int32 sums), the
// digits recombined in double, then the row scale. Compared with the float kernel (gemm_i8) on the real weights and embedding rows.
//
//   gemm-imma-bench TRUNK0_DIR EMBED_F32 [REPEATS]
// build: nvcc -O3 --fmad=false -arch=sm_86 -o gemm-imma-bench gemm-imma-bench.cu -lcublas
#include "bench-common.cuh"
#include <cublas_v2.h>

#define CUB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "%s:%d: cublas error %d\n", __FILE__, __LINE__, (int)s_); exit(1); } } while (0)

__global__ void quantize_planes(int8_t *X8, double *tscale, const float *X, int N, int K, int P)
{
    const int n = blockIdx.x, tid = threadIdx.x;
    const float *x = X + (size_t)n * K;
    float m = 0.0f;
    for (int c = tid; c < K; c += blockDim.x) m = fmaxf(m, fabsf(x[c]));
    __shared__ float sh[32], smax;
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, o));
    if ((tid & 31) == 0) sh[tid >> 5] = m;
    __syncthreads();
    m = tid < blockDim.x / 32 ? sh[tid] : 0.0f;
    if (tid < 32) for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, o));
    if (tid == 0) smax = m;
    __syncthreads();
    const double qmax = P == 3 ? 8355711.0 : 32639.0;   // the largest value whose balanced digits all fit in 8 bits
    const double inv = smax > 0.0f ? qmax / (double)smax : 0.0;
    if (tid == 0) tscale[n] = smax > 0.0f ? (double)smax / qmax : 0.0;
    for (int c = tid; c < K; c += blockDim.x) {
        long long t = llrint((double)x[c] * inv);
        for (int p = 0; p < P; p++) {
            int d;
            if (p < P - 1) { d = (int)(((t + 128) & 255) - 128); t = (t - d) >> 8; } else d = (int)t;
            X8[((size_t)p * N + n) * K + c] = (int8_t)d;
        }
    }
}

__global__ void combine_planes(float *out, const int *C32, const double *tscale, const unsigned char *scales, size_t scale_stride, int N, int R, int P)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)N * R) return;
    const int n = (int)(i / R), r = (int)(i % R);
    long long acc = 0;
    for (int p = 0; p < P; p++) acc += (long long)C32[((size_t)p * N + n) * R + r] << (8 * p);
    out[i] = (float)((double)acc * tscale[n] * (double)f32at(scales + (size_t)r * scale_stride));
}

struct Job { const Shape *s; const float *X; float *C; int N; int P; int8_t *X8; int *C32; double *ts; cublasHandle_t h; float ms[3]; cudaEvent_t e[4]; };
static void run_planes(void *p)
{
    Job *j = (Job *)p;
    CHECK(cudaEventRecord(j->e[0]));
    quantize_planes<<<j->N, 256>>>(j->X8, j->ts, j->X, j->N, j->s->K, j->P);
    CHECK(cudaEventRecord(j->e[1]));
    const int alpha = 1, beta = 0;
    CUB(cublasGemmEx(j->h, CUBLAS_OP_T, CUBLAS_OP_N, j->s->R, j->P * j->N, j->s->K, &alpha, j->s->codes, CUDA_R_8I, (int)j->s->code_stride, j->X8, CUDA_R_8I, j->s->K,
                     &beta, j->C32, CUDA_R_32I, j->s->R, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CHECK(cudaEventRecord(j->e[2]));
    combine_planes<<<(unsigned)(((size_t)j->N * j->s->R + 255) / 256), 256>>>(j->C, j->C32, j->ts, j->s->scales, j->s->scale_stride, j->N, j->s->R, j->P);
    CHECK(cudaEventRecord(j->e[3]));
}
struct FJob { const Shape *s; const float *X; float *C; int N; };
static void run_float(void *p)
{
    FJob *j = (FJob *)p;
    dim3 grid((j->s->R + BN - 1) / BN, (j->N + BM - 1) / BM);
    gemm_i8<<<grid, 256>>>(j->X, j->N, j->s->K, j->s->codes, j->s->code_stride, j->s->scales, j->s->scale_stride, j->s->R, j->C, j->s->R);
}

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: gemm-imma-bench TRUNK0_DIR EMBED_F32 [REPEATS]\n"); return 2; }
    const int reps = argc > 3 ? atoi(argv[3]) : 10;
    cudaDeviceProp prop; CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("%s\n", prop.name);
    Trunk T; load_trunk0(argv[1], T);
    std::vector<unsigned char> embed = slurp(argv[2]);
    const int N = (int)(embed.size() / (W * 4));
    Shape shapes[3] = { record_shape(T, 26, "dense gate/up 33792 x 7168"), record_shape(T, 28, "dense down 7168 x 33792"), qkv_shape(T, 0, "q/k/v 12288 x 7168") };
    cublasHandle_t h; CUB(cublasCreate(&h));
    cudaEvent_t a, b; CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
    for (int si = 0; si < 3; si++) {
        const Shape &s = shapes[si];
        float *X, *Cf, *Cq;
        CHECK(cudaMalloc(&X, (size_t)N * s.K * 4)); CHECK(cudaMalloc(&Cf, (size_t)N * s.R * 4)); CHECK(cudaMalloc(&Cq, (size_t)N * s.R * 4));
        if (s.K == W) CHECK(cudaMemcpy(X, embed.data(), (size_t)N * W * 4, cudaMemcpyHostToDevice));
        else fill_random<<<(unsigned)(((size_t)N * s.K + 255) / 256), 256>>>(X, (size_t)N * s.K, 7);
        FJob fj = { &s, X, Cf, N };
        const float tf = time_kernel(a, b, reps, run_float, &fj);
        std::vector<float> hf((size_t)N * s.R);
        CHECK(cudaMemcpy(hf.data(), Cf, hf.size() * 4, cudaMemcpyDeviceToHost));
        const double flop = 2.0 * N * s.R * s.K;
        printf("%-28s float kernel %7.2f ms (%5.2f TFLOP/s)\n", s.name, tf, flop / (tf * 1e-3) / 1e12);
        for (int P = 2; P <= 3; P++) {
            Job j; memset(&j, 0, sizeof j);
            j.s = &s; j.X = X; j.C = Cq; j.N = N; j.P = P; j.h = h;
            CHECK(cudaMalloc(&j.X8, (size_t)P * N * s.K)); CHECK(cudaMalloc(&j.C32, (size_t)P * N * s.R * 4)); CHECK(cudaMalloc(&j.ts, (size_t)N * 8));
            for (int e = 0; e < 4; e++) CHECK(cudaEventCreate(&j.e[e]));
            const float t = time_kernel(a, b, reps, run_planes, &j);
            float q, g, c;
            CHECK(cudaEventElapsedTime(&q, j.e[0], j.e[1])); CHECK(cudaEventElapsedTime(&g, j.e[1], j.e[2])); CHECK(cudaEventElapsedTime(&c, j.e[2], j.e[3]));
            std::vector<float> hq((size_t)N * s.R);
            CHECK(cudaMemcpy(hq.data(), Cq, hq.size() * 4, cudaMemcpyDeviceToHost));
            size_t same; double maxabs, rel; compare_floats(hq, hf, &same, &maxabs, &rel);
            printf("   %d planes: %7.2f ms in all (quantise %.2f, integer product %.2f = %.1f TOPS, combine %.2f) | %.2fx the float kernel | vs the float result: relative RMS difference %.3g, max abs %.3g, same bits %.3f%%\n",
                   P, t, q, g, 2.0 * P * N * s.R * s.K / (g * 1e-3) / 1e12, c, tf / t, rel, maxabs, 100.0 * same / hq.size());
            cudaFree(j.X8); cudaFree(j.C32); cudaFree(j.ts);
        }
        cudaFree(X); cudaFree(Cf); cudaFree(Cq);
    }
    return 0;
}
