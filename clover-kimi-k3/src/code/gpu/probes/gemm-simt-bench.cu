// Track A: the same 8-bit weight matrix product with a rewritten kernel (gemm_i8_v2 in ../kernels.cuh) against the first one (gemm_i8),
// on the real weights of layer 0 and the real embedding rows, for 1000 tokens. Reports time, TFLOP/s and whether the bits are the same.
//
//   gemm-simt-bench TRUNK0_DIR EMBED_F32 [REPEATS]
// build: nvcc -O3 --fmad=false -arch=sm_86 -o gemm-simt-bench gemm-simt-bench.cu
#include "bench-common.cuh"

struct Job { const Shape *s; const float *X; float *C; int N; int variant; };
static void launch(void *p)
{
    Job *j = (Job *)p;
    dim3 grid((j->s->R + BN - 1) / BN, (j->N + BM - 1) / BM);
    if (j->variant == 1) gemm_i8<<<grid, 256>>>(j->X, j->N, j->s->K, j->s->codes, j->s->code_stride, j->s->scales, j->s->scale_stride, j->s->R, j->C, j->s->R);
    else gemm_i8_v2<<<grid, 256>>>(j->X, j->N, j->s->K, j->s->codes, j->s->code_stride, j->s->scales, j->s->scale_stride, j->s->R, j->C, j->s->R);
}

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: gemm-simt-bench TRUNK0_DIR EMBED_F32 [REPEATS]\n"); return 2; }
    const int reps = argc > 3 ? atoi(argv[3]) : 10;
    cudaDeviceProp prop; CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("%s\n", prop.name);
    Trunk T; load_trunk0(argv[1], T);
    std::vector<unsigned char> embed = slurp(argv[2]);
    const int N = (int)(embed.size() / (W * 4));
    Shape shapes[3] = { record_shape(T, 26, "dense gate/up 33792 x 7168"), record_shape(T, 28, "dense down 7168 x 33792"), qkv_shape(T, 0, "q/k/v 12288 x 7168") };
    cudaEvent_t a, b; CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
    for (int si = 0; si < 3; si++) {
        const Shape &s = shapes[si];
        float *X, *C1, *C2;
        CHECK(cudaMalloc(&X, (size_t)N * s.K * 4)); CHECK(cudaMalloc(&C1, (size_t)N * s.R * 4)); CHECK(cudaMalloc(&C2, (size_t)N * s.R * 4));
        if (s.K == W) CHECK(cudaMemcpy(X, embed.data(), (size_t)N * W * 4, cudaMemcpyHostToDevice));
        else fill_random<<<(unsigned)(((size_t)N * s.K + 255) / 256), 256>>>(X, (size_t)N * s.K, 7);
        Job j1 = { &s, X, C1, N, 1 }, j2 = { &s, X, C2, N, 2 };
        const float t1 = time_kernel(a, b, reps, launch, &j1), t2 = time_kernel(a, b, reps, launch, &j2);
        std::vector<float> h1((size_t)N * s.R), h2((size_t)N * s.R);
        CHECK(cudaMemcpy(h1.data(), C1, h1.size() * 4, cudaMemcpyDeviceToHost)); CHECK(cudaMemcpy(h2.data(), C2, h2.size() * 4, cudaMemcpyDeviceToHost));
        size_t same; double maxabs, rel; compare_floats(h2, h1, &same, &maxabs, &rel);
        const double flop = 2.0 * N * s.R * s.K;
        printf("%-28s v1 %7.2f ms %6.2f TFLOP/s | v2 %7.2f ms %6.2f TFLOP/s | v2 is %.2fx | same bits as v1: %.4f%% (max abs diff %.3g)\n", s.name,
               t1, flop / (t1 * 1e-3) / 1e12, t2, flop / (t2 * 1e-3) / 1e12, t1 / t2, 100.0 * same / h1.size(), maxabs);
        cudaFree(X); cudaFree(C1); cudaFree(C2);
    }
    return 0;
}
