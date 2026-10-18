#define _GNU_SOURCE
/* Trunk projection batching ceiling.
   Same arithmetic and per-token reduction order as clover-one.c Qm (direct int8 path),
   swept over how many tokens share one pass over the weights.
     array  : accumulators indexed by a runtime t, as Qm does with NPOS_SLOTS-wide arrays
     blockN : tokens taken N at a time with N compile-time, so accumulators stay in registers
   Weights and inputs are synthetic: valid for timing, not for meaning. Four matrix copies
   are rotated so every call streams from memory rather than from the 128 MB L3. */
#include <immintrin.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { MAXT = 128, COPIES = 4 };

typedef void (*Kernel)(float *const *, const float *const *, int, const unsigned char *, int, int);

static double clk(void)
{
    struct timespec m; clock_gettime(CLOCK_MONOTONIC, &m);
    return (double)m.tv_sec + (double)m.tv_nsec / 1e9;
}

static inline __m256 decode8(const unsigned char *p)
{
    return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)p)));
}

static inline float reduce(__m256 a, __m256 b)
{
    __m256 vs = _mm256_add_ps(a, b);
    __m128 lo = _mm_add_ps(_mm256_castps256_ps128(vs), _mm256_extractf128_ps(vs, 1));
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    return _mm_cvtss_f32(lo);
}

static void k_array(float *const *Y, const float *const *Xs, int T,
                    const unsigned char *W, int in, int out)
{
    const size_t rowb = (size_t)4 + (size_t)in;
#pragma omp parallel for schedule(static)
    for (int o = 0; o < out; o++) {
        const unsigned char *row = W + (size_t)o * rowb;
        float scale; memcpy(&scale, row, 4);
        const unsigned char *w = row + 4;
        __m256 v0[MAXT], v1[MAXT];
        for (int t = 0; t < T; t++) { v0[t] = _mm256_setzero_ps(); v1[t] = _mm256_setzero_ps(); }
        int i = 0;
        for (; i + 15 < in; i += 16) {
            const __m256 w0 = decode8(w + i), w1 = decode8(w + i + 8);
            for (int t = 0; t < T; t++) {
                v0[t] = _mm256_fmadd_ps(w0, _mm256_loadu_ps(Xs[t] + i), v0[t]);
                v1[t] = _mm256_fmadd_ps(w1, _mm256_loadu_ps(Xs[t] + i + 8), v1[t]);
            }
        }
        for (int t = 0; t < T; t++) {
            float a = reduce(v0[t], v1[t]);
            for (int k = i; k < in; k++) a = a + (float)(int8_t)w[k] * Xs[t][k];
            Y[t][o] = a * scale;
        }
    }
}

#define BLOCK(TB) \
static void k_block##TB(float *const *Y, const float *const *Xs, int T, \
                        const unsigned char *W, int in, int out) \
{ \
    const size_t rowb = (size_t)4 + (size_t)in; \
    _Pragma("omp parallel for schedule(static)") \
    for (int o = 0; o < out; o++) { \
        const unsigned char *row = W + (size_t)o * rowb; \
        float scale; memcpy(&scale, row, 4); \
        const unsigned char *w = row + 4; \
        for (int t0 = 0; t0 + TB <= T; t0 += TB) { \
            const float *x[TB]; \
            __m256 a0[TB], a1[TB]; \
            _Pragma("GCC unroll 16") for (int t = 0; t < TB; t++) { \
                x[t] = Xs[t0 + t]; a0[t] = _mm256_setzero_ps(); a1[t] = _mm256_setzero_ps(); } \
            int i = 0; \
            for (; i + 15 < in; i += 16) { \
                const __m256 w0 = decode8(w + i), w1 = decode8(w + i + 8); \
                _Pragma("GCC unroll 16") for (int t = 0; t < TB; t++) { \
                    a0[t] = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x[t] + i), a0[t]); \
                    a1[t] = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x[t] + i + 8), a1[t]); } \
            } \
            _Pragma("GCC unroll 16") for (int t = 0; t < TB; t++) { \
                float a = reduce(a0[t], a1[t]); \
                for (int k = i; k < in; k++) a = a + (float)(int8_t)w[k] * x[t][k]; \
                Y[t0 + t][o] = a * scale; } \
        } \
    } \
}
BLOCK(2)
BLOCK(4)
BLOCK(6)
BLOCK(8)

typedef struct { const char *name; Kernel run; int block; } Entry;

int main(int argc, char **argv)
{
    const int in = argc > 1 ? atoi(argv[1]) : 7168;
    const int out = argc > 2 ? atoi(argv[2]) : 12288;
    const size_t rowb = (size_t)4 + (size_t)in, bytes = rowb * (size_t)out;
    unsigned char *mat[COPIES];
    uint64_t s = 88172645463325252ULL;
    for (int c = 0; c < COPIES; c++) {
        mat[c] = aligned_alloc(64, (bytes + 63) / 64 * 64);
        if (!mat[c]) { fputs("alloc failed\n", stderr); return 1; }
        for (int o = 0; o < out; o++) {
            unsigned char *row = mat[c] + (size_t)o * rowb;
            float scale = 0.01f; memcpy(row, &scale, 4);
            for (int i = 0; i < in; i++) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; row[4 + i] = (unsigned char)(s >> 11); }
        }
    }
    float *X = malloc((size_t)MAXT * in * sizeof(float));
    float *Yb = malloc((size_t)MAXT * out * sizeof(float));
    float *Yr = malloc((size_t)MAXT * out * sizeof(float));
    const float *Xp[MAXT]; float *Y[MAXT], *R[MAXT];
    if (!X || !Yb || !Yr) { fputs("alloc failed\n", stderr); return 1; }
    for (size_t k = 0; k < (size_t)MAXT * in; k++) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; X[k] = (float)((double)(s % 2000) / 1000.0 - 1.0); }
    for (int t = 0; t < MAXT; t++) { Xp[t] = X + (size_t)t * in; Y[t] = Yb + (size_t)t * out; R[t] = Yr + (size_t)t * out; }

    const Entry entries[] = { {"array ", k_array, 1}, {"block2", k_block2, 2}, {"block4", k_block4, 4},
                              {"block6", k_block6, 6}, {"block8", k_block8, 8} };
    int sweeps[32] = {1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128};
    size_t sweep_count = 12;
    if (argc > 3) {
        sweep_count = 0;
        for (char *token = strtok(argv[3], ","); token && sweep_count < 32; token = strtok(NULL, ","))
            sweeps[sweep_count++] = atoi(token);
    }
    printf("matrix %d x %d int8, %.1f MB, %d threads\n\n", out, in, bytes / 1e6, omp_get_max_threads());
    printf("%-7s %4s %10s %11s %9s %9s %10s %s\n", "kernel", "T", "ms/call", "ms/token", "GB/s", "GMAC/s", "vs T=1", "bits");
    double base = 0.0;
    for (size_t si = 0; si < sweep_count; si++) {
        const int T = sweeps[si];
        if (T < 1 || T > MAXT) continue;
        k_array(R, Xp, T, mat[0], in, out);                         /* reference output for this T */
        for (size_t e = 0; e < sizeof entries / sizeof *entries; e++) {
            const Entry *k = &entries[e];
            if (T % k->block) continue;
            k->run(Y, Xp, T, mat[0], in, out);
            int same = 1;
            for (int t = 0; t < T && same; t++) same = !memcmp(Y[t], R[t], (size_t)out * sizeof(float));
            k->run(Y, Xp, T, mat[1], in, out);                      /* warm, untimed */
            double t0 = clk(); k->run(Y, Xp, T, mat[2], in, out); double one = clk() - t0;
            int reps = (int)(0.8 / one); if (reps < 3) reps = 3; if (reps > 300) reps = 300;
            t0 = clk();
            for (int r = 0; r < reps; r++) k->run(Y, Xp, T, mat[(r + 3) % COPIES], in, out);
            double per = (clk() - t0) / reps;
            if (!base) base = per / T;
            printf("%-7s %4d %10.3f %11.4f %9.1f %9.1f %9.2fx %s\n", k->name, T, per * 1e3, per * 1e3 / T,
                   (double)bytes / per / 1e9, (double)in * out * T / per / 1e9, base / (per / T),
                   same ? "identical" : "DIFFERENT");
            fflush(stdout);
        }
    }
    {
        /* The batched result against one call per token, which is what the engine promises. */
        const int T = 32; int ok = 1;
        float *Z = malloc((size_t)out * sizeof(float)); float *zp[1] = { Z };
        k_block8(Y, Xp, T, mat[0], in, out);
        for (int t = 0; t < T && ok; t++) {
            const float *one[1] = { Xp[t] };
            k_array(zp, one, 1, mat[0], in, out);
            ok = !memcmp(Z, Y[t], (size_t)out * sizeof(float));
        }
        printf("\nblock8 at T=32 vs one call per token: %s\n", ok ? "identical" : "DIFFERENT");
    }
    return 0;
}
