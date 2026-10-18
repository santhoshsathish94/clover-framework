#define _GNU_SOURCE
/* Why do the trunk projections run at 325-470 GMAC/s, and what bounds them?
   Part 1: the raw FMA peak of this machine (registers only), one thread and all threads.
   Part 2: the layer-0/layer-1 projection kernel (1 weight row x 4 tokens per pass, AVX2) against
           variants that take R weight rows x G tokens per pass, so each activation load feeds
           more multiplies. Every variant must give a bit-identical result to the first.
   Shapes are those of the real matrices. Weights are random 8-bit codes; the arithmetic path is the
   identity-palette path of the production kernel.

     proj-bench ROWS COLS [REPS]                                                          */
#include <immintrin.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

enum { TOK = 32 };

#define KY(NAME, R, G) \
static inline __attribute__((always_inline)) void NAME(const float *const *x, const unsigned char *const *ids, \
    unsigned width, const float *scale, float *out) \
{ \
    __m256 a0[R][G], a1[R][G]; \
    _Pragma("GCC unroll 8") for (int r = 0; r < R; r++) _Pragma("GCC unroll 8") for (int g = 0; g < G; g++) { \
        a0[r][g] = _mm256_setzero_ps(); a1[r][g] = _mm256_setzero_ps(); } \
    for (unsigned c = 0; c < width; c += 16) { \
        __m256 w0[R], w1[R]; \
        _Pragma("GCC unroll 8") for (int r = 0; r < R; r++) { \
            w0[r] = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(ids[r] + c)))); \
            w1[r] = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(ids[r] + c + 8)))); } \
        _Pragma("GCC unroll 8") for (int r = 0; r < R; r++) _Pragma("GCC unroll 8") for (int g = 0; g < G; g++) { \
            a0[r][g] = _mm256_fmadd_ps(w0[r], _mm256_loadu_ps(x[g] + c), a0[r][g]); \
            a1[r][g] = _mm256_fmadd_ps(w1[r], _mm256_loadu_ps(x[g] + c + 8), a1[r][g]); } \
    } \
    _Pragma("GCC unroll 8") for (int r = 0; r < R; r++) _Pragma("GCC unroll 8") for (int g = 0; g < G; g++) { \
        float lanes[16], combined[8]; \
        _mm256_storeu_ps(lanes, a0[r][g]); _mm256_storeu_ps(lanes + 8, a1[r][g]); \
        for (unsigned lane = 0; lane < 8; lane++) combined[lane] = lanes[lane] + lanes[lane + 8]; \
        float sum0 = combined[0] + combined[4], sum1 = combined[1] + combined[5]; \
        float sum2 = combined[2] + combined[6], sum3 = combined[3] + combined[7]; \
        out[r * G + g] = ((sum0 + sum2) + (sum1 + sum3)) * scale[r]; } \
}

#define KZ(NAME, R, G) \
static inline __attribute__((always_inline)) void NAME(const float *const *x, const unsigned char *const *ids, \
    unsigned width, const float *scale, float *out) \
{ \
    __m512 a[R][G]; \
    _Pragma("GCC unroll 8") for (int r = 0; r < R; r++) _Pragma("GCC unroll 8") for (int g = 0; g < G; g++) a[r][g] = _mm512_setzero_ps(); \
    for (unsigned c = 0; c < width; c += 16) { \
        __m512 w[R]; \
        _Pragma("GCC unroll 8") for (int r = 0; r < R; r++) \
            w[r] = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(ids[r] + c)))); \
        _Pragma("GCC unroll 8") for (int r = 0; r < R; r++) _Pragma("GCC unroll 8") for (int g = 0; g < G; g++) \
            a[r][g] = _mm512_fmadd_ps(w[r], _mm512_loadu_ps(x[g] + c), a[r][g]); \
    } \
    _Pragma("GCC unroll 8") for (int r = 0; r < R; r++) _Pragma("GCC unroll 8") for (int g = 0; g < G; g++) { \
        float lanes[16], combined[8]; \
        _mm512_storeu_ps(lanes, a[r][g]); \
        for (unsigned lane = 0; lane < 8; lane++) combined[lane] = lanes[lane] + lanes[lane + 8]; \
        float sum0 = combined[0] + combined[4], sum1 = combined[1] + combined[5]; \
        float sum2 = combined[2] + combined[6], sum3 = combined[3] + combined[7]; \
        out[r * G + g] = ((sum0 + sum2) + (sum1 + sum3)) * scale[r]; } \
}

#define DRIVER(NAME, KERN, R, G) \
static void NAME(const unsigned char *ids, const float *scale, unsigned rows, unsigned cols, const float *X, float *Y) \
{ \
    _Pragma("omp parallel for schedule(static)") \
    for (unsigned row = 0; row < rows; row += R) { \
        const unsigned char *rid[R]; float sc[R]; \
        for (int r = 0; r < R; r++) { rid[r] = ids + (size_t)(row + r) * cols; sc[r] = scale[row + r]; } \
        for (unsigned k = 0; k < TOK; k += G) { \
            const float *x[G]; float out[R * G]; \
            for (int j = 0; j < G; j++) x[j] = X + (size_t)(k + j) * cols; \
            KERN(x, rid, cols, sc, out); \
            for (int r = 0; r < R; r++) for (int j = 0; j < G; j++) Y[(size_t)(k + j) * rows + row + r] = out[r * G + j]; \
        } \
    } \
}

KY(ky_1x4, 1, 4) DRIVER(run_y_1x4, ky_1x4, 1, 4)
KY(ky_2x2, 2, 2) DRIVER(run_y_2x2, ky_2x2, 2, 2)
KY(ky_2x3, 2, 3)
KY(ky_2x4, 2, 4)
KY(ky_4x1, 4, 1) DRIVER(run_y_4x1, ky_4x1, 4, 1)
KZ(kz_1x4, 1, 4) DRIVER(run_z_1x4, kz_1x4, 1, 4)
KZ(kz_2x4, 2, 4) DRIVER(run_z_2x4, kz_2x4, 2, 4)
KZ(kz_4x4, 4, 4) DRIVER(run_z_4x4, kz_4x4, 4, 4)
KZ(kz_4x2, 4, 2) DRIVER(run_z_4x2, kz_4x2, 4, 2)
KZ(kz_8x2, 8, 2) DRIVER(run_z_8x2, kz_8x2, 8, 2)
KZ(kz_4x8, 4, 8) DRIVER(run_z_4x8, kz_4x8, 4, 8)

typedef void (*Run)(const unsigned char *, const float *, unsigned, unsigned, const float *, float *);

/* Column order: 16 consecutive outputs live in the 16 lanes of one register; for each input column j every
   output gets x_j times its weight added to ONE running total. Weights are repacked as [16-row block][column][16 rows]. */
static void run_col_4x4(const unsigned char *pk, const float *scale, unsigned rows, unsigned cols, const float *X, float *Y)
{
#pragma omp parallel for schedule(static)
    for (unsigned row = 0; row < rows; row += 64) {
        const unsigned char *p[4];
        for (int r = 0; r < 4; r++) p[r] = pk + ((size_t)((row >> 4) + r) * cols) * 16;
        for (unsigned k = 0; k < TOK; k += 4) {
            __m512 a[4][4];
            _Pragma("GCC unroll 4") for (int r = 0; r < 4; r++) _Pragma("GCC unroll 4") for (int g = 0; g < 4; g++) a[r][g] = _mm512_setzero_ps();
            const float *x[4];
            for (int g = 0; g < 4; g++) x[g] = X + (size_t)(k + g) * cols;
            for (unsigned j = 0; j < cols; j++) {
                __m512 w[4], b[4];
                _Pragma("GCC unroll 4") for (int r = 0; r < 4; r++) w[r] = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(p[r] + (size_t)j * 16))));
                _Pragma("GCC unroll 4") for (int g = 0; g < 4; g++) b[g] = _mm512_set1_ps(x[g][j]);
                _Pragma("GCC unroll 4") for (int r = 0; r < 4; r++) _Pragma("GCC unroll 4") for (int g = 0; g < 4; g++) a[r][g] = _mm512_fmadd_ps(w[r], b[g], a[r][g]);
            }
            _Pragma("GCC unroll 4") for (int r = 0; r < 4; r++) {
                const __m512 sc = _mm512_loadu_ps(scale + row + r * 16);
                _Pragma("GCC unroll 4") for (int g = 0; g < 4; g++) _mm512_storeu_ps(Y + (size_t)(k + g) * rows + row + r * 16, _mm512_mul_ps(a[r][g], sc));
            }
        }
    }
}

/* ---- raw FMA peak ---- */
static double peak(int zmm, int threads, unsigned long iters)
{
    double start = omp_get_wtime();
    float sink = 0;
#pragma omp parallel num_threads(threads) reduction(+ : sink)
    {
        if (zmm) {
            __m512 a[12], b = _mm512_set1_ps(1.0000001f), c = _mm512_set1_ps(1e-7f);
            for (int k = 0; k < 12; k++) a[k] = _mm512_set1_ps((float)k);
            for (unsigned long i = 0; i < iters; i++)
                _Pragma("GCC unroll 12") for (int k = 0; k < 12; k++) a[k] = _mm512_fmadd_ps(a[k], b, c);
            for (int k = 0; k < 12; k++) sink += _mm512_reduce_add_ps(a[k]);
        } else {
            __m256 a[12], b = _mm256_set1_ps(1.0000001f), c = _mm256_set1_ps(1e-7f);
            for (int k = 0; k < 12; k++) a[k] = _mm256_set1_ps((float)k);
            for (unsigned long i = 0; i < iters; i++)
                _Pragma("GCC unroll 12") for (int k = 0; k < 12; k++) a[k] = _mm256_fmadd_ps(a[k], b, c);
            float t[8];
            for (int k = 0; k < 12; k++) { _mm256_storeu_ps(t, a[k]); sink += t[0]; }
        }
    }
    double sec = omp_get_wtime() - start;
    if (sink == 12345.678f) puts("");
    return (double)threads * iters * 12 * (zmm ? 16 : 8) / sec / 1e9;
}

int main(int argc, char **argv)
{
    if (argc < 3) { fputs("usage: proj-bench ROWS COLS [REPS]\n", stderr); return 2; }
    const unsigned rows = (unsigned)atoi(argv[1]), cols = (unsigned)atoi(argv[2]);
    const int reps = argc > 3 ? atoi(argv[3]) : 5;
    const int threads = omp_get_max_threads();
    if (rows % 8 || cols % 16) { fputs("rows must be a multiple of 8 and cols of 16\n", stderr); return 2; }

    printf("raw FMA peak, registers only (GMAC/s): 1 thread ymm %.1f, 1 thread zmm %.1f; %d threads ymm %.1f, zmm %.1f\n",
        peak(0, 1, 200000000UL), peak(1, 1, 200000000UL), threads, peak(0, threads, 200000000UL), peak(1, threads, 200000000UL));

    unsigned char *ids = aligned_alloc(64, (size_t)rows * cols);
    float *scale = malloc(rows * sizeof *scale), *X = aligned_alloc(64, (size_t)TOK * cols * 4);
    float *Yref = aligned_alloc(64, (size_t)TOK * rows * 4), *Y = aligned_alloc(64, (size_t)TOK * rows * 4);
    uint64_t s = 88172645463325252ULL;
    for (size_t i = 0; i < (size_t)rows * cols; i++) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; ids[i] = (unsigned char)(s >> 11); }
    for (unsigned i = 0; i < rows; i++) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; scale[i] = 0.001f + (float)(s >> 40) * 1e-9f; }
    for (size_t i = 0; i < (size_t)TOK * cols; i++) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; X[i] = ((float)(s >> 40) / 16777216.0f - 0.5f) * 0.1f; }

    struct { const char *name; Run run; } v[] = {
        {"AVX2   1 row  x 4 tokens (production)", run_y_1x4}, {"AVX2   2 rows x 2 tokens", run_y_2x2},
        {"AVX2   4 rows x 1 token ", run_y_4x1},
        {"AVX512 1 row  x 4 tokens", run_z_1x4}, {"AVX512 2 rows x 4 tokens", run_z_2x4},
        {"AVX512 4 rows x 2 tokens", run_z_4x2}, {"AVX512 4 rows x 4 tokens", run_z_4x4},
        {"AVX512 8 rows x 2 tokens", run_z_8x2}, {"AVX512 4 rows x 8 tokens", run_z_4x8}};
    printf("\nmatrix %u rows x %u columns (%.1f MB of weight codes), %d tokens per tile, %d threads, best of %d\n",
        rows, cols, (double)rows * cols / 1e6, TOK, threads, reps);
    for (unsigned i = 0; i < sizeof v / sizeof *v; i++) {
        v[i].run(ids, scale, rows, cols, X, Y);
        double best = 1e9;
        for (int r = 0; r < reps; r++) { double t = omp_get_wtime(); v[i].run(ids, scale, rows, cols, X, Y); t = omp_get_wtime() - t; if (t < best) best = t; }
        if (i == 0) memcpy(Yref, Y, (size_t)TOK * rows * 4);
        const int same = !memcmp(Yref, Y, (size_t)TOK * rows * 4);
        printf("  %-40s %7.1f GMAC/s  %s\n", v[i].name, (double)rows * cols * TOK / best / 1e9, same ? "bit-identical" : "DIFFERENT");
    }
    if (rows % 64 == 0) {
        unsigned char *pk = aligned_alloc(64, (size_t)rows * cols);
#pragma omp parallel for schedule(static)
        for (unsigned blk = 0; blk < rows / 16; blk++)
            for (unsigned j = 0; j < cols; j++)
                for (unsigned l = 0; l < 16; l++) pk[((size_t)blk * cols + j) * 16 + l] = ids[(size_t)(blk * 16 + l) * cols + j];
        run_col_4x4(pk, scale, rows, cols, X, Y);
        double best = 1e9;
        for (int r = 0; r < reps; r++) { double t = omp_get_wtime(); run_col_4x4(pk, scale, rows, cols, X, Y); t = omp_get_wtime() - t; if (t < best) best = t; }
        size_t differ = 0; double worst = 0;
        for (size_t i = 0; i < (size_t)TOK * rows; i++)
            if (memcmp(&Yref[i], &Y[i], 4)) { differ++; double d = fabs((double)Y[i] - Yref[i]) / (fabs((double)Yref[i]) + 1e-30); if (d > worst) worst = d; }
        printf("  %-40s %7.1f GMAC/s  %zu of %zu outputs differ from the 16-lane order\n", "COLUMN ORDER AVX512 64 rows x 4 tokens",
            (double)rows * cols * TOK / best / 1e9, differ, (size_t)TOK * rows);
        free(pk);
    }
    return 0;
}
