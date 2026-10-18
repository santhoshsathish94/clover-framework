#define _GNU_SOURCE
#define TRANSFORMER_NO_MAIN
/* Layer-1 expert kernel study. The expert matrices are 4-bit codes decoded through a per-group table of
   weight pairs, multiplied in DOUBLE precision (a multiply, then a separate add: no fused multiply-add) into
   4 sub-accumulators per token, then folded in a fixed order. Every variant here must give the same bits as
   the shipped arithmetic (proj_core in layer-batch.c).

   Part 1: the raw double-precision multiply-then-add peak of this machine.
   Part 2: the shipped kernel against variants that use 512-bit registers (8 doubles) and take R weight rows
           and G tokens per pass, on real expert data: one thread with the expert resident, and 16 threads on
           16 different experts as in the layer.

     expert-kernel-bench DATASET_DIR [REPS]                                                      */
#include "transformer-1.c"
#include <omp.h>
#include <time.h>

enum { LATW = 3584, EXPW = 3072, MAXTH = 32, MAXG = 8 };
static int nth = 16; /* threads for the multi-thread columns; env EK_THREADS */
#define THREADS nth

static double clk(void)
{
    struct timespec m; clock_gettime(CLOCK_MONOTONIC, &m);
    return (double)m.tv_sec + (double)m.tv_nsec / 1e9;
}
static void bail(const char *what) { fprintf(stderr, "expert-kernel-bench: %s\n", what); exit(1); }

/* ---- the shipped kernel, G tokens per pass (copy of proj_core in layer-batch.c) ---- */
static inline __attribute__((always_inline)) void proj_core(const Root *root, unsigned expert, unsigned matrix,
    const double *const *conv, float *const *out, const int G)
{
    const unsigned width = matrix == 2 ? 3072 : 3584, rows = matrix == 2 ? 3584 : 3072;
    const unsigned char *maps = root->constants + root->palette_count * 2;
    const unsigned char *offsets = maps + root->map_count * 16;
    const unsigned char *refs = offsets + (root->template_count + 1) * 2;
    const unsigned template_id = root_u16(root->index + (expert * 3 + matrix) * 2);
    const unsigned begin = root_u16(offsets + template_id * 2);
    for (unsigned block = 0; block < rows / 64; block++) {
        const unsigned char *data = root->direct + (size_t)expert * ROOT_EXPERT_RAW +
            (matrix < 2 ? (size_t)(matrix * 48 + block) * ROOT_RAW
                        : (size_t)96 * ROOT_RAW + (size_t)block * ROOT_DOWN_RAW);
        for (unsigned row = 0; row < 64; row++) {
            const unsigned char *codes = data + row * width / 2;
            const unsigned char *selectors = data + 64 * width / 2 + row * width / 32;
            __m256d l0[8], l4[8], l8[8], l12[8];
#pragma GCC unroll 8
            for (int g = 0; g < G; g++) {
                l0[g] = _mm256_setzero_pd(); l4[g] = _mm256_setzero_pd();
                l8[g] = _mm256_setzero_pd(); l12[g] = _mm256_setzero_pd();
            }
            for (unsigned coordinate = 0; coordinate < width; coordinate += 16) {
                const float (*pair)[2] = root->pairs[root_u16(refs + (begin + selectors[coordinate / 32]) * 2)];
                const unsigned char *packed = codes + coordinate / 2;
                __m256d w0 = _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd((const double *)pair[packed[0]]), _mm_load_sd((const double *)pair[packed[1]]))));
                __m256d w1 = _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd((const double *)pair[packed[2]]), _mm_load_sd((const double *)pair[packed[3]]))));
                __m256d w2 = _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd((const double *)pair[packed[4]]), _mm_load_sd((const double *)pair[packed[5]]))));
                __m256d w3 = _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd((const double *)pair[packed[6]]), _mm_load_sd((const double *)pair[packed[7]]))));
#pragma GCC unroll 8
                for (int g = 0; g < G; g++) {
                    const double *source = conv[g] + coordinate;
                    l0[g] = _mm256_add_pd(l0[g], _mm256_mul_pd(w0, _mm256_loadu_pd(source)));
                    l4[g] = _mm256_add_pd(l4[g], _mm256_mul_pd(w1, _mm256_loadu_pd(source + 4)));
                    l8[g] = _mm256_add_pd(l8[g], _mm256_mul_pd(w2, _mm256_loadu_pd(source + 8)));
                    l12[g] = _mm256_add_pd(l12[g], _mm256_mul_pd(w3, _mm256_loadu_pd(source + 12)));
                }
            }
#pragma GCC unroll 8
            for (int g = 0; g < G; g++) {
                double sums[4];
                _mm256_storeu_pd(sums, _mm256_add_pd(_mm256_add_pd(l0[g], l8[g]), _mm256_add_pd(l4[g], l12[g])));
                out[g][block * 64 + row] = (float)((sums[0] + sums[2]) + (sums[1] + sums[3]));
            }
        }
    }
}

/* ---- 512-bit version: R rows x G tokens per pass. acc A = [l0 | l4], acc B = [l8 | l12]; per lane the same
   products are added in the same order, and the fold (l0+l8)+(l4+l12) is the same additions. ---- */
#define KZ(NAME, R, G) \
static void NAME(const Root *root, unsigned expert, unsigned matrix, const double *const *conv, float *const *out) \
{ \
    const unsigned width = matrix == 2 ? 3072 : 3584, rows = matrix == 2 ? 3584 : 3072; \
    const unsigned char *maps = root->constants + root->palette_count * 2; \
    const unsigned char *offsets = maps + root->map_count * 16; \
    const unsigned char *refs = offsets + (root->template_count + 1) * 2; \
    const unsigned template_id = root_u16(root->index + (expert * 3 + matrix) * 2); \
    const unsigned begin = root_u16(offsets + template_id * 2); \
    for (unsigned block = 0; block < rows / 64; block++) { \
        const unsigned char *data = root->direct + (size_t)expert * ROOT_EXPERT_RAW + \
            (matrix < 2 ? (size_t)(matrix * 48 + block) * ROOT_RAW : (size_t)96 * ROOT_RAW + (size_t)block * ROOT_DOWN_RAW); \
        for (unsigned row = 0; row < 64; row += R) { \
            const unsigned char *codes[R], *selectors[R]; \
            _Pragma("GCC unroll 8") for (int r = 0; r < R; r++) { \
                codes[r] = data + (row + r) * width / 2; \
                selectors[r] = data + 64 * width / 2 + (row + r) * width / 32; } \
            __m512d a[R][G], b[R][G]; \
            _Pragma("GCC unroll 8") for (int r = 0; r < R; r++) _Pragma("GCC unroll 8") for (int g = 0; g < G; g++) { \
                a[r][g] = _mm512_setzero_pd(); b[r][g] = _mm512_setzero_pd(); } \
            for (unsigned coordinate = 0; coordinate < width; coordinate += 16) { \
                __m512d wa[R], wb[R]; \
                _Pragma("GCC unroll 8") for (int r = 0; r < R; r++) { \
                    const float (*pair)[2] = root->pairs[root_u16(refs + (begin + selectors[r][coordinate / 32]) * 2)]; \
                    const unsigned char *packed = codes[r] + coordinate / 2; \
                    __m256d w0 = _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd((const double *)pair[packed[0]]), _mm_load_sd((const double *)pair[packed[1]])))); \
                    __m256d w1 = _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd((const double *)pair[packed[2]]), _mm_load_sd((const double *)pair[packed[3]])))); \
                    __m256d w2 = _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd((const double *)pair[packed[4]]), _mm_load_sd((const double *)pair[packed[5]])))); \
                    __m256d w3 = _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd((const double *)pair[packed[6]]), _mm_load_sd((const double *)pair[packed[7]])))); \
                    wa[r] = _mm512_insertf64x4(_mm512_castpd256_pd512(w0), w1, 1); \
                    wb[r] = _mm512_insertf64x4(_mm512_castpd256_pd512(w2), w3, 1); } \
                _Pragma("GCC unroll 8") for (int g = 0; g < G; g++) { \
                    const double *source = conv[g] + coordinate; \
                    const __m512d xa = _mm512_loadu_pd(source), xb = _mm512_loadu_pd(source + 8); \
                    _Pragma("GCC unroll 8") for (int r = 0; r < R; r++) { \
                        a[r][g] = _mm512_add_pd(a[r][g], _mm512_mul_pd(wa[r], xa)); \
                        b[r][g] = _mm512_add_pd(b[r][g], _mm512_mul_pd(wb[r], xb)); } \
                } \
            } \
            _Pragma("GCC unroll 8") for (int r = 0; r < R; r++) _Pragma("GCC unroll 8") for (int g = 0; g < G; g++) { \
                const __m512d t = _mm512_add_pd(a[r][g], b[r][g]); \
                double sums[4]; \
                _mm256_storeu_pd(sums, _mm256_add_pd(_mm512_castpd512_pd256(t), _mm512_extractf64x4_pd(t, 1))); \
                out[g][block * 64 + row + r] = (float)((sums[0] + sums[2]) + (sums[1] + sums[3])); } \
        } \
    } \
}

typedef void (*Kernel)(const Root *, unsigned, unsigned, const double *const *, float *const *);
#define KY(NAME, G) static void NAME(const Root *r, unsigned e, unsigned m, const double *const *c, float *const *o) { proj_core(r, e, m, c, o, G); }
KY(ky1, 1) KY(ky2, 2) KY(ky4, 4) KY(ky8, 8)
KZ(kz_1_1, 1, 1) KZ(kz_1_2, 1, 2) KZ(kz_1_4, 1, 4) KZ(kz_1_8, 1, 8)
KZ(kz_2_1, 2, 1) KZ(kz_2_2, 2, 2) KZ(kz_2_4, 2, 4)
KZ(kz_4_1, 4, 1) KZ(kz_4_2, 4, 2) KZ(kz_8_1, 8, 1)

typedef struct { const char *name; int G; Kernel run; int base; } Variant; /* base: index of the shipped kernel at the same G */
static const Variant variants[] = {
    {"shipped, 256-bit, 1 row x 1 token ", 1, ky1, 0}, {"shipped, 256-bit, 1 row x 2 tokens", 2, ky2, 1},
    {"shipped, 256-bit, 1 row x 4 tokens", 4, ky4, 2}, {"shipped, 256-bit, 1 row x 8 tokens", 8, ky8, 3},
    {"512-bit, 1 row x 1 token ", 1, kz_1_1, 0}, {"512-bit, 1 row x 2 tokens", 2, kz_1_2, 1},
    {"512-bit, 1 row x 4 tokens", 4, kz_1_4, 2}, {"512-bit, 1 row x 8 tokens", 8, kz_1_8, 3},
    {"512-bit, 2 rows x 1 token ", 1, kz_2_1, 0}, {"512-bit, 2 rows x 2 tokens", 2, kz_2_2, 1},
    {"512-bit, 2 rows x 4 tokens", 4, kz_2_4, 2},
    {"512-bit, 4 rows x 1 token ", 1, kz_4_1, 0}, {"512-bit, 4 rows x 2 tokens", 2, kz_4_2, 1},
    {"512-bit, 8 rows x 1 token ", 1, kz_8_1, 0}};
enum { NVAR = sizeof variants / sizeof *variants };

/* ---- cost split (outputs are meaningless): the table decode alone, and the multiply-adds alone ---- */
static void k_decode_only(const Root *root, unsigned expert, unsigned matrix, const double *const *conv, float *const *out)
{
    (void)conv;
    const unsigned width = matrix == 2 ? 3072 : 3584, rows = matrix == 2 ? 3584 : 3072;
    const unsigned char *maps = root->constants + root->palette_count * 2;
    const unsigned char *offsets = maps + root->map_count * 16;
    const unsigned char *refs = offsets + (root->template_count + 1) * 2;
    const unsigned template_id = root_u16(root->index + (expert * 3 + matrix) * 2);
    const unsigned begin = root_u16(offsets + template_id * 2);
    for (unsigned block = 0; block < rows / 64; block++) {
        const unsigned char *data = root->direct + (size_t)expert * ROOT_EXPERT_RAW +
            (matrix < 2 ? (size_t)(matrix * 48 + block) * ROOT_RAW : (size_t)96 * ROOT_RAW + (size_t)block * ROOT_DOWN_RAW);
        for (unsigned row = 0; row < 64; row++) {
            const unsigned char *codes = data + row * width / 2;
            const unsigned char *selectors = data + 64 * width / 2 + row * width / 32;
            __m256d s0 = _mm256_setzero_pd(), s1 = s0, s2 = s0, s3 = s0;
            for (unsigned coordinate = 0; coordinate < width; coordinate += 16) {
                const float (*pair)[2] = root->pairs[root_u16(refs + (begin + selectors[coordinate / 32]) * 2)];
                const unsigned char *packed = codes + coordinate / 2;
                s0 = _mm256_add_pd(s0, _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd((const double *)pair[packed[0]]), _mm_load_sd((const double *)pair[packed[1]])))));
                s1 = _mm256_add_pd(s1, _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd((const double *)pair[packed[2]]), _mm_load_sd((const double *)pair[packed[3]])))));
                s2 = _mm256_add_pd(s2, _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd((const double *)pair[packed[4]]), _mm_load_sd((const double *)pair[packed[5]])))));
                s3 = _mm256_add_pd(s3, _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd((const double *)pair[packed[6]]), _mm_load_sd((const double *)pair[packed[7]])))));
            }
            double t[4]; _mm256_storeu_pd(t, _mm256_add_pd(_mm256_add_pd(s0, s1), _mm256_add_pd(s2, s3)));
            out[0][block * 64 + row] = (float)(t[0] + t[1] + t[2] + t[3]);
        }
    }
}

/* Same decode with every step using the SAME table (always in L1): isolates the cost of table cache misses. */
static void k_decode_hot(const Root *root, unsigned expert, unsigned matrix, const double *const *conv, float *const *out)
{
    (void)conv;
    const unsigned width = matrix == 2 ? 3072 : 3584, rows = matrix == 2 ? 3584 : 3072;
    const float (*pair)[2] = root->pairs[0];
    for (unsigned block = 0; block < rows / 64; block++) {
        const unsigned char *data = root->direct + (size_t)expert * ROOT_EXPERT_RAW +
            (matrix < 2 ? (size_t)(matrix * 48 + block) * ROOT_RAW : (size_t)96 * ROOT_RAW + (size_t)block * ROOT_DOWN_RAW);
        for (unsigned row = 0; row < 64; row++) {
            const unsigned char *codes = data + row * width / 2;
            __m256d s0 = _mm256_setzero_pd(), s1 = s0, s2 = s0, s3 = s0;
            for (unsigned coordinate = 0; coordinate < width; coordinate += 16) {
                const unsigned char *packed = codes + coordinate / 2;
                s0 = _mm256_add_pd(s0, _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd((const double *)pair[packed[0]]), _mm_load_sd((const double *)pair[packed[1]])))));
                s1 = _mm256_add_pd(s1, _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd((const double *)pair[packed[2]]), _mm_load_sd((const double *)pair[packed[3]])))));
                s2 = _mm256_add_pd(s2, _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd((const double *)pair[packed[4]]), _mm_load_sd((const double *)pair[packed[5]])))));
                s3 = _mm256_add_pd(s3, _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd((const double *)pair[packed[6]]), _mm_load_sd((const double *)pair[packed[7]])))));
            }
            double t[4]; _mm256_storeu_pd(t, _mm256_add_pd(_mm256_add_pd(s0, s1), _mm256_add_pd(s2, s3)));
            out[0][block * 64 + row] = (float)(t[0] + t[1] + t[2] + t[3]);
        }
    }
}

/* Same decode, the 8 code bytes of a step read as one 64-bit word and split with shifts (1 load instead of 8). */
static void k_decode_shift(const Root *root, unsigned expert, unsigned matrix, const double *const *conv, float *const *out)
{
    (void)conv;
    const unsigned width = matrix == 2 ? 3072 : 3584, rows = matrix == 2 ? 3584 : 3072;
    const unsigned char *maps = root->constants + root->palette_count * 2;
    const unsigned char *offsets = maps + root->map_count * 16;
    const unsigned char *refs = offsets + (root->template_count + 1) * 2;
    const unsigned template_id = root_u16(root->index + (expert * 3 + matrix) * 2);
    const unsigned begin = root_u16(offsets + template_id * 2);
    for (unsigned block = 0; block < rows / 64; block++) {
        const unsigned char *data = root->direct + (size_t)expert * ROOT_EXPERT_RAW +
            (matrix < 2 ? (size_t)(matrix * 48 + block) * ROOT_RAW : (size_t)96 * ROOT_RAW + (size_t)block * ROOT_DOWN_RAW);
        for (unsigned row = 0; row < 64; row++) {
            const unsigned char *codes = data + row * width / 2;
            const unsigned char *selectors = data + 64 * width / 2 + row * width / 32;
            __m256d s0 = _mm256_setzero_pd(), s1 = s0, s2 = s0, s3 = s0;
            for (unsigned coordinate = 0; coordinate < width; coordinate += 16) {
                const float (*pair)[2] = root->pairs[root_u16(refs + (begin + selectors[coordinate / 32]) * 2)];
                uint64_t word; memcpy(&word, codes + coordinate / 2, 8);
#define P(k) ((const double *)pair[(word >> (8 * (k))) & 255])
                s0 = _mm256_add_pd(s0, _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd(P(0)), _mm_load_sd(P(1))))));
                s1 = _mm256_add_pd(s1, _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd(P(2)), _mm_load_sd(P(3))))));
                s2 = _mm256_add_pd(s2, _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd(P(4)), _mm_load_sd(P(5))))));
                s3 = _mm256_add_pd(s3, _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd(P(6)), _mm_load_sd(P(7))))));
#undef P
            }
            double t[4]; _mm256_storeu_pd(t, _mm256_add_pd(_mm256_add_pd(s0, s1), _mm256_add_pd(s2, s3)));
            out[0][block * 64 + row] = (float)(t[0] + t[1] + t[2] + t[3]);
        }
    }
}

/* ---- tiled kernel: columns in tiles of CT so the activations of a token group stay in L1; each weight tile is
   decoded once for all T tokens; every (row, token) keeps its two 8-double partial sums in memory between tiles.
   Per lane the same products are added in the same order as the shipped kernel, and the final fold is the same. ---- */
#define FPGROUP(G) \
static inline __attribute__((always_inline)) void fpg##G(const __m512d *wbuf, unsigned steps, const double *const *x, __m512d *acc) \
{ \
    __m512d a[G], b[G]; \
    _Pragma("GCC unroll 8") for (int g = 0; g < G; g++) { a[g] = acc[g * 2]; b[g] = acc[g * 2 + 1]; } \
    for (unsigned s = 0; s < steps; s++) { \
        const __m512d wa = wbuf[s * 2], wb = wbuf[s * 2 + 1]; \
        _Pragma("GCC unroll 8") for (int g = 0; g < G; g++) { \
            const double *src = x[g] + s * 16; \
            a[g] = _mm512_add_pd(a[g], _mm512_mul_pd(wa, _mm512_loadu_pd(src))); \
            b[g] = _mm512_add_pd(b[g], _mm512_mul_pd(wb, _mm512_loadu_pd(src + 8))); } \
    } \
    _Pragma("GCC unroll 8") for (int g = 0; g < G; g++) { acc[g * 2] = a[g]; acc[g * 2 + 1] = b[g]; } \
}
FPGROUP(8) FPGROUP(4) FPGROUP(2) FPGROUP(1)

enum { MAXT = 32 };
#define KTILE(NAME, CT) \
static void NAME(const Root *root, unsigned expert, unsigned matrix, unsigned T, const double *const *conv, float *const *out) \
{ \
    const unsigned width = matrix == 2 ? 3072 : 3584, rows = matrix == 2 ? 3584 : 3072; \
    const unsigned char *maps = root->constants + root->palette_count * 2; \
    const unsigned char *offsets = maps + root->map_count * 16; \
    const unsigned char *refs = offsets + (root->template_count + 1) * 2; \
    const unsigned template_id = root_u16(root->index + (expert * 3 + matrix) * 2); \
    const unsigned begin = root_u16(offsets + template_id * 2); \
    __m512d acc[64][MAXT * 2] __attribute__((aligned(64))); \
    __m512d wbuf[(CT) / 16 * 2] __attribute__((aligned(64))); \
    for (unsigned block = 0; block < rows / 64; block++) { \
        const unsigned char *data = root->direct + (size_t)expert * ROOT_EXPERT_RAW + \
            (matrix < 2 ? (size_t)(matrix * 48 + block) * ROOT_RAW : (size_t)96 * ROOT_RAW + (size_t)block * ROOT_DOWN_RAW); \
        for (unsigned row = 0; row < 64; row++) for (unsigned i = 0; i < T * 2; i++) acc[row][i] = _mm512_setzero_pd(); \
        for (unsigned c0 = 0; c0 < width; c0 += (CT)) { \
            const unsigned steps = (width - c0 < (CT) ? width - c0 : (CT)) / 16; \
            const double *xt[MAXT]; \
            for (unsigned t = 0; t < T; t++) xt[t] = conv[t] + c0; \
            for (unsigned row = 0; row < 64; row++) { \
                const unsigned char *codes = data + row * width / 2; \
                const unsigned char *selectors = data + 64 * width / 2 + row * width / 32; \
                for (unsigned s = 0; s < steps; s++) { \
                    const unsigned coordinate = c0 + s * 16; \
                    const float (*pair)[2] = root->pairs[root_u16(refs + (begin + selectors[coordinate / 32]) * 2)]; \
                    const unsigned char *packed = codes + coordinate / 2; \
                    __m256d w0 = _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd((const double *)pair[packed[0]]), _mm_load_sd((const double *)pair[packed[1]])))); \
                    __m256d w1 = _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd((const double *)pair[packed[2]]), _mm_load_sd((const double *)pair[packed[3]])))); \
                    __m256d w2 = _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd((const double *)pair[packed[4]]), _mm_load_sd((const double *)pair[packed[5]])))); \
                    __m256d w3 = _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd((const double *)pair[packed[6]]), _mm_load_sd((const double *)pair[packed[7]])))); \
                    wbuf[s * 2] = _mm512_insertf64x4(_mm512_castpd256_pd512(w0), w1, 1); \
                    wbuf[s * 2 + 1] = _mm512_insertf64x4(_mm512_castpd256_pd512(w2), w3, 1); } \
                unsigned t = 0; \
                for (; t + 8 <= T; t += 8) fpg8(wbuf, steps, xt + t, &acc[row][t * 2]); \
                if (t + 4 <= T) { fpg4(wbuf, steps, xt + t, &acc[row][t * 2]); t += 4; } \
                if (t + 2 <= T) { fpg2(wbuf, steps, xt + t, &acc[row][t * 2]); t += 2; } \
                if (t < T) fpg1(wbuf, steps, xt + t, &acc[row][t * 2]); \
            } \
        } \
        for (unsigned row = 0; row < 64; row++) for (unsigned t = 0; t < T; t++) { \
            const __m512d sum = _mm512_add_pd(acc[row][t * 2], acc[row][t * 2 + 1]); \
            double sums[4]; \
            _mm256_storeu_pd(sums, _mm256_add_pd(_mm512_castpd512_pd256(sum), _mm512_extractf64x4_pd(sum, 1))); \
            out[t][block * 64 + row] = (float)((sums[0] + sums[2]) + (sums[1] + sums[3])); } \
    } \
}
typedef void (*TiledKernel)(const Root *, unsigned, unsigned, unsigned, const double *const *, float *const *);
KTILE(kt128, 128) KTILE(kt256, 256) KTILE(kt512, 512)

/* Shipped arithmetic and shipped decode inline (so decode overlaps the multiply-adds), but the columns are cut into
   tiles of CT and the loops are token group > row block > tile > row: the 4 tokens' activation tile stays in L1
   across the 64 rows, and each row's 16 partial sums wait in memory between tiles. Same additions, same order. */
#define KTI(NAME, CT) \
static void NAME(const Root *root, unsigned expert, unsigned matrix, unsigned T, const double *const *conv, float *const *out) \
{ \
    const unsigned width = matrix == 2 ? 3072 : 3584, rows = matrix == 2 ? 3584 : 3072; \
    const unsigned char *maps = root->constants + root->palette_count * 2; \
    const unsigned char *offsets = maps + root->map_count * 16; \
    const unsigned char *refs = offsets + (root->template_count + 1) * 2; \
    const unsigned template_id = root_u16(root->index + (expert * 3 + matrix) * 2); \
    const unsigned begin = root_u16(offsets + template_id * 2); \
    __m256d accm[64][4][4] __attribute__((aligned(32))); \
    for (unsigned tg = 0; tg + 4 <= T; tg += 4) { \
        const double *const *cv = conv + tg; float *const *ov = out + tg; \
        for (unsigned block = 0; block < rows / 64; block++) { \
            const unsigned char *data = root->direct + (size_t)expert * ROOT_EXPERT_RAW + \
                (matrix < 2 ? (size_t)(matrix * 48 + block) * ROOT_RAW : (size_t)96 * ROOT_RAW + (size_t)block * ROOT_DOWN_RAW); \
            for (unsigned row = 0; row < 64; row++) for (int g = 0; g < 4; g++) for (int j = 0; j < 4; j++) accm[row][g][j] = _mm256_setzero_pd(); \
            for (unsigned c0 = 0; c0 < width; c0 += (CT)) { \
                const unsigned cend = width - c0 < (CT) ? width : c0 + (CT); \
                for (unsigned row = 0; row < 64; row++) { \
                    const unsigned char *codes = data + row * width / 2; \
                    const unsigned char *selectors = data + 64 * width / 2 + row * width / 32; \
                    __m256d l0[4], l4[4], l8[4], l12[4]; \
                    _Pragma("GCC unroll 4") for (int g = 0; g < 4; g++) { l0[g] = accm[row][g][0]; l4[g] = accm[row][g][1]; l8[g] = accm[row][g][2]; l12[g] = accm[row][g][3]; } \
                    for (unsigned coordinate = c0; coordinate < cend; coordinate += 16) { \
                        const float (*pair)[2] = root->pairs[root_u16(refs + (begin + selectors[coordinate / 32]) * 2)]; \
                        const unsigned char *packed = codes + coordinate / 2; \
                        __m256d w0 = _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd((const double *)pair[packed[0]]), _mm_load_sd((const double *)pair[packed[1]])))); \
                        __m256d w1 = _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd((const double *)pair[packed[2]]), _mm_load_sd((const double *)pair[packed[3]])))); \
                        __m256d w2 = _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd((const double *)pair[packed[4]]), _mm_load_sd((const double *)pair[packed[5]])))); \
                        __m256d w3 = _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd((const double *)pair[packed[6]]), _mm_load_sd((const double *)pair[packed[7]])))); \
                        _Pragma("GCC unroll 4") for (int g = 0; g < 4; g++) { \
                            const double *source = cv[g] + coordinate; \
                            l0[g] = _mm256_add_pd(l0[g], _mm256_mul_pd(w0, _mm256_loadu_pd(source))); \
                            l4[g] = _mm256_add_pd(l4[g], _mm256_mul_pd(w1, _mm256_loadu_pd(source + 4))); \
                            l8[g] = _mm256_add_pd(l8[g], _mm256_mul_pd(w2, _mm256_loadu_pd(source + 8))); \
                            l12[g] = _mm256_add_pd(l12[g], _mm256_mul_pd(w3, _mm256_loadu_pd(source + 12))); } \
                    } \
                    _Pragma("GCC unroll 4") for (int g = 0; g < 4; g++) { accm[row][g][0] = l0[g]; accm[row][g][1] = l4[g]; accm[row][g][2] = l8[g]; accm[row][g][3] = l12[g]; } \
                } \
            } \
            for (unsigned row = 0; row < 64; row++) for (int g = 0; g < 4; g++) { \
                double sums[4]; \
                _mm256_storeu_pd(sums, _mm256_add_pd(_mm256_add_pd(accm[row][g][0], accm[row][g][2]), _mm256_add_pd(accm[row][g][1], accm[row][g][3]))); \
                ov[g][block * 64 + row] = (float)((sums[0] + sums[2]) + (sums[1] + sums[3])); } \
        } \
    } \
}
KTI(kti256, 256) KTI(kti512, 512) KTI(kti1024, 1024)

/* The shipped kernel applied to each token on its own: the reference for any token count. */
static void shipped_each(const Root *root, unsigned expert, unsigned matrix, unsigned T, const double *const *conv, float *const *out)
{
    for (unsigned t = 0; t < T; t++) ky1(root, expert, matrix, &conv[t], &out[t]);
}

/* Same decode through one 512-bit gather: 8 byte codes -> 8 table entries of 8 bytes (two floats each) = the 16
   weights of the step, in column order; the low 8 floats widen to the [l0|l4] weights, the high 8 to [l8|l12]. */
static void k_decode_gather(const Root *root, unsigned expert, unsigned matrix, const double *const *conv, float *const *out)
{
    (void)conv;
    const unsigned width = matrix == 2 ? 3072 : 3584, rows = matrix == 2 ? 3584 : 3072;
    const unsigned char *maps = root->constants + root->palette_count * 2;
    const unsigned char *offsets = maps + root->map_count * 16;
    const unsigned char *refs = offsets + (root->template_count + 1) * 2;
    const unsigned template_id = root_u16(root->index + (expert * 3 + matrix) * 2);
    const unsigned begin = root_u16(offsets + template_id * 2);
    for (unsigned block = 0; block < rows / 64; block++) {
        const unsigned char *data = root->direct + (size_t)expert * ROOT_EXPERT_RAW +
            (matrix < 2 ? (size_t)(matrix * 48 + block) * ROOT_RAW : (size_t)96 * ROOT_RAW + (size_t)block * ROOT_DOWN_RAW);
        for (unsigned row = 0; row < 64; row++) {
            const unsigned char *codes = data + row * width / 2;
            const unsigned char *selectors = data + 64 * width / 2 + row * width / 32;
            __m512d sa = _mm512_setzero_pd(), sb = sa;
            for (unsigned coordinate = 0; coordinate < width; coordinate += 16) {
                const float (*pair)[2] = root->pairs[root_u16(refs + (begin + selectors[coordinate / 32]) * 2)];
                const __m256i index = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(codes + coordinate / 2)));
                const __m512 weights = _mm512_castpd_ps(_mm512_i32gather_pd(index, (const double *)pair, 8));
                sa = _mm512_add_pd(sa, _mm512_cvtps_pd(_mm512_castps512_ps256(weights)));
                sb = _mm512_add_pd(sb, _mm512_cvtps_pd(_mm512_extractf32x8_ps(weights, 1)));
            }
            out[0][block * 64 + row] = (float)_mm512_reduce_add_pd(_mm512_add_pd(sa, sb));
        }
    }
}

#define KFP(NAME, G, WLIM) \
static void NAME(const Root *root, unsigned expert, unsigned matrix, const double *const *conv, float *const *out) \
{ \
    (void)root; (void)expert; \
    const unsigned width = (WLIM) ? (WLIM) : (matrix == 2 ? 3072 : 3584), rows = matrix == 2 ? 3584 : 3072; \
    static const double wconst[16] = {0.5, -0.25, 0.125, 0.75, -0.5, 0.25, -0.125, 0.625, 0.375, -0.75, 0.0625, 0.875, -0.375, 0.1875, -0.625, 0.3125}; \
    const __m256d w0 = _mm256_loadu_pd(wconst), w1 = _mm256_loadu_pd(wconst + 4), w2 = _mm256_loadu_pd(wconst + 8), w3 = _mm256_loadu_pd(wconst + 12); \
    for (unsigned row = 0; row < rows; row++) { \
        __m256d l0[G], l4[G], l8[G], l12[G]; \
        _Pragma("GCC unroll 8") for (int g = 0; g < G; g++) { l0[g] = _mm256_setzero_pd(); l4[g] = l0[g]; l8[g] = l0[g]; l12[g] = l0[g]; } \
        for (unsigned coordinate = 0; coordinate < width; coordinate += 16) { \
            _Pragma("GCC unroll 8") for (int g = 0; g < G; g++) { \
                const double *source = conv[g] + coordinate; \
                l0[g] = _mm256_add_pd(l0[g], _mm256_mul_pd(w0, _mm256_loadu_pd(source))); \
                l4[g] = _mm256_add_pd(l4[g], _mm256_mul_pd(w1, _mm256_loadu_pd(source + 4))); \
                l8[g] = _mm256_add_pd(l8[g], _mm256_mul_pd(w2, _mm256_loadu_pd(source + 8))); \
                l12[g] = _mm256_add_pd(l12[g], _mm256_mul_pd(w3, _mm256_loadu_pd(source + 12))); } \
        } \
        _Pragma("GCC unroll 8") for (int g = 0; g < G; g++) { \
            double sums[4]; \
            _mm256_storeu_pd(sums, _mm256_add_pd(_mm256_add_pd(l0[g], l8[g]), _mm256_add_pd(l4[g], l12[g]))); \
            out[g][row] = (float)((sums[0] + sums[2]) + (sums[1] + sums[3])); } \
    } \
}
KFP(kfp1, 1, 0) KFP(kfp2, 2, 0) KFP(kfp4, 4, 0) KFP(kfp8, 8, 0)
KFP(kfp4_w512, 4, 512) KFP(kfp4_w1024, 4, 1024) KFP(kfp4_w2048, 4, 2048)

/* ---- double multiply-then-add peak ---- */
static double peak(int zmm, int threads, unsigned long iters)
{
    double start = clk(), sink = 0;
#pragma omp parallel num_threads(threads) reduction(+ : sink)
    {
        if (zmm) {
            __m512d a[12], b = _mm512_set1_pd(1.0000001), c = _mm512_set1_pd(1e-7);
            for (int k = 0; k < 12; k++) a[k] = _mm512_set1_pd((double)k);
            for (unsigned long i = 0; i < iters; i++)
                _Pragma("GCC unroll 12") for (int k = 0; k < 12; k++) a[k] = _mm512_add_pd(a[k], _mm512_mul_pd(b, c));
            for (int k = 0; k < 12; k++) sink += _mm512_reduce_add_pd(a[k]);
        } else {
            __m256d a[12], b = _mm256_set1_pd(1.0000001), c = _mm256_set1_pd(1e-7);
            for (int k = 0; k < 12; k++) a[k] = _mm256_set1_pd((double)k);
            for (unsigned long i = 0; i < iters; i++)
                _Pragma("GCC unroll 12") for (int k = 0; k < 12; k++) a[k] = _mm256_add_pd(a[k], _mm256_mul_pd(b, c));
            double t[4];
            for (int k = 0; k < 12; k++) { _mm256_storeu_pd(t, a[k]); sink += t[0]; }
        }
    }
    double sec = clk() - start;
    if (sink == 12345.678) puts("");
    return (double)threads * iters * 12 * (zmm ? 8 : 4) / sec / 1e9;
}

int main(int argc, char **argv)
{
    if (argc < 2) { fputs("usage: expert-kernel-bench DATASET_DIR [REPS]\n", stderr); return 2; }
    const int reps = argc > 2 ? atoi(argv[2]) : 6;
    if (getenv("EK_THREADS")) nth = atoi(getenv("EK_THREADS"));
    if (nth < 1 || nth > MAXTH) bail("EK_THREADS must be 1..32");
    printf("raw double multiply-then-add peak (GMAC/s): 1 thread 256-bit %.1f, 512-bit %.1f; %d threads 256-bit %.1f, 512-bit %.1f\n",
        peak(0, 1, 100000000UL), peak(1, 1, 100000000UL), THREADS, peak(0, THREADS, 100000000UL), peak(1, THREADS, 100000000UL));

    Transformer *tr = NULL;
    if (!transformer_open(argv[1], &tr)) bail("transformer_open failed");
    const Root *root = tr->root;
    if (!root->direct) bail("no direct expert mapping");
    printf("pair tables: %u (2 KB each = %.1f MB); templates %u; references %u\n", root->map_count,
        (double)root->map_count * 2048 / 1e6, root->template_count, root->reference_count);

    /* Experts 0..15, one per thread; touch them so they are in memory before timing. */
    unsigned long touch = 0;
    for (unsigned e = 0; e < THREADS; e++)
        for (size_t i = 0; i < ROOT_EXPERT_RAW; i += 4096) touch += root->direct[(size_t)e * ROOT_EXPERT_RAW + i];
    if (touch == 1) puts("");

    static double conv[MAXTH][MAXG][LATW];
    uint64_t s = 88172645463325252ULL;
    for (int t = 0; t < THREADS; t++) for (int g = 0; g < MAXG; g++) for (unsigned i = 0; i < LATW; i++) {
        s ^= s << 13; s ^= s >> 7; s ^= s << 17;
        conv[t][g][i] = ((double)(s >> 11) / 9007199254740992.0 - 0.5) * 2.0;
    }
    static float outs[MAXTH][MAXG][3584], ref[MAXG][3584];

    for (int matrix = 0; matrix < 3; matrix += 2) {
        const unsigned width = matrix == 2 ? 3072 : 3584, rows = matrix == 2 ? 3584 : 3072;
        printf("\nexpert matrix %d: %u rows x %u columns (%.1f MB of codes); %d repeats\n", matrix, rows, width, (double)rows * width / 2 / 1e6, reps);
        printf("  %-36s %22s %26s %s\n", "variant", "1 thread, resident", "16 threads, 16 experts", "bits vs shipped");
        for (int v = 0; v < NVAR; v++) {
            const int G = variants[v].G;
            const double *cin[MAXG]; float *oout[MAXG];
            for (int g = 0; g < G; g++) { cin[g] = conv[0][g]; oout[g] = outs[0][g]; }
            variants[v].run(root, 0, matrix, cin, oout);                       /* warm */
            double best1 = 1e9;
            for (int r = 0; r < reps; r++) { double t = clk(); variants[v].run(root, 0, matrix, cin, oout); t = clk() - t; if (t < best1) best1 = t; }
            /* bits: compare with the shipped kernel at the same G on the same inputs */
            int same = 1;
            if (v == variants[v].base) { for (int g = 0; g < G; g++) memcpy(ref[g], outs[0][g], rows * sizeof(float)); }
            else {
                float tmp[MAXG][3584]; float *tp[MAXG]; for (int g = 0; g < G; g++) tp[g] = tmp[g];
                variants[variants[v].base].run(root, 0, matrix, cin, tp);
                for (int g = 0; g < G; g++) if (memcmp(tmp[g], outs[0][g], rows * sizeof(float))) same = 0;
            }
            double bestN = 1e9;
            for (int r = -1; r < reps; r++) {
                double t = clk();
#pragma omp parallel num_threads(THREADS)
                {
                    const int th = omp_get_thread_num();
                    const double *ci[MAXG]; float *oo[MAXG];
                    for (int g = 0; g < G; g++) { ci[g] = conv[th][g]; oo[g] = outs[th][g]; }
                    variants[v].run(root, (unsigned)th, (unsigned)matrix, ci, oo);
                }
                t = clk() - t;
                if (r >= 0 && t < bestN) bestN = t;
            }
            const double macs = (double)rows * width * G;
            printf("  %-36s %10.2f GMAC/s %9s %9.1f GMAC/s %10s   %s\n", variants[v].name, macs / best1 / 1e9, "",
                macs * THREADS / bestN / 1e9, "", same ? "identical" : "DIFFERENT");
            fflush(stdout);
        }
    }
    {   /* where one 16-column step goes, single thread, matrix 0 */
        const unsigned width = 3584, rows = 3072;
        const double steps = (double)rows * width / 16;
        struct { const char *name; Kernel run; int G; } split[] = {
            {"decode only (no multiplies)", k_decode_only, 1}, {"decode only, one table for every step", k_decode_hot, 1},
            {"decode only, code bytes by shifts", k_decode_shift, 1},
            {"decode only, one 512-bit gather per step", k_decode_gather, 1},
            {"multiply-adds only, 1 token", kfp1, 1},
            {"multiply-adds only, 2 tokens", kfp2, 2}, {"multiply-adds only, 4 tokens", kfp4, 4},
            {"multiply-adds only, 8 tokens", kfp8, 8}, {"shipped, 4 tokens (decode + multiply-adds)", ky4, 4}};
        printf("\ncost split, 1 thread, matrix 0 (time per 16-column step)\n");
        for (unsigned i = 0; i < sizeof split / sizeof *split; i++) {
            const double *cin[MAXG]; float *oout[MAXG];
            for (int g = 0; g < split[i].G; g++) { cin[g] = conv[0][g]; oout[g] = outs[0][g]; }
            split[i].run(root, 0, 0, cin, oout);
            double best = 1e9;
            for (int r = 0; r < reps; r++) { double t = clk(); split[i].run(root, 0, 0, cin, oout); t = clk() - t; if (t < best) best = t; }
            printf("  %-44s %6.2f ns per step  (%5.2f ns per token-step)\n", split[i].name, best * 1e9 / steps, best * 1e9 / steps / split[i].G);
        }
    }
    {   /* same multiply-adds, 4 tokens, but with the activations shrunk until they fit in the 32 KB L1 */
        struct { const char *name; Kernel run; unsigned width; } small[] = {
            {"multiply-adds only, 4 tokens, 512 columns (16 KB of activations)", kfp4_w512, 512},
            {"multiply-adds only, 4 tokens, 1024 columns (32 KB)", kfp4_w1024, 1024},
            {"multiply-adds only, 4 tokens, 2048 columns (64 KB)", kfp4_w2048, 2048},
            {"multiply-adds only, 4 tokens, 3584 columns (115 KB)", kfp4, 3584}};
        printf("\nactivation footprint test, 1 thread, 4 tokens, 3072 rows\n");
        for (unsigned i = 0; i < sizeof small / sizeof *small; i++) {
            const double *cin[MAXG]; float *oout[MAXG];
            for (int g = 0; g < 4; g++) { cin[g] = conv[0][g]; oout[g] = outs[0][g]; }
            small[i].run(root, 0, 0, cin, oout);
            double best = 1e9;
            for (int r = 0; r < reps; r++) { double t = clk(); small[i].run(root, 0, 0, cin, oout); t = clk() - t; if (t < best) best = t; }
            const double token_steps = 3072.0 * small[i].width / 16 * 4;
            printf("  %-66s %5.2f ns per token-step\n", small[i].name, best * 1e9 / token_steps);
        }
    }
    {   /* the tiled kernel against the shipped kernel (4 tokens per pass), many tokens on one expert */
        static double tok[MAXTH][MAXT][LATW];
        static float tout[MAXTH][MAXT][3584], tref[MAXT][3584];
        for (int t = 0; t < THREADS; t++) for (int g = 0; g < MAXT; g++) for (unsigned i = 0; i < LATW; i++) {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            tok[t][g][i] = ((double)(s >> 11) / 9007199254740992.0 - 0.5) * 2.0;
        }
        static const struct { const char *name; TiledKernel run; } tk[] = {
            {"row blocks, 256-col tiles, inline", kti256}, {"row blocks, 512-col tiles, inline", kti512},
            {"row blocks, 1024-col tiles, inline", kti1024}, {"zmm decode-buffer tiles, 256 cols", kt256},
            {"shipped, 4 tokens per pass", NULL}};
        static const unsigned counts[] = {4, 8, 16, 32};
        for (int matrix = 0; matrix < 3; matrix += 2) {
            const unsigned width = matrix == 2 ? 3072 : 3584, rows = matrix == 2 ? 3584 : 3072;
            printf("\ntiled kernel, expert matrix %d (%u rows x %u columns), T tokens on one expert\n", matrix, rows, width);
            printf("  %-30s %3s %24s %24s %s\n", "kernel", "T", "1 thread, resident", "16 threads, 16 experts", "bits vs shipped");
            for (unsigned c = 0; c < sizeof counts / sizeof *counts; c++) {
                const unsigned T = counts[c];
                const double *cin[MAXT]; float *oo[MAXT], *rr[MAXT];
                for (unsigned t = 0; t < T; t++) { cin[t] = tok[0][t]; oo[t] = tout[0][t]; rr[t] = tref[t]; }
                shipped_each(root, 0, (unsigned)matrix, T, cin, rr);
                for (unsigned k = 0; k < sizeof tk / sizeof *tk; k++) {
                    #define RUN_ONE(root_, expert_, ci_, oo_) do { \
                        if (tk[k].run) tk[k].run(root_, expert_, (unsigned)matrix, T, ci_, oo_); \
                        else for (unsigned t_ = 0; t_ < T; t_ += 4) ky4(root_, expert_, (unsigned)matrix, (ci_) + t_, (oo_) + t_); } while (0)
                    RUN_ONE(root, 0, cin, oo);
                    int same = 1;
                    for (unsigned t = 0; t < T; t++) if (memcmp(tref[t], tout[0][t], rows * sizeof(float))) same = 0;
                    double best1 = 1e9;
                    for (int r = 0; r < reps; r++) { double t0 = clk(); RUN_ONE(root, 0, cin, oo); t0 = clk() - t0; if (t0 < best1) best1 = t0; }
                    double bestN = 1e9;
                    for (int r = -1; r < reps; r++) {
                        double t0 = clk();
#pragma omp parallel num_threads(THREADS)
                        {
                            const int th = omp_get_thread_num();
                            const double *ci[MAXT]; float *o2[MAXT];
                            for (unsigned t = 0; t < T; t++) { ci[t] = tok[th][t]; o2[t] = tout[th][t]; }
                            RUN_ONE(root, (unsigned)th, ci, o2);
                        }
                        t0 = clk() - t0;
                        if (r >= 0 && t0 < bestN) bestN = t0;
                    }
                    #undef RUN_ONE
                    const double macs = (double)rows * width * T;
                    printf("  %-30s %3u %8.3f ns/token-step %6.1f GMAC/s %8.1f GMAC/s   %s\n", tk[k].name, T,
                        best1 * 1e9 / ((double)T * rows * width / 16), macs / best1 / 1e9, macs * THREADS / bestN / 1e9, same ? "identical" : "DIFFERENT");
                    fflush(stdout);
                }
            }
        }
    }
    return 0;
}
