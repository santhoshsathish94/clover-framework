#define _GNU_SOURCE
/* Why did the lookup win in the pasted test? Same shape as that test: 12,288 x 7,168 weights (88,080,384), a 16-value weight codebook,
   an 8-bit activation vector, a 16 x 256 table of products. Four ways to compute the same integer dot products, one thread:
     A  direct multiply, scalar (auto-vectorisation off)       -- the "direct multiplication" of the test
     B  table lookup, scalar (auto-vectorisation off)          -- the "lookup products" of the test
     C  direct multiply, compiler-vectorised
     D  direct multiply, written with AVX-512: the 16 codebook values looked up in a register (vpshufb), widened to int16,
        multiplied and added in pairs (vpmaddwd), 32 weights per step
   All results are integers, so they must be exactly equal.

     granular-bench [REPS]                                                                    */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <immintrin.h>

enum { ROWS = 12288, COLS = 7168 };
static double clk(void) { struct timespec m; clock_gettime(CLOCK_MONOTONIC, &m); return (double)m.tv_sec + (double)m.tv_nsec / 1e9; }

static const int8_t codebook[16] = { 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };   /* FP4-like, scaled to integers */
static int16_t table[16][256];

__attribute__((noinline, optimize("no-tree-vectorize")))
static void kernel_a(const uint8_t *w, const int8_t *act, int32_t *out)
{
    for (unsigned r = 0; r < ROWS; r++) {
        int32_t sum = 0; const uint8_t *row = w + (size_t)r * COLS;
        for (unsigned c = 0; c < COLS; c++) sum += (int32_t)codebook[row[c]] * (int32_t)act[c];
        out[r] = sum;
    }
}
__attribute__((noinline, optimize("no-tree-vectorize")))
static void kernel_b(const uint8_t *w, const uint8_t *act_code, int32_t *out)
{
    for (unsigned r = 0; r < ROWS; r++) {
        int32_t sum = 0; const uint8_t *row = w + (size_t)r * COLS;
        for (unsigned c = 0; c < COLS; c++) sum += table[row[c]][act_code[c]];
        out[r] = sum;
    }
}
__attribute__((noinline))
static void kernel_c(const uint8_t *w, const int8_t *act, int32_t *out)
{
    for (unsigned r = 0; r < ROWS; r++) {
        int32_t sum = 0; const uint8_t *row = w + (size_t)r * COLS;
        for (unsigned c = 0; c < COLS; c++) sum += (int32_t)codebook[row[c]] * (int32_t)act[c];
        out[r] = sum;
    }
}
__attribute__((noinline))
static void kernel_d(const uint8_t *w, const int8_t *act, int32_t *out)
{
    const __m256i cb = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)codebook));
    for (unsigned r = 0; r < ROWS; r++) {
        __m512i acc = _mm512_setzero_si512(); const uint8_t *row = w + (size_t)r * COLS;
        unsigned c = 0;
        for (; c + 32 <= COLS; c += 32) {
            const __m256i wv = _mm256_shuffle_epi8(cb, _mm256_loadu_si256((const __m256i *)(row + c)));
            acc = _mm512_add_epi32(acc, _mm512_madd_epi16(_mm512_cvtepi8_epi16(wv), _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i *)(act + c)))));
        }
        int32_t sum = _mm512_reduce_add_epi32(acc);
        for (; c < COLS; c++) sum += (int32_t)codebook[row[c]] * (int32_t)act[c];
        out[r] = sum;
    }
}

int main(int argc, char **argv)
{
    const int reps = argc > 1 ? atoi(argv[1]) : 3;
    uint8_t *w = aligned_alloc(64, (size_t)ROWS * COLS); int8_t *act = aligned_alloc(64, COLS); uint8_t *act_code = aligned_alloc(64, COLS);
    uint64_t s = 88172645463325252ULL;
    for (size_t i = 0; i < (size_t)ROWS * COLS; i++) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; w[i] = (uint8_t)(s & 15); }
    for (unsigned c = 0; c < COLS; c++) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; act[c] = (int8_t)(s >> 56); act_code[c] = (uint8_t)act[c]; }
    for (int k = 0; k < 16; k++) for (int a = 0; a < 256; a++) table[k][a] = (int16_t)((int)codebook[k] * (int)(int8_t)a);
    int32_t *o[4]; for (int i = 0; i < 4; i++) o[i] = aligned_alloc(64, ROWS * 4);
    double best[4] = { 1e9, 1e9, 1e9, 1e9 };
    for (int r = -1; r < reps; r++) {
        double t = clk(); kernel_a(w, act, o[0]); t = clk() - t; if (r >= 0 && t < best[0]) best[0] = t;
        t = clk(); kernel_b(w, act_code, o[1]); t = clk() - t; if (r >= 0 && t < best[1]) best[1] = t;
        t = clk(); kernel_c(w, act, o[2]); t = clk() - t; if (r >= 0 && t < best[2]) best[2] = t;
        t = clk(); kernel_d(w, act, o[3]); t = clk() - t; if (r >= 0 && t < best[3]) best[3] = t;
    }
    int same = 1; for (int i = 1; i < 4; i++) if (memcmp(o[0], o[i], ROWS * 4)) same = 0;
    const double macs = (double)ROWS * COLS;
    const char *names[4] = { "A direct multiply, scalar", "B table lookup, scalar", "C direct multiply, compiler-vectorised", "D direct multiply, AVX-512 (shuffle lookup of the codebook + pairwise int16 multiply-add)" };
    printf("%u x %u = %.0f weights, one byte per weight, one thread, best of %d\n", ROWS, COLS, macs, reps);
    for (int i = 0; i < 4; i++) printf("  %-88s %9.3f ms  %7.2f GMAC/s\n", names[i], best[i] * 1e3, macs / best[i] / 1e9);
    printf("  lookup (B) against direct (A): %.2fx faster; against the vectorised kernel D: %.1fx slower; integer results identical: %s\n",
        best[0] / best[1], best[1] / best[3], same ? "yes" : "NO");
    return 0;
}
