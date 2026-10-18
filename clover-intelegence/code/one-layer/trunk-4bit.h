/* The trunk matrices packed to 4 bits (USE_4BIT_TRUNK=1). NOT bit-identical to the 8-bit trunk.
   Per group of 64 columns: one integer step (smallest that fits the group's largest code into -8..7) and 64 nibbles, stored planar:
   the low nibbles of a group's 32 bytes are columns 0-31, the high nibbles columns 32-63. Weight = (nibble - 8) * step; the row's float scale is unchanged.
   Needs AVX-512; the 8-bit path stays for other CPUs. Included by layer-batch.c and layer0-batch.c. */
#ifndef TRUNK_4BIT_H
#define TRUNK_4BIT_H

#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { TRUNK_4BIT_GROUP = 64, TRUNK_4BIT_TOKEN_TILE = 32, TRUNK_4BIT_SLOTS = 96 };

/* The packed copy of one layer's trunk matrices, indexed by a slot number the layer file chooses per matrix. */
typedef struct { const unsigned char *nibbles[TRUNK_4BIT_SLOTS], *steps[TRUNK_4BIT_SLOTS]; } PackedTrunk;

static float trunk_4bit_read_float(const unsigned char *bytes)
{
    float value;
    memcpy(&value, bytes, sizeof value);
    return value;
}

/* codes: 8-bit weights, row r starting at codes + r * code_stride. */
static void pack_matrix_to_4bit(const unsigned char *codes, size_t code_stride, unsigned rows, unsigned cols,
                                const unsigned char **nibbles_out, const unsigned char **steps_out)
{
    const unsigned groups = cols / TRUNK_4BIT_GROUP;
    unsigned char *nibbles = aligned_alloc(64, ((size_t)rows * cols / 2 + 63) / 64 * 64);
    unsigned char *steps = malloc((size_t)rows * groups);
    if (!nibbles || !steps) { fputs("pack_matrix_to_4bit: allocation failed\n", stderr); exit(1); }
#pragma omp parallel for schedule(static)
    for (unsigned row = 0; row < rows; row++)
        for (unsigned group = 0; group < groups; group++) {
            const int8_t *code = (const int8_t *)(codes + (size_t)row * code_stride + (size_t)group * TRUNK_4BIT_GROUP);
            int largest = 0;
            for (int i = 0; i < TRUNK_4BIT_GROUP; i++) { const int magnitude = code[i] < 0 ? -code[i] : code[i]; if (magnitude > largest) largest = magnitude; }
            const int step = largest ? (largest + 6) / 7 : 1;
            steps[(size_t)row * groups + group] = (unsigned char)step;
            unsigned char *packed = nibbles + ((size_t)row * groups + group) * 32;
            for (int i = 0; i < 32; i++) {
                int low = (int)lrintf((float)code[i] / (float)step), high = (int)lrintf((float)code[i + 32] / (float)step);
                low = low < -8 ? -8 : low > 7 ? 7 : low;
                high = high < -8 ? -8 : high > 7 ? 7 : high;
                packed[i] = (unsigned char)((low + 8) | ((high + 8) << 4));
            }
        }
    *nibbles_out = nibbles; *steps_out = steps;
}

#if defined(__AVX512F__)
/* Four sixteen-column slices of one group, decoded to floats already multiplied by the group step. */
#define DECODE_GROUP_4BIT(nibbles, group, step, weights) \
    do { \
        const __m256i bytes = _mm256_loadu_si256((const __m256i *)((nibbles) + (size_t)(group) * 32)); \
        const __m256i low = _mm256_and_si256(bytes, _mm256_set1_epi8(0x0F)); \
        const __m256i high = _mm256_and_si256(_mm256_srli_epi16(bytes, 4), _mm256_set1_epi8(0x0F)); \
        const __m128i slice[4] = { _mm256_castsi256_si128(low), _mm256_extracti128_si256(low, 1), \
                                   _mm256_castsi256_si128(high), _mm256_extracti128_si256(high, 1) }; \
        for (int s = 0; s < 4; s++) \
            (weights)[s] = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_cvtepu8_epi32(slice[s]), _mm512_set1_epi32(8))), (step)); \
    } while (0)

/* `tokens` (1 or 4) activation vectors against one weight row. */
#define PROJECT_ROW_4BIT(tokens) \
    do { \
        __m512 sum[tokens]; \
        const float *activation[tokens]; \
        for (int t = 0; t < (tokens); t++) { sum[t] = _mm512_setzero_ps(); activation[t] = X + (size_t)(first + k + t) * xs; } \
        for (unsigned group = 0; group < groups; group++) { \
            __m512 weights[4]; \
            DECODE_GROUP_4BIT(row_nibbles, group, _mm512_set1_ps((float)row_steps[group]), weights); \
            for (int s = 0; s < 4; s++) \
                for (int t = 0; t < (tokens); t++) \
                    sum[t] = _mm512_fmadd_ps(weights[s], _mm512_loadu_ps(activation[t] + (size_t)group * TRUNK_4BIT_GROUP + s * 16), sum[t]); \
        } \
        for (int t = 0; t < (tokens); t++) Y[(size_t)(first + k + t) * ys + row] = _mm512_reduce_add_ps(sum[t]) * row_scale; \
    } while (0)

/* Y[t][row] = row scale * sum over columns of X[t][column] * weight, for tokens t in [0, n), token strides xs / ys floats.
   Tokens go through in tiles so a tile's activations stay in cache while the weight rows stream past once. */
static void project_with_4bit_weights(const unsigned char *nibbles, const unsigned char *steps, const unsigned char *row_scales, size_t row_scale_stride,
                                      unsigned rows, unsigned cols, const float *X, size_t xs, float *Y, size_t ys, unsigned n)
{
    const unsigned groups = cols / TRUNK_4BIT_GROUP;
    for (unsigned first = 0; first < n; first += TRUNK_4BIT_TOKEN_TILE) {
        const unsigned tile = n - first < TRUNK_4BIT_TOKEN_TILE ? n - first : TRUNK_4BIT_TOKEN_TILE;
#pragma omp parallel for schedule(static)
        for (unsigned row = 0; row < rows; row++) {
            const unsigned char *row_nibbles = nibbles + (size_t)row * cols / 2, *row_steps = steps + (size_t)row * groups;
            const float row_scale = trunk_4bit_read_float(row_scales + (size_t)row * row_scale_stride);
            unsigned k = 0;
            for (; k + 4 <= tile; k += 4) PROJECT_ROW_4BIT(4);
            for (; k < tile; k++) PROJECT_ROW_4BIT(1);
        }
    }
}
#endif

#endif
