#define _GNU_SOURCE
/* The two projection kernels side by side, same harness, same MAC count, swept from
   L2-resident to far past L3.

   Derived figures put the trunk path near 37 GMAC/s and the expert path near 70, but
   those came from dividing stage times by my own MAC counts, and a separate run of the
   trunk kernel alone gave 23.8. Three numbers from three methods is not a comparison.

   A: the trunk, transformer_project_row. One byte of palette index per weight, sixteen
      table lookups per sixteen coordinates, scalar fmaf into sixteen lanes.
   B: the expert, root_project's inner loop. Four bits per weight so one byte carries
      two, eight lookups per sixteen coordinates, AVX2 doubles.

   Same rows and width in both arms, so the MAC count is identical and only the format
   and the instruction mix differ. If the gap is bytes rather than code, A reads twice
   what B reads and the GB/s will agree where the GMAC/s does not. */
#include <immintrin.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { TABLES = 8 };

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static float palette[256];
static float pairs[TABLES][256][2];

/* A: one byte per weight. */
static float trunk_row(const float *input, const unsigned char *ids, unsigned width, float scale)
{
    float lanes[16] = {0}, combined[8];
    for (unsigned coordinate = 0; coordinate < width; coordinate += 16)
        for (unsigned lane = 0; lane < 16; lane++)
            lanes[lane] = fmaf(palette[ids[coordinate + lane]], input[coordinate + lane], lanes[lane]);
    for (unsigned lane = 0; lane < 8; lane++) combined[lane] = lanes[lane] + lanes[lane + 8];
    float s0 = combined[0] + combined[4], s1 = combined[1] + combined[5];
    float s2 = combined[2] + combined[6], s3 = combined[3] + combined[7];
    return ((s0 + s2) + (s1 + s3)) * scale;
}

/* A2: the same arithmetic as A, vectorised. Lane j of acc0 carries the same terms in
   the same order as lanes[j] above, and acc1 the same as lanes[j+8], so if fmaf and
   _mm256_fmadd_ps round alike the checksum must match A exactly. */
static float trunk_row_gather(const float *input, const unsigned char *ids, unsigned width, float scale)
{
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    for (unsigned coordinate = 0; coordinate < width; coordinate += 16) {
        __m256i index0 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(ids + coordinate)));
        __m256i index1 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(ids + coordinate + 8)));
        acc0 = _mm256_fmadd_ps(_mm256_i32gather_ps(palette, index0, 4),
            _mm256_loadu_ps(input + coordinate), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_i32gather_ps(palette, index1, 4),
            _mm256_loadu_ps(input + coordinate + 8), acc1);
    }
    float lanes[16], combined[8];
    _mm256_storeu_ps(lanes, acc0);
    _mm256_storeu_ps(lanes + 8, acc1);
    for (unsigned lane = 0; lane < 8; lane++) combined[lane] = lanes[lane] + lanes[lane + 8];
    float s0 = combined[0] + combined[4], s1 = combined[1] + combined[5];
    float s2 = combined[2] + combined[6], s3 = combined[3] + combined[7];
    return ((s0 + s2) + (s1 + s3)) * scale;
}

/* A3: sixteen indices a time. One 512-bit accumulator maps lane j to element c+j,
   which is exactly what lanes[j] held in the scalar form. */
#if defined(__AVX512F__)
static float trunk_row_gather512(const float *input, const unsigned char *ids, unsigned width, float scale)
{
    __m512 acc = _mm512_setzero_ps();
    for (unsigned coordinate = 0; coordinate < width; coordinate += 16) {
        __m512i index = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(ids + coordinate)));
        acc = _mm512_fmadd_ps(_mm512_i32gather_ps(index, palette, 4),
            _mm512_loadu_ps(input + coordinate), acc);
    }
    float lanes[16], combined[8];
    _mm512_storeu_ps(lanes, acc);
    for (unsigned lane = 0; lane < 8; lane++) combined[lane] = lanes[lane] + lanes[lane + 8];
    float s0 = combined[0] + combined[4], s1 = combined[1] + combined[5];
    float s2 = combined[2] + combined[6], s3 = combined[3] + combined[7];
    return ((s0 + s2) + (s1 + s3)) * scale;
}
#endif

/* B: four bits per weight, one byte carries two. */
static double expert_row(const double *converted, const unsigned char *codes,
    const unsigned char *selectors, unsigned width)
{
    double sums[4];
    __m256d lane0 = _mm256_setzero_pd(), lane4 = _mm256_setzero_pd();
    __m256d lane8 = _mm256_setzero_pd(), lane12 = _mm256_setzero_pd();
    for (unsigned coordinate = 0; coordinate < width; coordinate += 16) {
        const float (*pair)[2] = pairs[selectors[coordinate / 32] & (TABLES - 1)];
        const unsigned char *packed = codes + coordinate / 2;
        const double *source = converted + coordinate;
        __m256d w0 = _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd((const double *)pair[packed[0]]), _mm_load_sd((const double *)pair[packed[1]]))));
        __m256d w1 = _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd((const double *)pair[packed[2]]), _mm_load_sd((const double *)pair[packed[3]]))));
        __m256d w2 = _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd((const double *)pair[packed[4]]), _mm_load_sd((const double *)pair[packed[5]]))));
        __m256d w3 = _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd((const double *)pair[packed[6]]), _mm_load_sd((const double *)pair[packed[7]]))));
        lane0 = _mm256_add_pd(lane0, _mm256_mul_pd(w0, _mm256_loadu_pd(source)));
        lane4 = _mm256_add_pd(lane4, _mm256_mul_pd(w1, _mm256_loadu_pd(source + 4)));
        lane8 = _mm256_add_pd(lane8, _mm256_mul_pd(w2, _mm256_loadu_pd(source + 8)));
        lane12 = _mm256_add_pd(lane12, _mm256_mul_pd(w3, _mm256_loadu_pd(source + 12)));
    }
    _mm256_storeu_pd(sums, _mm256_add_pd(_mm256_add_pd(lane0, lane8), _mm256_add_pd(lane4, lane12)));
    return (sums[0] + sums[2]) + (sums[1] + sums[3]);
}

int main(int argc, char **argv)
{
    const unsigned width = argc > 1 ? (unsigned)atoi(argv[1]) : 3584;
    const int repeats = argc > 2 ? atoi(argv[2]) : 3;
    const unsigned sizes[] = {256, 4096, 16384, 65536};

    uint64_t s = 2463534242ULL;
    for (unsigned i = 0; i < 256; i++) palette[i] = (float)((double)i / 128.0 - 1.0);
    for (unsigned t = 0; t < TABLES; t++)
        for (unsigned i = 0; i < 256; i++) {
            pairs[t][i][0] = (float)((double)(i & 15) / 8.0 - 1.0);
            pairs[t][i][1] = (float)((double)(i >> 4) / 8.0 - 1.0);
        }

    float *input = malloc(width * sizeof(float));
    double *converted = malloc(width * sizeof(double));
    for (unsigned i = 0; i < width; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        input[i] = (float)((double)(s % 2000) / 1000.0 - 1.0);
        converted[i] = (double)input[i];
    }

    printf("width %u, %d threads, L3 is 128 MiB\n", width, omp_get_max_threads());
    printf("%8s %9s | %9s | %9s | %9s | %9s\n",
        "rows", "MMAC", "A scalar", "A2 g256", "A3 g512", "B 4-bit");

    for (unsigned k = 0; k < sizeof sizes / sizeof *sizes; k++) {
        const unsigned rows = sizes[k];
        const size_t a_bytes = (size_t)rows * width;
        unsigned char *ids = malloc(a_bytes);
        unsigned char *codes = malloc((size_t)rows * width / 2);
        unsigned char *selectors = malloc((size_t)rows * width / 32);
        float *out_a = malloc(rows * sizeof(float));
        float *out_a2 = malloc(rows * sizeof(float));
        float *out_a3 = calloc(rows, sizeof(float));
        double *out_b = malloc(rows * sizeof(double));
        if (!ids || !codes || !selectors || !out_a || !out_a2 || !out_a3 || !out_b) { fputs("alloc failed\n", stderr); return 1; }
#pragma omp parallel for schedule(static)
        for (long long r = 0; r < (long long)rows; r++) {
            uint64_t l = 88172645463325252ULL + (uint64_t)r * 2654435761ULL;
            for (unsigned c = 0; c < width; c++) { l ^= l << 13; l ^= l >> 7; l ^= l << 17; ids[(size_t)r * width + c] = (unsigned char)l; }
            for (unsigned c = 0; c < width / 2; c++) { l ^= l << 13; l ^= l >> 7; l ^= l << 17; codes[(size_t)r * (width / 2) + c] = (unsigned char)l; }
            for (unsigned c = 0; c < width / 32; c++) selectors[(size_t)r * (width / 32) + c] = (unsigned char)(c & (TABLES - 1));
        }

        double best_a = 1e18, best_a2 = 1e18, best_a3 = 1e18, best_b = 1e18;
        double sum_a = 0, sum_a2 = 0, sum_a3 = 0, sum_b = 0;
        for (int rep = 0; rep < repeats; rep++) {
            double t0 = now();
#pragma omp parallel for schedule(static)
            for (long long r = 0; r < (long long)rows; r++)
                out_a[r] = trunk_row(input, ids + (size_t)r * width, width, 0.01f);
            double t1 = now() - t0;
            if (t1 < best_a) best_a = t1;

            t0 = now();
#pragma omp parallel for schedule(static)
            for (long long r = 0; r < (long long)rows; r++)
                out_a2[r] = trunk_row_gather(input, ids + (size_t)r * width, width, 0.01f);
            t1 = now() - t0;
            if (t1 < best_a2) best_a2 = t1;

#if defined(__AVX512F__)
            t0 = now();
#pragma omp parallel for schedule(static)
            for (long long r = 0; r < (long long)rows; r++)
                out_a3[r] = trunk_row_gather512(input, ids + (size_t)r * width, width, 0.01f);
            t1 = now() - t0;
            if (t1 < best_a3) best_a3 = t1;
#endif
            t0 = now();
#pragma omp parallel for schedule(static)
            for (long long r = 0; r < (long long)rows; r++)
                out_b[r] = expert_row(converted, codes + (size_t)r * (width / 2),
                    selectors + (size_t)r * (width / 32), width);
            t1 = now() - t0;
            if (t1 < best_b) best_b = t1;
        }
        unsigned mismatch2 = 0, mismatch3 = 0;
        for (unsigned r = 0; r < rows; r++) {
            if (memcmp(&out_a[r], &out_a2[r], sizeof(float))) mismatch2++;
            if (memcmp(&out_a[r], &out_a3[r], sizeof(float))) mismatch3++;
            sum_a += out_a[r]; sum_a2 += out_a2[r]; sum_a3 += out_a3[r]; sum_b += out_b[r];
        }
        double macs = (double)rows * width;
        printf("%8u %9.1f | %9.2f | %9.2f | %9.2f | %9.2f   mismatch A2 %u A3 %u  (%.3f %.3f %.3f %.3f)\n",
            rows, macs / 1e6,
            macs / best_a / 1e9, macs / best_a2 / 1e9,
            best_a3 < 1e17 ? macs / best_a3 / 1e9 : 0.0, macs / best_b / 1e9,
            mismatch2, mismatch3, sum_a, sum_a2, sum_a3, sum_b);
        free(ids); free(codes); free(selectors); free(out_a); free(out_a2); free(out_a3); free(out_b);
    }
    free(input); free(converted);
    return 0;
}
