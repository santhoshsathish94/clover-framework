#define _GNU_SOURCE
/* What a trunk projection actually costs, copied out of transformer_project_row.

   KDA runs three of these at 12288x7168 for Q, K and V; MLA's four come to about a
   fifth of the arithmetic. The weights are one-byte palette indices, so a single
   12288x7168 record is 88 MB of ids streamed per position, and every element is an
   indexed palette gather rather than a plain multiply.

   Measured here on its own because two separate calculations of mine, one from MAC
   count and one from bandwidth, both disagreed with the layer. */
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static float palette[256];

static float project_row(const float *input, const unsigned char *ids,
    unsigned width, const float *pal, float scale)
{
    float lanes[16] = {0}, combined[8];
    for (unsigned coordinate = 0; coordinate < width; coordinate += 16)
        for (unsigned lane = 0; lane < 16; lane++)
            lanes[lane] = fmaf(pal[ids[coordinate + lane]], input[coordinate + lane], lanes[lane]);
    for (unsigned lane = 0; lane < 8; lane++) combined[lane] = lanes[lane] + lanes[lane + 8];
    float sum0 = combined[0] + combined[4], sum1 = combined[1] + combined[5];
    float sum2 = combined[2] + combined[6], sum3 = combined[3] + combined[7];
    return ((sum0 + sum2) + (sum1 + sum3)) * scale;
}

int main(int argc, char **argv)
{
    const unsigned rows = argc > 1 ? (unsigned)atoi(argv[1]) : 12288;
    const unsigned width = argc > 2 ? (unsigned)atoi(argv[2]) : 7168;
    const int repeats = argc > 3 ? atoi(argv[3]) : 5;

    size_t bytes = (size_t)rows * width;
    unsigned char *ids = malloc(bytes);
    float *input = malloc(width * sizeof(float));
    float *output = malloc(rows * sizeof(float));
    float *scales = malloc(rows * sizeof(float));
    if (!ids || !input || !output || !scales) { fputs("alloc failed\n", stderr); return 1; }

    uint64_t s = 2463534242ULL;
    for (unsigned i = 0; i < 256; i++) palette[i] = (float)((double)i / 128.0 - 1.0);
    for (unsigned i = 0; i < width; i++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        input[i] = (float)((double)(s % 2000) / 1000.0 - 1.0);
    }
    for (unsigned r = 0; r < rows; r++) scales[r] = 0.01f;
#pragma omp parallel for schedule(static)
    for (long long r = 0; r < (long long)rows; r++) {
        uint64_t local = 88172645463325252ULL + (uint64_t)r * 2654435761ULL;
        for (unsigned c = 0; c < width; c++) {
            local ^= local << 13; local ^= local >> 7; local ^= local << 17;
            ids[(size_t)r * width + c] = (unsigned char)(local & 0xFF);
        }
    }

    printf("record %ux%u = %.1f MB of ids, %.1f MMAC, %d threads\n",
        rows, width, (double)bytes / 1e6, (double)bytes / 1e6, omp_get_max_threads());

    for (int parallel = 0; parallel < 2; parallel++) {
        double best = 1e18, sum = 0.0;
        for (int r = 0; r < repeats; r++) {
            double start = now();
#pragma omp parallel for schedule(static) if (parallel)
            for (long long row = 0; row < (long long)rows; row++)
                output[row] = project_row(input, ids + (size_t)row * width, width, palette, scales[row]);
            double spent = now() - start;
            if (spent < best) best = spent;
        }
        for (unsigned r = 0; r < rows; r++) sum += (double)output[r];
        printf("%-8s %8.3f ms  %6.2f GB/s ids  %6.2f GMAC/s  checksum %.6f\n",
            parallel ? "parallel" : "serial", best * 1e3, (double)bytes / best / 1e9,
            (double)bytes / best / 1e9, sum);
    }
    free(ids); free(input); free(output); free(scales);
    return 0;
}
