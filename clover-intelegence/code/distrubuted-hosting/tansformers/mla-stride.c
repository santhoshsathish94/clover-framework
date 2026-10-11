/* Isolates the MLA attention step of a distributed layer: the cache layout and the
   head loop, at real sizes, with nothing else in the way.

   The layer stores K, V and the positional tail interleaved per position, so each of
   the 96 heads walks the sequence at a 98,560 byte stride and the bus delivers whole
   lines to use 512 bytes of each. Head-major stores one head's positions contiguously.
   Both arms do identical arithmetic in identical order, so the checksums must agree;
   only the addresses differ. The head loop is also serial in the layer, so each layout
   is timed once per core count. */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { ROWS = 12288, HEADS = 96, HEAD = 128, ROPE = 64, QUERY = 192 };
static const size_t STRIDE = 2 * ROWS + ROPE;

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

/* Layout independent, so the two arms hold the same logical cache. */
static inline float value_at(uint64_t head, uint64_t position, uint64_t coordinate)
{
    uint64_t x = head * 1000003u + position * 31u + coordinate;
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33;
    return (float)((double)(x & 0xFFFFFFu) / 8388608.0 - 1.0);
}

static void fill_position_major(float *cache, size_t count)
{
#pragma omp parallel for schedule(static)
    for (long long p = 0; p < (long long)count; p++) {
        float *base = cache + (size_t)p * STRIDE;
        for (unsigned h = 0; h < HEADS; h++)
            for (unsigned c = 0; c < HEAD; c++) {
                base[h * HEAD + c] = value_at(h, (uint64_t)p, c);
                base[ROWS + h * HEAD + c] = value_at(h + 512, (uint64_t)p, c);
            }
        for (unsigned c = 0; c < ROPE; c++) base[2 * ROWS + c] = value_at(9999, (uint64_t)p, c);
    }
}

static void fill_head_major(float *cache, size_t count)
{
#pragma omp parallel for schedule(static)
    for (long long h = 0; h < HEADS; h++)
        for (size_t p = 0; p < count; p++)
            for (unsigned c = 0; c < HEAD; c++) {
                cache[(size_t)h * HEAD * count + p * HEAD + c] = value_at((uint64_t)h, p, c);
                cache[((size_t)ROWS + h * HEAD) * count + p * HEAD + c] = value_at((uint64_t)h + 512, p, c);
            }
#pragma omp parallel for schedule(static)
    for (long long p = 0; p < (long long)count; p++)
        for (unsigned c = 0; c < ROPE; c++)
            cache[2 * (size_t)ROWS * count + (size_t)p * ROPE + c] = value_at(9999, (uint64_t)p, c);
}

/* One attention step over count positions. Arithmetic order matches the layer exactly. */
static double attend(const float *cache, const float *query, size_t count, int head_major, int parallel,
    float *output)
{
    const float scale = 1.0f / sqrtf(192.0f);
    double start = now();
#pragma omp parallel for schedule(static) if (parallel)
    for (long long head = 0; head < HEADS; head++) {
        float *scores = malloc(count * 2 * sizeof(float));
        if (!scores) continue;
        float *exponentials = scores + count;
        const float *q = query + head * QUERY;
        for (size_t position = 0; position < count; position++) {
            const float *cached, *positional;
            if (head_major) {
                cached = cache + (size_t)head * HEAD * count + position * HEAD;
                positional = cache + 2 * (size_t)ROWS * count + position * ROPE;
            } else {
                cached = cache + position * STRIDE + (size_t)head * HEAD;
                positional = cache + position * STRIDE + 2 * ROWS;
            }
            double score = 0.0;
            for (unsigned c = 0; c < HEAD; c++) score += (double)q[c] * (double)cached[c];
            for (unsigned c = 0; c < ROPE; c++) score += (double)q[HEAD + c] * (double)positional[c];
            scores[position] = (float)score * scale;
        }
        float maximum = scores[0];
        for (size_t position = 1; position < count; position++)
            if (scores[position] > maximum) maximum = scores[position];
        double total = 0.0;
        for (size_t position = 0; position < count; position++) {
            exponentials[position] = expf(scores[position] - maximum);
            total += (double)exponentials[position];
        }
        float *out = output + head * HEAD;
        memset(out, 0, HEAD * sizeof(float));
        for (size_t position = 0; position < count; position++) {
            float weight = (float)((double)exponentials[position] / total);
            const float *value = head_major
                ? cache + ((size_t)ROWS + head * HEAD) * count + position * HEAD
                : cache + position * STRIDE + ROWS + (size_t)head * HEAD;
            for (unsigned c = 0; c < HEAD; c++) out[c] = out[c] + weight * value[c];
        }
        free(scores);
    }
    return now() - start;
}

static double checksum(const float *output)
{
    double sum = 0.0;
    for (unsigned i = 0; i < ROWS; i++) sum += (double)output[i];
    return sum;
}

int main(int argc, char **argv)
{
    size_t count = argc > 1 ? strtoull(argv[1], NULL, 10) : 32768;
    int repeats = argc > 2 ? atoi(argv[2]) : 2;
    if (!count) return 1;

    float *query = malloc(HEADS * QUERY * sizeof(float));
    float *output = malloc(ROWS * sizeof(float));
    if (!query || !output) return 1;
    for (unsigned h = 0; h < HEADS; h++)
        for (unsigned c = 0; c < QUERY; c++) query[h * QUERY + c] = value_at(8888, h, c);

    double bytes = (double)count * (double)STRIDE * sizeof(float);
    printf("positions %zu  cache %.3f GB  repeats %d\n", count, bytes / 1e9, repeats);

    double best[4] = {1e18, 1e18, 1e18, 1e18};
    double sums[4] = {0, 0, 0, 0};
    const char *names[4] = {"position-major serial", "position-major parallel",
                            "head-major serial", "head-major parallel"};

    for (int layout = 0; layout < 2; layout++) {
        float *cache = malloc((size_t)count * STRIDE * sizeof(float));
        if (!cache) { printf("allocation failed for %.3f GB\n", bytes / 1e9); return 1; }
        if (layout) fill_head_major(cache, count); else fill_position_major(cache, count);
        for (int parallel = 0; parallel < 2; parallel++) {
            int arm = layout * 2 + parallel;
            for (int r = 0; r < repeats; r++) {
                double seconds = attend(cache, query, count, layout, parallel, output);
                if (seconds < best[arm]) best[arm] = seconds;
            }
            sums[arm] = checksum(output);
        }
        free(cache);
    }

    for (int arm = 0; arm < 4; arm++)
        printf("%-24s %8.1f ms  %7.2f GB/s  checksum %.6f\n",
            names[arm], best[arm] * 1e3, bytes / best[arm] / 1e9, sums[arm]);
    printf("layout gain serial %.2fx  parallel %.2fx  serial->parallel head-major %.2fx\n",
        best[0] / best[2], best[1] / best[3], best[2] / best[3]);
    free(query); free(output);
    return 0;
}
