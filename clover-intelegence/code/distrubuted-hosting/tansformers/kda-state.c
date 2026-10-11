#define _GNU_SOURCE
/* KDA layers cost 139.5 ms/position against MLA's 31.5, and snapshot count is not the
   reason: layers 2 and 46 agree to within 0.05% despite one and four snapshots, as do
   layers 3 and 43. So the difference is the KDA attention path.

   This is the recurrent state update copied out of transformer_update_attention, same
   loops in the same order, run on its own. It answers one question: is the 108 ms the
   state walk, or is it somewhere else in the KDA path? The parallel arm says what the
   missing pragma is worth, and the checksums must agree or it says nothing. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { HEADS = 96, HEAD = 128 };

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static float (*recurrent)[HEAD][HEAD];
static float query[HEADS * HEAD], key[HEADS * HEAD], value[HEADS * HEAD];
static float decay[HEADS * HEAD], beta[HEADS], attention[HEADS * HEAD];

static void update(int parallel)
{
    const float query_scale = 1.0f / sqrtf(128.0f);
#pragma omp parallel for schedule(static) if (parallel)
    for (int head = 0; head < HEADS; head++) {
        float prediction[HEAD] = {0};
        float *output = attention + head * HEAD;
        const float *q = query + head * HEAD;
        const float *k = key + head * HEAD;
        const float *v = value + head * HEAD;
        const float *d = decay + head * HEAD;
        for (unsigned row = 0; row < HEAD; row++)
            for (unsigned column = 0; column < HEAD; column++) {
                float state = d[row] * recurrent[head][row][column];
                recurrent[head][row][column] = state;
                prediction[column] = prediction[column] + state * k[row];
            }
        memset(output, 0, HEAD * sizeof *output);
        for (unsigned row = 0; row < HEAD; row++) {
            float key_beta = k[row] * beta[head];
            float scaled_query = q[row] * query_scale;
            for (unsigned column = 0; column < HEAD; column++) {
                float state = recurrent[head][row][column] + key_beta * (v[column] - prediction[column]);
                recurrent[head][row][column] = state;
                output[column] = output[column] + state * scaled_query;
            }
        }
    }
}

static void seed(void)
{
    uint64_t s = 2463534242ULL;
    float *all[] = {query, key, value, decay};
    for (unsigned a = 0; a < 4; a++)
        for (unsigned i = 0; i < HEADS * HEAD; i++) {
            s ^= s << 13; s ^= s >> 17; s ^= s << 5;
            all[a][i] = (float)((double)(s % 2000) / 1000.0 - 1.0);
        }
    /* The layer L2-normalises query and key per head; without it the recurrence
       diverges and the timing is measured on NaNs. */
    for (unsigned h = 0; h < HEADS; h++) {
        float *qh = query + h * HEAD, *kh = key + h * HEAD;
        double qn = 0.0, kn = 0.0;
        for (unsigned i = 0; i < HEAD; i++) { qn += (double)qh[i] * qh[i]; kn += (double)kh[i] * kh[i]; }
        qn = qn > 0 ? 1.0 / sqrt(qn) : 1.0; kn = kn > 0 ? 1.0 / sqrt(kn) : 1.0;
        for (unsigned i = 0; i < HEAD; i++) { qh[i] *= (float)qn; kh[i] *= (float)kn; }
    }
    for (unsigned i = 0; i < HEADS * HEAD; i++) decay[i] = 0.90f + 0.09f * fabsf(decay[i]);
    for (unsigned h = 0; h < HEADS; h++) beta[h] = 0.5f;
    for (unsigned h = 0; h < HEADS; h++)
        for (unsigned r = 0; r < HEAD; r++)
            for (unsigned c = 0; c < HEAD; c++) recurrent[h][r][c] = 0.001f * (float)((r + c) % 7);
}

int main(int argc, char **argv)
{
    const int positions = argc > 1 ? atoi(argv[1]) : 64;
    recurrent = malloc(sizeof(float) * HEADS * HEAD * HEAD);
    if (!recurrent) return 1;

    printf("state %.2f MB, %d positions\n", (double)sizeof(float) * HEADS * HEAD * HEAD / 1e6, positions);
    for (int parallel = 0; parallel < 2; parallel++) {
        seed();
        double start = now();
        for (int p = 0; p < positions; p++) update(parallel);
        double spent = now() - start;
        double sum = 0.0;
        for (unsigned i = 0; i < HEADS * HEAD; i++) sum += (double)attention[i];
        /* Two passes over the state, each reading and writing it. */
        double bytes = (double)positions * 4.0 * sizeof(float) * HEADS * HEAD * HEAD;
        printf("%-8s %9.3f s  %8.3f ms/position  %6.2f GB/s  checksum %.6f\n",
            parallel ? "parallel" : "serial", spent, 1000 * spent / positions, bytes / spent / 1e9, sum);
    }
    free(recurrent);
    return 0;
}
