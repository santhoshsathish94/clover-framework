/* head-shape.c - describe the vocabulary projection as a distribution.
 *
 * The previous pass asked the head "did you pick the expected token", which makes every
 * reading a test of my expectation rather than a description of the step. This asks only
 * what the distribution looks like: where the mass sits, how many entries are near the
 * top, how wide the spread is. No expected token appears anywhere in this file.
 *
 * The selected id is printed because it is what the step produces and the walk would be
 * incomplete without it, not because anything here is scored against it.
 *
 * Softmax statistics are NOT comparable between vectors of different norm - a dot
 * product scales with the vector, so the same direction at twice the length gives a
 * sharper distribution. The norm is printed beside every row so the reader can see when
 * a comparison would be unsound.
 *
 * build: gcc -std=c11 -O2 -march=native -ffp-contract=off -fno-fast-math -fopenmp head-shape.c -lm -o bin/head-shape
 * run:   ./bin/head-shape <head-path> <vectors.f32> [count]
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "numeric-table.h"

#define WIDTH 7168

static int descending(const void *a, const void *b)
{
    float x = *(const float *)a, y = *(const float *)b;
    return x < y ? 1 : x > y ? -1 : 0;
}

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: %s <head-path> <vectors.f32> [count]\n", argv[0]); return 2; }
    unsigned count = argc > 3 ? (unsigned)strtoul(argv[3], NULL, 10) : 1;

    ServerOutput *head = server_output_open(argv[1]);
    if (!head) { fputs("head open failed\n", stderr); return 1; }
    FILE *file = fopen(argv[2], "rb");
    if (!file) { fputs("vectors open failed\n", stderr); return 1; }

    float *logits = malloc((size_t)SERVER_TABLE_ROWS * sizeof *logits);
    float *sorted = malloc((size_t)SERVER_TABLE_ROWS * sizeof *sorted);
    if (!logits || !sorted) { fputs("alloc failed\n", stderr); return 1; }

    printf("%4s %10s %8s %10s %10s %9s %9s %9s %8s %8s %8s %7s\n",
           "slot", "norm", "token", "top", "mean", "sd", "margin1",
           "entropy", "p(top1)", "p(top10)", "within1", "within2");

    for (unsigned slot = 0; slot < count; slot++) {
        float vector[WIDTH];
        if (fread(vector, 4, WIDTH, file) != WIDTH) break;

        double square = 0.0;
        for (unsigned i = 0; i < WIDTH; i++) square += (double)vector[i] * (double)vector[i];

        uint32_t token = 0;
        for (unsigned row = 0; row < SERVER_TABLE_ROWS; row++) logits[row] = -INFINITY;
        if (!server_output_project(head, vector, logits, &token)) { fputs("project failed\n", stderr); return 1; }

        memcpy(sorted, logits, (size_t)SERVER_TABLE_ROWS * sizeof *logits);
        qsort(sorted, SERVER_TABLE_ROWS, sizeof *sorted, descending);

        double sum = 0.0, square_sum = 0.0;
        for (unsigned row = 0; row < SERVER_TABLE_ROWS; row++) {
            sum += logits[row];
            square_sum += (double)logits[row] * (double)logits[row];
        }
        double mean = sum / SERVER_TABLE_ROWS;
        double sd = sqrt(square_sum / SERVER_TABLE_ROWS - mean * mean);

        /* Shift by the maximum before exponentiating, the usual guard against overflow;
           it leaves every probability unchanged. */
        double total = 0.0;
        for (unsigned row = 0; row < SERVER_TABLE_ROWS; row++) total += exp((double)logits[row] - sorted[0]);
        double entropy = 0.0;
        for (unsigned row = 0; row < SERVER_TABLE_ROWS; row++) {
            double p = exp((double)logits[row] - sorted[0]) / total;
            if (p > 0.0) entropy -= p * log(p);
        }
        double top1 = exp(0.0) / total;
        double top10 = 0.0;
        for (unsigned rank = 0; rank < 10; rank++) top10 += exp((double)sorted[rank] - sorted[0]) / total;

        unsigned within1 = 0, within2 = 0;
        for (unsigned row = 0; row < SERVER_TABLE_ROWS; row++) {
            if (logits[row] > sorted[0] - 1.0f) within1++;
            if (logits[row] > sorted[0] - 2.0f) within2++;
        }

        printf("%4u %10.4f %8u %10.5f %10.5f %9.5f %9.5f %9.5f %8.5f %8.5f %8u %7u\n",
               slot, sqrt(square), token, sorted[0], mean, sd, sorted[0] - sorted[1],
               entropy, top1, top10, within1, within2);
    }

    free(logits); free(sorted); fclose(file);
    server_output_close(head);
    return 0;
}
