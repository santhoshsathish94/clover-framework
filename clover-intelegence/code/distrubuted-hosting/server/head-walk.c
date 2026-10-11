/* head-walk.c - observe the last transformation, vector to token.
 *
 * Reads the final normalised vectors that tail-walk dumped from the verified
 * reference run, projects each through the real vocabulary head, and reports
 * the top candidates with their scores. This is observation, not timing:
 * head-steady.c already measured how long the projection takes.
 *
 * build: gcc -std=c11 -O2 -march=native -fopenmp head-walk.c -lm -o bin/head-walk
 * run:   ./bin/head-walk <head-path> <vectors.f32> [count]
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "numeric-table.h"

#define WIDTH 7168
#define TOPK 10

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: %s <head-path> <vectors.f32> [count]\n", argv[0]); return 2; }
    unsigned count = argc > 3 ? (unsigned)strtoul(argv[3], NULL, 10) : 5;

    ServerOutput *head = server_output_open(argv[1]);
    if (!head) { fputs("head open failed\n", stderr); return 1; }
    if (!head->table->direct) { fputs("no direct mapping\n", stderr); return 1; }

    FILE *file = fopen(argv[2], "rb");
    if (!file) { fputs("vectors open failed\n", stderr); return 1; }

    float *logits = malloc((size_t)SERVER_TABLE_ROWS * sizeof *logits);
    if (!logits) { fputs("alloc failed\n", stderr); return 1; }

    for (unsigned position = 0; position < count; position++) {
        float vector[WIDTH];
        if (fread(vector, 4, WIDTH, file) != WIDTH) { fprintf(stderr, "short read at %u\n", position); break; }

        uint32_t token = 0;
        for (unsigned row = 0; row < SERVER_TABLE_ROWS; row++) logits[row] = -INFINITY;
        if (!server_output_project(head, vector, logits, &token)) { fputs("project failed\n", stderr); return 1; }

        /* Top-k by selection, which is cheap enough at this width and avoids
           sorting 163840 scores to look at ten of them. */
        unsigned best[TOPK];
        float taken[TOPK];
        for (unsigned rank = 0; rank < TOPK; rank++) { best[rank] = 0; taken[rank] = -INFINITY; }
        for (unsigned row = 0; row < SERVER_TABLE_ROWS; row++) {
            float score = logits[row];
            if (!isfinite(score)) continue;
            for (unsigned rank = 0; rank < TOPK; rank++) {
                if (score > taken[rank]) {
                    for (unsigned shift = TOPK - 1; shift > rank; shift--) {
                        taken[shift] = taken[shift - 1]; best[shift] = best[shift - 1];
                    }
                    taken[rank] = score; best[rank] = row;
                    break;
                }
            }
        }

        /* Softmax over the whole vocabulary, in a numerically safe order, to say
           how concentrated the choice actually is rather than guessing from gaps. */
        double total = 0.0;
        for (unsigned row = 0; row < SERVER_TABLE_ROWS; row++)
            if (isfinite(logits[row])) total += exp((double)logits[row] - taken[0]);

        double norm = 0.0;
        for (unsigned coordinate = 0; coordinate < WIDTH; coordinate++)
            norm += (double)vector[coordinate] * vector[coordinate];

        printf("HEADWALK\tpos\t%u\ttoken\t%u\tvector_norm\t%.4f\ttop\t%.5f\tsecond\t%.5f\tmargin\t%.5f\tprob\t%.6f\n",
            position, token, sqrt(norm), taken[0], taken[1], taken[0] - taken[1], 1.0 / total);
        for (unsigned rank = 0; rank < TOPK; rank++)
            printf("HEADTOP\t%u\t%u\t%u\t%.5f\t%.6f\n",
                position, rank, best[rank], taken[rank], exp((double)taken[rank] - taken[0]) / total);
        fflush(stdout);
    }

    free(logits);
    fclose(file);
    server_output_close(head);
    return 0;
}
