#define _GNU_SOURCE
/* head-cost.c times the first call to server_output_project, which also faults in a
   2.35 GB mapping, and compares it with a parallel arm that runs afterwards on warm
   pages. That flatters the parallel arm. This runs both repeatedly, warm, so the
   comparison is like for like, and checks they pick the same token. */
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "numeric-table.h"

static int head_parallel(ServerOutput *output, const float *vector, uint32_t *token)
{
    int bad = 0, have = 0;
    float best = 0.0f;
    uint32_t chosen = 0;
#pragma omp parallel
    {
        int local_have = 0, local_bad = 0;
        float local_best = 0.0f;
        uint32_t local_chosen = 0;
#pragma omp for schedule(static) nowait
        for (unsigned block = 0; block < SERVER_TABLE_BLOCKS; block++) {
            const unsigned char *rows = output->table->direct + (size_t)block * CLIENT_RAW_BYTES;
            for (unsigned local = 0; local < 16; local++) {
                unsigned id = block * 16 + local;
                float score = server_head_row(rows + (size_t)local * SERVER_TABLE_WIDTH * 2, vector);
                if (!isfinite(score)) { local_bad = 1; continue; }
                if (!local_have || score > local_best || (score == local_best && id < local_chosen)) {
                    local_best = score; local_chosen = id; local_have = 1;
                }
            }
        }
#pragma omp critical
        {
            if (local_bad) bad = 1;
            if (local_have && (!have || local_best > best || (local_best == best && local_chosen < chosen))) {
                best = local_best; chosen = local_chosen; have = 1;
            }
        }
    }
    *token = chosen;
    return !bad && have;
}

static double now(void)
{
    struct timespec moment;
    clock_gettime(CLOCK_MONOTONIC, &moment);
    return (double)moment.tv_sec + (double)moment.tv_nsec / 1e9;
}

int main(int argc, char **argv)
{
    if (argc != 2) { fputs("usage: head-steady FRUIT_BIN\n", stderr); return 2; }
    ServerOutput *head = server_output_open(argv[1]);
    if (!head) { fputs("open failed\n", stderr); return 1; }
    if (!head->table->direct) { fputs("no direct mapping; this measures nothing useful\n", stderr); return 1; }

    float vector[SERVER_TABLE_WIDTH];
    for (unsigned i = 0; i < SERVER_TABLE_WIDTH; i++) vector[i] = 0.001f * (float)((int)(i % 17) - 8);

    uint32_t token = 0, ptoken = 0;
    double first = now();
    if (!server_output_project(head, vector, NULL, &token)) return 1;
    first = now() - first;
    printf("first sequential call (faults in %.2f GB)  %8.3f s\n",
        (double)SERVER_TABLE_DIRECT_BYTES / 1e9, first);

    printf("\n%-28s %9s %9s\n", "warm, three runs each", "seconds", "GB/s");
    double best_seq = 1e18, best_par = 1e18;
    for (int rep = 0; rep < 3; rep++) {
        double t = now();
        if (!server_output_project(head, vector, NULL, &token)) return 1;
        t = now() - t;
        if (t < best_seq) best_seq = t;
        printf("%-28s %9.3f %9.2f\n", "  sequential", t, (double)SERVER_TABLE_DIRECT_BYTES / t / 1e9);
    }
    for (int rep = 0; rep < 3; rep++) {
        double t = now();
        if (!head_parallel(head, vector, &ptoken)) return 1;
        t = now() - t;
        if (t < best_par) best_par = t;
        printf("%-28s %9.3f %9.2f\n", "  parallel", t, (double)SERVER_TABLE_DIRECT_BYTES / t / 1e9);
    }
    printf("\nbest warm sequential %.3f s, best warm parallel %.3f s, %.1fx\n",
        best_seq, best_par, best_seq / best_par);
    printf("same token %s (%u vs %u), threads %d\n",
        token == ptoken ? "yes" : "NO", token, ptoken, omp_get_max_threads());
    server_output_close(head);
    return 0;
}
