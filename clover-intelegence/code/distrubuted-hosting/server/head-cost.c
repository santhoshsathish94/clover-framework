#define _GNU_SOURCE
/* Where the head's time actually goes. Each phase is timed on the real table, not
   modelled: the whole projection, then block loading alone, then the SHA-256 that
   block loading performs, then the dot product alone. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <omp.h>
#include "numeric-table.h"

/* Same scores, same tie rule as the sequential argmax: highest score, and the lowest
   id among equals. That rule is associative, so the winner does not depend on how the
   rows were split across threads. */
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
    if (argc != 2) { fputs("usage: head-cost FRUIT_BIN\n", stderr); return 2; }
    ServerOutput *head = server_output_open(argv[1]);
    if (!head) { fputs("open failed\n", stderr); return 1; }

    float vector[SERVER_TABLE_WIDTH];
    for (unsigned i = 0; i < SERVER_TABLE_WIDTH; i++) vector[i] = 0.001f * (float)((int)(i % 17) - 8);

    unsigned codecs[8] = {0};
    for (unsigned block = 0; block < SERVER_TABLE_BLOCKS; block++) {
        unsigned codec = table_u32(head->table->index + block * 56 + 16);
        if (codec < 8) codecs[codec]++;
    }

    uint32_t token = 0;
    double mark = now();
    if (!server_output_project(head, vector, NULL, &token)) { fputs("project failed\n", stderr); return 1; }
    double whole = now() - mark;

    mark = now();
    for (unsigned block = 0; block < SERVER_TABLE_BLOCKS; block++)
        if (!table_load(head->table, block)) { fputs("load failed\n", stderr); return 1; }
    double load = now() - mark;

    unsigned char digest[32];
    mark = now();
    for (unsigned block = 0; block < SERVER_TABLE_BLOCKS; block++)
        client_sha256(head->table->raw, CLIENT_RAW_BYTES, digest);
    double sha = now() - mark;

    double sink = 0;
    mark = now();
    for (unsigned block = 0; block < SERVER_TABLE_BLOCKS; block++)
        for (unsigned local = 0; local < 16; local++)
            sink += server_head_row(head->table->raw + (size_t)local * SERVER_TABLE_WIDTH * 2, vector);
    double dot = now() - mark;

    printf("blocks %u, rows %u, bytes verified per block %u\n",
        SERVER_TABLE_BLOCKS, SERVER_TABLE_ROWS, CLIENT_RAW_BYTES);
    printf("dictionary %u entries, %u bits per value\n", head->table->dictionary_count, head->table->bits);
    printf("codec histogram:");
    for (unsigned codec = 0; codec < 8; codec++) if (codecs[codec]) printf(" codec%u=%u", codec, codecs[codec]);
    printf("\n\n");
    printf("whole head (one token)   %8.3f s\n", whole);
    printf("  block load only        %8.3f s   %5.1f%%\n", load, 100 * load / whole);
    printf("    sha256 within it     %8.3f s   %5.1f%%\n", sha, 100 * sha / whole);
    printf("    load minus sha       %8.3f s   %5.1f%%\n", load - sha, 100 * (load - sha) / whole);
    printf("  dot product only       %8.3f s   %5.1f%%\n", dot, 100 * dot / whole);
    if (head->table->direct) {
        uint32_t parallel_token = 0;
        if (!head_parallel(head, vector, &parallel_token)) { fputs("parallel failed\n", stderr); return 1; }
        mark = now();
        if (!head_parallel(head, vector, &parallel_token)) { fputs("parallel failed\n", stderr); return 1; }
        double parallel = now() - mark;
        printf("\nthreads available        %8d\n", omp_get_max_threads());
        printf("head across threads      %8.3f s   %.1fx faster\n", parallel, whole / parallel);
        printf("same token as sequential %8s (%u vs %u)\n",
            parallel_token == token ? "yes" : "NO", parallel_token, token);
        printf("effective read rate      %8.2f GB/s\n",
            (double)SERVER_TABLE_DIRECT_BYTES / parallel / 1e9);
    }
    printf("\ntotal bytes sha256'd per token: %.2f GB\n",
        (double)SERVER_TABLE_BLOCKS * CLIENT_RAW_BYTES / 1e9);
    printf("multiply-accumulates per token: %.3f billion\n",
        (double)SERVER_TABLE_ROWS * SERVER_TABLE_WIDTH / 1e9);
    printf("(sink %g, token %u)\n", sink, token);
    server_output_close(head);
    return 0;
}
