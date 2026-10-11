#define _GNU_SOURCE
/* What a single token lookup actually costs. One row is 7168 values, 14 KB. If the
   table is reached through the compressed path, the whole 229,376-byte block that
   contains the row is read, inflated, de-interleaved and SHA-256'd to get it. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "numeric-table.h"

static double now(void)
{
    struct timespec moment;
    clock_gettime(CLOCK_MONOTONIC, &moment);
    return (double)moment.tv_sec + (double)moment.tv_nsec / 1e9;
}

int main(int argc, char **argv)
{
    if (argc != 2) { fputs("usage: input-cost SEED_BIN\n", stderr); return 2; }
    ServerInput *input = server_input_open(argv[1]);
    if (!input) { fputs("open failed\n", stderr); return 1; }
    printf("direct table mapped: %s\n", input->table->direct ? "yes" : "no");

    static float vector[SERVER_TABLE_WIDTH];
    double sink = 0;
    const unsigned rounds = 2000;

    /* Scattered ids, because consecutive ids share a block and would hide the cost. */
    uint32_t ids[2000];
    uint64_t state = 88172645463325252ULL;
    for (unsigned i = 0; i < rounds; i++) {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        ids[i] = (uint32_t)(state % SERVER_TABLE_ROWS);
    }

    if (!server_input(input, ids[0], vector)) { fputs("warm failed\n", stderr); return 1; }

    double mark = now();
    for (unsigned i = 0; i < rounds; i++) {
        if (!server_input(input, ids[i], vector)) { fputs("lookup failed\n", stderr); return 1; }
        sink += vector[0];
    }
    double scattered = now() - mark;

    mark = now();
    for (unsigned i = 0; i < rounds; i++) {
        if (!server_input(input, 1000 + (i % 16), vector)) { fputs("lookup failed\n", stderr); return 1; }
        sink += vector[0];
    }
    double same_block = now() - mark;

    printf("%u scattered lookups   %8.4f s   %8.3f us each\n", rounds, scattered, 1e6 * scattered / rounds);
    printf("%u same-block lookups  %8.4f s   %8.3f us each\n", rounds, same_block, 1e6 * same_block / rounds);
    printf("one row is %u bytes\n", SERVER_TABLE_WIDTH * 2);
    printf("(sink %g)\n", sink);
    server_input_close(input);
    return 0;
}
