#define _GNU_SOURCE
/* Write the decoded table beside the compressed one.
   Every block goes through table_load first, so its CRC32 and its recorded SHA-256
   are both checked here. That is the whole point: the contract is honoured once at
   build time rather than on every row read, and the bytes written are exactly the
   bytes table_load would have produced. */
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
    if (argc != 3) { fputs("usage: make-direct TABLE_BIN SOURCE_DIGEST\n", stderr); return 2; }

    char target[4096];
    if (snprintf(target, sizeof target, "%s.direct", argv[1]) >= (int)sizeof target) return 2;

    NumericTable *table = table_open(argv[1], argv[2]);
    if (!table) { fprintf(stderr, "make-direct: cannot open %s\n", argv[1]); return 1; }
    if (table->direct) {
        fprintf(stderr, "make-direct: %s already exists, refusing to rewrite\n", target);
        table_close(table);
        return 1;
    }

    FILE *out = fopen(target, "wb");
    if (!out) { fprintf(stderr, "make-direct: cannot write %s\n", target); table_close(table); return 1; }

    double started = now();
    for (unsigned block = 0; block < SERVER_TABLE_BLOCKS; block++) {
        if (!table_load(table, block)) {
            fprintf(stderr, "make-direct: block %u failed verification\n", block);
            fclose(out); remove(target); table_close(table); return 1;
        }
        if (fwrite(table->raw, 1, CLIENT_RAW_BYTES, out) != CLIENT_RAW_BYTES) {
            fprintf(stderr, "make-direct: short write at block %u\n", block);
            fclose(out); remove(target); table_close(table); return 1;
        }
    }
    if (fflush(out) || fclose(out)) { fprintf(stderr, "make-direct: close failed\n"); remove(target); table_close(table); return 1; }
    table_close(table);

    printf("PASS: wrote %s, %.0f bytes, %u blocks verified, %.1fs\n",
        target, (double)SERVER_TABLE_DIRECT_BYTES, SERVER_TABLE_BLOCKS, now() - started);
    return 0;
}
