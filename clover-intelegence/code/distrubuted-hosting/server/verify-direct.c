#define _GNU_SOURCE
/* Does the decoded file actually match the compressed table?
   For every block: decode it the slow way, which checks the CRC and the recorded
   SHA-256, then compare byte for byte against what the direct file holds. A direct
   file that disagrees anywhere is worse than no direct file at all. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include "numeric-table.h"

int main(int argc, char **argv)
{
    if (argc != 3) { fputs("usage: verify-direct TABLE_BIN SOURCE_DIGEST\n", stderr); return 2; }

    NumericTable *table = table_open(argv[1], argv[2]);
    if (!table) { fprintf(stderr, "verify-direct: cannot open %s\n", argv[1]); return 1; }
    if (!table->direct) { fprintf(stderr, "verify-direct: no direct file mapped\n"); table_close(table); return 1; }

    unsigned mismatched = 0, failed = 0, first_bad = SERVER_TABLE_BLOCKS;
    for (unsigned block = 0; block < SERVER_TABLE_BLOCKS; block++) {
        if (!table_load(table, block)) {
            failed++;
            if (first_bad == SERVER_TABLE_BLOCKS) first_bad = block;
            continue;
        }
        if (memcmp(table->raw, table->direct + (size_t)block * CLIENT_RAW_BYTES, CLIENT_RAW_BYTES)) {
            mismatched++;
            if (first_bad == SERVER_TABLE_BLOCKS) first_bad = block;
        }
    }

    printf("blocks %u, decode failures %u, byte mismatches %u\n",
        SERVER_TABLE_BLOCKS, failed, mismatched);
    if (failed || mismatched) printf("FAIL: first bad block %u\n", first_bad);
    else printf("PASS: direct file is byte-identical to the verified decode of every block\n");
    table_close(table);
    return failed || mismatched ? 1 : 0;
}
