#define _GNU_SOURCE
/* Can the trunk be packed to four bits without changing a single output?

   The trunk stores one byte of palette index per weight against the expert path's four
   bits, and that format difference is the rest of the gap between 50 and 79 GMAC/s.
   Packing to four bits is only lossless if the indices actually in use fit in sixteen
   values over whatever span shares a sub-palette. The expert path uses a selector per
   32 coordinates, so that is the span to check.

   This reads the real weights and counts. It writes nothing. If the counts come back
   above sixteen, a four-bit trunk means re-quantising, which changes the model, and
   that is a different decision from an optimisation. */
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static uint32_t u32(const unsigned char *b) { return (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24); }
static uint64_t u64(const unsigned char *b) { return (uint64_t)u32(b) | ((uint64_t)u32(b + 4) << 32); }

static const unsigned char *map_file(const char *path, size_t *bytes)
{
    int handle = open(path, O_RDONLY);
    struct stat info;
    if (handle < 0 || fstat(handle, &info)) { if (handle >= 0) close(handle); return NULL; }
    void *mapped = mmap(NULL, (size_t)info.st_size, PROT_READ, MAP_SHARED, handle, 0);
    close(handle);
    if (mapped == MAP_FAILED) return NULL;
    *bytes = (size_t)info.st_size;
    return mapped;
}

int main(int argc, char **argv)
{
    if (argc != 2) { fputs("usage: trunk-probe TRUNK_DIR\n", stderr); return 2; }
    char path[4096];
    size_t index_bytes = 0;
    snprintf(path, sizeof path, "%s/index.bin", argv[1]);
    const unsigned char *index = map_file(path, &index_bytes);
    if (!index || index_bytes < 1168) { fputs("cannot read index.bin\n", stderr); return 1; }
    if (memcmp(index, "K3TRK001", 8)) { fputs("bad magic\n", stderr); return 1; }
    const unsigned count = u32(index + 16);
    printf("records %u, index %zu bytes\n", count, index_bytes);

    const char *names[] = {"common.bin", "attention.bin", "routed-shared.bin"};
    const unsigned char *group_map[3] = {0};
    size_t group_bytes[3] = {0};
    for (unsigned g = 0; g < 3; g++) {
        snprintf(path, sizeof path, "%s/%s", argv[1], names[g]);
        group_map[g] = map_file(path, &group_bytes[g]);
    }

    printf("%4s %6s %7s %8s | %8s | %-28s | %-24s\n",
        "id", "rows", "cols", "MB", "distinct", "distinct per 32-coord group", "distinct per row");
    for (unsigned r = 0; r < count; r++) {
        const unsigned char *e = index + 1168 + (size_t)r * 72;
        unsigned id = u32(e), group = u32(e + 4), kind = u32(e + 8);
        unsigned rows = u32(e + 12), columns = u32(e + 16);
        uint64_t offset = u64(e + 32);
        if (kind != 1 || group > 2 || !group_map[group]) continue;
        const unsigned char *ids = group_map[group] + offset + (size_t)rows * 4;

        unsigned char seen[256] = {0};
        unsigned long groups_over16 = 0, groups_total = 0;
        unsigned worst_group = 0, worst_row = 0;
        for (unsigned row = 0; row < rows; row++) {
            unsigned char row_seen[256] = {0};
            unsigned row_distinct = 0;
            for (unsigned c = 0; c < columns; c += 32) {
                unsigned char g_seen[256] = {0};
                unsigned g_distinct = 0;
                unsigned span = columns - c < 32 ? columns - c : 32;
                for (unsigned k = 0; k < span; k++) {
                    unsigned char v = ids[(size_t)row * columns + c + k];
                    if (!g_seen[v]) { g_seen[v] = 1; g_distinct++; }
                    if (!row_seen[v]) { row_seen[v] = 1; row_distinct++; }
                    seen[v] = 1;
                }
                groups_total++;
                if (g_distinct > 16) groups_over16++;
                if (g_distinct > worst_group) worst_group = g_distinct;
            }
            if (row_distinct > worst_row) worst_row = row_distinct;
        }
        unsigned distinct = 0;
        for (unsigned v = 0; v < 256; v++) distinct += seen[v];
        printf("%4u %6u %7u %8.1f | %8u | worst %2u, over 16: %lu of %lu | worst %u\n",
            id, rows, columns, (double)rows * columns / 1e6,
            distinct, worst_group, groups_over16, groups_total, worst_row);
    }
    return 0;
}
