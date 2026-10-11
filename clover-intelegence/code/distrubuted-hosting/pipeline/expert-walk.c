/* Walks one layer's experts in the order real traffic touched them and reports major
   faults. On a large host the page cache hides everything, so this must be run inside a
   memory cgroup: major faults are the reads that would cross the network on a pod with
   no local disk. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>
#include <sys/mman.h>
#include <errno.h>
#include <omp.h>
#include "live-root.h"

static double clock_now(void)
{
    struct timespec moment; clock_gettime(CLOCK_MONOTONIC, &moment);
    return (double)moment.tv_sec + (double)moment.tv_nsec / 1e9;
}

/* Readahead turns most major faults into minor ones, so faults understate what the
   device actually served. read_bytes is the block-layer total. */
static unsigned long long storage_read_bytes(void)
{
    FILE *file = fopen("/proc/self/io", "r");
    if (!file) return 0;
    char label[64]; unsigned long long value = 0, found = 0;
    while (fscanf(file, "%63s %llu", label, &value) == 2)
        if (!strcmp(label, "read_bytes:")) { found = value; break; }
    fclose(file);
    return found;
}

int main(int argc, char **argv)
{
    if (argc < 4) { fprintf(stderr, "usage: expert-walk ROOT_DIR LAYER ID_FILE [lock]\n"); return 2; }
    unsigned layer = (unsigned)strtoul(argv[2], NULL, 10);
    int pin = argc > 4 && !strcmp(argv[4], "lock");
    Root *root = root_open(argv[1], layer);
    if (!root) { fprintf(stderr, "root_open failed for %s layer %u\n", argv[1], layer); return 1; }
    RootScratch *scratch = calloc((size_t)omp_get_max_threads(), sizeof *scratch);
    if (!scratch) { fprintf(stderr, "scratch allocation failed\n"); return 1; }

    /* Pinning the whole store is the only way to guarantee no expert ever comes from
       disk again; mmap alone lets the kernel evict under pressure. */
    if (pin) {
        unsigned long long locked_before = storage_read_bytes();
        double lock_started = clock_now();
        if (!root->direct) { fprintf(stderr, "no direct mapping to pin\n"); return 1; }
        if (mlock(root->direct, root->direct_bytes)) {
            printf("layer=%-3u PIN FAILED at %.2f GiB: %s\n",
                layer, (double)root->direct_bytes / 1073741824.0, strerror(errno));
            return 1;
        }
        printf("layer=%-3u pinned %.2f GiB in %.1fs, pulled %.2f GiB from device\n",
            layer, (double)root->direct_bytes / 1073741824.0, clock_now() - lock_started,
            (double)(storage_read_bytes() - locked_before) / 1073741824.0);
    }

    static float input[3584], gate[3072], up[3072], down[3584];
    for (unsigned i = 0; i < 3584; i++) input[i] = 1.0f / (float)(i + 1);

    FILE *file = fopen(argv[3], "r");
    if (!file) { fprintf(stderr, "cannot read %s\n", argv[3]); return 1; }

    struct rusage before; getrusage(RUSAGE_SELF, &before);
    unsigned long long read_before = storage_read_bytes();
    double started = clock_now();
    unsigned expert; size_t touched = 0;
    while (fscanf(file, "%u", &expert) == 1) {
        if (!root_project(root, scratch, expert, 0, input, gate) ||
            !root_project(root, scratch, expert, 1, input, up) ||
            !root_project(root, scratch, expert, 2, gate, down)) {
            fprintf(stderr, "projection failed at expert %u\n", expert); return 1;
        }
        touched++;
    }
    double elapsed = clock_now() - started;
    unsigned long long read_after = storage_read_bytes();
    struct rusage after; getrusage(RUSAGE_SELF, &after);
    fclose(file);

    long major = after.ru_majflt - before.ru_majflt;
    printf("layer=%-3u visits=%zu distinct_bytes_expected=%.2fGiB from_device=%.2fGiB "
           "wall=%.2fs major_faults=%ld per_visit=%.3fs\n",
        layer, touched,
        (double)touched * (double)ROOT_EXPERT_RAW / 1073741824.0,
        (double)(read_after - read_before) / 1073741824.0,
        elapsed, major, touched ? elapsed / (double)touched : 0.0);
    root_close(root);
    free(scratch);
    return 0;
}
