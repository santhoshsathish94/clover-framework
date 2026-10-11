#define _GNU_SOURCE
/* One layer pod driven far enough that attention stops being a rounding error.
   pod-cost.c answers what residency is worth over six positions; this answers what the
   cache layout is worth once the sequence is long, which six positions cannot show
   because six positions of K/V is 592 KB and never leaves cache.

   Experts are pinned before timing so expert arrival is identical in every arm and the
   only thing left moving is the attention walk. The running checksum is the whole point
   of comparing two builds: it must agree exactly, or the speed means nothing. */
#include <dlfcn.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

enum { WIDTH = 7168 };

typedef int (*OpenFn)(const char *, void **);
typedef void *(*CreateFn)(const void *);
typedef void (*CloseSeqFn)(void *);
typedef void (*CloseFn)(void *);
typedef int (*ManyFn)(void *, void *, const float *, const float *, unsigned, float *, float *);
typedef int (*OneFn)(void *, void *, const float *, const float *, float *);

static double now(void)
{
    struct timespec moment;
    clock_gettime(CLOCK_MONOTONIC, &moment);
    return (double)moment.tv_sec + (double)moment.tv_nsec / 1e9;
}

int main(int argc, char **argv)
{
    if (argc != 7) {
        fputs("usage: pod-context LAYER SO DATASET EXPERTS POSITIONS PIN\n", stderr);
        return 2;
    }
    const unsigned layer = (unsigned)atoi(argv[1]);
    const size_t positions = strtoull(argv[5], NULL, 10);
    const int pin = atoi(argv[6]);
    const unsigned snapshot_count = (layer + 11) / 12;
    if (!positions) return 2;

    void *library = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
    if (!library) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }
    OpenFn layer_open = (OpenFn)dlsym(library, "transformer_open");
    CreateFn make = (CreateFn)dlsym(library, "transformer_sequence_create");
    CloseSeqFn drop = (CloseSeqFn)dlsym(library, "transformer_sequence_close");
    CloseFn shut = (CloseFn)dlsym(library, "transformer_close");
    void *process = dlsym(library, "transformer_process");
    if (!layer_open || !make || !drop || !shut || !process) { fputs("missing symbol\n", stderr); return 1; }

    void *model = NULL;
    if (!layer_open(argv[3], &model)) { fputs("open failed\n", stderr); return 1; }

    /* A different vector per position, so the router does not pick one expert set. */
    float *inputs = malloc(positions * WIDTH * sizeof(float));
    float *snapshots = calloc((size_t)8 * WIDTH, sizeof(float));
    float *output = malloc(WIDTH * sizeof(float));
    float *output_snapshots = malloc((size_t)8 * WIDTH * sizeof(float));
    if (!inputs || !snapshots || !output || !output_snapshots) { fputs("alloc failed\n", stderr); return 1; }
    uint64_t state = 2463534242ULL;
    for (size_t p = 0; p < positions; p++)
        for (unsigned i = 0; i < WIDTH; i++) {
            state ^= state << 13; state ^= state >> 17; state ^= state << 5;
            inputs[p * WIDTH + i] = (float)((double)(state % 2000) / 1000.0 - 1.0);
        }
    for (unsigned i = 0; i < 8 * WIDTH; i++) snapshots[i] = inputs[i % WIDTH] * 0.5f;

    double pinned = 0.0;
    double resident_gb = 0.0;
    if (pin) {
        int handle = open(argv[4], O_RDONLY);
        struct stat info;
        if (handle < 0 || fstat(handle, &info)) { fputs("cannot open experts\n", stderr); return 1; }
        size_t bytes = (size_t)info.st_size;
        void *mapped = mmap(NULL, bytes, PROT_READ, MAP_SHARED, handle, 0);
        if (mapped == MAP_FAILED) { fputs("map failed\n", stderr); return 1; }
        double mark = now();
        if (mlock(mapped, bytes)) { perror("mlock"); return 1; }
        pinned = now() - mark;
        resident_gb = (double)bytes / 1e9;
    }

    void *sequence = make(model);
    if (!sequence) { fputs("sequence failed\n", stderr); return 1; }

    double checksum = 0.0;
    double mark = now();
    for (size_t p = 0; p < positions; p++) {
        int ok = layer == 1
            ? ((OneFn)process)(model, sequence, inputs + p * WIDTH, snapshots, output)
            : ((ManyFn)process)(model, sequence, inputs + p * WIDTH, snapshots, snapshot_count,
                  output, output_snapshots);
        if (!ok) { fprintf(stderr, "process failed at position %zu\n", p); return 1; }
        for (unsigned i = 0; i < WIDTH; i++) checksum += (double)output[i];
    }
    double spent = now() - mark;
    drop(sequence);

    printf("layer %u  positions %zu  pin %d (%.2f GB in %.2f s)  total %9.3f s  "
           "%8.3f ms/position  checksum %.6f\n",
        layer, positions, pin, resident_gb, pinned, spent, 1000 * spent / (double)positions, checksum);
    fflush(stdout);
    shut(model);
    return 0;
}
