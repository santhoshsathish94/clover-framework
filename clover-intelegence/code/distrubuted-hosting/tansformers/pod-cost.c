#define _GNU_SOURCE
/* One layer pod, measured the way it would run if it owned its experts.
   Cold pass with the experts evicted, then the experts read in and locked, then the
   same six positions again. The gap is what the pod design removes.
   The layer is loaded by dlopen so one binary covers all ninety-two. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

enum { WIDTH = 7168, POSITIONS = 6 };

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

static float input[POSITIONS][WIDTH], snapshots[8 * WIDTH], output[WIDTH], output_snapshots[8 * WIDTH];

int main(int argc, char **argv)
{
    if (argc != 5) { fputs("usage: pod-cost LAYER SO DATASET EXPERTS\n", stderr); return 2; }
    unsigned layer = (unsigned)atoi(argv[1]);
    unsigned snapshot_count = (layer + 11) / 12;

    void *library = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
    if (!library) { fprintf(stderr, "%u dlopen: %s\n", layer, dlerror()); return 1; }
    OpenFn layer_open = (OpenFn)dlsym(library, "transformer_open");
    CreateFn make = (CreateFn)dlsym(library, "transformer_sequence_create");
    CloseSeqFn drop = (CloseSeqFn)dlsym(library, "transformer_sequence_close");
    CloseFn shut = (CloseFn)dlsym(library, "transformer_close");
    void *process = dlsym(library, "transformer_process");
    if (!layer_open || !make || !drop || !shut || !process) { fprintf(stderr, "%u missing symbol\n", layer); return 1; }

    void *model = NULL;
    if (!layer_open(argv[3], &model)) { fprintf(stderr, "%u open failed\n", layer); return 1; }

    /* A different vector per position, so the router picks a different sixteen each
       time. One vector reused would touch the same 268 MB and hide the real spread. */
    uint64_t state = 2463534242ULL;
    for (unsigned position = 0; position < POSITIONS; position++)
        for (unsigned i = 0; i < WIDTH; i++) {
            state ^= state << 13; state ^= state >> 17; state ^= state << 5;
            input[position][i] = (float)((double)(state % 2000) / 1000.0 - 1.0);
        }
    for (unsigned i = 0; i < 8 * WIDTH; i++) snapshots[i] = input[0][i % WIDTH] * 0.5f;

    int handle = open(argv[4], O_RDONLY);
    struct stat info;
    if (handle < 0 || fstat(handle, &info)) { fprintf(stderr, "%u cannot open experts\n", layer); return 1; }
    size_t bytes = (size_t)info.st_size;

    double cold = -1, warm = -1, pinned = -1;
    for (int phase = 0; phase < 2; phase++) {
        void *sequence = make(model);
        if (!sequence) { fprintf(stderr, "%u sequence failed\n", layer); return 1; }
        double mark = now();
        int ok = 1;
        for (unsigned position = 0; position < POSITIONS && ok; position++)
            ok = layer == 1
                ? ((OneFn)process)(model, sequence, input[position], snapshots, output)
                : ((ManyFn)process)(model, sequence, input[position], snapshots, snapshot_count, output, output_snapshots);
        double spent = now() - mark;
        drop(sequence);
        if (!ok) { fprintf(stderr, "%u process failed in phase %d\n", layer, phase); return 1; }
        if (phase == 0) {
            cold = spent;
            void *mapped = mmap(NULL, bytes, PROT_READ, MAP_SHARED, handle, 0);
            if (mapped == MAP_FAILED) { fprintf(stderr, "%u map failed\n", layer); return 1; }
            mark = now();
            if (mlock(mapped, bytes)) { fprintf(stderr, "%u mlock failed\n", layer); return 1; }
            pinned = now() - mark;
        } else warm = spent;
    }

    printf("%u,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f\n", layer, cold, pinned, warm,
        1000 * warm / POSITIONS, 1000 * cold / POSITIONS, (double)bytes / 1e9);
    fflush(stdout);

    shut(model);
    close(handle);
    return 0;
}
