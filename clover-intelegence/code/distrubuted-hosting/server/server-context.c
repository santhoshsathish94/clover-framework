#define _GNU_SOURCE
/* Layer 0 lives in the server rather than in a pod, and the server is built without
   OpenMP, so every loop in it runs on one core. This drives server_process the way
   pod-context drives a layer, so the two are comparable.

   Layer 0 is KDA with a dense feed-forward instead of a router and experts: three
   33792x7168 projections for gate, up and down, plus qkv and the attention output. */
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { WIDTH = 7168 };

typedef int (*OpenFn)(const char *, const char *, void **);
typedef void *(*CreateFn)(const void *);
typedef int (*ProcessFn)(const void *, void *, const float *, float *, float *);
typedef void (*CloseSeqFn)(void *);
typedef void (*CloseFn)(void *);

static double now(void)
{
    struct timespec moment;
    clock_gettime(CLOCK_MONOTONIC, &moment);
    return (double)moment.tv_sec + (double)moment.tv_nsec / 1e9;
}

int main(int argc, char **argv)
{
    if (argc != 5) { fputs("usage: server-context SO QKV_PATH TRUNK_DIR POSITIONS\n", stderr); return 2; }
    const size_t positions = strtoull(argv[4], NULL, 10);
    if (!positions) return 2;

    void *library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!library) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }
    OpenFn open_model = (OpenFn)dlsym(library, "server_open");
    CreateFn make = (CreateFn)dlsym(library, "server_sequence_create");
    ProcessFn process = (ProcessFn)dlsym(library, "server_process");
    CloseSeqFn drop = (CloseSeqFn)dlsym(library, "server_sequence_close");
    CloseFn shut = (CloseFn)dlsym(library, "server_close");
    if (!open_model || !make || !process || !drop || !shut) { fputs("missing symbol\n", stderr); return 1; }

    void *model = NULL;
    if (!open_model(argv[2], argv[3], &model)) { fputs("open failed\n", stderr); return 1; }
    void *sequence = make(model);
    if (!sequence) { fputs("sequence failed\n", stderr); return 1; }

    float *inputs = malloc(positions * WIDTH * sizeof(float));
    float *output = malloc(WIDTH * sizeof(float));
    float *snapshot = malloc(WIDTH * sizeof(float));
    if (!inputs || !output || !snapshot) { fputs("alloc failed\n", stderr); return 1; }
    uint64_t state = 2463534242ULL;
    for (size_t p = 0; p < positions; p++)
        for (unsigned i = 0; i < WIDTH; i++) {
            state ^= state << 13; state ^= state >> 17; state ^= state << 5;
            inputs[p * WIDTH + i] = (float)((double)(state % 2000) / 1000.0 - 1.0);
        }

    double checksum = 0.0;
    double mark = now();
    for (size_t p = 0; p < positions; p++) {
        if (!process(model, sequence, inputs + p * WIDTH, output, snapshot)) {
            fprintf(stderr, "process failed at position %zu\n", p);
            return 1;
        }
        for (unsigned i = 0; i < WIDTH; i++) checksum += (double)output[i];
    }
    double spent = now() - mark;

    printf("layer 0  positions %zu  total %9.3f s  %9.3f ms/position  checksum %.6f\n",
        positions, spent, 1000 * spent / (double)positions, checksum);
    drop(sequence); shut(model);
    return 0;
}
