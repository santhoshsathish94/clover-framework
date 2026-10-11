#define _GNU_SOURCE
/* Feed controlled vectors through one stage-120 block and record the whole trajectory.
   A fresh sequence per probe, so each run starts from the same state and the block is
   exercised as a function of its input rather than of its history.

   The probes are chosen so the observations can decide the transformation class
   instead of assuming it: zero gives any constant offset, u and v give two unrelated
   trajectories, u+v tests additivity, 2u tests homogeneity, and w is held back from
   every derivation for independent validation. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dlfcn.h>

enum { WIDTH = 7168, SNAPSHOTS = 8 };

typedef int (*OpenFn)(const char *, void **);
typedef void *(*CreateFn)(const void *);
typedef void (*CloseSeqFn)(void *);
typedef int (*ManyFn)(void *, void *, const float *, const float *, unsigned, float *, float *);

static float probe[6][WIDTH], snapshots[SNAPSHOTS * WIDTH];
static float output[WIDTH], output_snapshots[SNAPSHOTS * WIDTH];

static void fill(float *target, uint64_t state)
{
    for (unsigned i = 0; i < WIDTH; i++) {
        state ^= state << 13; state ^= state >> 7; state ^= state << 17;
        target[i] = (float)((double)(state % 2000) / 1000.0 - 1.0);
    }
}

int main(int argc, char **argv)
{
    if (argc != 5) { fputs("usage: probe-category LAYER SO DATASET OUTPUT\n", stderr); return 2; }
    unsigned layer = (unsigned)atoi(argv[1]);
    unsigned snapshot_count = (layer + 11) / 12;

    void *library = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
    if (!library) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }
    OpenFn layer_open = (OpenFn)dlsym(library, "transformer_open");
    CreateFn make = (CreateFn)dlsym(library, "transformer_sequence_create");
    CloseSeqFn drop = (CloseSeqFn)dlsym(library, "transformer_sequence_close");
    ManyFn run = (ManyFn)dlsym(library, "transformer_process");
    if (!layer_open || !make || !drop || !run) { fputs("missing symbol\n", stderr); return 1; }

    void *model = NULL;
    if (!layer_open(argv[3], &model)) { fputs("open failed\n", stderr); return 1; }

    /* probe 0 zero, 1 u, 2 v, 3 u+v, 4 2u, 5 w held out */
    memset(probe[0], 0, sizeof probe[0]);
    fill(probe[1], 2463534242ULL);
    fill(probe[2], 88172645463325252ULL);
    for (unsigned i = 0; i < WIDTH; i++) probe[3][i] = probe[1][i] + probe[2][i];
    for (unsigned i = 0; i < WIDTH; i++) probe[4][i] = probe[1][i] + probe[1][i];
    fill(probe[5], 99194853094755497ULL);
    fill(snapshots, 1234567890123ULL);
    for (unsigned i = WIDTH; i < SNAPSHOTS * WIDTH; i++) snapshots[i] = snapshots[i % WIDTH] * 0.5f;

    FILE *finals = fopen(argv[4], "wb");
    if (!finals) { fputs("cannot write output\n", stderr); return 1; }

    for (unsigned p = 0; p < 6; p++) {
        void *sequence = make(model);
        if (!sequence) { fputs("sequence failed\n", stderr); return 1; }
        if (!run(model, sequence, probe[p], snapshots, snapshot_count, output, output_snapshots)) {
            fprintf(stderr, "probe %u failed\n", p);
            return 1;
        }
        drop(sequence);
        if (fwrite(probe[p], sizeof(float), WIDTH, finals) != WIDTH ||
            fwrite(output, sizeof(float), WIDTH, finals) != WIDTH) { fputs("short write\n", stderr); return 1; }
        printf("probe %u done\n", p);
    }
    fclose(finals);
    printf("PASS: 6 probes, input and output pairs written to %s\n", argv[4]);
    return 0;
}
