#ifndef CLOVER_TRACE_H
#define CLOVER_TRACE_H
/* Records every stage that actually executes, so the path can be read off a run
   instead of reasoned about. Off unless CLOVER_TRACE names a file, and when off it
   costs one pointer test per stage.

   What is recorded is deliberately narrow: which stage ran, how long it took, and a
   hash of the residual afterwards. The hash is what makes repetition visible without
   storing 28 KB per stage. It answers "did this stage change the carried state", not
   "was this stage useful". */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

static FILE *clover_trace_sink;
static int clover_trace_opened;

static void clover_trace_open(void)
{
    const char *path;
    if (clover_trace_opened) return;
    clover_trace_opened = 1;
    path = getenv("CLOVER_TRACE");
    if (!path || !*path) return;
    /* O_APPEND keeps short lines from different pods from tearing each other. */
    clover_trace_sink = fopen(path, "a");
    if (clover_trace_sink) setvbuf(clover_trace_sink, NULL, _IOLBF, 0);
}

static int clover_trace_active(void)
{
    return clover_trace_sink != NULL;
}

static double clover_trace_clock(void)
{
    struct timespec moment;
    clock_gettime(CLOCK_MONOTONIC, &moment);
    return (double)moment.tv_sec * 1e9 + (double)moment.tv_nsec;
}

static uint64_t clover_trace_hash(const void *data, size_t bytes)
{
    const unsigned char *walk = data;
    uint64_t hash = 14695981039346656037ULL;
    for (size_t index = 0; index < bytes; index++) {
        hash ^= walk[index];
        hash *= 1099511628211ULL;
    }
    return hash;
}

/* owner: 0 server, 1..92 layer, 93 tail. detail: expert id, or -1. */
static void clover_trace_stage(int owner, int local, unsigned long position,
    double nanos, uint64_t residual, long detail)
{
    if (!clover_trace_sink) return;
    fprintf(clover_trace_sink, "%d,%d,%lu,%.0f,%016llx,%ld\n",
        owner, local, position, nanos, (unsigned long long)residual, detail);
}

/* The vectors themselves, for one owner only. A hash shows that state changed; it
   cannot show what it became, and nothing about the shape of a transformation can be
   read from it. Written as a flat binary record so no precision is lost on the way
   out: an 8-byte header of four int32 then count floats, little endian as written.

   One file per owner. A record is 28 KB and O_APPEND only keeps small writes whole,
   so ninety-two owners sharing one file tear each other's records apart. */

static FILE *clover_vectors_sink;
static int clover_vectors_opened, clover_vectors_owner = -1, clover_vectors_keep;

enum { CLOVER_KEEP_ALL = 0, CLOVER_KEEP_WEIGHTS = 1, CLOVER_KEEP_SNAPSHOTS = 2 };

static void clover_vectors_open(int owner)
{
    const char *path, *only;
    char named[4096];
    if (clover_vectors_opened) return;
    clover_vectors_opened = 1;
    path = getenv("CLOVER_VECTORS");
    if (!path || !*path) return;
    only = getenv("CLOVER_VECTORS_OWNER");
    clover_vectors_owner = only && *only ? atoi(only) : -1;
    only = getenv("CLOVER_VECTORS_KEEP");
    if (only && !strcmp(only, "weights")) clover_vectors_keep = CLOVER_KEEP_WEIGHTS;
    else if (only && !strcmp(only, "snapshots")) clover_vectors_keep = CLOVER_KEEP_SNAPSHOTS;
    if (clover_vectors_owner >= 0 && clover_vectors_owner != owner) return;
    if (snprintf(named, sizeof named, "%s.%d.bin", path, owner) >= (int)sizeof named) return;
    clover_vectors_sink = fopen(named, "wb");
}

static int clover_vectors_active(int owner)
{
    return clover_vectors_sink != NULL &&
        (clover_vectors_owner < 0 || clover_vectors_owner == owner);
}

static void clover_vectors_write(int owner, int local, unsigned long position,
    const float *values, unsigned count)
{
    int32_t header[4];
    if (!clover_vectors_active(owner)) return;
    /* A full dump is about 1.7 GB per prompt, so a run that only needs the fold
       weights or only the snapshots says so and keeps the rest out.
       "weights" also keeps 4000+stage, the vector the fold produced. Asking what a
       fold did and being handed only its weights is what made this filter wrong
       once already, when 2000 and 3000 were added above a bare lower bound. */
    if (clover_vectors_keep == CLOVER_KEEP_WEIGHTS
        && !(local >= 2000 && local < 3000) && !(local >= 4000 && local < 5000)) return;
    if (clover_vectors_keep == CLOVER_KEEP_SNAPSHOTS && (local < 1000 || local >= 1008)) return;
    header[0] = owner;
    header[1] = local;
    header[2] = (int32_t)position;
    header[3] = (int32_t)count;
    if (fwrite(header, sizeof header, 1, clover_vectors_sink) == 1)
        (void)fwrite(values, sizeof *values, count, clover_vectors_sink);
    /* The fleet is normally stopped with a signal, which would discard a buffered
       tail and leave a half record behind. */
    (void)fflush(clover_vectors_sink);
}

/* Experiment only, off unless CLOVER_FREEZE names a capture prefix: replace this
   layer's incoming snapshots, from slot CLOVER_FREEZE_FROM upward, with the ones a
   different prompt produced. If the output does not move, those slots were not
   carrying anything the answer depended on. Deliberately not bit-exact. */

static float clover_frozen[8][7168];
static int clover_frozen_loaded, clover_frozen_have[8], clover_frozen_from = -1, clover_frozen_to = 7;

static void clover_freeze_load(int owner)
{
    const char *path, *from;
    char named[4096];
    FILE *file;
    if (clover_frozen_loaded) return;
    clover_frozen_loaded = 1;
    path = getenv("CLOVER_FREEZE");
    if (!path || !*path) return;
    from = getenv("CLOVER_FREEZE_FROM");
    clover_frozen_from = from && *from ? atoi(from) : 0;
    from = getenv("CLOVER_FREEZE_TO");
    if (from && *from) clover_frozen_to = atoi(from);
    if (snprintf(named, sizeof named, "%s.%d.bin", path, owner) >= (int)sizeof named) return;
    file = fopen(named, "rb");
    if (!file) { clover_frozen_from = -1; return; }
    for (;;) {
        int32_t header[4];
        if (fread(header, sizeof header, 1, file) != 1) break;
        if (header[3] <= 0 || header[3] > 7168) break;
        if (header[1] >= 1000 && header[1] < 1008 && header[3] == 7168 && !clover_frozen_have[header[1] - 1000]) {
            int slot = header[1] - 1000;
            if (fread(clover_frozen[slot], sizeof(float), 7168, file) != 7168) break;
            clover_frozen_have[slot] = 1;
        } else if (fseek(file, 4L * header[3], SEEK_CUR)) break;
    }
    fclose(file);
}

/* Stage 4 only ever appends at index snapshot_count, so slots below it are the same
   going in as coming out. That is why a capture of the outgoing snapshots can be used
   to replace the incoming ones. */
static void clover_freeze_apply(int owner, float *snapshots, unsigned count)
{
    clover_freeze_load(owner);
    if (clover_frozen_from < 0) return;
    for (unsigned slot = (unsigned)clover_frozen_from; slot < count && slot < 8; slot++)
        if ((int)slot <= clover_frozen_to && clover_frozen_have[slot])
            memcpy(snapshots + (size_t)slot * 7168, clover_frozen[slot], 7168 * sizeof(float));
}

#endif
