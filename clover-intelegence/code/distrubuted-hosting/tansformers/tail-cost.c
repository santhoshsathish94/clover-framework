#define _GNU_SOURCE
/* The tail's own split. transformer-93 already accumulates normalize, head and deliver
   seconds and exposes them, so this reads those rather than subtracting one harness
   from another. The tail is the last piece of the chain: final aggregation and RMS
   normalisation, the vocabulary head, then the token back to text. */
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { WIDTH = 7168, SNAPSHOTS = 8, TOKENS = 8 };

typedef struct { const unsigned char *bytes; size_t length; } Text;

static double now(void)
{
    struct timespec moment;
    clock_gettime(CLOCK_MONOTONIC, &moment);
    return (double)moment.tv_sec + (double)moment.tv_nsec / 1e9;
}

int main(int argc, char **argv)
{
    if (argc != 2) { fputs("usage: tail-cost CODE_DIR\n", stderr); return 2; }
    const char *code = argv[1];
    char path[4096], leaves[4096], head[4096], tik[4096], vocab[4096], config[4096];

    snprintf(path, sizeof path, "%s/tansformers/transformer-93/bin/pipeline-stage.so", code);
    void *library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!library) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }
    int (*tail_open)(const char *, const char *, const char *, const char *, const char *, void **) =
        dlsym(library, "transformer_93_open");
    void *(*tail_make)(const void *) = dlsym(library, "transformer_93_sequence_create");
    int (*tail_run)(const void *, void *, const float *, const float *, unsigned, int,
        const char *, uint32_t *, Text *) = dlsym(library, "transformer_93_process");
    int (*tail_timing)(const void *, double *, double *, double *) = dlsym(library, "transformer_93_timing");
    void (*tail_drop)(void *) = dlsym(library, "transformer_93_sequence_close");
    if (!tail_open || !tail_make || !tail_run || !tail_timing || !tail_drop) { fputs("missing symbol\n", stderr); return 1; }

    snprintf(leaves, sizeof leaves, "%s/server/bin/dataset/leaves.json", code);
    snprintf(head, sizeof head, "%s/server/bin/dataset/outputs/fruit.bin", code);
    snprintf(tik, sizeof tik, "%s/server/bin/dataset/tiktoken.model", code);
    snprintf(vocab, sizeof vocab, "%s/server/bin/dataset/vocabulary.bin", code);
    snprintf(config, sizeof config, "%s/server/bin/configs/tokenizer_config.json", code);

    void *model = NULL;
    if (!tail_open(leaves, head, tik, vocab, config, &model)) { fputs("tail open failed\n", stderr); return 1; }

    static float residual[WIDTH], snapshots[SNAPSHOTS * WIDTH];
    uint64_t state = 2463534242ULL;
    for (unsigned i = 0; i < WIDTH; i++) {
        state ^= state << 13; state ^= state >> 17; state ^= state << 5;
        residual[i] = (float)((double)(state % 2000) / 1000.0 - 1.0);
    }
    for (unsigned i = 0; i < SNAPSHOTS * WIDTH; i++) snapshots[i] = residual[i % WIDTH] * 0.5f;

    uint32_t token = 0; Text word;
    void *sequence = tail_make(model);
    if (!tail_run(model, sequence, residual, snapshots, SNAPSHOTS, 0, "", &token, &word)) { fputs("warmup failed\n", stderr); return 1; }
    tail_drop(sequence);

    sequence = tail_make(model);
    double mark = now();
    for (unsigned t = 0; t < TOKENS; t++)
        if (!tail_run(model, sequence, residual, snapshots, SNAPSHOTS, 0, "", &token, &word)) { fputs("run failed\n", stderr); return 1; }
    double whole = now() - mark;

    double normalize = 0, head_s = 0, deliver = 0;
    tail_timing(sequence, &normalize, &head_s, &deliver);
    printf("%u tokens, whole tail %8.4f s  %8.3f ms/token\n", TOKENS, whole, 1000 * whole / TOKENS);
    printf("  normalisation and fold   %8.4f s  %8.3f ms/token  %5.1f%%\n",
        normalize, 1000 * normalize / TOKENS, 100 * normalize / whole);
    printf("  head and detokenise      %8.4f s  %8.3f ms/token  %5.1f%%\n",
        head_s, 1000 * head_s / TOKENS, 100 * head_s / whole);
    printf("  deliver                  %8.4f s  %8.3f ms/token  %5.1f%%\n",
        deliver, 1000 * deliver / TOKENS, 100 * deliver / whole);
    printf("  unaccounted              %8.4f s  %8.3f ms/token\n",
        whole - normalize - head_s - deliver, 1000 * (whole - normalize - head_s - deliver) / TOKENS);
    printf("(token %u)\n", token);
    tail_drop(sequence);
    return 0;
}
