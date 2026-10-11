#define _GNU_SOURCE
/* The two edge pods, measured with their data resident.
   The server turns a token into a vector and runs trunk 0. Layer 93 turns a vector
   back into a token. Both hold small enough data that it is simply read in first. */
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

enum { WIDTH = 7168, POSITIONS = 6, TOKENS = 2, SNAPSHOTS = 8 };

typedef struct { const unsigned char *bytes; size_t length; } Text;

static double now(void)
{
    struct timespec moment;
    clock_gettime(CLOCK_MONOTONIC, &moment);
    return (double)moment.tv_sec + (double)moment.tv_nsec / 1e9;
}

/* Read the file through once so the pod is measured warm, as it would run. */
static double warm(const char *path)
{
    int handle = open(path, O_RDONLY);
    struct stat info;
    if (handle < 0 || fstat(handle, &info)) { if (handle >= 0) close(handle); return 0; }
    double mark = now();
    void *mapped = mmap(NULL, (size_t)info.st_size, PROT_READ, MAP_SHARED, handle, 0);
    volatile unsigned char sink = 0;
    if (mapped != MAP_FAILED) {
        long page = sysconf(_SC_PAGESIZE);
        for (size_t at = 0; at < (size_t)info.st_size; at += (size_t)page) sink ^= ((unsigned char *)mapped)[at];
        munmap(mapped, (size_t)info.st_size);
    }
    close(handle);
    (void)sink;
    return now() - mark;
}

static float input[POSITIONS][WIDTH], snapshots[SNAPSHOTS * WIDTH], output[WIDTH], snapshot[WIDTH];

int main(int argc, char **argv)
{
    if (argc != 3) { fputs("usage: edge-cost CODE_ROOT DATASET\n", stderr); return 2; }
    const char *code = argv[1], *data = argv[2];
    char path[4096], qkv[4096], trunk[4096], leaves[4096], head[4096], tik[4096], vocab[4096], config[4096];

    uint64_t state = 99194853094755497ULL;
    for (unsigned p = 0; p < POSITIONS; p++)
        for (unsigned i = 0; i < WIDTH; i++) {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17;
            input[p][i] = (float)((double)(state % 2000) / 1000.0 - 1.0);
        }
    for (unsigned i = 0; i < SNAPSHOTS * WIDTH; i++) snapshots[i] = input[0][i % WIDTH] * 0.25f;

    /* ---- server pod ---- */
    snprintf(path, sizeof path, "%s/server/bin/pipeline-stage.so", code);
    void *library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!library) { fprintf(stderr, "server dlopen: %s\n", dlerror()); return 1; }
    int (*server_open)(const char *, const char *, void **) = dlsym(library, "server_open");
    void *(*server_make)(const void *) = dlsym(library, "server_sequence_create");
    void (*server_drop)(void *) = dlsym(library, "server_sequence_close");
    int (*server_run)(const void *, void *, const float *, float *, float *) = dlsym(library, "server_process");
    if (!server_open || !server_make || !server_drop || !server_run) { fputs("server symbols missing\n", stderr); return 1; }

    snprintf(qkv, sizeof qkv, "%s/server/bin/dataset/trunk-0-qkv/qkv.bin", code);
    snprintf(trunk, sizeof trunk, "%s/server/bin/dataset/trunk-0", code);
    double server_warm = warm(qkv);
    snprintf(path, sizeof path, "%s/inputs/seed.bin.direct", data);
    server_warm += warm(path);

    void *server_model = NULL;
    if (!server_open(qkv, trunk, &server_model)) { fputs("server open failed\n", stderr); return 1; }
    void *server_seq = server_make(server_model);
    if (!server_seq) { fputs("server sequence failed\n", stderr); return 1; }
    for (unsigned p = 0; p < POSITIONS; p++)
        if (!server_run(server_model, server_seq, input[p], output, snapshot)) { fputs("server warmup failed\n", stderr); return 1; }
    server_drop(server_seq);

    server_seq = server_make(server_model);
    double mark = now();
    for (unsigned p = 0; p < POSITIONS; p++)
        if (!server_run(server_model, server_seq, input[p], output, snapshot)) { fputs("server failed\n", stderr); return 1; }
    double server_time = now() - mark;
    server_drop(server_seq);

    /* ---- layer 93 pod ---- */
    snprintf(path, sizeof path, "%s/tansformers/transformer-93/bin/pipeline-stage.so", code);
    void *tail = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!tail) { fprintf(stderr, "tail dlopen: %s\n", dlerror()); return 1; }
    int (*tail_open)(const char *, const char *, const char *, const char *, const char *, void **) = dlsym(tail, "transformer_93_open");
    void *(*tail_make)(const void *) = dlsym(tail, "transformer_93_sequence_create");
    void (*tail_drop)(void *) = dlsym(tail, "transformer_93_sequence_close");
    int (*tail_run)(const void *, void *, const float *, const float *, unsigned, int, const char *, uint32_t *, Text *) = dlsym(tail, "transformer_93_process");
    if (!tail_open || !tail_make || !tail_drop || !tail_run) { fputs("tail symbols missing\n", stderr); return 1; }

    snprintf(leaves, sizeof leaves, "%s/server/bin/dataset/leaves.json", code);
    snprintf(head, sizeof head, "%s/server/bin/dataset/outputs/fruit.bin", code);
    snprintf(tik, sizeof tik, "%s/server/bin/dataset/tiktoken.model", code);
    snprintf(vocab, sizeof vocab, "%s/server/bin/dataset/vocabulary.bin", code);
    snprintf(config, sizeof config, "%s/server/bin/configs/tokenizer_config.json", code);

    snprintf(path, sizeof path, "%s/outputs/fruit.bin.direct", data);
    double tail_warm = warm(path);

    void *tail_model = NULL;
    if (!tail_open(leaves, head, tik, vocab, config, &tail_model)) { fputs("tail open failed\n", stderr); return 1; }
    void *tail_seq = tail_make(tail_model);
    uint32_t token = 0; Text word;
    if (!tail_run(tail_model, tail_seq, input[0], snapshots, SNAPSHOTS, 0, "", &token, &word)) { fputs("tail warmup failed\n", stderr); return 1; }
    tail_drop(tail_seq);

    tail_seq = tail_make(tail_model);
    mark = now();
    for (unsigned t = 0; t < TOKENS; t++)
        if (!tail_run(tail_model, tail_seq, input[t], snapshots, SNAPSHOTS, 0, "", &token, &word)) { fputs("tail failed\n", stderr); return 1; }
    double tail_time = now() - mark;
    tail_drop(tail_seq);

    printf("pod,warm_s,work_s,per_unit_ms,units\n");
    printf("server,%.3f,%.4f,%.2f,%d\n", server_warm, server_time, 1000 * server_time / POSITIONS, POSITIONS);
    printf("layer93,%.3f,%.4f,%.2f,%d\n", tail_warm, tail_time, 1000 * tail_time / TOKENS, TOKENS);
    return 0;
}
