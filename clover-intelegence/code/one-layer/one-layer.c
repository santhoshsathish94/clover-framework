#define _GNU_SOURCE
/* One layer, everything resident, many prompts.
   Embedding + layer 0 (the server) turn each prompt into real layer-1 input. Layer 1
   then runs over every position of every prompt with its experts locked in RAM (pin),
   evicted first (cold) or left as found (warm). Only the layer-1 stage is timed for
   the layer figure; the server stage is reported separately. */
#include "ends.h"
#include <dlfcn.h>
#include <fcntl.h>
#include <math.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

enum { OL_W = 7168, OL_MAX_PROMPTS = 2048, OL_MAX_LEN = 4096 };

typedef int (*OlServerOpen)(const char *, const char *, void **);
typedef int (*OlServerProcess)(const void *, void *, const float *, float *, float *);
typedef int (*OlLayerOpen)(const char *, void **);
typedef int (*OlLayerOne)(void *, void *, const float *, const float *, float *);
typedef void *(*OlCreate)(const void *);
typedef void (*OlClose)(void *);

static double ol_clock(void)
{
    struct timespec m; clock_gettime(CLOCK_MONOTONIC, &m);
    return (double)m.tv_sec + (double)m.tv_nsec / 1e9;
}

static void ol_die(const char *what) { fprintf(stderr, "one-layer: %s\n", what); exit(1); }

static void *ol_need(void *library, const char *name)
{
    void *symbol = dlsym(library, name);
    if (!symbol) { fprintf(stderr, "one-layer: missing symbol %s\n", name); exit(1); }
    return symbol;
}

static unsigned long long ol_read_bytes(void)
{
    FILE *file = fopen("/proc/self/io", "r");
    if (!file) return 0;
    char label[64]; unsigned long long value = 0, found = 0;
    while (fscanf(file, "%63s %llu", label, &value) == 2)
        if (!strcmp(label, "read_bytes:")) { found = value; break; }
    fclose(file);
    return found;
}

static unsigned ol_parse(const char *line, uint32_t *ids)
{
    unsigned n = 0; char *end;
    while (*line && n < OL_MAX_LEN) {
        while (*line == ',' || *line == ' ') line++;
        if (*line == '\n' || *line == '\r' || !*line) break;
        ids[n++] = (uint32_t)strtoul(line, &end, 10);
        if (end == line) break;
        line = end;
    }
    while (*line == ',' || *line == ' ') line++;
    if (*line && *line != '\n' && *line != '\r') ol_die("input longer than OL_MAX_LEN or malformed");
    return n;
}

int main(int argc, char **argv)
{
    if (argc != 4 && argc != 5) { fputs("usage: one-layer CODE_DIR IDS_FILE pin|cold|warm [DUMP_PREFIX]\n", stderr); return 2; }
    const char *code = argv[1], *mode = argv[3];
    if (strcmp(mode, "pin") && strcmp(mode, "cold") && strcmp(mode, "warm")) ol_die("mode must be pin, cold or warm");

    static uint32_t ids[OL_MAX_PROMPTS][OL_MAX_LEN];
    static unsigned lengths[OL_MAX_PROMPTS];
    unsigned prompts = 0; size_t positions = 0;
    FILE *list = fopen(argv[2], "r");
    if (!list) ol_die("cannot read ids file");
    char line[OL_MAX_LEN * 8];
    while (prompts < OL_MAX_PROMPTS && fgets(line, sizeof line, list)) {
        size_t text = strlen(line);
        if (text == sizeof line - 1 && line[text - 1] != '\n') ol_die("ids line too long");
        unsigned n = ol_parse(line, ids[prompts]);
        if (!n) continue;
        lengths[prompts++] = n; positions += n;
    }
    fclose(list);
    if (!prompts) ol_die("no prompts");

    char server_so[4096], qkv[4096], trunk[4096], seed[4096], layer_so[4096], layer_data[4096], experts[4096];
    snprintf(server_so, sizeof server_so, "%s/server/bin/pipeline-stage.so", code);
    snprintf(qkv, sizeof qkv, "%s/server/bin/dataset/trunk-0-qkv/qkv.bin", code);
    snprintf(trunk, sizeof trunk, "%s/server/bin/dataset/trunk-0", code);
    snprintf(seed, sizeof seed, "%s/server/bin/dataset/inputs/seed.bin", code);
    snprintf(layer_so, sizeof layer_so, "%s/tansformers/transformer-1/bin/pipeline-stage.so", code);
    snprintf(layer_data, sizeof layer_data, "%s/tansformers/transformer-1/bin/dataset", code);
    snprintf(experts, sizeof experts, "%s/tansformers/transformer-1/bin/dataset/root-1/experts.direct", code);

    int handle = open(experts, O_RDONLY);
    struct stat info;
    if (handle < 0 || fstat(handle, &info)) ol_die("cannot open experts.direct");
    size_t expert_bytes = (size_t)info.st_size;
    /* Evict before any mapping of the file exists, or mapped pages would survive it. */
    if (!strcmp(mode, "cold")) posix_fadvise(handle, 0, 0, POSIX_FADV_DONTNEED);

    void *server_lib = dlopen(server_so, RTLD_NOW | RTLD_LOCAL);
    if (!server_lib) { fprintf(stderr, "one-layer: %s\n", dlerror()); return 1; }
    void *layer_lib = dlopen(layer_so, RTLD_NOW | RTLD_LOCAL);
    if (!layer_lib) { fprintf(stderr, "one-layer: %s\n", dlerror()); return 1; }
    OlServerOpen server_open_fn = (OlServerOpen)ol_need(server_lib, "server_open");
    OlServerProcess server_process_fn = (OlServerProcess)ol_need(server_lib, "server_process");
    OlCreate server_create = (OlCreate)ol_need(server_lib, "server_sequence_create");
    OlClose server_drop = (OlClose)ol_need(server_lib, "server_sequence_close");
    OlClose server_shut = (OlClose)ol_need(server_lib, "server_close");
    OlLayerOpen layer_open_fn = (OlLayerOpen)ol_need(layer_lib, "transformer_open");
    OlLayerOne layer_process_fn = (OlLayerOne)ol_need(layer_lib, "transformer_process");
    OlCreate layer_create = (OlCreate)ol_need(layer_lib, "transformer_sequence_create");
    OlClose layer_drop = (OlClose)ol_need(layer_lib, "transformer_sequence_close");
    OlClose layer_shut = (OlClose)ol_need(layer_lib, "transformer_close");

    void *server_model = NULL, *layer_model = NULL;
    if (!server_open_fn(qkv, trunk, &server_model)) ol_die("server_open failed");
    if (!layer_open_fn(layer_data, &layer_model)) ol_die("transformer_open failed");
    ServerInput *input = server_input_open(seed);
    if (!input) ol_die("server_input_open failed");

    float *resid = malloc(positions * OL_W * sizeof(float));
    float *s0 = malloc(positions * OL_W * sizeof(float));
    float *out = malloc(positions * OL_W * sizeof(float));
    float *embedding = malloc(OL_W * sizeof(float));
    double *per_prompt = calloc(prompts, sizeof(double));
    if (!resid || !s0 || !out || !embedding || !per_prompt) ol_die("allocation failed");

    /* Stage A: embedding and layer 0 for every prompt, so layer 1 sees real input. */
    double mark = ol_clock();
    size_t at = 0;
    for (unsigned p = 0; p < prompts; p++) {
        void *sequence = server_create(server_model);
        if (!sequence) ol_die("server sequence failed");
        for (unsigned k = 0; k < lengths[p]; k++, at++) {
            if (!server_input(input, ids[p][k], embedding)) ol_die("embedding lookup failed");
            if (!server_process_fn(server_model, sequence, embedding, resid + at * OL_W, s0 + at * OL_W))
                ol_die("server_process failed");
        }
        server_drop(sequence);
    }
    double server_s = ol_clock() - mark;
    if (argc == 5) {
        char path[4096];
        snprintf(path, sizeof path, "%s.resid.f32", argv[4]);
        FILE *dump = fopen(path, "wb");
        if (!dump || fwrite(resid, sizeof(float), positions * OL_W, dump) != positions * OL_W || fclose(dump)) ol_die("cannot write residual dump");
        snprintf(path, sizeof path, "%s.s0.f32", argv[4]);
        dump = fopen(path, "wb");
        if (!dump || fwrite(s0, sizeof(float), positions * OL_W, dump) != positions * OL_W || fclose(dump)) ol_die("cannot write s0 dump");
        printf("dumped %zu layer-1 inputs to %s.{resid,s0}.f32\n", positions, argv[4]);
        return 0;
    }

    double pin_s = 0.0, pin_gb = 0.0;
    if (!strcmp(mode, "pin")) {
        void *mapped = mmap(NULL, expert_bytes, PROT_READ, MAP_SHARED, handle, 0);
        if (mapped == MAP_FAILED) ol_die("map failed");
        mark = ol_clock();
        if (mlock(mapped, expert_bytes)) { perror("mlock"); return 1; }
        pin_s = ol_clock() - mark;
        pin_gb = (double)expert_bytes / 1e9;
    }

    /* Stage B: layer 1 over every position of every prompt. */
    struct rusage before, after;
    getrusage(RUSAGE_SELF, &before);
    unsigned long long io_before = ol_read_bytes();
    mark = ol_clock();
    at = 0;
    for (unsigned p = 0; p < prompts; p++) {
        void *sequence = layer_create(layer_model);
        if (!sequence) ol_die("layer sequence failed");
        double start = ol_clock();
        for (unsigned k = 0; k < lengths[p]; k++, at++)
            if (!layer_process_fn(layer_model, sequence, resid + at * OL_W, s0 + at * OL_W, out + at * OL_W)) {
                fprintf(stderr, "one-layer: layer 1 failed at prompt %u position %u\n", p, k);
                return 1;
            }
        per_prompt[p] = ol_clock() - start;
        layer_drop(sequence);
    }
    double layer_s = ol_clock() - mark;
    unsigned long long io_after = ol_read_bytes();
    getrusage(RUSAGE_SELF, &after);

    uint64_t hash = 1469598103934665603ULL;
    size_t nonfinite = 0;
    const unsigned char *bytes = (const unsigned char *)out;
    for (size_t i = 0; i < positions * OL_W * sizeof(float); i++) { hash ^= bytes[i]; hash *= 1099511628211ULL; }
    for (size_t i = 0; i < positions * OL_W; i++) if (!isfinite(out[i])) nonfinite++;

    printf("layer 1  mode %-4s  prompts %u  positions %zu\n", mode, prompts, positions);
    printf("  experts pinned     %.2f GB in %.2f s\n", pin_gb, pin_s);
    printf("  server (layer 0)   %9.3f s  %8.3f ms/position\n", server_s, 1000 * server_s / (double)positions);
    printf("  layer 1            %9.3f s  %8.3f ms/position\n", layer_s, 1000 * layer_s / (double)positions);
    if (prompts > 1)
        printf("  first prompt %.3f s (%u pos); rest %.3f ms/position\n", per_prompt[0], lengths[0],
            1000 * (layer_s - per_prompt[0]) / (double)(positions - lengths[0]));
    printf("  layer-1 stage read %.3f GiB from device, major faults %ld\n",
        (double)(io_after - io_before) / 1073741824.0, after.ru_majflt - before.ru_majflt);
    printf("  output fnv1a %016llx  nonfinite %zu\n", (unsigned long long)hash, nonfinite);
    if (prompts <= 12)
        for (unsigned p = 0; p < prompts; p++) printf("    prompt %2u  %3u pos  %8.3f s\n", p + 1, lengths[p], per_prompt[p]);
    fflush(stdout);

    server_input_close(input);
    layer_shut(layer_model);
    server_shut(server_model);
    return 0;
}
