#define _GNU_SOURCE
/* Words -> token ids -> embedding vectors -> layer 0 -> layer 1, in one process, as ONE request.

   The three stages are the existing programs (input-stage.c, layer0-batch.c, layer-batch.c), built with
   -DPIPELINE_ENTRY and called through stageN_setup / stageN_run. No data is copied between them: the
   embedding buffer `s0` is written by stage 1, read by stage 2 (as its input) and by stage 3 (as the
   snapshot); the buffer `resid` is written by stage 2 and read by stage 3.

   One-time setup (open files, allocate, tokenise once to size the buffers, one 64-token warm-up pass that
   touches the weights) is timed on its own. Then the request is timed from the words in memory to the layer 1
   output in memory; the token count is whatever the words give.

     pipeline CODE_DIR TOKFILES WORDS REF_S0 REF_RESID REF_OUT

   REF_* are the recorded values for the token ids that WORDS stands for (ids-to-words.c); pass "-" for REF_S0
   when no recorded values exist, and nothing is compared.                                                   */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

double stage1_setup(const char *tokfiles, const char *seed);
int stage1_ids(const char *text, long size, int *ids, int cap);
void stage1_lookup(const unsigned *ids, unsigned n, float *out);
double stage2_setup(const char *code_dir, unsigned max_n);
void stage2_run(const uint32_t *ids, const float *emb, unsigned n, unsigned window, float *out);
double stage3_setup(const char *dataset, unsigned max_n);
void stage3_run(const float *resid, const float *s0, unsigned n, unsigned window, float *out);

enum { W = 7168 };

static double clk(void)
{
    struct timespec m; clock_gettime(CLOCK_MONOTONIC, &m);
    return (double)m.tv_sec + (double)m.tv_nsec / 1e9;
}
static void bail(const char *what) { fprintf(stderr, "pipeline: %s\n", what); exit(1); }

static char *slurp(const char *path, long *size)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "pipeline: cannot read %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); *size = ftell(f); fseek(f, 0, SEEK_SET);
    char *data = malloc((size_t)*size + 1);
    if (!data || fread(data, 1, (size_t)*size, f) != (size_t)*size) bail("read failed");
    data[*size] = 0; fclose(f);
    return data;
}

static unsigned same_rows(const float *a, const float *b, unsigned n)
{
    unsigned ok = 0;
    for (unsigned i = 0; i < n; i++) ok += !memcmp(a + (size_t)i * W, b + (size_t)i * W, W * sizeof(float));
    return ok;
}

int main(int argc, char **argv)
{
    if (argc != 7) { fputs("usage: pipeline CODE_DIR TOKFILES WORDS REF_S0 REF_RESID REF_OUT\n", stderr); return 2; }
    const char *code = argv[1];
    const int compare = strcmp(argv[4], "-") != 0;
    char seed[4096], dataset[4096];
    snprintf(seed, sizeof seed, "%s/server/bin/dataset/inputs/seed.bin", code);
    snprintf(dataset, sizeof dataset, "%s/tansformers/transformer-1/bin/dataset", code);

    long words_size; char *words = slurp(argv[3], &words_size);
    unsigned long word_count = 0; int inside = 0;
    for (long i = 0; i < words_size; i++) {
        const int space = words[i] == ' ' || words[i] == '\n' || words[i] == '\t' || words[i] == '\r';
        if (!space && !inside) word_count++;
        inside = !space;
    }

    /* one-time setup, with a warm-up pass over the first tokens so every weight page has been touched once */
    const double setup0 = clk();
    stage1_setup(argv[2], seed);
    const int cap = (int)words_size + 16;
    int *ids = malloc((size_t)cap * sizeof *ids);
    const int found = stage1_ids(words, words_size, ids, cap);
    if (found <= 0) bail("the words gave no tokens");
    const unsigned N = (unsigned)found;
    float *s0 = malloc((size_t)N * W * sizeof *s0), *resid = malloc((size_t)N * W * sizeof *resid), *out = malloc((size_t)N * W * sizeof *out);
    if (!s0 || !resid || !out) bail("allocation failed");
    float *ref_s0 = NULL, *ref_resid = NULL, *ref_out = NULL;
    if (compare) {
        long l;
        ref_s0 = (float *)slurp(argv[4], &l); if ((size_t)l < (size_t)N * W * 4) bail("REF_S0 too short");
        ref_resid = (float *)slurp(argv[5], &l); if ((size_t)l < (size_t)N * W * 4) bail("REF_RESID too short");
        ref_out = (float *)slurp(argv[6], &l); if ((size_t)l < (size_t)N * W * 4) bail("REF_OUT too short");
    }
    stage2_setup(code, N);
    stage3_setup(dataset, N);
    {
        const unsigned warm = N < 64 ? N : 64;
        stage1_lookup((const unsigned *)ids, warm, s0);
        stage2_run((const uint32_t *)ids, s0, warm, warm, resid);
        stage3_run(resid, s0, warm, warm, out);
    }
    const double setup = clk() - setup0;

    /* the request: words in memory -> layer 1 output in memory */
    const double a = clk();
    const int got = stage1_ids(words, words_size, ids, cap);
    const double b = clk();
    if (got != (int)N) bail("the second tokenisation gave a different token count");
    stage1_lookup((const unsigned *)ids, N, s0);
    const double c = clk();
    stage2_run((const uint32_t *)ids, s0, N, N, resid);
    const double d = clk();
    stage3_run(resid, s0, N, N, out);
    const double e = clk();

    printf("one-time setup (open, allocate, warm-up pass): %.2f s\n", setup);
    printf("REQUEST, words -> layer 1 output: %lu words = %u tokens (%ld bytes): %.1f ms = %.3f s in total (%.3f ms/token)\n",
        word_count, N, words_size, 1000 * (e - a), e - a, 1000 * (e - a) / N);
    printf("   tokenise %.2f ms, embedding lookup %.2f ms, layer 0 %.1f ms, layer 1 %.1f ms\n",
        1000 * (b - a), 1000 * (c - b), 1000 * (d - c), 1000 * (e - d));
    printf("   setup + request: %.2f s\n", setup + (e - a));
    if (compare)
        printf("   against the recorded values: embeddings %u of %u, layer 0 output %u of %u, layer 1 output %u of %u rows bit-identical\n",
            same_rows(s0, ref_s0, N), N, same_rows(resid, ref_resid, N), N, same_rows(out, ref_out, N), N);
    else
        printf("   not compared: no recorded values exist for these words\n");
    return 0;
}
