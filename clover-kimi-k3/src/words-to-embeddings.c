#define _GNU_SOURCE
/* Step 1 of the model: input text -> token ids -> embedding vectors. No layer runs.

   text -> ids     the C BPE tokenizer built from tiktoken.model (kimi-k3-in-c), checked
                   against the ids recorded for the 34 prompts in prompts.tsv, and a text
                   round trip; then a large real text, to get many real tokens.
   id -> vector    the server's own embedding lookup (server_input), checked value by value
                   against the S0 the pod chain saved for the same 1000 tokens.

     input-stage TOKFILES_DIR SEED_BIN PROMPTS_TSV BIGTEXT S0_F32 IDS_FILE OUT_IDS          */
#include "k3_tok.h"
#include "numeric-table.h"
#include <math.h>
#include <omp.h>
#include <time.h>

static double clk(void)
{
    struct timespec m; clock_gettime(CLOCK_MONOTONIC, &m);
    return (double)m.tv_sec + (double)m.tv_nsec / 1e9;
}
static void bail(const char *what) { fprintf(stderr, "input-stage: %s\n", what); exit(1); }

static char *slurp(const char *path, long *size)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "input-stage: cannot read %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); *size = ftell(f); fseek(f, 0, SEEK_SET);
    char *data = malloc((size_t)*size + 1);
    if (!data || fread(data, 1, (size_t)*size, f) != (size_t)*size) bail("read failed");
    data[*size] = 0; fclose(f);
    return data;
}

int main(int argc, char **argv)
{
    if (argc != 8) { fputs("usage: input-stage TOKFILES SEED_BIN PROMPTS_TSV BIGTEXT S0_F32 IDS_FILE OUT_IDS\n", stderr); return 2; }
    double t = clk();
    Tok tok; k3_tok_load(&tok, argv[1]);
    printf("1. tokenizer: loaded in %.0f ms (vocabulary %d ids)\n", 1000 * (clk() - t), K3_TOK_VOCAB);

    /* ---- text -> ids on the recorded prompts ---- */
    long tsv_size; char *tsv = slurp(argv[3], &tsv_size);
    int prompts = 0, id_match = 0, round_trip = 0, tokens_checked = 0;
    double encode_s = 0;
    for (char *line = strtok(tsv, "\n"); line; line = strtok(NULL, "\n")) {
        char *f1 = strchr(line, '\t'); if (!f1) continue;
        char *f2 = strchr(f1 + 1, '\t'); if (!f2) continue;
        char *f3 = strchr(f2 + 1, '\t'); if (!f3) continue;
        *f1 = *f2 = *f3 = 0;
        int expected[512], ne = 0;
        for (char *p = f2 + 1; *p && ne < 512; ) { expected[ne++] = (int)strtol(p, &p, 10); if (*p == ',') p++; }
        const char *text = f3 + 1;
        int got[512]; double s = clk();
        int ng = tok_encode(&tok, text, (int)strlen(text), got, 512);
        encode_s += clk() - s;
        char back[4096];
        int bl = tok_decode(&tok, got, ng, back, sizeof back);
        prompts++; tokens_checked += ne;
        if (ng == ne && !memcmp(got, expected, (size_t)ne * sizeof(int))) id_match++;
        else printf("   prompt %d: encoded %d ids, recorded %d\n", prompts, ng, ne);
        if (bl == (int)strlen(text) && !memcmp(back, text, (size_t)bl)) round_trip++;
    }
    printf("2. text -> ids: %d of %d recorded prompts give exactly the recorded ids (%d tokens); %d of %d decode back to the same text; %.3f ms in total\n",
        id_match, prompts, tokens_checked, round_trip, prompts, 1000 * encode_s);

    /* ---- text -> ids on a large real text ---- */
    long text_size; char *text = slurp(argv[4], &text_size);
    int cap = (int)text_size + 16; int *ids = malloc((size_t)cap * sizeof(int));
    t = clk();
    int n = tok_encode(&tok, text, (int)text_size, ids, cap);
    double spent = clk() - t;
    char *back = malloc((size_t)text_size * 4 + 64);
    int bl = tok_decode(&tok, ids, n, back, (int)text_size * 4 + 64);
    unsigned char *seen = calloc(K3_TOK_VOCAB, 1); int distinct = 0;
    for (int i = 0; i < n; i++) if (!seen[ids[i]]++) distinct++;
    printf("3. real text: %ld bytes -> %d tokens (%.2f bytes per token), %d distinct ids, %.1f ms = %.0f tokens/s; decodes back exactly: %s\n",
        text_size, n, (double)text_size / n, distinct, 1000 * spent, n / spent, bl == (int)text_size && !memcmp(back, text, (size_t)bl) ? "yes" : "NO");
    { FILE *o = fopen(argv[7], "w"); if (!o) bail("cannot write ids");
      for (int i = 0; i < n; i++) fprintf(o, "%d%c", ids[i], i + 1 == n ? '\n' : ','); fclose(o); }

    /* ---- id -> vector ---- */
    long idl; char *idtext = slurp(argv[6], &idl);
    unsigned N = 0; static unsigned ids1000[4096];
    for (char *p = idtext; *p && N < 4096; ) { ids1000[N++] = (unsigned)strtol(p, &p, 10); while (*p == ',' || *p == '\n') p++; }
    long s0_size; float *s0 = (float *)slurp(argv[5], &s0_size);
    if ((size_t)s0_size < (size_t)N * 7168 * 4) bail("s0 file shorter than the id list");
    char *real = realpath(argv[2], NULL);
    char direct[4096]; snprintf(direct, sizeof direct, "%s.direct", real ? real : argv[2]);
    int dfd = open(direct, O_RDONLY);
    if (dfd >= 0) { posix_fadvise(dfd, 0, 0, POSIX_FADV_DONTNEED); close(dfd); }
    t = clk();
    ServerInput *input = server_input_open(argv[2]);
    if (!input) bail("server_input_open failed");
    printf("4. embedding table: opened in %.1f ms; %d rows x %d values (bf16, 2 bytes each) = %.2f GB, read in blocks of %d rows (%d bytes)\n",
        1000 * (clk() - t), SERVER_TABLE_ROWS, SERVER_TABLE_WIDTH, (double)SERVER_TABLE_ROWS * SERVER_TABLE_WIDTH * 2 / 1e9,
        SERVER_TABLE_BLOCK_ROWS, CLIENT_RAW_BYTES);
    float (*vec)[7168] = malloc((size_t)N * sizeof *vec);
    double first_total = 0, first_max = 0;
    for (unsigned i = 0; i < N; i++) {
        t = clk();
        if (!server_input(input, ids1000[i], vec[i])) bail("lookup failed");
        double d = clk() - t; first_total += d; if (d > first_max) first_max = d;
    }
    double warm = 1e30;
    for (int round = 0; round < 5; round++) {
        t = clk();
        for (unsigned i = 0; i < N; i++) if (!server_input(input, ids1000[i], vec[i])) bail("lookup failed");
        double d = clk() - t; if (d < warm) warm = d;
    }
    printf("5. lookup, %u tokens: first touch %.3f ms in total (%.4f ms each, slowest %.3f ms); repeated %.3f ms in total (%.5f ms each)\n",
        N, 1000 * first_total, 1000 * first_total / N, 1000 * first_max, 1000 * warm, 1000 * warm / N);
    unsigned exact = 0;
    for (unsigned i = 0; i < N; i++) exact += !memcmp(vec[i], s0 + (size_t)i * 7168, 7168 * sizeof(float));
    printf("6. vectors vs the S0 saved by the pod chain: %u of %u tokens bit-identical in all 7168 values\n", exact, N);
    unsigned pairs = 0, same = 0;
    for (unsigned a = 0; a < N; a++) for (unsigned b = a + 1; b < N; b++)
        if (ids1000[a] == ids1000[b]) { pairs++; same += !memcmp(vec[a], vec[b], 7168 * sizeof(float)); }
    double nmin = 1e30, nmax = 0, nsum = 0, cmin = 2, cmax = -2, csum = 0; unsigned cn = 0;
    for (unsigned i = 0; i < N; i++) {
        double q = 0; for (int c = 0; c < 7168; c++) q += (double)vec[i][c] * vec[i][c];
        q = sqrt(q); if (q < nmin) nmin = q; if (q > nmax) nmax = q; nsum += q;
    }
    for (unsigned i = 0; i + 1 < N; i++) if (ids1000[i] != ids1000[i + 1]) {
        double dot = 0, na = 0, nb = 0;
        for (int c = 0; c < 7168; c++) { dot += (double)vec[i][c] * vec[i + 1][c]; na += (double)vec[i][c] * vec[i][c]; nb += (double)vec[i + 1][c] * vec[i + 1][c]; }
        double cs = dot / sqrt(na * nb); if (cs < cmin) cmin = cs; if (cs > cmax) cmax = cs; csum += cs; cn++;
    }
    printf("7. same id -> same vector: %u of %u position pairs that share an id are bit-identical\n", same, pairs);
    printf("8. the vectors: length (L2 norm) min %.3f mean %.3f max %.3f; neighbours with different ids: cosine min %.3f mean %.3f max %.3f\n",
        nmin, nsum / N, nmax, cmin, csum / cn, cmax);
    server_input_close(input);
    return 0;
}

#ifdef PIPELINE_ENTRY
/* Entry points for the one-process pipeline (input-to-layer1/pipeline.c); build with -DPIPELINE_ENTRY -Dmain=input_stage_main. */
static Tok p_tok;
static ServerInput *p_input;

double stage1_setup(const char *tokfiles, const char *seed)
{
    const double t = clk();
    k3_tok_load(&p_tok, tokfiles);
    char *real = realpath(seed, NULL);
    char direct[4096]; snprintf(direct, sizeof direct, "%s.direct", real ? real : seed);
    int dfd = open(direct, O_RDONLY);
    if (dfd >= 0) { posix_fadvise(dfd, 0, 0, POSIX_FADV_DONTNEED); close(dfd); }
    p_input = server_input_open(seed);
    if (!p_input) bail("server_input_open failed");
    return clk() - t;
}

int stage1_ids(const char *text, long size, int *ids, int cap) { return tok_encode(&p_tok, text, (int)size, ids, cap); }
/* Plain text only: a spelling such as "<|end_of_msg|>" in it stays ordinary text (the chat template builds its control tokens itself). */
int stage1_ids_plain(const char *text, long size, int *ids, int cap) { return tok_encode_mode(&p_tok, text, (int)size, ids, cap, 0); }
int stage1_id_of(const char *name) { return tok_id_of(&p_tok, name); }

void stage1_lookup(const unsigned *ids, unsigned n, float *out)
{
    for (unsigned i = 0; i < n; i++) if (!server_input(p_input, ids[i], out + (size_t)i * 7168)) bail("lookup failed");
}
#endif
