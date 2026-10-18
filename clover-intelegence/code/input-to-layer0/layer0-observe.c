#define _GNU_SOURCE
/* Layer 0 (the server) opened up, stage by stage, over real token ids.
   Pass A runs the server's own stage functions in the server's own order, split only so each
   can be timed, and checks the output against the recorded layer-0 output bit for bit.
   It also checks which stages give identical values when the same token id comes back.
   Pass B computes those token-only stages once per distinct id and reuses them. */
#define SERVER_NO_MAIN
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "numeric-table.h"
#include "server.c"
#include <omp.h>

enum { W = SERVER_WIDTH, R = SERVER_ROWS, NSTAGE = 13, VOCAB = 163840 };

static const char *stage_name[NSTAGE] = {
    "T normalise the input", "T raw Q/K/V projections (3 x 12288 rows)", "P conv + silu + l2 of Q,K",
    "T beta + decay projections", "P recurrent state update + read", "T output-gate projection",
    "P head norm x gate", "P attention output projection", "P aggregate with the original vector",
    "P dense: normalise", "P dense: gate and up projections", "P dense: activation", "P dense: down projection + add"};

typedef struct { float norm[W]; float raw[3][R]; float beta[SERVER_HEADS]; float decay[R]; float gate[R]; } Tok;
typedef struct { double in, attn, wsnap, wres, dense, out, cosio, state; } Obs;

static double lap[NSTAGE];
static float *dump[3]; /* attention branch, aggregate, dense output; set only for pass A */
#define LAP(k, t0) do { double t1_ = omp_get_wtime(); lap[k] += t1_ - t0; t0 = t1_; } while (0)

static double rms(const float *v, size_t n)
{
    double s = 0; for (size_t i = 0; i < n; i++) s += (double)v[i] * v[i];
    return sqrt(s / (double)n);
}

static double cosine(const float *a, const float *b, size_t n)
{
    double d = 0, x = 0, y = 0;
    for (size_t i = 0; i < n; i++) { d += (double)a[i] * b[i]; x += (double)a[i] * a[i]; y += (double)b[i] * b[i]; }
    return d / sqrt(x * y);
}

/* Everything here is a function of the token's own vector only. */
static void token_part(const Server *s, const float *emb, Tok *t)
{
    double t0 = omp_get_wtime();
    server_normalize(t->norm, emb, s->qkv + 88, W);
    LAP(0, t0);
#pragma omp parallel for schedule(static)
    for (unsigned row = 0; row < R; row++)
        for (unsigned c = 0; c < 3; c++) {
            const unsigned char *rec = s->qkv + SERVER_QKV_PREFIX + ((size_t)row * 3 + c) * SERVER_QKV_COMPONENT;
            t->raw[c][row] = server_project_row(t->norm, rec + 20, W, s->palette, server_f32(rec));
        }
    LAP(1, t0);
    float hidden[SERVER_HEAD];
    server_project(s, 11, t->norm, t->beta);
    server_project(s, 12, t->norm, hidden);
    server_project(s, 13, hidden, t->decay);
    for (unsigned head = 0; head < SERVER_HEADS; head++) {
        t->beta[head] = server_sigmoid(t->beta[head]);
        float exponent = server_f32(s->records[39].data + head * 4);
        for (unsigned k = 0; k < SERVER_HEAD; k++) {
            unsigned index = head * SERVER_HEAD + k;
            float value = exponent * (t->decay[index] + server_f32(s->records[18].data + index * 4));
            t->decay[index] = expf(-5.0f * server_sigmoid(value));
        }
    }
    LAP(3, t0);
    server_project(s, 6, t->norm, t->gate);
    LAP(5, t0);
}

static void weights(const Server *s, const float *a, const float *b, double w[2])
{
    const float *src[2] = {a, b};
    float sc[2];
    for (int k = 0; k < 2; k++) {
        float inv = server_inverse_rms(src[k], W);
        double score = 0;
        for (unsigned c = 0; c < W; c++) score += (double)(src[k][c] * inv) * (double)server_f32(s->records[38].data + c * 4);
        sc[k] = (float)score;
    }
    float mx = sc[1] > sc[0] ? sc[1] : sc[0];
    double e0 = expf(sc[0] - mx), e1 = expf(sc[1] - mx);
    w[0] = e0 / (e0 + e1); w[1] = e1 / (e0 + e1);
}

/* Everything here depends on the positions that came before. */
static void position_part(const Server *s, ServerSequence *q, const float *emb, const Tok *t, float *out, Obs *o)
{
    memcpy(q->snapshot, emb, sizeof q->snapshot);
    double t0 = omp_get_wtime();
#pragma omp parallel for schedule(static)
    for (unsigned row = 0; row < R; row++)
        for (unsigned c = 0; c < 3; c++) {
            const unsigned char *rec = s->qkv + SERVER_QKV_PREFIX + ((size_t)row * 3 + c) * SERVER_QKV_COMPONENT;
            float conv = server_convolve(t->raw[c][row], rec + 4, q->history[c][row]);
            q->qkv[c][row] = conv * server_sigmoid(conv);
        }
    server_l2_heads(q->qkv[0]);
    server_l2_heads(q->qkv[1]);
    memcpy(q->beta, t->beta, sizeof q->beta);
    memcpy(q->decay, t->decay, sizeof q->decay);
    LAP(2, t0);
    server_update_attention(q);
    LAP(4, t0);
    memcpy(q->gate, t->gate, sizeof q->gate);
    for (unsigned head = 0; head < SERVER_HEADS; head++) {
        float *block = q->attention + head * SERVER_HEAD;
        server_normalize(block, block, s->records[19].data, SERVER_HEAD);
    }
    for (unsigned c = 0; c < R; c++) q->gate[c] = q->attention[c] * server_sigmoid(q->gate[c]);
    LAP(6, t0);
    server_project(s, 7, q->gate, q->residual);
    LAP(7, t0);
    if (dump[0]) memcpy(dump[0] + q->positions * W, q->residual, W * sizeof(float));
    double w[2] = {0, 0};
    if (o) { o->attn = rms(q->residual, W); weights(s, q->snapshot, q->residual, w); o->wsnap = w[0]; o->wres = w[1]; }
    t0 = omp_get_wtime();
    server_aggregate(s, q);
    LAP(8, t0);
    if (dump[1]) memcpy(dump[1] + q->positions * W, q->aggregate, W * sizeof(float));
    server_normalize(q->postnorm, q->aggregate, s->records[5].data, W);
    LAP(9, t0);
    server_project(s, 26, q->postnorm, q->dense_gate);
    server_project(s, 27, q->postnorm, q->dense_up);
    LAP(10, t0);
    server_dense_activation(q->dense_gate, q->dense_up);
    LAP(11, t0);
    server_project(s, 28, q->dense_gate, q->dense_output);
    if (dump[2]) memcpy(dump[2] + q->positions * W, q->dense_output, W * sizeof(float));
    for (unsigned c = 0; c < W; c++) q->residual[c] = q->residual[c] + q->dense_output[c];
    LAP(12, t0);
    memcpy(out, q->residual, sizeof q->residual);
    q->positions++;
    if (o) {
        o->in = rms(emb, W); o->dense = rms(q->dense_output, W); o->out = rms(out, W);
        o->cosio = cosine(emb, out, W); o->state = rms(&q->recurrent[0][0][0], (size_t)SERVER_HEADS * SERVER_HEAD * SERVER_HEAD);
    }
}

static void report(const char *title, int n)
{
    double total = 0, tok = 0, pos = 0;
    printf("\n%s  (ms per token, over %d tokens; T = depends on the token only, P = depends on earlier positions)\n", title, n);
    for (int k = 0; k < NSTAGE; k++) {
        double ms = lap[k] * 1e3 / n; total += ms;
        if (stage_name[k][0] == 'T') tok += ms; else pos += ms;
    }
    for (int k = 0; k < NSTAGE; k++) {
        double ms = lap[k] * 1e3 / n;
        printf("  %-46s %8.3f ms  %5.1f%%\n", stage_name[k], ms, 100.0 * ms / total);
    }
    printf("  %-46s %8.3f ms   (T stages %.3f, P stages %.3f)\n", "total", total, tok, pos);
}

int main(int argc, char **argv)
{
    if (argc != 6 && argc != 7) { fputs("usage: layer0-observe CODE_DIR IDS_FILE REF_RESID_F32 N SHOW [DUMP_PREFIX]\n", stderr); return 2; }
    const unsigned N = (unsigned)atoi(argv[4]), SHOW = (unsigned)atoi(argv[5]);
    char qkv[4096], trunk[4096], seed[4096];
    snprintf(qkv, sizeof qkv, "%s/server/bin/dataset/trunk-0-qkv/qkv.bin", argv[1]);
    snprintf(trunk, sizeof trunk, "%s/server/bin/dataset/trunk-0", argv[1]);
    snprintf(seed, sizeof seed, "%s/server/bin/dataset/inputs/seed.bin", argv[1]);

    uint32_t *ids = malloc(N * sizeof *ids);
    FILE *f = fopen(argv[2], "r");
    if (!f) { perror("ids"); return 1; }
    unsigned have = 0; unsigned long v;
    while (have < N && fscanf(f, "%lu", &v) == 1) { ids[have++] = (uint32_t)v; int ch = fgetc(f); if (ch != ',' && ch != ' ' && ch != '\n' && ch != EOF) ungetc(ch, f); }
    fclose(f);
    if (have != N) { fprintf(stderr, "ids file has %u ids, wanted %u\n", have, N); return 1; }
    float *ref = malloc((size_t)N * W * sizeof *ref);
    f = fopen(argv[3], "rb");
    if (!f || fread(ref, sizeof *ref, (size_t)N * W, f) != (size_t)N * W) { fputs("cannot read reference\n", stderr); return 1; }
    fclose(f);

    Server *server = NULL;
    if (!server_open(qkv, trunk, &server)) { fputs("server_open failed\n", stderr); return 1; }
    ServerInput *input = server_input_open(seed);
    if (!input) { fputs("server_input_open failed\n", stderr); return 1; }

    int *slot = malloc(VOCAB * sizeof *slot), *count = calloc(VOCAB, sizeof *count);
    for (int i = 0; i < VOCAB; i++) slot[i] = -1;
    unsigned distinct = 0;
    for (unsigned i = 0; i < N; i++) { if (slot[ids[i]] < 0) slot[ids[i]] = (int)distinct++; count[ids[i]]++; }
    printf("layer 0 over %u real tokens: %u distinct ids, %u repeats\n", N, distinct, N - distinct);

    float *emb = malloc((size_t)W * sizeof *emb), *outA = malloc((size_t)N * W * sizeof *outA);
    static Tok tok, scratch;
    Tok *cache = malloc((size_t)distinct * sizeof *cache);
    unsigned char *filled = calloc(distinct, 1);
    Obs *obs = calloc(N, sizeof *obs);

    ServerSequence *q = server_sequence_create(server);
    for (unsigned i = 0; i < 4; i++) {
        server_input(input, ids[i], emb); token_part(server, emb, &tok); position_part(server, q, emb, &tok, outA, NULL);
    }
    server_sequence_reset(q);
    memset(lap, 0, sizeof lap);

    /* Pass A */
    if (argc == 7) for (int k = 0; k < 3; k++) dump[k] = malloc((size_t)N * W * sizeof(float));
    const char *group[7] = {"normalised input", "raw Q", "raw K", "raw V", "beta", "decay", "output-gate projection"};
    unsigned same[7] = {0}, repeats = 0, matches = 0;
    for (unsigned i = 0; i < N; i++) {
        if (!server_input(input, ids[i], emb)) { fputs("lookup failed\n", stderr); return 1; }
        token_part(server, emb, &tok);
        int sl = slot[ids[i]];
        if (!filled[sl]) { memcpy(&cache[sl], &tok, sizeof tok); filled[sl] = 1; }
        else {
            repeats++;
            same[0] += !memcmp(tok.norm, cache[sl].norm, sizeof tok.norm);
            for (int c = 0; c < 3; c++) same[1 + c] += !memcmp(tok.raw[c], cache[sl].raw[c], sizeof tok.raw[c]);
            same[4] += !memcmp(tok.beta, cache[sl].beta, sizeof tok.beta);
            same[5] += !memcmp(tok.decay, cache[sl].decay, sizeof tok.decay);
            same[6] += !memcmp(tok.gate, cache[sl].gate, sizeof tok.gate);
        }
        int watch = i < SHOW || i % 100 == 99;
        position_part(server, q, emb, &tok, outA + (size_t)i * W, watch ? &obs[i] : NULL);
        matches += !memcmp(outA + (size_t)i * W, ref + (size_t)i * W, W * sizeof *ref);
    }
    printf("\nPASS A  output of layer 0 equal to the recorded one, bit for bit: %u of %u positions\n", matches, N);
    if (argc == 7) {
        const char *suffix[3] = {"attn", "agg", "dense"};
        for (int k = 0; k < 3; k++) {
            char path[4096]; snprintf(path, sizeof path, "%s.%s.f32", argv[6], suffix[k]);
            FILE *df = fopen(path, "wb");
            if (!df || fwrite(dump[k], sizeof(float), (size_t)N * W, df) != (size_t)N * W || fclose(df)) { fputs("dump failed\n", stderr); return 1; }
            free(dump[k]); dump[k] = NULL;
        }
    }
    report("PASS A  stage by stage", (int)N);
    printf("\nWhen an id came back (%u repeats), was the stage's result bit-identical to its first occurrence?\n", repeats);
    for (int k = 0; k < 7; k++) printf("  %-26s %u of %u\n", group[k], same[k], repeats);

    printf("\nWhat happens to each token (first %u positions, then every 100th). RMS = typical size of one coordinate.\n", SHOW);
    printf("  pos    id   in_rms attn_rms w(orig) w(attn) dense_rms  out_rms cos(in,out) state_rms\n");
    for (unsigned i = 0; i < N; i++) {
        if (!(i < SHOW || i % 100 == 99)) continue;
        const Obs *o = &obs[i];
        printf("  %4u %6u %8.4f %8.4f %7.3f %7.3f %9.4f %8.4f %10.4f %9.5f\n", i, ids[i], o->in, o->attn, o->wsnap, o->wres,
            o->dense, o->out, o->cosio, o->state);
    }
    int top = 0;
    for (int i = 0; i < VOCAB; i++) if (count[i] > count[top]) top = i;
    printf("\nThe most repeated id is %d (%d times). Same token, same vector in; layer-0 output compared with its first occurrence:\n", top, count[top]);
    const float *first = NULL; unsigned shown = 0;
    for (unsigned i = 0; i < N && shown < 8; i++) {
        if ((int)ids[i] != top) continue;
        const float *o = outA + (size_t)i * W;
        if (!first) first = o;
        printf("  pos %4u  cos(out, first occurrence) = %.6f   out_rms %.4f\n", i, cosine(first, o, W), rms(o, W));
        shown++;
    }

    /* Pass B */
    server_sequence_reset(q);
    memset(lap, 0, sizeof lap);
    memset(filled, 0, distinct);
    float *outB = malloc((size_t)W * sizeof *outB);
    unsigned matchesB = 0, computed = 0;
    for (unsigned i = 0; i < N; i++) {
        server_input(input, ids[i], emb);
        int sl = slot[ids[i]];
        if (!filled[sl]) { token_part(server, emb, &cache[sl]); filled[sl] = 1; computed++; }
        position_part(server, q, emb, &cache[sl], outB, NULL);
        matchesB += !memcmp(outB, ref + (size_t)i * W, W * sizeof *ref);
    }
    printf("\nPASS B  token-only stages computed once per distinct id (%u of %u positions); output equal to the recorded one: %u of %u\n",
        computed, N, matchesB, N);
    report("PASS B  stage by stage", (int)N);

    (void)scratch; (void)group;
    server_input_close(input);
    server_sequence_close(q);
    server_close(server);
    return 0;
}
