#define _GNU_SOURCE
#define SERVER_NO_MAIN
/* Layer 0 (the server) for a window of tokens at a time.

   What is shared across tokens: every trunk matrix is streamed once per tile of 32 tokens
   instead of once per token (4 tokens per pass over each weight row, same accumulation order
   per token as server_project_row). The stages that depend only on the token id (input norm,
   raw Q/K/V, beta, decay, output gate) are computed once per DISTINCT id in the window and
   read by every position that uses that id. What stays per position, in order: the 3-tap
   convolution, the recurrent state, and everything after it.

     layer0-batch CODE_DIR IDS_FILE REF_RESID_F32 N WINDOW[,WINDOW...] DEDUP(0|1)           */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "numeric-table.h"
#include "server.c"
#include <omp.h>
#include "../one-layer/trunk-4bit.h"

enum { W = SERVER_WIDTH, R = SERVER_ROWS, H = SERVER_HEADS, D = SERVER_HEAD, DENSE = SERVER_DENSE,
       CHUNK = 32, VOCAB = 163840, LAPS = 16 };

static double clk(void) { return omp_get_wtime(); }
static void bail(const char *what) { fprintf(stderr, "layer0-batch: %s\n", what); exit(1); }
static void *grab(size_t bytes)
{
    void *p = aligned_alloc(64, (bytes + 63) / 64 * 64);
    if (!p) bail("allocation failed");
    memset(p, 0, bytes);
    return p;
}

static double lap_time[LAPS], lap_last;
static const char *lap_name[LAPS];
static void lap(int id, const char *name)
{
    double now = clk();
    lap_time[id] += now - lap_last; lap_name[id] = name; lap_last = now;
}

typedef struct { const unsigned char *ids, *scale; size_t id_stride, scale_stride; unsigned rows, cols; const char *name;
                 const unsigned char *packed_nibbles, *packed_steps; } Mat;   /* the 4-bit copy (USE_4BIT_TRUNK=1), NULL if not packed */
typedef struct { const char *name; unsigned rows, cols; double sec; unsigned long tokens; } MatStat;
static MatStat mat_stats[16];
static unsigned mat_stat_n;
/* Packed copies by slot: record ids 0-47, then the q/k/v components at 48-50. */
static PackedTrunk g_packed_trunk;
enum { PACKED_SLOT_QKV = 48 };

static Mat mat_record(const Server *s, unsigned id, const char *name)
{
    const ServerRecord *r = &s->records[id];
    return (Mat){ r->data + (size_t)r->rows * 4, r->data, r->columns, 4, r->rows, r->columns, name, g_packed_trunk.nibbles[id], g_packed_trunk.steps[id] };
}
static Mat mat_qkv(const Server *s, unsigned component)
{
    static const char *const names[3] = { "q", "k", "v" };
    const unsigned char *base = s->qkv + SERVER_QKV_PREFIX + (size_t)component * SERVER_QKV_COMPONENT;
    return (Mat){ base + 20, base, 3 * (size_t)SERVER_QKV_COMPONENT, 3 * (size_t)SERVER_QKV_COMPONENT,
                  SERVER_ROWS, SERVER_WIDTH, names[component], g_packed_trunk.nibbles[PACKED_SLOT_QKV + component], g_packed_trunk.steps[PACKED_SLOT_QKV + component] };
}
static int palette_identity(const float *palette)
{
    for (int code = 0; code < 256; code++) if (palette[code] != (float)(int8_t)(unsigned char)code) return 0;
    return 1;
}

#define DOTG(NAME, G) \
static inline __attribute__((always_inline)) void NAME(const float *const *x, const unsigned char *ids, \
    unsigned width, const float *palette, int ident, float scale, float *out) \
{ \
    __m256 a0[G], a1[G]; \
    _Pragma("GCC unroll 4") for (int k = 0; k < G; k++) { a0[k] = _mm256_setzero_ps(); a1[k] = _mm256_setzero_ps(); } \
    for (unsigned c = 0; c < width; c += 16) { \
        __m256 w0, w1; \
        if (ident) { \
            w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(ids + c)))); \
            w1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(ids + c + 8)))); \
        } else { \
            w0 = _mm256_i32gather_ps(palette, _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(ids + c))), 4); \
            w1 = _mm256_i32gather_ps(palette, _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(ids + c + 8))), 4); \
        } \
        _Pragma("GCC unroll 4") for (int k = 0; k < G; k++) { \
            a0[k] = _mm256_fmadd_ps(w0, _mm256_loadu_ps(x[k] + c), a0[k]); \
            a1[k] = _mm256_fmadd_ps(w1, _mm256_loadu_ps(x[k] + c + 8), a1[k]); } \
    } \
    _Pragma("GCC unroll 4") for (int k = 0; k < G; k++) { \
        float lanes[16], combined[8]; \
        _mm256_storeu_ps(lanes, a0[k]); _mm256_storeu_ps(lanes + 8, a1[k]); \
        for (unsigned lane = 0; lane < 8; lane++) combined[lane] = lanes[lane] + lanes[lane + 8]; \
        float sum0 = combined[0] + combined[4], sum1 = combined[1] + combined[5]; \
        float sum2 = combined[2] + combined[6], sum3 = combined[3] + combined[7]; \
        out[k] = ((sum0 + sum2) + (sum1 + sum3)) * scale; } \
}
DOTG(dot4, 4)
DOTG(dot2, 2)
DOTG(dot1, 1)

#if defined(__AVX512F__)
/* 4 weight rows x 4 tokens per pass: every activation load feeds 4 multiplies and every converted weight 4.
   One 16-lane accumulator per (row, token): lane l holds the same terms in the same order as lane l of the
   two 8-lane accumulators above, and the reduction after the loop is identical. */
static inline __attribute__((always_inline)) void dotz4x4(const float *const *x, const unsigned char *const *ids,
    unsigned width, const float *scale, float *out)
{
    __m512 a[4][4];
    _Pragma("GCC unroll 4") for (int r = 0; r < 4; r++) _Pragma("GCC unroll 4") for (int g = 0; g < 4; g++) a[r][g] = _mm512_setzero_ps();
    for (unsigned c = 0; c < width; c += 16) {
        __m512 w[4];
#if defined(PFX)
        _Pragma("GCC unroll 4") for (int g = 0; g < 4; g++) _mm_prefetch((const char *)(x[g] + c + PFX), _MM_HINT_T0);
#endif
        _Pragma("GCC unroll 4") for (int r = 0; r < 4; r++)
            w[r] = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(ids[r] + c))));
        _Pragma("GCC unroll 4") for (int r = 0; r < 4; r++) _Pragma("GCC unroll 4") for (int g = 0; g < 4; g++)
            a[r][g] = _mm512_fmadd_ps(w[r], _mm512_loadu_ps(x[g] + c), a[r][g]);
    }
    _Pragma("GCC unroll 4") for (int r = 0; r < 4; r++) _Pragma("GCC unroll 4") for (int g = 0; g < 4; g++) {
        float lanes[16], combined[8];
        _mm512_storeu_ps(lanes, a[r][g]);
        for (unsigned lane = 0; lane < 8; lane++) combined[lane] = lanes[lane] + lanes[lane + 8];
        float sum0 = combined[0] + combined[4], sum1 = combined[1] + combined[5];
        float sum2 = combined[2] + combined[6], sum3 = combined[3] + combined[7];
        out[r * 4 + g] = ((sum0 + sum2) + (sum1 + sum3)) * scale[r];
    }
}
#endif

static const float *g_palette; static int g_ident;
static int g_wide = 1; /* 1: use the 4 rows x 4 tokens kernel where it applies */
static unsigned g_chunk = 32; /* tokens per tile: the weights stream once per tile */
static int g_vecrec = 1; /* 1: recurrent update vectorised across value columns (env L0_VEC) */
static int g_pf; /* 1 / 2: prefetch the next row group's weights into L2 / L1 while the current one is computed (env L0_PF) */

/* Y[t][row] for t in [0, n); token strides xs / ys floats; weights stream once per tile of CHUNK tokens. */
static void bproject(const Mat *m, const float *X, size_t xs, float *Y, size_t ys, unsigned n)
{
    const double began = clk();
#if defined(__AVX512F__)
    if (m->packed_nibbles) {
        project_with_4bit_weights(m->packed_nibbles, m->packed_steps, m->scale, m->scale_stride, m->rows, m->cols, X, xs, Y, ys, n);
        MatStat *stat = NULL;
        for (unsigned i = 0; i < mat_stat_n; i++) if (mat_stats[i].name == m->name) stat = &mat_stats[i];
        if (!stat) { stat = &mat_stats[mat_stat_n++]; stat->name = m->name; stat->rows = m->rows; stat->cols = m->cols; stat->sec = 0; stat->tokens = 0; }
        stat->sec += clk() - began; stat->tokens += n;
        return;
    }
#endif
    for (unsigned c0 = 0; c0 < n; c0 += g_chunk) {
        const unsigned cn = n - c0 < g_chunk ? n - c0 : g_chunk;
#if defined(__AVX512F__)
        if (g_wide && g_ident && m->rows % 4 == 0) {
#pragma omp parallel for schedule(static)
            for (unsigned row = 0; row < m->rows; row += 4) {
                const unsigned char *rid[4]; float sc[4];
                for (int r = 0; r < 4; r++) {
                    rid[r] = m->ids + (size_t)(row + r) * m->id_stride;
                    sc[r] = server_f32(m->scale + (size_t)(row + r) * m->scale_stride);
                }
                unsigned k = 0;
                for (; k + 4 <= cn; k += 4) {
                    const float *x[4]; float out[16];
                    for (int j = 0; j < 4; j++) x[j] = X + (size_t)(c0 + k + j) * xs;
                    dotz4x4(x, rid, m->cols, sc, out);
                    if (g_pf && row + 4 < m->rows) {   /* the next row group's weights, 1/ngroups of them per token pass */
                        const unsigned lpr = m->cols / 64, total = 4 * lpr, ng = cn / 4, gi = k / 4;
                        for (unsigned j = total * gi / ng; j < total * (gi + 1) / ng; j++) {
                            const char *pp = (const char *)(m->ids + (size_t)(row + 4 + j / lpr) * m->id_stride + (size_t)(j % lpr) * 64);
                            if (g_pf == 1) _mm_prefetch(pp, _MM_HINT_T1); else _mm_prefetch(pp, _MM_HINT_T0);
                        }
                    }
                    for (int r = 0; r < 4; r++) for (int j = 0; j < 4; j++) Y[(size_t)(c0 + k + j) * ys + row + r] = out[r * 4 + j];
                }
                for (; k < cn; k++) {
                    const float *x[1] = { X + (size_t)(c0 + k) * xs }; float out[1];
                    for (int r = 0; r < 4; r++) { dot1(x, rid[r], m->cols, g_palette, 1, sc[r], out); Y[(size_t)(c0 + k) * ys + row + r] = out[0]; }
                }
            }
            continue;
        }
#endif
#pragma omp parallel for schedule(static)
        for (unsigned row = 0; row < m->rows; row++) {
            const unsigned char *ids = m->ids + (size_t)row * m->id_stride;
            const float scale = server_f32(m->scale + (size_t)row * m->scale_stride);
            unsigned k = 0;
            while (k < cn) {
                const float *x[4]; float out[4];
                const unsigned g = cn - k >= 4 ? 4 : cn - k >= 2 ? 2 : 1;
                for (unsigned j = 0; j < g; j++) x[j] = X + (size_t)(c0 + k + j) * xs;
                if (g == 4) dot4(x, ids, m->cols, g_palette, g_ident, scale, out);
                else if (g == 2) dot2(x, ids, m->cols, g_palette, g_ident, scale, out);
                else dot1(x, ids, m->cols, g_palette, g_ident, scale, out);
                for (unsigned j = 0; j < g; j++) Y[(size_t)(c0 + k + j) * ys + row] = out[j];
                k += g;
            }
        }
    }
    MatStat *stat = NULL;
    for (unsigned i = 0; i < mat_stat_n; i++) if (mat_stats[i].name == m->name) stat = &mat_stats[i];
    if (!stat) { stat = &mat_stats[mat_stat_n++]; stat->name = m->name; stat->rows = m->rows; stat->cols = m->cols; stat->sec = 0; stat->tokens = 0; }
    stat->sec += clk() - began; stat->tokens += n;
}

/* The server's own aggregate (server_aggregate) over plain arrays. */
static void agg0(const Server *s, const float *snapshot, const float *residual, float *aggregate)
{
    const float *sources[] = {snapshot, residual};
    float scores[2], exponentials[2], weights[2];
    for (unsigned source = 0; source < 2; source++) {
        float inverse = server_inverse_rms(sources[source], W);
        double score = 0.0;
        for (unsigned c = 0; c < W; c++) {
            float normalized = sources[source][c] * inverse;
            score += (double)normalized * (double)server_f32(s->records[38].data + c * 4);
        }
        scores[source] = (float)score;
    }
    float maximum = scores[1] > scores[0] ? scores[1] : scores[0];
    double total = 0.0;
    for (unsigned source = 0; source < 2; source++) {
        exponentials[source] = expf(scores[source] - maximum);
        total += (double)exponentials[source];
    }
    for (unsigned source = 0; source < 2; source++) weights[source] = (float)((double)exponentials[source] / total);
    for (unsigned c = 0; c < W; c++) {
        float value = 0.0f;
        for (unsigned source = 0; source < 2; source++) value = value + weights[source] * sources[source][c];
        aggregate[c] = value;
    }
}

typedef struct {
    float *norm, *raw[3], *beta, *dhid, *decay, *gate;       /* one row per distinct id in the window */
    float *qkv[3], *attention, *gmul, *aout, *postnorm, *dgate, *dup, *dout; /* one row per position */
    unsigned *first; int *uidx;
} Buf;

static Buf buf_make(unsigned w)
{
    Buf b;
    b.norm = grab((size_t)w * W * 4);
    for (int c = 0; c < 3; c++) { b.raw[c] = grab((size_t)w * R * 4); b.qkv[c] = grab((size_t)w * R * 4); }
    b.beta = grab((size_t)w * H * 4); b.dhid = grab((size_t)w * D * 4);
    b.decay = grab((size_t)w * R * 4); b.gate = grab((size_t)w * R * 4);
    b.attention = grab((size_t)w * R * 4); b.gmul = grab((size_t)w * R * 4);
    b.aout = grab((size_t)w * W * 4); b.postnorm = grab((size_t)w * W * 4);
    b.dgate = grab((size_t)w * DENSE * 4); b.dup = grab((size_t)w * DENSE * 4); b.dout = grab((size_t)w * W * 4);
    b.first = grab((size_t)w * sizeof(unsigned)); b.uidx = grab((size_t)w * sizeof(int));
    return b;
}

static int *g_slot;
static int g_keep_state; /* 1: do not reset the convolution history and recurrent state at the start of a call (a decode step) */

#if defined(__AVX512F__)
/* The recurrent update of one head for one token, 16 value columns per instruction. Columns do not depend on each
   other; per column the operations and their order are those of the scalar loops (separate multiply and add, no
   fused multiply-add), so the result is the same bits. */
static inline void recurrent_vec(float (*state)[D], const float *query, const float *key, const float *value,
    const float *decay, float beta, float query_scale, float *output)
{
    __m512 pred[8], diff[8], out[8];
    _Pragma("GCC unroll 8") for (int v = 0; v < 8; v++) pred[v] = _mm512_setzero_ps();
    for (unsigned row = 0; row < D; row++) {
        const __m512 dk = _mm512_set1_ps(decay[row]), kk = _mm512_set1_ps(key[row]);
        float *r = state[row];
        _Pragma("GCC unroll 8") for (int v = 0; v < 8; v++) {
            const __m512 s = _mm512_mul_ps(dk, _mm512_loadu_ps(r + v * 16));
            _mm512_storeu_ps(r + v * 16, s);
            pred[v] = _mm512_add_ps(pred[v], _mm512_mul_ps(s, kk));
        }
    }
    _Pragma("GCC unroll 8") for (int v = 0; v < 8; v++) {
        diff[v] = _mm512_sub_ps(_mm512_loadu_ps(value + v * 16), pred[v]);
        out[v] = _mm512_setzero_ps();
    }
    for (unsigned row = 0; row < D; row++) {
        const __m512 kb = _mm512_set1_ps(key[row] * beta), sq = _mm512_set1_ps(query[row] * query_scale);
        float *r = state[row];
        _Pragma("GCC unroll 8") for (int v = 0; v < 8; v++) {
            const __m512 s = _mm512_add_ps(_mm512_loadu_ps(r + v * 16), _mm512_mul_ps(kb, diff[v]));
            _mm512_storeu_ps(r + v * 16, s);
            out[v] = _mm512_add_ps(out[v], _mm512_mul_ps(s, sq));
        }
    }
    _Pragma("GCC unroll 8") for (int v = 0; v < 8; v++) _mm512_storeu_ps(output + v * 16, out[v]);
}
#endif

static void run_layer0(const Server *s, const uint32_t *ids, const float *emb, unsigned n, unsigned window,
    int dedup, Buf *b, float *out, float (*hist)[R][3], float (*rec)[D][D])
{
    const Mat mq[3] = { mat_qkv(s, 0), mat_qkv(s, 1), mat_qkv(s, 2) };
    const Mat m6 = mat_record(s, 6, "gate"), m7 = mat_record(s, 7, "attention_out"), m11 = mat_record(s, 11, "beta"),
              m12 = mat_record(s, 12, "decay_down"), m13 = mat_record(s, 13, "decay_up"),
              m26 = mat_record(s, 26, "dense_gate"), m27 = mat_record(s, 27, "dense_up"), m28 = mat_record(s, 28, "dense_down");
    if (!g_keep_state) {
        memset(hist, 0, 3 * sizeof(float[R][3]));
        memset(rec, 0, (size_t)H * D * D * sizeof(float));
    }
    for (unsigned p = 0; p < n; p += window) {
        const unsigned w = n - p < window ? n - p : window;
        const float *e = emb + (size_t)p * W;
        lap_last = clk();

        unsigned nu = 0;
        for (unsigned t = 0; t < w; t++) {
            int u = dedup ? g_slot[ids[p + t]] : -1;
            if (u < 0) { u = (int)nu++; b->first[u] = t; if (dedup) g_slot[ids[p + t]] = u; }
            b->uidx[t] = u;
        }
        if (dedup) for (unsigned t = 0; t < w; t++) g_slot[ids[p + t]] = -1;
        lap(0, "find distinct ids");

#pragma omp parallel for schedule(static)
        for (unsigned u = 0; u < nu; u++)
            server_normalize(b->norm + (size_t)u * W, e + (size_t)b->first[u] * W, s->qkv + 88, W);
        lap(1, "T input norm");
        for (int c = 0; c < 3; c++) bproject(&mq[c], b->norm, W, b->raw[c], R, nu);
        lap(2, "T raw Q/K/V projections");
        bproject(&m11, b->norm, W, b->beta, H, nu);
        bproject(&m12, b->norm, W, b->dhid, D, nu);
        bproject(&m13, b->dhid, D, b->decay, R, nu);
#pragma omp parallel for schedule(static)
        for (unsigned u = 0; u < nu; u++) {
            float *beta = b->beta + (size_t)u * H, *decay = b->decay + (size_t)u * R;
            for (unsigned head = 0; head < H; head++) {
                beta[head] = server_sigmoid(beta[head]);
                float exponent = server_f32(s->records[39].data + head * 4);
                for (unsigned k = 0; k < D; k++) {
                    unsigned index = head * D + k;
                    float value = exponent * (decay[index] + server_f32(s->records[18].data + index * 4));
                    decay[index] = expf(-5.0f * server_sigmoid(value));
                }
            }
        }
        lap(3, "T beta + decay");
        bproject(&m6, b->norm, W, b->gate, R, nu);
        lap(4, "T output-gate projection");

#pragma omp parallel for schedule(static)
        for (unsigned blk = 0; blk < R / 16; blk++)
            for (int c = 0; c < 3; c++)
                for (unsigned t = 0; t < w; t++)
                    for (unsigned rr = 0; rr < 16; rr++) {
                        const unsigned row = blk * 16 + rr;
                        const unsigned char *record = s->qkv + SERVER_QKV_PREFIX + ((size_t)row * 3 + c) * SERVER_QKV_COMPONENT;
                        float convolved = server_convolve(b->raw[c][(size_t)b->uidx[t] * R + row], record + 4, hist[c][row]);
                        b->qkv[c][(size_t)t * R + row] = convolved * server_sigmoid(convolved);
                    }
#pragma omp parallel for schedule(static)
        for (unsigned t = 0; t < w; t++) {
            server_l2_heads(b->qkv[0] + (size_t)t * R);
            server_l2_heads(b->qkv[1] + (size_t)t * R);
        }
        lap(5, "P conv + silu + l2");

        const float query_scale = 1.0f / sqrtf(128.0f);
#pragma omp parallel for schedule(static)
        for (unsigned head = 0; head < H; head++)
            for (unsigned t = 0; t < w; t++) {
                float prediction[D] = {0};
                float *output = b->attention + (size_t)t * R + head * D;
                const float *query = b->qkv[0] + (size_t)t * R + head * D;
                const float *key = b->qkv[1] + (size_t)t * R + head * D;
                const float *value = b->qkv[2] + (size_t)t * R + head * D;
                const float *decay = b->decay + (size_t)b->uidx[t] * R + head * D;
                const float beta = b->beta[(size_t)b->uidx[t] * H + head];
#if defined(__AVX512F__)
                if (g_vecrec) { recurrent_vec(rec[head], query, key, value, decay, beta, query_scale, output); continue; }
#endif
                for (unsigned row = 0; row < D; row++)
                    for (unsigned column = 0; column < D; column++) {
                        float state = decay[row] * rec[head][row][column];
                        rec[head][row][column] = state;
                        prediction[column] = prediction[column] + state * key[row];
                    }
                memset(output, 0, D * sizeof *output);
                for (unsigned row = 0; row < D; row++) {
                    float key_beta = key[row] * beta;
                    float scaled_query = query[row] * query_scale;
                    for (unsigned column = 0; column < D; column++) {
                        float state = rec[head][row][column] + key_beta * (value[column] - prediction[column]);
                        rec[head][row][column] = state;
                        output[column] = output[column] + state * scaled_query;
                    }
                }
            }
        lap(6, "P recurrent state update");

#pragma omp parallel for schedule(static)
        for (unsigned t = 0; t < w; t++) {
            float *attention = b->attention + (size_t)t * R, *gm = b->gmul + (size_t)t * R;
            const float *gate = b->gate + (size_t)b->uidx[t] * R;
            for (unsigned head = 0; head < H; head++) {
                float *block = attention + head * D;
                server_normalize(block, block, s->records[19].data, D);
            }
            for (unsigned c = 0; c < R; c++) gm[c] = attention[c] * server_sigmoid(gate[c]);
        }
        lap(7, "P head norm + gating");
        bproject(&m7, b->gmul, R, b->aout, W, w);
        lap(8, "P attention output projection");

#pragma omp parallel for schedule(static)
        for (unsigned t = 0; t < w; t++) {
            float aggregate[W];
            agg0(s, e + (size_t)t * W, b->aout + (size_t)t * W, aggregate);
            server_normalize(b->postnorm + (size_t)t * W, aggregate, s->records[5].data, W);
        }
        lap(9, "P aggregate + dense norm");
        bproject(&m26, b->postnorm, W, b->dgate, DENSE, w);
        bproject(&m27, b->postnorm, W, b->dup, DENSE, w);
        lap(10, "P dense gate + up projections");
#pragma omp parallel for schedule(static)
        for (unsigned t = 0; t < w; t++)
            server_dense_activation(b->dgate + (size_t)t * DENSE, b->dup + (size_t)t * DENSE);
        lap(11, "P dense activation");
        bproject(&m28, b->dgate, DENSE, b->dout, W, w);
#pragma omp parallel for schedule(static)
        for (unsigned t = 0; t < w; t++) {
            float *o = out + (size_t)(p + t) * W;
            for (unsigned c = 0; c < W; c++) o[c] = b->aout[(size_t)t * W + c] + b->dout[(size_t)t * W + c];
        }
        lap(12, "P dense down projection + add");
    }
}

int main(int argc, char **argv)
{
    if (argc != 7) { fputs("usage: layer0-batch CODE_DIR IDS_FILE REF_RESID_F32 N WINDOW[,WINDOW...] DEDUP\n", stderr); return 2; }
    const unsigned N = (unsigned)atoi(argv[4]);
    const int dedup = atoi(argv[6]);
    char qkv[4096], trunk[4096], seed[4096];
    snprintf(qkv, sizeof qkv, "%s/server/bin/dataset/trunk-0-qkv/qkv.bin", argv[1]);
    snprintf(trunk, sizeof trunk, "%s/server/bin/dataset/trunk-0", argv[1]);
    snprintf(seed, sizeof seed, "%s/server/bin/dataset/inputs/seed.bin", argv[1]);

    uint32_t *ids = malloc(N * sizeof *ids);
    FILE *f = fopen(argv[2], "r");
    unsigned have = 0; unsigned long v;
    while (f && have < N && fscanf(f, "%lu", &v) == 1) { ids[have++] = (uint32_t)v; int ch = fgetc(f); if (ch != ',' && ch != ' ' && ch != '\n' && ch != EOF) ungetc(ch, f); }
    if (!f || have != N) bail("ids file problem");
    fclose(f);
    float *ref = malloc((size_t)N * W * sizeof *ref);
    const int self_ref = !strcmp(argv[3], "-"); /* no recorded output for this many tokens: use server_process itself */
    if (!self_ref) {
        f = fopen(argv[3], "rb");
        if (!f || fread(ref, sizeof *ref, (size_t)N * W, f) != (size_t)N * W) bail("cannot read reference");
        fclose(f);
    }

    Server *server = NULL;
    if (!server_open(qkv, trunk, &server)) bail("server_open failed");
    ServerInput *input = server_input_open(seed);
    if (!input) bail("server_input_open failed");
    g_palette = server->palette; g_ident = palette_identity(server->palette);
    if (getenv("L0_WIDE")) g_wide = atoi(getenv("L0_WIDE"));
    if (getenv("L0_VEC")) g_vecrec = atoi(getenv("L0_VEC"));
    if (getenv("L0_PF")) g_pf = atoi(getenv("L0_PF"));
    g_slot = malloc(VOCAB * sizeof *g_slot);
    for (int i = 0; i < VOCAB; i++) g_slot[i] = -1;

    float *emb = malloc((size_t)N * W * sizeof *emb);
    for (unsigned i = 0; i < N; i++) if (!server_input(input, ids[i], emb + (size_t)i * W)) bail("lookup failed");
    unsigned distinct = 0;
    for (unsigned i = 0; i < N; i++) if (g_slot[ids[i]] < 0) g_slot[ids[i]] = (int)distinct++;
    for (unsigned i = 0; i < N; i++) g_slot[ids[i]] = -1;
    printf("layer 0, %u real tokens (%u distinct ids), palette identity: %d, threads: %d, dedup: %d, 4x4 kernel: %d, vector recurrence: %d\n",
        N, distinct, g_ident, omp_get_max_threads(), dedup, g_wide, g_vecrec);

    /* One token at a time through the server's own function: the baseline and a second check. */
    float *outS = malloc((size_t)N * W * sizeof *outS), snapshot[W];
    ServerSequence *q = server_sequence_create(server);
    for (unsigned i = 0; i < 4; i++) server_process(server, q, emb + (size_t)i * W, outS, snapshot);
    server_sequence_reset(q);
    double t0 = clk();
    for (unsigned i = 0; i < N; i++)
        if (!server_process(server, q, emb + (size_t)i * W, outS + (size_t)i * W, snapshot)) bail("server_process failed");
    const double seq_ms = (clk() - t0) * 1e3 / N;
    if (self_ref) memcpy(ref, outS, (size_t)N * W * sizeof *ref);
    unsigned seq_ok = 0;
    for (unsigned i = 0; i < N; i++) seq_ok += !memcmp(outS + (size_t)i * W, ref + (size_t)i * W, W * sizeof *ref);
    printf("one token at a time (server_process): %.3f ms/token, %s at %u of %u positions\n", seq_ms,
        self_ref ? "this output is the reference for the checks below" : "equal to the recorded output", seq_ok, N);

    float (*hist)[R][3] = grab(3 * sizeof(float[R][3]));
    float (*rec)[D][D] = grab((size_t)H * D * D * sizeof(float));
    float *out = malloc((size_t)N * W * sizeof *out);
    unsigned largest = 0;
    for (char *list = strdup(argv[5]), *tok = strtok(list, ","); tok; tok = strtok(NULL, ",")) { unsigned w = (unsigned)atoi(tok); if (w > largest) largest = w; }
    if (largest > N) largest = N;
    Buf b = buf_make(largest);

    {   /* warm-up: touch every weight page and buffer once */
        const unsigned warm = N < 64 ? N : 64;
        run_layer0(server, ids, emb, warm, 32, dedup, &b, out, hist, rec);
    }

    for (char *list = strdup(argv[5]), *tok = strtok(list, ","); tok; tok = strtok(NULL, ",")) {
        char *colon = strchr(tok, ':');
        g_chunk = 32;
        if (colon) { g_chunk = (unsigned)atoi(colon + 1); *colon = 0; }
        unsigned window = (unsigned)atoi(tok);
        if (window > N) window = N;
        memset(lap_time, 0, sizeof lap_time); mat_stat_n = 0;
        memset(out, 0, (size_t)N * W * sizeof *out);
        double start = clk();
        run_layer0(server, ids, emb, N, window, dedup, &b, out, hist, rec);
        double total = clk() - start;
        unsigned ok = 0;
        for (unsigned i = 0; i < N; i++) ok += !memcmp(out + (size_t)i * W, ref + (size_t)i * W, W * sizeof *ref);
        printf("\nwindow %u, weight tile %u tokens: %.3f ms/token (one at a time: %.3f, %.2fx); output equal to the reference at %u of %u positions\n",
            window, g_chunk, total * 1e3 / N, seq_ms, seq_ms / (total * 1e3 / N), ok, N);
        for (int k = 0; k < LAPS; k++) if (lap_name[k] && lap_time[k] > 0)
            printf("  %-36s %8.3f ms/token  %5.1f%%\n", lap_name[k], lap_time[k] * 1e3 / N, 100.0 * lap_time[k] / total);
        printf("  per matrix (all positions run, or distinct ids for T matrices):\n");
        for (unsigned i = 0; i < mat_stat_n; i++) {
            const MatStat *m = &mat_stats[i];
            double gmac = (double)m->rows * m->cols * m->tokens / m->sec / 1e9;
            printf("    %-14s %6u x %-6u %8.3f ms/token over %lu rows of input  %6.1f GMAC/s\n",
                m->name, m->rows, m->cols, m->sec * 1e3 / N, m->tokens, gmac);
        }
        fflush(stdout);
    }
    {   /* the vectorised recurrence against the scalar one: final memory state, convolution history, outputs */
        const unsigned n = N < 2000 ? N : 2000, cw = largest < 1000 ? largest : 1000;
        float *state0 = malloc((size_t)H * D * D * sizeof(float)), *out0 = malloc((size_t)n * W * sizeof(float));
        float (*hist0)[R][3] = malloc(3 * sizeof(float[R][3]));
        const int saved = g_vecrec;
        g_vecrec = 0;
        run_layer0(server, ids, emb, n, cw, dedup, &b, out, hist, rec);
        memcpy(state0, rec, (size_t)H * D * D * sizeof(float)); memcpy(hist0, hist, 3 * sizeof(float[R][3]));
        memcpy(out0, out, (size_t)n * W * sizeof(float));
        g_vecrec = 1;
        run_layer0(server, ids, emb, n, cw, dedup, &b, out, hist, rec);
        printf("\nscalar against vectorised recurrence after %u tokens: memory state %s, convolution history %s, outputs %s\n", n,
            memcmp(state0, rec, (size_t)H * D * D * sizeof(float)) ? "DIFFERENT" : "identical",
            memcmp(hist0, hist, 3 * sizeof(float[R][3])) ? "DIFFERENT" : "identical",
            memcmp(out0, out, (size_t)n * W * sizeof(float)) ? "DIFFERENT" : "identical");
        g_vecrec = saved;
    }
    server_input_close(input);
    return 0;
}

#ifdef PIPELINE_ENTRY
/* Entry points for the one-process pipeline (input-to-layer1/pipeline.c); build with -DPIPELINE_ENTRY -Dmain=layer0_batch_main. */
static Server *p_server;
static Buf p_buf;
static float (*p_hist)[R][3], (*p_rec)[D][D];

double stage2_setup(const char *code_dir, unsigned max_n)
{
    const double t = clk();
    char qkv[4096], trunk[4096];
    snprintf(qkv, sizeof qkv, "%s/server/bin/dataset/trunk-0-qkv/qkv.bin", code_dir);
    snprintf(trunk, sizeof trunk, "%s/server/bin/dataset/trunk-0", code_dir);
    if (!server_open(qkv, trunk, &p_server)) bail("server_open failed");
    g_palette = p_server->palette; g_ident = palette_identity(p_server->palette);
    if (getenv("L0_WIDE")) g_wide = atoi(getenv("L0_WIDE"));
    if (getenv("L0_VEC")) g_vecrec = atoi(getenv("L0_VEC"));
    if (getenv("L0_PF")) g_pf = atoi(getenv("L0_PF"));
    g_slot = malloc(VOCAB * sizeof *g_slot);
    for (int i = 0; i < VOCAB; i++) g_slot[i] = -1;
    p_hist = grab(3 * sizeof(float[R][3]));
    p_rec = grab((size_t)H * D * D * sizeof(float));
    p_buf = buf_make(max_n);
    const char *use_4bit_trunk = getenv("USE_4BIT_TRUNK");
    if (use_4bit_trunk && atoi(use_4bit_trunk) != 0) {   /* qkv, records 6 7 11 12 13, dense 26 27 28 */
        static const unsigned record_ids[] = { 6, 7, 11, 12, 13, 26, 27, 28 };
        for (unsigned component = 0; component < 3; component++) {
            const Mat m = mat_qkv(p_server, component);
            pack_matrix_to_4bit(m.ids, m.id_stride, m.rows, m.cols, &g_packed_trunk.nibbles[PACKED_SLOT_QKV + component], &g_packed_trunk.steps[PACKED_SLOT_QKV + component]);
        }
        for (unsigned i = 0; i < sizeof record_ids / sizeof *record_ids; i++) {
            const Mat m = mat_record(p_server, record_ids[i], "");
            if (m.cols % TRUNK_4BIT_GROUP) bail("4-bit trunk: columns not a multiple of 64");
            pack_matrix_to_4bit(m.ids, m.id_stride, m.rows, m.cols, &g_packed_trunk.nibbles[record_ids[i]], &g_packed_trunk.steps[record_ids[i]]);
        }
    }
    return clk() - t;
}

void stage2_run(const uint32_t *ids, const float *emb, unsigned n, unsigned window, float *out)
{
    g_chunk = 32;
    run_layer0(p_server, ids, emb, n, window, 1, &p_buf, out, p_hist, p_rec);
}

/* One more token after the prompt: the state stays as the prompt left it. */
void stage2_step(const uint32_t *id, const float *emb, float *out)
{
    g_chunk = 32;
    g_keep_state = 1;
    run_layer0(p_server, id, emb, 1, 1, 1, &p_buf, out, p_hist, p_rec);
    g_keep_state = 0;
}

/* n more tokens after the prompt in one pass, on the state the earlier tokens left. */
void stage2_stepk(const uint32_t *ids, const float *emb, unsigned n, float *out)
{
    g_chunk = 32;
    g_keep_state = 1;
    run_layer0(p_server, ids, emb, n, n, 1, &p_buf, out, p_hist, p_rec);
    g_keep_state = 0;
}

/* Copies of the recurrent state and convolution history, so a rejected draft can be undone (speculative decoding). */
static float (*p_saved_hist)[R][3], (*p_saved_rec)[D][D];
void stage2_save_state(void)
{
    if (!p_saved_hist) { p_saved_hist = grab(3 * sizeof(float[R][3])); p_saved_rec = grab((size_t)H * D * D * sizeof(float)); }
    memcpy(p_saved_hist, p_hist, 3 * sizeof(float[R][3]));
    memcpy(p_saved_rec, p_rec, (size_t)H * D * D * sizeof(float));
}
void stage2_restore_state(void)
{
    memcpy(p_hist, p_saved_hist, 3 * sizeof(float[R][3]));
    memcpy(p_rec, p_saved_rec, (size_t)H * D * D * sizeof(float));
}
#endif
