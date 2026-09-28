/* eq.c - k3-model-equation.md, sections 2 to 5, in C.
 *
 * Standalone. No cache manager, no prefetcher, no scheduler. It reads the
 * weights, evaluates the equation for one prompt, and prints the token.
 *
 * Metadata comes from eqidx.bin, produced by dump_eqidx.py, because parsing
 * two JSON indexes and 93 safetensors headers is not part of the equation.
 *
 * build: gcc -O2 -march=native -ffp-contract=off -fopenmp eq.c -o eq -lm
 *
 * -ffp-contract=off is required. Without it the compiler fuses a*b+c into a
 * single-rounded FMA in places the engine rounds twice.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ---------------------------------------------------------------- 1.2 shapes */
#define E      7168
#define NPOS   5
#define H      96
#define D      128
#define P      12288
#define KC     4
#define VOCAB  163840
#define NLAY   93
#define LAT    3584
#define I_     3072
#define SI     6144
#define DI     33792
#define NEXP   896
#define TOPK   16
#define GRP    32
#define QN     128
#define QR     64
#define QH     192
#define VH     128
#define KVL    512
#define KVW    576
#define KVD    256
#define QLORA  1536

/* ---------------------------------------------------------------- 1.3 scalars */
static const double EPS5 = (double)1.0e-5f;
static const double EPS6 = (double)1.0e-6f;
static const float  LAM  = -5.0f;
static const float  B1C  = 4.0f;
static const float  B2C  = 25.0f;

/* ---------------------------------------------------------------- index */
typedef struct { int present, dtype; int64_t off, nbytes; int d0, d1; } Slot;
typedef struct { int file_id; int64_t off, nbytes; int d0, d1, dtype; } MRec;
typedef struct { int layer, expert, which, kind, file_id; int64_t off, nbytes; int d0, d1; } ERec;

static int   n_slots;
static Slot *slots;            /* [NLAY * n_slots] */
static MRec  mrec[8];
static int   n_model;
static char  fpath[128][512];
static int   n_files;
static ERec *erec;
static int   n_erec;

static unsigned char *trunk;   /* mmap of trunk.bin */
static unsigned char *fmap[128];
static size_t          fsize[128];

enum {
    S_ARN = 0, S_ARP, S_MRN, S_MRP, S_IN_LN, S_POST_LN, S_G, S_O,
    S_Q, S_K, S_V, S_B, S_FA, S_FB, S_CQ, S_CK, S_CV, S_ALOG, S_DTB, S_ONORM,
    S_QA, S_QAN, S_QB, S_KA, S_KAN, S_KB,
    S_MGATE, S_MUP, S_MDOWN,
    S_GATE, S_GBIAS, S_EDOWN, S_EUP, S_ENORM, S_SH1, S_SH3, S_SH2
};

static double now_s(void)
{
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static void die(const char *m) { fprintf(stderr, "eq: %s\n", m); exit(1); }

static unsigned char *map_file(const char *path, size_t *sz)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) die(path);
    struct stat st;
    if (fstat(fd, &st)) die("fstat");
    void *p = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE | MAP_NORESERVE, fd, 0);
    close(fd);
    if (p == MAP_FAILED) die("mmap");
    *sz = st.st_size;
    return (unsigned char *)p;
}

static unsigned char *file_ptr(int id)
{
    if (!fmap[id]) fmap[id] = map_file(fpath[id], &fsize[id]);
    return fmap[id];
}

/* read the flat index, field by field: the on-disk records are packed and a
   C struct of the same fields is not */
static void load_index(const char *path)
{
    size_t sz; unsigned char *b = map_file(path, &sz);
    size_t c = 0;
    if (memcmp(b, "K3EQ", 4)) die("bad index magic");
    c = 4;
    int nl; memcpy(&nl, b + c, 4); c += 4;
    memcpy(&n_slots, b + c, 4); c += 4;
    if (nl != NLAY) die("layer count");
    slots = calloc((size_t)NLAY * n_slots, sizeof(Slot));
    for (int i = 0; i < NLAY * n_slots; i++) {
        memcpy(&slots[i].present, b + c, 4); c += 4;
        memcpy(&slots[i].dtype,   b + c, 4); c += 4;
        memcpy(&slots[i].off,     b + c, 8); c += 8;
        memcpy(&slots[i].nbytes,  b + c, 8); c += 8;
        memcpy(&slots[i].d0,      b + c, 4); c += 4;
        memcpy(&slots[i].d1,      b + c, 4); c += 4;
    }
    memcpy(&n_model, b + c, 4); c += 4;
    for (int i = 0; i < n_model; i++) {
        memcpy(&mrec[i].file_id, b + c, 4); c += 4;
        memcpy(&mrec[i].off,     b + c, 8); c += 8;
        memcpy(&mrec[i].nbytes,  b + c, 8); c += 8;
        memcpy(&mrec[i].d0,      b + c, 4); c += 4;
        memcpy(&mrec[i].d1,      b + c, 4); c += 4;
        memcpy(&mrec[i].dtype,   b + c, 4); c += 4;
    }
    memcpy(&n_files, b + c, 4); c += 4;
    for (int i = 0; i < n_files; i++) {
        int len; memcpy(&len, b + c, 4); c += 4;
        memcpy(fpath[i], b + c, len); fpath[i][len] = 0; c += len;
    }
    memcpy(&n_erec, b + c, 4); c += 4;
    erec = calloc(n_erec, sizeof(ERec));
    for (int i = 0; i < n_erec; i++) {
        memcpy(&erec[i].layer,   b + c, 4); c += 4;
        memcpy(&erec[i].expert,  b + c, 4); c += 4;
        memcpy(&erec[i].which,   b + c, 4); c += 4;
        memcpy(&erec[i].kind,    b + c, 4); c += 4;
        memcpy(&erec[i].file_id, b + c, 4); c += 4;
        memcpy(&erec[i].off,     b + c, 8); c += 8;
        memcpy(&erec[i].nbytes,  b + c, 8); c += 8;
        memcpy(&erec[i].d0,      b + c, 4); c += 4;
        memcpy(&erec[i].d1,      b + c, 4); c += 4;
    }
}

static const ERec *expert_rec(int L, int e, int which, int kind)
{
    long idx = (long)(L - 1) * (NEXP * 6) + (long)e * 6 + which * 2 + kind;
    const ERec *r = &erec[idx];
    if (r->layer != L || r->expert != e || r->which != which || r->kind != kind)
        die("expert index layout");
    return r;
}

static const unsigned char *slot_ptr(int L, int s)
{
    Slot *sl = &slots[(size_t)L * n_slots + s];
    if (!sl->present) die("absent slot");
    return trunk + sl->off;
}

/* dequantize any slot to float32. dtype 0=F32 1=BF16 2=I8R */
static float *slot_vec(int L, int s, int want)
{
    Slot *sl = &slots[(size_t)L * n_slots + s];
    if (!sl->present) die("absent slot vec");
    const unsigned char *p = trunk + sl->off;
    float *out = malloc(sizeof(float) * (size_t)want);
    if (sl->dtype == 0) {
        memcpy(out, p, sizeof(float) * (size_t)want);
    } else if (sl->dtype == 1) {
        const uint16_t *h = (const uint16_t *)p;
        for (int i = 0; i < want; i++) {
            uint32_t u = ((uint32_t)h[i]) << 16;
            float f; memcpy(&f, &u, 4); out[i] = f;
        }
    } else {
        int rows = (int)((sl->nbytes - want) / 4);
        int cols = want / rows;
        for (int r = 0; r < rows; r++) {
            const unsigned char *row = p + (size_t)r * (4 + cols);
            float sc; memcpy(&sc, row, 4);
            const int8_t *w = (const int8_t *)(row + 4);
            for (int i = 0; i < cols; i++) out[r * cols + i] = (float)w[i] * sc;
        }
    }
    return out;
}

/* ================================================================ section 2 */

/* Q: int8 projection, sixteen float32 lanes then the fixed tree */
static void Q(float *y, const float *x, const unsigned char *W, int in, int out)
{
    const size_t rowb = (size_t)4 + (size_t)in;
#pragma omp parallel for schedule(static) if (out > 64)
    for (int o = 0; o < out; o++) {
        const unsigned char *row = W + (size_t)o * rowb;
        float scale; memcpy(&scale, row, 4);
        const int8_t *w = (const int8_t *)(row + 4);
        int i = 0; float acc;
#if defined(__AVX2__)
        __m256 v0 = _mm256_setzero_ps(), v1 = _mm256_setzero_ps();
        for (; i + 15 < in; i += 16) {
            const __m128i b0 = _mm_loadl_epi64((const __m128i *)(w + i));
            const __m128i b1 = _mm_loadl_epi64((const __m128i *)(w + i + 8));
            v0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b0)),
                                 _mm256_loadu_ps(x + i), v0);
            v1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b1)),
                                 _mm256_loadu_ps(x + i + 8), v1);
        }
        __m256 vs = _mm256_add_ps(v0, v1);
        __m128 lo = _mm_add_ps(_mm256_castps256_ps128(vs), _mm256_extractf128_ps(vs, 1));
        lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
        lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
        acc = _mm_cvtss_f32(lo);
#else
        float B[16] = {0};
        for (; i + 15 < in; i += 16)
            for (int c = 0; c < 16; c++) B[c] = fmaf((float)w[i + c], x[i + c], B[c]);
        float A[8];
        for (int j = 0; j < 8; j++) A[j] = B[j] + B[j + 8];
        float l0 = A[0] + A[4], l1 = A[1] + A[5], l2 = A[2] + A[6], l3 = A[3] + A[7];
        acc = (l0 + l2) + (l1 + l3);
#endif
        for (; i < in; i++) acc = acc + (float)w[i] * x[i];
        y[o] = acc * scale;
    }
}

/* X: MXFP4 projection, four 4-lane double accumulators */
static const float E2M1[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
                               -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

static void X(float *y, const float *x, const unsigned char *pk,
              const unsigned char *sc, int inn, int rows)
{
    const int ngrp = inn / GRP;
    const int cpg  = GRP >> 4;
#pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        const unsigned char *pr = pk + (size_t)r * (inn / 2);
        const unsigned char *sr = sc + (size_t)r * ngrp;
        double v[4][4] = {{0}};
        int g = 0, c = 0;
        for (int i = 0; i < inn; i += 16) {
            float e8 = (sr[g] == 255) ? 0.0f : exp2f((float)sr[g] - 127.0f);
            float wd[16];
            for (int b = 0; b < 8; b++) {
                unsigned char by = pr[(i >> 1) + b];
                wd[2 * b]     = E2M1[by & 0x0F] * e8;
                wd[2 * b + 1] = E2M1[by >> 4]   * e8;
            }
            for (int m = 0; m < 4; m++)
                for (int k = 0; k < 4; k++)
                    v[m][k] += (double)wd[m * 4 + k] * (double)x[i + m * 4 + k];
            if (++c == cpg) { c = 0; g++; }
        }
        double q[4];
        for (int k = 0; k < 4; k++) q[k] = (v[0][k] + v[2][k]) + (v[1][k] + v[3][k]);
        double t0 = q[0] + q[2], t1 = q[1] + q[3];
        y[r] = (float)(t0 + t1);
    }
}

/* Bf: BF16 projection, sixteen double lanes then the fixed tree */
static void Bf(float *y, const float *x, const uint16_t *W, int rows, int in)
{
#pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        const uint16_t *w = W + (size_t)r * in;
        double C[16] = {0};
        for (int i = 0; i < in; i += 16)
            for (int c = 0; c < 16; c++) {
                uint32_t u = ((uint32_t)w[i + c]) << 16;
                float f; memcpy(&f, &u, 4);
                C[c] += (double)f * (double)x[i + c];
            }
        double U[4];
        for (int c = 0; c < 4; c++) U[c] = (C[c] + C[c + 4]) + (C[c + 8] + C[c + 12]);
        y[r] = (float)((U[0] + U[1]) + (U[2] + U[3]));
    }
}

static void rmsnorm(float *y, const float *x, const float *w, int n, double eps)
{
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
    const float inv = (float)(1.0 / sqrt(ss / (double)n + eps));
    for (int i = 0; i < n; i++) y[i] = (w[i] * x[i]) * inv;
}

static void rmsnorm_blocks(float *y, const float *x, const float *w, int nb, int blk)
{
    for (int b = 0; b < nb; b++) {
        const float *xs = x + (size_t)b * blk;
        double ss = 0.0;
        for (int i = 0; i < blk; i++) ss += (double)xs[i] * (double)xs[i];
        const float inv = (float)(1.0 / sqrt(ss / (double)blk + EPS5));
        for (int i = 0; i < blk; i++) y[b * blk + i] = (w[i] * xs[i]) * inv;
    }
}

static void l2_blocks(float *y, const float *x, int nb, int blk)
{
    for (int b = 0; b < nb; b++) {
        const float *xs = x + (size_t)b * blk;
        double ss = 0.0;
        for (int i = 0; i < blk; i++) ss += (double)xs[i] * (double)xs[i];
        const float inv = (float)(1.0 / sqrt(ss + EPS6));
        for (int i = 0; i < blk; i++) y[b * blk + i] = xs[i] * inv;
    }
}

static inline float sigf(float x) { return 1.0f / (1.0f + expf(-x)); }

static void situ(float *y, const float *g, const float *u, int n)
{
    for (int i = 0; i < n; i++) {
        float a  = (B1C * tanhf(g[i] / B1C)) * sigf(g[i]);
        float uu = B2C * tanhf(u[i] / B2C);
        y[i] = a * uu;
    }
}

/* AR: snapshot aggregation, float32 source-major weighted sum */
static void AR(float *out, float *const *srcs, int nsrc, const float *fold)
{
    float sc[16];
    for (int s = 0; s < nsrc; s++) {
        const float *v = srcs[s];
        double ss = 0.0;
        for (int i = 0; i < E; i++) ss += (double)v[i] * (double)v[i];
        const float inv = (float)(1.0 / sqrt(ss / (double)E + EPS5));
        double acc = 0.0;
        for (int i = 0; i < E; i++) {
            float pr = v[i] * inv;
            acc += (double)pr * (double)fold[i];
        }
        sc[s] = (float)acc;
    }
    float m = sc[0];
    for (int s = 1; s < nsrc; s++) if (sc[s] > m) m = sc[s];
    float ex[16]; double z = 0.0;
    for (int s = 0; s < nsrc; s++) { ex[s] = expf(sc[s] - m); z += (double)ex[s]; }
    for (int i = 0; i < E; i++) out[i] = 0.0f;
    for (int s = 0; s < nsrc; s++) {
        const float pi = (float)((double)ex[s] / z);
        const float *v = srcs[s];
        for (int i = 0; i < E; i++) out[i] = out[i] + pi * v[i];
    }
}

/* ================================================================ state */
static float  resid[NPOS][E];
static float *snap[NPOS][16];
static int    nsnap = 0;
static int    have_prefix = 1;

static float  St[H][D][D];            /* KDA recurrent state */
static float  convbuf[3][P][KC - 1];  /* q,k,v history */

static float  x1b[NPOS][E], x2b[NPOS][E], hb[NPOS][E], h2b[NPOS][E];
static float  aout[NPOS][E], ffn[NPOS][E];

/* MLA cache */
static float  mla_klat[NPOS][H * QN], mla_v[NPOS][H * VH], mla_rp[NPOS][QR];

int main(int argc, char **argv)
{
    const char *base = "/root/k3raw";
    char idxp[512]; snprintf(idxp, sizeof idxp, "%s/eqidx.bin", base);
    load_index(idxp);
    size_t tsz; trunk = map_file("/root/k3trunk_i8/trunk.bin", &tsz);

    const int ids[NPOS] = {1008, 10484, 318, 15383, 387};
    int last = (argc > 1) ? atoi(argv[1]) : NLAY - 1;

    /* section 5, initial conditions: the embedding */
    {
        const MRec *m = &mrec[0];
        const uint16_t *tab = (const uint16_t *)(file_ptr(m->file_id) + m->off);
        for (int t = 0; t < NPOS; t++)
            for (int i = 0; i < E; i++) {
                uint32_t u = ((uint32_t)tab[(size_t)ids[t] * E + i]) << 16;
                memcpy(&resid[t][i], &u, 4);
            }
    }

    double T0 = now_s();
    printf("k3-model-equation.md in C, layers 0..%d\n", last);
    fflush(stdout);

    float *fa = malloc(sizeof(float) * E), *fm = malloc(sizeof(float) * E);
    float *srcbuf[16];

    for (int L = 0; L <= last; L++) {
        double t0 = now_s();
        const int isMLA = ((L % 4) == 3 && L <= 91) || (L == 92);
        const int isMoE = (L >= 1);

        float *arn = slot_vec(L, S_ARN, E), *arp = slot_vec(L, S_ARP, E);
        float *mrn = slot_vec(L, S_MRN, E), *mrp = slot_vec(L, S_MRP, E);
        for (int i = 0; i < E; i++) { fa[i] = arn[i] * arp[i]; fm[i] = mrn[i] * mrp[i]; }
        free(arn); free(arp); free(mrn); free(mrp);

        /* (1) pre-attention aggregation, guarded */
        for (int t = 0; t < NPOS; t++) {
            if (nsnap > 0) {
                for (int s = 0; s < nsnap; s++) srcbuf[s] = snap[t][s];
                srcbuf[nsnap] = resid[t];
                AR(hb[t], srcbuf, nsnap + 1, fa);
            } else {
                memcpy(hb[t], resid[t], sizeof(float) * E);
            }
        }
        /* (2) snapshot push */
        if (L % 12 == 0) {
            for (int t = 0; t < NPOS; t++) {
                snap[t][nsnap] = malloc(sizeof(float) * E);
                memcpy(snap[t][nsnap], resid[t], sizeof(float) * E);
            }
            nsnap++; have_prefix = 0;
        }
        /* (3) pre-attention norm */
        float *win = slot_vec(L, S_IN_LN, E);
        for (int t = 0; t < NPOS; t++) rmsnorm(x1b[t], hb[t], win, E, EPS5);
        free(win);

        /* the attention block */
        if (isMLA) {
            const unsigned char *WQA = slot_ptr(L, S_QA), *WQB = slot_ptr(L, S_QB);
            const unsigned char *WKA = slot_ptr(L, S_KA), *WKB = slot_ptr(L, S_KB);
            const unsigned char *WG  = slot_ptr(L, S_G),  *WO  = slot_ptr(L, S_O);
            float *wqan = slot_vec(L, S_QAN, QLORA), *wkan = slot_vec(L, S_KAN, KVL);
            float *ql = malloc(sizeof(float) * QLORA), *qn = malloc(sizeof(float) * QLORA);
            float *qs = malloc(sizeof(float) * (size_t)NPOS * H * QH);
            float *cc = malloc(sizeof(float) * KVW), *kv = malloc(sizeof(float) * H * KVD);
            float *gb = malloc(sizeof(float) * H * VH), *acc = malloc(sizeof(float) * H * VH);

            for (int t = 0; t < NPOS; t++) {
                Q(ql, x1b[t], WQA, E, QLORA);
                rmsnorm(qn, ql, wqan, QLORA, EPS5);
                Q(qs + (size_t)t * H * QH, qn, WQB, QLORA, H * QH);
                float *craw = malloc(sizeof(float) * KVW);
                Q(craw, x1b[t], WKA, E, KVW);
                rmsnorm(cc, craw, wkan, KVL, EPS5);
                memcpy(cc + KVL, craw + KVL, sizeof(float) * QR);
                free(craw);
                memcpy(mla_rp[t], cc + KVL, sizeof(float) * QR);
                Q(kv, cc, WKB, KVL, H * KVD);
                for (int h = 0; h < H; h++) {
                    memcpy(mla_klat[t] + (size_t)h * QN, kv + (size_t)h * KVD, sizeof(float) * QN);
                    memcpy(mla_v[t]    + (size_t)h * VH, kv + (size_t)h * KVD + QN, sizeof(float) * VH);
                }
            }
            const float msc = 1.0f / sqrtf(192.0f);
            for (int t = 0; t < NPOS; t++) {
#pragma omp parallel for schedule(static)
                for (int h = 0; h < H; h++) {
                    const float *qh = qs + (size_t)t * H * QH + (size_t)h * QH;
                    float scv[NPOS];
                    for (int s = 0; s <= t; s++) {
                        const float *kl = mla_klat[s] + (size_t)h * QN;
                        const float *kr = mla_rp[s];
                        double d = 0.0;
                        for (int i = 0; i < QN; i++) d += (double)qh[i] * (double)kl[i];
                        for (int i = 0; i < QR; i++) d += (double)qh[QN + i] * (double)kr[i];
                        scv[s] = (float)d * msc;
                    }
                    float m = scv[0];
                    for (int s = 1; s <= t; s++) if (scv[s] > m) m = scv[s];
                    float ex[NPOS]; double z = 0.0;
                    for (int s = 0; s <= t; s++) { ex[s] = expf(scv[s] - m); z += (double)ex[s]; }
                    float *o = acc + (size_t)h * VH;
                    for (int i = 0; i < VH; i++) o[i] = 0.0f;
                    for (int s = 0; s <= t; s++) {
                        const float pr = (float)((double)ex[s] / z);
                        const float *vs = mla_v[s] + (size_t)h * VH;
                        for (int i = 0; i < VH; i++) o[i] = o[i] + pr * vs[i];
                    }
                }
                Q(gb, x1b[t], WG, E, H * VH);
                for (int i = 0; i < H * VH; i++) gb[i] = acc[i] * sigf(gb[i]);
                Q(aout[t], gb, WO, H * VH, E);
            }
            free(wqan); free(wkan); free(ql); free(qn); free(qs);
            free(cc); free(kv); free(gb); free(acc);
        } else {
            const unsigned char *WQ = slot_ptr(L, S_Q), *WK = slot_ptr(L, S_K);
            const unsigned char *WV = slot_ptr(L, S_V), *WB = slot_ptr(L, S_B);
            const unsigned char *WFA = slot_ptr(L, S_FA), *WFB = slot_ptr(L, S_FB);
            const unsigned char *WG = slot_ptr(L, S_G), *WO = slot_ptr(L, S_O);
            float *cw[3] = { slot_vec(L, S_CQ, P * KC), slot_vec(L, S_CK, P * KC),
                             slot_vec(L, S_CV, P * KC) };
            float *alog = slot_vec(L, S_ALOG, D);
            float *dtb  = slot_vec(L, S_DTB, P);
            float *won  = slot_vec(L, S_ONORM, D);

            float *raw = malloc(sizeof(float) * P), *cv[3];
            for (int j = 0; j < 3; j++) cv[j] = malloc(sizeof(float) * P);
            float *zz = malloc(sizeof(float) * P), *fatmp = malloc(sizeof(float) * D);
            float *beta = malloc(sizeof(float) * H), *alpha = malloc(sizeof(float) * P);
            float *o = malloc(sizeof(float) * P), *on = malloc(sizeof(float) * P);
            float *gt = malloc(sizeof(float) * P);

            memset(St, 0, sizeof St);
            memset(convbuf, 0, sizeof convbuf);
            float ah[H];
            for (int h = 0; h < H; h++) ah[h] = expf(alog[h]);   /* width 128, first 96 used */

            for (int t = 0; t < NPOS; t++) {
                const unsigned char *WW[3] = {WQ, WK, WV};
                for (int j = 0; j < 3; j++) {
                    Q(raw, x1b[t], WW[j], E, P);
                    for (int i = 0; i < P; i++) {
                        float a = cw[j][(size_t)i * KC + 3] * raw[i];
                        a = a + cw[j][(size_t)i * KC + 0] * convbuf[j][i][0];
                        a = a + cw[j][(size_t)i * KC + 1] * convbuf[j][i][1];
                        a = a + cw[j][(size_t)i * KC + 2] * convbuf[j][i][2];
                        cv[j][i] = a * sigf(a);
                        convbuf[j][i][0] = convbuf[j][i][1];
                        convbuf[j][i][1] = convbuf[j][i][2];
                        convbuf[j][i][2] = raw[i];
                    }
                }
                Q(beta, x1b[t], WB, E, H);
                for (int h = 0; h < H; h++) beta[h] = sigf(beta[h]);
                Q(fatmp, x1b[t], WFA, E, D);
                Q(zz, fatmp, WFB, D, P);
                for (int h = 0; h < H; h++)
                    for (int d = 0; d < D; d++) {
                        int i = h * D + d;
                        float u = ah[h] * (zz[i] + dtb[i]);
                        alpha[i] = expf(LAM * sigf(u));
                    }
                /* q and k are L2 normalized per head; v is not */
                l2_blocks(cv[0], cv[0], H, D);
                l2_blocks(cv[1], cv[1], H, D);
                const float qsc = 1.0f / sqrtf(128.0f);
#pragma omp parallel for schedule(static)
                for (int h = 0; h < H; h++) {
                    float u[D];
                    const float *k = cv[1] + (size_t)h * D;
                    const float *v = cv[2] + (size_t)h * D;
                    const float *q = cv[0] + (size_t)h * D;
                    const float *al = alpha + (size_t)h * D;
                    for (int i = 0; i < D; i++)
                        for (int j = 0; j < D; j++) St[h][i][j] = al[i] * St[h][i][j];
                    for (int j = 0; j < D; j++) u[j] = 0.0f;
                    for (int i = 0; i < D; i++)
                        for (int j = 0; j < D; j++) u[j] = u[j] + St[h][i][j] * k[i];
                    for (int i = 0; i < D; i++) {
                        float kb = k[i] * beta[h];
                        for (int j = 0; j < D; j++) St[h][i][j] = St[h][i][j] + kb * (v[j] - u[j]);
                    }
                    float *oo = o + (size_t)h * D;
                    for (int j = 0; j < D; j++) oo[j] = 0.0f;
                    for (int i = 0; i < D; i++) {
                        float qq = q[i] * qsc;
                        for (int j = 0; j < D; j++) oo[j] = oo[j] + St[h][i][j] * qq;
                    }
                }
                rmsnorm_blocks(on, o, won, H, D);
                Q(gt, x1b[t], WG, E, P);
                for (int i = 0; i < P; i++) gt[i] = on[i] * sigf(gt[i]);
                Q(aout[t], gt, WO, P, E);
            }
            for (int j = 0; j < 3; j++) { free(cw[j]); free(cv[j]); }
            free(alog); free(dtb); free(won); free(raw); free(zz); free(fatmp);
            free(beta); free(alpha); free(o); free(on); free(gt);
        }

        /* (4) residual: replace at a snapshot layer, add otherwise */
        for (int t = 0; t < NPOS; t++) {
            if (have_prefix) for (int i = 0; i < E; i++) resid[t][i] = resid[t][i] + aout[t][i];
            else             memcpy(resid[t], aout[t], sizeof(float) * E);
        }
        have_prefix = 1;

        /* (5) pre-MLP aggregation, unguarded, then norm */
        float *wpost = slot_vec(L, S_POST_LN, E);
        for (int t = 0; t < NPOS; t++) {
            for (int s = 0; s < nsnap; s++) srcbuf[s] = snap[t][s];
            srcbuf[nsnap] = resid[t];
            AR(h2b[t], srcbuf, nsnap + 1, fm);
            rmsnorm(x2b[t], h2b[t], wpost, E, EPS5);
        }
        free(wpost);

        /* the MLP */
        if (!isMoE) {
            const unsigned char *WG2 = slot_ptr(L, S_MGATE), *WU = slot_ptr(L, S_MUP);
            const unsigned char *WD = slot_ptr(L, S_MDOWN);
            float *g = malloc(sizeof(float) * DI), *u = malloc(sizeof(float) * DI);
            for (int t = 0; t < NPOS; t++) {
                Q(g, x2b[t], WG2, E, DI);
                Q(u, x2b[t], WU, E, DI);
                situ(g, g, u, DI);
                Q(ffn[t], g, WD, DI, E);
            }
            free(g); free(u);
        } else {
            float *gw = slot_vec(L, S_GATE, NEXP * E);
            float *gbias = slot_vec(L, S_GBIAS, NEXP);
            const unsigned char *WDN = slot_ptr(L, S_EDOWN), *WUP = slot_ptr(L, S_EUP);
            const unsigned char *S1 = slot_ptr(L, S_SH1), *S3 = slot_ptr(L, S_SH3),
                                *S2 = slot_ptr(L, S_SH2);
            float *lnw = slot_vec(L, S_ENORM, LAT);
            float *zl = malloc(sizeof(float) * LAT), *aL = malloc(sizeof(float) * LAT);
            float *nl = malloc(sizeof(float) * LAT), *ro = malloc(sizeof(float) * E);
            float *sh = malloc(sizeof(float) * E);
            float *sg = malloc(sizeof(float) * SI), *su = malloc(sizeof(float) * SI);
            float *eg = malloc(sizeof(float) * I_), *eu = malloc(sizeof(float) * I_);
            float *ed = malloc(sizeof(float) * LAT);
            float *score = malloc(sizeof(float) * NEXP), *ch = malloc(sizeof(float) * NEXP);

            for (int t = 0; t < NPOS; t++) {
                Q(zl, x2b[t], WDN, E, LAT);
#pragma omp parallel for schedule(static)
                for (int e = 0; e < NEXP; e++) {
                    double a = 0.0;
                    const float *gr = gw + (size_t)e * E;
                    for (int i = 0; i < E; i++) a += (double)gr[i] * (double)x2b[t][i];
                    score[e] = sigf((float)a);
                }
                for (int e = 0; e < NEXP; e++) ch[e] = score[e] + gbias[e];
                int idsel[TOPK]; float wts[TOPK];
                for (int j = 0; j < TOPK; j++) {
                    int best = -1; float bv = -INFINITY;
                    for (int e = 0; e < NEXP; e++) if (ch[e] > bv) { bv = ch[e]; best = e; }
                    idsel[j] = best; wts[j] = score[best]; ch[best] = -INFINITY;
                }
                double ssum = 0.0;
                for (int j = 0; j < TOPK; j++) ssum += (double)wts[j];
                const float iv = (float)(1.0 / (ssum + 1e-20));
                for (int j = 0; j < TOPK; j++) wts[j] = wts[j] * iv;

                for (int i = 0; i < LAT; i++) aL[i] = 0.0f;
                for (int j = 0; j < TOPK; j++) {
                    const ERec *p1 = expert_rec(L, idsel[j], 0, 0), *s1 = expert_rec(L, idsel[j], 0, 1);
                    const ERec *p3 = expert_rec(L, idsel[j], 1, 0), *s3 = expert_rec(L, idsel[j], 1, 1);
                    const ERec *p2 = expert_rec(L, idsel[j], 2, 0), *s2 = expert_rec(L, idsel[j], 2, 1);
                    X(eg, zl, file_ptr(p1->file_id) + p1->off, file_ptr(s1->file_id) + s1->off, LAT, I_);
                    X(eu, zl, file_ptr(p3->file_id) + p3->off, file_ptr(s3->file_id) + s3->off, LAT, I_);
                    situ(eg, eg, eu, I_);
                    X(ed, eg, file_ptr(p2->file_id) + p2->off, file_ptr(s2->file_id) + s2->off, I_, LAT);
                    for (int i = 0; i < LAT; i++) aL[i] = aL[i] + wts[j] * ed[i];
                }
                rmsnorm(nl, aL, lnw, LAT, EPS5);
                Q(ro, nl, WUP, LAT, E);
                Q(sg, x2b[t], S1, E, SI);
                Q(su, x2b[t], S3, E, SI);
                situ(sg, sg, su, SI);
                Q(sh, sg, S2, SI, E);
                for (int i = 0; i < E; i++) ffn[t][i] = ro[i] + sh[i];
            }
            free(gw); free(gbias); free(lnw); free(zl); free(aL); free(nl);
            free(ro); free(sh); free(sg); free(su); free(eg); free(eu); free(ed);
            free(score); free(ch);
        }

        /* (6) MLP residual, unconditional */
        for (int t = 0; t < NPOS; t++)
            for (int i = 0; i < E; i++) resid[t][i] = resid[t][i] + ffn[t][i];

        printf("  layer %2d  %s %-5s  %.2fs\n", L, isMLA ? "MLA" : "KDA",
               isMoE ? "MoE" : "dense", now_s() - t0);
        fflush(stdout);
    }

    /* ---------------------------------------------------------------- tail */
    float *orn = malloc(sizeof(float) * E), *orp = malloc(sizeof(float) * E);
    float *mn  = malloc(sizeof(float) * E);
    for (int k = 1; k <= 3; k++) {
        const MRec *m = &mrec[k];
        const uint16_t *h = (const uint16_t *)(file_ptr(m->file_id) + m->off);
        float *dst = (k == 1) ? orn : (k == 2) ? orp : mn;
        for (int i = 0; i < E; i++) {
            uint32_t u = ((uint32_t)h[i]) << 16;
            memcpy(&dst[i], &u, 4);
        }
    }
    float *foldO = malloc(sizeof(float) * E);
    for (int i = 0; i < E; i++) foldO[i] = orn[i] * orp[i];

    float hf[E], nrm[E];
    for (int s = 0; s < nsnap; s++) srcbuf[s] = snap[NPOS - 1][s];
    srcbuf[nsnap] = resid[NPOS - 1];
    AR(hf, srcbuf, nsnap + 1, foldO);
    rmsnorm(nrm, hf, mn, E, EPS5);

    const MRec *lm = &mrec[4];
    const uint16_t *LMW = (const uint16_t *)(file_ptr(lm->file_id) + lm->off);
    float *logits = malloc(sizeof(float) * VOCAB);
    Bf(logits, nrm, LMW, VOCAB, E);

    { FILE *f = fopen("/root/k3raw/eq_c_logits.bin", "wb");
      fwrite(nrm, 4, E, f); fwrite(logits, 4, VOCAB, f); fclose(f); }

    int am = 0;
    for (int i = 1; i < VOCAB; i++) if (logits[i] > logits[am]) am = i;

    printf("\nsnapshots at the tail : %d sources\n", nsnap + 1);
    printf("total wall time       : %.2f s\n", now_s() - T0);
    printf("emitted token         : %d   engine emitted 17374   %s\n",
           am, am == 17374 ? "MATCH" : "DIFFERENT");
    return 0;
}
