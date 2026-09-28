/* clover-server-k3.c - clover-k3.c on a different data layer.
 *
 * clover-k3.c is the reference and is not modified. This is the same program
 * with the same arithmetic; only where the bytes come from changes, and it
 * changes in two steps so that a broken gate says which step broke it.
 *
 *   step 1, this file today: the trunk arrives as 93 independent per-layer
 *           slices instead of one 54.47 GB file. Experts still come from the
 *           checkpoint, exactly as the reference reads them.
 *   step 2, next: experts come from SQLite instead of the checkpoint.
 *           read_range() is the only place expert bytes come off disk.
 *
 * Step 1 touches four things and nothing else:
 *   - the trunk is mapped 93 times, one file per layer, not once
 *   - slot_ptr and slot_vec address slice[L] + (off - layer_off[L])
 *   - cover() maps a slice address back to the same logical page it had in
 *     the whole trunk, so K3_COVER stays comparable with the reference
 *   - teardown unmaps 93 mappings
 * Every kernel, every operator and every buffer is untouched, so the logits
 * must be bit-identical. That is the test:
 *   The capital of France is -> 17374, md5 23d162dcefb18211a7540ef12948f1eb
 *
 * layer_off[] is not read from anywhere. It is the running sum of the slice
 * file sizes, which is only correct if the layers tile the trunk exactly, so
 * every slot is bounds-checked against its slice at startup rather than
 * trusted.
 *
 * K3_SLICES replaces K3_TRUNKPATH: a directory holding L00.bin .. L92.bin.
 *
 * build: gcc -O3 -march=native -ffp-contract=off -fopenmp -DNPOS=<n> \
 *          -o clover-server-k3 clover-server-k3.c -lm
 *
 * -ffp-contract=off is required. Without it the compiler fuses a*b+c into a
 * single-rounded FMA in places the engine rounds twice.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <sqlite3.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ---------------------------------------------------------------- 1.2 shapes */
#define E      7168
#ifndef NPOS
#define NPOS   5
#endif
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

static unsigned char *slice[NLAY];   /* one mmap per layer, L00.bin .. L92.bin */
static size_t slice_sz[NLAY];
static size_t slice_map_n[NLAY];     /* hugetlb rounds up, so munmap needs its own length */
static int64_t layer_off[NLAY];      /* where this layer began in the whole trunk */
static size_t trunk_sz;              /* the 93 slices added up */
static uint16_t *covc;          /* K3_COVER: touches per 4 KB page of the trunk */
static FILE *dumplay;           /* K3_DUMPLAY: each layer input, for the prefix check */

/* Prefix reuse. Positions below TLO are not recomputed; their contribution
   reaches the rest through the cached attention KV and recurrent state.
   TLO is 0 everywhere except a load run, so the normal path is unchanged. */
static int TLO;
static int pfx_n;               /* how many positions the cache covers */
static int pfx_mode;            /* 0 off, 1 save, 2 load */
static FILE *pfx_f;

/* Counts how often each trunk page is read, to answer whether a run needs all
   54.47 GB and whether it reads any of it more than once.
   The address now lands in one of 93 slices, so it is translated back to the
   page it would have had in the whole trunk and the counts stay comparable. */
static inline void cover(const void *p, size_t n)
{
    if (!covc) return;
    uintptr_t a = (uintptr_t)p;
    for (int L = 0; L < NLAY; L++) {
        uintptr_t b = (uintptr_t)slice[L];
        if (!b || a < b || a >= b + slice_sz[L]) continue;
        size_t d = (size_t)(a - b) + (size_t)layer_off[L];
        size_t s = d >> 12, e = (d + n + 4095) >> 12, np = trunk_sz >> 12;
        if (e > np) e = np;
        for (size_t i = s; i < e; i++) if (covc[i] < 65535) covc[i]++;
        return;
    }
}
static unsigned char *fmap[128];
static size_t          fsize[128];

enum {
    S_ARN = 0, S_ARP, S_MRN, S_MRP, S_IN_LN, S_POST_LN, S_G, S_O,
    S_Q, S_K, S_V, S_B, S_FA, S_FB, S_CQ, S_CK, S_CV, S_ALOG, S_DTB, S_ONORM,
    S_QA, S_QAN, S_QB, S_KA, S_KAN, S_KB,
    S_MGATE, S_MUP, S_MDOWN,
    S_GATE, S_GBIAS, S_EDOWN, S_EUP, S_ENORM, S_SH1, S_SH3, S_SH2
};
#define N_SLOTN 37

/* K3_PROV: log the identity of every weight block a run consumes - which trunk
   slot, which expert block, at what offset - instead of the float values, which
   are ~3.7 GB a run. What varies between prompts is the identity, not the bytes. */
static FILE *prov_fp = NULL;
static const char *SLOTN[N_SLOTN] = {
    "ARN", "ARP", "MRN", "MRP", "IN_LN", "POST_LN", "G", "O",
    "Q", "K", "V", "B", "FA", "FB", "CQ", "CK", "CV", "ALOG", "DTB", "ONORM",
    "QA", "QAN", "QB", "KA", "KAN", "KB",
    "MGATE", "MUP", "MDOWN",
    "GATE", "GBIAS", "EDOWN", "EUP", "ENORM", "SH1", "SH3", "SH2"
};

/* K3_LSTAT: the equation's named variables, per layer and position. K3_PROV
   records which weights a value came from; this records what the value was.
   Statistics, not tensors - the tensors are 3.7 GB a run. */
static FILE *lstat_fp = NULL;
static void lstat_vec(int L, int t, const char *name, const float *v, int n)
{
    double s = 0.0, s2 = 0.0;
    float mn = v[0], mx = v[0];
    long nz = 0, neg = 0, nbad = 0;
    for (int i = 0; i < n; i++) {
        const float a = v[i];
        if (a != a || a > 3.0e38f || a < -3.0e38f) nbad++;
        if (a < mn) mn = a;
        if (a > mx) mx = a;
        if (a == 0.0f) nz++;
        else if (a < 0.0f) neg++;
        s += (double)a;
        s2 += (double)a * (double)a;
    }
    fprintf(lstat_fp, "V\t%d\t%d\t%s\t%d\t%.6g\t%.6g\t%.6g\t%.6g\t%ld\t%ld\t%ld\n",
            L, t, name, n, mn, mx, s / n, sqrt(s2 / n), nz, neg, nbad);
}

/* K3_STAGE: one row per operator INVOCATION, not per layer. Q is called about
   1171 times in a five-token run and X about 17049 (5683 distinct experts x 3
   parts); this records what each individual call produced. */
static FILE  *stg_fp = NULL;
static int    cur_L = -1, cur_e = -1;
static const char *cur_part = "";
static long   stg_q = 0, stg_x = 0;

/* Q takes a raw weight pointer, so recover which trunk slot it is by address. */
static const char *slot_of_ptr(int L, const unsigned char *W)
{
    if (L < 0 || !slice[L]) return "?";
    for (int s = 0; s < n_slots && s < N_SLOTN; s++) {
        const Slot *sl = &slots[(size_t)L * n_slots + s];
        if (sl->present && slice[L] + (sl->off - layer_off[L]) == W) return SLOTN[s];
    }
    return "?";
}

static void stg_emit(const char *op, const char *label, int expert,
                     int T, int in, int out, float *const *Y)
{
    double s = 0.0, s2 = 0.0;
    float mn = Y[0][0], mx = Y[0][0];
    long nz = 0;
    for (int t = 0; t < T; t++)
        for (int i = 0; i < out; i++) {
            const float a = Y[t][i];
            if (a < mn) mn = a;
            if (a > mx) mx = a;
            if (a == 0.0f) nz++;
            s += (double)a;
            s2 += (double)a * (double)a;
        }
    const long n = (long)T * (long)out;
    const long seq = (op[0] == 'Q') ? ++stg_q : ++stg_x;
    fprintf(stg_fp, "%s\t%ld\t%d\t%s\t%d\t%d\t%d\t%d\t%ld\t%.6g\t%.6g\t%.6g\t%.6g\t%ld\n",
            op, seq, cur_L, label, expert, T, in, out, n,
            mn, mx, s / n, sqrt(s2 / n), nz);
}

static double now_s(void)
{
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static void die(const char *m) { fprintf(stderr, "eq: %s\n", m); exit(1); }

/* ----------------------------------------------- per-operator accounting */
/* One bucket per primitive operator of k3-model-equation.md section 2, plus
   the router dot product, which is deliberately not one of the ten. */
enum { OP_Q, OP_X, OP_B, OP_N, OP_L, OP_SITU, OP_CONV, OP_AR,
       OP_DELTA, OP_SA, OP_ROUT, OP_TOPK, OP_ALPHA, OP_RESID, OP_COUNT };
static const char *OPN[OP_COUNT] = {
    "Q   int8 projection", "X   mxfp4 expert proj", "B   bf16 lm_head",
    "N   rmsnorm", "L   l2 per-head", "SiTU + sigma", "C   shortconv",
    "AR  snapshot aggregate", "D   kda delta-rule", "SA  softmax attention",
    "router dot product", "top-k selection", "alpha / beta / gate", "residual + copies" };
static double  op_t[OP_COUNT];
static int64_t op_n[OP_COUNT], op_wb[OP_COUNT], op_fl[OP_COUNT], op_out[OP_COUNT];
static int64_t vh[OP_COUNT][6];   /* |y|: exact 0, <1e-6, <1e-3, <1, <1e3, rest */
static int val_on;

/* Twenty-one steps measured bytes and seconds. This measures what the bytes buy. */
static void vstat(int o, float *const *Y, int T, int n)
{
    op_out[o] += (int64_t)T * n;
    if (!val_on) return;
    for (int t = 0; t < T; t++)
        for (int i = 0; i < n; i++) {
            float a = Y[t][i] < 0 ? -Y[t][i] : Y[t][i];
            int b = (a == 0.0f) ? 0 : (a < 1e-6f) ? 1 : (a < 1e-3f) ? 2
                  : (a < 1.0f) ? 3 : (a < 1e3f) ? 4 : 5;
            vh[o][b]++;
        }
}

static inline void op_add(int o, double t0, int64_t wbytes)
{
    op_t[o] += now_s() - t0; op_n[o]++; op_wb[o] += wbytes;
}

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

static int use_huge = 1;   /* 0 = 4 KB, 1 = THP (madvise-only here), 2 = hugetlb pool */
static size_t last_map_n;  /* hugetlb rounds the mapping up, so munmap needs its own length */
static int quiet_load;     /* 93 slices would otherwise print 93 lines */

/* Pull a whole file into anonymous RAM with O_DIRECT, bypassing the page cache
   so it cannot be evicted by expert traffic and is never faulted again. */
static unsigned char *load_ram(const char *path, size_t *szout)
{
    int probe = open(path, O_RDONLY | O_DIRECT);
    int dio = (probe >= 0);
    if (dio) close(probe);
    int fd = open(path, O_RDONLY);
    if (fd < 0) die(path);
    struct stat st;
    if (fstat(fd, &st)) die("fstat");
    close(fd);
    size_t n = (size_t)st.st_size;

    size_t map_n = n;
    int flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE;
    if (use_huge == 2) {
        const size_t HP = 2u << 20;
        map_n = (n + HP - 1) / HP * HP;
        /* NORESERVE on hugetlb defers an empty pool into SIGBUS at first touch. */
        flags = (flags & ~MAP_NORESERVE) | MAP_HUGETLB;
    }
    unsigned char *buf = (unsigned char *)mmap(NULL, map_n, PROT_READ | PROT_WRITE, flags, -1, 0);
    if (buf == MAP_FAILED && use_huge == 2) {
        fprintf(stderr, "hugetlb mmap failed (pool not reserved?) - falling back to THP\n");
        use_huge = 1;
        map_n = n;
        buf = (unsigned char *)mmap(NULL, map_n, PROT_READ | PROT_WRITE,
                                    MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    }
    if (buf == MAP_FAILED) die("anon mmap");
    if (use_huge == 1) madvise(buf, map_n, MADV_HUGEPAGE);
    last_map_n = map_n;

    const size_t CH = 4u << 20;
    size_t nch = (n + CH - 1) / CH;
    double t0 = now_s();
#pragma omp parallel
    {
        int f = open(path, O_RDONLY | (dio ? O_DIRECT : 0));
        int fb = open(path, O_RDONLY);
#pragma omp for schedule(dynamic, 1)
        for (size_t c = 0; c < nch; c++) {
            size_t off = c * CH;
            size_t len = (off + CH <= n) ? CH : (n - off);
            int use = (dio && (len % 4096) == 0) ? f : fb;
            size_t got = 0;
            while (got < len) {
                ssize_t r = pread(use, buf + off + got, len - got, (off_t)(off + got));
                if (r <= 0) break;
                got += (size_t)r;
            }
        }
        close(f); close(fb);
    }
    double dt = now_s() - t0;
    if (!quiet_load)
        fprintf(stderr, "slice -> RAM: %.2f GB in %.2f s = %.2f GB/s (%s, %s)\n",
                n / 1e9, dt, n / 1e9 / dt, dio ? "O_DIRECT" : "buffered",
                use_huge == 2 ? "hugetlb 2M" : use_huge == 1 ? "THP" : "4K");
    *szout = n;
    return buf;
}

static unsigned char *file_ptr(int id)
{
    if (!fmap[id]) fmap[id] = map_file(fpath[id], &fsize[id]);
    return fmap[id];
}

/* ------------------------------------------------------------- prefetch */
/* mmap demand paging delivers one 4 KB fault at a time from inside the
   arithmetic, so only one read stream is ever in flight. These bring a whole
   layer's expert bytes in first, sorted, with every thread faulting at once. */
typedef struct {
    int fid; int64_t off, nb;
    unsigned char *mem;            /* arena address of off, mode 3 only */
    int64_t aoff; size_t alen, apos;
} PRange;

static double  pf_secs = 0.0;
static int par2 = 31;      /* K3_PAR2 bitmask: 1 rmsnorm, 2 rmsnorm_blocks, 4 l2, 8 AR, 16 conv */
static double  pr_secs = 0.0;   /* the per-layer progress line, also uncounted */
static int64_t pf_bytes = 0;
static int     pf_on = 1;
static int     trunk_ram = 0;

static unsigned char *arena = NULL;
static size_t arena_cap = 0;
static int    dfd[128];
static PRange g_rg[NPOS * TOPK * 6];
static int    g_nrg = 0;
static FILE  *route_fp = NULL;
static FILE  *sel_fp = NULL;      /* K3_DUMPSEL: layer, position, token id, its 16 experts */
static const int *g_ids = NULL;
static int    hstat_on = 0;
static double hs_top[4], hs_tiny[4]; static int64_t hs_n = 0;

static double hs2_keep[5], hs2_blk[5]; static int64_t hs2_n = 0;
static double hs3_ov = 0.0, hs3_blk = 0.0; static int64_t hs3_n = 0, hs3_pairs = 0, hs3_bn = 0;

static inline float sigf(float x);

typedef struct { float a, h; int i; } AH;
static int ahdesc(const void *p, const void *q)
{
    float x = ((const AH *)p)->a, y = ((const AH *)q)->a;
    return (x < y) - (x > y);
}

/* Rank neurons by the gate factor b1*tanh(g/b1)*sigma(g), which needs only w1,
   then measure two things: how much of the realized sum|h| the top fraction
   captures, and how many 4096-byte disk blocks of w3 the rest would actually free. */
static void hstat2(const float *g, const float *h, int n)
{
    AH *v = malloc(sizeof(AH) * n);
    double tot = 0.0;
    for (int i = 0; i < n; i++) {
        v[i].a = fabsf((B1C * tanhf(g[i] / B1C)) * sigf(g[i]));
        v[i].h = fabsf(h[i]);
        v[i].i = i;
        tot += v[i].h;
    }
    if (tot <= 0.0) { free(v); return; }
    qsort(v, n, sizeof(AH), ahdesc);

    const double fr[5] = {0.10, 0.25, 0.50, 0.75, 0.90};
    double run = 0.0; int k = 0;
    for (int i = 0; i < n && k < 5; i++) {
        run += v[i].h;
        if (i + 1 == (int)(fr[k] * n)) { hs2_keep[k] += run / tot; k++; }
    }

    const int rowb = 1792, nblk = (n * rowb + 4095) / 4096;
    unsigned char *blk = malloc(nblk);
    for (int q = 0; q < 5; q++) {
        int keep = (int)(fr[q] * n);
        memset(blk, 1, nblk);
        for (int i = 0; i < keep; i++) {
            int j = v[i].i;
            int b0 = (j * rowb) / 4096, b1 = ((j + 1) * rowb - 1) / 4096;
            for (int b = b0; b <= b1 && b < nblk; b++) blk[b] = 0;
        }
        int freed = 0;
        for (int b = 0; b < nblk; b++) freed += blk[b];
        hs2_blk[q] += (double)freed / (double)nblk;
    }
    free(blk);
    hs2_n++;
    free(v);
}

/* Do the positions sharing an expert switch off the SAME neurons? Two independent
   random half-sets would overlap 50%, so that is the null this compares against. */
static void hstat3(float *const *gs, int m, int n)
{
    const int half = n / 2;
    unsigned char *inset = calloc((size_t)m * n, 1);
    AH *v = malloc(sizeof(AH) * n);
    for (int q = 0; q < m; q++) {
        for (int i = 0; i < n; i++) {
            v[i].a = fabsf((B1C * tanhf(gs[q][i] / B1C)) * sigf(gs[q][i]));
            v[i].h = 0.0f; v[i].i = i;
        }
        qsort(v, n, sizeof(AH), ahdesc);
        for (int i = half; i < n; i++) inset[(size_t)q * n + v[i].i] = 1;
    }
    for (int a = 0; a < m; a++)
        for (int b = a + 1; b < m; b++) {
            int inter = 0;
            for (int i = 0; i < n; i++)
                inter += inset[(size_t)a * n + i] & inset[(size_t)b * n + i];
            hs3_ov += (double)inter / (double)half;
            hs3_pairs++;
        }

    /* Best case for a static file layout: order neurons by their MEAN gate across
       the very positions being tested (in-sample, so an upper bound), then count
       4096-byte blocks that every position's own skip set frees. */
    for (int i = 0; i < n; i++) {
        double s = 0.0;
        for (int q = 0; q < m; q++)
            s += fabs((double)(B1C * tanhf(gs[q][i] / B1C)) * sigf(gs[q][i]));
        v[i].a = (float)(s / m); v[i].h = 0.0f; v[i].i = i;
    }
    qsort(v, n, sizeof(AH), ahdesc);
    int *rank = malloc(sizeof(int) * n);
    for (int p = 0; p < n; p++) rank[v[p].i] = p;      /* neuron -> static position */
    const int rowb = 1792, nblk = (n * rowb + 4095) / 4096;
    unsigned char *blk = malloc(nblk);
    for (int q = 0; q < m; q++) {
        memset(blk, 1, nblk);
        for (int i = 0; i < n; i++) {
            if (inset[(size_t)q * n + i]) continue;    /* skipped: leaves block free */
            int p = rank[i];
            int b0 = (p * rowb) / 4096, b1 = ((p + 1) * rowb - 1) / 4096;
            for (int b = b0; b <= b1 && b < nblk; b++) blk[b] = 0;
        }
        int freed = 0;
        for (int b = 0; b < nblk; b++) freed += blk[b];
        hs3_blk += (double)freed / (double)nblk;
        hs3_bn++;
    }
    free(rank); free(blk);
    hs3_n++;
    free(inset); free(v);
}

static int fdesc(const void *a, const void *b)
{
    float x = *(const float *)a, y = *(const float *)b;
    return (x < y) - (x > y);
}

/* realized concentration of |SiTU| across an expert's 3072 neurons */
static void hstat(const float *h, int n)
{
    float *a = malloc(sizeof(float) * n);
    double tot = 0.0;
    for (int i = 0; i < n; i++) { a[i] = fabsf(h[i]); tot += a[i]; }
    if (tot <= 0.0) { free(a); return; }
    qsort(a, n, sizeof(float), fdesc);
    const double frac[4] = {0.10, 0.25, 0.50, 0.75};
    double run = 0.0; int k = 0;
    for (int i = 0; i < n && k < 4; i++) {
        run += a[i];
        if (i + 1 == (int)(frac[k] * n)) { hs_top[k] += run / tot; k++; }
    }
    double mx = a[0], c[4] = {0, 0, 0, 0};
    const double thr[4] = {1e-6, 1e-4, 1e-2, 1e-1};
    for (int i = 0; i < n; i++)
        for (int q = 0; q < 4; q++) if (a[i] < thr[q] * mx) c[q] += 1.0;
    for (int q = 0; q < 4; q++) hs_tiny[q] += c[q] / n;
    hs_n++;
    free(a);
}

static void arena_reserve(size_t need)
{
    if (need <= arena_cap) return;
    if (arena) munmap(arena, arena_cap);
    size_t cap = need + (need >> 2);
    arena = (unsigned char *)mmap(NULL, cap, PROT_READ | PROT_WRITE,
                                  MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (arena == MAP_FAILED) die("arena");
    if (use_huge) madvise(arena, cap, MADV_HUGEPAGE);
    arena_cap = cap;
}

/* Expert bytes live here in mode 3, so the kernels must not read the mmap. */
static PRange *find_rg(int fid, int64_t off)
{
    int lo = 0, hi = g_nrg - 1;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        PRange *g = &g_rg[mid];
        if (g->fid == fid && g->off == off) return g;
        if (g->fid < fid || (g->fid == fid && g->off < off)) lo = mid + 1;
        else hi = mid - 1;
    }
    return NULL;
}

static const unsigned char *res_ptr(int fid, int64_t off)
{
    PRange *g = find_rg(fid, off);
    return g ? g->mem : file_ptr(fid) + off;
}

/* ------------------------------------------------- experts out of SQLite */
/* A layer reads from its store if one exists and from the checkpoint if not,
   so the gate can be run with some layers converted and the rest not. 92
   stores would be 1448 GB and the box has 87 GB.

   Addressing is arithmetic, not a search: a layer is one contiguous run and
   every expert in it is the same size, so a file offset gives the position
   directly. Expert ids are NOT in file order - 894 of 896 are permuted - so
   position is mapped through pos2id, built from the index at startup.

   Connections are a claimed pool, not one per thread. The reader threads are
   pthreads created and joined per layer, so a run makes about 1288 of them;
   indexing by thread identity would either collide (omp_get_thread_num is 0
   for a pthread) or exhaust. A slot keeps its connection and its blob handle
   across claims, so both are opened once. */
#define MAXCONN 32
typedef struct {
    int           on, fid, nexp;
    int64_t       base, esz;      /* first byte of the layer, bytes per expert */
    int           pos2id[NEXP];
    sqlite3      *db[MAXCONN];
    sqlite3_blob *bh[MAXCONN];
    int           row[MAXCONN];   /* which rowid bh is currently open on */
    char          path[512];
} Store;
static Store  store[NLAY];
static int    n_store;
static int64_t sq_bytes;
static double  sq_secs;
static volatile int conn_busy[MAXCONN];

static int claim_conn(void)
{
    for (;;)
        for (int i = 0; i < MAXCONN; i++) {
            int e = 0;
            if (__atomic_compare_exchange_n(&conn_busy[i], &e, 1, 0,
                                            __ATOMIC_ACQUIRE, __ATOMIC_RELAXED))
                return i;
        }
}

static void release_conn(int i)
{
    __atomic_store_n(&conn_busy[i], 0, __ATOMIC_RELEASE);
}

static void sqdie(sqlite3 *db, const char *what)
{
    fprintf(stderr, "eq: %s: %s\n", what, db ? sqlite3_errmsg(db) : "?");
    exit(1);
}

static int store_for(int fid, int64_t off)
{
    if (!n_store) return -1;
    for (int L = 1; L < NLAY; L++) {
        const Store *s = &store[L];
        if (s->on && s->fid == fid && off >= s->base &&
            off < s->base + s->esz * s->nexp) return L;
    }
    return -1;
}

/* One range out of a store. Exactly the bytes the range names, at g->mem,
   so no alignment padding is read and res_ptr keeps working unchanged. */
static void read_range_db(PRange *g, int L)
{
    Store *s = &store[L];
    const int64_t d = g->off - s->base;
    const int pos = (int)(d / s->esz);
    const int64_t delta = d % s->esz;
    if (delta + g->nb > s->esz) die("expert range crosses an expert boundary");
    const int id = s->pos2id[pos];
    const int c = claim_conn();

    if (!s->db[c]) {
        if (sqlite3_open_v2(s->path, &s->db[c],
                            SQLITE_OPEN_READONLY | SQLITE_OPEN_NOMUTEX, NULL))
            sqdie(s->db[c], "store open");
        s->row[c] = -1;
    }
    if (!s->bh[c]) {
        if (sqlite3_blob_open(s->db[c], "main", "expert", "data", id, 0, &s->bh[c]))
            sqdie(s->db[c], "blob open");
        s->row[c] = id;
    } else if (s->row[c] != id) {
        if (sqlite3_blob_reopen(s->bh[c], id)) sqdie(s->db[c], "blob reopen");
        s->row[c] = id;
    }
    if (sqlite3_blob_read(s->bh[c], g->mem, (int)g->nb, (int)delta))
        sqdie(s->db[c], "blob read");
    release_conn(c);
}

static int64_t meta_int(sqlite3 *db, const char *k)
{
    sqlite3_stmt *st;
    if (sqlite3_prepare_v2(db, "SELECT v FROM meta WHERE k = ?", -1, &st, NULL))
        die("meta prepare");
    sqlite3_bind_text(st, 1, k, -1, SQLITE_STATIC);
    if (sqlite3_step(st) != SQLITE_ROW) die("meta row missing");
    const int64_t v = (int64_t)strtoll((const char *)sqlite3_column_text(st, 0), NULL, 10);
    sqlite3_finalize(st);
    return v;
}

static const ERec *expert_rec(int L, int e, int which, int kind);

/* Opens every store present in dir. Absent layers simply stay on the
   checkpoint, which is what makes a partial migration testable. */
static void open_stores(const char *dir)
{
    for (int L = 1; L < NLAY; L++) {
        Store *s = &store[L];
        snprintf(s->path, sizeof s->path, "%s/L%02d.db", dir, L);
        if (access(s->path, R_OK)) continue;
        sqlite3 *db;
        if (sqlite3_open_v2(s->path, &db, SQLITE_OPEN_READONLY, NULL))
            die("store open");
        s->base = meta_int(db, "source_off");
        s->esz  = meta_int(db, "expert_bytes");
        s->nexp = (int)meta_int(db, "nexpert");
        const int64_t sb = meta_int(db, "source_bytes");
        sqlite3_close(db);
        if (s->nexp != NEXP || s->esz * s->nexp != sb) die("store meta disagrees");

        /* position in the file -> expert id, and prove the layer tiles */
        int64_t o[NEXP];
        for (int e = 0; e < NEXP; e++) {
            const ERec *r = expert_rec(L, e, 0, 0);
            if (e == 0) s->fid = r->file_id;
            else if (r->file_id != s->fid) die("layer spans files");
            o[e] = r->off;
        }
        for (int e = 0; e < NEXP; e++) {
            const int64_t d = o[e] - s->base;
            if (d < 0 || d % s->esz || d / s->esz >= NEXP) die("expert off grid");
            s->pos2id[d / s->esz] = e;
        }
        s->on = 1;
        n_store++;
    }
    if (n_store)
        fprintf(stderr, "expert stores: %d of %d layers from SQLite\n", n_store, NLAY - 1);
}

/* One range, O_DIRECT where the alignment allows and buffered for the tail. */
static void read_range(PRange *g)
{
    const int L = store_for(g->fid, g->off);
    if (L >= 0) {
        const double t0 = now_s();
        read_range_db(g, L);
#pragma omp atomic
        sq_bytes += g->nb;
        const double dt = now_s() - t0;
#pragma omp atomic
        sq_secs += dt;
        return;
    }
    size_t needed = (size_t)(g->off + g->nb - g->aoff);
    ssize_t rd = pread(dfd[g->fid], arena + g->apos, g->alen, g->aoff);
    if (rd >= 0 && (size_t)rd >= needed) return;
    int fb = open(fpath[g->fid], O_RDONLY);
    size_t got = 0;
    while (got < needed) {
        ssize_t q = pread(fb, arena + g->apos + got, needed - got, g->aoff + (int64_t)got);
        if (q <= 0) break;
        got += (size_t)q;
    }
    close(fb);
}

/* Mode 4 pipelines the layer: reader threads fetch expert k+1.. while the
   compute threads are still on expert k. Same bytes, same order, same math. */
#define NREADER 32
static int nreader = 8;
static const ERec *expert_rec(int L, int e, int which, int kind);
static int  pl_L, pl_nexp;
static int  pl_e[NPOS * TOPK];
static volatile int pl_done[NPOS * TOPK];
static volatile int pl_next;
static volatile int pl_rnext, pl_nrange;
static volatile int pl_rem[NPOS * TOPK];
static PRange *pl_rg[NPOS * TOPK * 6];
static int pl_gran = 1;            /* 0 = one expert per reader, 1 = one range */
static pthread_t pl_th[NREADER];
static double pl_wait = 0.0, pl_wait0 = 0.0, pl_waitE = 0.0, pl_waitL = 0.0;
static int64_t mhist[NPOS + 1];

/* Selection index. The old code scanned a growing list to dedup each draw and
   then rescanned the whole table once per expert; both are one pass here.
   Order is preserved exactly: experts in first-appearance order, positions
   ascending, which is what the prefetch pipeline and route_fp depend on. */
static int sx_epoch[NEXP], sx_slot[NEXP], sx_ep = 0;
static int sx_head[NPOS * TOPK], sx_tail[NPOS * TOPK];
static int sx_next[NPOS * TOPK], sx_t[NPOS * TOPK], sx_j[NPOS * TOPK];

static void *pl_reader(void *arg)
{
    (void)arg;
    if (pl_gran) {
        /* range granularity: the first expert is finished by all readers at once
           instead of by one reader walking its six ranges in turn */
        for (;;) {
            int r = __atomic_fetch_add((int *)&pl_rnext, 1, __ATOMIC_SEQ_CST);
            if (r >= pl_nrange) break;
            if (pl_rg[r]) read_range(pl_rg[r]);
            int k = r / 6;
            if (__atomic_sub_fetch((int *)&pl_rem[k], 1, __ATOMIC_ACQ_REL) == 0)
                __atomic_store_n((int *)&pl_done[k], 1, __ATOMIC_RELEASE);
        }
        return NULL;
    }
    for (;;) {
        int k = __atomic_fetch_add((int *)&pl_next, 1, __ATOMIC_SEQ_CST);
        if (k >= pl_nexp) break;
        for (int w = 0; w < 3; w++)
            for (int kind = 0; kind < 2; kind++) {
                const ERec *rr = expert_rec(pl_L, pl_e[k], w, kind);
                PRange *g = find_rg(rr->file_id, rr->off);
                if (g) read_range(g);
            }
        __atomic_store_n((int *)&pl_done[k], 1, __ATOMIC_RELEASE);
    }
    return NULL;
}

static void pl_start(int L, const int *experts, int nexp)
{
    pl_L = L; pl_nexp = nexp; pl_next = 0; pl_rnext = 0;
    int n = 0;
    for (int i = 0; i < nexp; i++) {
        pl_e[i] = experts[i]; pl_done[i] = 0; pl_rem[i] = 6;
        for (int w = 0; w < 3; w++)
            for (int kind = 0; kind < 2; kind++) {
                const ERec *rr = expert_rec(L, experts[i], w, kind);
                pl_rg[n++] = find_rg(rr->file_id, rr->off);
            }
    }
    pl_nrange = n;
    for (int i = 0; i < nreader; i++) pthread_create(&pl_th[i], NULL, pl_reader, NULL);
}

static void pl_wait_for(int k)
{
    double t0 = now_s();
    while (!__atomic_load_n((int *)&pl_done[k], __ATOMIC_ACQUIRE)) sched_yield();
    double d = now_s() - t0;
    pl_wait += d;
    if (k == 0) pl_wait0 += d; else if (k < nreader) pl_waitE += d; else pl_waitL += d;
}

static void pl_finish(void)
{
    for (int i = 0; i < nreader; i++) pthread_join(pl_th[i], NULL);
}

static int pr_cmp(const void *a, const void *b)
{
    const PRange *x = (const PRange *)a, *y = (const PRange *)b;
    if (x->fid != y->fid) return x->fid - y->fid;
    return (x->off < y->off) ? -1 : (x->off > y->off);
}

static void prefetch_ranges(PRange *r, int n)
{
    if (!pf_on || n <= 0) return;
    double t0 = now_s();
    qsort(r, n, sizeof(PRange), pr_cmp);
    int m = 0;
    for (int i = 0; i < n; i++)
        if (m == 0 || r[i].fid != r[m - 1].fid || r[i].off != r[m - 1].off) r[m++] = r[i];
    for (int i = 0; i < m; i++) if (r[i].fid >= 0) file_ptr(r[i].fid);   /* not thread-safe */
    int64_t tot = 0;
    for (int i = 0; i < m; i++) tot += r[i].nb;

    if (pf_on == 3 || pf_on == 4) {
        /* O_DIRECT into an arena. The buffered path cannot exceed ~8.8 GB/s on this
           disk while the device gives 14.4, and O_DIRECT cannot warm an mmap, so the
           bytes have to be read somewhere the kernels can address directly. */
        size_t need = 0;
        for (int i = 0; i < m; i++) {
            r[i].aoff = r[i].off & ~(int64_t)4095;
            r[i].alen = (size_t)((r[i].off + r[i].nb - r[i].aoff + 4095) & ~(int64_t)4095);
            r[i].apos = need;
            need += r[i].alen;
        }
        arena_reserve(need);
        for (int i = 0; i < m; i++) {
            r[i].mem = arena + r[i].apos + (size_t)(r[i].off - r[i].aoff);
            if (dfd[r[i].fid] < 0)
                dfd[r[i].fid] = open(fpath[r[i].fid], O_RDONLY | O_DIRECT);
        }
        memcpy(g_rg, r, (size_t)m * sizeof(PRange));
        g_nrg = m;
        if (pf_on == 4) { pf_bytes += tot; return; }   /* mode 4 reads in the pipeline */
#pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < m; i++) read_range(&r[i]);
        pf_secs += now_s() - t0;
        pf_bytes += tot;
        return;
    }
    g_nrg = 0;

#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < m; i++) {
        unsigned char *p = fmap[r[i].fid] + r[i].off;   /* experts only; no trunk range is ever queued */
        uintptr_t a = (uintptr_t)p & ~(uintptr_t)4095;
        madvise((void *)a, (size_t)(r[i].nb + ((uintptr_t)p - a)), MADV_WILLNEED);
        if (pf_on != 2) {            /* mode 2 leaves the readahead async */
            unsigned char s = 0;
            for (int64_t o = 0; o < r[i].nb; o += 4096) s = (unsigned char)(s ^ p[o]);
            if (r[i].nb) s = (unsigned char)(s ^ p[r[i].nb - 1]);
            __asm__ volatile("" :: "r"(s));
        }
    }
    pf_secs += now_s() - t0;
    pf_bytes += tot;
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

/* Called from the layer body only, which is serial; the parallelism is inside
   the kernels. Writing here would not be safe from inside one. */
static void prov_slot(int L, int s, const char *how)
{
    const Slot *sl = &slots[(size_t)L * n_slots + s];
    fprintf(prov_fp, "T\t%d\t%d\t%s\t%s\t%d\t%lld\t%lld\t%d\t%d\n",
            L, s, (s >= 0 && s < N_SLOTN) ? SLOTN[s] : "?", how,
            sl->dtype, (long long)sl->off, (long long)sl->nbytes, sl->d0, sl->d1);
}

static const unsigned char *slot_ptr(int L, int s)
{
    Slot *sl = &slots[(size_t)L * n_slots + s];
    if (!sl->present) die("absent slot");
    if (prov_fp) prov_slot(L, s, "ptr");
    return slice[L] + (sl->off - layer_off[L]);
}

/* dequantize any slot to float32. dtype 0=F32 1=BF16 2=I8R */
static double sv_secs; static int64_t sv_n, sv_elem;   /* no operator counts this */
static int gfuse = 1;   /* K3_GFUSE: fold the gate's scale into the router dot product */
static float *slot_vec(int L, int s, int want)
{
    const double _t = now_s();
    Slot *sl = &slots[(size_t)L * n_slots + s];
    if (!sl->present) die("absent slot vec");
    if (prov_fp) prov_slot(L, s, "vec");
    const unsigned char *p = slice[L] + (sl->off - layer_off[L]);
    /* cover() is called from the kernels, but slot_vec reads the trunk directly,
       so without this the norm weights look untouched. */
    if (sl->dtype == 0)      cover(p, sizeof(float) * (size_t)want);
    else if (sl->dtype == 1) cover(p, 2 * (size_t)want);
    else                     cover(p, (size_t)sl->nbytes);
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
    sv_secs += now_s() - _t; sv_n++; sv_elem += want;
    return out;
}

/* ================================================================ section 2 */

/* Q: int8 projection, sixteen float32 lanes then the fixed tree */
static void Q(float *y, const float *x, const unsigned char *W, int in, int out)
{
    const double _t = now_s();
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
    op_add(OP_Q, _t, (int64_t)out * (int64_t)rowb);
    op_fl[OP_Q] += 2 * (int64_t)out * (int64_t)in;
    vstat(OP_Q, &y, 1, out);
    if (stg_fp) { float *yy = y; stg_emit("Q", slot_of_ptr(cur_L, W), -1, 1, in, out, &yy); }
    cover(W, (size_t)out * rowb);
}

/* X: MXFP4 projection, four 4-lane double accumulators */
static const float E2M1[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
                               -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

/* The decode E2M1[c] * 2^(s-127) has a domain of 16 codes x 256 scale bytes, and
   both factors are fixed by the equation, so the whole map tabulates in 16 KB.
   Written as the literal expression to keep the 263 negative zeros. */
static float DQ[256][16];
static double DQd[256][16];
static double DQ2[256][256][2];   /* scale byte, packed byte -> that byte's two doubles */
static int xdec = 2;

static void dq_init(void)
{
    for (int s = 0; s < 256; s++) {
        float e8 = (s == 255) ? 0.0f : exp2f((float)s - 127.0f);
        for (int c = 0; c < 16; c++) {
            DQ[s][c] = E2M1[c] * e8;
            DQd[s][c] = (double)DQ[s][c];   /* exact widening of the same float */
        }
        for (int b = 0; b < 256; b++) {
            DQ2[s][b][0] = DQd[s][b & 15];
            DQ2[s][b][1] = DQd[s][b >> 4];
        }
    }
}

static void X(float *y, const float *x, const unsigned char *pk,
              const unsigned char *sc, int inn, int rows)
{
    const double _t = now_s();
    const int ngrp = inn / GRP;
    const int cpg  = GRP >> 4;
#pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        const unsigned char *pr = pk + (size_t)r * (inn / 2);
        const unsigned char *sr = sc + (size_t)r * ngrp;
        double v[4][4] = {{0}};
        int g = 0, c = 0;
        for (int i = 0; i < inn; i += 16) {
            const float *dq = DQ[sr[g]];
            float wd[16];
            for (int b = 0; b < 8; b++) {
                unsigned char by = pr[(i >> 1) + b];
                wd[2 * b]     = dq[by & 0x0F];
                wd[2 * b + 1] = dq[by >> 4];
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
    op_add(OP_X, _t, (int64_t)rows * ((int64_t)inn / 2 + (int64_t)ngrp));
}

/* ---- batched forms: the weight is read once and applied to every position ----
   Each output element is still an independent reduction over i in the original
   order, so these are bit-identical to calling Q or X once per position. */

static void Qm(float *const *Y, const float *const *Xs, int T,
               const unsigned char *W, int in, int out)
{
    const double _t = now_s();
    const size_t rowb = (size_t)4 + (size_t)in;
    int omain = 0;
#if defined(__AVX2__)
    /* One x vector is 4*in bytes, so T of them overflow L1 and it is the input
       streams, not the accumulators, that set the rate. Blocking output rows
       reuses each x load across OB rows. Summation order per output is
       unchanged, so results are bit-identical; below T=8 the old path wins. */
    enum { OB = 8 };
    if (T >= 8) omain = (out / OB) * OB;
#pragma omp parallel for schedule(static) if (out > 64)
    for (int o0 = 0; o0 < omain; o0 += OB) {
        const int8_t *wp[OB]; float sc[OB];
        for (int b = 0; b < OB; b++) {
            const unsigned char *row = W + (size_t)(o0 + b) * rowb;
            memcpy(&sc[b], row, 4);
            wp[b] = (const int8_t *)(row + 4);
        }
        for (int t = 0; t < T; t++) {
            const float *xt = Xs[t];
            __m256 a0[OB], a1[OB];
            for (int b = 0; b < OB; b++) {
                a0[b] = _mm256_setzero_ps(); a1[b] = _mm256_setzero_ps();
            }
            int i = 0;
            for (; i + 15 < in; i += 16) {
                const __m256 x0 = _mm256_loadu_ps(xt + i);
                const __m256 x1 = _mm256_loadu_ps(xt + i + 8);
                for (int b = 0; b < OB; b++) {
                    const __m256 w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                        _mm_loadl_epi64((const __m128i *)(wp[b] + i))));
                    const __m256 w1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                        _mm_loadl_epi64((const __m128i *)(wp[b] + i + 8))));
                    a0[b] = _mm256_fmadd_ps(w0, x0, a0[b]);
                    a1[b] = _mm256_fmadd_ps(w1, x1, a1[b]);
                }
            }
            for (int b = 0; b < OB; b++) {
                __m256 vs = _mm256_add_ps(a0[b], a1[b]);
                __m128 lo = _mm_add_ps(_mm256_castps256_ps128(vs),
                                       _mm256_extractf128_ps(vs, 1));
                lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
                lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
                float a = _mm_cvtss_f32(lo);
                for (int k = i; k < in; k++) a = a + (float)wp[b][k] * xt[k];
                Y[t][o0 + b] = a * sc[b];
            }
        }
    }
#endif
#pragma omp parallel for schedule(static) if (out > 64)
    for (int o = omain; o < out; o++) {
        const unsigned char *row = W + (size_t)o * rowb;
        float scale; memcpy(&scale, row, 4);
        const int8_t *w = (const int8_t *)(row + 4);
        int i = 0; float acc[NPOS];
#if defined(__AVX2__)
        __m256 v0[NPOS], v1[NPOS];
        for (int t = 0; t < T; t++) { v0[t] = _mm256_setzero_ps(); v1[t] = _mm256_setzero_ps(); }
        for (; i + 15 < in; i += 16) {
            const __m256 w0 = _mm256_cvtepi32_ps(
                _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(w + i))));
            const __m256 w1 = _mm256_cvtepi32_ps(
                _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(w + i + 8))));
            for (int t = 0; t < T; t++) {
                v0[t] = _mm256_fmadd_ps(w0, _mm256_loadu_ps(Xs[t] + i), v0[t]);
                v1[t] = _mm256_fmadd_ps(w1, _mm256_loadu_ps(Xs[t] + i + 8), v1[t]);
            }
        }
        for (int t = 0; t < T; t++) {
            __m256 vs = _mm256_add_ps(v0[t], v1[t]);
            __m128 lo = _mm_add_ps(_mm256_castps256_ps128(vs), _mm256_extractf128_ps(vs, 1));
            lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
            lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
            acc[t] = _mm_cvtss_f32(lo);
        }
#else
        float B[NPOS][16];
        for (int t = 0; t < T; t++) for (int c = 0; c < 16; c++) B[t][c] = 0.0f;
        for (; i + 15 < in; i += 16)
            for (int t = 0; t < T; t++)
                for (int c = 0; c < 16; c++)
                    B[t][c] = fmaf((float)w[i + c], Xs[t][i + c], B[t][c]);
        for (int t = 0; t < T; t++) {
            float A[8];
            for (int j = 0; j < 8; j++) A[j] = B[t][j] + B[t][j + 8];
            float l0 = A[0] + A[4], l1 = A[1] + A[5], l2 = A[2] + A[6], l3 = A[3] + A[7];
            acc[t] = (l0 + l2) + (l1 + l3);
        }
#endif
        for (int t = 0; t < T; t++) {
            float a = acc[t];
            for (int k = i; k < in; k++) a = a + (float)w[k] * Xs[t][k];
            Y[t][o] = a * scale;
        }
    }
    op_add(OP_Q, _t, (int64_t)out * (int64_t)rowb);
    op_fl[OP_Q] += 2 * (int64_t)out * (int64_t)in * (int64_t)T;
    vstat(OP_Q, Y, T, out);
    if (stg_fp) stg_emit("Q", slot_of_ptr(cur_L, W), -1, T, in, out, Y);
    cover(W, (size_t)out * rowb);
}

static void Xm(float *const *Y, const float *const *Xs, int T,
               const unsigned char *pk, const unsigned char *sc, int inn, int rows)
{
    const double _t = now_s();
    const int ngrp = inn / GRP;
    const int cpg  = GRP >> 4;
#if defined(__AVX2__)
    /* acc[t][m] below is indexed by a runtime t, so it cannot live in registers
       and every accumulation becomes a load-modify-store. T==1 is 77.6% of calls. */
    if (T == 1 && xdec == 2) {
#pragma omp parallel for schedule(static)
        for (int r = 0; r < rows; r++) {
            const unsigned char *pr = pk + (size_t)r * (inn / 2);
            const unsigned char *sr = sc + (size_t)r * ngrp;
            __m256d a0 = _mm256_setzero_pd(), a1 = _mm256_setzero_pd();
            __m256d a2 = _mm256_setzero_pd(), a3 = _mm256_setzero_pd();
            const float *X0 = Xs[0];
            int g = 0, c = 0;
            for (int i = 0; i < inn; i += 16) {
                const double (*d2)[2] = DQ2[sr[g]];
                const unsigned char *p = pr + (i >> 1);
                const __m256d w0 = _mm256_loadu2_m128d(d2[p[1]], d2[p[0]]);
                const __m256d w1 = _mm256_loadu2_m128d(d2[p[3]], d2[p[2]]);
                const __m256d w2 = _mm256_loadu2_m128d(d2[p[5]], d2[p[4]]);
                const __m256d w3 = _mm256_loadu2_m128d(d2[p[7]], d2[p[6]]);
                const float *xp = X0 + i;
                a0 = _mm256_add_pd(a0, _mm256_mul_pd(w0, _mm256_cvtps_pd(_mm_loadu_ps(xp))));
                a1 = _mm256_add_pd(a1, _mm256_mul_pd(w1, _mm256_cvtps_pd(_mm_loadu_ps(xp + 4))));
                a2 = _mm256_add_pd(a2, _mm256_mul_pd(w2, _mm256_cvtps_pd(_mm_loadu_ps(xp + 8))));
                a3 = _mm256_add_pd(a3, _mm256_mul_pd(w3, _mm256_cvtps_pd(_mm_loadu_ps(xp + 12))));
                if (++c == cpg) { c = 0; g++; }
            }
            __m256d qv = _mm256_add_pd(_mm256_add_pd(a0, a2), _mm256_add_pd(a1, a3));
            double q[4]; _mm256_storeu_pd(q, qv);
            double b0 = q[0] + q[2], b1 = q[1] + q[3];
            Y[0][r] = (float)(b0 + b1);
        }
        op_add(OP_X, _t, (int64_t)rows * ((int64_t)inn / 2 + (int64_t)ngrp));
        op_fl[OP_X] += 2 * (int64_t)rows * (int64_t)inn;
        vstat(OP_X, Y, 1, rows);
        if (stg_fp) stg_emit("X", cur_part, cur_e, 1, inn, rows, Y);
        return;
    }
#endif
#pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        const unsigned char *pr = pk + (size_t)r * (inn / 2);
        const unsigned char *sr = sc + (size_t)r * ngrp;
#if defined(__AVX2__)
        __m256d acc[NPOS][4];
        for (int t = 0; t < T; t++)
            for (int m = 0; m < 4; m++) acc[t][m] = _mm256_setzero_pd();
        int g = 0, c = 0;

/* The four weight vectors are built the same way in every variant; only how the
   table is read differs. The wd[] form costs a 128-byte store-load round trip. */
#define XACC                                                                         \
        for (int t = 0; t < T; t++) {                                                \
            const float *xp = Xs[t] + i;                                             \
            acc[t][0] = _mm256_add_pd(acc[t][0],                                     \
                _mm256_mul_pd(w0, _mm256_cvtps_pd(_mm_loadu_ps(xp))));               \
            acc[t][1] = _mm256_add_pd(acc[t][1],                                     \
                _mm256_mul_pd(w1, _mm256_cvtps_pd(_mm_loadu_ps(xp + 4))));           \
            acc[t][2] = _mm256_add_pd(acc[t][2],                                     \
                _mm256_mul_pd(w2, _mm256_cvtps_pd(_mm_loadu_ps(xp + 8))));           \
            acc[t][3] = _mm256_add_pd(acc[t][3],                                     \
                _mm256_mul_pd(w3, _mm256_cvtps_pd(_mm_loadu_ps(xp + 12))));          \
        }                                                                            \
        if (++c == cpg) { c = 0; g++; }

        if (xdec == 2) {
            for (int i = 0; i < inn; i += 16) {
                const double (*d2)[2] = DQ2[sr[g]];
                const unsigned char *p = pr + (i >> 1);
                const __m256d w0 = _mm256_loadu2_m128d(d2[p[1]], d2[p[0]]);
                const __m256d w1 = _mm256_loadu2_m128d(d2[p[3]], d2[p[2]]);
                const __m256d w2 = _mm256_loadu2_m128d(d2[p[5]], d2[p[4]]);
                const __m256d w3 = _mm256_loadu2_m128d(d2[p[7]], d2[p[6]]);
                XACC
            }
        } else if (xdec == 1) {
            for (int i = 0; i < inn; i += 16) {
                const double *dq = DQd[sr[g]];
                const unsigned char *p = pr + (i >> 1);
                const __m256d w0 = _mm256_set_pd(dq[p[1] >> 4], dq[p[1] & 15],
                                                 dq[p[0] >> 4], dq[p[0] & 15]);
                const __m256d w1 = _mm256_set_pd(dq[p[3] >> 4], dq[p[3] & 15],
                                                 dq[p[2] >> 4], dq[p[2] & 15]);
                const __m256d w2 = _mm256_set_pd(dq[p[5] >> 4], dq[p[5] & 15],
                                                 dq[p[4] >> 4], dq[p[4] & 15]);
                const __m256d w3 = _mm256_set_pd(dq[p[7] >> 4], dq[p[7] & 15],
                                                 dq[p[6] >> 4], dq[p[6] & 15]);
                XACC
            }
        } else {
            for (int i = 0; i < inn; i += 16) {
                const double *dq = DQd[sr[g]];
                double wd[16];
                for (int b = 0; b < 8; b++) {
                    unsigned char by = pr[(i >> 1) + b];
                    wd[2 * b]     = dq[by & 0x0F];
                    wd[2 * b + 1] = dq[by >> 4];
                }
                const __m256d w0 = _mm256_loadu_pd(wd);
                const __m256d w1 = _mm256_loadu_pd(wd + 4);
                const __m256d w2 = _mm256_loadu_pd(wd + 8);
                const __m256d w3 = _mm256_loadu_pd(wd + 12);
                XACC
            }
        }
#undef XACC
        for (int t = 0; t < T; t++) {
            __m256d qv = _mm256_add_pd(_mm256_add_pd(acc[t][0], acc[t][2]),
                                       _mm256_add_pd(acc[t][1], acc[t][3]));
            double q[4]; _mm256_storeu_pd(q, qv);
            double a0 = q[0] + q[2], a1 = q[1] + q[3];
            Y[t][r] = (float)(a0 + a1);
        }
#else
        double v[NPOS][4][4];
        for (int t = 0; t < T; t++)
            for (int m = 0; m < 4; m++) for (int k = 0; k < 4; k++) v[t][m][k] = 0.0;
        int g = 0, c = 0;
        for (int i = 0; i < inn; i += 16) {
            const float *dq = DQ[sr[g]];
            float wd[16];
            for (int b = 0; b < 8; b++) {
                unsigned char by = pr[(i >> 1) + b];
                wd[2 * b]     = dq[by & 0x0F];
                wd[2 * b + 1] = dq[by >> 4];
            }
            for (int t = 0; t < T; t++)
                for (int m = 0; m < 4; m++)
                    for (int k = 0; k < 4; k++)
                        v[t][m][k] += (double)wd[m * 4 + k] * (double)Xs[t][i + m * 4 + k];
            if (++c == cpg) { c = 0; g++; }
        }
        for (int t = 0; t < T; t++) {
            double q[4];
            for (int k = 0; k < 4; k++)
                q[k] = (v[t][0][k] + v[t][2][k]) + (v[t][1][k] + v[t][3][k]);
            double a0 = q[0] + q[2], a1 = q[1] + q[3];
            Y[t][r] = (float)(a0 + a1);
        }
#endif
    }
    op_add(OP_X, _t, (int64_t)rows * ((int64_t)inn / 2 + (int64_t)ngrp));
    op_fl[OP_X] += 2 * (int64_t)rows * (int64_t)inn * (int64_t)T;
    vstat(OP_X, Y, T, rows);
    if (stg_fp) stg_emit("X", cur_part, cur_e, T, inn, rows, Y);
    cover(pk, (size_t)rows * (size_t)(inn / 2));
    cover(sc, (size_t)rows * (size_t)ngrp);
}

/* Bf: BF16 projection, sixteen double lanes then the fixed tree */
static void Bf(float *y, const float *x, const uint16_t *W, int rows, int in)
{
    const double _t = now_s();
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
    op_add(OP_B, _t, (int64_t)rows * (int64_t)in * 2);
    cover(W, (size_t)rows * (size_t)in * 2);
}

static void rmsnorm(float *y, const float *x, const float *w, int n, double eps)
{
    const double _t = now_s();
    /* The sum stays serial: splitting it would reorder the additions. */
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
    const float inv = (float)(1.0 / sqrt(ss / (double)n + eps));
#pragma omp parallel for schedule(static) if ((par2 & 1) && n > 4096)
    for (int i = 0; i < n; i++) y[i] = (w[i] * x[i]) * inv;
    op_add(OP_N, _t, (int64_t)n * 4);
}

static void rmsnorm_blocks(float *y, const float *x, const float *w, int nb, int blk)
{
    const double _t = now_s();
    /* Each block owns its own reduction, so splitting by block reorders nothing. */
#pragma omp parallel for schedule(static) if ((par2 & 2) && nb > 1)
    for (int b = 0; b < nb; b++) {
        const float *xs = x + (size_t)b * blk;
        double ss = 0.0;
        for (int i = 0; i < blk; i++) ss += (double)xs[i] * (double)xs[i];
        const float inv = (float)(1.0 / sqrt(ss / (double)blk + EPS5));
        for (int i = 0; i < blk; i++) y[b * blk + i] = (w[i] * xs[i]) * inv;
    }
    op_add(OP_N, _t, (int64_t)blk * 4);
}

static void l2_blocks(float *y, const float *x, int nb, int blk)
{
    const double _t = now_s();
#pragma omp parallel for schedule(static) if ((par2 & 4) && nb > 1)
    for (int b = 0; b < nb; b++) {
        const float *xs = x + (size_t)b * blk;
        double ss = 0.0;
        for (int i = 0; i < blk; i++) ss += (double)xs[i] * (double)xs[i];
        const float inv = (float)(1.0 / sqrt(ss + EPS6));
        for (int i = 0; i < blk; i++) y[b * blk + i] = xs[i] * inv;
    }
    op_add(OP_L, _t, 0);
}

static inline float sigf(float x) { return 1.0f / (1.0f + expf(-x)); }

/* K3_SITUSTAT: what range does SiTU actually see? tanhf and the sigmoid both
   reach exact values well inside float range, so saturation may be exploitable. */
static int situstat;
static int situ_par = 1;   /* K3_SITUPAR=0 restores the serial loop */
static int64_t si_n, si_gsat, si_ssat_hi, si_ssat_lo, si_usat;
static float si_gmin = 1e30f, si_gmax = -1e30f, si_umin = 1e30f, si_umax = -1e30f;

static void situ(float *y, const float *g, const float *u, int n)
{
    const double _t = now_s();
    /* Elementwise, so splitting it changes nothing about the arithmetic. Three
       scalar libm calls per element make it worth the fork even at n=3072. */
#pragma omp parallel for schedule(static) if (situ_par && n > 512)
    for (int i = 0; i < n; i++) {
        float a  = (B1C * tanhf(g[i] / B1C)) * sigf(g[i]);
        float uu = B2C * tanhf(u[i] / B2C);
        y[i] = a * uu;
    }
    op_add(OP_SITU, _t, 0);
    if (situstat) {
        for (int i = 0; i < n; i++) {
            const float gi = g[i], ui = u[i];
            si_n++;
            if (gi < si_gmin) si_gmin = gi;
            if (gi > si_gmax) si_gmax = gi;
            if (ui < si_umin) si_umin = ui;
            if (ui > si_umax) si_umax = ui;
            if (gi > 34.7f || gi < -34.7f) si_gsat++;      /* tanhf(g/4) at +-1 */
            if (gi >  17.0f) si_ssat_hi++;                  /* sigf(g) at 1 */
            if (gi < -17.0f) si_ssat_lo++;
            if (ui > 216.6f || ui < -216.6f) si_usat++;     /* tanhf(u/25) at +-1 */
        }
    }
}

/* AR: snapshot aggregation, float32 source-major weighted sum */
static void AR(float *out, float *const *srcs, int nsrc, const float *fold)
{
    const double _t = now_s();
    float sc[16];
#pragma omp parallel for schedule(static) if ((par2 & 8) && nsrc > 1)
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
    /* Summing per output instead of per source keeps each out[i] in s order. */
    float pis[16];
    for (int s = 0; s < nsrc; s++) pis[s] = (float)((double)ex[s] / z);
#pragma omp parallel for schedule(static) if (par2 & 8)
    for (int i = 0; i < E; i++) {
        float o = 0.0f;
        for (int s = 0; s < nsrc; s++) o = o + pis[s] * srcs[s][i];
        out[i] = o;
    }
    op_add(OP_AR, _t, (int64_t)E * 4);
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

/* Per-position scratch, allocated once instead of ~176,000 malloc/free pairs a
   run. Laid out BUFFER-major: every position of one buffer is contiguous, which
   is how Qm walks them. Position-major cost 1.4 s at 64 tokens by spreading each
   buffer's positions 389 KB apart. */
static float mp_zlm[NPOS][LAT], mp_nlm[NPOS][LAT];
static float mp_rom[NPOS][E],   mp_shm[NPOS][E];
static float mp_sgm[NPOS][SI],  mp_sum2[NPOS][SI];
static float mp_egm[NPOS][I_],  mp_eum[NPOS][I_];
static float mp_slot[NPOS][TOPK][LAT];
static float ap_raw[3][NPOS][P];
static float ap_beta[NPOS][H], ap_fa[NPOS][D];
static float ap_zz[NPOS][P], ap_gtm[NPOS][P], ap_gtf[NPOS][P];
static float ap_qlm[NPOS][QLORA], ap_qnm[NPOS][QLORA];
static float ap_craw[NPOS][KVW], ap_cc[NPOS][KVW];
static float ap_kvm[NPOS][H * KVD];
static float ap_gbm[NPOS][H * VH], ap_gbf[NPOS][H * VH], ap_accm[NPOS][H * VH];
static float qs_pool[(size_t)NPOS * H * QH] __attribute__((aligned(64)));
static float snap_pool[16][NPOS][E] __attribute__((aligned(64)));

int main(int argc, char **argv)
{
    const double T_start = now_s();
    /* Allocations above 128 KB would otherwise be mmap'd and munmap'd every
       layer, faulting their pages in again each time. */
    mallopt(M_MMAP_THRESHOLD, 256 * 1024 * 1024);
    mallopt(M_TRIM_THRESHOLD, 256 * 1024 * 1024);
    /* Paths come from config.env via the scripts. Defaulting them would let a
       run silently pick up another machine's index. */
    const char *idxp = getenv("K3_INDEX");
    if (!idxp) die("K3_INDEX is not set - source config.env, or run ./build.sh");
    load_index(idxp);
    { const char *v = getenv("K3_STORES"); if (v) open_stores(v); }
    dq_init();
    for (int i = 0; i < 128; i++) dfd[i] = -1;
    { const char *v = getenv("K3_STAGE"); if (v) {
        stg_fp = fopen(v, "w");
        if (!stg_fp) die("cannot open K3_STAGE path");
        fprintf(stg_fp,
          "op\tseq\tlayer\tlabel\texpert\tT\tin\tout\tn\tmin\tmax\tmean\trms\tn_zero\n"
          "#Q rows: label is the trunk slot the weight came from, expert -1\n"
          "#X rows: label is gate/up/down, expert is the expert id, T is how many\n"
          "#        positions shared that expert so the block was decoded once\n");
      } }
    { const char *v = getenv("K3_LSTAT"); if (v) {
        lstat_fp = fopen(v, "w");
        if (!lstat_fp) die("cannot open K3_LSTAT path");
        fprintf(lstat_fp,
          "#V\tlayer\tpos\tvariable\tn\tmin\tmax\tmean\trms\tn_zero\tn_neg\tn_nonfinite\n"
          "#note\tpos -1 means the value is per layer, not per position\n"
          "#note\tvariables are numbered in execution order within a layer\n"
          "#note\tlayer 93 is the tail: final aggregate, its norm, and the logits\n");
      } }
    { const char *v = getenv("K3_DUMPROUTE"); if (v) route_fp = fopen(v, "w"); }
    { const char *v = getenv("K3_DUMPSEL"); if (v) sel_fp = fopen(v, "w"); }
    { const char *v = getenv("K3_PROV"); if (v) {
        prov_fp = fopen(v, "w");
        if (!prov_fp) die("cannot open K3_PROV path");
        fprintf(prov_fp,
          "#kind\tT = trunk slot resolved: layer slot name how dtype off nbytes d0 d1\n"
          "#kind\tS = expert chosen:       layer pos token_id rank expert weight\n"
          "#kind\tX = expert block read:   layer expert m part file_id off nbytes d0 d1\n"
          "#kind\tO = output:              rank token logit\n"
          "#note\thow: ptr = used in place, vec = dequantized to float32\n"
          "#note\tpart: gate/up/down x .w = mxfp4 payload, .s = shared exponents\n"
          "#note\tm = how many positions of this run chose that expert in that layer\n"
          "#note\tweight is the post-softmax routing weight, after normalization\n"
          "#note\ttimings from a K3_PROV run are not comparable; the writes are in the loop\n");
      } }
    { const char *v = getenv("K3_HSTAT"); hstat_on = v ? atoi(v) : 0; }
    { const char *v = getenv("K3_NREADER"); if (v) { nreader = atoi(v);
        if (nreader < 1) nreader = 1; if (nreader > NREADER) nreader = NREADER; } }
    { const char *v = getenv("K3_XDEC"); if (v) xdec = atoi(v); }
    { const char *v = getenv("K3_PLGRAN"); if (v) pl_gran = atoi(v); }
    { const char *v = getenv("K3_HUGE"); if (v) use_huge = atoi(v); }
    { const char *v = getenv("K3_PREFETCH"); if (v) pf_on = atoi(v); }
    { const char *v = getenv("K3_TRUNKRAM"); trunk_ram = v ? atoi(v) : 0; }
    size_t tsz;
    const char *sdir = getenv("K3_SLICES");
    if (!sdir) die("K3_SLICES is not set - the directory holding L00.bin .. L92.bin");
    {
        const double t0 = now_s();
        char sp[1024];
        int64_t off = 0;
        quiet_load = 1;
        for (int L = 0; L < NLAY; L++) {
            snprintf(sp, sizeof sp, "%s/L%02d.bin", sdir, L);
            size_t n;
            slice[L] = trunk_ram ? load_ram(sp, &n) : map_file(sp, &n);
            slice_sz[L] = n;
            slice_map_n[L] = trunk_ram ? last_map_n : n;
            layer_off[L] = off;
            off += (int64_t)n;
        }
        quiet_load = 0;
        trunk_sz = tsz = (size_t)off;

        /* layer_off is the running sum of the slice sizes, not something read
           from a file, so it is only right if the layers tile the trunk with
           no gap. Prove that against the index before addressing one weight. */
        int bad = 0;
        for (int L = 0; L < NLAY; L++)
            for (int s = 0; s < n_slots; s++) {
                const Slot *sl = &slots[(size_t)L * n_slots + s];
                if (!sl->present) continue;
                if (sl->off >= layer_off[L] &&
                    sl->off + sl->nbytes <= layer_off[L] + (int64_t)slice_sz[L]) continue;
                if (++bad <= 5)
                    fprintf(stderr, "L%d slot %d [%lld,%lld) outside slice [%lld,%lld)\n",
                            L, s, (long long)sl->off, (long long)(sl->off + sl->nbytes),
                            (long long)layer_off[L],
                            (long long)(layer_off[L] + (int64_t)slice_sz[L]));
            }
        if (bad) die("slices do not tile the trunk the index describes");
        fprintf(stderr, "trunk: %d slices, %.2f GB, %s, %.2f s\n",
                NLAY, trunk_sz / 1e9, trunk_ram ? "RAM" : "mmap", now_s() - t0);
    }
    { const char *v = getenv("K3_COVER");
      if (v && atoi(v)) covc = calloc(trunk_sz >> 12, sizeof(uint16_t)); }
    { const char *v = getenv("K3_VALUE"); if (v) val_on = atoi(v); }
    { const char *v = getenv("K3_GFUSE"); if (v) gfuse = atoi(v); }
    { const char *v = getenv("K3_SITUSTAT"); if (v) situstat = atoi(v); }
    { const char *v = getenv("K3_SITUPAR"); if (v) situ_par = atoi(v); }
    { const char *v = getenv("K3_PAR2"); if (v) par2 = atoi(v); }
    { const char *v = getenv("K3_DUMPLAY"); if (v) dumplay = fopen(v, "wb"); }
    { const char *p = getenv("K3_PFXSAVE"), *q = getenv("K3_PFXLOAD");
      const char *n = getenv("K3_PFXN");
      pfx_n = n ? atoi(n) : 0;
      if (p && pfx_n > 0) { pfx_mode = 1; pfx_f = fopen(p, "wb"); }
      else if (q && pfx_n > 0) { pfx_mode = 2; pfx_f = fopen(q, "rb"); TLO = pfx_n; }
      if ((p || q) && !pfx_f) die("prefix cache open");
      if (pfx_n >= NPOS) die("K3_PFXN must be less than NPOS");
      if (pfx_mode) fprintf(stderr, "prefix cache: %s %d of %d positions\n",
                            pfx_mode == 1 ? "saving" : "loading", pfx_n, NPOS); }

    static const int ids_all[5] = {1008, 10484, 318, 15383, 387};
    static int ids_env[NPOS];
    const int *ids = ids_all;
    /* K3_IDS lets a different prompt be measured; its length must equal NPOS. */
    { const char *v = getenv("K3_IDS");
      if (v) {
          int n = 0;
          for (const char *p = v; *p && n < NPOS; ) {
              ids_env[n++] = (int)strtol(p, (char **)&p, 10);
              while (*p == ',' || *p == ' ') p++;
          }
          if (n != NPOS) { fprintf(stderr, "K3_IDS has %d ids, NPOS is %d\n", n, NPOS); return 2; }
          ids = ids_env;
      } else if (NPOS != 5) {
          fprintf(stderr, "NPOS is %d but no K3_IDS given\n", NPOS); return 2;
      } }
    g_ids = ids;   /* after K3_IDS, or the dump labels the default prompt */
    int last = (argc > 1) ? atoi(argv[1]) : NLAY - 1;
    fprintf(stderr, "prefetch %s  trunk %s\n", pf_on ? "ON" : "OFF",
            trunk_ram ? "RAM" : "mmap");

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
        cur_L = L;
        const int isMLA = ((L % 4) == 3 && L <= 91) || (L == 92);
        const int isMoE = (L >= 1);

        float *arn = slot_vec(L, S_ARN, E), *arp = slot_vec(L, S_ARP, E);
        float *mrn = slot_vec(L, S_MRN, E), *mrp = slot_vec(L, S_MRP, E);
        for (int i = 0; i < E; i++) { fa[i] = arn[i] * arp[i]; fm[i] = mrn[i] * mrp[i]; }
        free(arn); free(arp); free(mrn); free(mrp);

        /* (1) pre-attention aggregation, guarded */
        for (int t = TLO; t < NPOS; t++) {
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
            for (int t = TLO; t < NPOS; t++) {
                snap[t][nsnap] = snap_pool[nsnap][t];
                memcpy(snap[t][nsnap], resid[t], sizeof(float) * E);
            }
            nsnap++; have_prefix = 0;
        }
        /* (3) pre-attention norm */
        float *win = slot_vec(L, S_IN_LN, E);
        /* Prefix reuse needs the state to match at every layer, not just the
           last one, so this dumps each layer's input for comparison. */
        if (dumplay) { for (int t = 0; t < NPOS; t++) fwrite(hb[t], 4, E, dumplay); }
        for (int t = TLO; t < NPOS; t++) rmsnorm(x1b[t], hb[t], win, E, EPS5);
        free(win);

        /* the attention block */
        if (isMLA) {
            const unsigned char *WQA = slot_ptr(L, S_QA), *WQB = slot_ptr(L, S_QB);
            const unsigned char *WKA = slot_ptr(L, S_KA), *WKB = slot_ptr(L, S_KB);
            const unsigned char *WG  = slot_ptr(L, S_G),  *WO  = slot_ptr(L, S_O);
            float *wqan = slot_vec(L, S_QAN, QLORA), *wkan = slot_vec(L, S_KAN, KVL);
            float *qs = qs_pool;
            float *qlm[NPOS], *qnm[NPOS], *crawm[NPOS], *ccm[NPOS], *kvm[NPOS];
            float *qsm[NPOS], *gbm[NPOS], *gbf[NPOS], *accm[NPOS];
            const float *x1p[NPOS]; float *aoutp[NPOS];
            for (int t = TLO; t < NPOS; t++) {
                qlm[t]   = ap_qlm[t];
                qnm[t]   = ap_qnm[t];
                crawm[t] = ap_craw[t];
                ccm[t]   = ap_cc[t];
                kvm[t]   = ap_kvm[t];
                gbm[t]   = ap_gbm[t];
                gbf[t]   = ap_gbf[t];
                accm[t]  = ap_accm[t];
                qsm[t]   = qs + (size_t)t * H * QH;
                x1p[t] = x1b[t]; aoutp[t] = aout[t];
            }
            const int NACT = NPOS - TLO;

            Qm(qlm + TLO, x1p + TLO, NACT, WQA, E, QLORA);
            for (int t = TLO; t < NPOS; t++) rmsnorm(qnm[t], qlm[t], wqan, QLORA, EPS5);
            Qm(qsm + TLO, (const float *const *)qnm + TLO, NACT, WQB, QLORA, H * QH);
            Qm(crawm + TLO, x1p + TLO, NACT, WKA, E, KVW);
            for (int t = TLO; t < NPOS; t++) {
                rmsnorm(ccm[t], crawm[t], wkan, KVL, EPS5);
                memcpy(ccm[t] + KVL, crawm[t] + KVL, sizeof(float) * QR);
                memcpy(mla_rp[t], ccm[t] + KVL, sizeof(float) * QR);
            }
            Qm(kvm + TLO, (const float *const *)ccm + TLO, NACT, WKB, KVL, H * KVD);
            for (int t = TLO; t < NPOS; t++)
                for (int h = 0; h < H; h++) {
                    memcpy(mla_klat[t] + (size_t)h * QN, kvm[t] + (size_t)h * KVD,
                           sizeof(float) * QN);
                    memcpy(mla_v[t] + (size_t)h * VH, kvm[t] + (size_t)h * KVD + QN,
                           sizeof(float) * VH);
                }
            /* Attention at an active position reads every earlier position, so
               the cached ones must be in place before the scores are formed. */
            if (pfx_mode == 1) {
                for (int t = 0; t < pfx_n; t++) {
                    fwrite(mla_klat[t], 4, (size_t)H * QN, pfx_f);
                    fwrite(mla_v[t],    4, (size_t)H * VH, pfx_f);
                    fwrite(mla_rp[t],   4, QR, pfx_f);
                }
            } else if (pfx_mode == 2) {
                for (int t = 0; t < pfx_n; t++) {
                    if (fread(mla_klat[t], 4, (size_t)H * QN, pfx_f) != (size_t)H * QN ||
                        fread(mla_v[t],    4, (size_t)H * VH, pfx_f) != (size_t)H * VH ||
                        fread(mla_rp[t],   4, QR, pfx_f) != QR) die("prefix cache short read");
                }
            }
            const float msc = 1.0f / sqrtf(192.0f);
            Qm(gbm + TLO, x1p + TLO, NACT, WG, E, H * VH);
            for (int t = TLO; t < NPOS; t++) {
                float *acc = accm[t];
                const double _tsa = now_s();
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
                op_add(OP_SA, _tsa, 0);
                for (int i = 0; i < H * VH; i++) gbf[t][i] = acc[i] * sigf(gbm[t][i]);
            }
            Qm(aoutp + TLO, (const float *const *)gbf + TLO, NACT, WO, H * VH, E);

            free(wqan); free(wkan);
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
            /* KDA carries a recurrent state rather than per-position KV, and it
               is a fixed size whatever the prefix length. */
            if (pfx_mode == 2) {
                if (fread(St, 1, sizeof St, pfx_f) != sizeof St ||
                    fread(convbuf, 1, sizeof convbuf, pfx_f) != sizeof convbuf)
                    die("prefix cache short read");
            }
            float ah[H];
            for (int h = 0; h < H; h++) ah[h] = expf(alog[h]);   /* width 128, first 96 used */

            float *rawm[3][NPOS], *betam[NPOS], *fam[NPOS], *zzm[NPOS];
            float *gtm[NPOS], *gtf[NPOS];
            const float *x1p[NPOS]; float *aoutp[NPOS];
            for (int t = TLO; t < NPOS; t++) {
                for (int j = 0; j < 3; j++) rawm[j][t] = ap_raw[j][t];
                betam[t] = ap_beta[t];
                fam[t]   = ap_fa[t];
                zzm[t]   = ap_zz[t];
                gtm[t]   = ap_gtm[t];
                gtf[t]   = ap_gtf[t];
                x1p[t] = x1b[t]; aoutp[t] = aout[t];
            }
            const int NACT = NPOS - TLO;

            /* every projection that depends only on x1 is taken one weight at a time */
            const unsigned char *WW[3] = {WQ, WK, WV};
            for (int j = 0; j < 3; j++) Qm(rawm[j] + TLO, x1p + TLO, NACT, WW[j], E, P);
            Qm(betam + TLO, x1p + TLO, NACT, WB, E, H);
            Qm(fam + TLO,   x1p + TLO, NACT, WFA, E, D);
            Qm(zzm + TLO, (const float *const *)fam + TLO, NACT, WFB, D, P);
            Qm(gtm + TLO, x1p + TLO, NACT, WG, E, P);

            /* conv history and delta-rule state carry, so this stays in order */
            for (int t = TLO; t < NPOS; t++) {
                for (int j = 0; j < 3; j++) {
                    const float *rw = rawm[j][t];
                    const double _tc = now_s();
                    /* Index i touches only its own conv history, so this splits cleanly. */
#pragma omp parallel for schedule(static) if (par2 & 16)
                    for (int i = 0; i < P; i++) {
                        float a = cw[j][(size_t)i * KC + 3] * rw[i];
                        a = a + cw[j][(size_t)i * KC + 0] * convbuf[j][i][0];
                        a = a + cw[j][(size_t)i * KC + 1] * convbuf[j][i][1];
                        a = a + cw[j][(size_t)i * KC + 2] * convbuf[j][i][2];
                        cv[j][i] = a * sigf(a);
                        convbuf[j][i][0] = convbuf[j][i][1];
                        convbuf[j][i][1] = convbuf[j][i][2];
                        convbuf[j][i][2] = rw[i];
                    }
                    op_add(OP_CONV, _tc, (int64_t)P * KC * 4);
                }
                for (int h = 0; h < H; h++) beta[h] = sigf(betam[t][h]);
                const double _tal = now_s();
                for (int h = 0; h < H; h++)
                    for (int d = 0; d < D; d++) {
                        int i = h * D + d;
                        float u = ah[h] * (zzm[t][i] + dtb[i]);
                        alpha[i] = expf(LAM * sigf(u));
                    }
                op_add(OP_ALPHA, _tal, 0);
                /* q and k are L2 normalized per head; v is not */
                l2_blocks(cv[0], cv[0], H, D);
                l2_blocks(cv[1], cv[1], H, D);
                const float qsc = 1.0f / sqrtf(128.0f);
                const double _td = now_s();
#pragma omp parallel for schedule(static)
                for (int h = 0; h < H; h++) {
                    float u[D];
                    const float *k = cv[1] + (size_t)h * D;
                    const float *v = cv[2] + (size_t)h * D;
                    const float *q = cv[0] + (size_t)h * D;
                    const float *al = alpha + (size_t)h * D;
                    /* Four passes over St[h] (64 KB) became two. The i order and the
                       j order are unchanged, so every accumulation is in the same
                       sequence as before and the result is identical. */
                    for (int j = 0; j < D; j++) u[j] = 0.0f;
                    for (int i = 0; i < D; i++) {
                        const float ali = al[i], ki = k[i];
                        for (int j = 0; j < D; j++) {
                            const float s = ali * St[h][i][j];
                            St[h][i][j] = s;
                            u[j] = u[j] + s * ki;
                        }
                    }
                    float *oo = o + (size_t)h * D;
                    for (int j = 0; j < D; j++) oo[j] = 0.0f;
                    for (int i = 0; i < D; i++) {
                        const float kb = k[i] * beta[h];
                        const float qq = q[i] * qsc;
                        for (int j = 0; j < D; j++) {
                            const float s = St[h][i][j] + kb * (v[j] - u[j]);
                            St[h][i][j] = s;
                            oo[j] = oo[j] + s * qq;
                        }
                    }
                }
                op_add(OP_DELTA, _td, 0);
                rmsnorm_blocks(on, o, won, H, D);
                for (int i = 0; i < P; i++) gtf[t][i] = on[i] * sigf(gtm[t][i]);
                if (pfx_mode == 1 && t == pfx_n - 1) {
                    fwrite(St, 1, sizeof St, pfx_f);
                    fwrite(convbuf, 1, sizeof convbuf, pfx_f);
                }
            }
            Qm(aoutp + TLO, (const float *const *)gtf + TLO, NACT, WO, P, E);

            for (int j = 0; j < 3; j++) { free(cw[j]); free(cv[j]); }
            free(alog); free(dtb); free(won); free(raw); free(zz); free(fatmp);
            free(beta); free(alpha); free(o); free(on); free(gt);
        }

        /* (4) residual: replace at a snapshot layer, add otherwise */
        for (int t = TLO; t < NPOS; t++) {
            if (have_prefix) for (int i = 0; i < E; i++) resid[t][i] = resid[t][i] + aout[t][i];
            else             memcpy(resid[t], aout[t], sizeof(float) * E);
        }
        have_prefix = 1;

        /* (5) pre-MLP aggregation, unguarded, then norm */
        float *wpost = slot_vec(L, S_POST_LN, E);
        for (int t = TLO; t < NPOS; t++) {
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
            /* Batched like every other weight: these three are 727 MB, and the
               per-position loop re-read all of it for each position. */
            const int NACT0 = NPOS - TLO;
            float *gm[NPOS], *um[NPOS], *fp[NPOS];
            const float *x2p0[NPOS];
            for (int t = TLO; t < NPOS; t++) {
                gm[t] = malloc(sizeof(float) * DI);
                um[t] = malloc(sizeof(float) * DI);
                fp[t] = ffn[t];
                x2p0[t] = x2b[t];
            }
            Qm(gm + TLO, x2p0 + TLO, NACT0, WG2, E, DI);
            Qm(um + TLO, x2p0 + TLO, NACT0, WU, E, DI);
            for (int t = TLO; t < NPOS; t++) situ(gm[t], gm[t], um[t], DI);
            Qm(fp + TLO, (const float *const *)gm + TLO, NACT0, WD, DI, E);
            for (int t = TLO; t < NPOS; t++) { free(gm[t]); free(um[t]); }
        } else {
            /* The gate is int8 rows with a per-row scale, the same form Q reads
               without ever building a float copy. K3_GFUSE=0 restores the copy. */
            float *gw = gfuse ? NULL : slot_vec(L, S_GATE, NEXP * E);
            const unsigned char *gp = slot_ptr(L, S_GATE);
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

            int idsel_all[NPOS][TOPK]; float wts_all[NPOS][TOPK];

            /* phase 1: routing only. It needs x2b[t], which step (5) already
               produced for every position, so no expert weight is touched.
               Expert-major: each gate row is read once and applied to every
               position, instead of the whole gate being re-read per position. */
            float (*sc_all)[NEXP] = malloc(sizeof(float) * NPOS * NEXP);
            {
                const double _tr = now_s();
#pragma omp parallel for schedule(static)
                for (int e = 0; e < NEXP; e++) {
                    float tmp[E];
                    const float *grow;
                    if (gfuse) {
                        const unsigned char *row = gp + (size_t)e * (4 + E);
                        float sc; memcpy(&sc, row, 4);
                        const int8_t *w = (const int8_t *)(row + 4);
                        for (int i = 0; i < E; i++)
                            tmp[i] = (float)w[i] * sc;   /* must round to float here */
                        grow = tmp;
                    } else {
                        grow = gw + (size_t)e * E;
                    }
                    for (int t = TLO; t < NPOS; t++) {
                        double a = 0.0;
                        for (int i = 0; i < E; i++)
                            a += (double)grow[i] * (double)x2b[t][i];
                        sc_all[t][e] = sigf((float)a);
                    }
                }
                op_add(OP_ROUT, _tr, gfuse ? (int64_t)NEXP * (4 + E)
                                           : (int64_t)NEXP * E * 4);
                if (gfuse) cover(gp, (size_t)NEXP * (4 + E));   /* read directly, not via Q */
            }
            for (int t = TLO; t < NPOS; t++) {
                const float *score = sc_all[t];
                const double _tk = now_s();
                /* One pass replaces sixteen linear maxima. Scanning e ascending and
                   inserting only on a strict > gives descending value with the lower
                   index winning ties, which is what the repeated maximum produced. */
                int ti[TOPK]; float tv[TOPK]; int nt = 0;
                for (int e = 0; e < NEXP; e++) {
                    const float v = score[e] + gbias[e];
                    if (nt == TOPK && !(v > tv[TOPK - 1])) continue;
                    int p = (nt < TOPK) ? nt : TOPK - 1;
                    while (p > 0 && v > tv[p - 1]) { tv[p] = tv[p - 1]; ti[p] = ti[p - 1]; p--; }
                    tv[p] = v; ti[p] = e;
                    if (nt < TOPK) nt++;
                }
                for (int j = 0; j < TOPK; j++) {
                    idsel_all[t][j] = ti[j]; wts_all[t][j] = score[ti[j]];
                }
                double ssum = 0.0;
                for (int j = 0; j < TOPK; j++) ssum += (double)wts_all[t][j];
                const float iv = (float)(1.0 / (ssum + 1e-20));
                for (int j = 0; j < TOPK; j++) wts_all[t][j] = wts_all[t][j] * iv;
                op_add(OP_TOPK, _tk, 0);
                if (sel_fp) {
                    fprintf(sel_fp, "%d\t%d\t%d", L, t, g_ids ? g_ids[t] : -1);
                    for (int j = 0; j < TOPK; j++) fprintf(sel_fp, "\t%d", idsel_all[t][j]);
                    fputc('\n', sel_fp);
                }
                if (prov_fp)
                    for (int j = 0; j < TOPK; j++)
                        fprintf(prov_fp, "S\t%d\t%d\t%d\t%d\t%d\t%.9g\n",
                                L, t, g_ids ? g_ids[t] : -1, j,
                                idsel_all[t][j], wts_all[t][j]);
            }
            free(sc_all);

            /* phase 2: every byte this layer will read is now known */
            {
                PRange rg[NPOS * TOPK * 6]; int nr = 0;
                for (int t = TLO; t < NPOS; t++)
                    for (int j = 0; j < TOPK; j++)
                        for (int w = 0; w < 3; w++)
                            for (int k = 0; k < 2; k++) {
                                const ERec *rr = expert_rec(L, idsel_all[t][j], w, k);
                                rg[nr].fid = rr->file_id; rg[nr].off = rr->off;
                                rg[nr].nb = rr->nbytes; nr++;
                            }
                prefetch_ranges(rg, nr);
            }

            /* phase 3: batched. Each trunk weight is streamed once for all
               positions, and each distinct expert is decoded once for every
               position that chose it. Section 4.2 fixes the routed sum as
               float32 expert-major in rank order, so each expert's output is
               parked in that position's rank slot and summed in j order after. */
            {
                float *zlm[NPOS], *nlm[NPOS], *rom[NPOS], *shm[NPOS];
                float *sgm[NPOS], *sum2[NPOS], *egm[NPOS], *eum[NPOS];
                float *slot[NPOS][TOPK];
                const float *x2p[NPOS];
                for (int t = TLO; t < NPOS; t++) {
                    zlm[t]  = malloc(sizeof(float) * LAT);
                    nlm[t]  = malloc(sizeof(float) * LAT);
                    rom[t]  = malloc(sizeof(float) * E);
                    shm[t]  = malloc(sizeof(float) * E);
                    sgm[t]  = malloc(sizeof(float) * SI);
                    sum2[t] = malloc(sizeof(float) * SI);
                    egm[t]  = malloc(sizeof(float) * I_);
                    eum[t]  = malloc(sizeof(float) * I_);
                    for (int j = 0; j < TOPK; j++) slot[t][j] = malloc(sizeof(float) * LAT);
                    x2p[t] = x2b[t];
                }

                const int NACT = NPOS - TLO;
                Qm(zlm + TLO, x2p + TLO, NACT, WDN, E, LAT);

                int seen[NPOS * TOPK], nseen = 0;
                sx_ep++;
                {
                    int d = 0;
                    for (int t0 = TLO; t0 < NPOS; t0++)
                        for (int j0 = 0; j0 < TOPK; j0++, d++) {
                            const int e = idsel_all[t0][j0];
                            int s;
                            if (sx_epoch[e] == sx_ep) {
                                s = sx_slot[e];
                            } else {
                                sx_epoch[e] = sx_ep; sx_slot[e] = s = nseen;
                                seen[nseen] = e;
                                sx_head[nseen] = -1; sx_tail[nseen] = -1;
                                nseen++;
                                if (route_fp) fprintf(route_fp, "%d\t%d\n", L, e);
                            }
                            sx_t[d] = t0; sx_j[d] = j0; sx_next[d] = -1;
                            if (sx_tail[s] < 0) sx_head[s] = d; else sx_next[sx_tail[s]] = d;
                            sx_tail[s] = d;
                        }
                }

                if (pf_on == 4) pl_start(L, seen, nseen);

                for (int sx = 0; sx < nseen; sx++) {
                        if (pf_on == 4) pl_wait_for(sx);
                        const int e = seen[sx];

                        int mt[NPOS], mj[NPOS], m = 0;
                        for (int d2 = sx_head[sx]; d2 >= 0; d2 = sx_next[d2]) {
                            mt[m] = sx_t[d2]; mj[m] = sx_j[d2]; m++;
                        }

                        const float *zin[NPOS];
                        float *go[NPOS], *uo[NPOS], *dgo[NPOS];
                        for (int q = 0; q < m; q++) {
                            zin[q] = zlm[mt[q]]; go[q] = egm[mt[q]];
                            uo[q]  = eum[mt[q]]; dgo[q] = slot[mt[q]][mj[q]];
                        }
                        mhist[m]++;
                        const ERec *p1 = expert_rec(L, e, 0, 0), *s1 = expert_rec(L, e, 0, 1);
                        const ERec *p3 = expert_rec(L, e, 1, 0), *s3 = expert_rec(L, e, 1, 1);
                        const ERec *p2 = expert_rec(L, e, 2, 0), *s2 = expert_rec(L, e, 2, 1);
                        if (prov_fp) {
                            const ERec *ee[6] = { p1, s1, p3, s3, p2, s2 };
                            static const char *PN[6] = { "gate.w", "gate.s", "up.w",
                                                         "up.s", "down.w", "down.s" };
                            for (int z = 0; z < 6; z++)
                                fprintf(prov_fp, "X\t%d\t%d\t%d\t%s\t%d\t%lld\t%lld\t%d\t%d\n",
                                        L, e, m, PN[z], ee[z]->file_id,
                                        (long long)ee[z]->off, (long long)ee[z]->nbytes,
                                        ee[z]->d0, ee[z]->d1);
                        }
                        cur_e = e;
                        cur_part = "gate";
                        Xm(go, zin, m, res_ptr(p1->file_id, p1->off),
                           res_ptr(s1->file_id, s1->off), LAT, I_);
                        cur_part = "up";
                        Xm(uo, zin, m, res_ptr(p3->file_id, p3->off),
                           res_ptr(s3->file_id, s3->off), LAT, I_);
                        if (hstat_on == 3 && m > 1) {
                            float *gs[NPOS];          /* situ is in place, so copy g first */
                            for (int q = 0; q < m; q++) {
                                gs[q] = malloc(sizeof(float) * I_);
                                memcpy(gs[q], go[q], sizeof(float) * I_);
                            }
                            for (int q = 0; q < m; q++) situ(go[q], go[q], uo[q], I_);
                            hstat3(gs, m, I_);
                            for (int q = 0; q < m; q++) free(gs[q]);
                        } else {
                            for (int q = 0; q < m; q++) {
                                if (hstat_on == 2) {
                                    float *gsave = malloc(sizeof(float) * I_);
                                    memcpy(gsave, go[q], sizeof(float) * I_);
                                    situ(go[q], go[q], uo[q], I_);
                                    hstat2(gsave, go[q], I_);
                                    free(gsave);
                                } else {
                                    situ(go[q], go[q], uo[q], I_);
                                    if (hstat_on == 1) hstat(go[q], I_);
                                }
                            }
                        }
                        cur_part = "down";
                        Xm(dgo, (const float *const *)go, m,
                           res_ptr(p2->file_id, p2->off),
                           res_ptr(s2->file_id, s2->off), I_, LAT);
                    }
                if (pf_on == 4) pl_finish();

                for (int t = TLO; t < NPOS; t++) {
                    for (int i = 0; i < LAT; i++) aL[i] = 0.0f;
                    for (int j = 0; j < TOPK; j++) {
                        const float pi = wts_all[t][j];
                        const float *ed = slot[t][j];
                        for (int i = 0; i < LAT; i++) aL[i] = aL[i] + pi * ed[i];
                    }
                    rmsnorm(nlm[t], aL, lnw, LAT, EPS5);
                }

                Qm(rom + TLO, (const float *const *)nlm + TLO, NACT, WUP, LAT, E);
                Qm(sgm + TLO, x2p + TLO, NACT, S1, E, SI);
                Qm(sum2 + TLO, x2p + TLO, NACT, S3, E, SI);
                for (int t = TLO; t < NPOS; t++) situ(sgm[t], sgm[t], sum2[t], SI);
                Qm(shm + TLO, (const float *const *)sgm + TLO, NACT, S2, SI, E);
                for (int t = TLO; t < NPOS; t++)
                    for (int i = 0; i < E; i++) ffn[t][i] = rom[t][i] + shm[t][i];

                for (int t = TLO; t < NPOS; t++) {
                    free(zlm[t]); free(nlm[t]); free(rom[t]); free(shm[t]);
                    free(sgm[t]); free(sum2[t]); free(egm[t]); free(eum[t]);
                    for (int j = 0; j < TOPK; j++) free(slot[t][j]);
                }
            }
            free(gw); free(gbias); free(lnw); free(zl); free(aL); free(nl);
            free(ro); free(sh); free(sg); free(su); free(eg); free(eu); free(ed);
            free(score); free(ch);
        }

        /* (6) MLP residual, unconditional */
#pragma omp parallel for collapse(2) schedule(static)
        for (int t = TLO; t < NPOS; t++)
            for (int i = 0; i < E; i++) resid[t][i] = resid[t][i] + ffn[t][i];

        if (lstat_fp) {
            for (int t = TLO; t < NPOS; t++) {
                lstat_vec(L, t, "1_resid_in_aggregated", hb[t],    E);
                lstat_vec(L, t, "2_x1_prenorm",          x1b[t],   E);
                lstat_vec(L, t, "3_attn_out",            aout[t],  E);
                lstat_vec(L, t, "4_h2_after_attn",       h2b[t],   E);
                lstat_vec(L, t, "5_x2_postnorm",         x2b[t],   E);
                lstat_vec(L, t, "6_ffn_out",             ffn[t],   E);
                lstat_vec(L, t, "7_resid_out",           resid[t], E);
            }
            /* the carried recurrent state is per layer, not per position */
            if (!isMLA) lstat_vec(L, -1, "8_kda_state", &St[0][0][0], H * D * D);
        }

        { const double _tp = now_s();
          printf("  layer %2d  %s %-5s  %.2fs\n", L, isMLA ? "MLA" : "KDA",
                 isMoE ? "MoE" : "dense", now_s() - t0);
          fflush(stdout);
          pr_secs += now_s() - _tp; }
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
    /* Causality says a position depends only on tokens at or before it, so the
       same prefix must give the same state whatever follows. This dumps it. */
    { const char *p = getenv("K3_DUMPRES");
      if (p) {
          FILE *f = fopen(p, "wb");
          if (f) { for (int t = 0; t < NPOS; t++) fwrite(resid[t], 4, E, f); fclose(f); }
      } }
    for (int s = 0; s < nsnap; s++) srcbuf[s] = snap[NPOS - 1][s];
    srcbuf[nsnap] = resid[NPOS - 1];
    AR(hf, srcbuf, nsnap + 1, foldO);
    rmsnorm(nrm, hf, mn, E, EPS5);

    const MRec *lm = &mrec[4];
    const uint16_t *LMW = (const uint16_t *)(file_ptr(lm->file_id) + lm->off);
    float *logits = malloc(sizeof(float) * VOCAB);
    Bf(logits, nrm, LMW, VOCAB, E);

    if (lstat_fp) {
        lstat_vec(93, NPOS - 1, "9_tail_aggregate", hf,     E);
        lstat_vec(93, NPOS - 1, "9_tail_norm",      nrm,    E);
        lstat_vec(93, NPOS - 1, "9_logits",         logits, VOCAB);
        fclose(lstat_fp); lstat_fp = NULL;
    }
    if (stg_fp) {
        cur_L = 93;
        { float *ll = logits; stg_emit("B", "lm_head", -1, 1, E, VOCAB, &ll); }
        fprintf(stg_fp, "#totals\tQ calls %ld\tX calls %ld\n", stg_q, stg_x);
        fclose(stg_fp); stg_fp = NULL;
    }

    { const char *lp = getenv("K3_LOGITS");
      FILE *f = fopen(lp ? lp : "clover-k3-logits.bin", "wb");
      fwrite(nrm, 4, E, f); fwrite(logits, 4, VOCAB, f); fclose(f); }

    int am = 0;
    for (int i = 1; i < VOCAB; i++) if (logits[i] > logits[am]) am = i;

    if (prov_fp) {
        /* the top eight, so a prompt's answer can be judged for confidence and
           not just for which token won */
        int top[8];
        for (int r = 0; r < 8; r++) {
            int b = -1;
            for (int i = 0; i < VOCAB; i++) {
                int taken = 0;
                for (int z = 0; z < r; z++) if (top[z] == i) { taken = 1; break; }
                if (taken) continue;
                if (b < 0 || logits[i] > logits[b]) b = i;
            }
            top[r] = b;
            fprintf(prov_fp, "O\t%d\t%d\t%.9g\n", r, b, logits[b]);
        }
        fclose(prov_fp); prov_fp = NULL;
    }

    printf("\nsnapshots at the tail : %d sources\n", nsnap + 1);
    printf("total wall time       : %.2f s\n", now_s() - T0);
    { FILE *io = fopen("/proc/self/io", "r"); char k[64]; long long v;
      while (io && fscanf(io, "%63s %lld", k, &v) == 2)
          if (!strcmp(k, "read_bytes:"))
              printf("read_bytes            : %.2f GB   effective %.2f GB/s\n",
                     v / 1e9, v / 1e9 / (now_s() - T0));
      if (io) fclose(io); }
    printf("prefetch              : %s  %.2f s  %.2f GB  %.2f GB/s\n",
           pf_on ? "on" : "off", pf_secs, pf_bytes / 1e9,
           pf_secs > 0 ? pf_bytes / 1e9 / pf_secs : 0.0);
    if (n_store)
        printf("  of which SQLite     : %d layers  %.2f GB  %.2f thread-s  %.2f GB/s\n",
               n_store, sq_bytes / 1e9, sq_secs,
               sq_secs > 0 ? sq_bytes / 1e9 / sq_secs : 0.0);

    {
        double tot = 0.0; int64_t wb = 0;
        for (int i = 0; i < OP_COUNT; i++) { tot += op_t[i]; wb += op_wb[i]; }
        double wall = now_s() - T0;
        struct rusage ru; getrusage(RUSAGE_SELF, &ru);
        printf("\n--- where the time goes, by operator of k3-model-equation.md section 2 ---\n");
        printf("%-24s %10s %8s %14s %12s %10s %10s\n",
               "operator", "seconds", "% wall", "calls", "weight GB", "GB/s", "GFLOP/s");
        for (int i = 0; i < OP_COUNT; i++) {
            if (op_n[i] == 0) continue;
            printf("%-24s %10.3f %7.2f%% %14lld %12.2f %10.2f %10.1f\n", OPN[i], op_t[i],
                   100.0 * op_t[i] / wall, (long long)op_n[i], op_wb[i] / 1e9,
                   op_t[i] > 0 ? op_wb[i] / 1e9 / op_t[i] : 0.0,
                   op_t[i] > 0 ? op_fl[i] / 1e9 / op_t[i] : 0.0);
        }
        printf("%-24s %10.3f %7.2f%% %14s %12.2f\n", "SUM of operators", tot,
               100.0 * tot / wall, "", wb / 1e9);
        {
            printf("\n--- what the reading buys: values produced ---\n");
            printf("%-24s %14s %14s %12s\n",
                   "operator", "output floats", "weight B/out", "flops/out");
            for (int i = 0; i < OP_COUNT; i++) {
                if (op_out[i] == 0) continue;
                printf("%-24s %14lld %14.1f %12.1f\n", OPN[i], (long long)op_out[i],
                       (double)op_wb[i] / (double)op_out[i],
                       (double)op_fl[i] / (double)op_out[i]);
            }
            if (val_on) {
                printf("%-24s %10s %10s %10s %10s %10s %10s\n", "magnitude of output",
                       "exact 0", "<1e-6", "<1e-3", "<1", "<1e3", ">=1e3");
                for (int i = 0; i < OP_COUNT; i++) {
                    int64_t s = 0;
                    for (int b = 0; b < 6; b++) s += vh[i][b];
                    if (s == 0) continue;
                    printf("%-24s", OPN[i]);
                    for (int b = 0; b < 6; b++) printf(" %9.4f%%", 100.0 * vh[i][b] / s);
                    printf("\n");
                }
            }
        }
        printf("%-24s %10.3f %7.2f%%\n", "prefetch (pure I/O)", pf_secs,
               100.0 * pf_secs / wall);
        printf("%-24s %10.3f %7.2f%%   waiting on reader threads\n",
               "pipeline stall", pl_wait, 100.0 * pl_wait / wall);
        printf("%-24s first expert %.3f   next %d experts %.3f   rest %.3f\n",
               "  stall split", pl_wait0, nreader - 1, pl_waitE, pl_waitL);
        { int64_t tot = 0, drw = 0;
          for (int q = 1; q <= NPOS; q++) { tot += mhist[q]; drw += mhist[q] * q; }
          if (tot) {
              printf("Xm calls by position count:");
              for (int q = 1; q <= NPOS; q++)
                  printf("  m=%d:%lld(%.1f%%)", q, (long long)mhist[q],
                         100.0 * mhist[q] / tot);
              printf("   distinct %lld  draws %lld  mean m %.3f\n",
                     (long long)tot, (long long)drw, (double)drw / tot);
          } }
        printf("%-24s %10.3f %7.2f%%   unattributed\n", "wall - ops - prefetch",
               wall - tot - pf_secs, 100.0 * (wall - tot - pf_secs) / wall);
        printf("  of which: stall %.3f   slot_vec %.3f (%lld calls, %.1fM floats)"
               "   progress print %.3f   residual %.3f\n",
               pl_wait, sv_secs, (long long)sv_n, sv_elem / 1e6, pr_secs,
               wall - tot - pf_secs - pl_wait - sv_secs - pr_secs);
        printf("faults: %ld major, %ld minor\n", ru.ru_majflt, ru.ru_minflt);
        if (situstat && si_n) {
            printf("SiTU inputs: %lld elements   g [%.3f, %.3f]   u [%.3f, %.3f]\n",
                   (long long)si_n, si_gmin, si_gmax, si_umin, si_umax);
            printf("  |g|>34.7 (tanh sat) %.4f%%   g>17 (sig->1) %.4f%%   g<-17 %.4f%%"
                   "   |u|>216.6 (tanh sat) %.4f%%\n",
                   100.0 * si_gsat / si_n, 100.0 * si_ssat_hi / si_n,
                   100.0 * si_ssat_lo / si_n, 100.0 * si_usat / si_n);
        }
        if (hs_n) {
            const double fr[4] = {10, 25, 50, 75};
            const double th[4] = {1e-6, 1e-4, 1e-2, 1e-1};
            printf("\n--- realized |SiTU| concentration over %lld expert-position pairs ---\n",
                   (long long)hs_n);
            for (int q = 0; q < 4; q++)
                printf("   top %4.0f%% of neurons carry %6.2f%% of sum|h|\n",
                       fr[q], 100.0 * hs_top[q] / (double)hs_n);
            for (int q = 0; q < 4; q++)
                printf("   neurons below %7.0e x max : %6.2f%%\n",
                       th[q], 100.0 * hs_tiny[q] / (double)hs_n);
        }
        if (hs2_n) {
            const double fr[5] = {10, 25, 50, 75, 90};
            printf("\n--- ranking neurons by the w1-only gate factor, over %lld pairs ---\n",
                   (long long)hs2_n);
            for (int q = 0; q < 5; q++)
                printf("   keep top %4.0f%% by gate -> %6.2f%% of sum|h| ; frees %5.2f%% of w3 blocks\n",
                       fr[q], 100.0 * hs2_keep[q] / (double)hs2_n,
                       100.0 * hs2_blk[q] / (double)hs2_n);
        }
        if (hs3_pairs) {
            printf("\n--- do positions sharing an expert switch off the SAME neurons? ---\n");
            printf("   experts used by >1 position : %lld\n", (long long)hs3_n);
            printf("   position pairs compared     : %lld\n", (long long)hs3_pairs);
            printf("   bottom-half gate sets share : %6.2f%%\n",
                   100.0 * hs3_ov / (double)hs3_pairs);
            printf("   two independent halves would: 50.00%%\n");
            printf("   best static order frees     : %6.2f%% of w3 blocks (in-sample upper bound)\n",
                   100.0 * hs3_blk / (double)hs3_bn);
        }
    }
    printf("emitted token         : %d   engine emitted 17374   %s\n",
           am, am == 17374 ? "MATCH" : "DIFFERENT");
    if (covc) {
        size_t np = trunk_sz >> 12, touched = 0, once = 0, many = 0;
        long long reads = 0; unsigned mx = 0;
        for (size_t i = 0; i < np; i++) {
            if (!covc[i]) continue;
            touched++; reads += covc[i];
            if (covc[i] == 1) once++; else many++;
            if (covc[i] > mx) mx = covc[i];
        }
        printf("trunk coverage        : %.2f GB of %.2f GB touched (%.2f%%)\n",
               touched * 4096.0 / 1e9, trunk_sz / 1e9, 100.0 * touched / np);
        printf("  never read          : %.2f GB (%zu pages)\n",
               (np - touched) * 4096.0 / 1e9, np - touched);
        printf("  read exactly once   : %.2f GB (%zu pages)\n", once * 4096.0 / 1e9, once);
        printf("  read more than once : %.2f GB (%zu pages)  max %u\n",
               many * 4096.0 / 1e9, many, mx);
        printf("  total page reads    : %lld = %.2f GB of traffic\n",
               reads, reads * 4096.0 / 1e9);
        /* Attribute the untouched pages, so "the equation does not need all of the
           stored trunk" can name what it does not need. */
        {
            long long unc[N_SLOTN]; long long usedb[N_SLOTN]; long long gap = 0;
            for (int s = 0; s < N_SLOTN; s++) { unc[s] = 0; usedb[s] = 0; }
            char *claimed = calloc(np, 1);
            for (int L = 0; L < NLAY; L++)
                for (int s = 0; s < n_slots && s < N_SLOTN; s++) {
                    Slot *sl = &slots[(size_t)L * n_slots + s];
                    if (!sl->present) continue;
                    size_t p0 = (size_t)sl->off >> 12;
                    size_t p1 = (size_t)(sl->off + sl->nbytes + 4095) >> 12;
                    if (p1 > np) p1 = np;
                    for (size_t i = p0; i < p1; i++) {
                        claimed[i] = 1;
                        if (covc[i]) usedb[s] += 4096; else unc[s] += 4096;
                    }
                }
            for (size_t i = 0; i < np; i++) if (!claimed[i] && !covc[i]) gap += 4096;
            printf("  untouched, by slot  :\n");
            for (int s = 0; s < N_SLOTN; s++)
                if (unc[s] > 0)
                    printf("      %-8s %8.3f GB untouched of %8.3f GB claimed\n",
                           SLOTN[s], unc[s] / 1e9, (unc[s] + usedb[s]) / 1e9);
            printf("      %-8s %8.3f GB  (padding between slots, claimed by none)\n",
                   "GAP", gap / 1e9);
            free(claimed);
        }
    }
    printf("span: start->T0 %.2f s   timed %.2f s\n", T0 - T_start, now_s() - T0);

    /* Exit unmaps these anyway; doing it here moves the cost where a clock can see it. */
    {
        double u0 = now_s();
        if (arena) munmap(arena, arena_cap);
        double u1 = now_s();
        if (trunk_ram)
            for (int L = 0; L < NLAY; L++)
                if (slice[L]) munmap(slice[L], slice_map_n[L]);
        double u2 = now_s();
        printf("unmap: arena %.3f s (%.2f GB)   trunk %.3f s (%.2f GB)   total %.3f s\n",
               u1 - u0, arena_cap / 1e9, u2 - u1, trunk_ram ? tsz / 1e9 : 0.0, u2 - u0);
    }
    fflush(stdout);
    return 0;
}
