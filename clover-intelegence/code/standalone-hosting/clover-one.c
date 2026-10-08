/* clover-k3.c - k3-model-equation.md, sections 2 to 5, in C.
 *
 * Standalone. No cache manager, no prefetcher, no scheduler. It reads the
 * weights, evaluates the equation for one prompt, and prints the token.
 *
 * Metadata comes from eqidx.bin, produced by dump_eqidx.py, because parsing
 * two JSON indexes and 93 safetensors headers is not part of the equation.
 *
 * build: ./build.sh, or
 *   gcc -O3 -march=native -ffp-contract=off -fopenmp -DNPOS=<n> \
 *       -o clover-k3 clover-k3.c -lm
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
#include <poll.h>
#include <time.h>
#include "numeric-table.h"
#define NORMALIZATION_NO_MAIN
#include "normalization-values.h"
#include "generation.h"
#include "live-root.h"
#include "head-cache.h"
#include "cache-budget.h"
#include "expert-pipeline.h"
#include "cross-layer-prefetch.h"
#include "stage-profile.h"
#include "expert-results.h"

static char resident_dataset[4096];
static ClientInput *resident_input;
static ClientOutput *resident_output;
static Normalization *resident_leaves;
static GenerationConfig resident_config;
static HeadCache resident_head;
static CacheBudget resident_cache_budget;
static ExpertPipeline resident_pipeline;
static CrossLayerPrefetch resident_cross_layer;
static ExpertResults resident_results;
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ---------------------------------------------------------------- 1.2 shapes */
#define E      7168
#ifndef NPOS_SLOTS
#define NPOS_SLOTS 1
#endif
static int NPOS = 1;   /* positions active in this pass; NPOS_SLOTS bounds every per-position array */
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
static size_t trunk_sz;
static uint16_t *covc;          /* K3_COVER: touches per 4 KB page of the trunk */
static FILE *dumplay;           /* K3_DUMPLAY: each layer input, for the prefix check */

/* Prefix reuse. Positions below TLO are not recomputed; their contribution
   reaches the rest through the cached attention KV and recurrent state.
   TLO is 0 everywhere except a load run, so the normal path is unchanged. */
static int TLO;
static int pfx_n;               /* how many positions the cache covers */
static int pfx_mode;            /* 0 off, 1 save, 2 load */
static FILE *pfx_f;
/* K3_PFXOUT: write a cache covering every position this run computed, which
   may include ones it loaded. Saving and loading are otherwise exclusive, so
   without this a decode step cannot produce the cache the next step needs. */
static FILE *pfx_of;

/* Counts how often each trunk page is read, to answer whether a run needs all
   54.47 GB and whether it reads any of it more than once. */
static inline void cover(const void *p, size_t n)
{
    if (!covc || !trunk) return;
    uintptr_t a = (uintptr_t)p, b = (uintptr_t)trunk;
    if (a < b || a >= b + trunk_sz) return;
    size_t s = (a - b) >> 12, e = (a - b + n + 4095) >> 12, np = trunk_sz >> 12;
    if (e > np) e = np;
    for (size_t i = s; i < e; i++) if (covc[i] < 65535) covc[i]++;
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
    if (!trunk || L < 0) return "?";
    for (int s = 0; s < n_slots && s < N_SLOTN; s++) {
        const Slot *sl = &slots[(size_t)L * n_slots + s];
        if (sl->present && trunk + sl->off == W) return SLOTN[s];
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

static void resident_path(char *path, size_t capacity, const char *relative)
{
    int length = snprintf(path, capacity, "%s/%s", resident_dataset, relative);
    if (length < 0 || (size_t)length >= capacity) die("central dataset path too long");
}

static void resident_configure(const char *directory)
{
    char path[4096];
    if (!directory) directory = getenv("CLOVER_DATASET");
    if (!directory || !*directory) {
        ssize_t length = readlink("/proc/self/exe", path, sizeof(path) - 1);
        if (length <= 0 || (size_t)length >= sizeof(path) - 1) die("executable path unavailable");
        path[length] = 0;
        char *separator = strrchr(path, '/');
        if (!separator || (size_t)(separator - path) + sizeof("/dataset") > sizeof(path)) die("executable dataset path");
        strcpy(separator, "/dataset");
        directory = path;
    }
    char *resolved = realpath(directory, NULL);
    if (!resolved || strlen(resolved) >= sizeof(resident_dataset)) die("central dataset root unavailable");
    strcpy(resident_dataset, resolved);
    free(resolved);
    const char *names[] = {"K3_INDEX", "K3_PREPARED_DATA", "K3_OPERATOR_DIRECTORY", "K3_TRUNK0_QKV"};
    const char *suffixes[] = {"eqidx.bin", "", "operators/qkv-all", "operators/trunk-0-qkv/qkv.bin"};
    for (unsigned index = 0; index < 4; index++) {
        resident_path(path, sizeof(path), suffixes[index]);
        if (setenv(names[index], path, 1)) die("central dataset configuration failed");
    }
}

static void resident_values_open(void)
{
    char path[4096];
    resident_path(path, sizeof(path), "inputs/seed.bin");
    resident_input = client_input_open(path);
    resident_path(path, sizeof(path), "outputs/fruit.bin");
    resident_output = client_output_open(path);
    resident_path(path, sizeof(path), "leaves.json");
    if (!resident_input || !resident_output || !normalization_open(path, &resident_leaves))
        die("current input/output/leaves dataset validation failed");
}

static void resident_values_close(void)
{
    client_input_close(resident_input);
    client_output_close(resident_output);
    normalization_close(resident_leaves);
    resident_input = NULL;
    resident_output = NULL;
    resident_leaves = NULL;
}

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

/* Trunk and operator weights are the same bytes for every prompt, so they are read
   once into anonymous huge pages. Left as file mappings they are page-cache backed,
   so expert streaming evicts them and the next request faults them again 4 KB at a
   time; anonymous pages cannot be evicted while RAM holds. Falls back to the mapping
   when the allocation will not fit, which keeps a small machine working. */
static unsigned char *payload_file(const char *path, size_t *sz)
{
    static int enabled = -1;
    if (enabled < 0) {
        const char *setting = getenv("K3_PAYLOAD_RAM");
        if (setting && strcmp(setting, "0") && strcmp(setting, "1"))
            die("K3_PAYLOAD_RAM must be 0 or 1");
        enabled = !setting || strcmp(setting, "0");
    }
    if (!enabled) return map_file(path, sz);
    int fd = open(path, O_RDONLY);
    if (fd < 0) die(path);
    struct stat st;
    if (fstat(fd, &st)) die("fstat");
    size_t bytes = (size_t)st.st_size;
    size_t rounded = (bytes + 4095) & ~(size_t)4095;
    void *p = mmap(NULL, rounded, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) { close(fd); return map_file(path, sz); }
#ifdef MADV_HUGEPAGE
    (void)madvise(p, rounded, MADV_HUGEPAGE);
#endif
    /* O_DIRECT lands the bytes in the arena without the page-cache copy, which is the
       difference between 6 GB/s and 11 GB/s here. It needs 4 KB granularity, so the
       sub-block tail is read buffered. */
    int direct = open(path, O_RDONLY | O_DIRECT);
    size_t bulk = direct >= 0 ? (bytes & ~(size_t)4095) : 0;
    const size_t chunk = 32u << 20;
    const long units = (long)((bulk + chunk - 1) / chunk);
    int ok = 1;
#pragma omp parallel for schedule(dynamic, 1) num_threads(8) reduction(&:ok)
    for (long unit = 0; unit < units; unit++) {
        size_t base = (size_t)unit * chunk, want = bulk - base < chunk ? bulk - base : chunk, got = 0;
        while (got < want) {
            ssize_t n = pread(direct, (unsigned char *)p + base + got, want - got, (off_t)(base + got));
            if (n <= 0) break;
            got += (size_t)n;
        }
        if (got != want) ok = 0;
    }
    if (direct >= 0) close(direct);
    for (size_t offset = bulk; ok && offset < bytes; ) {
        ssize_t n = pread(fd, (unsigned char *)p + offset, bytes - offset, (off_t)offset);
        if (n <= 0) { ok = 0; break; }
        offset += (size_t)n;
    }
    close(fd);
    if (!ok) { munmap(p, rounded); return map_file(path, sz); }
    *sz = bytes;
    return (unsigned char *)p;
}

static int use_huge = 1;   /* 0 = 4 KB, 1 = THP (madvise-only here), 2 = hugetlb pool */
static size_t trunk_map_n; /* hugetlb rounds the mapping up, so munmap needs its own length */

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
    trunk_map_n = map_n;

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
    fprintf(stderr, "trunk -> RAM: %.2f GB in %.2f s = %.2f GB/s (%s, %s)\n",
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
static PRange g_rg[NPOS_SLOTS * TOPK * 6];
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

/* K3_ARENA2: two fixed halves rather than one growing buffer, so layer L+1 can
   be read into one while layer L is still being multiplied out of the other.
   Fixed size because a realloc would move bytes a reader is writing into. */
static size_t pf_half;      /* 0 = one buffer, as before */
static size_t pf_base;      /* where the current layer's ranges start */

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

/* One range, O_DIRECT where the alignment allows and buffered for the tail. */
static void read_range(PRange *g)
{
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
static int  pl_e[NPOS_SLOTS * TOPK];
static volatile int pl_done[NPOS_SLOTS * TOPK];
static volatile int pl_next;
static volatile int pl_rnext, pl_nrange;
static volatile int pl_rem[NPOS_SLOTS * TOPK];
static PRange *pl_rg[NPOS_SLOTS * TOPK * 6];
static int pl_gran = 1;            /* 0 = one expert per reader, 1 = one range */
static pthread_t pl_th[NREADER];
static double pl_wait = 0.0, pl_wait0 = 0.0, pl_waitE = 0.0, pl_waitL = 0.0;
static int64_t mhist[NPOS_SLOTS + 1];

/* Selection index. The old code scanned a growing list to dedup each draw and
   then rescanned the whole table once per expert; both are one pass here.
   Order is preserved exactly: experts in first-appearance order, positions
   ascending, which is what the prefetch pipeline and route_fp depend on. */
static int sx_epoch[NEXP], sx_slot[NEXP], sx_ep = 0;
static int sx_head[NPOS_SLOTS * TOPK], sx_tail[NPOS_SLOTS * TOPK];
static int sx_next[NPOS_SLOTS * TOPK], sx_t[NPOS_SLOTS * TOPK], sx_j[NPOS_SLOTS * TOPK];

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

/* ------------------------------------------------------- routing cache */
/* Measured: routing at position t is a pure function of tokens 0..t, so the 16
   ids per (layer, position) are a cache keyed by the prompt, not a prediction.
   It is used to start reads earlier and for nothing else - the router still
   runs every time and every cached row is compared against what it produced,
   so a wrong cache is a crash, never a wrong answer. */
static uint16_t *route_tab;
static int       route_load_on, route_save_on;
static int64_t   route_checked, route_bad;

static void route_alloc(void)
{
    if (!route_tab) route_tab = calloc((size_t)NLAY * NPOS_SLOTS * TOPK, sizeof(uint16_t));
    if (!route_tab) die("route table");
}

static int route_file_rw(const char *path, int write, const int *ids)
{
    FILE *f = fopen(path, write ? "wb" : "rb");
    if (!f) return -1;
    uint32_t hdr[4] = { 0x5452334BU, (uint32_t)NLAY, (uint32_t)NPOS, (uint32_t)TOPK };
    uint32_t kid[NPOS_SLOTS];
    const size_t n = (size_t)NLAY * NPOS_SLOTS * TOPK;
    int rc = 0;
    if (write) {
        for (int i = 0; i < NPOS; i++) kid[i] = (uint32_t)ids[i];
        if (fwrite(hdr, sizeof hdr, 1, f) != 1) rc = -1;
        if (fwrite(kid, sizeof kid, 1, f) != 1) rc = -1;
        if (fwrite(route_tab, sizeof(uint16_t), n, f) != n) rc = -1;
    } else {
        uint32_t got[4];
        if (fread(got, sizeof got, 1, f) != 1) rc = -1;
        else if (memcmp(got, hdr, sizeof hdr)) { fprintf(stderr,
                 "route cache: header is for a different model or prompt shape\n"); rc = -1; }
        else if (fread(kid, sizeof kid, 1, f) != 1) rc = -1;
        else {
            for (int i = 0; i < NPOS && !rc; i++)
                if (kid[i] != (uint32_t)ids[i]) { fprintf(stderr,
                    "route cache: built for different tokens, ignoring\n"); rc = -1; }
            if (!rc && fread(route_tab, sizeof(uint16_t), n, f) != n) rc = -1;
        }
    }
    fclose(f);
    return rc;
}

static void route_note(int L, int t, const int *sel)
{
    uint16_t *row = route_tab + ((size_t)L * NPOS_SLOTS + t) * TOPK;
    if (route_save_on)
        for (int j = 0; j < TOPK; j++) row[j] = (uint16_t)sel[j];
    if (route_load_on) {
        route_checked++;
        for (int j = 0; j < TOPK; j++)
            if (row[j] != (uint16_t)sel[j]) {
                route_bad++;
                fprintf(stderr, "route cache MISMATCH layer %d pos %d rank %d: "
                        "cached %u, router chose %d\n", L, t, j, row[j], sel[j]);
                die("route cache does not match the router");
            }
    }
}

/* -------------------------------------------- one layer of expert lookahead */
/* The device is idle ~1.75 s of an 8.75 s run - exactly the 18.9 ms per layer
   that attention, Q, the router and the trunk take while no expert read is
   queued, because pl_start cannot fire until the router has produced the ids.
   With the routing cached the ids are known a layer early, so that gap can be
   filled. */
static PRange       nx_rg[NPOS_SLOTS * TOPK * 6];
static int          nx_n, nx_L = -1, nx_on, nx_nread = 7;
static pthread_t    nx_th[NREADER];
static volatile int nx_next;
static double       nx_wait;
static int64_t      nx_bytes, nx_hit, nx_miss;

static void *nx_reader(void *u)
{
    (void)u;
    for (;;) {
        const int i = __atomic_fetch_add(&nx_next, 1, __ATOMIC_RELAXED);
        if (i >= nx_n) break;
        read_range(&nx_rg[i]);
    }
    return NULL;
}

static int nx_build(int L)
{
    if (L < 1 || L >= NLAY || !route_tab) return 0;
    static PRange tmp[NPOS_SLOTS * TOPK * 6];
    int n = 0;
    for (int t = 0; t < NPOS; t++) {
        const uint16_t *row = route_tab + ((size_t)L * NPOS_SLOTS + t) * TOPK;
        for (int j = 0; j < TOPK; j++)
            for (int w = 0; w < 3; w++)
                for (int k = 0; k < 2; k++) {
                    const ERec *rr = expert_rec(L, (int)row[j], w, k);
                    tmp[n].fid = rr->file_id; tmp[n].off = rr->off;
                    tmp[n].nb = rr->nbytes; n++;
                }
    }
    qsort(tmp, n, sizeof(PRange), pr_cmp);
    int m = 0;
    for (int i = 0; i < n; i++)
        if (m == 0 || tmp[i].fid != tmp[m - 1].fid || tmp[i].off != tmp[m - 1].off)
            tmp[m++] = tmp[i];

    size_t need = 0;
    const size_t base = (size_t)(L & 1) * pf_half;
    for (int i = 0; i < m; i++) {
        tmp[i].aoff = tmp[i].off & ~(int64_t)4095;
        tmp[i].alen = (size_t)((tmp[i].off + tmp[i].nb - tmp[i].aoff + 4095) & ~(int64_t)4095);
        tmp[i].apos = base + need;
        need += tmp[i].alen;
        tmp[i].mem = arena + tmp[i].apos + (size_t)(tmp[i].off - tmp[i].aoff);
        nx_bytes += tmp[i].nb;
    }
    if (need > pf_half) die("lookahead layer exceeds arena half");
    /* opening a file is not thread safe, so it happens here and not in a reader */
    for (int i = 0; i < m; i++)
        if (tmp[i].fid >= 0 && dfd[tmp[i].fid] < 0)
            dfd[tmp[i].fid] = open(fpath[tmp[i].fid], O_RDONLY | O_DIRECT);
    memcpy(nx_rg, tmp, (size_t)m * sizeof(PRange));
    return m;
}

static void nx_begin(int L)
{
    if (!nx_on || !route_load_on || !pf_half) return;
    nx_n = nx_build(L);
    if (nx_n <= 0) { nx_L = -1; return; }
    nx_L = L; nx_next = 0;
    for (int i = 0; i < nx_nread; i++) pthread_create(&nx_th[i], NULL, nx_reader, NULL);
}

static void nx_join(void)
{
    for (int i = 0; i < nx_nread; i++) pthread_join(nx_th[i], NULL);
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
        pf_base = pf_half ? (size_t)(cur_L & 1) * pf_half : 0;
        for (int i = 0; i < m; i++) {
            r[i].aoff = r[i].off & ~(int64_t)4095;
            r[i].alen = (size_t)((r[i].off + r[i].nb - r[i].aoff + 4095) & ~(int64_t)4095);
            r[i].apos = pf_base + need;
            need += r[i].alen;
        }
        if (pf_half) { if (need > pf_half) die("arena half too small for a layer"); }
        else arena_reserve(need);
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
        unsigned char *p = (r[i].fid < 0 ? trunk : fmap[r[i].fid]) + r[i].off;
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

typedef struct {
    char magic[8]; uint32_t version, layer, count, reserved;
    uint64_t lengths[3]; unsigned char hashes[3][32]; float palette[256];
} PreparedHeader;
typedef struct {
    uint32_t id, group, kind, rows, columns, count;
    uint64_t offset, length; unsigned char hash[32];
} PreparedRecord;
_Static_assert(sizeof(PreparedHeader) == 1168, "prepared header layout");
_Static_assert(sizeof(PreparedRecord) == 72, "prepared record layout");
static PreparedHeader prepared_header;
static PreparedRecord prepared_records[40];
static const PreparedRecord *prepared_slots[40];
static unsigned char *prepared_groups[3];
static const char *prepared_directory;
static int prepared_layer = -1;
static unsigned prepared_trunk_count;

static void prepared_init(void)
{
    prepared_directory = getenv("K3_PREPARED_DATA");
    if (!prepared_directory || !*prepared_directory) die("K3_PREPARED_DATA is required");
}

static void prepared_close(void)
{
    if (prepared_layer < 0) die("prepared layer not open");
    memset(prepared_groups, 0, sizeof(prepared_groups));
    memset(prepared_slots, 0, sizeof(prepared_slots));
    prepared_layer = -1;
}

static void prepared_load(int layer)
{
    if (prepared_layer >= 0 || layer < 0 || layer >= NLAY) die("prepared layer lifetime");
    char path[4096];
    int length = snprintf(path, sizeof(path), "%s/trunk-%d/index.bin", prepared_directory, layer);
    if (length < 0 || (size_t)length >= sizeof(path)) die("prepared index path");
    size_t bytes;
    unsigned char *index = map_file(path, &bytes);
    if (bytes < sizeof(prepared_header)) die("short prepared index");
    memcpy(&prepared_header, index, sizeof(prepared_header));
    unsigned count = layer == 0 ? 26 : (layer == 92 || layer % 4 == 3) ? 24 : 31;
    if (memcmp(prepared_header.magic, "K3TRK001", 8) || prepared_header.version != 1 ||
        prepared_header.layer != (unsigned)layer || prepared_header.count != count ||
        prepared_header.reserved || bytes != sizeof(prepared_header) + count * sizeof(PreparedRecord))
        die("prepared index identity");
    memcpy(prepared_records, index + sizeof(prepared_header), count * sizeof(PreparedRecord));
    if (munmap(index, bytes)) die("prepared index unmap");
    const char *names[3] = {"common.bin", layer ? "attention.bin" : "kda.bin",
                          layer ? "routed-shared.bin" : "dense.bin"};
    for (unsigned group = 0; group < 3; group++) {
        length = snprintf(path, sizeof(path), "%s/trunk-%d/%s", prepared_directory, layer, names[group]);
        if (length < 0 || (size_t)length >= sizeof(path)) die("prepared group path");
        prepared_groups[group] = payload_file(path, &bytes);
        if (bytes != prepared_header.lengths[group]) die("prepared group size");
    }
    uint64_t ends[3] = {0};
    for (unsigned position = 0; position < count; position++) {
        const PreparedRecord *record = &prepared_records[position];
        if (record->id >= 40 || prepared_slots[record->id] || record->group >= 3 ||
            !record->rows || !record->columns || (uint64_t)record->rows * record->columns != record->count ||
            record->kind < 1 || record->kind > 3 || record->offset % 4)
            die("prepared record identity");
        uint64_t wanted = record->kind == 1 ? record->rows * 4ULL + record->count : record->count * 4ULL;
        if (record->length != wanted || record->offset != ends[record->group] ||
            record->offset > prepared_header.lengths[record->group] ||
            record->length > prepared_header.lengths[record->group] - record->offset)
            die("prepared record bounds");
        if (record->id < 37) {
            const Slot *original = &slots[(size_t)layer * n_slots + record->id];
            if (!original->present || (record->kind != 2 &&
                (original->d0 != (int)record->rows || original->d1 != (int)record->columns)))
                die("prepared/source slot shape");
        }
        ends[record->group] += record->length;
        prepared_slots[record->id] = record;
    }
    for (unsigned group = 0; group < 3; group++)
        if (ends[group] != prepared_header.lengths[group]) die("prepared group coverage");
    prepared_layer = layer;
    prepared_trunk_count++;
}

typedef struct {
    char magic[8];
    uint32_t version, layer, width, rows, head, components, taps, records, record_bytes, reserved;
    float rms_epsilon, l2_epsilon;
    uint64_t bytes;
} OperatorHeader;
typedef struct {
    uint32_t slot, kind, rows, width;
    uint64_t offset, length;
} OperatorRecord;
typedef struct {
    const unsigned char *scales, *codes;
    const float *palette;
    size_t scale_stride, row_stride;
} OperatorMatrix;
_Static_assert(sizeof(OperatorHeader) == 64, "operator header ABI");
_Static_assert(sizeof(OperatorRecord) == 32, "operator record ABI");
static const unsigned char *operator_data;
static size_t operator_bytes;
static OperatorHeader operator_header;
static const float *operator_palette;
static const unsigned char *operator_pointers[37];
static const float *operator_vectors[37];
static unsigned operator_counts[37], operator_hits[37];
static OperatorMatrix operator_matrices[37];
static float operator_taps[3][P * KC];
static unsigned operator_layers, operator_matrices_used, operator_vectors_used;

static void operator_load(int layer)
{
    if (operator_data || prepared_layer != layer) die("operator layer lifetime");
    const char *root = getenv("K3_OPERATOR_DIRECTORY");
    const char *first = getenv("K3_TRUNK0_QKV");
    if (!root || !*root || !first || !*first) die("operator paths required");
    char path[4096];
    int length = layer ? snprintf(path, sizeof(path), "%s/layer-%d/operator.bin", root, layer)
                       : snprintf(path, sizeof(path), "%s", first);
    if (length < 0 || (size_t)length >= sizeof(path)) die("operator path length");
    operator_data = payload_file(path, &operator_bytes);
    if (operator_bytes < sizeof(operator_header)) die("short operator header");
    memcpy(&operator_header, operator_data, sizeof(operator_header));
    int mla = layer == 92 || layer % 4 == 3;
    const uint32_t expected[10] = {1, (unsigned)layer, E, mla ? H * QH : P,
        mla ? QH : D, 3, mla ? 0 : KC, mla ? 7 : 6, mla ? 32 : E + 20, 0};
    if (memcmp(operator_header.magic, mla ? "K3MLA001" : "K3QKV001", 8) ||
        memcmp(&operator_header.version, expected, sizeof(expected)) ||
        operator_header.bytes != operator_bytes ||
        operator_header.rms_epsilon != (float)EPS5 || operator_header.l2_epsilon != (float)EPS6)
        die("operator header or arithmetic contract differs");
    if (!mla) {
        size_t payload = 64 + 24 + E * 4 + 1024;
        size_t component_bytes = E + 20;
        const uint32_t program[6] = {1, 2, 3, 4, 5, 6};
        if (operator_bytes != payload + (size_t)P * 3 * component_bytes ||
            memcmp(operator_data + 64, program, sizeof(program))) die("KDA operator layout");
        operator_palette = (const float *)(operator_data + 88 + E * 4);
        operator_vectors[S_IN_LN] = (const float *)(operator_data + 88);
        operator_counts[S_IN_LN] = E;
        for (unsigned component = 0; component < 3; component++) {
            unsigned slot = S_Q + component;
            const unsigned char *record = operator_data + payload + component * component_bytes;
            operator_pointers[slot] = record;
            operator_matrices[slot] = (OperatorMatrix){record, record + 20, operator_palette,
                                                       3 * component_bytes, 3 * component_bytes};
            for (unsigned row = 0; row < P; row++) {
                const unsigned char *taps = record + (size_t)row * 3 * component_bytes + 4;
                memcpy(operator_taps[component] + row * KC, taps + 4, 3 * sizeof(float));
                memcpy(operator_taps[component] + row * KC + 3, taps, sizeof(float));
            }
            operator_vectors[S_CQ + component] = operator_taps[component];
            operator_counts[S_CQ + component] = P * KC;
        }
    } else {
        const uint32_t schema[7][4] = {{4,2,E,1},{20,1,QLORA,E},{21,2,QLORA,1},
            {22,1,H*QH,QLORA},{23,1,KVW,E},{24,2,KVL,1},{25,1,H*KVD,KVL}};
        size_t next = 64 + 7 * sizeof(OperatorRecord) + 1024;
        if (operator_bytes < next) die("short MLA operator descriptors");
        operator_palette = (const float *)(operator_data + 64 + 7 * sizeof(OperatorRecord));
        for (unsigned index = 0; index < 7; index++) {
            OperatorRecord record;
            memcpy(&record, operator_data + 64 + index * sizeof(record), sizeof(record));
            uint64_t bytes = record.kind == 1 ? (uint64_t)record.rows * (record.width + 4ULL)
                                             : (uint64_t)record.rows * record.width * 4;
            if (memcmp(&record, schema[index], 16) || record.offset != next || record.length != bytes ||
                next > operator_bytes || bytes > operator_bytes - next) die("MLA operator record layout");
            const unsigned char *values = operator_data + next;
            if (record.kind == 1) {
                operator_pointers[record.slot] = values;
                operator_matrices[record.slot] = (OperatorMatrix){values, values + (size_t)record.rows * 4,
                                                                   operator_palette, 4, record.width};
            } else {
                operator_vectors[record.slot] = (const float *)values;
                operator_counts[record.slot] = record.rows * record.width;
            }
            next += (size_t)bytes;
        }
        if (next != operator_bytes) die("MLA unclaimed operator bytes");
    }
    operator_layers++;
}

static const float *operator_vector(unsigned slot, unsigned count)
{
    if (slot >= 37 || !operator_vectors[slot] || operator_counts[slot] != count)
        die("operator vector binding differs");
    operator_hits[slot]++;
    operator_vectors_used++;
    return operator_vectors[slot];
}

static OperatorMatrix operator_matrix_view(const unsigned char *pointer, int width, int rows)
{
    for (unsigned slot = 0; slot < 37; slot++) {
        if (!operator_pointers[slot] || operator_pointers[slot] != pointer) continue;
        const PreparedRecord *record = prepared_slots[slot];
        if (!record || record->kind != 1 || record->rows != (unsigned)rows || record->columns != (unsigned)width)
            die("operator matrix shape differs from layer");
        operator_hits[slot]++;
        operator_matrices_used++;
        return operator_matrices[slot];
    }
    for (unsigned slot = 0; slot < 37; slot++) {
        const PreparedRecord *record = prepared_slots[slot];
        if (!record || prepared_groups[record->group] + record->offset != pointer) continue;
        if (record->kind != 1 || record->rows != (unsigned)rows || record->columns != (unsigned)width)
            die("original prepared matrix binding differs");
        return (OperatorMatrix){pointer, pointer + (size_t)rows * 4, prepared_header.palette, 4, (size_t)width};
    }
    die("matrix outside active parameter sources");
    return (OperatorMatrix){0};
}

static void operator_close(void)
{
    if (!operator_data) die("operator not open");
    unsigned vectors = 0, matrices = 0;
    for (unsigned slot = 0; slot < 37; slot++) {
        int used = operator_vectors[slot] || operator_pointers[slot];
        if (operator_hits[slot] != (unsigned)used) die("operator value first-use coverage differs");
        vectors += operator_vectors[slot] != NULL;
        matrices += operator_pointers[slot] != NULL;
    }
    printf("operator values       : layer=%d matrices=%u vectors=%u each_used_once=1\n",
        prepared_layer, matrices, vectors);
    operator_bytes = 0;
    operator_data = NULL;
    operator_palette = NULL;
    memset(operator_pointers, 0, sizeof(operator_pointers));
    memset(operator_vectors, 0, sizeof(operator_vectors));
    memset(operator_counts, 0, sizeof(operator_counts));
    memset(operator_hits, 0, sizeof(operator_hits));
    memset(operator_matrices, 0, sizeof(operator_matrices));
}
static const PreparedRecord *prepared_record(int layer, unsigned slot)
{
    if (prepared_layer != layer || slot >= 40 || !prepared_slots[slot]) die("prepared slot unavailable");
    return prepared_slots[slot];
}

static const unsigned char *prepared_pointer(int layer, unsigned slot)
{
    const PreparedRecord *record = prepared_record(layer, slot);
    if (slot < 37 && operator_pointers[slot]) return operator_pointers[slot];
    return prepared_groups[record->group] + record->offset;
}

static const float *prepared_vector(int layer, unsigned slot, unsigned count)
{
    const PreparedRecord *record = prepared_record(layer, slot);
    if (record->kind != 2 || record->count != count) die("prepared vector shape");
    if (slot < 37 && operator_vectors[slot]) return operator_vector(slot, count);
    return (const float *)prepared_pointer(layer, slot);
}

static const float *prepared_router(unsigned row)
{
    const PreparedRecord *record = prepared_record(prepared_layer, S_GATE);
    if (record->kind != 3 || record->rows != NEXP || record->columns != E || row >= NEXP)
        die("prepared router shape");
    return (const float *)prepared_pointer(prepared_layer, S_GATE) + (size_t)row * E;
}

static void prepared_matrix_check(const unsigned char *pointer, int width, int rows)
{
    for (unsigned slot = 0; slot < 37; slot++) {
        const PreparedRecord *record = prepared_slots[slot];
        if (record && prepared_pointer(prepared_layer, slot) == pointer) {
            if (record->kind != 1 || record->rows != (unsigned)rows || record->columns != (unsigned)width)
                die("prepared matrix shape");
            return;
        }
    }
    die("unbound prepared matrix");
}

static const unsigned char *slot_ptr(int L, int s)
{
    Slot *sl = &slots[(size_t)L * n_slots + s];
    if (!sl->present) die("absent slot");
    if (prov_fp) prov_slot(L, s, "ptr");
    return prepared_pointer(L, (unsigned)s);
}

/* dequantize any slot to float32. dtype 0=F32 1=BF16 2=I8R */
static double sv_secs; static int64_t sv_n, sv_elem;   /* no operator counts this */
static int gfuse = 1;   /* K3_GFUSE: fold the gate's scale into the router dot product */
static float *slot_vec(int L, int s, int want)
{
    const double started = now_s();
    const float *values = prepared_vector(L, (unsigned)s, (unsigned)want);
    float *output = malloc((size_t)want * sizeof(float));
    if (!output) die("prepared vector allocation");
    memcpy(output, values, (size_t)want * sizeof(float));
    sv_secs += now_s() - started; sv_n++; sv_elem += want;
    return output;
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

static int xdec = 2, pfc_on;

/* The qkv palette is palette[c] == (float)(int8_t)c on every KDA layer, so the
   gather reproduces a sign-extend. Verified at run time rather than assumed,
   because the MLA layers carry a different table. */
static int palette_identity(const float *palette)
{
    static const float *seen;
    static int verdict;
    if (palette == seen) return verdict;
    int identity = 1;
    for (int code = 0; code < 256; code++)
        if (palette[code] != (float)(int8_t)(unsigned char)code) { identity = 0; break; }
    seen = palette;
    verdict = identity;
    return identity;
}

/* ---- batched forms: the weight is read once and applied to every position ----
   Each output element is still an independent reduction over i in the original
   order, so these are bit-identical to calling Q or X once per position. */

static void Qm(float *const *Y, const float *const *Xs, int T,
               const unsigned char *W, int in, int out)
{
    const double _t = now_s();
    prepared_matrix_check(W, in, out);
    OperatorMatrix matrix = operator_matrix_view(W, in, out);
    const float *palette = matrix.palette;
    const int direct_codes = palette_identity(palette);
    const size_t rowb = (size_t)4 + (size_t)in;
    /* An output-blocked variant used to run at T>=8. It hoisted the x load across
       eight output rows but carried the t loop outside, so it re-read every weight
       row T times, and it never took the direct int8 decode below. Weights are what
       binds here, so at eight lanes it cost 8.12 s a step against 1.45 s at four. */
#pragma omp parallel for schedule(static) if (out > 64)
    for (int o = 0; o < out; o++) {
        const unsigned char *row = matrix.scales + (size_t)o * matrix.scale_stride;
        float scale; memcpy(&scale, row, 4);
        const unsigned char *w = matrix.codes + (size_t)o * matrix.row_stride;
        int i = 0; float acc[NPOS_SLOTS];
#if defined(__AVX2__)
        __m256 v0[NPOS_SLOTS], v1[NPOS_SLOTS];
        for (int t = 0; t < T; t++) { v0[t] = _mm256_setzero_ps(); v1[t] = _mm256_setzero_ps(); }
        if (direct_codes) {
            for (; i + 15 < in; i += 16) {
                const __m256 w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(w + i))));
                const __m256 w1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(w + i + 8))));
                for (int t = 0; t < T; t++) {
                    v0[t] = _mm256_fmadd_ps(w0, _mm256_loadu_ps(Xs[t] + i), v0[t]);
                    v1[t] = _mm256_fmadd_ps(w1, _mm256_loadu_ps(Xs[t] + i + 8), v1[t]);
                }
            }
        } else
        for (; i + 15 < in; i += 16) {
            const __m256 w0 = _mm256_i32gather_ps(palette, _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(w + i))), 4);
            const __m256 w1 = _mm256_i32gather_ps(palette, _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(w + i + 8))), 4);
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
        float B[NPOS_SLOTS][16];
        for (int t = 0; t < T; t++) for (int c = 0; c < 16; c++) B[t][c] = 0.0f;
        for (; i + 15 < in; i += 16)
            for (int t = 0; t < T; t++)
                for (int c = 0; c < 16; c++)
                    B[t][c] = fmaf(palette[w[i + c]], Xs[t][i + c], B[t][c]);
        for (int t = 0; t < T; t++) {
            float A[8];
            for (int j = 0; j < 8; j++) A[j] = B[t][j] + B[t][j + 8];
            float l0 = A[0] + A[4], l1 = A[1] + A[5], l2 = A[2] + A[6], l3 = A[3] + A[7];
            acc[t] = (l0 + l2) + (l1 + l3);
        }
#endif
        for (int t = 0; t < T; t++) {
            float a = acc[t];
            for (int k = i; k < in; k++) a = a + palette[w[k]] * Xs[t][k];
            Y[t][o] = a * scale;
        }
    }
    op_add(OP_Q, _t, (int64_t)out * (int64_t)rowb);
    op_fl[OP_Q] += 2 * (int64_t)out * (int64_t)in * (int64_t)T;
    PROFILE_PROJECTION(W,now_s()-_t);
    vstat(OP_Q, Y, T, out);
    if (stg_fp) stg_emit("Q", slot_of_ptr(cur_L, W), -1, T, in, out, Y);
    cover(W, (size_t)out * rowb);
}

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

/* Only the first snapshot, pushed at layer 0, is meant to follow the prompt. At the
   later snapshot layers the fold can take the residual at weight 1 instead of scoring
   the sources. Off unless CLOVER_SNAPFOLD=residual: the distributed model defaults it
   off too, and the two drifting apart cost a day once already. */
static int clover_snapfold_residual(int layer)
{
    static int decided, value;
    if (!decided) {
        const char *setting = getenv("CLOVER_SNAPFOLD");
        value = setting && !strcmp(setting, "residual");
        decided = 1;
    }
    return value && layer > 0 && layer % 12 == 0;
}

/* Each snapshot after the first can become the previous one plus the residual, so the
   series is built by adding rather than by what the intervening layers produced.
   Off unless CLOVER_SNAPSHOT=add; measurement showed the additive scheme false
   (cos 0.0298 median against the residual, ratio 5.94). */
static int clover_snapshot_add(void)
{
    static int decided, value;
    if (!decided) {
        const char *setting = getenv("CLOVER_SNAPSHOT");
        value = setting && !strcmp(setting, "add");
        decided = 1;
    }
    return value;
}

/* Same record layout as the distributed capture in common/trace.h: owner, local,
   position, count as int32, then count floats. local is 2000 + stage so the two
   sides can be diffed without a translation step. */
static FILE *fold_sink;
static int fold_sink_tried;

static void fold_record(int owner, int stage, int position, const float *weights, int count)
{
    if (!fold_sink_tried) {
        const char *path = getenv("CLOVER_FOLD_DUMP");
        fold_sink_tried = 1;
        if (path && *path) fold_sink = fopen(path, "wb");
    }
    if (!fold_sink) return;
    int32_t header[4] = {owner, 2000 + stage, position, count};
    if (fwrite(header, sizeof header, 1, fold_sink) == 1)
        (void)fwrite(weights, sizeof *weights, (size_t)count, fold_sink);
    (void)fflush(fold_sink);
}

/* AR: snapshot aggregation, float32 source-major weighted sum */
static void AR(float *out, float *const *srcs, int nsrc, const float *fold, float *weights_out)
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
    if (weights_out) memcpy(weights_out, pis, (size_t)nsrc * sizeof *pis);
#pragma omp parallel for schedule(static) if (par2 & 8)
    for (int i = 0; i < E; i++) {
        float o = 0.0f;
        for (int s = 0; s < nsrc; s++) o = o + pis[s] * srcs[s][i];
        out[i] = o;
    }
    op_add(OP_AR, _t, (int64_t)E * 4);
}

/* ================================================================ state */
static float  resid[NPOS_SLOTS][E];
static float *snap[NPOS_SLOTS][16];
static int    nsnap = 0;
static int    have_prefix = 1;

/* KDA recurrent state and conv history. These point straight at the owning
   lane's storage, so a batch of lanes carries independent state with no copy. */
enum { ST_FLOATS = H * D * D, CONV_FLOATS = 3 * P * (KC - 1) };
#define ST_BYTES   (sizeof(float) * ST_FLOATS)
#define CONV_BYTES (sizeof(float) * CONV_FLOATS)
static float (*St)[D][D];
static float (*convbuf)[P][KC - 1];

static float  x1b[NPOS_SLOTS][E], x2b[NPOS_SLOTS][E], hb[NPOS_SLOTS][E], h2b[NPOS_SLOTS][E];
static float  aout[NPOS_SLOTS][E], ffn[NPOS_SLOTS][E];

/* MLA cache */
static float  mla_klat[NPOS_SLOTS][H * QN], mla_v[NPOS_SLOTS][H * VH], mla_rp[NPOS_SLOTS][QR];

/* Per-position scratch, allocated once instead of ~176,000 malloc/free pairs a
   run. Laid out BUFFER-major: every position of one buffer is contiguous, which
   is how Qm walks them. Position-major cost 1.4 s at 64 tokens by spreading each
   buffer's positions 389 KB apart. */
static float mp_zlm[NPOS_SLOTS][LAT], mp_nlm[NPOS_SLOTS][LAT];
static float mp_rom[NPOS_SLOTS][E],   mp_shm[NPOS_SLOTS][E];
static float mp_sgm[NPOS_SLOTS][SI],  mp_sum2[NPOS_SLOTS][SI];
static float mp_egm[NPOS_SLOTS][I_],  mp_eum[NPOS_SLOTS][I_];
static float mp_slot[NPOS_SLOTS][TOPK][LAT];
static float ap_raw[3][NPOS_SLOTS][P];
static float ap_beta[NPOS_SLOTS][H], ap_fa[NPOS_SLOTS][D];
static float ap_zz[NPOS_SLOTS][P], ap_gtm[NPOS_SLOTS][P], ap_gtf[NPOS_SLOTS][P];
static float ap_qlm[NPOS_SLOTS][QLORA], ap_qnm[NPOS_SLOTS][QLORA];
static float ap_craw[NPOS_SLOTS][KVW], ap_cc[NPOS_SLOTS][KVW];
static float ap_kvm[NPOS_SLOTS][H * KVD];
static float ap_gbm[NPOS_SLOTS][H * VH], ap_gbf[NPOS_SLOTS][H * VH], ap_accm[NPOS_SLOTS][H * VH];
static float qs_pool[(size_t)NPOS_SLOTS * H * QH] __attribute__((aligned(64)));
static float snap_pool[16][NPOS_SLOTS][E] __attribute__((aligned(64)));

/* Expert-major grouping: the distinct experts a batch needs, and who asked for each. */
_Static_assert(NPOS_SLOTS <= ROOT_MAX_ROWS, "batched expert projection bounds the slot count");
static float expert_down[NPOS_SLOTS][TOPK][LAT] __attribute__((aligned(64)));
static uint64_t expert_hash[NPOS_SLOTS];
static unsigned expert_hit[NPOS_SLOTS][TOPK];
static unsigned group_expert[NPOS_SLOTS * TOPK], group_members[NPOS_SLOTS * TOPK];
static unsigned group_position[NPOS_SLOTS * TOPK][NPOS_SLOTS];
static unsigned group_rank[NPOS_SLOTS * TOPK][NPOS_SLOTS];
static uint64_t resident_group_passes, resident_group_pairs;

static const unsigned char *recorded_roots, *recorded_inputs;
static size_t recorded_root_bytes, recorded_input_bytes;
static unsigned recorded_expert_hits;
static int recorded_layer = -1;
static unsigned recorded_qkv_hits, recorded_shared_hits;
static const float *recorded_qkv_values, *recorded_shared_values;

static void result_miss(const char *message)
{
    fprintf(stderr, "RESULT_MISS: %s\n", message);
    exit(3);
}

static void recorded_open(int layer)
{
    if (recorded_layer == layer) return;
    if (recorded_roots) {
        if (munmap((void *)recorded_roots, recorded_root_bytes) ||
            munmap((void *)recorded_inputs, recorded_input_bytes)) die("result unmap");
    }
    const char *directory = getenv("K3_PREPARED_DATA");
    const char *name = getenv("K3_RESULT_SET");
    if (!directory || !name || (strcmp(name, "france") && strcmp(name, "japan")))
        result_miss("unsupported observation set");
    char path[4096];
    int length = snprintf(path, sizeof(path), "%s/root-%d/observations/%s/results.bin", directory, layer, name);
    if (length < 0 || (size_t)length >= sizeof(path)) die("result path");
    recorded_roots = map_file(path, &recorded_root_bytes);
    length = snprintf(path, sizeof(path), "%s/root-%d/observations/%s/inputs.f32", directory, layer, name);
    if (length < 0 || (size_t)length >= sizeof(path)) die("result input path");
    recorded_inputs = map_file(path, &recorded_input_bytes);
    if (recorded_root_bytes != 80ULL * 51204 || recorded_input_bytes != 80ULL * LAT * 4)
        result_miss("observation layout differs");
    recorded_layer = layer;
}

static const float *recorded_down(int layer, int expert, const float *input)
{
    recorded_open(layer);
    for (unsigned record = 0; record < 80; record++) {
        const unsigned char *values = recorded_roots + (size_t)record * 51204;
        int32_t identity;
        memcpy(&identity, values, sizeof(identity));
        if (identity == expert && !memcmp(recorded_inputs + (size_t)record * LAT * 4, input, LAT * 4)) {
            recorded_expert_hits++;
            return (const float *)(values + 4 + 3 * I_ * 4);
        }
    }
    fprintf(stderr, "result binding: layer=%d expert=%d\n", layer, expert);
    result_miss("no stored result matches the live expert input");
    return NULL;
}

/* The observation sets hold gate/up/activation/down for the experts a prompt selects.
   A record is usable only when its stored input is the same vector the live step is
   about to project: over 7,360 measured steps every exact input match reproduced the
   stored down bit for bit, while matching on expert id alone differed by up to 2.2. */
static const unsigned char *stored_results[NLAY];
static const float *stored_inputs[NLAY];
static int stored_state;
static unsigned long long stored_expert_hits;

static void stored_open(void)
{
    const char *name = getenv("K3_RESULT_SET");
    const char *root = getenv("K3_PREPARED_DATA");
    if (!name || !root || (strcmp(name, "france") && strcmp(name, "japan"))) { stored_state = -1; return; }
    char path[4096];
    size_t bytes;
    for (unsigned layer = 1; layer < NLAY; layer++) {
        snprintf(path, sizeof path, "%s/root-%u/observations/%s/results.bin", root, layer, name);
        stored_results[layer] = map_file(path, &bytes);
        if (bytes != 80ULL * 51204) die("observation results layout");
        snprintf(path, sizeof path, "%s/root-%u/observations/%s/inputs.f32", root, layer, name);
        stored_inputs[layer] = (const float *)map_file(path, &bytes);
        if (bytes != 80ULL * LAT * 4) die("observation inputs layout");
    }
    stored_state = 1;
}

static int stored_expert_down(unsigned layer, unsigned expert, const float *input, float *out)
{
    if (!stored_state) stored_open();
    if (stored_state < 0 || layer < 1 || layer >= NLAY) return 0;
    for (unsigned record = 0; record < 80; record++) {
        const unsigned char *values = stored_results[layer] + (size_t)record * 51204;
        int32_t identity;
        memcpy(&identity, values, sizeof identity);
        if ((unsigned)identity != expert) continue;
        if (memcmp(stored_inputs[layer] + (size_t)record * LAT, input, LAT * 4)) continue;
        memcpy(out, values + 4 + 3 * I_ * 4, LAT * 4);
        stored_expert_hits++;
        return 1;
    }
    return 0;
}

static void result_options(void)
{
    const char *forbidden[] = {"K3_STAGE", "K3_LSTAT", "K3_DUMPROUTE", "K3_DUMPSEL", "K3_PROV",
        "K3_DUMPLAY", "K3_PFXSAVE", "K3_PFXLOAD", "K3_PFXOUT", "K3_ROUTESAVE", "K3_ROUTELOAD",
        "K3_ARENA2", "K3_LOGITS", "K3_DUMPRES", "K3_COVER", "K3_TRUNKRAM", "K3_NX", "K3_PREFETCH"};
    for (unsigned setting = 0; setting < sizeof(forbidden) / sizeof(forbidden[0]); setting++)
        if (getenv(forbidden[setting])) result_miss("unsupported persistence or raw-reader option");
    if (NPOS < 1) result_miss("at least one active token per step");
    pf_on = 0;
}

static void recorded_contract(const int *tokens)
{
    const char *setting = getenv("K3_RESULT_CONTRACT_FD");
    if (!setting) result_miss("matching-result launcher contract required");
    char *end;
    long descriptor = strtol(setting, &end, 10);
    if (*end || descriptor < 3 || descriptor > INT32_MAX) result_miss("invalid contract descriptor");
    unsigned char contract[28];
    size_t received = 0;
    while (received < sizeof(contract)) {
        ssize_t count = read((int)descriptor, contract + received, sizeof(contract) - received);
        if (count <= 0) result_miss("incomplete recorded-input contract");
        received += (size_t)count;
    }
    close((int)descriptor);
    if (NPOS != 5 || memcmp(contract, "K3RES001", 8)) result_miss("unsupported result contract");
    for (unsigned position = 0; position < 5; position++) {
        int32_t expected;
        memcpy(&expected, contract + 8 + position * 4, 4);
        if (tokens[position] != expected) result_miss("input has no matching completed trunk results");
    }
}

static void recorded_qkv(int layer, int position, float *query, float *key, float *value)
{
    if (layer < 0 || layer >= NLAY || position < 0 || position >= 5) result_miss("QKV result unavailable");
    if (!recorded_qkv_values) {
        const char *path = getenv("K3_QKV_RESULTS");
        size_t bytes;
        if (!path) result_miss("QKV result path required");
        recorded_qkv_values = (const float *)map_file(path, &bytes);
        if (bytes != 74465280) result_miss("QKV result layout differs");
    }
    size_t offset = 0;
    for (int previous = 0; previous < layer; previous++)
        offset += (size_t)5 * ((previous == 92 || previous % 4 == 3) ? 49152 : 36864);
    unsigned width = layer == 92 || layer % 4 == 3 ? 18432 : 12288;
    const float *result = recorded_qkv_values + offset + (size_t)position * (2 * width + P);
    memcpy(query, result, width * sizeof(float));
    memcpy(key, result + width, width * sizeof(float));
    memcpy(value, result + 2 * width, P * sizeof(float));
    recorded_qkv_hits++;
}

static const float *recorded_shared(int layer, int position)
{
    if (layer < 1 || layer > 92 || position < 0 || position >= 5 || (layer == 92 && position != 4))
        result_miss("shared result unavailable");
    if (!recorded_shared_values) {
        const char *path = getenv("K3_SHARED_RESULTS");
        size_t bytes;
        if (!path) result_miss("shared result path required");
        recorded_shared_values = (const float *)map_file(path, &bytes);
        if (bytes != 46694400) result_miss("shared result layout differs");
    }
    unsigned record = layer == 92 ? 455 : (unsigned)(layer - 1) * 5 + (unsigned)position;
    recorded_shared_hits++;
    return recorded_shared_values + (size_t)record * 25600 + 3 * SI;
}
typedef struct {
    PreparedHeader prepared;
    PreparedRecord records[40];
    unsigned char *groups[3];
    OperatorHeader header;
    const unsigned char *data;
    size_t bytes;
    const float *palette;
    const unsigned char *pointers[37];
    const float *vectors[37];
    unsigned counts[37];
    OperatorMatrix matrices[37];
    float *taps;
} ResidentLayer;

static ResidentLayer resident_layers[NLAY];
static int resident_ready;
static unsigned resident_maps, resident_unmaps, resident_tap_layers;
static size_t resident_payload_bytes;
static double resident_setup_seconds, resident_warmup_seconds;
static volatile unsigned resident_touch_value;

typedef struct {
    float *state, *convolution, *keys, *values, *positions;
    unsigned length;
} ResidentSequence;

/* A lane is one in-flight request: its own attention cache and recurrent state.
   Slots stay the batch dimension; a lane owns one slot while decoding and all of
   them while prefilling. */
#ifndef K3_LANES
#define K3_LANES 1
#endif
_Static_assert(K3_LANES >= 1 && K3_LANES <= NPOS_SLOTS, "lane count must fit the slot array");
static ResidentSequence resident_sequences[K3_LANES][NLAY];
static unsigned lane_position[K3_LANES];   /* absolute position reached, per lane */
static unsigned pos_lane[NPOS_SLOTS];      /* lane owning each active slot */
static unsigned pos_abs[NPOS_SLOTS];       /* absolute position of each active slot */
static unsigned active_lane;               /* lane being prefilled, when not batching */
static int      batch_decode;              /* set while slot t means lane t, one new position each */
static unsigned batch_next[NPOS_SLOTS];    /* argmax per slot, filled by every evaluation */
#define SEQ(slot, layer) (&resident_sequences[pos_lane[(slot)]][(layer)])
static Root *resident_roots[NLAY];
static RootScratch *resident_root_scratch;
static unsigned resident_position;
static unsigned long long resident_expert_calls;
static unsigned long long resident_result_hits;
static float resident_gate[I_], resident_up[I_], resident_down[LAT];
#ifdef CLOVER_CHECK_EXPERT
static void check_live_expert(unsigned layer, unsigned expert, const float *input, const float *output);
#endif

static void resident_sequence_open(void)
{
    unsigned capacity=resident_config.max_input_tokens+resident_config.max_output_tokens;
    if (!capacity) capacity=GENERATION_CAPACITY;
    resident_root_scratch=calloc((size_t)omp_get_max_threads(),sizeof *resident_root_scratch);
    if (!resident_root_scratch) die("root scratch allocation failed");
    { const char *setting=getenv("K3_EXPERT_CACHE");
      long experts=setting?strtol(setting,NULL,10):0;
      if (experts>0 && !root_cache_open(NLAY,(unsigned)experts))
          die("per-layer expert cache allocation failed"); }
    for (unsigned layer=0; layer<NLAY; layer++) {
        for (unsigned lane=0; lane<K3_LANES; lane++) {
        ResidentSequence *sequence=&resident_sequences[lane][layer];
        if (layer==92 || layer%4==3) {
            sequence->keys=calloc((size_t)capacity*H*QN,sizeof(float));
            sequence->values=calloc((size_t)capacity*H*VH,sizeof(float));
            sequence->positions=calloc((size_t)capacity*QR,sizeof(float));
            if (!sequence->keys || !sequence->values || !sequence->positions) die("MLA sequence allocation failed");
        } else {
            sequence->state=calloc(1,ST_BYTES);
            sequence->convolution=calloc(1,CONV_BYTES);
            if (!sequence->state || !sequence->convolution) die("KDA sequence allocation failed");
        }
        }
        if (layer) {
            char suffix[128],path[4096];
            snprintf(suffix,sizeof suffix,"root-%u",layer);
            resident_path(path,sizeof path,suffix);
            resident_roots[layer]=root_open(path,layer);
            if (!resident_roots[layer]) die("root metadata or stored constants invalid");
        }
    }
}

static void resident_shared(float *gate,float *up,float *output,const float *input,
    const unsigned char *gate_weights,const unsigned char *up_weights,const unsigned char *down_weights)
{
    Qm(&gate,&input,1,gate_weights,E,SI);
    Qm(&up,&input,1,up_weights,E,SI);
    situ(gate,gate,up,SI);
    Qm(&output,(const float *const *)&gate,1,down_weights,SI,E);
}

/* The last dense weights still read once per position; batched like every other one. */
static void resident_shared_rows(float *const *gate,float *const *up,float *const *output,
    const float *const *input,int count,
    const unsigned char *gate_weights,const unsigned char *up_weights,const unsigned char *down_weights)
{
    Qm(gate,input,count,gate_weights,E,SI);
    Qm(up,input,count,up_weights,E,SI);
    for (int entry=0;entry<count;entry++) situ(gate[entry],gate[entry],up[entry],SI);
    Qm(output,(const float *const *)gate,count,down_weights,SI,E);
}

static const float *resident_expert(unsigned layer,unsigned expert,const float *input)
{
    double started=now_s();
    Root *root=resident_roots[layer];
    if (!root_project(root,resident_root_scratch,expert,0,input,resident_gate)) die("live expert gate failed");
    PROFILE_DURATION(layer,"detail:expert-gate",now_s()-started);
    PROFILE_CLOCK(phase);
    if (!root_project(root,resident_root_scratch,expert,1,input,resident_up)) die("live expert up failed");
    PROFILE_DURATION(layer,"detail:expert-up",now_s()-phase);
    PROFILE_RESTART(phase);
    situ(resident_gate,resident_gate,resident_up,I_);
    PROFILE_DURATION(layer,"detail:expert-activation",now_s()-phase);
    PROFILE_RESTART(phase);
    if (!root_project(root,resident_root_scratch,expert,2,resident_gate,resident_down)) die("live expert down failed");
    PROFILE_DURATION(layer,"detail:expert-down",now_s()-phase);
#ifdef CLOVER_CHECK_EXPERT
    check_live_expert(layer,expert,input,resident_down);
#endif
    resident_expert_calls+=3;
    op_add(OP_X,started,0);
    return resident_down;
}

/* Same three projections, but every position that chose this expert rides the one
   weight pass. Each row keeps its own accumulators, so the values are unchanged. */
static float batch_gate[ROOT_MAX_ROWS][I_], batch_up[ROOT_MAX_ROWS][I_];
static void resident_expert_rows(unsigned layer,unsigned expert,
    const float *const *inputs,float *const *outputs,unsigned count)
{
    double started=now_s();
    Root *root=resident_roots[layer];
    float *gate_rows[ROOT_MAX_ROWS],*up_rows[ROOT_MAX_ROWS];
    const float *activated[ROOT_MAX_ROWS];
    for (unsigned entry=0;entry<count;entry++) {
        gate_rows[entry]=batch_gate[entry];
        up_rows[entry]=batch_up[entry];
        activated[entry]=batch_gate[entry];
    }
    if (!root_project_rows(root,resident_root_scratch,expert,0,inputs,gate_rows,count)) die("live expert gate failed");
    PROFILE_DURATION(layer,"detail:expert-gate",now_s()-started);
    PROFILE_CLOCK(phase);
    if (!root_project_rows(root,resident_root_scratch,expert,1,inputs,up_rows,count)) die("live expert up failed");
    PROFILE_DURATION(layer,"detail:expert-up",now_s()-phase);
    PROFILE_RESTART(phase);
    for (unsigned entry=0;entry<count;entry++) situ(batch_gate[entry],batch_gate[entry],batch_up[entry],I_);
    PROFILE_DURATION(layer,"detail:expert-activation",now_s()-phase);
    PROFILE_RESTART(phase);
    if (!root_project_rows(root,resident_root_scratch,expert,2,activated,outputs,count)) die("live expert down failed");
    PROFILE_DURATION(layer,"detail:expert-down",now_s()-phase);
#ifdef CLOVER_CHECK_EXPERT
    for (unsigned entry=0;entry<count;entry++) check_live_expert(layer,expert,inputs[entry],outputs[entry]);
#endif
    resident_expert_calls+=3*count;
    op_add(OP_X,started,0);
}

static void resident_caches_open(void)
{
    FILE *file=fopen("/proc/meminfo","r");
    uint64_t available=0;
    if (file) {
        char line[256]; unsigned long long kib;
        while (fgets(line,sizeof line,file))
            if (sscanf(line,"MemAvailable: %llu kB",&kib)==1) { available=(uint64_t)kib*1024; break; }
        fclose(file);
    }
    resident_cache_budget=cache_budget(&resident_config,available,resident_payload_bytes);
    if (!head_cache_open(&resident_head,resident_output,resident_cache_budget.head)) die("head cache initialization failed");
    if (!expert_pipeline_open(&resident_pipeline,resident_cache_budget.pipeline)) die("expert pipeline initialization failed");
    memset(&resident_cross_layer,0,sizeof resident_cross_layer);
    if (!expert_results_open(&resident_results,resident_cache_budget.results)) die("expert result cache initialization failed");
}

typedef struct {
    uint64_t head_decodes,head_hits;
    uint64_t pipeline_reads,pipeline_bytes;
    double pipeline_wait;
    uint64_t result_hits,result_misses,result_evictions,result_stores;
    uint64_t cross_submitted,cross_hits,cross_misses;
    double cross_prediction;
} ResidentCacheStats;
static ResidentCacheStats resident_cache_stats(void)
{
    ResidentCacheStats stats={0};
    stats.head_decodes=resident_head.decoded_blocks; stats.head_hits=resident_head.reused_blocks;
    stats.pipeline_reads=resident_pipeline.reads; stats.pipeline_bytes=resident_pipeline.bytes;
    stats.pipeline_wait=resident_pipeline.wait_seconds;
    stats.result_hits=resident_results.hits; stats.result_misses=resident_results.misses;
    stats.result_evictions=resident_results.evictions; stats.result_stores=resident_results.stores;
    stats.cross_submitted=resident_cross_layer.submitted;
    stats.cross_hits=resident_cross_layer.hits; stats.cross_misses=resident_cross_layer.misses;
    stats.cross_prediction=resident_cross_layer.prediction_seconds;
    return stats;
}

static void resident_cache_report(unsigned request,ResidentCacheStats before)
{
    ResidentCacheStats after=resident_cache_stats();
    printf("CACHE_JSON {\"request\":%u,"
        "\"head_decoded_blocks\":%llu,\"head_reused_blocks\":%llu,\"head_ram_bytes\":%zu,"
        "\"pipeline_reads\":%llu,\"pipeline_bytes\":%llu,\"pipeline_wait_seconds\":%.9f,\"pipeline_ram_bytes\":%zu,"
        "\"result_hits\":%llu,\"result_misses\":%llu,\"result_stores\":%llu,\"result_evictions\":%llu,"
        "\"result_cache_bytes\":%zu,\"result_cache_entries\":%zu,"
        "\"cross_layer_submitted\":%llu,\"cross_layer_hits\":%llu,\"cross_layer_misses\":%llu,"
        "\"cross_layer_prediction_seconds\":%.9f}\n",
        request,
        (unsigned long long)(after.head_decodes-before.head_decodes),(unsigned long long)(after.head_hits-before.head_hits),
        resident_head.allocated,(unsigned long long)(after.pipeline_reads-before.pipeline_reads),
        (unsigned long long)(after.pipeline_bytes-before.pipeline_bytes),after.pipeline_wait-before.pipeline_wait,
        resident_pipeline.buffers[0].capacity+resident_pipeline.buffers[1].capacity,
        (unsigned long long)(after.result_hits-before.result_hits),(unsigned long long)(after.result_misses-before.result_misses),
        (unsigned long long)(after.result_stores-before.result_stores),(unsigned long long)(after.result_evictions-before.result_evictions),
        resident_results.allocated,resident_results.sets*RESULT_WAYS,
        (unsigned long long)(after.cross_submitted-before.cross_submitted),
        (unsigned long long)(after.cross_hits-before.cross_hits),(unsigned long long)(after.cross_misses-before.cross_misses),
        after.cross_prediction-before.cross_prediction);
    fflush(stdout);
}

static void resident_detach(void)
{
    memset(prepared_slots, 0, sizeof(prepared_slots));
    memset(prepared_groups, 0, sizeof(prepared_groups));
    prepared_layer = -1;
    operator_data = NULL;
    operator_palette = NULL;
    memset(operator_pointers, 0, sizeof(operator_pointers));
    memset(operator_vectors, 0, sizeof(operator_vectors));
    memset(operator_counts, 0, sizeof(operator_counts));
    memset(operator_hits, 0, sizeof(operator_hits));
    memset(operator_matrices, 0, sizeof(operator_matrices));
}

static void resident_touch(const unsigned char *data, size_t bytes)
{
    long page = sysconf(_SC_PAGESIZE);
    if (page <= 0) die("cannot determine page size");
    unsigned value = 0;
    for (size_t offset = 0; offset < bytes; offset += (size_t)page) value ^= data[offset];
    if (bytes) value ^= data[bytes - 1];
    resident_touch_value ^= value;
}

static void resident_startup(int warmup)
{
    if (resident_ready) die("resident parameters already initialized");
    double started = now_s();
    resident_values_open();
    resident_sequence_open();
    for (int layer = 0; layer < NLAY; layer++) {
        prepared_load(layer);
        operator_load(layer);
        ResidentLayer *saved = &resident_layers[layer];
        saved->prepared = prepared_header;
        memcpy(saved->records, prepared_records, sizeof(saved->records));
        memcpy(saved->groups, prepared_groups, sizeof(saved->groups));
        saved->header = operator_header;
        saved->data = operator_data;
        saved->bytes = operator_bytes;
        saved->palette = operator_palette;
        memcpy(saved->pointers, operator_pointers, sizeof(saved->pointers));
        memcpy(saved->vectors, operator_vectors, sizeof(saved->vectors));
        memcpy(saved->counts, operator_counts, sizeof(saved->counts));
        memcpy(saved->matrices, operator_matrices, sizeof(saved->matrices));
        if (layer != 92 && layer % 4 != 3) {
            saved->taps = malloc(sizeof(operator_taps));
            if (!saved->taps) die("resident taps allocation failed");
            memcpy(saved->taps, operator_taps, sizeof(operator_taps));
            for (unsigned component = 0; component < 3; component++)
                saved->vectors[S_CQ + component] = saved->taps + (size_t)component * P * KC;
            resident_tap_layers++;
        }
        resident_maps += 4;
        resident_payload_bytes += saved->bytes;
        for (unsigned group = 0; group < 3; group++) resident_payload_bytes += saved->prepared.lengths[group];
        resident_detach();
    }
    resident_caches_open();
    resident_setup_seconds = now_s() - started;
    started = now_s();
    if (warmup) {
        for (int layer = 0; layer < NLAY; layer++) {
            ResidentLayer *saved = &resident_layers[layer];
            for (unsigned group = 0; group < 3; group++)
                resident_touch(saved->groups[group], (size_t)saved->prepared.lengths[group]);
            resident_touch(saved->data, saved->bytes);
        }
        for (int file = 0; file < n_files; file++)
            if (fmap[file]) resident_touch(fmap[file], fsize[file]);
    }
    resident_warmup_seconds = now_s() - started;
    if (resident_maps != NLAY * 4 || resident_tap_layers != 69) die("startup parameter coverage differs");
    resident_ready = 1;
}

static void prepared_open(int layer)
{
    if (!resident_ready || prepared_layer >= 0 || layer < 0 || layer >= NLAY)
        die("resident prepared layer binding");
    ResidentLayer *saved = &resident_layers[layer];
    prepared_header = saved->prepared;
    memcpy(prepared_records, saved->records, sizeof(prepared_records));
    memcpy(prepared_groups, saved->groups, sizeof(prepared_groups));
    memset(prepared_slots, 0, sizeof(prepared_slots));
    for (unsigned record = 0; record < prepared_header.count; record++)
        prepared_slots[prepared_records[record].id] = prepared_records + record;
    prepared_layer = layer;
    prepared_trunk_count++;
}

static void operator_open(int layer)
{
    if (!resident_ready || prepared_layer != layer || operator_data) die("resident operator layer binding");
    ResidentLayer *saved = &resident_layers[layer];
    operator_header = saved->header;
    operator_data = saved->data;
    operator_bytes = saved->bytes;
    operator_palette = saved->palette;
    memcpy(operator_pointers, saved->pointers, sizeof(operator_pointers));
    memcpy(operator_vectors, saved->vectors, sizeof(operator_vectors));
    memcpy(operator_counts, saved->counts, sizeof(operator_counts));
    memcpy(operator_matrices, saved->matrices, sizeof(operator_matrices));
    memset(operator_hits, 0, sizeof(operator_hits));
    operator_layers++;
}

static void resident_shutdown(void)
{
    if (!resident_ready || prepared_layer >= 0 || operator_data) die("resident shutdown while bound");
    expert_pipeline_close(&resident_pipeline);
    memset(&resident_cross_layer,0,sizeof resident_cross_layer);
    expert_results_close(&resident_results);
    for (int layer = 0; layer < NLAY; layer++) {
        ResidentLayer *saved = &resident_layers[layer];
        for (unsigned group = 0; group < 3; group++) {
            if (munmap(saved->groups[group], (size_t)saved->prepared.lengths[group])) die("shutdown trunk unmap");
            resident_unmaps++;
        }
        if (munmap((void *)saved->data, saved->bytes)) die("shutdown operator unmap");
        resident_unmaps++;
        free(saved->taps);
    }
    for (int file = 0; file < n_files; file++)
        if (fmap[file] && munmap(fmap[file], fsize[file])) die("shutdown global model unmap");
    if (recorded_roots) {
        if (munmap((void *)recorded_roots, recorded_root_bytes) ||
            munmap((void *)recorded_inputs, recorded_input_bytes)) die("shutdown expert observations unmap");
        recorded_roots = recorded_inputs = NULL;
        recorded_layer = -1;
    }
    free(slots);
    free(erec);
    for (unsigned layer=0; layer<NLAY; layer++) {
        for (unsigned lane=0; lane<K3_LANES; lane++) {
            ResidentSequence *sequence=&resident_sequences[lane][layer];
            free(sequence->state); free(sequence->convolution); free(sequence->keys);
            free(sequence->values); free(sequence->positions);
            memset(sequence,0,sizeof *sequence);
        }
        root_close(resident_roots[layer]); resident_roots[layer]=NULL;
    }
    free(resident_root_scratch); resident_root_scratch=NULL;
    root_cache_close();
    head_cache_close(&resident_head);
    resident_values_close();
    resident_ready = 0;
}
static void request_reset(void)
{
    if (!resident_ready || prepared_layer >= 0 || operator_data) die("request started while layer bound");
    if (resident_cross_layer.armed) die("cross-layer candidate escaped token boundary");
    nsnap = 0;
    have_prefix = 1;
    TLO = 0;
    cur_L = -1;
    memset(snap, 0, sizeof(snap));
    memset(snap_pool, 0, sizeof(snap_pool));
    memset(resid, 0, sizeof(resid));
    /* St and convbuf alias the owning lane's carried state; only a new sequence clears them. */
    memset(op_t, 0, sizeof(op_t));
    memset(op_n, 0, sizeof(op_n));
    memset(op_wb, 0, sizeof(op_wb));
    memset(op_fl, 0, sizeof(op_fl));
    memset(op_out, 0, sizeof(op_out));
    memset(vh, 0, sizeof(vh));
    operator_layers = operator_matrices_used = operator_vectors_used = prepared_trunk_count = 0;
    recorded_expert_hits = 0;
    sv_secs = pr_secs = 0;
    sv_n = sv_elem = 0;
}

static unsigned evaluate_tokens(const unsigned *tokens, unsigned count, unsigned position, int project)
{
    PROFILE_BOUNDARY(-1,NULL);
    double request_started = now_s();
    request_reset();
    int ids[NPOS_SLOTS];
    if (!count || count > NPOS_SLOTS) die("active position count out of range");
    NPOS = (int)count;
    for (unsigned entry = 0; entry < count; entry++) ids[entry] = (int)tokens[entry];
    /* One lane holding every slot while it prefills, or one slot per lane while a
       batch decodes. Either way a slot knows its lane and its absolute position. */
    if (batch_decode) {
        /* the scheduler has already mapped each slot to the lane it is decoding for */
        if (count > K3_LANES) die("decode batch wider than the lane count");
        for (unsigned entry = 0; entry < count; entry++)
            pos_abs[entry] = lane_position[pos_lane[entry]];
    } else {
        for (unsigned entry = 0; entry < count; entry++) {
            pos_lane[entry] = active_lane;
            pos_abs[entry]  = position + entry;
        }
    }
    resident_position=position;
    resident_expert_calls=0;
    resident_result_hits=0;
    unsigned context_capacity=resident_config.max_input_tokens+resident_config.max_output_tokens;
    if (!context_capacity) context_capacity=GENERATION_CAPACITY;
    for (unsigned entry = 0; entry < count; entry++)
        if (tokens[entry]>=VOCAB || position+entry>=context_capacity) die("token or context position out of range");
    g_ids = ids;
    const int last = NLAY - 1;
    unsigned maps_before = resident_maps, unmaps_before = resident_unmaps, taps_before = resident_tap_layers;
    PROFILE_BOUNDARY(-1,"request-reset");
    /* section 5, initial conditions: the embedding */
    for (int position = 0; position < NPOS; position++)
        if (!client_input(resident_input, (uint32_t)ids[position], resid[position]))
            die("input row could not be read from seed dataset");

            PROFILE_BOUNDARY(-1,"embedding");
    double T0 = now_s();

    float *fa = malloc(sizeof(float) * E), *fm = malloc(sizeof(float) * E);
    float *srcbuf[16];
    PROFILE_BOUNDARY(-1,"scratch-setup");

    for (int L = 0; L <= last; L++) {
        PROFILE_BOUNDARY(L,NULL);
        double t0 = now_s();
        cur_L = L;
        prepared_open(L);
        operator_open(L);
        const int isMLA = ((L % 4) == 3 && L <= 91) || (L == 92);
        const int isMoE = (L >= 1);
        for (int t = 0; t < NPOS; t++)
            if (SEQ(t,L)->length != pos_abs[t] - (batch_decode ? 0 : (unsigned)t))
                die("attention cache position mismatch");

        memcpy(fa, prepared_vector(L, 37, E), sizeof(float) * E);
        memcpy(fm, prepared_vector(L, 38, E), sizeof(float) * E);
        PROFILE_BOUNDARY(L,"binding-and-stored-folds");

        /* (1) pre-attention aggregation, guarded */
        for (int t = TLO; t < NPOS; t++) {
            if (nsnap > 0 && !clover_snapfold_residual(L)) {
                for (int s = 0; s < nsnap; s++) srcbuf[s] = snap[t][s];
                srcbuf[nsnap] = resid[t];
                float used[16];
                AR(hb[t], srcbuf, nsnap + 1, fa, used);
                fold_record(L, 3, t, used, nsnap + 1);
            } else {
                memcpy(hb[t], resid[t], sizeof(float) * E);
            }
        }
        PROFILE_BOUNDARY(L,"pre-attention-aggregation");
        /* (2) snapshot push */
        if (L % 12 == 0) {
            for (int t = TLO; t < NPOS; t++) {
                snap[t][nsnap] = snap_pool[nsnap][t];
                if (nsnap > 0 && clover_snapshot_add())
                    for (int i = 0; i < E; i++)
                        snap[t][nsnap][i] = snap[t][nsnap - 1][i] + resid[t][i];
                else
                    memcpy(snap[t][nsnap], resid[t], sizeof(float) * E);
            }
            nsnap++; have_prefix = 0;
        }
        PROFILE_BOUNDARY(L,"snapshot-push");
        /* (3) pre-attention norm */
        float *win = slot_vec(L, S_IN_LN, E);
        /* Prefix reuse needs the state to match at every layer, not just the
           last one, so this dumps each layer's input for comparison. */
        if (dumplay) { for (int t = 0; t < NPOS; t++) fwrite(hb[t], 4, E, dumplay); }
        for (int t = TLO; t < NPOS; t++) rmsnorm(x1b[t], hb[t], win, E, EPS5);
        free(win);
        PROFILE_BOUNDARY(L,"pre-attention-normalization");

        /* the attention block */
        if (isMLA) {
            const unsigned char *WQA = slot_ptr(L, S_QA), *WQB = slot_ptr(L, S_QB);
            const unsigned char *WKA = slot_ptr(L, S_KA), *WKB = slot_ptr(L, S_KB);
            const unsigned char *WG  = slot_ptr(L, S_G),  *WO  = slot_ptr(L, S_O);
            float *wqan = slot_vec(L, S_QAN, QLORA), *wkan = slot_vec(L, S_KAN, KVL);
            float *qs = qs_pool;
            float *qlm[NPOS_SLOTS], *qnm[NPOS_SLOTS], *crawm[NPOS_SLOTS], *ccm[NPOS_SLOTS], *kvm[NPOS_SLOTS];
            float *qsm[NPOS_SLOTS], *gbm[NPOS_SLOTS], *gbf[NPOS_SLOTS], *accm[NPOS_SLOTS];
            const float *x1p[NPOS_SLOTS]; float *aoutp[NPOS_SLOTS];
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
                for (int t = TLO; t < NPOS; t++) {
                    ResidentSequence *sequence = SEQ(t,L);
                    size_t slot = pos_abs[t];
                    memcpy(sequence->keys+slot*H*QN,mla_klat[t],sizeof mla_klat[t]);
                    memcpy(sequence->values+slot*H*VH,mla_v[t],sizeof mla_v[t]);
                    memcpy(sequence->positions+slot*QR,mla_rp[t],sizeof mla_rp[t]);
                }
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
            /* loaded positions and freshly computed ones are both in place here */
            if (pfx_of) {
                for (int t = 0; t < NPOS; t++) {
                    fwrite(mla_klat[t], 4, (size_t)H * QN, pfx_of);
                    fwrite(mla_v[t],    4, (size_t)H * VH, pfx_of);
                    fwrite(mla_rp[t],   4, QR, pfx_of);
                }
            }
            const float msc = 1.0f / sqrtf(192.0f);
            Qm(gbm + TLO, x1p + TLO, NACT, WG, E, H * VH);
            for (int t = TLO; t < NPOS; t++) {
                float *acc = accm[t];
                const ResidentSequence *sequence = SEQ(t,L);
                const unsigned last = pos_abs[t];
                const double _tsa = now_s();
#pragma omp parallel for schedule(static)
                for (int h = 0; h < H; h++) {
                    const float *qh = qs + (size_t)t * H * QH + (size_t)h * QH;
                    float scv[GENERATION_CAPACITY];
                    for (unsigned s = 0; s <= last; s++) {
                        const float *kl = sequence->keys + ((size_t)s * H + h) * QN;
                        const float *kr = sequence->positions + (size_t)s * QR;
                        double d = 0.0;
                        for (int i = 0; i < QN; i++) d += (double)qh[i] * (double)kl[i];
                        for (int i = 0; i < QR; i++) d += (double)qh[QN + i] * (double)kr[i];
                        scv[s] = (float)d * msc;
                    }
                    float m = scv[0];
                    for (unsigned s = 1; s <= last; s++) if (scv[s] > m) m = scv[s];
                    float ex[GENERATION_CAPACITY]; double z = 0.0;
                    for (unsigned s = 0; s <= last; s++) { ex[s] = expf(scv[s] - m); z += (double)ex[s]; }
                    float *o = acc + (size_t)h * VH;
                    for (int i = 0; i < VH; i++) o[i] = 0.0f;
                    for (unsigned s = 0; s <= last; s++) {
                        const float pr = (float)((double)ex[s] / z);
                        const float *vs = sequence->values + ((size_t)s * H + h) * VH;
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

            St      = (float (*)[D][D])     SEQ(TLO,L)->state;
            convbuf = (float (*)[P][KC - 1]) SEQ(TLO,L)->convolution;
            /* KDA carries a recurrent state rather than per-position KV, and it
               is a fixed size whatever the prefix length. */
            if (pfx_mode == 2) {
                if (fread(St, 1, ST_BYTES, pfx_f) != ST_BYTES ||
                    fread(convbuf, 1, CONV_BYTES, pfx_f) != CONV_BYTES)
                    die("prefix cache short read");
            }
            float ah[H];
            memcpy(ah, prepared_vector(L, 39, H), sizeof ah);

            float *rawm[3][NPOS_SLOTS], *betam[NPOS_SLOTS], *fam[NPOS_SLOTS], *zzm[NPOS_SLOTS];
            float *gtm[NPOS_SLOTS], *gtf[NPOS_SLOTS];
            const float *x1p[NPOS_SLOTS]; float *aoutp[NPOS_SLOTS];
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
                St      = (float (*)[D][D])     SEQ(t,L)->state;
                convbuf = (float (*)[P][KC - 1]) SEQ(t,L)->convolution;
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
                    fwrite(St, 1, ST_BYTES, pfx_f);
                    fwrite(convbuf, 1, CONV_BYTES, pfx_f);
                }
            }
            /* after the last position, so the next run resumes from NPOS */
            if (pfx_of) {
                fwrite(St, 1, ST_BYTES, pfx_of);
                fwrite(convbuf, 1, CONV_BYTES, pfx_of);
            }
            Qm(aoutp + TLO, (const float *const *)gtf + TLO, NACT, WO, P, E);

            for (int j = 0; j < 3; j++) { free(cw[j]); free(cv[j]); }
            free(alog); free(dtb); free(won); free(raw); free(zz); free(fatmp);
            free(beta); free(alpha); free(o); free(on); free(gt);
        }

        PROFILE_BOUNDARY(L,"attention");
        /* (4) residual: replace at a snapshot layer, add otherwise */
        for (int t = TLO; t < NPOS; t++) {
            if (have_prefix) for (int i = 0; i < E; i++) resid[t][i] = resid[t][i] + aout[t][i];
            else             memcpy(resid[t], aout[t], sizeof(float) * E);
        }
        have_prefix = 1;
        PROFILE_BOUNDARY(L,"attention-residual");

        /* (5) pre-MLP aggregation, unguarded, then norm */
        float *wpost = slot_vec(L, S_POST_LN, E);
        for (int t = TLO; t < NPOS; t++) {
            if (clover_snapfold_residual(L)) {
                memcpy(h2b[t], resid[t], sizeof(float) * E);
            } else {
                for (int s = 0; s < nsnap; s++) srcbuf[s] = snap[t][s];
                srcbuf[nsnap] = resid[t];
                float used[16];
                AR(h2b[t], srcbuf, nsnap + 1, fm, used);
                fold_record(L, 21, t, used, nsnap + 1);
            }
            rmsnorm(x2b[t], h2b[t], wpost, E, EPS5);
        }
        free(wpost);
        PROFILE_BOUNDARY(L,"pre-mlp-aggregation-and-normalization");

        /* the MLP */
        if (!isMoE) {
            const unsigned char *WG2 = slot_ptr(L, S_MGATE), *WU = slot_ptr(L, S_MUP);
            const unsigned char *WD = slot_ptr(L, S_MDOWN);
            /* Batched like every other weight: these three are 727 MB, and the
               per-position loop re-read all of it for each position. */
            const int NACT0 = NPOS - TLO;
            float *gm[NPOS_SLOTS], *um[NPOS_SLOTS], *fp[NPOS_SLOTS];
            const float *x2p0[NPOS_SLOTS];
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
            float *gw = NULL;
            const unsigned char *gp = slot_ptr(L, S_GATE);
            float *gbias = slot_vec(L, S_GBIAS, NEXP);
            const unsigned char *WDN = slot_ptr(L, S_EDOWN), *WUP = slot_ptr(L, S_EUP);
            const unsigned char *S1 = slot_ptr(L, S_SH1), *S3 = slot_ptr(L, S_SH3), *S2 = slot_ptr(L, S_SH2);
            float *lnw = slot_vec(L, S_ENORM, LAT);

            int idsel_all[NPOS_SLOTS][TOPK]; float wts_all[NPOS_SLOTS][TOPK];

            PROFILE_BOUNDARY(L,"moe-scratch-and-parameter-bind");
            /* phase 1: routing only. It needs x2b[t], which step (5) already
               produced for every position, so no expert weight is touched.
               Expert-major: each gate row is read once and applied to every
               position, instead of the whole gate being re-read per position. */
            float (*sc_all)[NEXP] = malloc(sizeof(float) * NPOS * NEXP);
            {
                const double _tr = now_s();
#pragma omp parallel for schedule(static)
                for (int e = 0; e < NEXP; e++) {
                    const float *grow = prepared_router((unsigned)e);
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
                if (route_tab) route_note(L, t, idsel_all[t]);
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
            PROFILE_BOUNDARY(L,"router-and-top16");

            float latent[NPOS_SLOTS][LAT], mixed[NPOS_SLOTS][LAT], routed[NPOS_SLOTS][E];
            float shared_gate[NPOS_SLOTS][SI], shared_up[NPOS_SLOTS][SI], shared_down[NPOS_SLOTS][E];
            float *latent_rows[NPOS_SLOTS], *mixed_rows[NPOS_SLOTS], *routed_rows[NPOS_SLOTS];
            float *gate_rows[NPOS_SLOTS], *up_rows[NPOS_SLOTS], *shared_rows[NPOS_SLOTS];
            const float *input_rows[NPOS_SLOTS];
            for (int position = 0; position < NPOS; position++) {
                latent_rows[position] = latent[position]; mixed_rows[position] = mixed[position];
                routed_rows[position] = routed[position]; input_rows[position] = x2b[position];
                gate_rows[position] = shared_gate[position]; up_rows[position] = shared_up[position];
                shared_rows[position] = shared_down[position];
            }
            Qm(latent_rows, input_rows, NPOS, WDN, E, LAT);
            PROFILE_BOUNDARY(L,"prefetch-submit-and-latent-projection");
            {
                /* Group by expert so one weight pass serves every position that picked it. */
                unsigned groups=0;
                for (int position=0;position<NPOS;position++) {
                    expert_hash[position]=resident_results.sets?expert_results_hash(latent[position]):0;
                    for (unsigned rank=0;rank<TOPK;rank++) {
                        unsigned expert=(unsigned)idsel_all[position][rank];
                        expert_hit[position][rank]=(unsigned)expert_results_find(&resident_results,(unsigned)L,
                            expert,expert_hash[position],latent[position],expert_down[position][rank]);
                        if (!expert_hit[position][rank])
                            expert_hit[position][rank]=(unsigned)stored_expert_down((unsigned)L,expert,
                                latent[position],expert_down[position][rank]);
                        if (expert_hit[position][rank]) continue;
                        unsigned group=0;
                        while (group<groups && group_expert[group]!=expert) group++;
                        if (group==groups) { group_expert[groups]=expert; group_members[groups]=0; groups++; }
                        group_position[group][group_members[group]]=(unsigned)position;
                        group_rank[group][group_members[group]]=rank;
                        group_members[group]++;
                    }
                }
                PROFILE_BOUNDARY(L,"result-lookup-and-first-expert-submit");
                for (unsigned group=0;group<groups;group++) resident_group_pairs+=group_members[group];
                resident_group_passes+=groups;
                /* Expert n's maths needs only expert n's bytes, so the reads run on
                   dedicated threads and each group waits for just its own slot. */
                root_pull_begin(resident_roots[L],group_expert,groups);
                resident_shared_rows(gate_rows,up_rows,shared_rows,input_rows,NPOS,S1,S3,S2);
                PROFILE_BOUNDARY(L,"shared-expert-during-read");
                for (unsigned group=0;group<groups;group++) {
                    const float *ins[ROOT_MAX_ROWS]; float *outs[ROOT_MAX_ROWS];
                    for (unsigned member=0;member<group_members[group];member++) {
                        ins[member]=latent[group_position[group][member]];
                        outs[member]=expert_down[group_position[group][member]][group_rank[group][member]];
                    }
                    root_arena_wait(group);
                    resident_expert_rows((unsigned)L,group_expert[group],ins,outs,group_members[group]);
                    for (unsigned member=0;member<group_members[group];member++) {
                        unsigned position=group_position[group][member];
                        expert_results_store(&resident_results,(unsigned)L,group_expert[group],
                            expert_hash[position],latent[position],expert_down[position][group_rank[group][member]]);
                    }
                }
                root_pull_end();
                for (int position=0;position<NPOS;position++) {
                    memset(mixed[position], 0, sizeof(mixed[position]));
                    for (int rank=0;rank<TOPK;rank++) {
                        const float *down=expert_down[position][rank];
                        if (expert_hit[position][rank]) {
                            resident_result_hits++;
#ifdef CLOVER_CHECK_EXPERT
                            check_live_expert((unsigned)L,(unsigned)idsel_all[position][rank],latent[position],down);
#endif
                        }
                        const float weight = wts_all[position][rank];
                        for (int coordinate = 0; coordinate < LAT; coordinate++)
                            mixed[position][coordinate] = mixed[position][coordinate] + weight * down[coordinate];
                    }
                    rmsnorm(mixed[position], mixed[position], lnw, LAT, EPS5);
                }
            }
            Qm(routed_rows, (const float *const *)mixed_rows, NPOS, WUP, LAT, E);
            PROFILE_BOUNDARY(L,"experts-mix-normalize-up");
#ifdef CLOVER_DEFER_SHARED
            resident_shared_rows(gate_rows,up_rows,shared_rows,input_rows,NPOS,S1,S3,S2);
#endif
            for (int position = 0; position < NPOS; position++)
                for (int coordinate = 0; coordinate < E; coordinate++)
                    ffn[position][coordinate] = routed[position][coordinate] + shared_down[position][coordinate];
            free(gw); free(gbias); free(lnw);
        }

    #ifdef CLOVER_DEFER_SHARED
        PROFILE_BOUNDARY(L,isMoE?"shared-expert-and-cleanup":"dense-mlp");
    #else
        PROFILE_BOUNDARY(L,isMoE?"mlp-merge-and-cleanup":"dense-mlp");
    #endif
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

        operator_close();
        prepared_close();
        /* St and convbuf alias the lane's own storage, so the carry needs no writeback. */
        for (int t = 0; t < NPOS; t++) SEQ(t,L)->length++;
        { const double _tp = now_s();
          (void)t0;
          pr_secs += now_s() - _tp; }
                PROFILE_BOUNDARY(L,"mlp-residual-cache-save-and-unbind");
                PROFILE_DURATION(L,"total:layer",now_s()-t0);
    }

    if (operator_layers != 93 || operator_matrices_used != 303 || operator_vectors_used != 348 ||
        prepared_trunk_count != 93 || resident_expert_calls + resident_result_hits * 3 != 92ULL * 16 * 3 * (unsigned)NPOS || recorded_expert_hits)
        die("live token operator coverage differs");
    if (resident_maps != maps_before || resident_unmaps != unmaps_before || resident_tap_layers != taps_before)
        die("fixed parameters changed lifetime during token evaluation");
    if (!project) { free(fa); free(fm); return 0; }

    /* ---------------------------------------------------------------- tail */
    const float *mn = resident_leaves->gains[2];
    if (pfx_of) { fclose(pfx_of); pfx_of = NULL; }   /* the next step reads it immediately */

    const float *foldO = resident_leaves->fold;

    float hf[E], nrm[E];
    /* Causality says a position depends only on tokens at or before it, so the
       same prefix must give the same state whatever follows. This dumps it. */
    { const char *p = getenv("K3_DUMPRES");
      if (p) {
          FILE *f = fopen(p, "wb");
          if (f) { for (int t = 0; t < NPOS; t++) fwrite(resid[t], 4, E, f); fclose(f); }
      } }
    float *logits = malloc(sizeof(float) * VOCAB);
    if (!logits) die("logits allocation failed");
    uint32_t selected;

    /* A batch carries one lane per slot, so every slot needs its own head. The
       last one falls through to the existing path and keeps the diagnostics. */
    if (batch_decode)
        for (int t = 0; t < NPOS - 1; t++) {
            for (int s = 0; s < nsnap; s++) srcbuf[s] = snap[t][s];
            srcbuf[nsnap] = resid[t];
            AR(hf, srcbuf, nsnap + 1, foldO, NULL);
            rmsnorm(nrm, hf, mn, E, EPS5);
            double slot_started = now_s();
            if (!head_cache_project(&resident_head, nrm, logits, &selected)) die("fruit projection failed");
            op_add(OP_B, slot_started, (int64_t)VOCAB * E * 2);
            op_fl[OP_B] += 2 * (int64_t)VOCAB * E;
            int best = 0;
            for (int i = 1; i < VOCAB; i++) if (logits[i] > logits[best]) best = i;
            batch_next[t] = (unsigned)best;
        }

    for (int s = 0; s < nsnap; s++) srcbuf[s] = snap[NPOS - 1][s];
    srcbuf[nsnap] = resid[NPOS - 1];
    /* The distributed layer 93 blends this one with a plain softmax, so it stays a
       blend here too or the two stop being the same model. */
    AR(hf, srcbuf, nsnap + 1, foldO, NULL);
    rmsnorm(nrm, hf, mn, E, EPS5);
    PROFILE_BOUNDARY(93,"final-aggregation-and-normalization");

    double head_started = now_s();
    if (!head_cache_project(&resident_head, nrm, logits, &selected)) die("fruit projection failed");
    op_add(OP_B, head_started, (int64_t)VOCAB * E * 2);
    op_fl[OP_B] += 2 * (int64_t)VOCAB * E;
    PROFILE_BOUNDARY(93,"head-fill-and-projection");

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



    int am = 0;
    for (int i = 1; i < VOCAB; i++) if (logits[i] > logits[am]) am = i;
    batch_next[NPOS - 1] = (unsigned)am;

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

    if (operator_layers != 93 || operator_matrices_used != 303 || operator_vectors_used != 348)
        die("all-layer operator field coverage differs");
    if (prepared_trunk_count != 93 || recorded_expert_hits || resident_expert_calls + resident_result_hits * 3 != 92ULL * 16 * 3 * (unsigned)NPOS)
        die("resident request expert or trunk coverage differs");
    if (resident_maps != maps_before || resident_unmaps != unmaps_before || resident_tap_layers != taps_before)
        die("fixed parameters changed lifetime during a request");
    free(fa); free(fm); free(logits);
    PROFILE_BOUNDARY(93,"argmax-and-cleanup");
    fprintf(stderr,"STEP_JSON {\"position\":%u,\"token\":%d,\"seconds\":%.9f,\"timed_seconds\":%.9f,"
        "\"fixed_mapping_opens\":%u,\"fixed_mapping_releases\":%u,\"tap_layout_builds\":%u,"
        "\"layers\":%u,\"matrix_records\":%u,\"vector_records\":%u,\"expert_matches\":%u,"
        "\"expert_projection_calls\":%lld,\"expert_result_hits\":%llu,\"vector_copy_seconds\":%.9f,"
        "\"expert_read_bytes\":%llu,\"expert_read_seconds\":%.9f,"
        "\"expert_cache_hits\":%llu,\"expert_cache_misses\":%llu,"
        "\"expert_launch_seconds\":%.9f,\"expert_tail_seconds\":%.9f,"
        "\"expert_stall_seconds\":%.9f,\"expert_drain_seconds\":%.9f,\"operators\":{",
        resident_position, am, now_s() - request_started, now_s() - T0,
        resident_maps - maps_before, resident_unmaps - unmaps_before, resident_tap_layers - taps_before,
        operator_layers, operator_matrices_used, operator_vectors_used, recorded_expert_hits,
        (long long)resident_expert_calls, resident_result_hits, sv_secs,
        root_pull_bytes, root_pull_seconds, root_cache_hits, root_cache_misses,
        root_launch_seconds, root_tail_seconds,
        root_wait_seconds, root_drain_seconds);
    for (int operation = 0; operation < OP_COUNT; operation++)
        fprintf(stderr,"%s\"%s\":%.9f", operation ? "," : "", OPN[operation], op_t[operation]);
    fprintf(stderr,"}}\n");
    fprintf(stderr,"OPSTAT_JSON {");
    for (int operation = 0; operation < OP_COUNT; operation++)
        fprintf(stderr,"%s\"%s\":{\"s\":%.6f,\"gflop\":%.4f,\"wgb\":%.4f,\"calls\":%lld}",
            operation ? "," : "", OPN[operation], op_t[operation],
            (double)op_fl[operation] / 1e9, (double)op_wb[operation] / 1e9,
            (long long)op_n[operation]);
    fprintf(stderr,"}\n");
    return (unsigned)am;
}

static unsigned evaluate_token(unsigned token,unsigned position,int project)
{
    return evaluate_tokens(&token,1,position,project);
}

/* Only the last prefill position projects, so earlier ones can wait and share one weight sweep. */
static unsigned prefill_tokens[NPOS_SLOTS], prefill_count, prefill_base;

static int resident_step(void *state,unsigned token,unsigned position,int project,unsigned *next)
{
    (void)state;
    if (!project) {
        if (prefill_count == NPOS_SLOTS || (prefill_count && prefill_base + prefill_count != position)) {
            evaluate_tokens(prefill_tokens,prefill_count,prefill_base,0);
            prefill_count = 0;
        }
        if (!prefill_count) prefill_base = position;
        prefill_tokens[prefill_count++] = token;
        return 1;
    }
    if (prefill_count && prefill_count < NPOS_SLOTS && prefill_base + prefill_count == position) {
        prefill_tokens[prefill_count++] = token;
        *next = evaluate_tokens(prefill_tokens,prefill_count,prefill_base,1);
        prefill_count = 0;
        return 1;
    }
    if (prefill_count) {
        evaluate_tokens(prefill_tokens,prefill_count,prefill_base,0);
        prefill_count = 0;
    }
    *next = evaluate_token(token,position,1);
    return 1;
}

/* ---------------------------------------------------------------- batching
   A lane is one in-flight request. Prefill runs a lane at a time because it
   already fills every slot, then the whole batch decodes together: one weight
   sweep carries a new position for every lane instead of one. */
typedef struct {
    GenerationRequest request;
    unsigned id, produced, next;
    int active, eos;
    double started;
} Lane;
static Lane batch_lanes[K3_LANES];

static void lane_sequence_clear(unsigned lane)
{
    lane_position[lane]=0;
    for (unsigned layer=0; layer<NLAY; layer++) {
        ResidentSequence *sequence=&resident_sequences[lane][layer];
        sequence->length=0;
        if (sequence->state) {
            memset(sequence->state,0,ST_BYTES);
            memset(sequence->convolution,0,CONV_BYTES);
        }
    }
}

static unsigned lane_prefill(unsigned lane, const GenerationRequest *request)
{
    unsigned next=0;
    active_lane=lane; batch_decode=0; prefill_count=0;
    for (unsigned position=0; position<request->count; position++)
        if (!resident_step(NULL,request->tokens[position],position,
                           position+1==request->count,&next)) die("prefill failed");
    lane_position[lane]=request->count;
    return next;
}

static void lane_finish(Lane *lane)
{
    lane->active=0;
    printf("DONE_JSON {\"request\":%u,\"input_tokens\":%u,\"output_tokens\":%u,\"stop_reason\":\"%s\",\"seconds\":%.9f}\n",
        lane->id,lane->request.count,lane->produced,lane->eos?"eos":"length",now_s()-lane->started);
}

/* stdio would swallow a whole pipe write into its own buffer and leave poll
   with nothing to report, so requests are split out of a buffer we own. */
static char request_buffer[1<<16];
static size_t request_filled, request_taken;
static int request_eof;
static unsigned request_count;

static int request_next(char *line, size_t size, int wait)
{
    for (;;) {
        for (size_t at=request_taken; at<request_filled; at++)
            if (request_buffer[at]=='\n') {
                size_t length=at-request_taken;
                if (length>=size) length=size-1;
                memcpy(line,request_buffer+request_taken,length);
                line[length]=0;
                request_taken=at+1;
                return 1;
            }
        if (request_taken) {
            memmove(request_buffer,request_buffer+request_taken,request_filled-request_taken);
            request_filled-=request_taken; request_taken=0;
        }
        if (request_filled==sizeof request_buffer) die("request line too long");
        if (!wait) {
            struct pollfd watch={ .fd=0, .events=POLLIN };
            if (poll(&watch,1,0)<=0) return 0;
        }
        ssize_t got=read(0,request_buffer+request_filled,sizeof request_buffer-request_filled);
        if (got<=0) { request_eof=1; return 0; }
        request_filled+=(size_t)got;
    }
}

/* Take one waiting request into a free lane and prefill it. Prefill needs every
   slot for its own tokens, so it runs between decode steps rather than inside one. */
static int lane_admit(unsigned lane, int wait)
{
    char line[8192];
    GenerationRequest parsed;
    for (;;) {
        if (!request_next(line,sizeof line,wait)) return 0;
        if (generation_request_parse((const unsigned char *)line,strlen(line),&resident_config,&parsed) &&
            parsed.count && parsed.count<=resident_config.max_input_tokens &&
            parsed.max_new_tokens && parsed.max_new_tokens<=resident_config.max_output_tokens) break;
        puts("REQUEST_ERROR {\"reason\":\"invalid input_ids or token budget\"}");
        fflush(stdout);
    }
    Lane *entry=&batch_lanes[lane];
    entry->request=parsed;
    entry->id=++request_count;
    entry->produced=0; entry->eos=0; entry->active=1;
    entry->started=now_s();
    lane_sequence_clear(lane);
    entry->next=lane_prefill(lane,&entry->request);
    return 1;
}

/* Emit the token a lane is holding and queue it for the next sweep, or retire
   the lane when that token was its last. */
static int lane_emit(unsigned lane, unsigned *tokens, unsigned *slots)
{
    Lane *entry=&batch_lanes[lane];
    printf("TOKEN_JSON {\"request\":%u,\"index\":%u,\"token\":%u}\n",
        entry->id,entry->produced,entry->next);
    entry->produced++;
    if (entry->next==resident_config.eos_token_id) entry->eos=1;
    if (entry->eos || entry->produced>=entry->request.max_new_tokens) { lane_finish(entry); return 0; }
    pos_lane[*slots]=lane;
    tokens[(*slots)++]=entry->next;
    return 1;
}

static void serve(void)
{
    unsigned tokens[NPOS_SLOTS];
    for (;;) {
        unsigned slots=0;
        for (unsigned lane=0; lane<K3_LANES; lane++)
            if (batch_lanes[lane].active) lane_emit(lane,tokens,&slots);
        /* a lane that just retired is capacity for this sweep, not the next one:
           the sweep costs the same whether it carries one position or eight */
        for (unsigned lane=0; lane<K3_LANES; lane++) {
            if (batch_lanes[lane].active || request_eof) continue;
            if (lane_admit(lane,0)) lane_emit(lane,tokens,&slots);
        }
        fflush(stdout);
        if (!slots) {
            if (request_eof) break;
            if (!lane_admit(0,1)) break;
            continue;
        }
        fprintf(stderr,"STEP_WIDTH %u\n",slots);
        batch_decode=1;
        evaluate_tokens(tokens,slots,0,1);
        batch_decode=0;
        for (unsigned slot=0; slot<slots; slot++) {
            unsigned lane=pos_lane[slot];
            batch_lanes[lane].next=batch_next[slot];
            lane_position[lane]++;
        }
    }
}

static int resident_read_config(void)
{
    const char *setting=getenv("CLOVER_CONFIG");
    char path[4096];
    if (!setting || !*setting) {
        ssize_t length=readlink("/proc/self/exe",path,sizeof path-1);
        if (length<=0 || (size_t)length>=sizeof path-1) return 0;
        path[length]=0;
        char *separator=strrchr(path,'/');
        if (!separator || (size_t)(separator-path)+sizeof("/configs/generation.json")>sizeof path) return 0;
        strcpy(separator,"/configs/generation.json");
        setting=path;
    }
    return generation_config_load(setting,&resident_config);
}

int main(int argc, char **argv)
{
    int inspect = argc > 1 && !strcmp(argv[1], "--inspect");
    if (argc > (inspect ? 3 : 2)) {
        fputs("usage: clover-one [--inspect] [DATASET_DIRECTORY]\n", stderr);
        return 2;
    }
    resident_configure(argc > 1 + inspect ? argv[1 + inspect] : NULL);
    if (!resident_read_config()) die("invalid generation config: input+output budget must be at most256");
    double started = now_s();
    result_options();
    const char *index = getenv("K3_INDEX");
    if (!index) die("K3_INDEX required");
    mallopt(M_MMAP_THRESHOLD, 256 * 1024 * 1024);
    mallopt(M_TRIM_THRESHOLD, 256 * 1024 * 1024);
    pf_on = 0;
    load_index(index);
    prepared_init();
    const char *setting = getenv("K3_RESIDENT_WARMUP");
    if (setting && strcmp(setting, "0") && strcmp(setting, "1")) die("K3_RESIDENT_WARMUP must be 0 or 1");
    int warmup = !inspect && setting && !strcmp(setting, "1");
    resident_startup(warmup);
    printf("READY_JSON {\"startup_seconds\":%.9f,\"mapping_and_tap_seconds\":%.9f,"
        "\"page_warmup_seconds\":%.9f,\"payload_bytes\":%zu,\"fixed_mappings\":%u,"
        "\"tap_layout_layers\":%u,\"pages_touched_at_startup\":%s,\"pid\":%ld,"
        "\"max_input_tokens\":%u,\"max_output_tokens\":%u,\"context_capacity\":%u,"
        "\"head_cache_bytes\":%zu,\"expert_pipeline_bytes\":%zu,"
        "\"result_cache_bytes\":%zu,\"result_cache_entries\":%zu}\n",
        now_s() - started, resident_setup_seconds, resident_warmup_seconds,
        resident_payload_bytes, resident_maps, resident_tap_layers, warmup ? "true" : "false", (long)getpid(),
        resident_config.max_input_tokens, resident_config.max_output_tokens,
        resident_config.max_input_tokens + resident_config.max_output_tokens,
        resident_head.allocated,
        resident_pipeline.buffers[0].capacity+resident_pipeline.buffers[1].capacity,
        resident_results.allocated,resident_results.sets*RESULT_WAYS);
    fflush(stdout);
    if (inspect) {
        resident_shutdown();
        puts("INSPECT_PASS: current prepared datasets loaded; 372 mappings released; zero requests executed");
        return 0;
    }
    ResidentCacheStats before = resident_cache_stats();
    serve();
    resident_cache_report(request_count, before);
    unsigned requests = request_count;
    int failed = 0;
    started = now_s();
    resident_shutdown();
    printf("STOP_JSON {\"requests\":%u,\"shutdown_seconds\":%.9f,\"fixed_mapping_releases\":%u}\n",
        requests, now_s() - started, resident_unmaps);
    fflush(stdout);
    return failed ? 1 : 0;
}
