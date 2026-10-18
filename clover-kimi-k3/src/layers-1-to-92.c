#define _GNU_SOURCE
#define _FILE_OFFSET_BITS 64
#define TRANSFORMER_NO_MAIN
/* One layer for a window of tokens at a time (layer 1 by default; -DLB_GENERIC -DLB_TEMPLATE="tpl-kda.c" builds the
   run-time-layer version for the non-MLA layers 2..92: the layer number and the number of input snapshots are
   chosen at run time, and the template is the deployed transformer-2.c with only those constants made variables).

   Layer 1 for a window of tokens at a time.

   Every heavy stage is batched over the window; the light elementwise stages reuse the
   pod's own functions. Per-token results are compared bit for bit with the pod
   (transformer_process, one token per call), and the first tokens are also compared stage
   by stage so a mismatch says where it starts.

   Batched here: all trunk projections (tiles of tokens, 4 per pass over each weight row),
   the router (8 tokens per pass over each expert's router row, same accumulation order),
   the recurrent state update (head by head, no fork per token), the experts (grouped by
   expert across the window, 4 tokens per pass), the shared expert and the latent
   projections.  Not changed: arithmetic order inside any single token.

     layer-batch DATASET_DIR RESID_F32 S0_F32 NTOKENS WINDOW[,WINDOW...]                  (layer 1)
     layer-batch DATASET_DIR RESID_F32 SNAPS_F32 NTOKENS WINDOW[,WINDOW...] LAYER K       (generic; SNAPS = K slots of NTOKENS x 7168)  */
#include <stdio.h>
#ifdef LB_GENERIC
static unsigned lb_layer_no, lb_k_in;
static const char *lb_name(int which)
{
    static char names[3][256];
    static const char *const formats[3] = { "operators/qkv-all/layer-%u/operator.bin", "trunk-%u", "root-%u" };
    snprintf(names[which], sizeof names[which], formats[which], lb_layer_no);
    return names[which];
}
#include LB_TEMPLATE
#define LB_LAYER lb_layer_no
#define LB_K lb_k_in
#define LB_MLA TRANSFORMER_MLA
#define LB_INPUT_GAINS(tr) ((tr)->input_gains)
#else
#include "transformer-1.c"
#define LB_LAYER 1u
#define LB_K 1u
#define LB_MLA 0
#define LB_INPUT_GAINS(tr) ((tr)->qkv + 88)
#endif
#define LB_BOUNDARY (LB_LAYER % 12 == 0)
#include <omp.h>
#include "trunk-4bit.h"
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>

enum { INW = 7168, ROWS = 12288, LATW = 3584, EXPW = 3072, SHRW = 6144, NHEAD = 96, HDIM = 128,
       CHUNK = 32, THREADS = 16, CHECK = 8 };

static double clk(void)
{
    struct timespec m; clock_gettime(CLOCK_MONOTONIC, &m);
    return (double)m.tv_sec + (double)m.tv_nsec / 1e9;
}
static void bail(const char *what) { fprintf(stderr, "layer-batch: %s\n", what); exit(1); }

/* ---- experts from disk (env LB_DISK): 0 = locked in RAM as before; 1 = before EVERY window the layer's expert file is
   dropped from memory, so the window reads what it needs from disk on first touch; 2 = the same, plus a read-ahead
   request for each used expert, in processing order, as the window starts its expert stage. ---- */
static int g_disk, g_expert_fd = -1;
static double g_evict_s;
static unsigned long g_distinct_sum, g_window_count, g_hits[896];
static void evict_experts(const Root *root)
{
    madvise((void *)root->direct, root->direct_bytes, MADV_DONTNEED);
    posix_fadvise(g_expert_fd, 0, 0, POSIX_FADV_DONTNEED);
}
static unsigned long long proc_read_bytes(void)
{
    FILE *file = fopen("/proc/self/io", "r");
    if (!file) return 0;
    char label[64]; unsigned long long value = 0, found = 0;
    while (fscanf(file, "%63s %llu", label, &value) == 2) if (!strcmp(label, "read_bytes:")) { found = value; break; }
    fclose(file);
    return found;
}

/* ---- mode 3: experts by direct reads (O_DIRECT, 4 MB chunks) into one in-memory copy of the layer's experts.
   Reader threads start when the window starts and read a speculative list first (experts 0..LB_SPEC-1); as soon as the
   router has chosen, they switch to the experts the window needs, in the order the compute takes them (most tokens
   first), and stop when those are all in memory. Compute threads wait for the expert they are about to use. ---- */
#include <pthread.h>
#include <sched.h>
enum { IO_CHUNK = 4 << 20, MAX_IO = 32 };
static unsigned char *g_buf;
static int g_direct_fd = -1, g_io_n = 8, g_spec = 896;
static int g_io_lo = -1, g_io_hi = -1; /* env LB_IOLO / LB_IOHI: pin the reader threads to these CPUs */
static int g_state[896];                 /* 0 not read, 1 being read, 2 in memory */
static unsigned g_spec_idx, g_need[896], g_need_n, g_need_idx;
static int g_need_ready, g_cancel;
static pthread_t g_io[MAX_IO];
static unsigned long g_read_experts, g_wasted_experts;
static double g_wait_t[64]; /* per compute thread: time spent waiting for its expert to arrive from disk */

/* Decode steps keep the experts a layer used lately in RAM slots of their own, so the next token reads only the ones it did not use
   before. `where[e]` is the entry holding expert e (-1: none); the entry's slot is a fixed 17.5 MB region handed in by the caller.
   Used by single-token calls only; a prompt pass uses g_buf by expert id as before and clears the cache. */
enum { CACHE_MAX = 64 };
typedef struct {
    unsigned cap;
    unsigned char *slots[CACHE_MAX];
    int expert[CACHE_MAX];
    unsigned long stamp[CACHE_MAX];
    short where[896];
} ExpertCache;
static ExpertCache *g_cache;                  /* non-NULL: this call is a decode step and experts live in the layer's slots */
static unsigned char *g_ptr[896];             /* slot mode: where each expert of the running layer is */
/* A window of several tokens (verifying a draft) needs more distinct experts than a layer has slots: the ones that find no free slot
   are read into this pool, shared by all layers (only one layer runs at a time) and not remembered afterwards. */
enum { SCRATCH_MAX = 160 };
static unsigned char *g_scratch[SCRATCH_MAX];
static unsigned g_scratch_n, g_scratch_used;
static unsigned long g_cache_clock, g_cache_hits, g_cache_misses;

/* Early reads (env LB_GUESS=G): at the start of a decode step the readers already start on the G most likely experts, guessed by routing
   on the layer input before attention, so the disk works during the attention part instead of waiting for the router. */
static int g_guess_max;
static unsigned g_spec_ids[64], g_guess_list[64], g_guess_count;
static unsigned g_spec_list_n;
static int g_spec_list_on;
static unsigned char g_guessed[896];
static unsigned long g_guess_issued, g_guess_used;

static ExpertCache *cache_make(unsigned char *const *slots, unsigned cap)
{
    if (cap < 16 || cap > CACHE_MAX) bail("expert cache needs 16 to 64 slots per layer");
    ExpertCache *c = calloc(1, sizeof *c);
    if (!c) bail("allocation failed");
    c->cap = cap;
    for (unsigned j = 0; j < cap; j++) { c->slots[j] = slots[j]; c->expert[j] = -1; }
    memset(c->where, 0xff, sizeof c->where);
    return c;
}
static void cache_clear(ExpertCache *c)
{
    for (unsigned j = 0; j < c->cap; j++) { c->expert[j] = -1; c->stamp[j] = 0; }
    memset(c->where, 0xff, sizeof c->where);
}
/* For the experts this step needs: cached ones are ready at once, the others get the least recently used slot and are read. */
static void cache_assign(const unsigned *order, unsigned n)
{
    ExpertCache *c = g_cache;
    if (n > 16 * 9) bail("a decode window needs more than 144 experts");
    if (n > c->cap && !g_scratch_n) bail("a decode window needs more experts than the cache holds and there is no scratch pool");
    const unsigned long now = ++g_cache_clock;
    unsigned miss[144], nm = 0;
    g_scratch_used = 0;
    for (unsigned g = 0; g < n; g++) {
        const unsigned e = order[g]; const short w = c->where[e];
        if (w >= 0) {
            c->stamp[w] = now; g_ptr[e] = c->slots[w];
            if (g_guessed[e]) g_guess_used++;                       /* its read was started early; the state already says where it is */
            else __atomic_store_n(&g_state[e], 2, __ATOMIC_RELEASE);
            g_cache_hits++;
        }
        else miss[nm++] = e;
    }
    for (unsigned m = 0; m < nm; m++) {
        unsigned v = CACHE_MAX; unsigned long best = ~0UL;
        for (unsigned j = 0; j < c->cap; j++) {
            /* an expert read early and not needed keeps its slot until the layer is done: its read may still be running */
            if (c->stamp[j] == now || (c->expert[j] >= 0 && g_guessed[c->expert[j]])) continue;
            if (c->stamp[j] < best) { best = c->stamp[j]; v = j; }
        }
        if (v == CACHE_MAX) {
            if (g_scratch_used >= g_scratch_n) bail("no free expert slot: LB_GUESS plus 16 must fit in the cache, or the scratch pool is too small");
            g_ptr[miss[m]] = g_scratch[g_scratch_used++];
            g_cache_misses++;
            continue;
        }
        if (c->expert[v] >= 0) c->where[c->expert[v]] = -1;
        c->expert[v] = (int)miss[m]; c->where[miss[m]] = (short)v; c->stamp[v] = now; g_ptr[miss[m]] = c->slots[v];
        g_cache_misses++;
    }
}
/* After the layer's reads are done: an expert read early that did not finish (the readers moved on to the real list) holds no valid data. */
static void cache_finish(void)
{
    ExpertCache *c = g_cache;
    for (unsigned k = 0; k < g_guess_count; k++) {
        const unsigned e = g_guess_list[k];
        if (g_state[e] != 2 && c->where[e] >= 0) { c->expert[c->where[e]] = -1; c->stamp[c->where[e]] = 0; c->where[e] = -1; }
        g_guessed[e] = 0;
    }
    g_guess_count = 0;
}

static inline const unsigned char *expert_base(const Root *root, unsigned expert)
{
    if (g_disk == 3 && g_cache) return g_ptr[expert];
    return (g_disk == 3 ? g_buf : root->direct) + (size_t)expert * ROOT_EXPERT_RAW;
}
static void read_expert(unsigned e)
{
    unsigned char *dst = g_cache ? g_ptr[e] : g_buf + (size_t)e * ROOT_EXPERT_RAW;
    for (size_t off = 0; off < ROOT_EXPERT_RAW; off += IO_CHUNK) {
        const size_t len = ROOT_EXPERT_RAW - off < IO_CHUNK ? ROOT_EXPERT_RAW - off : IO_CHUNK;
        size_t got = 0;
        while (got < len) {
            const ssize_t n = pread(g_direct_fd, dst + off + got, len - got, (off_t)((size_t)e * ROOT_EXPERT_RAW + off + got));
            if (n <= 0) bail("direct read failed");
            got += (size_t)n;
        }
    }
}
static void *io_main(void *unused)
{
    (void)unused;
    while (!__atomic_load_n(&g_cancel, __ATOMIC_ACQUIRE)) {
        unsigned e;
        if (__atomic_load_n(&g_need_ready, __ATOMIC_ACQUIRE)) {
            const unsigned i = __atomic_fetch_add(&g_need_idx, 1, __ATOMIC_ACQ_REL);
            if (i >= g_need_n) break;
            e = g_need[i];
        } else {
            const unsigned i = __atomic_fetch_add(&g_spec_idx, 1, __ATOMIC_ACQ_REL);
            const unsigned limit = g_spec_list_on ? __atomic_load_n(&g_spec_list_n, __ATOMIC_ACQUIRE) : (unsigned)g_spec;
            if (i >= limit) { usleep(100); __atomic_fetch_sub(&g_spec_idx, 1, __ATOMIC_ACQ_REL); continue; }
            e = g_spec_list_on ? g_spec_ids[i] : i;
        }
        int expect = 0;
        if (!__atomic_compare_exchange_n(&g_state[e], &expect, 1, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) continue;
        read_expert(e);
        __atomic_store_n(&g_state[e], 2, __ATOMIC_RELEASE);
    }
    return NULL;
}
static void prefetch_start(void)
{
    memset(g_state, 0, sizeof g_state);
    g_spec_idx = 0; g_need_n = 0; g_need_idx = 0; g_need_ready = 0; g_cancel = 0;
    g_spec_list_n = 0; g_spec_list_on = g_cache != NULL && g_guess_max > 0;
    cpu_set_t all; CPU_ZERO(&all);
    for (int c = 0; c < (int)sysconf(_SC_NPROCESSORS_ONLN) && c < CPU_SETSIZE; c++)
        if (g_io_lo < 0 || (c >= g_io_lo && c <= g_io_hi)) CPU_SET(c, &all);
    pthread_attr_t attr; pthread_attr_init(&attr); pthread_attr_setaffinity_np(&attr, sizeof all, &all);
    for (int i = 0; i < g_io_n; i++) if (pthread_create(&g_io[i], &attr, io_main, NULL)) bail("cannot start a reader thread");
    pthread_attr_destroy(&attr);
}
static void prefetch_publish(const unsigned *order, unsigned n)
{
    memcpy(g_need, order, n * sizeof *order);
    g_need_n = n;
    __atomic_store_n(&g_need_ready, 1, __ATOMIC_RELEASE);
}
static void prefetch_finish(const unsigned *total)
{
    __atomic_store_n(&g_cancel, 1, __ATOMIC_RELEASE);
    for (int i = 0; i < g_io_n; i++) pthread_join(g_io[i], NULL);
    for (unsigned e = 0; e < 896; e++) if (g_state[e] == 2) { g_read_experts++; if (!total[e]) g_wasted_experts++; }
}
static inline void wait_expert(unsigned e)
{
    while (__atomic_load_n(&g_state[e], __ATOMIC_ACQUIRE) != 2) { _mm_pause(); sched_yield(); }
}
static void *grab(size_t bytes)
{
    void *p = aligned_alloc(64, (bytes + 63) / 64 * 64);
    if (!p) bail("allocation failed");
    memset(p, 0, bytes);
    return p;
}

/* ---- stage clock ---- */
enum { LAPS = 24 };
static double lap_time[LAPS], lap_last;
static const char *lap_name[LAPS];
static void lap(int id, const char *name)
{
    double now = clk();
    lap_time[id] += now - lap_last; lap_name[id] = name; lap_last = now;
}

/* ---- batched projection: same arithmetic as transformer_project_row, per token ---- */
typedef struct { const unsigned char *ids, *scale; size_t id_stride, scale_stride; unsigned rows, cols; const char *name; const unsigned char *packed_nibbles, *packed_steps; } Mat;

static const PackedTrunk *g_active_packed_trunk;   /* set while a layer that has a 4-bit copy runs (USE_4BIT_TRUNK=1) */

/* ---- LB_REPEAT=1 (diagnostic, standalone harness, first window only): how often the same float value occurs, in the vectors every
   stage produces, in the weights, and in the operand pairs (activation, weight) of the multiplications. Exact bit patterns. ---- */
static int g_rep, g_rep_active, g_rep_done;
static int cmp_u32(const void *a, const void *b) { const unsigned x = *(const unsigned *)a, y = *(const unsigned *)b; return (x > y) - (x < y); }
static void rep_buf(const char *name, const float *p, size_t vectors, size_t width)
{
    if (!g_rep_active || !vectors) return;
    unsigned *v = malloc(width * sizeof *v), *all = malloc(vectors * width * sizeof *all);
    double distinct = 0, repeating = 0;
    for (size_t t = 0; t < vectors; t++) {
        memcpy(v, p + t * width, width * sizeof *v);
        qsort(v, width, sizeof *v, cmp_u32);
        size_t run = 1;
        for (size_t i = 1; i <= width; i++) {
            if (i < width && v[i] == v[i - 1]) { run++; continue; }
            distinct++; if (run > 1) repeating += (double)run;
            run = 1;
        }
        memcpy(all + t * width, v, width * sizeof *v);
    }
    qsort(all, vectors * width, sizeof *all, cmp_u32);
    size_t whole = 1;
    for (size_t i = 1; i < vectors * width; i++) whole += all[i] != all[i - 1];
    printf("REP %-27s %4zu x %-5zu distinct per vector %7.1f (%6.2f%%), elements sharing a value inside their vector %6.2f%%, distinct in the whole buffer %zu of %zu (%.2f%%)\n",
        name, vectors, width, distinct / (double)vectors, 100 * distinct / (double)(vectors * width), 100 * repeating / (double)(vectors * width),
        whole, vectors * width, 100 * (double)whole / (double)(vectors * width));
    free(v); free(all);
}
static void rep_buf_d(const char *name, const double *p, size_t vectors, size_t width)   /* the doubles here are widened floats, so the cast is exact */
{
    if (!g_rep_active) return;
    float *f = malloc(vectors * width * sizeof *f);
    for (size_t i = 0; i < vectors * width; i++) f[i] = (float)p[i];
    rep_buf(name, f, vectors, width);
    free(f);
}
/* Whole-vector and 16-column-segment reuse: how many of the `vectors` rows are bit-identical to an earlier row, and how many 16-column
   segments (same column block) are identical to the same block of an earlier row. */
static int cmp_u64(const void *a, const void *b) { const unsigned long long x = *(const unsigned long long *)a, y = *(const unsigned long long *)b; return (x > y) - (x < y); }
static unsigned long long fnv64(const void *p, size_t n)
{
    const unsigned char *b = p; unsigned long long h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}
static size_t count_dups(unsigned long long *h, size_t n)
{
    qsort(h, n, sizeof *h, cmp_u64);
    size_t distinct = n ? 1 : 0;
    for (size_t i = 1; i < n; i++) distinct += h[i] != h[i - 1];
    return n - distinct;
}
static void rep_dups(const char *name, const float *p, size_t vectors, size_t width)
{
    if (!g_rep_active || vectors < 2) return;
    unsigned long long *h = malloc(vectors * sizeof *h);
    for (size_t t = 0; t < vectors; t++) h[t] = fnv64(p + t * width, width * 4);
    const size_t dup_vectors = count_dups(h, vectors);
    size_t dup_segments = 0;
    for (size_t blk = 0; blk < width / 16; blk++) {
        for (size_t t = 0; t < vectors; t++) h[t] = fnv64(p + t * width + blk * 16, 64);
        dup_segments += count_dups(h, vectors);
    }
    printf("REP duplicates %-24s %zu vectors of %zu floats: %zu identical to an earlier vector; 16-column segments: %zu of %zu identical to an earlier vector's segment at the same columns (%.4f%%)\n",
        name, vectors, width, dup_vectors, dup_segments, vectors * (width / 16), 100.0 * (double)dup_segments / (double)(vectors * (width / 16)));
    free(h);
}
enum { HS_BITS = 24 };
static unsigned long long *g_hs;
static unsigned long g_hs_distinct;
static void hs_reset(void)
{
    if (!g_hs) g_hs = malloc(8ULL << HS_BITS);
    if (!g_hs) bail("pair table allocation failed");
    memset(g_hs, 0, 8ULL << HS_BITS);
    g_hs_distinct = 0;
}
static inline int hs_add(unsigned long long key)
{
    key += 1;
    size_t i = (size_t)((key * 0x9E3779B97F4A7C15ULL) >> (64 - HS_BITS));
    for (;; i = (i + 1) & ((1u << HS_BITS) - 1)) {
        if (!g_hs[i]) {
            g_hs[i] = key;
            if (++g_hs_distinct > (3ul << HS_BITS) / 4) bail("pair table full");
            return 0;
        }
        if (g_hs[i] == key) return 1;
    }
}
static inline int hs_member(unsigned long long key)
{
    key += 1;
    size_t i = (size_t)((key * 0x9E3779B97F4A7C15ULL) >> (64 - HS_BITS));
    for (;; i = (i + 1) & ((1u << HS_BITS) - 1)) {
        if (!g_hs[i]) return 0;
        if (g_hs[i] == key) return 1;
    }
}
/* A list keyed by the activation's own float value (what a "token value" list would hold): feed the tokens in order, count how many
   elements of each token have a value already in the list from earlier tokens (any column), then add the token's values. */
static void rep_growth(const char *name, const float *p, size_t vectors, size_t width)
{
    if (!g_rep_active || vectors < 2) return;
    if (vectors * width > (3ul << HS_BITS) / 4) return;
    hs_reset();
    unsigned long hits_all = 0, seen_all = 0; size_t mark = 1;
    printf("REP growth %-24s", name);
    for (size_t t = 0; t < vectors; t++) {
        unsigned long hits = 0;
        unsigned *bits = malloc(width * sizeof *bits);
        memcpy(bits, p + t * width, width * sizeof *bits);
        for (size_t c = 0; c < width; c++) hits += (unsigned long)hs_member(bits[c]);
        for (size_t c = 0; c < width; c++) hs_add(bits[c]);
        free(bits);
        if (t > 0) { hits_all += hits; seen_all += width; }
        if (t + 1 == mark || t + 1 == vectors) {
            printf(" | after %zu tokens: list %lu values, %.3f%% of the latest token's values were already listed", t + 1, g_hs_distinct, 100.0 * (double)hits / (double)width);
            mark *= 10;
        }
    }
    printf(" | over tokens 2..%zu: %.3f%% of all values already in the list\n", vectors, 100.0 * (double)hits_all / (double)seen_all);
}
static inline unsigned long long pair_key(float x, unsigned weight_bits)
{
    unsigned xb; memcpy(&xb, &x, 4);
    return ((unsigned long long)xb << 32) | weight_bits;
}
static void rep_pairs_print(const char *name, unsigned long mults)
{
    printf("REP pairs %-22s %lu multiplications, %lu distinct (activation, weight) pairs = %.3f%%, so %.2f%% of the products repeat an earlier one\n",
        name, mults, g_hs_distinct, 100.0 * (double)g_hs_distinct / (double)mults, 100.0 - 100.0 * (double)g_hs_distinct / (double)mults);
}
/* Trunk matrix against the first token's vector: the weight in each product is the 8-bit code (the row scale is applied after the sum). */
static void rep_trunk_pairs(const Mat *m, const float *x0)
{
    unsigned long hist[256] = {0};
    hs_reset();
    for (unsigned r = 0; r < m->rows; r++) {
        const unsigned char *ids = m->ids + (size_t)r * m->id_stride;
        for (unsigned c = 0; c < m->cols; c++) { hist[ids[c]]++; hs_add(pair_key(x0[c], ids[c])); }
    }
    unsigned used = 0; for (int i = 0; i < 256; i++) used += hist[i] != 0;
    unsigned long top = 0, all = (unsigned long)m->rows * m->cols;
    for (int pick = 0; pick < 16; pick++) { int b = 0; for (int i = 1; i < 256; i++) if (hist[i] > hist[b]) b = i; top += hist[b]; hist[b] = 0; }
    char label[64]; snprintf(label, sizeof label, "trunk %s %ux%u", m->name, m->rows, m->cols);
    printf("REP weights %-34s %u distinct 8-bit codes; the 16 commonest hold %.1f%% of the weights\n", label, used, 100.0 * (double)top / (double)all);
    rep_pairs_print(label, all);
}
typedef struct { const char *name; unsigned rows, cols; double sec; } MatStat;
static MatStat mat_stats[16];
static unsigned mat_stat_n;

static Mat mat_record(const Transformer *t, unsigned id, const char *name)
{
    const TransformerRecord *r = &t->records[id];
    Mat m = (Mat){ r->data + (size_t)r->rows * 4, r->data, r->columns, 4, r->rows, r->columns, name };
    if (g_active_packed_trunk) { m.packed_nibbles = g_active_packed_trunk->nibbles[id]; m.packed_steps = g_active_packed_trunk->steps[id]; }
    return m;
}
#if LB_MLA
/* The MLA operator records (ids 20-25) hold the same layout as a trunk record: row scales, then the 8-bit codes. */
static Mat mat_oprec(const Transformer *t, unsigned id, const char *name)
{
    const TransformerRecord *r = &t->operator_records[id];
    Mat m = (Mat){ r->data + (size_t)r->rows * 4, r->data, r->columns, 4, r->rows, r->columns, name };
    if (g_active_packed_trunk) { m.packed_nibbles = g_active_packed_trunk->nibbles[70 + id]; m.packed_steps = g_active_packed_trunk->steps[70 + id]; }
    return m;
}
#endif
static Mat mat_qkv(const Transformer *t, unsigned component)
{
    static const char *const names[3] = { "q", "k", "v" };
    const unsigned char *base = t->qkv + TRANSFORMER_QKV_PREFIX + (size_t)component * TRANSFORMER_QKV_COMPONENT;
    Mat m = (Mat){ base + 20, base, 3 * (size_t)TRANSFORMER_QKV_COMPONENT, 3 * (size_t)TRANSFORMER_QKV_COMPONENT,
                  TRANSFORMER_ROWS, TRANSFORMER_WIDTH, names[component] };
    if (g_active_packed_trunk) { m.packed_nibbles = g_active_packed_trunk->nibbles[40 + component]; m.packed_steps = g_active_packed_trunk->steps[40 + component]; }
    return m;
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
/* 4 weight rows x 4 tokens per pass (same kernel as input-to-layer0/layer0-batch.c): each activation load feeds 4 rows,
   each widened weight feeds 4 tokens. One 16-lane accumulator per (row, token); lane l holds the same terms in the
   same order as lane l of the two 8-lane accumulators above, and the reduction after the loop is identical. */
static inline __attribute__((always_inline)) void dotz4x4(const float *const *x, const unsigned char *const *ids,
    unsigned width, const float *scale, float *out)
{
    __m512 a[4][4];
    _Pragma("GCC unroll 4") for (int r = 0; r < 4; r++) _Pragma("GCC unroll 4") for (int g = 0; g < 4; g++) a[r][g] = _mm512_setzero_ps();
    for (unsigned c = 0; c < width; c += 16) {
        __m512 w[4];
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
static int g_wide = 1; /* 1: use the 4 rows x 4 tokens kernel where it applies (env LB_WIDE) */
static int g_vecrec = 1; /* 1: recurrent update vectorised across the 128 value columns (env LB_VEC) */

#if defined(__AVX512F__)
/* One head, one token: 16 value columns per instruction. Columns do not depend on each other; per column the
   operations and their order are those of the scalar loops (separate multiply and add), so the bits are the same. */
static inline void recurrent_vec(float (*state)[HDIM], const float *query, const float *key, const float *value,
    const float *decay, float beta, float query_scale, float *output)
{
    __m512 pred[8], diff[8], out[8];
    _Pragma("GCC unroll 8") for (int v = 0; v < 8; v++) pred[v] = _mm512_setzero_ps();
    for (unsigned row = 0; row < HDIM; row++) {
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
    for (unsigned row = 0; row < HDIM; row++) {
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

/* The 4-bit copy of every trunk matrix of a layer that the projections read. Slots: 0-39 record ids, 40-42 the q/k/v components, 70-75 MLA operator records. */
static PackedTrunk *pack_layer_trunk_to_4bit(const Transformer *layer)
{
    PackedTrunk *packed = calloc(1, sizeof *packed);
    if (!packed) bail("allocation failed");
#if LB_MLA
    static const unsigned record_ids[] = { 6, 7, 31, 32, 34, 35, 36 };
    static const unsigned operator_ids[] = { 20, 22, 23, 25 };
    for (unsigned i = 0; i < sizeof operator_ids / sizeof *operator_ids; i++) {
        const Mat m = mat_oprec(layer, operator_ids[i], "");
        pack_matrix_to_4bit(m.ids, m.id_stride, m.rows, m.cols, &packed->nibbles[70 + operator_ids[i]], &packed->steps[70 + operator_ids[i]]);
    }
#else
    static const unsigned record_ids[] = { 6, 7, 11, 12, 13, 31, 32, 34, 35, 36 };
    for (unsigned component = 0; component < 3; component++) {
        const Mat m = mat_qkv(layer, component);
        pack_matrix_to_4bit(m.ids, m.id_stride, m.rows, m.cols, &packed->nibbles[40 + component], &packed->steps[40 + component]);
    }
#endif
    for (unsigned i = 0; i < sizeof record_ids / sizeof *record_ids; i++) {
        const Mat m = mat_record(layer, record_ids[i], "");
        pack_matrix_to_4bit(m.ids, m.id_stride, m.rows, m.cols, &packed->nibbles[record_ids[i]], &packed->steps[record_ids[i]]);
    }
    return packed;
}

static int use_4bit_trunk_requested(void)
{
    const char *value = getenv("USE_4BIT_TRUNK");
    return value && atoi(value) != 0;
}

/* ---- LB_A8T=1 (experiment, NOT bit-identical): trunk matrices with 8-bit activations. A trunk weight is an int8 code times a per-row float
   scale, so with the token's activations rounded to int8 (one scale per group of LB_A8T_GS columns, default 256) the row's dot product is
   integer arithmetic: the unsigned-by-signed byte multiply-add (64 products per instruction) on activation + 128, the surplus 128 x (sum
   of the row's codes in the group) subtracted, then scaled and added in float. Windows of 8 or more tokens only (decode steps stay exact). ---- */
static int g_a8t, g_a8t_gs = 256, g_a8t_init;
#if defined(__AVX512VNNI__)
#define A8T_BLOCK(G, TOK0) \
    for (unsigned gi = 0; gi < ng; gi++) { \
        __m512i iacc[4][G], sw[4]; \
        for (int r = 0; r < 4; r++) { sw[r] = _mm512_setzero_si512(); for (int g = 0; g < G; g++) iacc[r][g] = _mm512_setzero_si512(); } \
        for (unsigned c = gi * gs; c < (gi + 1) * gs; c += 64) { \
            __m512i w[4], au[G]; \
            for (int r = 0; r < 4; r++) w[r] = _mm512_loadu_si512((const void *)(rid[r] + c)); \
            for (int g = 0; g < G; g++) au[g] = _mm512_loadu_si512((const void *)(aq + (size_t)(TOK0 + g) * cols + c)); \
            for (int r = 0; r < 4; r++) sw[r] = _mm512_dpbusd_epi32(sw[r], ones, w[r]); \
            for (int r = 0; r < 4; r++) for (int g = 0; g < G; g++) iacc[r][g] = _mm512_dpbusd_epi32(iacc[r][g], au[g], w[r]); \
        } \
        for (int r = 0; r < 4; r++) { \
            const __m512i corr = _mm512_slli_epi32(sw[r], 7); \
            for (int g = 0; g < G; g++) \
                fl[r][g] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(iacc[r][g], corr)), _mm512_set1_ps(asc[(size_t)(TOK0 + g) * ng + gi]), fl[r][g]); \
        } \
    }
static void bproject_a8(const Mat *m, const float *X, size_t xs, float *Y, size_t ys, unsigned n)
{
    const unsigned cols = m->cols, gs = cols < (unsigned)g_a8t_gs ? cols : (unsigned)g_a8t_gs, ng = cols / gs;
    uint8_t *aq = aligned_alloc(64, (size_t)n * cols); float *asc = malloc((size_t)n * ng * sizeof(float));
#pragma omp parallel for schedule(static)
    for (unsigned t = 0; t < n; t++)
        for (unsigned gi = 0; gi < ng; gi++) {
            const float *x = X + (size_t)t * xs + (size_t)gi * gs; float amax = 0;
            for (unsigned j = 0; j < gs; j++) { const float a = fabsf(x[j]); if (a > amax) amax = a; }
            const float inv = amax > 0 ? 127.0f / amax : 0.0f;
            asc[(size_t)t * ng + gi] = amax / 127.0f;
            for (unsigned j = 0; j < gs; j++) aq[(size_t)t * cols + (size_t)gi * gs + j] = (uint8_t)(lrintf(x[j] * inv) + 128);
        }
    const __m512i ones = _mm512_set1_epi8(1);
    for (unsigned c0 = 0; c0 < n; c0 += CHUNK) {
        const unsigned cn = n - c0 < CHUNK ? n - c0 : CHUNK;
#pragma omp parallel for schedule(static)
        for (unsigned row = 0; row < m->rows; row += 4) {
            const unsigned char *rid[4]; float sc[4];
            for (int r = 0; r < 4; r++) { rid[r] = m->ids + (size_t)(row + r) * m->id_stride; sc[r] = transformer_f32(m->scale + (size_t)(row + r) * m->scale_stride); }
            unsigned k = 0;
            for (; k + 4 <= cn; k += 4) {
                __m512 fl[4][4];
                for (int r = 0; r < 4; r++) for (int g = 0; g < 4; g++) fl[r][g] = _mm512_setzero_ps();
                A8T_BLOCK(4, c0 + k)
                for (int r = 0; r < 4; r++) for (int g = 0; g < 4; g++) Y[(size_t)(c0 + k + g) * ys + row + r] = _mm512_reduce_add_ps(fl[r][g]) * sc[r];
            }
            for (; k < cn; k++) {
                __m512 fl[4][1];
                for (int r = 0; r < 4; r++) fl[r][0] = _mm512_setzero_ps();
                A8T_BLOCK(1, c0 + k)
                for (int r = 0; r < 4; r++) Y[(size_t)(c0 + k) * ys + row + r] = _mm512_reduce_add_ps(fl[r][0]) * sc[r];
            }
        }
    }
    free(aq); free(asc);
}
#endif

/* Y[t][row] for t in [0, n): token stride xs / ys floats. Tokens go through in tiles of CHUNK so the
   tile's inputs stay in cache while the weights stream past once per tile. */
static void bproject(const Mat *m, const float *X, size_t xs, float *Y, size_t ys, unsigned n)
{
    if (g_rep_active) { rep_trunk_pairs(m, X); }
    if (g_active_packed_trunk && m->packed_nibbles) {
        project_with_4bit_weights(m->packed_nibbles, m->packed_steps, m->scale, m->scale_stride, m->rows, m->cols, X, xs, Y, ys, n);
        return;
    }
    if (!g_a8t_init) { g_a8t_init = 1; if (getenv("LB_A8T")) g_a8t = atoi(getenv("LB_A8T")); if (getenv("LB_A8T_GS") && *getenv("LB_A8T_GS")) g_a8t_gs = atoi(getenv("LB_A8T_GS")); }
#if defined(__AVX512VNNI__)
    if (g_a8t && g_ident && n >= 8 && m->rows % 4 == 0 && m->cols % 64 == 0 && (m->cols < (unsigned)g_a8t_gs ? m->cols : (unsigned)g_a8t_gs) % 64 == 0 &&
        m->cols % (m->cols < (unsigned)g_a8t_gs ? m->cols : (unsigned)g_a8t_gs) == 0) {
        const double began8 = clk();
        bproject_a8(m, X, xs, Y, ys, n);
        MatStat *stat8 = NULL;
        for (unsigned i = 0; i < mat_stat_n; i++) if (mat_stats[i].name == m->name) stat8 = &mat_stats[i];
        if (!stat8) { stat8 = &mat_stats[mat_stat_n++]; stat8->name = m->name; stat8->rows = m->rows; stat8->cols = m->cols; stat8->sec = 0; }
        stat8->sec += clk() - began8;
        return;
    }
#endif
    const double began = clk();
    for (unsigned c0 = 0; c0 < n; c0 += CHUNK) {
        const unsigned cn = n - c0 < CHUNK ? n - c0 : CHUNK;
#if defined(__AVX512F__)
        if (g_wide && g_ident && m->rows % 4 == 0) {
#pragma omp parallel for schedule(static)
            for (unsigned row = 0; row < m->rows; row += 4) {
                const unsigned char *rid[4]; float sc[4];
                for (int r = 0; r < 4; r++) {
                    rid[r] = m->ids + (size_t)(row + r) * m->id_stride;
                    sc[r] = transformer_f32(m->scale + (size_t)(row + r) * m->scale_stride);
                }
                unsigned k = 0;
                for (; k + 4 <= cn; k += 4) {
                    const float *x[4]; float out[16];
                    for (int j = 0; j < 4; j++) x[j] = X + (size_t)(c0 + k + j) * xs;
                    dotz4x4(x, rid, m->cols, sc, out);
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
            const float scale = transformer_f32(m->scale + (size_t)row * m->scale_stride);
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
    if (!stat) { stat = &mat_stats[mat_stat_n++]; stat->name = m->name; stat->rows = m->rows; stat->cols = m->cols; stat->sec = 0; }
    stat->sec += clk() - began;
}

/* ---- copies of the pod's small stages, over plain arrays ---- */
/* The fold over `count` sources (the snapshots, then the residual last); same operations, same order, as the pod's
   transformer_aggregate. */
static void agg(const Transformer *tr, unsigned fold, const float *const *sources, unsigned count, float *aggregate)
{
    float scores[10], exponentials[10], weights[10];
    for (unsigned source = 0; source < count; source++) {
        float inverse = transformer_inverse_rms(sources[source], INW);
        double score = 0.0;
        for (unsigned c = 0; c < INW; c++) {
            float normalized = sources[source][c] * inverse;
            score += (double)normalized * (double)transformer_f32(tr->records[fold].data + c * 4);
        }
        scores[source] = (float)score;
    }
    float maximum = scores[0];
    for (unsigned source = 1; source < count; source++) if (scores[source] > maximum) maximum = scores[source];
    double total = 0.0;
    for (unsigned source = 0; source < count; source++) {
        exponentials[source] = expf(scores[source] - maximum);
        total += (double)exponentials[source];
    }
    for (unsigned source = 0; source < count; source++) weights[source] = (float)((double)exponentials[source] / total);
    for (unsigned c = 0; c < INW; c++) {
        float value = 0.0f;
        for (unsigned source = 0; source < count; source++) value = value + weights[source] * sources[source][c];
        aggregate[c] = value;
    }
}

static int route_select(const float *scores, const float *choices, unsigned *selected, float *weights)
{
    float best[16]; unsigned count = 0;
    for (unsigned expert = 0; expert < 896; expert++) {
        float choice = choices[expert];
        if (!isfinite(choice)) return 0;
        if (count == 16 && !(choice > best[15])) continue;
        unsigned position = count < 16 ? count : 15;
        while (position && choice > best[position - 1]) {
            best[position] = best[position - 1];
            selected[position] = selected[position - 1];
            position--;
        }
        best[position] = choice;
        selected[position] = expert;
        if (count < 16) count++;
    }
    double total = 0.0;
    for (unsigned r = 0; r < 16; r++) total += (double)scores[selected[r]];
    float inverse = (float)(1.0 / (total + 1e-20));
    for (unsigned r = 0; r < 16; r++) weights[r] = scores[selected[r]] * inverse;
    return 1;
}

/* Router: 8 tokens per pass over each expert's row, one double accumulator per token in coordinate
   order, mul then add, as the pod does it. */
static int g_wideroute = 1; /* 1: 32 tokens per pass instead of 8, 512-bit registers (env LB_ROUTE) */
#if defined(__AVX512F__)
/* Every (token, expert) score is one chain of 7168 dependent double additions, so 8 tokens per pass (two accumulators)
   leave the adder waiting on its own result. 32 tokens per pass give 4 independent accumulators per expert row. Each
   token's chain is untouched: the same products added in the same order. The activations are kept as floats (they are
   floats, and the widening to double is exact) so the 32-token block is 0.9 MB instead of 1.8 MB. */
enum { RT = 32 };
static void route_scores_wide(const Transformer *tr, const float *postnorm, unsigned n, float *scores, float *choices)
{
    const unsigned char *weights = tr->records[29].data;
    float *xt = grab((size_t)INW * RT * sizeof(float));
    for (unsigned g0 = 0; g0 < n; g0 += RT) {
        const unsigned cnt = n - g0 < RT ? n - g0 : RT;
#pragma omp parallel for schedule(static)
        for (unsigned c = 0; c < INW; c++)
            for (unsigned k = 0; k < RT; k++) xt[(size_t)c * RT + k] = k < cnt ? postnorm[(size_t)(g0 + k) * INW + c] : 0.0f;
#pragma omp parallel for schedule(static)
        for (unsigned expert = 0; expert < 896; expert++) {
            const unsigned char *row = weights + (size_t)expert * INW * 4;
            __m512d acc[RT / 8];
            _Pragma("GCC unroll 4") for (int j = 0; j < RT / 8; j++) acc[j] = _mm512_setzero_pd();
            for (unsigned c = 0; c < INW; c++) {
                const __m512d wb = _mm512_set1_pd((double)transformer_f32(row + c * 4));
                const float *xc = xt + (size_t)c * RT;
                _Pragma("GCC unroll 4") for (int j = 0; j < RT / 8; j++)
                    acc[j] = _mm512_add_pd(acc[j], _mm512_mul_pd(wb, _mm512_cvtps_pd(_mm256_loadu_ps(xc + j * 8))));
            }
            double sc[RT];
            _Pragma("GCC unroll 4") for (int j = 0; j < RT / 8; j++) _mm512_storeu_pd(sc + j * 8, acc[j]);
            const float bias = transformer_f32(tr->records[30].data + expert * 4);
            for (unsigned k = 0; k < cnt; k++) {
                float s = transformer_sigmoid((float)sc[k]);
                scores[(size_t)(g0 + k) * 896 + expert] = s;
                choices[(size_t)(g0 + k) * 896 + expert] = s + bias;
            }
        }
    }
    free(xt);
}
#endif
static void route_scores(const Transformer *tr, const float *postnorm, unsigned n, float *scores, float *choices)
{
#if defined(__AVX512F__)
    if (g_wideroute && n >= 16) { route_scores_wide(tr, postnorm, n, scores, choices); return; }
#endif
    const unsigned char *weights = tr->records[29].data;
    double (*x8)[8] = grab((size_t)INW * 8 * sizeof(double));
    for (unsigned g0 = 0; g0 < n; g0 += 8) {
        const unsigned cnt = n - g0 < 8 ? n - g0 : 8;
        for (unsigned c = 0; c < INW; c++)
            for (unsigned k = 0; k < 8; k++) x8[c][k] = k < cnt ? (double)postnorm[(size_t)(g0 + k) * INW + c] : 0.0;
#pragma omp parallel for schedule(static)
        for (unsigned expert = 0; expert < 896; expert++) {
            const unsigned char *row = weights + (size_t)expert * INW * 4;
            __m256d s0 = _mm256_setzero_pd(), s1 = _mm256_setzero_pd();
            for (unsigned c = 0; c < INW; c++) {
                const __m256d wb = _mm256_set1_pd((double)transformer_f32(row + c * 4));
                s0 = _mm256_add_pd(s0, _mm256_mul_pd(wb, _mm256_load_pd(x8[c])));
                s1 = _mm256_add_pd(s1, _mm256_mul_pd(wb, _mm256_load_pd(x8[c] + 4)));
            }
            double sc[8];
            _mm256_storeu_pd(sc, s0); _mm256_storeu_pd(sc + 4, s1);
            const float bias = transformer_f32(tr->records[30].data + expert * 4);
            for (unsigned k = 0; k < cnt; k++) {
                float s = transformer_sigmoid((float)sc[k]);
                scores[(size_t)(g0 + k) * 896 + expert] = s;
                choices[(size_t)(g0 + k) * 896 + expert] = s + bias;
            }
        }
    }
    free(x8);
}

#ifdef DETAIL
/* Diagnostic build only: every expert projection is repeated once straight away (into scratch) so a
   cold read can be compared with a warm one, and each thread's time is split by what it was doing. */
static double d_first[THREADS][3], d_second[THREADS][3], d_act[THREADS], d_busy[THREADS], d_region;
static unsigned long d_passes[THREADS], d_pairs[THREADS];
#define DT(...) __VA_ARGS__
#else
#define DT(...)
#endif

/* ---- experts, grouped by expert (see expert-bench.c; same arithmetic as root_project) ---- */
static inline __attribute__((always_inline)) void proj_core(const Root *root, unsigned expert, unsigned matrix,
    const double *const *conv, float *const *out, const int G)
{
    const unsigned width = matrix == 2 ? 3072 : 3584, rows = matrix == 2 ? 3584 : 3072;
    const unsigned char *maps = root->constants + root->palette_count * 2;
    const unsigned char *offsets = maps + root->map_count * 16;
    const unsigned char *refs = offsets + (root->template_count + 1) * 2;
    const unsigned template_id = root_u16(root->index + (expert * 3 + matrix) * 2);
    const unsigned begin = root_u16(offsets + template_id * 2);
    for (unsigned block = 0; block < rows / 64; block++) {
        const unsigned char *data = expert_base(root, expert) +
            (matrix < 2 ? (size_t)(matrix * 48 + block) * ROOT_RAW
                        : (size_t)96 * ROOT_RAW + (size_t)block * ROOT_DOWN_RAW);
        for (unsigned row = 0; row < 64; row++) {
            const unsigned char *codes = data + row * width / 2;
            const unsigned char *selectors = data + 64 * width / 2 + row * width / 32;
            __m256d l0[8], l4[8], l8[8], l12[8];
#pragma GCC unroll 8
            for (int g = 0; g < G; g++) {
                l0[g] = _mm256_setzero_pd(); l4[g] = _mm256_setzero_pd();
                l8[g] = _mm256_setzero_pd(); l12[g] = _mm256_setzero_pd();
            }
            for (unsigned coordinate = 0; coordinate < width; coordinate += 16) {
                const float (*pair)[2] = root->pairs[root_u16(refs + (begin + selectors[coordinate / 32]) * 2)];
                const unsigned char *packed = codes + coordinate / 2;
                __m256d w0 = _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd((const double *)pair[packed[0]]), _mm_load_sd((const double *)pair[packed[1]]))));
                __m256d w1 = _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd((const double *)pair[packed[2]]), _mm_load_sd((const double *)pair[packed[3]]))));
                __m256d w2 = _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd((const double *)pair[packed[4]]), _mm_load_sd((const double *)pair[packed[5]]))));
                __m256d w3 = _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd((const double *)pair[packed[6]]), _mm_load_sd((const double *)pair[packed[7]]))));
#pragma GCC unroll 8
                for (int g = 0; g < G; g++) {
                    const double *source = conv[g] + coordinate;
                    l0[g] = _mm256_add_pd(l0[g], _mm256_mul_pd(w0, _mm256_loadu_pd(source)));
                    l4[g] = _mm256_add_pd(l4[g], _mm256_mul_pd(w1, _mm256_loadu_pd(source + 4)));
                    l8[g] = _mm256_add_pd(l8[g], _mm256_mul_pd(w2, _mm256_loadu_pd(source + 8)));
                    l12[g] = _mm256_add_pd(l12[g], _mm256_mul_pd(w3, _mm256_loadu_pd(source + 12)));
                }
            }
#pragma GCC unroll 8
            for (int g = 0; g < G; g++) {
                double sums[4];
                _mm256_storeu_pd(sums, _mm256_add_pd(_mm256_add_pd(l0[g], l8[g]), _mm256_add_pd(l4[g], l12[g])));
                out[g][block * 64 + row] = (float)((sums[0] + sums[2]) + (sums[1] + sums[3]));
            }
        }
    }
}
typedef void (*ProjFn)(const Root *, unsigned, unsigned, const double *const *, float *const *);

#if defined(__AVX512F__)
/* 512-bit expert kernels (see expert-kernel-bench.c): R weight rows x G tokens per pass. Accumulator A = [l0 | l4],
   B = [l8 | l12] of the 256-bit kernel; per lane the same products are added in the same order (multiply, then a
   separate add), and the fold (l0+l8)+(l4+l12) is the same additions, so the bits are the same. */
#define KZ(NAME, R, G) \
static void NAME(const Root *root, unsigned expert, unsigned matrix, const double *const *conv, float *const *out) \
{ \
    const unsigned width = matrix == 2 ? 3072 : 3584, rows = matrix == 2 ? 3584 : 3072; \
    const unsigned char *maps = root->constants + root->palette_count * 2; \
    const unsigned char *offsets = maps + root->map_count * 16; \
    const unsigned char *refs = offsets + (root->template_count + 1) * 2; \
    const unsigned template_id = root_u16(root->index + (expert * 3 + matrix) * 2); \
    const unsigned begin = root_u16(offsets + template_id * 2); \
    for (unsigned block = 0; block < rows / 64; block++) { \
        const unsigned char *data = expert_base(root, expert) + \
            (matrix < 2 ? (size_t)(matrix * 48 + block) * ROOT_RAW : (size_t)96 * ROOT_RAW + (size_t)block * ROOT_DOWN_RAW); \
        for (unsigned row = 0; row < 64; row += R) { \
            const unsigned char *codes[R], *selectors[R]; \
            _Pragma("GCC unroll 8") for (int r = 0; r < R; r++) { \
                codes[r] = data + (row + r) * width / 2; \
                selectors[r] = data + 64 * width / 2 + (row + r) * width / 32; } \
            __m512d a[R][G], b[R][G]; \
            _Pragma("GCC unroll 8") for (int r = 0; r < R; r++) _Pragma("GCC unroll 8") for (int g = 0; g < G; g++) { \
                a[r][g] = _mm512_setzero_pd(); b[r][g] = _mm512_setzero_pd(); } \
            for (unsigned coordinate = 0; coordinate < width; coordinate += 16) { \
                __m512d wa[R], wb[R]; \
                _Pragma("GCC unroll 8") for (int r = 0; r < R; r++) { \
                    const float (*pair)[2] = root->pairs[root_u16(refs + (begin + selectors[r][coordinate / 32]) * 2)]; \
                    const unsigned char *packed = codes[r] + coordinate / 2; \
                    __m256d w0 = _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd((const double *)pair[packed[0]]), _mm_load_sd((const double *)pair[packed[1]])))); \
                    __m256d w1 = _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd((const double *)pair[packed[2]]), _mm_load_sd((const double *)pair[packed[3]])))); \
                    __m256d w2 = _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd((const double *)pair[packed[4]]), _mm_load_sd((const double *)pair[packed[5]])))); \
                    __m256d w3 = _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd((const double *)pair[packed[6]]), _mm_load_sd((const double *)pair[packed[7]])))); \
                    wa[r] = _mm512_insertf64x4(_mm512_castpd256_pd512(w0), w1, 1); \
                    wb[r] = _mm512_insertf64x4(_mm512_castpd256_pd512(w2), w3, 1); } \
                _Pragma("GCC unroll 8") for (int g = 0; g < G; g++) { \
                    const double *source = conv[g] + coordinate; \
                    const __m512d xa = _mm512_loadu_pd(source), xb = _mm512_loadu_pd(source + 8); \
                    _Pragma("GCC unroll 8") for (int r = 0; r < R; r++) { \
                        a[r][g] = _mm512_add_pd(a[r][g], _mm512_mul_pd(wa[r], xa)); \
                        b[r][g] = _mm512_add_pd(b[r][g], _mm512_mul_pd(wb[r], xb)); } \
                } \
            } \
            _Pragma("GCC unroll 8") for (int r = 0; r < R; r++) _Pragma("GCC unroll 8") for (int g = 0; g < G; g++) { \
                const __m512d t = _mm512_add_pd(a[r][g], b[r][g]); \
                double sums[4]; \
                _mm256_storeu_pd(sums, _mm256_add_pd(_mm512_castpd512_pd256(t), _mm512_extractf64x4_pd(t, 1))); \
                out[g][block * 64 + row + r] = (float)((sums[0] + sums[2]) + (sums[1] + sums[3])); } \
        } \
    } \
}
KZ(kz_1_1, 1, 1) KZ(kz_1_2, 1, 2) KZ(kz_1_4, 1, 4) KZ(kz_1_8, 1, 8)
KZ(kz_2_1, 2, 1) KZ(kz_2_2, 2, 2) KZ(kz_2_4, 2, 4)
#endif
/* ---- LB_EXPK=3 (experiment, NOT bit-identical): 8-bit activations. Every pair table of the experts is power-of-two scale x small
   integer q (0, 1, 2, 3, 4, 6, 8, 12, with sign), so a weight is exactly wscale[table] * q[nibble]. The activations of a token are cut
   into groups of 32 columns (the weights' group size), each scaled by its own max / 127 and rounded to int8; per group the integer dot
   product is exact (int16 multiply-add), then scaled and added in float. The only error is the rounding of the activations. ---- */
typedef struct { const Root *root; int8_t (*q)[16]; float *scale; } A8Tables;
static A8Tables a8_tabs[128];
static unsigned a8_n;
static const A8Tables *a8_get(const Root *root)
{
    for (unsigned i = 0; i < a8_n; i++) if (a8_tabs[i].root == root) return &a8_tabs[i];
    if (a8_n == 128) bail("too many layers for the 8-bit tables");
    A8Tables *t = &a8_tabs[a8_n];
    t->q = malloc((size_t)root->map_count * 16); t->scale = malloc((size_t)root->map_count * sizeof(float));
    for (unsigned m = 0; m < root->map_count; m++) {
        float v[16]; int a = 1 << 20;
        for (unsigned b = 0; b < 256; b++) {
            if (root->pairs[m][b][0] != root->pairs[m][b & 15][0] || root->pairs[m][b][1] != root->pairs[m][b >> 4][0]) bail("a pair table is not low nibble x high nibble");
        }
        for (unsigned b = 0; b < 16; b++) {
            v[b] = root->pairs[m][b][0];
            if (v[b] == 0.0f) continue;
            int e; const float mant = frexpf(fabsf(v[b]), &e);
            const int unit = mant == 0.5f ? e - 1 : mant == 0.75f ? e - 2 : (bail("a weight value is not 2^k or 3 x 2^k"), 0);
            if (unit < a) a = unit;
        }
        if (a == 1 << 20) a = 0;
        const float ws = ldexpf(1.0f, a);
        t->scale[m] = ws;
        for (unsigned b = 0; b < 16; b++) {
            const int q = (int)lrintf(v[b] / ws);
            if ((float)q * ws != v[b] || q < -127 || q > 127) bail("a weight value is not an integer multiple of its table's unit");
            t->q[m][b] = (int8_t)q;
        }
    }
    t->root = root; a8_n++;
    return t;
}
#define KA8(NAME, G) \
static void NAME(const Root *root, unsigned expert, unsigned matrix, const double *const *conv, float *const *out) \
{ \
    const A8Tables *tb = a8_get(root); \
    const unsigned width = matrix == 2 ? 3072 : 3584, rows = matrix == 2 ? 3584 : 3072, groups = width / 32; \
    const unsigned char *maps = root->constants + root->palette_count * 2; \
    const unsigned char *offsets = maps + root->map_count * 16; \
    const unsigned char *refs = offsets + (root->template_count + 1) * 2; \
    const unsigned template_id = root_u16(root->index + (expert * 3 + matrix) * 2); \
    const unsigned begin = root_u16(offsets + template_id * 2); \
    static __thread int16_t qa[G][3584] __attribute__((aligned(64))); \
    static __thread float as[G][112]; \
    for (int g = 0; g < G; g++) \
        for (unsigned gi = 0; gi < groups; gi++) { \
            const double *x = conv[g] + gi * 32; double amax = 0; \
            for (unsigned j = 0; j < 32; j++) { const double a = fabs(x[j]); if (a > amax) amax = a; } \
            const float inv = amax > 0 ? (float)(127.0 / amax) : 0.0f; \
            as[g][gi] = (float)(amax / 127.0); \
            for (unsigned j = 0; j < 16; j++) { \
                qa[g][gi * 32 + j] = (int16_t)lrintf((float)x[2 * j] * inv); \
                qa[g][gi * 32 + 16 + j] = (int16_t)lrintf((float)x[2 * j + 1] * inv); } \
        } \
    const __m128i low = _mm_set1_epi8(0x0F); \
    for (unsigned block = 0; block < rows / 64; block++) { \
        const unsigned char *data = expert_base(root, expert) + \
            (matrix < 2 ? (size_t)(matrix * 48 + block) * ROOT_RAW : (size_t)96 * ROOT_RAW + (size_t)block * ROOT_DOWN_RAW); \
        for (unsigned row = 0; row < 64; row++) { \
            const unsigned char *codes = data + row * width / 2, *selectors = data + 64 * width / 2 + row * width / 32; \
            __m512 acc[G]; \
            _Pragma("GCC unroll 4") for (int g = 0; g < G; g++) acc[g] = _mm512_setzero_ps(); \
            for (unsigned gi = 0; gi < groups; gi++) { \
                const unsigned ref = root_u16(refs + (begin + selectors[gi]) * 2); \
                const __m128i bytes = _mm_loadu_si128((const __m128i *)(codes + gi * 16)); \
                const __m128i lut = _mm_loadu_si128((const __m128i *)tb->q[ref]); \
                const __m128i qe = _mm_shuffle_epi8(lut, _mm_and_si128(bytes, low)); \
                const __m128i qo = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(bytes, 4), low)); \
                const __m512i w32 = _mm512_cvtepi8_epi16(_mm256_set_m128i(qo, qe)); \
                const float ws = tb->scale[ref]; \
                _Pragma("GCC unroll 4") for (int g = 0; g < G; g++) { \
                    const __m512i prod = _mm512_madd_epi16(w32, _mm512_load_si512((const void *)(qa[g] + gi * 32))); \
                    acc[g] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(prod), _mm512_set1_ps(ws * as[g][gi]), acc[g]); } \
            } \
            _Pragma("GCC unroll 4") for (int g = 0; g < G; g++) out[g][block * 64 + row] = _mm512_reduce_add_ps(acc[g]); \
        } \
    } \
}
KA8(ka8_1, 1) KA8(ka8_2, 2) KA8(ka8_4, 4)
static int g_expk = 1; /* env LB_EXPK: 0 = shipped 256-bit, 4 tokens per pass; 1 = 512-bit, 2 rows x 4 tokens; 2 = 512-bit, 1 row x 8 tokens; 3 = 8-bit activations (not bit-identical) */
static int g_sweep_expk = 0; /* env LB_SWEEP_EXPK: the same choice for a window of several decode tokens (_stepk) */
static int g_pred; /* env LB_PRED=1: report how well routing on the layer input predicts the real experts (diagnostic) */
static int g_lin; /* env LB_LIN=1: report whether experts behave like one linear map (diagnostic, needs LB_DISK=0) */
static void proj_g1(const Root *r, unsigned e, unsigned m, const double *const *c, float *const *o) { proj_core(r, e, m, c, o, 1); }
static void proj_g2(const Root *r, unsigned e, unsigned m, const double *const *c, float *const *o) { proj_core(r, e, m, c, o, 2); }
static void proj_g4(const Root *r, unsigned e, unsigned m, const double *const *c, float *const *o) { proj_core(r, e, m, c, o, 4); }
static ProjFn proj_for(int g)
{
#if defined(__AVX512F__)
    if (g_expk == 3) return g >= 4 ? ka8_4 : g == 2 ? ka8_2 : ka8_1;
    if (g_expk == 1) return g >= 4 ? kz_2_4 : g == 2 ? kz_2_2 : kz_2_1;
    if (g_expk == 2) return g >= 8 ? kz_1_8 : g >= 4 ? kz_1_4 : g == 2 ? kz_1_2 : kz_1_1;
#endif
    return g >= 4 ? proj_g4 : g == 2 ? proj_g2 : proj_g1;
}

/* LB_REPEAT: one expert matrix decoded to float32 (rows x width), as dump_expert does. */
static float *rep_decode(const Root *root, unsigned expert, unsigned matrix, unsigned *rows_out, unsigned *width_out)
{
    const unsigned width = matrix == 2 ? 3072 : 3584, rows = matrix == 2 ? 3584 : 3072;
    const unsigned char *maps = root->constants + root->palette_count * 2;
    const unsigned char *offsets = maps + root->map_count * 16;
    const unsigned char *refs = offsets + (root->template_count + 1) * 2;
    const unsigned begin = root_u16(offsets + root_u16(root->index + (expert * 3 + matrix) * 2) * 2);
    float *w = malloc((size_t)rows * width * 4);
    for (unsigned block = 0; block < rows / 64; block++) {
        const unsigned char *data = expert_base(root, expert) +
            (matrix < 2 ? (size_t)(matrix * 48 + block) * ROOT_RAW : (size_t)96 * ROOT_RAW + (size_t)block * ROOT_DOWN_RAW);
        for (unsigned row = 0; row < 64; row++) {
            const unsigned char *codes = data + row * width / 2, *selectors = data + 64 * width / 2 + row * width / 32;
            for (unsigned c = 0; c < width; c++) {
                const float (*pair)[2] = root->pairs[root_u16(refs + (begin + selectors[c / 32]) * 2)];
                w[(size_t)(block * 64 + row) * width + c] = pair[codes[c / 2]][c & 1];
            }
        }
    }
    *rows_out = rows; *width_out = width;
    return w;
}
static void rep_pairs_matrix(const char *name, const float *w, unsigned rows, unsigned width, const double *x)
{
    hs_reset();
    for (unsigned r = 0; r < rows; r++)
        for (unsigned c = 0; c < width; c++) { unsigned wb; memcpy(&wb, w + (size_t)r * width + c, 4); hs_add(pair_key((float)x[c], wb)); }
    rep_pairs_print(name, (unsigned long)rows * width);
}
/* Captured expert-internal vectors (raw gate, raw up, activated) of up to 128 (token, expert) calls, and one expert's operand pairs. */
static float g_cap[3][128][EXPW];
static unsigned g_cap_n;
static int g_rep_expert_done;
static void rep_expert_capture(const Root *root, unsigned expert, unsigned g, const float (*raw_gate)[EXPW], const float (*raw_up)[EXPW],
    const double (*activated)[EXPW], const double *const *x_lat)
{
#pragma omp critical(rep_expert)
    {
        for (unsigned k = 0; k < g && g_cap_n < 128; k++, g_cap_n++) {
            memcpy(g_cap[0][g_cap_n], raw_gate[k], EXPW * 4); memcpy(g_cap[1][g_cap_n], raw_up[k], EXPW * 4);
            for (unsigned i = 0; i < EXPW; i++) g_cap[2][g_cap_n][i] = (float)activated[k][i];
        }
        if (!g_rep_expert_done) {
            g_rep_expert_done = 1;
            static const char *const names[3] = { "expert gate matrix", "expert up matrix", "expert down matrix" };
            for (unsigned matrix = 0; matrix < 3; matrix++) {
                unsigned rows, width;
                float *w = rep_decode(root, expert, matrix, &rows, &width);
                unsigned *sorted = malloc((size_t)rows * width * 4);
                memcpy(sorted, w, (size_t)rows * width * 4);
                qsort(sorted, (size_t)rows * width, 4, cmp_u32);
                unsigned distinct = 1; for (size_t i = 1; i < (size_t)rows * width; i++) distinct += sorted[i] != sorted[i - 1];
                printf("REP weights %-34s expert %u: %u distinct float values among %u weights\n", names[matrix], expert, distinct, rows * width);
                free(sorted);
                if (matrix < 2) {
                    rep_pairs_matrix(names[matrix], w, rows, width, x_lat[0]);
                } else {
                    double act0[EXPW];
                    for (unsigned i = 0; i < EXPW; i++) act0[i] = activated[0][i];
                    rep_pairs_matrix(names[matrix], w, rows, width, act0);
                }
                free(w);
            }
        }
    }
}

static void expert_apply(const Root *root, unsigned expert, const unsigned *token, const unsigned *rank,
    unsigned n, const double (*lat)[LATW], float (*out)[16][LATW])
{
    float gate[8][EXPW], up[8][EXPW];
    double act[8][EXPW];
    unsigned done = 0;
    if (g_disk == 3) { const double w0 = clk(); wait_expert(expert); g_wait_t[omp_get_thread_num() & 63] += clk() - w0; }
    DT(const int th = omp_get_thread_num(); float g2[8][EXPW], u2[8][EXPW], dn2[8][LATW];
       float *go2[8], *uo2[8], *do2[8]; const double e0 = clk(); double extra = 0;)
    while (done < n) {
        int g = g_expk == 2 ? 8 : 4;
        while (g > (int)(n - done)) g >>= 1;
        const double *cin[8], *ain[8];
        float *go[8], *uo[8], *dout[8];
        for (int k = 0; k < g; k++) { cin[k] = lat[token[done + k]]; go[k] = gate[k]; uo[k] = up[k]; }
        DT(double a0 = clk();)
        proj_for(g)(root, expert, 0, cin, go);
        DT(for (int k = 0; k < g; k++) { go2[k] = g2[k]; uo2[k] = u2[k]; do2[k] = dn2[k]; }
           double a1 = clk(); d_first[th][0] += a1 - a0;
           proj_for(g)(root, expert, 0, cin, go2); double a2 = clk(); extra += a2 - a1; d_second[th][0] += a2 - a1;)
        DT(double b0 = clk();)
        proj_for(g)(root, expert, 1, cin, uo);
        DT(double b1 = clk(); d_first[th][1] += b1 - b0;
           proj_for(g)(root, expert, 1, cin, uo2); double b2 = clk(); extra += b2 - b1; d_second[th][1] += b2 - b1;)
        DT(double c0 = clk();)
        static __thread float rg[8][EXPW], ru[8][EXPW];
        if (g_rep_active) for (int k = 0; k < g; k++) { memcpy(rg[k], gate[k], EXPW * 4); memcpy(ru[k], up[k], EXPW * 4); }
        for (int k = 0; k < g; k++) {
            transformer_activation(gate[k], up[k], EXPW);
            for (unsigned i = 0; i < EXPW; i++) act[k][i] = (double)gate[k][i];
            ain[k] = act[k];
            dout[k] = out[token[done + k]][rank[done + k]];
        }
        if (g_rep_active) rep_expert_capture(root, expert, (unsigned)g, (const float (*)[EXPW])rg, (const float (*)[EXPW])ru, (const double (*)[EXPW])act, cin);
        DT(double c1 = clk(); d_act[th] += c1 - c0;)
        proj_for(g)(root, expert, 2, ain, dout);
        DT(double f1 = clk(); d_first[th][2] += f1 - c1;
           proj_for(g)(root, expert, 2, ain, do2); double f2 = clk(); extra += f2 - f1; d_second[th][2] += f2 - f1;
           d_passes[th]++; d_pairs[th] += (unsigned long)g;)
        done += (unsigned)g;
    }
    DT(d_busy[th] += (clk() - e0) - extra;)
}

typedef struct { unsigned expert, count, start; } Group;
static int by_count_desc(const void *a, const void *b)
{
    const Group *x = a, *y = b;
    return x->count != y->count ? (x->count < y->count ? 1 : -1) : (x->expert > y->expert) - (x->expert < y->expert);
}

static void run_experts(const Root *root, unsigned count, unsigned (*sel)[16], const double (*lat)[LATW], float (*out)[16][LATW])
{
    unsigned total[896] = {0}, fill[896];
    for (unsigned t = 0; t < count; t++) for (unsigned r = 0; r < 16; r++) total[sel[t][r]]++;
    for (unsigned e = 0; e < 896; e++) g_hits[e] += total[e];
    unsigned *ptok = grab((size_t)count * 16 * sizeof(unsigned)), *prank = grab((size_t)count * 16 * sizeof(unsigned));
    Group *groups = grab(896 * sizeof(Group));
    unsigned n = 0, offset = 0;
    for (unsigned e = 0; e < 896; e++) if (total[e]) { groups[n++] = (Group){e, total[e], offset}; offset += total[e]; }
    for (unsigned g = 0; g < n; g++) fill[groups[g].expert] = groups[g].start;
    for (unsigned t = 0; t < count; t++)
        for (unsigned r = 0; r < 16; r++) { unsigned e = sel[t][r]; ptok[fill[e]] = t; prank[fill[e]] = r; fill[e]++; }
    qsort(groups, n, sizeof(Group), by_count_desc);
    g_distinct_sum += n; g_window_count++;
    if (g_disk == 2)
        for (unsigned g = 0; g < n; g++)
            madvise((void *)(root->direct + (size_t)groups[g].expert * ROOT_EXPERT_RAW), ROOT_EXPERT_RAW, MADV_WILLNEED);
    if (g_disk == 3) { unsigned order[896]; for (unsigned g = 0; g < n; g++) order[g] = groups[g].expert; if (g_cache) cache_assign(order, n); prefetch_publish(order, n); }
    DT(const double r0 = clk();)
#pragma omp parallel for schedule(dynamic, 1) num_threads(THREADS)
    for (unsigned g = 0; g < n; g++)
        expert_apply(root, groups[g].expert, ptok + groups[g].start, prank + groups[g].start, groups[g].count, lat, out);
    if (g_disk == 3) prefetch_finish(total);
    if (g_disk == 3 && g_cache) cache_finish();
    DT(d_region += clk() - r0;)
    free(ptok); free(prank); free(groups);
}

/* ---- buffers for one window ---- */
typedef struct {
    float *normalized, *raw[3], *qkv[3], *beta, *dhid, *decay, *attention, *gate, *aout, *residual, *postnorm;
    float *scores, *choices, *latent, *mixture, *lnorm, *routed, *sgate, *sup, *sout;
    unsigned (*sel)[16]; float (*wts)[16];
    double (*lat)[LATW]; float (*down)[16][LATW];
    float *ql, *qq, *kvl, *exq; /* MLA only: query latent, query, key/value latent, expanded key/value */
} Buf;

/* MLA only. Per head: the 128 key coordinates, position-contiguous (so 8 positions are one load), the 128 value
   coordinates of each position, and the 64 positional coordinates shared by all heads, also position-contiguous.
   Grows by one position per token like the pod's cache; positions are the token index within the request. */
enum { MLA_QLAT = 1536, MLA_QROWS = 18432, MLA_KVW = 576, MLA_LAT = 512, MLA_EXROWS = 24576 };
typedef struct { float *key, *value, *rope; size_t cap; float *scratch; } MlaCache;
static MlaCache g_mla;
static size_t g_pos0;   /* position of the first token of this call within the sequence (decode steps continue a prompt) */
/* Speculative decoding: when set, a KDA pass of n tokens stores the recurrent state and convolution history after each of its first n-1 tokens here
   (indexed by token), so a rejected draft is undone by copying one back instead of running the accepted tokens again. */
static __attribute__((unused)) float (*g_checkpoint_rec)[NHEAD][HDIM][HDIM];
static __attribute__((unused)) float (*g_checkpoint_hist)[3][ROWS][3];
static void mla_cache_make(unsigned max_n)
{
    g_mla.cap = ((size_t)max_n + 7) / 8 * 8 + 8;
    g_mla.key = grab((size_t)NHEAD * 128 * g_mla.cap * 4);
    g_mla.value = grab((size_t)NHEAD * g_mla.cap * 128 * 4);
    g_mla.rope = grab((size_t)64 * g_mla.cap * 4);
    g_mla.scratch = grab((size_t)64 * 2 * g_mla.cap * 4);
}

static Buf buf_make(unsigned w)
{
    Buf b;
    b.normalized = grab((size_t)w * INW * 4);
    for (int c = 0; c < 3; c++) { b.raw[c] = grab((size_t)w * ROWS * 4); b.qkv[c] = grab((size_t)w * ROWS * 4); }
    b.beta = grab((size_t)w * NHEAD * 4); b.dhid = grab((size_t)w * HDIM * 4); b.decay = grab((size_t)w * ROWS * 4);
    b.attention = grab((size_t)w * ROWS * 4); b.gate = grab((size_t)w * ROWS * 4); b.aout = grab((size_t)w * INW * 4);
    b.residual = grab((size_t)w * INW * 4); b.postnorm = grab((size_t)w * INW * 4);
    b.scores = grab((size_t)w * 896 * 4); b.choices = grab((size_t)w * 896 * 4);
    b.latent = grab((size_t)w * LATW * 4); b.mixture = grab((size_t)w * LATW * 4); b.lnorm = grab((size_t)w * LATW * 4);
    b.routed = grab((size_t)w * INW * 4); b.sgate = grab((size_t)w * SHRW * 4); b.sup = grab((size_t)w * SHRW * 4);
    b.sout = grab((size_t)w * INW * 4);
    b.sel = grab((size_t)w * 16 * sizeof(unsigned)); b.wts = grab((size_t)w * 16 * sizeof(float));
    b.lat = grab((size_t)w * LATW * sizeof(double)); b.down = grab((size_t)w * 16 * LATW * sizeof(float));
#if LB_MLA
    b.ql = grab((size_t)w * MLA_QLAT * 4); b.qq = grab((size_t)w * MLA_QROWS * 4);
    b.kvl = grab((size_t)w * MLA_KVW * 4); b.exq = grab((size_t)w * MLA_EXROWS * 4);
#endif
    return b;
}

/* Pod state of the first CHECK tokens, for the stage-by-stage comparison. */
typedef struct {
    float normalized[CHECK][INW], qkv[3][CHECK][ROWS], decay[CHECK][ROWS], beta[CHECK][NHEAD];
    float attention[CHECK][ROWS], gate[CHECK][ROWS], postnorm[CHECK][INW], latent[CHECK][LATW];
    float mixture[CHECK][LATW], routed[CHECK][INW], sout[CHECK][INW];
    unsigned sel[CHECK][16]; float wts[CHECK][16];
#if LB_MLA
    float mq[CHECK][MLA_QROWS], mex[CHECK][MLA_EXROWS];
#endif
} PodStages;

static int cmp_stage(const char *name, const void *mine, const void *pod, size_t elems, size_t width, size_t size)
{
    for (size_t t = 0; t < CHECK; t++) {
        const unsigned char *a = (const unsigned char *)mine + t * elems * size, *b = (const unsigned char *)pod + t * width * size;
        if (memcmp(a, b, width * size)) {
            size_t i = 0; while (!memcmp(a + i * size, b + i * size, size)) i++;
            printf("    stage %-16s DIFFERS at token %zu element %zu\n", name, t, i);
            return 1;
        }
    }
    printf("    stage %-16s identical (first %d tokens)\n", name, CHECK);
    return 0;
}

/* Routes each token on the layer INPUT (attention skipped) and compares with the real top-16. Not bit-exact by design. */
static void pred_report(const Transformer *tr, const float *in, const float *const *snp, unsigned w, const Buf *b)
{
    float *pn = grab((size_t)w * INW * 4), *ps = grab((size_t)w * 896 * 4), *pc = grab((size_t)w * 896 * 4);
#pragma omp parallel for schedule(static)
    for (unsigned t = 0; t < w; t++) {
        float aggregate[INW];
        const float *src[10];
        for (unsigned s = 0; s < LB_K; s++) src[s] = snp[s] + (size_t)t * INW;
        src[LB_K] = in + (size_t)t * INW;
        agg(tr, 38, src, LB_K + 1, aggregate);
        transformer_normalize(pn + (size_t)t * INW, aggregate, tr->records[5].data, INW);
    }
    route_scores(tr, pn, w, ps, pc);
    enum { NK = 4 };
    static const unsigned ks[NK] = { 16, 32, 64, 128 };
    unsigned char (*pk)[NK][896] = calloc(w, sizeof *pk);
    for (unsigned t = 0; t < w; t++) {
        unsigned char taken[896] = {0};
        for (unsigned r = 0, ki = 0; r < ks[NK - 1]; r++) {
            int best = -1;
            for (int e = 0; e < 896; e++)
                if (!taken[e] && (best < 0 || pc[(size_t)t * 896 + e] > pc[(size_t)t * 896 + best])) best = e;
            taken[best] = 1;
            for (ki = 0; ki < NK; ki++) if (r < ks[ki]) pk[t][ki][best] = 1;
        }
    }
    printf("  expert prediction from the layer input (pre-attention), top-K guesses per token:\n");
    for (unsigned k = 0; k < NK; k++) {
        double hit = 0;
        for (unsigned t = 0; t < w; t++) for (unsigned r = 0; r < 16; r++) hit += pk[t][k][b->sel[t][r]];
        printf("    K=%-3u per token: %.2f of the real 16 are in the guess (%.1f%%)\n", ks[k], hit / w, 100.0 * hit / (16.0 * w));
    }
    static const unsigned gs[] = { 8, 32, 128, 1000 };
    for (unsigned gi = 0; gi < 4; gi++) {
        const unsigned G = gs[gi];
        if (G > w) continue;
        double need = 0, cov[NK] = {0}, size[NK] = {0}; unsigned groups = 0;
        for (unsigned g0 = 0; g0 + G <= w; g0 += G, groups++) {
            unsigned char tu[896] = {0}, pu[NK][896] = {{0}};
            for (unsigned t = g0; t < g0 + G; t++) {
                for (unsigned r = 0; r < 16; r++) tu[b->sel[t][r]] = 1;
                for (unsigned k = 0; k < NK; k++) for (int e = 0; e < 896; e++) pu[k][e] |= pk[t][k][e];
            }
            for (int e = 0; e < 896; e++) {
                need += tu[e];
                for (unsigned k = 0; k < NK; k++) { cov[k] += tu[e] && pu[k][e]; size[k] += pu[k][e]; }
            }
        }
        printf("    window %-4u real distinct experts %.1f;", G, need / groups);
        for (unsigned k = 0; k < NK; k++) printf("  K=%u: guessed %.1f, covers %.1f%%", ks[k], size[k] / groups, 100.0 * cov[k] / need);
        printf("\n");
    }
    free(pk); free(pn); free(ps); free(pc);
}

/* A linear map f would satisfy f(s*x) = s*f(x). Runs every expert again on scaled inputs; outputs are not used further. */
static void lin_report(const Root *root, const Buf *b, unsigned w)
{
    double (*lat2)[LATW] = grab((size_t)w * LATW * sizeof(double));
    float (*down2)[16][LATW] = grab((size_t)w * 16 * LATW * sizeof(float));
    static const double scales[2] = { 0.5, 2.0 };
    printf("  linearity test, f(s*x) against s*f(x) for the 16 routed experts of each token:\n");
    for (int si = 0; si < 2; si++) {
        const double s = scales[si];
        for (unsigned t = 0; t < w; t++) for (unsigned c = 0; c < LATW; c++) lat2[t][c] = b->lat[t][c] * s;
        run_experts(root, w, b->sel, (const double (*)[LATW])lat2, down2);
        double sum = 0, lo = 1e300, hi = 0, nsum = 0; unsigned cnt = 0;
        for (unsigned t = 0; t < w; t++)
            for (unsigned r = 0; r < 16; r++) {
                double e = 0, ref = 0, got = 0;
                for (unsigned c = 0; c < LATW; c++) {
                    const double d = down2[t][r][c], x = s * (double)b->down[t][r][c];
                    e += (d - x) * (d - x); ref += x * x; got += d * d;
                }
                const double rel = sqrt(e / ref);
                sum += rel; nsum += sqrt(got / ref); cnt++;
                if (rel < lo) lo = rel;
                if (rel > hi) hi = rel;
            }
        printf("    s=%.1f: relative error mean %.3f (min %.3f, max %.3f); size of f(s*x) against s*f(x): %.3f (a linear map gives 1.000)\n",
            s, sum / cnt, lo, hi, nsum / cnt);
    }
    free(lat2); free(down2);
}

/* Diagnostic for single-token calls (env PF_ROUTES in the whole-model driver): the 16 experts the token really chose and
   the 32 best by routing on the layer input before attention (what could be guessed before this layer's attention ran). */
static int g_route_log;
static unsigned short g_route_rec[48];
static void route_log(const Transformer *tr, const float *in, const float *const *snp, const Buf *b)
{
    float aggregate[INW], pn[INW], ps[896], pc[896];
    const float *src[10];
    for (unsigned s = 0; s < LB_K; s++) src[s] = snp[s];
    src[LB_K] = in;
    agg(tr, 38, src, LB_K + 1, aggregate);
    transformer_normalize(pn, aggregate, tr->records[5].data, INW);
    route_scores(tr, pn, 1, ps, pc);
    for (unsigned r = 0; r < 16; r++) g_route_rec[r] = (unsigned short)b->sel[0][r];
    unsigned char taken[896] = {0};
    for (unsigned r = 0; r < 32; r++) {
        int best = -1;
        for (int e = 0; e < 896; e++) if (!taken[e] && (best < 0 || pc[e] > pc[best])) best = e;
        taken[best] = 1; g_route_rec[16 + r] = (unsigned short)best;
    }
}

/* Single-token decode step with the cache on: guess the experts from the layer input and have the readers start on the likeliest ones
   that are not in RAM yet, best first, at most g_guess_max (and never more than the cache can hold besides the 16 a step needs). */
static void guess_experts(const Transformer *tr, const float *in, const float *const *snp)
{
    ExpertCache *c = g_cache;
    float aggregate[INW], pn[INW], ps[896], pc[896];
    const float *src[10];
    for (unsigned s = 0; s < LB_K; s++) src[s] = snp[s];
    src[LB_K] = in;
    agg(tr, 38, src, LB_K + 1, aggregate);
    transformer_normalize(pn, aggregate, tr->records[5].data, INW);
    route_scores(tr, pn, 1, ps, pc);
    unsigned limit = (unsigned)g_guess_max;
    if (limit > c->cap - 16) limit = c->cap - 16;
    const unsigned long now = ++g_cache_clock;
    unsigned char taken[896] = {0};
    g_guess_count = 0;
    for (unsigned r = 0; r < 24 && g_guess_count < limit; r++) {
        int best = -1;
        for (int e = 0; e < 896; e++) if (!taken[e] && (best < 0 || pc[e] > pc[best])) best = e;
        taken[best] = 1;
        if (c->where[best] >= 0) { c->stamp[c->where[best]] = now; continue; }   /* already in RAM: keep it */
        unsigned v = CACHE_MAX; unsigned long old = ~0UL;
        for (unsigned j = 0; j < c->cap; j++) if (c->stamp[j] != now && c->stamp[j] < old) { old = c->stamp[j]; v = j; }
        if (v == CACHE_MAX) break;
        if (c->expert[v] >= 0) c->where[c->expert[v]] = -1;
        c->expert[v] = best; c->where[best] = (short)v; c->stamp[v] = now; g_ptr[best] = c->slots[v];
        g_guessed[best] = 1; g_guess_list[g_guess_count] = (unsigned)best; g_spec_ids[g_guess_count] = (unsigned)best;
        __atomic_store_n(&g_spec_list_n, ++g_guess_count, __ATOMIC_RELEASE);
    }
    g_guess_issued += g_guess_count;
}

static void run_layer(const Transformer *tr, const Root *root, const float *resid, const float *const *snaps, unsigned n,
    unsigned window, Buf *b, float *final_out, float (*hist)[ROWS][3], float (*rec)[HDIM][HDIM],
    const PodStages *pod)
{
#if !LB_MLA
    const Mat mq[3] = { mat_qkv(tr, 0), mat_qkv(tr, 1), mat_qkv(tr, 2) };
    const Mat m11 = mat_record(tr, 11, "beta"), m12 = mat_record(tr, 12, "decay_down"), m13 = mat_record(tr, 13, "decay_up");
#else
    const Mat ma20 = mat_oprec(tr, 20, "mla_q_down"), ma22 = mat_oprec(tr, 22, "mla_q_up"),
              ma23 = mat_oprec(tr, 23, "mla_kv_down"), ma25 = mat_oprec(tr, 25, "mla_kv_up");
#endif
    const Mat m6 = mat_record(tr, 6, "gate"), m7 = mat_record(tr, 7, "attention_out"),
              m31 = mat_record(tr, 31, "latent_down"), m32 = mat_record(tr, 32, "latent_up"),
              m34 = mat_record(tr, 34, "shared_gate"), m35 = mat_record(tr, 35, "shared_up"), m36 = mat_record(tr, 36, "shared_down");
    for (unsigned p = 0; p < n; p += window) {
        const unsigned w = n - p < window ? n - p : window;
        if (g_rep && !g_rep_done && p == 0) g_rep_active = 1;
        const float *in = resid + (size_t)p * INW;
        const float *snp[8];
        for (unsigned s = 0; s < LB_K; s++) snp[s] = snaps[s] + (size_t)p * INW;
        if (g_disk == 1 || g_disk == 2) { const double e0 = clk(); evict_experts(root); g_evict_s += clk() - e0; }
        lap_last = clk();
        if (g_disk == 3) prefetch_start();
        if (g_disk == 3 && g_cache && g_guess_max > 0 && w == 1) { guess_experts(tr, in, snp); lap(19, "start readers, early guess"); }

#pragma omp parallel for schedule(static)
        for (unsigned t = 0; t < w; t++) {
            float aggregate[INW];
            const float *src[10];
            for (unsigned s = 0; s < LB_K; s++) src[s] = snp[s] + (size_t)t * INW;
            src[LB_K] = in + (size_t)t * INW;
            agg(tr, 37, src, LB_K + 1, aggregate);
            transformer_normalize(b->normalized + (size_t)t * INW, aggregate, LB_INPUT_GAINS(tr), INW);
        }
        lap(0, "aggregate + norm (pre-attn)");
        if (g_rep_active) {
            rep_buf("layer input (residual)", in, w, INW);
            for (unsigned s = 0; s < LB_K; s++) rep_buf("input snapshot", snp[s], w, INW);
            rep_buf("normalized (pre-attention)", b->normalized, w, INW);
            rep_growth("normalized (pre-attention)", b->normalized, w, INW);
        }

#if LB_MLA
        {   /* MLA: query and key/value latents, then the causal cache; each token reads every earlier position. */
            bproject(&ma20, b->normalized, INW, b->ql, MLA_QLAT, w);
#pragma omp parallel for schedule(static)
            for (unsigned t = 0; t < w; t++)
                transformer_normalize(b->ql + (size_t)t * MLA_QLAT, b->ql + (size_t)t * MLA_QLAT, tr->operator_records[21].data, MLA_QLAT);
            bproject(&ma22, b->ql, MLA_QLAT, b->qq, MLA_QROWS, w);
            bproject(&ma23, b->normalized, INW, b->kvl, MLA_KVW, w);
#pragma omp parallel for schedule(static)
            for (unsigned t = 0; t < w; t++)
                transformer_normalize(b->kvl + (size_t)t * MLA_KVW, b->kvl + (size_t)t * MLA_KVW, tr->operator_records[24].data, MLA_LAT);
            bproject(&ma25, b->kvl, MLA_KVW, b->exq, MLA_EXROWS, w);
            lap(1, "MLA q and key/value projections");
            rep_buf("MLA query latent", b->ql, w, MLA_QLAT); rep_buf("MLA query", b->qq, w, MLA_QROWS);
            rep_buf("MLA key/value latent", b->kvl, w, MLA_KVW); rep_buf("MLA expanded key/value", b->exq, w, MLA_EXROWS);

            const size_t cap = g_mla.cap;
#pragma omp parallel for schedule(static)
            for (unsigned head = 0; head < NHEAD; head++)
                for (unsigned t = 0; t < w; t++) {
                    const size_t pos = g_pos0 + p + t;
                    const float *ex = b->exq + (size_t)t * MLA_EXROWS + head * 256;
                    for (unsigned c = 0; c < 128; c++) g_mla.key[((size_t)head * 128 + c) * cap + pos] = ex[c];
                    memcpy(g_mla.value + ((size_t)head * cap + pos) * 128, ex + 128, 128 * sizeof(float));
                }
#pragma omp parallel for schedule(static)
            for (unsigned c = 0; c < 64; c++)
                for (unsigned t = 0; t < w; t++) g_mla.rope[(size_t)c * cap + g_pos0 + p + t] = b->kvl[(size_t)t * MLA_KVW + MLA_LAT + c];
            lap(2, "MLA cache store");

            /* Per (head, token): the score of each earlier position is one double accumulator added in the pod's order
               (128 key coordinates, then 64 positional), here eight positions per instruction; softmax and the weighted
               sum of values follow the pod's loops, the sum vectorised over the 128 value coordinates. */
            const float scale = 1.0f / sqrtf(192.0f);
#pragma omp parallel
            {
                float *scores = g_mla.scratch + (size_t)omp_get_thread_num() * 2 * cap, *exps = scores + cap;
#pragma omp for schedule(dynamic, 1) collapse(2)
                for (unsigned head = 0; head < NHEAD; head++)
                    for (unsigned tt = 0; tt < w; tt++) {
                        const unsigned t = w - 1 - tt;   /* longest walks first */
                        const size_t count = g_pos0 + (size_t)p + t + 1;
                        const float *q = b->qq + (size_t)t * MLA_QROWS + head * 192;
                        const float *kt = g_mla.key + (size_t)head * 128 * cap;
                        for (size_t j0 = 0; j0 < count; j0 += 8) {
                            __m512d acc = _mm512_setzero_pd();
                            for (unsigned c = 0; c < 128; c++)
                                acc = _mm512_add_pd(acc, _mm512_mul_pd(_mm512_set1_pd((double)q[c]),
                                    _mm512_cvtps_pd(_mm256_loadu_ps(kt + (size_t)c * cap + j0))));
                            for (unsigned c = 0; c < 64; c++)
                                acc = _mm512_add_pd(acc, _mm512_mul_pd(_mm512_set1_pd((double)q[128 + c]),
                                    _mm512_cvtps_pd(_mm256_loadu_ps(g_mla.rope + (size_t)c * cap + j0))));
                            _mm256_storeu_ps(scores + j0, _mm256_mul_ps(_mm512_cvtpd_ps(acc), _mm256_set1_ps(scale)));
                        }
                        float maximum = scores[0];
                        for (size_t j = 1; j < count; j++) if (scores[j] > maximum) maximum = scores[j];
                        double total = 0.0;
                        for (size_t j = 0; j < count; j++) { exps[j] = expf(scores[j] - maximum); total += (double)exps[j]; }
                        __m512 acc[8];
                        _Pragma("GCC unroll 8") for (int v = 0; v < 8; v++) acc[v] = _mm512_setzero_ps();
                        const float *values = g_mla.value + (size_t)head * cap * 128;
                        for (size_t j = 0; j < count; j++) {
                            const __m512 wv = _mm512_set1_ps((float)((double)exps[j] / total));
                            const float *vj = values + j * 128;
                            _Pragma("GCC unroll 8") for (int v = 0; v < 8; v++)
                                acc[v] = _mm512_add_ps(acc[v], _mm512_mul_ps(wv, _mm512_loadu_ps(vj + v * 16)));
                        }
                        float *out = b->attention + (size_t)t * ROWS + head * 128;
                        _Pragma("GCC unroll 8") for (int v = 0; v < 8; v++) _mm512_storeu_ps(out + v * 16, acc[v]);
                    }
            }
            lap(4, "MLA attention over the cache");
            rep_buf("MLA attention output", b->attention, w, ROWS);

            bproject(&m6, b->normalized, INW, b->gate, ROWS, w);
            lap(5, "gate projection");
#pragma omp parallel for schedule(static)
            for (unsigned t = 0; t < w; t++) {
                const float *attention = b->attention + (size_t)t * ROWS; float *gate = b->gate + (size_t)t * ROWS;
                for (unsigned c = 0; c < ROWS; c++) gate[c] = attention[c] * transformer_sigmoid(gate[c]);
            }
            lap(6, "gating");
        }
#else
        for (int c = 0; c < 3; c++) bproject(&mq[c], b->normalized, INW, b->raw[c], ROWS, w);
        lap(1, "Q,K,V projections");
        rep_buf("raw q (before conv)", b->raw[0], w, ROWS); rep_buf("raw k (before conv)", b->raw[1], w, ROWS); rep_buf("raw v (before conv)", b->raw[2], w, ROWS);
#pragma omp parallel for schedule(static)
        for (unsigned blk = 0; blk < ROWS / 16; blk++)
            for (int c = 0; c < 3; c++)
                for (unsigned t = 0; t < w; t++)
                    for (unsigned rr = 0; rr < 16; rr++) {
                        const unsigned row = blk * 16 + rr;
                        const unsigned char *record = tr->qkv + TRANSFORMER_QKV_PREFIX + ((size_t)row * 3 + c) * TRANSFORMER_QKV_COMPONENT;
                        float convolved = transformer_convolve(b->raw[c][(size_t)t * ROWS + row], record + 4, hist[c][row]);
                        if (g_checkpoint_hist && t + 1 < w) memcpy(g_checkpoint_hist[t][c][row], hist[c][row], sizeof hist[c][row]);
                        b->qkv[c][(size_t)t * ROWS + row] = convolved * transformer_sigmoid(convolved);
                    }
#pragma omp parallel for schedule(static)
        for (unsigned t = 0; t < w; t++) {
            transformer_l2_heads(b->qkv[0] + (size_t)t * ROWS);
            transformer_l2_heads(b->qkv[1] + (size_t)t * ROWS);
        }
        lap(2, "conv + silu + l2");
        rep_buf("q (conv, silu, l2)", b->qkv[0], w, ROWS); rep_buf("k (conv, silu, l2)", b->qkv[1], w, ROWS); rep_buf("v (conv, silu)", b->qkv[2], w, ROWS);

        bproject(&m11, b->normalized, INW, b->beta, NHEAD, w);
        bproject(&m12, b->normalized, INW, b->dhid, HDIM, w);
        bproject(&m13, b->dhid, HDIM, b->decay, ROWS, w);
#pragma omp parallel for schedule(static)
        for (unsigned t = 0; t < w; t++) {
            float *beta = b->beta + (size_t)t * NHEAD, *decay = b->decay + (size_t)t * ROWS;
            for (unsigned head = 0; head < NHEAD; head++) {
                beta[head] = transformer_sigmoid(beta[head]);
                float exponent = transformer_f32(tr->records[39].data + head * 4);
                for (unsigned coordinate = 0; coordinate < HDIM; coordinate++) {
                    unsigned index = head * HDIM + coordinate;
                    float value = exponent * (decay[index] + transformer_f32(tr->records[18].data + index * 4));
                    decay[index] = expf(-5.0f * transformer_sigmoid(value));
                }
            }
        }
        lap(3, "beta, decay");
        rep_buf("beta", b->beta, w, NHEAD); rep_buf("decay down", b->dhid, w, HDIM); rep_buf("decay", b->decay, w, ROWS);

        const float query_scale = 1.0f / sqrtf(128.0f);
#pragma omp parallel for schedule(static)
        for (unsigned head = 0; head < NHEAD; head++)
            for (unsigned t = 0; t < w; t++) {
                float prediction[HDIM] = {0};
                float *output = b->attention + (size_t)t * ROWS + head * HDIM;
                const float *query = b->qkv[0] + (size_t)t * ROWS + head * HDIM;
                const float *key = b->qkv[1] + (size_t)t * ROWS + head * HDIM;
                const float *value = b->qkv[2] + (size_t)t * ROWS + head * HDIM;
                const float *decay = b->decay + (size_t)t * ROWS + head * HDIM;
                const float beta = b->beta[(size_t)t * NHEAD + head];
#if defined(__AVX512F__)
                if (g_vecrec) {
                    recurrent_vec(rec[head], query, key, value, decay, beta, query_scale, output);
                    if (g_checkpoint_rec && t + 1 < w) memcpy(g_checkpoint_rec[t][head], rec[head], sizeof rec[head]);
                    continue;
                }
#endif
                for (unsigned row = 0; row < HDIM; row++)
                    for (unsigned column = 0; column < HDIM; column++) {
                        float state = decay[row] * rec[head][row][column];
                        rec[head][row][column] = state;
                        prediction[column] = prediction[column] + state * key[row];
                    }
                memset(output, 0, HDIM * sizeof *output);
                for (unsigned row = 0; row < HDIM; row++) {
                    float key_beta = key[row] * beta;
                    float scaled_query = query[row] * query_scale;
                    for (unsigned column = 0; column < HDIM; column++) {
                        float state = rec[head][row][column] + key_beta * (value[column] - prediction[column]);
                        rec[head][row][column] = state;
                        output[column] = output[column] + state * scaled_query;
                    }
                }
                if (g_checkpoint_rec && t + 1 < w) memcpy(g_checkpoint_rec[t][head], rec[head], sizeof rec[head]);
            }
        lap(4, "recurrent state update");
        rep_buf("KDA attention output", b->attention, w, ROWS);
        if (g_rep_active) rep_buf("KDA recurrent state (after)", &rec[0][0][0], NHEAD, (size_t)HDIM * HDIM);

        bproject(&m6, b->normalized, INW, b->gate, ROWS, w);
        lap(5, "gate projection");
#pragma omp parallel for schedule(static)
        for (unsigned t = 0; t < w; t++) {
            float *attention = b->attention + (size_t)t * ROWS, *gate = b->gate + (size_t)t * ROWS;
            for (unsigned head = 0; head < NHEAD; head++) {
                float *block = attention + head * HDIM;
                transformer_normalize(block, block, tr->records[19].data, HDIM);
            }
            for (unsigned c = 0; c < ROWS; c++) gate[c] = attention[c] * transformer_sigmoid(gate[c]);
        }
        lap(6, "head norm + gating");
#endif
        bproject(&m7, b->gate, ROWS, b->aout, INW, w);
        lap(7, "attention output projection");
        rep_buf("attention x gate", b->gate, w, ROWS); rep_buf("attention output (aout)", b->aout, w, INW);

#pragma omp parallel for schedule(static)
        for (unsigned t = 0; t < w; t++) {
            float aggregate[INW];
            float *residual = b->residual + (size_t)t * INW;
            /* At every twelfth layer the attention output replaces the residual instead of adding to it, and the layer
               input becomes one more snapshot, seen by this second fold (after the first fold, before this one). */
            if (LB_BOUNDARY) memcpy(residual, b->aout + (size_t)t * INW, INW * sizeof(float));
            else for (unsigned c = 0; c < INW; c++) residual[c] = in[(size_t)t * INW + c] + b->aout[(size_t)t * INW + c];
            const float *src[10];
            unsigned count = 0;
            for (unsigned s = 0; s < LB_K; s++) src[count++] = snp[s] + (size_t)t * INW;
            if (LB_BOUNDARY) src[count++] = in + (size_t)t * INW;
            src[count++] = residual;
            agg(tr, 38, src, count, aggregate);
            transformer_normalize(b->postnorm + (size_t)t * INW, aggregate, tr->records[5].data, INW);
        }
        lap(8, "residual, aggregate, norm (pre-MoE)");
        rep_buf("residual after attention", b->residual, w, INW); rep_buf("normalized (pre-MoE)", b->postnorm, w, INW);

        route_scores(tr, b->postnorm, w, b->scores, b->choices);
        lap(9, "router scoring");
        if (g_rep_active) {   /* router: first token's vector against the 896 weight rows (floats) */
            const unsigned char *rw = tr->records[29].data;
            unsigned *sorted = malloc((size_t)896 * INW * 4);
            hs_reset();
            for (unsigned e = 0; e < 896; e++)
                for (unsigned c = 0; c < INW; c++) {
                    unsigned wb; memcpy(&wb, rw + ((size_t)e * INW + c) * 4, 4); sorted[(size_t)e * INW + c] = wb;
                    hs_add(pair_key(b->postnorm[c], wb));
                }
            qsort(sorted, (size_t)896 * INW, 4, cmp_u32);
            unsigned distinct = 1; for (size_t i = 1; i < (size_t)896 * INW; i++) distinct += sorted[i] != sorted[i - 1];
            printf("REP weights %-34s %u distinct float values among %u weights\n", "router 896x7168", distinct, 896 * INW);
            rep_pairs_print("router 896x7168", (unsigned long)896 * INW);
            free(sorted);
            rep_buf("router scores", b->scores, w, 896);
        }
#pragma omp parallel for schedule(static)
        for (unsigned t = 0; t < w; t++)
            if (!route_select(b->scores + (size_t)t * 896, b->choices + (size_t)t * 896, b->sel[t], b->wts[t])) {
                fprintf(stderr, "non-finite routing choice\n"); exit(1);
            }
        lap(10, "top-16 selection");
        if (g_pred && p == 0) { pred_report(tr, in, snp, w, b); lap_last = clk(); }
        if (g_route_log && w == 1) { route_log(tr, in, snp, b); lap_last = clk(); }

        bproject(&m31, b->postnorm, INW, b->latent, LATW, w);
        lap(11, "latent projection");
        rep_buf("latent (to experts)", b->latent, w, LATW);
        rep_dups("latent (expert input)", b->latent, w, LATW); rep_dups("normalized (pre-MoE)", b->postnorm, w, INW);
        rep_growth("latent (expert input)", b->latent, w, LATW); rep_growth("normalized (pre-MoE)", b->postnorm, w, INW);
#pragma omp parallel for schedule(static)
        for (unsigned t = 0; t < w; t++)
            for (unsigned c = 0; c < LATW; c++) b->lat[t][c] = (double)b->latent[(size_t)t * LATW + c];
        rep_buf_d("latent as double", &b->lat[0][0], w, LATW);
        run_experts(root, w, b->sel, (const double (*)[LATW])b->lat, b->down);
        lap(12, "experts (grouped)");
        if (g_rep_active) {
            rep_buf("expert raw gate (captured)", g_cap[0][0], g_cap_n, EXPW); rep_buf("expert raw up (captured)", g_cap[1][0], g_cap_n, EXPW);
            rep_buf("expert activated (captured)", g_cap[2][0], g_cap_n, EXPW);
            rep_buf("expert outputs (down)", &b->down[0][0][0], (size_t)w * 16, LATW);
        }
        if (g_lin && g_disk != 3 && p == 0) { lin_report(root, b, w); lap_last = clk(); }

#pragma omp parallel for schedule(static)
        for (unsigned t = 0; t < w; t++) {
            float *mixture = b->mixture + (size_t)t * LATW;
            memset(mixture, 0, LATW * sizeof(float));
            for (unsigned r = 0; r < 16; r++)
                for (unsigned c = 0; c < LATW; c++) mixture[c] = mixture[c] + b->wts[t][r] * b->down[t][r][c];
            transformer_normalize(b->lnorm + (size_t)t * LATW, mixture, tr->records[33].data, LATW);
        }
        lap(13, "mix + norm");
        rep_buf("expert mixture", b->mixture, w, LATW); rep_buf("mixture normalized", b->lnorm, w, LATW);
        bproject(&m32, b->lnorm, LATW, b->routed, INW, w);
        lap(14, "latent up projection");
        rep_buf("routed output", b->routed, w, INW);

        bproject(&m34, b->postnorm, INW, b->sgate, SHRW, w);
        bproject(&m35, b->postnorm, INW, b->sup, SHRW, w);
        lap(15, "shared gate, up");
        rep_buf("shared gate (raw)", b->sgate, w, SHRW); rep_buf("shared up (raw)", b->sup, w, SHRW);
#pragma omp parallel for schedule(static)
        for (unsigned t = 0; t < w; t++) transformer_activation(b->sgate + (size_t)t * SHRW, b->sup + (size_t)t * SHRW, SHRW);
        lap(16, "shared activation");
        rep_buf("shared activated", b->sgate, w, SHRW);
        bproject(&m36, b->sgate, SHRW, b->sout, INW, w);
        lap(17, "shared down");
        rep_buf("shared output", b->sout, w, INW);

#pragma omp parallel for schedule(static)
        for (unsigned t = 0; t < w; t++) {
            float *residual = b->residual + (size_t)t * INW;
            for (unsigned c = 0; c < INW; c++)
                residual[c] = residual[c] + (b->routed[(size_t)t * INW + c] + b->sout[(size_t)t * INW + c]);
            memcpy(final_out + (size_t)(p + t) * INW, residual, INW * sizeof(float));
        }
        lap(18, "final add");
        rep_buf("layer output", final_out + (size_t)p * INW, w, INW);
        if (g_rep_active) { g_rep_active = 0; g_rep_done = 1; }

        if (p == 0 && pod) {
            printf("  stage comparison against the pod:\n");
            int bad = 0;
            bad |= cmp_stage("normalized", b->normalized, pod->normalized, INW, INW, 4);
#if LB_MLA
            if (!bad) bad |= cmp_stage("mla query", b->qq, pod->mq, MLA_QROWS, MLA_QROWS, 4);
            if (!bad) bad |= cmp_stage("mla expanded kv", b->exq, pod->mex, MLA_EXROWS, MLA_EXROWS, 4);
#else
            for (int c = 0; c < 3 && !bad; c++) bad |= cmp_stage(c == 0 ? "q" : c == 1 ? "k" : "v", b->qkv[c], pod->qkv[c], ROWS, ROWS, 4);
            if (!bad) bad |= cmp_stage("beta", b->beta, pod->beta, NHEAD, NHEAD, 4);
            if (!bad) bad |= cmp_stage("decay", b->decay, pod->decay, ROWS, ROWS, 4);
#endif
            if (!bad) bad |= cmp_stage("attention", b->attention, pod->attention, ROWS, ROWS, 4);
            if (!bad) bad |= cmp_stage("gate", b->gate, pod->gate, ROWS, ROWS, 4);
            if (!bad) bad |= cmp_stage("postnorm", b->postnorm, pod->postnorm, INW, INW, 4);
            if (!bad) bad |= cmp_stage("routes", b->sel, pod->sel, 16, 16, 4);
            if (!bad) bad |= cmp_stage("route weights", b->wts, pod->wts, 16, 16, 4);
            if (!bad) bad |= cmp_stage("latent", b->latent, pod->latent, LATW, LATW, 4);
            if (!bad) bad |= cmp_stage("mixture", b->mixture, pod->mixture, LATW, LATW, 4);
            if (!bad) bad |= cmp_stage("routed", b->routed, pod->routed, INW, INW, 4);
            if (!bad) bad |= cmp_stage("shared", b->sout, pod->sout, INW, INW, 4);
        }
    }
}

static void report_tables(unsigned N)
{
    static const struct { int lap; double macs; } macs[] = {
        {1, 264241152.0}, {5, 88080384.0}, {7, 88080384.0}, {9, 6422528.0}, {11, 25690112.0},
        {12, 528482304.0}, {14, 25690112.0}, {15, 88080384.0}, {17, 44040192.0} };
    double total = 0;
    for (int k = 0; k < LAPS; k++) total += lap_time[k];
    printf("  where the time goes (per token): time, share of the layer, arithmetic done, rate\n");
    for (int k = 0; k < LAPS; k++) {
        if (!lap_name[k] || lap_time[k] <= 0) continue;
        const double ms = 1000 * lap_time[k] / N;
        double m = 0;
        for (unsigned i = 0; i < sizeof macs / sizeof *macs; i++) if (macs[i].lap == k) m = macs[i].macs;
        if (LB_MLA && k == 1) m = 0;   /* lap 1 is the layer-1 Q,K,V figure; the MLA projections are listed one by one below */
        if (m) printf("    %-36s %7.3f ms %5.1f%%  %7.1f MMAC  %5.0f GMAC/s\n", lap_name[k], ms, 100 * lap_time[k] / total, m / 1e6, m / (ms / 1000) / 1e9);
        else printf("    %-36s %7.3f ms %5.1f%%\n", lap_name[k], ms, 100 * lap_time[k] / total);
    }
    printf("    %-36s %7.3f ms\n", "sum of stages", 1000 * total / N);
    printf("  trunk matrices one by one:\n");
    for (unsigned i = 0; i < mat_stat_n; i++) {
        const double ms = 1000 * mat_stats[i].sec / N, m = (double)mat_stats[i].rows * mat_stats[i].cols;
        printf("    %-14s %5u x %-5u %7.3f ms  %6.1f MMAC  %5.0f GMAC/s  %6.1f MB of weights\n", mat_stats[i].name,
            mat_stats[i].rows, mat_stats[i].cols, ms, m / 1e6, m / (ms / 1000) / 1e9, m / 1e6);
    }
#ifdef DETAIL
    {
        double first[3] = {0}, second[3] = {0}, act = 0, busy_max = 0, busy_sum = 0;
        unsigned long passes = 0, pairs = 0;
        for (int th = 0; th < THREADS; th++) {
            for (int m = 0; m < 3; m++) { first[m] += d_first[th][m]; second[m] += d_second[th][m]; }
            act += d_act[th]; busy_sum += d_busy[th]; if (d_busy[th] > busy_max) busy_max = d_busy[th];
            passes += d_passes[th]; pairs += d_pairs[th];
        }
        const double repeats = (second[0] + second[1] + second[2]) / THREADS;
        const double wall = d_region - repeats;
        const char *mn[3] = {"gate matrix", "up matrix", "down matrix"};
        printf("  experts stage in detail (diagnostic build: each projection is repeated once at once, to compare a cold read with a warm one)\n");
        printf("    wall with the repeats removed (estimate)  %7.3f ms/token\n", 1000 * wall / N);
        printf("    thread utilisation (busy / wall x %d)      %5.1f%%   busiest thread %.3f s, mean %.3f s\n", THREADS,
            100 * busy_sum / (THREADS * wall), busy_max, busy_sum / THREADS);
        for (int m = 0; m < 3; m++)
            printf("    %-12s as run %7.3f ms/token   repeated at once %7.3f ms/token  (%.2fx)\n", mn[m],
                1000 * first[m] / THREADS / N, 1000 * second[m] / THREADS / N, first[m] / second[m]);
        printf("    activation + convert to double       %7.3f ms/token\n", 1000 * act / THREADS / N);
        printf("    %lu weight passes (%.2f tokens each): %.1f MB of expert weights read per token = %.1f GB/s\n", passes,
            (double)pairs / passes, (double)passes * ROOT_EXPERT_RAW / N / 1e6,
            (double)passes * ROOT_EXPERT_RAW / wall / 1e9);
    }
#endif
}

/* Writes one expert's gate, up and down as row-major float32 (rows x width) for offline analysis. */
static void dump_expert(const Root *root, unsigned expert, const char *dir)
{
    static const char *const names[3] = { "gate", "up", "down" };
    const unsigned char *maps = root->constants + root->palette_count * 2;
    const unsigned char *offsets = maps + root->map_count * 16;
    const unsigned char *refs = offsets + (root->template_count + 1) * 2;
    for (unsigned matrix = 0; matrix < 3; matrix++) {
        const unsigned width = matrix == 2 ? 3072 : 3584, rows = matrix == 2 ? 3584 : 3072;
        const unsigned begin = root_u16(offsets + root_u16(root->index + (expert * 3 + matrix) * 2) * 2);
        float *w = grab((size_t)rows * width * 4);
        for (unsigned block = 0; block < rows / 64; block++) {
            const unsigned char *data = expert_base(root, expert) +
                (matrix < 2 ? (size_t)(matrix * 48 + block) * ROOT_RAW : (size_t)96 * ROOT_RAW + (size_t)block * ROOT_DOWN_RAW);
            for (unsigned row = 0; row < 64; row++) {
                const unsigned char *codes = data + row * width / 2, *selectors = data + 64 * width / 2 + row * width / 32;
                for (unsigned c = 0; c < width; c++) {
                    const float (*pair)[2] = root->pairs[root_u16(refs + (begin + selectors[c / 32]) * 2)];
                    w[(size_t)(block * 64 + row) * width + c] = pair[codes[c / 2]][c & 1];
                }
            }
        }
        char path[4096];
        snprintf(path, sizeof path, "%s/exp-%u-%s.f32", dir, expert, names[matrix]);
        FILE *out = fopen(path, "wb");
        if (!out || fwrite(w, 4, (size_t)rows * width, out) != (size_t)rows * width) bail("dump write");
        fclose(out); free(w);
        printf("wrote %s (%u x %u)\n", path, rows, width);
    }
}

static long anon_huge_kb(void)
{
    FILE *f = fopen("/proc/self/smaps_rollup", "r"); char line[256]; long kb = -1;
    while (f && fgets(line, sizeof line, f)) if (sscanf(line, "AnonHugePages: %ld kB", &kb) == 1) break;
    if (f) fclose(f);
    return kb;
}

int main(int argc, char **argv)
{
#ifdef LB_GENERIC
    if (argc != 8) { fputs("usage: layer-batch DATASET_DIR RESID_F32 SNAPS_F32 NTOKENS WINDOW[,WINDOW...] LAYER K\n", stderr); return 2; }
    lb_layer_no = (unsigned)atoi(argv[6]); lb_k_in = (unsigned)atoi(argv[7]);
    if (lb_layer_no < 2 || lb_layer_no > 92 || lb_k_in < 1 || lb_k_in > 8) bail("LAYER must be 2..92 and K 1..8");
#else
    if (argc != 6) { fputs("usage: layer-batch DATASET_DIR RESID_F32 S0_F32 NTOKENS WINDOW[,WINDOW...]\n", stderr); return 2; }
#endif
    static const char *const forbidden[] = {"CLOVER_FOLD", "CLOVER_FOLD_LAYERS", "CLOVER_DROP_SLOTS", "CLOVER_TOPK", "CLOVER_CONST_ATTN"};
    for (unsigned i = 0; i < sizeof forbidden / sizeof *forbidden; i++)
        if (getenv(forbidden[i])) bail("an experiment variable is set; the pod would not match this harness");
    const unsigned N = (unsigned)atoi(argv[4]);
    const unsigned K = LB_K;
    float *resid = grab((size_t)N * INW * 4);
    float *snaps[9];
    for (unsigned s = 0; s < K + 1; s++) snaps[s] = grab((size_t)N * INW * 4); /* slot K is only used as the boundary layer's new snapshot */
    FILE *f = fopen(argv[2], "rb"); if (!f || fread(resid, 4, (size_t)N * INW, f) != (size_t)N * INW) bail("resid file"); fclose(f);
    f = fopen(argv[3], "rb");
    for (unsigned s = 0; s < K; s++) if (!f || fread(snaps[s], 4, (size_t)N * INW, f) != (size_t)N * INW) bail("snapshot file");
    fclose(f);

    Transformer *tr = NULL;
    if (!transformer_open(argv[1], &tr)) bail("transformer_open failed");
    g_palette = tr->palette; g_ident = palette_identity(tr->palette);
    if (getenv("LB_WIDE")) g_wide = atoi(getenv("LB_WIDE"));
    if (getenv("LB_VEC")) g_vecrec = atoi(getenv("LB_VEC"));
    if (getenv("LB_ROUTE")) g_wideroute = atoi(getenv("LB_ROUTE"));
    if (getenv("LB_EXPK")) g_expk = atoi(getenv("LB_EXPK"));
    if (getenv("LB_PRED")) g_pred = atoi(getenv("LB_PRED"));
    if (getenv("LB_LIN")) g_lin = atoi(getenv("LB_LIN"));
    printf("palette is %s, %d threads\n", g_ident ? "the int8 identity (decode path)" : "not identity (gather path)", THREADS);
    const Root *root = tr->root;
    if (!root->direct) bail("no direct expert mapping");
    if (getenv("LB_REPEAT") && *getenv("LB_REPEAT")) {
        g_rep = 1;
        /* weights at rest: distinct float values over the first four experts' gate, up and down matrices together */
        const unsigned n_exp = 4;
        size_t total = 0; unsigned *sorted = malloc((size_t)n_exp * 3 * 3584 * 3072 * 4);
        for (unsigned e = 0; e < n_exp; e++) for (unsigned m = 0; m < 3; m++) {
            unsigned rows, width; float *w = rep_decode(root, e, m, &rows, &width);
            memcpy(sorted + total, w, (size_t)rows * width * 4); total += (size_t)rows * width; free(w);
        }
        qsort(sorted, total, 4, cmp_u32);
        unsigned distinct = 1; for (size_t i = 1; i < total; i++) distinct += sorted[i] != sorted[i - 1];
        printf("REP weights experts 0-%u, all three matrices together: %u distinct float values among %zu weights\n", n_exp - 1, distinct, total);
        free(sorted);
        /* identical weight segments: 16 columns = 8 code bytes under one group table; counted per column block over the rows of the matrix */
        {
            size_t segs = 0, dups = 0;
            for (unsigned e = 0; e < n_exp; e++) for (unsigned m = 0; m < 3; m++) {
                const unsigned width = m == 2 ? 3072 : 3584, rows = m == 2 ? 3584 : 3072;
                const unsigned char *maps = root->constants + root->palette_count * 2;
                const unsigned char *offsets = maps + root->map_count * 16;
                const unsigned char *refs = offsets + (root->template_count + 1) * 2;
                const unsigned begin = root_u16(offsets + root_u16(root->index + (e * 3 + m) * 2) * 2);
                unsigned long long *keys = malloc((size_t)(width / 16) * rows * sizeof *keys);
                for (unsigned block = 0; block < rows / 64; block++) {
                    const unsigned char *data = expert_base(root, e) +
                        (m < 2 ? (size_t)(m * 48 + block) * ROOT_RAW : (size_t)96 * ROOT_RAW + (size_t)block * ROOT_DOWN_RAW);
                    for (unsigned row = 0; row < 64; row++) {
                        const unsigned char *codes = data + row * width / 2, *selectors = data + 64 * width / 2 + row * width / 32;
                        for (unsigned cb = 0; cb < width / 16; cb++) {
                            unsigned char buf[10];
                            const unsigned ref = root_u16(refs + (begin + selectors[cb / 2]) * 2);
                            buf[0] = (unsigned char)ref; buf[1] = (unsigned char)(ref >> 8); memcpy(buf + 2, codes + cb * 8, 8);
                            keys[(size_t)cb * rows + block * 64 + row] = fnv64(buf, 10);
                        }
                    }
                }
                for (unsigned cb = 0; cb < width / 16; cb++) dups += count_dups(keys + (size_t)cb * rows, rows);
                segs += (size_t)(width / 16) * rows;
                free(keys);
            }
            printf("REP weight segments, experts 0-%u: %zu of %zu 16-column segments (table + 8 code bytes) are identical to another row's segment at the same columns (%.3f%%)\n",
                n_exp - 1, dups, segs, 100.0 * (double)dups / (double)segs);
        }
    }
    if (getenv("LB_DUMP") && *getenv("LB_DUMP")) {
        char list[256]; snprintf(list, sizeof list, "%s", getenv("LB_DUMP"));
        for (char *tok = strtok(list, ","); tok; tok = strtok(NULL, ",")) dump_expert(root, (unsigned)atoi(tok), "/tmp/one-layer");
        return 0;
    }
    double mark = clk();
    if (mlock((void *)root->direct, root->direct_bytes)) { perror("mlock"); return 1; }
    printf("layer %u: %.2f GB of experts locked in %.2f s\n", LB_LAYER, (double)root->direct_bytes / 1e9, clk() - mark);

    /* The pod, one token per call: reference outputs and the first tokens' stage state. */
    TransformerSequence *seq = transformer_sequence_create(tr);
    if (!seq) bail("sequence failed");
    float *ref = grab((size_t)N * INW * 4);
    PodStages *pod = grab(sizeof *pod);
    mark = clk();
    unsigned long snapshot_mismatch = 0;
    for (unsigned t = 0; t < N; t++) {
#ifdef LB_GENERIC
        float snap_in[8 * INW], snap_out[9 * INW];
        for (unsigned s = 0; s < K; s++) memcpy(snap_in + (size_t)s * INW, snaps[s] + (size_t)t * INW, INW * sizeof(float));
        if (!transformer_process(tr, seq, resid + (size_t)t * INW, snap_in, K, ref + (size_t)t * INW, snap_out)) bail("pod process failed");
        /* What the pod hands on as snapshots: the ones it was given, and at a boundary layer also this layer's input. */
        if (memcmp(snap_out, snap_in, (size_t)K * INW * sizeof(float)) ||
            (LB_BOUNDARY && memcmp(snap_out + (size_t)K * INW, resid + (size_t)t * INW, INW * sizeof(float)))) snapshot_mismatch++;
#else
        if (!transformer_process(tr, seq, resid + (size_t)t * INW, snaps[0] + (size_t)t * INW, ref + (size_t)t * INW)) bail("pod process failed");
#endif
        if (t < CHECK) {
            memcpy(pod->normalized[t], seq->normalized, sizeof pod->normalized[t]);
            for (int c = 0; c < 3; c++) memcpy(pod->qkv[c][t], seq->qkv[c], sizeof pod->qkv[c][t]);
            memcpy(pod->decay[t], seq->decay, sizeof pod->decay[t]); memcpy(pod->beta[t], seq->beta, sizeof pod->beta[t]);
            memcpy(pod->attention[t], seq->attention, sizeof pod->attention[t]); memcpy(pod->gate[t], seq->gate, sizeof pod->gate[t]);
            memcpy(pod->postnorm[t], seq->postnorm, sizeof pod->postnorm[t]); memcpy(pod->latent[t], seq->latent, sizeof pod->latent[t]);
            memcpy(pod->mixture[t], seq->mixture, sizeof pod->mixture[t]); memcpy(pod->routed[t], seq->routed, sizeof pod->routed[t]);
            memcpy(pod->sout[t], seq->shared_output, sizeof pod->sout[t]);
            memcpy(pod->sel[t], seq->selected, sizeof pod->sel[t]); memcpy(pod->wts[t], seq->weights, sizeof pod->wts[t]);
#if LB_MLA
            memcpy(pod->mq[t], seq->mla_query, sizeof pod->mq[t]); memcpy(pod->mex[t], seq->mla_expanded, sizeof pod->mex[t]);
#endif
        }
    }
    const double pod_s = clk() - mark;
    printf("pod, one token per call: %.3f s for %u tokens = %.2f ms/token\n\n", pod_s, N, 1000 * pod_s / N);
#ifdef LB_GENERIC
    printf("pod snapshots handed on (%u input slots%s): %s\n", K, LB_BOUNDARY ? " + the layer input, boundary layer" : "",
        snapshot_mismatch ? "DIFFER from the expected" : "as expected");
#endif
    if (getenv("LB_SAVE_REF") && *getenv("LB_SAVE_REF")) {
        FILE *rf = fopen(getenv("LB_SAVE_REF"), "wb");
        if (!rf || fwrite(ref, 4, (size_t)N * INW, rf) != (size_t)N * INW) bail("cannot save the reference");
        fclose(rf);
    }
#ifdef LB_GENERIC
    if (getenv("LB_SAVE_SNAPS") && *getenv("LB_SAVE_SNAPS")) {   /* the snapshots this layer hands to the next, slot-major */
        FILE *rf = fopen(getenv("LB_SAVE_SNAPS"), "wb");
        if (!rf) bail("cannot save the snapshots");
        for (unsigned s = 0; s < K; s++) if (fwrite(snaps[s], 4, (size_t)N * INW, rf) != (size_t)N * INW) bail("cannot save the snapshots");
        if (LB_BOUNDARY && fwrite(resid, 4, (size_t)N * INW, rf) != (size_t)N * INW) bail("cannot save the snapshots");
        fclose(rf);
    }
    if (getenv("LB_POD_ONLY") && *getenv("LB_POD_ONLY")) return snapshot_mismatch ? 1 : 0;
#endif
    if (getenv("LB_DISK")) g_disk = atoi(getenv("LB_DISK"));
    if (g_disk) {
        char path[4096];
        snprintf(path, sizeof path, "%s/root-%u/experts.direct", argv[1], LB_LAYER);
        g_expert_fd = open(path, O_RDONLY);
        if (g_expert_fd < 0) bail("cannot open experts.direct");
        munlock((void *)root->direct, root->direct_bytes);
        if (g_disk == 3) {
            if (getenv("LB_IO")) g_io_n = atoi(getenv("LB_IO"));
            if (getenv("LB_SPEC")) g_spec = atoi(getenv("LB_SPEC"));
            if (getenv("LB_IOLO") && *getenv("LB_IOLO") && getenv("LB_IOHI") && *getenv("LB_IOHI")) { g_io_lo = atoi(getenv("LB_IOLO")); g_io_hi = atoi(getenv("LB_IOHI")); }
            if (g_io_n < 1 || g_io_n > MAX_IO || g_spec < 0 || g_spec > 896) bail("LB_IO must be 1..32 and LB_SPEC 0..896");
            g_direct_fd = open(path, O_RDONLY | O_DIRECT);
            if (g_direct_fd < 0) bail("cannot open experts.direct with O_DIRECT");
            mark = clk();
            const int thp = getenv("LB_THP") ? atoi(getenv("LB_THP")) : 1; /* env LB_THP=0 turns huge pages off */
            g_buf = mmap(NULL, root->direct_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | (thp ? 0 : MAP_POPULATE), -1, 0);
            if (g_buf == MAP_FAILED) bail("cannot allocate the in-memory copy of the experts");
            if (thp) { madvise(g_buf, root->direct_bytes, MADV_HUGEPAGE); memset(g_buf, 0, root->direct_bytes); }
            printf("expert buffer: %ld MB of it on huge pages%s\n", anon_huge_kb() / 1024, thp ? " (huge pages requested)" : "");
            printf("experts by direct reads: %d reader threads, %d MB chunks, speculative list %d experts; %.2f GB buffer ready in %.1f s\n\n",
                g_io_n, IO_CHUNK >> 20, g_spec, (double)root->direct_bytes / 1e9, clk() - mark);
        } else
        printf("experts from disk, mode %d (%s): the expert file is dropped from memory before every window\n\n", g_disk,
            g_disk == 2 ? "with a read-ahead request per used expert" : "demand paging only");
    }

    unsigned windows[16], nw = 0;
    { char *copy = strdup(argv[5]); for (char *part = strtok(copy, ","); part && nw < 16; part = strtok(NULL, ",")) windows[nw++] = (unsigned)atoi(part); free(copy); }
    unsigned wmax = 1; for (unsigned i = 0; i < nw; i++) if (windows[i] > wmax) wmax = windows[i];
    if (wmax > N) wmax = N;
    Buf b = buf_make(wmax);
#if LB_MLA
    mla_cache_make(N);
#endif
    float *out = grab((size_t)N * INW * 4);
    float (*hist)[ROWS][3] = grab(3 * (size_t)ROWS * 3 * sizeof(float));
    float (*rec)[HDIM][HDIM] = grab((size_t)NHEAD * HDIM * HDIM * sizeof(float));
    for (unsigned i = 0; i < nw; i++) {
        const unsigned W = windows[i] > N ? N : windows[i];
        double best = 1e30; int same = 0;
        double disk_gb = 0, mean_used = 0, read_per = 0, wasted_per = 0; unsigned long windows_run = 0;
        for (int round = 0; round < 2; round++) {
            memset(hist, 0, 3 * (size_t)ROWS * 3 * sizeof(float)); memset(rec, 0, (size_t)NHEAD * HDIM * HDIM * sizeof(float));
            memset(lap_time, 0, sizeof lap_time); memset(out, 0, (size_t)N * INW * 4);
            for (unsigned q = 0; q < mat_stat_n; q++) mat_stats[q].sec = 0;
            DT(memset(d_first, 0, sizeof d_first); memset(d_second, 0, sizeof d_second); memset(d_act, 0, sizeof d_act);
               memset(d_busy, 0, sizeof d_busy); memset(d_passes, 0, sizeof d_passes); memset(d_pairs, 0, sizeof d_pairs);
               d_region = 0;)
            if (round == 0) printf("window %u\n", W);
            g_evict_s = 0; g_distinct_sum = 0; g_window_count = 0; memset(g_hits, 0, sizeof g_hits);
            g_read_experts = 0; g_wasted_experts = 0; memset(g_wait_t, 0, sizeof g_wait_t);
            const unsigned long long rb0 = proc_read_bytes();
            mark = clk();
            run_layer(tr, root, resid, (const float *const *)snaps, N, W, &b, out, hist, rec, round == 0 ? pod : NULL);
            double spent = clk() - mark - g_evict_s;
            if (round == 1 || best > 1e29) { best = spent; disk_gb = (double)(proc_read_bytes() - rb0) / 1e9; mean_used = g_window_count ? (double)g_distinct_sum / g_window_count : 0; windows_run = g_window_count;
                read_per = g_window_count ? (double)g_read_experts / g_window_count : 0; wasted_per = g_window_count ? (double)g_wasted_experts / g_window_count : 0; }
            same = !memcmp(out, ref, (size_t)N * INW * 4);
            if (round == 0 && !same) {
                size_t t = 0; while (!memcmp(out + t * INW, ref + t * INW, INW * 4)) t++;
                printf("  output differs from the pod first at token %zu\n", t);
            }
            if (round == 1 && !same) {   /* how far the output is from the pod's (the 8-bit-activation experiment) */
                double worst = 0, mean = 0, min_cos = 1;
                for (unsigned t = 0; t < N; t++) {
                    double e = 0, r2 = 0, o2 = 0, dot = 0;
                    for (unsigned c = 0; c < INW; c++) { const double d = (double)out[(size_t)t * INW + c] - (double)ref[(size_t)t * INW + c], x = ref[(size_t)t * INW + c], y = out[(size_t)t * INW + c]; e += d * d; r2 += x * x; o2 += y * y; dot += x * y; }
                    const double rel = sqrt(e / (r2 + 1e-30)), cs = dot / (sqrt(r2 * o2) + 1e-30);
                    mean += rel; if (rel > worst) worst = rel; if (cs < min_cos) min_cos = cs;
                }
                printf("  distance of the output from the pod's, per token (relative RMS): mean %.3e, worst %.3e; lowest cosine %.9f\n", mean / N, worst, min_cos);
            }
        }
        printf("  layer %u, %u tokens, window %u: %.3f s = %.3f ms/token   (pod %.2f ms/token, %.2fx)   output vs pod: %s\n",
            LB_LAYER, N, W, best, 1000 * best / N, 1000 * pod_s / N, pod_s / best, same ? "bit-identical" : "DIFFERENT");
        printf("  experts used per window: %.1f of 896 on average over %lu windows (%.1f MB of expert weights per window); read from disk during the run: %.2f GB (%.1f%% of the layer's experts per window)\n",
            mean_used, windows_run, mean_used * ROOT_EXPERT_RAW / 1e6, disk_gb, 100 * disk_gb * 1e9 / (windows_run ? windows_run : 1) / root->direct_bytes);
        if (g_disk == 3) {
            double wsum = 0, wmax = 0;
            for (int i = 0; i < 64; i++) { wsum += g_wait_t[i]; if (g_wait_t[i] > wmax) wmax = g_wait_t[i]; }
            printf("  direct reads: %.1f experts in memory per window at the end, of which %.1f were read speculatively and not needed\n", read_per, wasted_per);
            printf("  compute threads waiting for expert data from disk: mean %.3f s per thread, busiest thread %.3f s, in a run of %.3f s\n", wsum / THREADS, wmax, best);
        }
        {   /* how skewed expert use is over these N tokens (one text; not a general property) */
            unsigned long sorted[896], sum = 0, run = 0;
            memcpy(sorted, g_hits, sizeof sorted);
            for (int i = 0; i < 896; i++) sum += g_hits[i];
            for (int i = 0; i < 896; i++) for (int j = i + 1; j < 896; j++) if (sorted[j] > sorted[i]) { unsigned long t = sorted[i]; sorted[i] = sorted[j]; sorted[j] = t; }
            printf("  expert use over these tokens: the most used 45 / 90 / 224 / 448 experts (5%% / 10%% / 25%% / 50%%) take");
            for (int i = 0; i < 896; i++) {
                run += sorted[i];
                if (i + 1 == 45 || i + 1 == 90 || i + 1 == 224 || i + 1 == 448) printf(" %.1f%%", 100.0 * run / (sum ? sum : 1));
            }
            printf(" of the selections; least used expert chosen %lu times, most used %lu times\n", sorted[895], sorted[0]);
        }
        report_tables(N);
        printf("\n");
        fflush(stdout);
    }
    return 0;
}

#ifdef PIPELINE_ENTRY
/* Entry points for the one-process pipeline (input-to-layer1/pipeline.c); build with -DPIPELINE_ENTRY -Dmain=layer_batch_main.
   Experts come from disk by direct reads (LB_DISK=3), as in the standalone harness. */
static Transformer *p_tr;
static const Root *p_root;
static Buf p_b;
static float (*p_hist)[ROWS][3];
static float (*p_rec)[HDIM][HDIM];
static unsigned char *p_external_buffer;   /* set with stage3_use_buffer before stage3_setup to share one expert buffer */
void stage3_use_buffer(unsigned char *buffer) { p_external_buffer = buffer; }
void stage3_route_log(int on) { g_route_log = on; }
void stage3_routes(unsigned short *out) { memcpy(out, g_route_rec, sizeof g_route_rec); }
void stage3_hits(unsigned long *out) { memcpy(out, g_hits, sizeof g_hits); }
static ExpertCache *p_cache;
static PackedTrunk *p_packed_trunk;   /* layer 1's 4-bit copy, used by decode steps only */
static float (*p_checkpoint_rec)[NHEAD][HDIM][HDIM];   /* layer 1's state after each token of a draft pass, see g_checkpoint_rec */
static float (*p_checkpoint_hist)[3][ROWS][3];
static unsigned p_checkpoint_count;
void stage3_set_slots(unsigned char *const *slots, unsigned cap) { p_cache = cap ? cache_make(slots, cap) : NULL; }
void stage3_cache_stats(unsigned long *hit, unsigned long *miss) { *hit = g_cache_hits; *miss = g_cache_misses; }
void stage3_guess_stats(unsigned long *issued, unsigned long *used) { *issued = g_guess_issued; *used = g_guess_used; }
void stage3_laps(double *out) { memcpy(out, lap_time, LAPS * sizeof(double)); }
double stage3_setup(const char *dataset, unsigned max_n)
{
    const double t0 = clk();
    if (!transformer_open(dataset, &p_tr)) bail("transformer_open failed");
    g_palette = p_tr->palette; g_ident = palette_identity(p_tr->palette);
    if (getenv("LB_WIDE")) g_wide = atoi(getenv("LB_WIDE"));
    if (getenv("LB_VEC")) g_vecrec = atoi(getenv("LB_VEC"));
    if (getenv("LB_ROUTE")) g_wideroute = atoi(getenv("LB_ROUTE"));
    if (getenv("LB_EXPK")) g_expk = atoi(getenv("LB_EXPK"));
    p_root = p_tr->root;
    if (!p_root->direct) bail("no direct expert mapping");
    g_disk = 3;
    char path[4096];
    snprintf(path, sizeof path, "%s/root-%u/experts.direct", dataset, LB_LAYER);
    g_expert_fd = open(path, O_RDONLY);
    if (g_expert_fd < 0) bail("cannot open experts.direct");
    if (getenv("LB_IO")) g_io_n = atoi(getenv("LB_IO"));
    if (getenv("LB_SPEC")) g_spec = atoi(getenv("LB_SPEC"));
    if (getenv("LB_GUESS")) g_guess_max = atoi(getenv("LB_GUESS"));
    g_direct_fd = open(path, O_RDONLY | O_DIRECT);
    if (g_direct_fd < 0) bail("cannot open experts.direct with O_DIRECT");
    const int thp = getenv("LB_THP") ? atoi(getenv("LB_THP")) : 1;
    if (p_external_buffer) g_buf = p_external_buffer;
    else {
        g_buf = mmap(NULL, p_root->direct_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | (thp ? 0 : MAP_POPULATE), -1, 0);
        if (g_buf == MAP_FAILED) bail("cannot allocate the in-memory copy of the experts");
        if (thp) { madvise(g_buf, p_root->direct_bytes, MADV_HUGEPAGE); memset(g_buf, 0, p_root->direct_bytes); }
    }
    p_b = buf_make(max_n);
    p_hist = grab(3 * (size_t)ROWS * 3 * sizeof(float));
    p_rec = grab((size_t)NHEAD * HDIM * HDIM * sizeof(float));
    if (use_4bit_trunk_requested()) p_packed_trunk = pack_layer_trunk_to_4bit(p_tr);
    return clk() - t0;
}

void stage3_run(const float *resid, const float *s0, unsigned n, unsigned window, float *out)
{
    memset(p_hist, 0, 3 * (size_t)ROWS * 3 * sizeof(float));
    memset(p_rec, 0, (size_t)NHEAD * HDIM * HDIM * sizeof(float));
    g_evict_s = 0; g_distinct_sum = 0; g_window_count = 0; memset(g_hits, 0, sizeof g_hits);
    g_read_experts = 0; g_wasted_experts = 0; memset(g_wait_t, 0, sizeof g_wait_t);
    const float *snapshots[1] = { s0 };
    g_cache = NULL;
    if (p_cache) cache_clear(p_cache);
    g_active_packed_trunk = NULL;
    run_layer(p_tr, p_root, resid, snapshots, n, window, &p_b, out, p_hist, p_rec, NULL);
}

/* n more tokens after earlier ones as one prompt-style pass (a chat turn): the state stays as the earlier tokens left it. */
void stage3_run_from(const float *resid, const float *s0, unsigned n, unsigned window, float *out)
{
    g_evict_s = 0; g_distinct_sum = 0; g_window_count = 0; memset(g_hits, 0, sizeof g_hits);
    g_read_experts = 0; g_wasted_experts = 0; memset(g_wait_t, 0, sizeof g_wait_t);
    const float *snapshots[1] = { s0 };
    g_cache = NULL;
    if (p_cache) cache_clear(p_cache);
    g_active_packed_trunk = NULL;
    run_layer(p_tr, p_root, resid, snapshots, n, window, &p_b, out, p_hist, p_rec, NULL);
}

/* One more token after the prompt: the state stays as the prompt left it. */
void stage3_step(const float *resid, const float *s0, float *out)
{
    g_evict_s = 0; g_distinct_sum = 0; g_window_count = 0; memset(g_hits, 0, sizeof g_hits);
    g_read_experts = 0; g_wasted_experts = 0; memset(g_wait_t, 0, sizeof g_wait_t);
    const float *snapshots[1] = { s0 };
    g_cache = p_cache;
    const int kernel = g_expk;
    g_expk = 0;   /* see the layer steps above */
    g_active_packed_trunk = p_packed_trunk;
    run_layer(p_tr, p_root, resid, snapshots, 1, 1, &p_b, out, p_hist, p_rec, NULL);
    g_cache = NULL;
    g_expk = kernel;
}

/* n tokens after the prompt in one pass (verifying a draft): the state stays as the prompt left it, experts come from the cache. */
void stage3_stepk(const float *resid, const float *s0, unsigned n, float *out)
{
    g_evict_s = 0; g_distinct_sum = 0; g_window_count = 0; memset(g_hits, 0, sizeof g_hits);
    g_read_experts = 0; g_wasted_experts = 0; memset(g_wait_t, 0, sizeof g_wait_t);
    const float *snapshots[1] = { s0 };
    if (getenv("LB_SWEEP_EXPK")) g_sweep_expk = atoi(getenv("LB_SWEEP_EXPK"));
    g_cache = p_cache;
    const int kernel = g_expk;
    g_expk = g_sweep_expk;
    g_active_packed_trunk = p_packed_trunk;
    if (n - 1 <= p_checkpoint_count) { g_checkpoint_rec = p_checkpoint_rec; g_checkpoint_hist = p_checkpoint_hist; }
    run_layer(p_tr, p_root, resid, snapshots, n, n, &p_b, out, p_hist, p_rec, NULL);
    g_checkpoint_rec = NULL; g_checkpoint_hist = NULL;
    g_cache = NULL;
    g_expk = kernel;
}
void stage3_set_scratch(unsigned char *const *slots, unsigned n)
{
    if (n > SCRATCH_MAX) bail("scratch pool too large");
    for (unsigned i = 0; i < n; i++) g_scratch[i] = slots[i];
    g_scratch_n = n;
}
/* Layer 1's state after each token of a draft pass (see g_checkpoint_rec): kept so a rejected draft can be undone. */
void stage3_keep_positions(unsigned count)
{
    p_checkpoint_rec = grab(count * sizeof *p_checkpoint_rec); p_checkpoint_hist = grab(count * sizeof *p_checkpoint_hist); p_checkpoint_count = count;
}
/* The state as it was after token `position` of the last stepk pass. */
void stage3_use_position(unsigned position)
{
    memcpy(p_rec, p_checkpoint_rec[position], sizeof *p_checkpoint_rec);
    memcpy(p_hist, p_checkpoint_hist[position], sizeof *p_checkpoint_hist);
}
#endif

#ifdef LB_ENTRY
/* Entry points for the whole-model process (full-model/pipeline-full.c). One object per attention kind (-DLB_PFX=lbk for
   the KDA layers, -DLB_PFX=lbm for the MLA layers; -Dmain=... so the standalone main is unused), each able to hold any
   number of opened layers. The 15.7 GB in-memory copy of the experts is shared by all layers and handed in once.
   A layer handle owns what one sequence needs of that layer: the recurrent state and convolution history (KDA) or the
   key/value cache (MLA). Experts come from disk by direct reads (LB_DISK=3), as in the standalone harness. */
#define LB_CAT2(a, b) a##b
#define LB_CAT(a, b) LB_CAT2(a, b)
typedef struct {
    unsigned layer, k;
    Transformer *tr;
    int plain_fd, direct_fd, ident;
    float (*hist)[ROWS][3];
    float (*rec)[HDIM][HDIM];
    MlaCache mla;
    ExpertCache *cache;
    PackedTrunk *packed_trunk;   /* NULL unless USE_4BIT_TRUNK=1 */
    float (*checkpoint_rec)[NHEAD][HDIM][HDIM];   /* KDA only, see g_checkpoint_rec */
    float (*checkpoint_hist)[3][ROWS][3];
    unsigned checkpoint_count;
} LbLayer;
static Buf e_b;

static void touch_range(const void *start, size_t bytes)
{
    volatile unsigned char sink = 0;
    for (size_t at = 0; at < bytes; at += 4096) sink ^= ((const unsigned char *)start)[at];
    (void)sink;
}

void LB_CAT(LB_PFX, _set_buffer)(unsigned char *buffer) { g_buf = buffer; }
void LB_CAT(LB_PFX, _route_log)(int on) { g_route_log = on; }
void LB_CAT(LB_PFX, _routes)(unsigned short *out) { memcpy(out, g_route_rec, sizeof g_route_rec); }
void LB_CAT(LB_PFX, _hits)(unsigned long *out) { memcpy(out, g_hits, sizeof g_hits); }
/* Gives the layer `cap` expert slots (decode steps keep recently used experts there); cap 0 leaves the cache off. */
void LB_CAT(LB_PFX, _set_slots)(void *handle, unsigned char *const *slots, unsigned cap)
{
    LbLayer *h = handle;
    h->cache = cap ? cache_make(slots, cap) : NULL;
}
void LB_CAT(LB_PFX, _cache_stats)(unsigned long *hit, unsigned long *miss) { *hit = g_cache_hits; *miss = g_cache_misses; }
void LB_CAT(LB_PFX, _guess_stats)(unsigned long *issued, unsigned long *used) { *issued = g_guess_issued; *used = g_guess_used; }
void LB_CAT(LB_PFX, _laps)(double *out) { memcpy(out, lap_time, LAPS * sizeof(double)); }

double LB_CAT(LB_PFX, _init)(unsigned max_n)
{
    const double t0 = clk();
    if (getenv("LB_WIDE")) g_wide = atoi(getenv("LB_WIDE"));
    if (getenv("LB_VEC")) g_vecrec = atoi(getenv("LB_VEC"));
    if (getenv("LB_ROUTE")) g_wideroute = atoi(getenv("LB_ROUTE"));
    if (getenv("LB_EXPK")) g_expk = atoi(getenv("LB_EXPK"));
    if (getenv("LB_IO")) g_io_n = atoi(getenv("LB_IO"));
    if (getenv("LB_SPEC")) g_spec = atoi(getenv("LB_SPEC"));
    if (getenv("LB_GUESS")) g_guess_max = atoi(getenv("LB_GUESS"));
    g_disk = 3;
    e_b = buf_make(max_n);
    return clk() - t0;
}

/* Opens layer `layer` (K input snapshots) and reads its trunk and attention weights through once, so the request finds
   them in memory. The experts stay on disk. */
void *LB_CAT(LB_PFX, _open)(const char *dataset, unsigned layer, unsigned k, unsigned max_n)
{
    LbLayer *h = calloc(1, sizeof *h);
    if (!h) bail("allocation failed");
    h->layer = layer; h->k = k;
    lb_layer_no = layer; lb_k_in = k;
    if (!transformer_open(dataset, &h->tr)) bail("transformer_open failed");
    const Root *root = h->tr->root;
    if (!root->direct) bail("no direct expert mapping");
    h->ident = palette_identity(h->tr->palette);
    char path[4096];
    snprintf(path, sizeof path, "%s/root-%u/experts.direct", dataset, layer);
    h->plain_fd = open(path, O_RDONLY);
    h->direct_fd = open(path, O_RDONLY | O_DIRECT);
    if (h->plain_fd < 0 || h->direct_fd < 0) bail("cannot open experts.direct");
    touch_range(h->tr->qkv, h->tr->qkv_bytes);
    for (unsigned group = 0; group < 3; group++) touch_range(h->tr->group_map[group], h->tr->group_bytes[group]);
    if (use_4bit_trunk_requested()) h->packed_trunk = pack_layer_trunk_to_4bit(h->tr);
#if LB_MLA
    mla_cache_make(max_n);
    h->mla = g_mla;
#else
    (void)max_n;
    h->hist = grab(3 * (size_t)ROWS * 3 * sizeof(float));
    h->rec = grab((size_t)NHEAD * HDIM * HDIM * sizeof(float));
#endif
    return h;
}

/* One layer over `n` tokens as one request: input residual and the K snapshots in, output residual out.
   A boundary layer (every twelfth) does not change the snapshots it was given; the caller appends its own input. */
double LB_CAT(LB_PFX, _run)(void *handle, const float *resid, const float *const *snaps, unsigned n, unsigned window, float *out)
{
    LbLayer *h = handle;
    lb_layer_no = h->layer; lb_k_in = h->k;
    g_palette = h->tr->palette; g_ident = h->ident;
    g_direct_fd = h->direct_fd; g_expert_fd = h->plain_fd;
    g_mla = h->mla;
    if (h->hist) memset(h->hist, 0, 3 * (size_t)ROWS * 3 * sizeof(float));
    if (h->rec) memset(h->rec, 0, (size_t)NHEAD * HDIM * HDIM * sizeof(float));
    g_evict_s = 0; g_distinct_sum = 0; g_window_count = 0; memset(g_hits, 0, sizeof g_hits);
    g_read_experts = 0; g_wasted_experts = 0; memset(g_wait_t, 0, sizeof g_wait_t);
    g_pos0 = 0;
    g_cache = NULL;
    if (h->cache) cache_clear(h->cache);   /* a prompt pass fills the shared expert buffer by expert id, so cached slots may be overwritten */
    g_active_packed_trunk = NULL;   /* the prompt pass is compute-bound, where the 4-bit kernel is slower than the 8-bit one */
    const double t0 = clk();
    run_layer(h->tr, h->tr->root, resid, snaps, n, window, &e_b, out, h->hist, h->rec, NULL);
    return clk() - t0;
}

/* n more tokens at positions pos0.. after earlier ones, as one prompt-style pass (a chat turn): the state and the attention cache stay as
   the earlier tokens left them; the experts are read the way a prompt pass reads them. */
double LB_CAT(LB_PFX, _run_from)(void *handle, const float *resid, const float *const *snaps, unsigned n, unsigned window, size_t pos0, float *out)
{
    LbLayer *h = handle;
    lb_layer_no = h->layer; lb_k_in = h->k;
    g_palette = h->tr->palette; g_ident = h->ident;
    g_direct_fd = h->direct_fd; g_expert_fd = h->plain_fd;
    g_mla = h->mla;
    g_evict_s = 0; g_distinct_sum = 0; g_window_count = 0; memset(g_hits, 0, sizeof g_hits);
    g_read_experts = 0; g_wasted_experts = 0; memset(g_wait_t, 0, sizeof g_wait_t);
    g_pos0 = pos0;
    g_cache = NULL;
    if (h->cache) cache_clear(h->cache);
    g_active_packed_trunk = NULL;
    const double t0 = clk();
    run_layer(h->tr, h->tr->root, resid, snaps, n, window, &e_b, out, h->hist, h->rec, NULL);
    g_pos0 = 0;
    return clk() - t0;
}

/* One more token of a sequence whose prompt this layer has already seen: state and cache are kept, the token is at
   position `pos0`. Same arithmetic as the prompt pass. */
double LB_CAT(LB_PFX, _step)(void *handle, const float *resid, const float *const *snaps, size_t pos0, float *out)
{
    LbLayer *h = handle;
    lb_layer_no = h->layer; lb_k_in = h->k;
    g_palette = h->tr->palette; g_ident = h->ident;
    g_direct_fd = h->direct_fd; g_expert_fd = h->plain_fd;
    g_mla = h->mla;
    g_evict_s = 0; g_distinct_sum = 0; g_window_count = 0; memset(g_hits, 0, sizeof g_hits);
    g_read_experts = 0; g_wasted_experts = 0; memset(g_wait_t, 0, sizeof g_wait_t);
    g_pos0 = pos0;
    g_cache = h->cache;
    g_active_packed_trunk = h->packed_trunk;
    const int kernel = g_expk;
    g_expk = 0;   /* one token per call: the 4-token expert kernel is 10% faster than the 2-row one here (6.3 against 7.0 ms per layer) */
    const double t0 = clk();
    run_layer(h->tr, h->tr->root, resid, snaps, 1, 1, &e_b, out, h->hist, h->rec, NULL);
    g_pos0 = 0;
    g_cache = NULL;
    g_expk = kernel;
    return clk() - t0;
}

/* n tokens at positions pos0.. in one pass (verifying a draft), on the state the earlier tokens left; experts through the cache,
   those that do not fit its slots through the scratch pool. */
double LB_CAT(LB_PFX, _stepk)(void *handle, const float *resid, const float *const *snaps, unsigned n, size_t pos0, float *out)
{
    LbLayer *h = handle;
    lb_layer_no = h->layer; lb_k_in = h->k;
    g_palette = h->tr->palette; g_ident = h->ident;
    g_direct_fd = h->direct_fd; g_expert_fd = h->plain_fd;
    g_mla = h->mla;
    g_evict_s = 0; g_distinct_sum = 0; g_window_count = 0; memset(g_hits, 0, sizeof g_hits);
    g_read_experts = 0; g_wasted_experts = 0; memset(g_wait_t, 0, sizeof g_wait_t);
    if (getenv("LB_SWEEP_EXPK")) g_sweep_expk = atoi(getenv("LB_SWEEP_EXPK"));
    g_pos0 = pos0;
    g_cache = h->cache;
    const int kernel = g_expk;
    g_expk = g_sweep_expk;
    g_active_packed_trunk = h->packed_trunk;
    if (n - 1 <= h->checkpoint_count) { g_checkpoint_rec = h->checkpoint_rec; g_checkpoint_hist = h->checkpoint_hist; }
    const double t0 = clk();
    run_layer(h->tr, h->tr->root, resid, snaps, n, n, &e_b, out, h->hist, h->rec, NULL);
    g_checkpoint_rec = NULL; g_checkpoint_hist = NULL;
    g_pos0 = 0;
    g_cache = NULL;
    g_expk = kernel;
    return clk() - t0;
}
void LB_CAT(LB_PFX, _set_scratch)(unsigned char *const *slots, unsigned n)
{
    if (n > SCRATCH_MAX) bail("scratch pool too large");
    for (unsigned i = 0; i < n; i++) g_scratch[i] = slots[i];
    g_scratch_n = n;
}
unsigned long LB_CAT(LB_PFX, _distinct)(void) { return g_window_count ? g_distinct_sum / g_window_count : 0; }

/* Speculative decoding: keep the state after each token of a pass over a draft (up to `count` tokens), so a rejected draft is undone by copying
   one back. An MLA layer has no such state: the key/value entries of rejected positions are overwritten by the next pass. */
void LB_CAT(LB_PFX, _keep_positions)(void *handle, unsigned count)
{
    LbLayer *h = handle;
    if (!h->hist) return;
    h->checkpoint_rec = grab(count * sizeof *h->checkpoint_rec); h->checkpoint_hist = grab(count * sizeof *h->checkpoint_hist); h->checkpoint_count = count;
}
/* The state as it was after token `position` of the last stepk pass. */
void LB_CAT(LB_PFX, _use_position)(void *handle, unsigned position)
{
    LbLayer *h = handle;
    if (!h->hist) return;
    memcpy(h->rec, h->checkpoint_rec[position], sizeof *h->checkpoint_rec);
    memcpy(h->hist, h->checkpoint_hist[position], sizeof *h->checkpoint_hist);
}
#endif
