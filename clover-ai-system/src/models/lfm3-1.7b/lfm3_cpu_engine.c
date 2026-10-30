#define _GNU_SOURCE
/*
 * Native C / OpenMP CPU Engine for Liquid AI LFM3-1.7B (`liblfm3_cpu_engine.so`)
 * (`clover-ai-system/src/models/lfm3-1.7b/lfm3_cpu_engine.c`)
 *
 * Executes all 24 tri-hybrid layers of `LiquidAI/LFM3-1.7B` (`D_model = 2048`, `V = 128,000` tied)
 * directly from the pruned `LFM3TRK1` trunks (`trunk-0..23`), `LFM3_OP1` composed `operator.bin`
 * files (`attention.bin` deleted from all 24 trunks), and `global-0/` (`embed_tokens.bf16.bin`,
 * `embedding_norm.bf16.bin`, `rope_inv_freq.bin`):
 *   - 13 Gated ShortConv layers (`kind = 0`, width K=7, B*x pre-gate, C*conv post-gate)
 *   - 5 Sliding-Window GQA layers (`kind = 1`, W=256, full RoPE 1.0, per-head sigmoid gate)
 *   - 6 Full Global GQA layers (`kind = 2`, zero-tail partial RoPE 0.5, per-head sigmoid gate)
 *   - 24 Sandwich RMSNorm + SwiGLU FFN blocks (`2048 -> 7680 -> 2048`)
 */

#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#ifndef _WIN32
#include <sched.h>
#endif
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#define LFM3_API __declspec(dllexport)
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#define LFM3_API
#endif

enum {
    LFM3_D = 2048,
    LFM3_FFN = 7680,
    LFM3_LAYERS = 24,
    LFM3_HEADS = 16,
    LFM3_KV_HEADS = 4,
    LFM3_HDIM = 128,
    LFM3_CONV_K = 7,
    LFM3_SLIDING_W = 256,
    LFM3_HEADER_SIZE = 72,
    LFM3_RECORD_SIZE = 36,
    LFM3_FFN_MAT_4BIT_BYTES = 8140800 /* 7680x2048 or 2048x7680 in planar 4-bit group-64 */
};

static inline float bf16_to_f32(uint16_t b)
{
    union { uint32_t u; float f; } v;
    v.u = ((uint32_t)b) << 16;
    return v.f;
}

static inline uint16_t f32_to_bf16_rne(float x)
{
    union { float f; uint32_t u; } v;
    v.f = x;
    if ((v.u & 0x7FFFFFFFU) > 0x7F800000U) {
        return (uint16_t)((v.u >> 16) | 0x0040U);
    }
    uint32_t rounding_bias = 0x00007FFFU + ((v.u >> 16) & 1U);
    return (uint16_t)((v.u + rounding_bias) >> 16);
}

static inline float bf16_round_f32(float x)
{
    return bf16_to_f32(f32_to_bf16_rne(x));
}

typedef struct {
    void *ptr;
    size_t size;
    int fd;
#ifdef _WIN32
    HANDLE hfile;
    HANDLE hmap;
#endif
} MmapFile;

static double now_ms(void)
{
#ifdef _WIN32
    static LARGE_INTEGER freq;
    LARGE_INTEGER c;
    if (!freq.QuadPart) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / (double)freq.QuadPart;
#else
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
#endif
}

static int file_readable(const char *path)
{
#ifdef _WIN32
    return GetFileAttributesA(path) != INVALID_FILE_ATTRIBUTES;
#else
    return access(path, R_OK) == 0;
#endif
}

#ifdef _WIN32
static int mmap_readonly(const char *path, MmapFile *out)
{
    out->ptr = NULL;
    out->size = 0;
    out->fd = -1;
    out->hmap = NULL;
    out->hfile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (out->hfile == INVALID_HANDLE_VALUE) {
        out->hfile = NULL;
        return -1;
    }
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(out->hfile, &sz)) {
        CloseHandle(out->hfile);
        out->hfile = NULL;
        return -1;
    }
    out->size = (size_t)sz.QuadPart;
    out->fd = 0;
    if (out->size == 0) return 0;
    out->hmap = CreateFileMappingA(out->hfile, NULL, PAGE_READONLY, 0, 0, NULL);
    out->ptr = out->hmap ? MapViewOfFile(out->hmap, FILE_MAP_READ, 0, 0, 0) : NULL;
    if (!out->ptr) {
        if (out->hmap) CloseHandle(out->hmap);
        CloseHandle(out->hfile);
        out->hmap = NULL;
        out->hfile = NULL;
        out->fd = -1;
        return -1;
    }
    return 0;
}
#else
static int mmap_readonly(const char *path, MmapFile *out)
{
    out->ptr = NULL;
    out->size = 0;
    out->fd = open(path, O_RDONLY);
    if (out->fd < 0) return -1;
    struct stat st;
    if (fstat(out->fd, &st) != 0) {
        close(out->fd);
        out->fd = -1;
        return -1;
    }
    out->size = (size_t)st.st_size;
    if (out->size == 0) return 0;
    out->ptr = mmap(NULL, out->size, PROT_READ, MAP_PRIVATE, out->fd, 0);
    if (out->ptr == MAP_FAILED) {
        out->ptr = NULL;
        close(out->fd);
        out->fd = -1;
        return -1;
    }
    return 0;
}
#endif

#pragma pack(push, 1)
typedef struct {
    char magic[8];
    uint32_t version;
    uint32_t layer_id;
    uint32_t layer_kind;
    uint32_t record_count;
    uint64_t g0_off, g0_len;
    uint64_t g1_off, g1_len;
    uint64_t g2_off, g2_len;
} Lfm3TrunkHeader;

typedef struct {
    uint32_t rec_id;
    uint32_t grp_id;
    uint32_t dtype_code;
    uint32_t rows;
    uint64_t offset;
    uint64_t length;
    uint32_t cols;
} Lfm3RecordDesc;
#pragma pack(pop)

typedef struct {
    MmapFile index_mf;
    MmapFile group_mf[3];
    MmapFile operator_mf;
    MmapFile ffn_4bit_mf;
    Lfm3TrunkHeader hdr;
    Lfm3RecordDesc recs[20];
} Lfm3TrunkMapped;

static int load_lfm3_trunk(const char *trunk_dir, Lfm3TrunkMapped *t)
{
    memset(t, 0, sizeof(*t));
    t->index_mf.fd = -1;
    t->operator_mf.fd = -1;
    t->ffn_4bit_mf.fd = -1;
    for (int g = 0; g < 3; g++) t->group_mf[g].fd = -1;

    char path[2048];
    snprintf(path, sizeof(path), "%s/index.bin", trunk_dir);
    if (mmap_readonly(path, &t->index_mf) != 0 || t->index_mf.size < LFM3_HEADER_SIZE) return -1;
    memcpy(&t->hdr, t->index_mf.ptr, sizeof(Lfm3TrunkHeader));
    if (memcmp(t->hdr.magic, "LFM3TRK1", 8) != 0) return -2;

    const uint8_t *p = (const uint8_t *)t->index_mf.ptr + LFM3_HEADER_SIZE;
    uint32_t nrec = t->hdr.record_count < 20 ? t->hdr.record_count : 20;
    for (uint32_t i = 0; i < nrec; i++) {
        memcpy(&t->recs[i], p + i * LFM3_RECORD_SIZE, sizeof(Lfm3RecordDesc));
    }

    const char *gnames[3] = { "common.bin", "attention.bin", "ffn.bin" };
    for (int g = 0; g < 3; g++) {
        snprintf(path, sizeof(path), "%s/%s", trunk_dir, gnames[g]);
        if (file_readable(path)) {
            if (mmap_readonly(path, &t->group_mf[g]) != 0) return -3;
        }
    }
    snprintf(path, sizeof(path), "%s/operator.bin", trunk_dir);
    if (file_readable(path)) {
        if (mmap_readonly(path, &t->operator_mf) != 0) return -4;
    }
    snprintf(path, sizeof(path), "%s/ffn.4bit.bin", trunk_dir);
    if (file_readable(path)) {
        mmap_readonly(path, &t->ffn_4bit_mf);
    }
    return 0;
}

static const void *get_lfm3_record_ptr(const Lfm3TrunkMapped *t, uint32_t rec_id)
{
    for (uint32_t i = 0; i < t->hdr.record_count && i < 20; i++) {
        if (t->recs[i].rec_id == rec_id) {
            if (t->recs[i].length == 0) return NULL;
            uint32_t g = t->recs[i].grp_id;
            if (g >= 3 || !t->group_mf[g].ptr) return NULL;
            return (const uint8_t *)t->group_mf[g].ptr + t->recs[i].offset;
        }
    }
    return NULL;
}

/* SpecMD Policy 3 (Least-Stale + R4 Clean-Drop) & Eyeriss 4-Tier Ledger for LFM3-Omni / Samsung S23 */
static struct {
    int initialized;
    int use_4bit_ffn;
    int num_threads;
    int s23_pin_requested;
    int s23_big_cores_pinned;
    MmapFile emb_mf;
    MmapFile emb_norm_mf;
    MmapFile rope_mf;
    float inv_freq_full[64];
    float inv_freq_sliding[64];
    Lfm3TrunkMapped trunks[LFM3_LAYERS];

    unsigned long clock;
    int pool_keys[12];
    unsigned long pool_ts[12];
    float pool_heat[12];
    int host_keys[LFM3_LAYERS];

    unsigned long total_layer_requests;
    unsigned long tier1_sram_l3_hits;
    unsigned long tier2_lpddr5x_hits;
    unsigned long tier3_ufs4_fetches;
    unsigned long r4_clean_drops;
    unsigned long dirty_writebacks;
} g_lfm3 = { 0, 0, 8, 0 };

static void lfm3_dm_reset(void)
{
    g_lfm3.clock = 0;
    for (int i = 0; i < 12; i++) {
        g_lfm3.pool_keys[i] = -1;
        g_lfm3.pool_ts[i] = 0;
        g_lfm3.pool_heat[i] = 0.0f;
    }
    for (int l = 0; l < LFM3_LAYERS; l++) g_lfm3.host_keys[l] = 0;
    g_lfm3.total_layer_requests = 0;
    g_lfm3.tier1_sram_l3_hits = 0;
    g_lfm3.tier2_lpddr5x_hits = 0;
    g_lfm3.tier3_ufs4_fetches = 0;
    g_lfm3.r4_clean_drops = 0;
    g_lfm3.dirty_writebacks = 0;
}

static void lfm3_dm_record_layer(int layer_id, int tokens_in_pass)
{
    g_lfm3.clock++;
    g_lfm3.total_layer_requests++;
    float w = (float)(tokens_in_pass > 0 ? tokens_in_pass : 1);
    for (int i = 0; i < 12; i++) {
        if (g_lfm3.pool_keys[i] == layer_id) {
            g_lfm3.tier1_sram_l3_hits++;
            g_lfm3.pool_ts[i] = g_lfm3.clock;
            g_lfm3.pool_heat[i] = 0.75f * g_lfm3.pool_heat[i] + w;
            return;
        }
    }
    if (g_lfm3.host_keys[layer_id]) {
        g_lfm3.tier2_lpddr5x_hits++;
    } else {
        g_lfm3.tier3_ufs4_fetches++;
        g_lfm3.host_keys[layer_id] = 1;
    }
    int slot = -1;
    for (int i = 0; i < 12; i++) {
        if (g_lfm3.pool_keys[i] < 0) {
            slot = i;
            break;
        }
    }
    if (slot < 0) {
        int best_v = 0;
        double best_sc = 1e30;
        int next_layer = (layer_id + 1) % LFM3_LAYERS;
        for (int i = 0; i < 12; i++) {
            int k = g_lfm3.pool_keys[i];
            if (k == layer_id || k == next_layer) continue; /* SpecMD Lookahead: protect Layer L and L+1 */
            unsigned long stale = g_lfm3.clock > g_lfm3.pool_ts[i] ? (g_lfm3.clock - g_lfm3.pool_ts[i]) : 1UL;
            double sc = (double)g_lfm3.pool_heat[i] / (double)(1UL + stale);
            if (sc < best_sc) {
                best_sc = sc;
                best_v = i;
            }
        }
        slot = best_v;
        int victim = g_lfm3.pool_keys[slot];
        if (victim >= 0 && g_lfm3.host_keys[victim]) {
            g_lfm3.r4_clean_drops++; /* Red-Blue Pebble R4 Clean-Drop: 0 dirty writebacks! */
        } else {
            g_lfm3.dirty_writebacks++;
        }
    }
    g_lfm3.pool_keys[slot] = layer_id;
    g_lfm3.pool_ts[slot] = g_lfm3.clock;
    g_lfm3.pool_heat[slot] = w;
}

LFM3_API void lfm3_set_use_4bit_ffn(int enable)
{
    g_lfm3.use_4bit_ffn = enable ? 1 : 0;
}

LFM3_API int lfm3_configure_s23_snapdragon8gen2(int pin_big_cores_3_to_7)
{
    /* Samsung Galaxy S23 (Snapdragon 8 Gen 2 for Galaxy, SM8550-AC):
     *   - Cores 0..2: 3x Cortex-A510 efficiency cores (2.0 GHz, in-order)
     *   - Cores 3..4: 2x Cortex-A710 performance cores (2.8 GHz)
     *   - Cores 5..6: 2x Cortex-A715 performance cores (2.8 GHz)
     *   - Core 7:     1x Cortex-X3 prime core (3.36 GHz)
     * Inspects Android cpuset affinity first: if cores 3..7 are available (foreground/top-app),
     * pins to the 4 or 5 out-of-order cores (3..7); if restricted by background cpuset (e.g. 0..2),
     * matches OpenMP thread count to allowed cores to prevent barrier spinlock oversubscription. */
    g_lfm3.s23_pin_requested = pin_big_cores_3_to_7 ? 1 : 0;
    g_lfm3.num_threads = pin_big_cores_3_to_7 ? 5 : 8;
#if defined(__linux__) && defined(__aarch64__)
    cpu_set_t current_mask;
    CPU_ZERO(&current_mask);
    if (sched_getaffinity(0, sizeof(current_mask), &current_mask) == 0) {
        cpu_set_t big_mask;
        CPU_ZERO(&big_mask);
        int big_avail = 0;
        for (int c = 3; c <= 7; c++) {
            if (CPU_ISSET(c, &current_mask)) {
                CPU_SET(c, &big_mask);
                big_avail++;
            }
        }
        if (pin_big_cores_3_to_7 && big_avail >= 4) {
            if (sched_setaffinity(0, sizeof(big_mask), &big_mask) == 0) {
                g_lfm3.s23_big_cores_pinned = 1;
                g_lfm3.num_threads = big_avail;
            }
        } else {
            int total_avail = CPU_COUNT(&current_mask);
            if (total_avail > 0) {
                g_lfm3.num_threads = total_avail;
            }
        }
    }
#endif
    omp_set_num_threads(g_lfm3.num_threads);
    return g_lfm3.num_threads;
}

LFM3_API void lfm3_dm_stats(unsigned long *out_stats6)
{
    if (!out_stats6) return;
    out_stats6[0] = g_lfm3.total_layer_requests;
    out_stats6[1] = g_lfm3.tier1_sram_l3_hits;
    out_stats6[2] = g_lfm3.tier2_lpddr5x_hits;
    out_stats6[3] = g_lfm3.tier3_ufs4_fetches;
    out_stats6[4] = g_lfm3.r4_clean_drops;
    out_stats6[5] = g_lfm3.dirty_writebacks;
}

static void linear_bf16_omp(
    const float *x,
    const uint16_t *W_bf16,
    float *y,
    int T,
    int out_dim,
    int in_dim
)
{
    const int n_thr = g_lfm3.num_threads > 0 ? g_lfm3.num_threads : 8;
    #pragma omp parallel for num_threads(n_thr) schedule(static)
    for (int o = 0; o < out_dim; o++) {
        const uint16_t *w_row = W_bf16 + (size_t)o * (size_t)in_dim;
        int t = 0;
        for (; t + 8 <= T; t += 8) {
            const float *x0 = x + (size_t)(t + 0) * in_dim;
            const float *x1 = x + (size_t)(t + 1) * in_dim;
            const float *x2 = x + (size_t)(t + 2) * in_dim;
            const float *x3 = x + (size_t)(t + 3) * in_dim;
            const float *x4 = x + (size_t)(t + 4) * in_dim;
            const float *x5 = x + (size_t)(t + 5) * in_dim;
            const float *x6 = x + (size_t)(t + 6) * in_dim;
            const float *x7 = x + (size_t)(t + 7) * in_dim;
#if defined(__aarch64__) || defined(__ARM_NEON)
            float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f), acc3 = vdupq_n_f32(0.0f);
            float32x4_t acc4 = vdupq_n_f32(0.0f), acc5 = vdupq_n_f32(0.0f);
            float32x4_t acc6 = vdupq_n_f32(0.0f), acc7 = vdupq_n_f32(0.0f);
            int k = 0;
            for (; k + 4 <= in_dim; k += 4) {
                uint16x4_t w_u16 = vld1_u16(w_row + k);
                float32x4_t wv = vreinterpretq_f32_u32(vshll_n_u16(w_u16, 16));
                acc0 = vfmaq_f32(acc0, vld1q_f32(x0 + k), wv);
                acc1 = vfmaq_f32(acc1, vld1q_f32(x1 + k), wv);
                acc2 = vfmaq_f32(acc2, vld1q_f32(x2 + k), wv);
                acc3 = vfmaq_f32(acc3, vld1q_f32(x3 + k), wv);
                acc4 = vfmaq_f32(acc4, vld1q_f32(x4 + k), wv);
                acc5 = vfmaq_f32(acc5, vld1q_f32(x5 + k), wv);
                acc6 = vfmaq_f32(acc6, vld1q_f32(x6 + k), wv);
                acc7 = vfmaq_f32(acc7, vld1q_f32(x7 + k), wv);
            }
            float s0 = vaddvq_f32(acc0), s1 = vaddvq_f32(acc1);
            float s2 = vaddvq_f32(acc2), s3 = vaddvq_f32(acc3);
            float s4 = vaddvq_f32(acc4), s5 = vaddvq_f32(acc5);
            float s6 = vaddvq_f32(acc6), s7 = vaddvq_f32(acc7);
            for (; k < in_dim; k++) {
                float w = bf16_to_f32(w_row[k]);
                s0 += x0[k] * w; s1 += x1[k] * w; s2 += x2[k] * w; s3 += x3[k] * w;
                s4 += x4[k] * w; s5 += x5[k] * w; s6 += x6[k] * w; s7 += x7[k] * w;
            }
#else
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            float s4 = 0.0f, s5 = 0.0f, s6 = 0.0f, s7 = 0.0f;
            for (int k = 0; k < in_dim; k++) {
                float w = bf16_to_f32(w_row[k]);
                s0 += x0[k] * w; s1 += x1[k] * w; s2 += x2[k] * w; s3 += x3[k] * w;
                s4 += x4[k] * w; s5 += x5[k] * w; s6 += x6[k] * w; s7 += x7[k] * w;
            }
#endif
            y[(size_t)(t + 0) * out_dim + o] = bf16_round_f32(s0);
            y[(size_t)(t + 1) * out_dim + o] = bf16_round_f32(s1);
            y[(size_t)(t + 2) * out_dim + o] = bf16_round_f32(s2);
            y[(size_t)(t + 3) * out_dim + o] = bf16_round_f32(s3);
            y[(size_t)(t + 4) * out_dim + o] = bf16_round_f32(s4);
            y[(size_t)(t + 5) * out_dim + o] = bf16_round_f32(s5);
            y[(size_t)(t + 6) * out_dim + o] = bf16_round_f32(s6);
            y[(size_t)(t + 7) * out_dim + o] = bf16_round_f32(s7);
        }
        for (; t + 4 <= T; t += 4) {
            const float *x0 = x + (size_t)(t + 0) * in_dim;
            const float *x1 = x + (size_t)(t + 1) * in_dim;
            const float *x2 = x + (size_t)(t + 2) * in_dim;
            const float *x3 = x + (size_t)(t + 3) * in_dim;
#if defined(__aarch64__) || defined(__ARM_NEON)
            float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f), acc3 = vdupq_n_f32(0.0f);
            int k = 0;
            for (; k + 4 <= in_dim; k += 4) {
                float32x4_t wv = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w_row + k), 16));
                acc0 = vfmaq_f32(acc0, vld1q_f32(x0 + k), wv);
                acc1 = vfmaq_f32(acc1, vld1q_f32(x1 + k), wv);
                acc2 = vfmaq_f32(acc2, vld1q_f32(x2 + k), wv);
                acc3 = vfmaq_f32(acc3, vld1q_f32(x3 + k), wv);
            }
            float s0 = vaddvq_f32(acc0), s1 = vaddvq_f32(acc1);
            float s2 = vaddvq_f32(acc2), s3 = vaddvq_f32(acc3);
            for (; k < in_dim; k++) {
                float w = bf16_to_f32(w_row[k]);
                s0 += x0[k] * w; s1 += x1[k] * w; s2 += x2[k] * w; s3 += x3[k] * w;
            }
#else
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            for (int k = 0; k < in_dim; k++) {
                float w = bf16_to_f32(w_row[k]);
                s0 += x0[k] * w; s1 += x1[k] * w; s2 += x2[k] * w; s3 += x3[k] * w;
            }
#endif
            y[(size_t)(t + 0) * out_dim + o] = bf16_round_f32(s0);
            y[(size_t)(t + 1) * out_dim + o] = bf16_round_f32(s1);
            y[(size_t)(t + 2) * out_dim + o] = bf16_round_f32(s2);
            y[(size_t)(t + 3) * out_dim + o] = bf16_round_f32(s3);
        }
        for (; t < T; t++) {
            const float *x_row = x + (size_t)t * (size_t)in_dim;
#if defined(__aarch64__) || defined(__ARM_NEON)
            float32x4_t acc = vdupq_n_f32(0.0f);
            int k = 0;
            for (; k + 4 <= in_dim; k += 4) {
                float32x4_t wv = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(w_row + k), 16));
                acc = vfmaq_f32(acc, vld1q_f32(x_row + k), wv);
            }
            float sum = vaddvq_f32(acc);
            for (; k < in_dim; k++) sum += x_row[k] * bf16_to_f32(w_row[k]);
#else
            float sum = 0.0f;
            for (int k = 0; k < in_dim; k++) {
                sum += x_row[k] * bf16_to_f32(w_row[k]);
            }
#endif
            y[(size_t)t * (size_t)out_dim + (size_t)o] = bf16_round_f32(sum);
        }
    }
}

/* OpenMP 4-Bit Planar Group-64 Linear with In-Register 8-Token + 4-Token + 1-Token ARMv9 NEON (`vfmaq_f32`) Blocking:
 * Zero stack buffer (`w_f32`) so `x` stays 100% resident in the 32 KB L1D cache across all output rows! */
static void linear_4bit_group64_omp(
    const float *x,
    const uint8_t *nibbles_base,
    const uint8_t *steps_base,
    const float *scales_base,
    float *y,
    int T,
    int out_dim,
    int in_dim
)
{
    const int groups = in_dim / 64;
    const int n_thr = g_lfm3.num_threads > 0 ? g_lfm3.num_threads : 8;
    #pragma omp parallel for num_threads(n_thr) schedule(static)
    for (int o = 0; o < out_dim; o++) {
        const uint8_t *nib_row = nibbles_base + (size_t)o * (size_t)(groups * 32);
        const uint8_t *stp_row = steps_base + (size_t)o * (size_t)groups;
        const float row_scale = scales_base[o];
        int t = 0;
        for (; t + 8 <= T; t += 8) {
            const float *x0 = x + (size_t)(t + 0) * in_dim;
            const float *x1 = x + (size_t)(t + 1) * in_dim;
            const float *x2 = x + (size_t)(t + 2) * in_dim;
            const float *x3 = x + (size_t)(t + 3) * in_dim;
            const float *x4 = x + (size_t)(t + 4) * in_dim;
            const float *x5 = x + (size_t)(t + 5) * in_dim;
            const float *x6 = x + (size_t)(t + 6) * in_dim;
            const float *x7 = x + (size_t)(t + 7) * in_dim;
#if defined(__aarch64__) || defined(__ARM_NEON)
            float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
            float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
            float32x4_t a4 = vdupq_n_f32(0.0f), a5 = vdupq_n_f32(0.0f);
            float32x4_t a6 = vdupq_n_f32(0.0f), a7 = vdupq_n_f32(0.0f);
            const int16x8_t eight_s16 = vdupq_n_s16(8);
            const uint8x8_t mask_lo8 = vdup_n_u8(0x0F);
            for (int g = 0; g < groups; g++) {
                const uint8_t *nb = nib_row + g * 32;
                const float32x4_t gsc_v = vdupq_n_f32((float)stp_row[g] * row_scale);
                const int col0 = g * 64;
                for (int j = 0; j < 32; j += 8) {
                    uint8x8_t pk8 = vld1_u8(nb + j);
                    int16x8_t qlo = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vand_u8(pk8, mask_lo8))), eight_s16);
                    int16x8_t qhi = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vshr_n_u8(pk8, 4))), eight_s16);
                    float32x4_t wlo0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(qlo))), gsc_v);
                    float32x4_t wlo1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(qlo))), gsc_v);
                    float32x4_t whi0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(qhi))), gsc_v);
                    float32x4_t whi1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(qhi))), gsc_v);
                    int clo = col0 + j;
                    int chi = col0 + 32 + j;
                    a0 = vfmaq_f32(vfmaq_f32(a0, vld1q_f32(x0 + clo), wlo0), vld1q_f32(x0 + clo + 4), wlo1);
                    a0 = vfmaq_f32(vfmaq_f32(a0, vld1q_f32(x0 + chi), whi0), vld1q_f32(x0 + chi + 4), whi1);
                    a1 = vfmaq_f32(vfmaq_f32(a1, vld1q_f32(x1 + clo), wlo0), vld1q_f32(x1 + clo + 4), wlo1);
                    a1 = vfmaq_f32(vfmaq_f32(a1, vld1q_f32(x1 + chi), whi0), vld1q_f32(x1 + chi + 4), whi1);
                    a2 = vfmaq_f32(vfmaq_f32(a2, vld1q_f32(x2 + clo), wlo0), vld1q_f32(x2 + clo + 4), wlo1);
                    a2 = vfmaq_f32(vfmaq_f32(a2, vld1q_f32(x2 + chi), whi0), vld1q_f32(x2 + chi + 4), whi1);
                    a3 = vfmaq_f32(vfmaq_f32(a3, vld1q_f32(x3 + clo), wlo0), vld1q_f32(x3 + clo + 4), wlo1);
                    a3 = vfmaq_f32(vfmaq_f32(a3, vld1q_f32(x3 + chi), whi0), vld1q_f32(x3 + chi + 4), whi1);
                    a4 = vfmaq_f32(vfmaq_f32(a4, vld1q_f32(x4 + clo), wlo0), vld1q_f32(x4 + clo + 4), wlo1);
                    a4 = vfmaq_f32(vfmaq_f32(a4, vld1q_f32(x4 + chi), whi0), vld1q_f32(x4 + chi + 4), whi1);
                    a5 = vfmaq_f32(vfmaq_f32(a5, vld1q_f32(x5 + clo), wlo0), vld1q_f32(x5 + clo + 4), wlo1);
                    a5 = vfmaq_f32(vfmaq_f32(a5, vld1q_f32(x5 + chi), whi0), vld1q_f32(x5 + chi + 4), whi1);
                    a6 = vfmaq_f32(vfmaq_f32(a6, vld1q_f32(x6 + clo), wlo0), vld1q_f32(x6 + clo + 4), wlo1);
                    a6 = vfmaq_f32(vfmaq_f32(a6, vld1q_f32(x6 + chi), whi0), vld1q_f32(x6 + chi + 4), whi1);
                    a7 = vfmaq_f32(vfmaq_f32(a7, vld1q_f32(x7 + clo), wlo0), vld1q_f32(x7 + clo + 4), wlo1);
                    a7 = vfmaq_f32(vfmaq_f32(a7, vld1q_f32(x7 + chi), whi0), vld1q_f32(x7 + chi + 4), whi1);
                }
            }
            float s0 = vaddvq_f32(a0), s1 = vaddvq_f32(a1);
            float s2 = vaddvq_f32(a2), s3 = vaddvq_f32(a3);
            float s4 = vaddvq_f32(a4), s5 = vaddvq_f32(a5);
            float s6 = vaddvq_f32(a6), s7 = vaddvq_f32(a7);
#else
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            float s4 = 0.0f, s5 = 0.0f, s6 = 0.0f, s7 = 0.0f;
            for (int g = 0; g < groups; g++) {
                const uint8_t *nb = nib_row + g * 32;
                const float g_sc = (float)stp_row[g] * row_scale;
                const int col0 = g * 64;
                for (int j = 0; j < 32; j++) {
                    uint8_t pk = nb[j];
                    float wl = (float)((int)(pk & 0x0F) - 8) * g_sc;
                    float wh = (float)((int)(pk >> 4) - 8) * g_sc;
                    int c_lo = col0 + j, c_hi = col0 + 32 + j;
                    s0 += x0[c_lo] * wl + x0[c_hi] * wh; s1 += x1[c_lo] * wl + x1[c_hi] * wh;
                    s2 += x2[c_lo] * wl + x2[c_hi] * wh; s3 += x3[c_lo] * wl + x3[c_hi] * wh;
                    s4 += x4[c_lo] * wl + x4[c_hi] * wh; s5 += x5[c_lo] * wl + x5[c_hi] * wh;
                    s6 += x6[c_lo] * wl + x6[c_hi] * wh; s7 += x7[c_lo] * wl + x7[c_hi] * wh;
                }
            }
#endif
            y[(size_t)(t + 0) * out_dim + o] = bf16_round_f32(s0);
            y[(size_t)(t + 1) * out_dim + o] = bf16_round_f32(s1);
            y[(size_t)(t + 2) * out_dim + o] = bf16_round_f32(s2);
            y[(size_t)(t + 3) * out_dim + o] = bf16_round_f32(s3);
            y[(size_t)(t + 4) * out_dim + o] = bf16_round_f32(s4);
            y[(size_t)(t + 5) * out_dim + o] = bf16_round_f32(s5);
            y[(size_t)(t + 6) * out_dim + o] = bf16_round_f32(s6);
            y[(size_t)(t + 7) * out_dim + o] = bf16_round_f32(s7);
        }
        for (; t + 4 <= T; t += 4) {
            const float *x0 = x + (size_t)(t + 0) * in_dim;
            const float *x1 = x + (size_t)(t + 1) * in_dim;
            const float *x2 = x + (size_t)(t + 2) * in_dim;
            const float *x3 = x + (size_t)(t + 3) * in_dim;
#if defined(__aarch64__) || defined(__ARM_NEON)
            float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
            float32x4_t a2 = vdupq_n_f32(0.0f), a3 = vdupq_n_f32(0.0f);
            const int16x8_t eight_s16 = vdupq_n_s16(8);
            const uint8x8_t mask_lo8 = vdup_n_u8(0x0F);
            for (int g = 0; g < groups; g++) {
                const uint8_t *nb = nib_row + g * 32;
                const float32x4_t gsc_v = vdupq_n_f32((float)stp_row[g] * row_scale);
                const int col0 = g * 64;
                for (int j = 0; j < 32; j += 8) {
                    uint8x8_t pk8 = vld1_u8(nb + j);
                    int16x8_t qlo = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vand_u8(pk8, mask_lo8))), eight_s16);
                    int16x8_t qhi = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vshr_n_u8(pk8, 4))), eight_s16);
                    float32x4_t wlo0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(qlo))), gsc_v);
                    float32x4_t wlo1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(qlo))), gsc_v);
                    float32x4_t whi0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(qhi))), gsc_v);
                    float32x4_t whi1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(qhi))), gsc_v);
                    int clo = col0 + j;
                    int chi = col0 + 32 + j;
                    a0 = vfmaq_f32(vfmaq_f32(a0, vld1q_f32(x0 + clo), wlo0), vld1q_f32(x0 + clo + 4), wlo1);
                    a0 = vfmaq_f32(vfmaq_f32(a0, vld1q_f32(x0 + chi), whi0), vld1q_f32(x0 + chi + 4), whi1);
                    a1 = vfmaq_f32(vfmaq_f32(a1, vld1q_f32(x1 + clo), wlo0), vld1q_f32(x1 + clo + 4), wlo1);
                    a1 = vfmaq_f32(vfmaq_f32(a1, vld1q_f32(x1 + chi), whi0), vld1q_f32(x1 + chi + 4), whi1);
                    a2 = vfmaq_f32(vfmaq_f32(a2, vld1q_f32(x2 + clo), wlo0), vld1q_f32(x2 + clo + 4), wlo1);
                    a2 = vfmaq_f32(vfmaq_f32(a2, vld1q_f32(x2 + chi), whi0), vld1q_f32(x2 + chi + 4), whi1);
                    a3 = vfmaq_f32(vfmaq_f32(a3, vld1q_f32(x3 + clo), wlo0), vld1q_f32(x3 + clo + 4), wlo1);
                    a3 = vfmaq_f32(vfmaq_f32(a3, vld1q_f32(x3 + chi), whi0), vld1q_f32(x3 + chi + 4), whi1);
                }
            }
            float s0 = vaddvq_f32(a0), s1 = vaddvq_f32(a1);
            float s2 = vaddvq_f32(a2), s3 = vaddvq_f32(a3);
#else
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            for (int g = 0; g < groups; g++) {
                const uint8_t *nb = nib_row + g * 32;
                const float g_sc = (float)stp_row[g] * row_scale;
                const int col0 = g * 64;
                for (int j = 0; j < 32; j++) {
                    uint8_t pk = nb[j];
                    float wl = (float)((int)(pk & 0x0F) - 8) * g_sc;
                    float wh = (float)((int)(pk >> 4) - 8) * g_sc;
                    int c_lo = col0 + j, c_hi = col0 + 32 + j;
                    s0 += x0[c_lo] * wl + x0[c_hi] * wh; s1 += x1[c_lo] * wl + x1[c_hi] * wh;
                    s2 += x2[c_lo] * wl + x2[c_hi] * wh; s3 += x3[c_lo] * wl + x3[c_hi] * wh;
                }
            }
#endif
            y[(size_t)(t + 0) * out_dim + o] = bf16_round_f32(s0);
            y[(size_t)(t + 1) * out_dim + o] = bf16_round_f32(s1);
            y[(size_t)(t + 2) * out_dim + o] = bf16_round_f32(s2);
            y[(size_t)(t + 3) * out_dim + o] = bf16_round_f32(s3);
        }
        for (; t < T; t++) {
            const float *xt = x + (size_t)t * in_dim;
#if defined(__aarch64__) || defined(__ARM_NEON)
            float32x4_t acc_v = vdupq_n_f32(0.0f);
            const int16x8_t eight_s16 = vdupq_n_s16(8);
            const uint8x8_t mask_lo8 = vdup_n_u8(0x0F);
            for (int g = 0; g < groups; g++) {
                const uint8_t *nb = nib_row + g * 32;
                const float32x4_t gsc_v = vdupq_n_f32((float)stp_row[g] * row_scale);
                const int col0 = g * 64;
                for (int j = 0; j < 32; j += 8) {
                    uint8x8_t pk8 = vld1_u8(nb + j);
                    int16x8_t qlo = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vand_u8(pk8, mask_lo8))), eight_s16);
                    int16x8_t qhi = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(vshr_n_u8(pk8, 4))), eight_s16);
                    float32x4_t wlo0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(qlo))), gsc_v);
                    float32x4_t wlo1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(qlo))), gsc_v);
                    float32x4_t whi0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(qhi))), gsc_v);
                    float32x4_t whi1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(qhi))), gsc_v);
                    acc_v = vfmaq_f32(acc_v, vld1q_f32(xt + col0 + j), wlo0);
                    acc_v = vfmaq_f32(acc_v, vld1q_f32(xt + col0 + j + 4), wlo1);
                    acc_v = vfmaq_f32(acc_v, vld1q_f32(xt + col0 + 32 + j), whi0);
                    acc_v = vfmaq_f32(acc_v, vld1q_f32(xt + col0 + 32 + j + 4), whi1);
                }
            }
            float sum = vaddvq_f32(acc_v);
#else
            float sum = 0.0f;
            for (int g = 0; g < groups; g++) {
                const uint8_t *nb = nib_row + g * 32;
                const float g_sc = (float)stp_row[g] * row_scale;
                const int col0 = g * 64;
                for (int j = 0; j < 32; j++) {
                    uint8_t pk = nb[j];
                    sum += xt[col0 + j] * ((float)((int)(pk & 0x0F) - 8) * g_sc)
                         + xt[col0 + 32 + j] * ((float)((int)(pk >> 4) - 8) * g_sc);
                }
            }
#endif
            y[(size_t)t * out_dim + o] = bf16_round_f32(sum);
        }
    }
}

static void rms_norm_bf16(
    const float *x,
    const uint16_t *w_bf16,
    float *y,
    int n_rows,
    int dim,
    float eps
)
{
    const int n_thr = g_lfm3.num_threads > 0 ? g_lfm3.num_threads : 8;
    #pragma omp parallel for num_threads(n_thr) schedule(static)
    for (int r = 0; r < n_rows; r++) {
        const float *xr = x + (size_t)r * dim;
        float *yr = y + (size_t)r * dim;
        double ss = 0.0;
        for (int d = 0; d < dim; d++) ss += (double)xr[d] * (double)xr[d];
        float inv = 1.0f / sqrtf((float)(ss / (double)dim) + eps);
        for (int d = 0; d < dim; d++) {
            float normed_bf16 = bf16_round_f32(xr[d] * inv);
            yr[d] = bf16_round_f32(normed_bf16 * bf16_to_f32(w_bf16[d]));
        }
    }
}

LFM3_API int lfm3_engine_init(const char *pruned_dataset_dir)
{
    if (g_lfm3.initialized) return 0;
    char path[1024];
    snprintf(path, sizeof(path), "%s/global-0/embed_tokens.bf16.bin", pruned_dataset_dir);
    if (mmap_readonly(path, &g_lfm3.emb_mf) != 0) return -1;

    snprintf(path, sizeof(path), "%s/global-0/embedding_norm.bf16.bin", pruned_dataset_dir);
    if (mmap_readonly(path, &g_lfm3.emb_norm_mf) != 0) return -2;

    snprintf(path, sizeof(path), "%s/global-0/rope_inv_freq.bin", pruned_dataset_dir);
    if (mmap_readonly(path, &g_lfm3.rope_mf) != 0 || g_lfm3.rope_mf.size < 20 + 512) return -3;
    const uint8_t *rp = (const uint8_t *)g_lfm3.rope_mf.ptr;
    memcpy(g_lfm3.inv_freq_full, rp + 20, 256);
    memcpy(g_lfm3.inv_freq_sliding, rp + 20 + 256, 256);

    for (int i = 0; i < LFM3_LAYERS; i++) {
        snprintf(path, sizeof(path), "%s/trunk-%d", pruned_dataset_dir, i);
        if (load_lfm3_trunk(path, &g_lfm3.trunks[i]) != 0) return -(10 + i);
    }
    if (g_lfm3.trunks[0].ffn_4bit_mf.ptr && !g_lfm3.trunks[0].group_mf[2].ptr) {
        g_lfm3.use_4bit_ffn = 1;
    }

    lfm3_dm_reset();
    g_lfm3.initialized = 1;
    return 0;
}

static void execute_single_lfm3_layer_c(
    const Lfm3TrunkMapped *trk,
    const float *h_in,
    float *h_out,
    int T,
    const int *seg_starts
)
{
    const uint8_t *op = (const uint8_t *)trk->operator_mf.ptr;
    uint32_t kind = trk->hdr.layer_kind;
    const int n_thr = g_lfm3.num_threads > 0 ? g_lfm3.num_threads : 8;

    float *op_in = (float *)malloc((size_t)T * LFM3_D * sizeof(float));
    float *op_out_raw = (float *)malloc((size_t)T * LFM3_D * sizeof(float));
    float *op_normed = (float *)malloc((size_t)T * LFM3_D * sizeof(float));
    float *h_mid = (float *)malloc((size_t)T * LFM3_D * sizeof(float));

    if (kind == 0) {
        /* Gated ShortConv (`kind = 0`, width K=7) isolated per segment `seg_starts[t]` */
        uint32_t len_op_norm, len_in_proj, len_conv_w;
        memcpy(&len_op_norm, op + 16, 4);
        memcpy(&len_in_proj, op + 20, 4);
        memcpy(&len_conv_w, op + 24, 4);

        size_t cur = 64;
        const uint16_t *op_norm_w = (const uint16_t *)(op + cur); cur += len_op_norm;
        const uint16_t *in_proj_w = (const uint16_t *)(op + cur); cur += len_in_proj;
        const uint16_t *conv_w = (const uint16_t *)(op + cur); cur += len_conv_w;
        const uint16_t *out_proj_w = (const uint16_t *)get_lfm3_record_ptr(trk, 2);

        rms_norm_bf16(h_in, op_norm_w, op_in, T, LFM3_D, 1e-5f);

        float *bcx = (float *)malloc((size_t)T * 6144 * sizeof(float));
        float *post_gated = (float *)malloc((size_t)T * LFM3_D * sizeof(float));
        linear_bf16_omp(op_in, in_proj_w, bcx, T, 6144, LFM3_D);

        #pragma omp parallel for num_threads(n_thr) schedule(static)
        for (int c = 0; c < LFM3_D; c++) {
            float w_k[7];
            for (int k = 0; k < 7; k++) w_k[k] = bf16_to_f32(conv_w[c * 7 + k]);
            for (int t = 0; t < T; t++) {
                int s0 = seg_starts ? seg_starts[t] : 0;
                float acc = 0.0f;
                for (int k = 0; k < 7; k++) {
                    int ts = t - 6 + k;
                    if (ts >= s0) {
                        float B_val = bcx[(size_t)ts * 6144 + c];
                        float x_val = bcx[(size_t)ts * 6144 + 4096 + c];
                        float pre = bf16_round_f32(B_val * x_val);
                        acc += pre * w_k[k];
                    }
                }
                float conv_c = bf16_round_f32(acc);
                float C_val = bcx[(size_t)t * 6144 + 2048 + c];
                post_gated[(size_t)t * LFM3_D + c] = bf16_round_f32(C_val * conv_c);
            }
        }
        linear_bf16_omp(post_gated, out_proj_w, op_out_raw, T, LFM3_D, LFM3_D);
        free(bcx);
        free(post_gated);
    } else {
        /* Sliding (`kind = 1`) or Full (`kind = 2`) GQA Attention with per-head Sigmoid Gate & segment isolation */
        uint32_t len_op_norm, len_q, len_k, len_v, len_gate, len_qn, len_kn;
        memcpy(&len_op_norm, op + 16, 4);
        memcpy(&len_q, op + 20, 4);
        memcpy(&len_k, op + 24, 4);
        memcpy(&len_v, op + 28, 4);
        memcpy(&len_gate, op + 32, 4);
        memcpy(&len_qn, op + 36, 4);
        memcpy(&len_kn, op + 40, 4);

        size_t cur = 64;
        const uint16_t *op_norm_w = (const uint16_t *)(op + cur); cur += len_op_norm;
        const uint16_t *q_w = (const uint16_t *)(op + cur); cur += len_q;
        const uint16_t *k_w = (const uint16_t *)(op + cur); cur += len_k;
        const uint16_t *v_w = (const uint16_t *)(op + cur); cur += len_v;
        const uint16_t *gate_w = (const uint16_t *)(op + cur); cur += len_gate;
        const uint16_t *qn_w = (const uint16_t *)(op + cur); cur += len_qn;
        const uint16_t *kn_w = (const uint16_t *)(op + cur); cur += len_kn;
        const uint16_t *o_proj_w = (const uint16_t *)get_lfm3_record_ptr(trk, 2);

        rms_norm_bf16(h_in, op_norm_w, op_in, T, LFM3_D, 1e-5f);

        float *q = (float *)malloc((size_t)T * 2048 * sizeof(float));
        float *k = (float *)malloc((size_t)T * 512 * sizeof(float));
        float *v = (float *)malloc((size_t)T * 512 * sizeof(float));
        float *gate_raw = (float *)malloc((size_t)T * 16 * sizeof(float));
        float *gated_attn = (float *)malloc((size_t)T * 2048 * sizeof(float));

        linear_bf16_omp(op_in, q_w, q, T, 2048, LFM3_D);
        linear_bf16_omp(op_in, k_w, k, T, 512, LFM3_D);
        linear_bf16_omp(op_in, v_w, v, T, 512, LFM3_D);
        linear_bf16_omp(op_in, gate_w, gate_raw, T, 16, LFM3_D);

        /* Per-head q_norm and k_norm (dim=128) */
        rms_norm_bf16(q, qn_w, q, T * 16, 128, 1e-5f);
        rms_norm_bf16(k, kn_w, k, T * 4, 128, 1e-5f);

        /* Apply RoPE in float32 using segment-relative position `pos_t = t - seg_starts[t]` */
        const float *inv_freq = (kind == 1) ? g_lfm3.inv_freq_sliding : g_lfm3.inv_freq_full;
        #pragma omp parallel for num_threads(n_thr) schedule(static)
        for (int t = 0; t < T; t++) {
            int pos_t = t - (seg_starts ? seg_starts[t] : 0);
            float cos_t[64], sin_t[64];
            for (int d = 0; d < 64; d++) {
                float angle = (float)pos_t * inv_freq[d];
                cos_t[d] = bf16_round_f32(cosf(angle));
                sin_t[d] = bf16_round_f32(sinf(angle));
            }
            for (int h = 0; h < 16; h++) {
                float *qh = q + (size_t)t * 2048 + h * 128;
                for (int d = 0; d < 64; d++) {
                    float x1 = qh[d];
                    float x2 = qh[d + 64];
                    qh[d] = bf16_round_f32(x1 * cos_t[d] - x2 * sin_t[d]);
                    qh[d + 64] = bf16_round_f32(x2 * cos_t[d] + x1 * sin_t[d]);
                }
            }
            for (int h = 0; h < 4; h++) {
                float *kh = k + (size_t)t * 512 + h * 128;
                for (int d = 0; d < 64; d++) {
                    float x1 = kh[d];
                    float x2 = kh[d + 64];
                    kh[d] = bf16_round_f32(x1 * cos_t[d] - x2 * sin_t[d]);
                    kh[d + 64] = bf16_round_f32(x2 * cos_t[d] + x1 * sin_t[d]);
                }
            }
        }

        const float scale = 1.0f / sqrtf(128.0f);
        #pragma omp parallel for num_threads(n_thr) schedule(static)
        for (int h = 0; h < 16; h++) {
            int kv_h = h / 4;
            float *scores = (float *)malloc((size_t)T * sizeof(float));
            for (int t = 0; t < T; t++) {
                const float *qh = q + (size_t)t * 2048 + h * 128;
                int s0 = seg_starts ? seg_starts[t] : 0;
                int j_start = s0;
                if (kind == 1 && t - LFM3_SLIDING_W + 1 > j_start) {
                    j_start = t - LFM3_SLIDING_W + 1;
                }
                float max_s = -1e30f;
                for (int j = j_start; j <= t; j++) {
                    const float *kh = k + (size_t)j * 512 + kv_h * 128;
                    float dot = 0.0f;
                    for (int d = 0; d < 128; d++) dot += qh[d] * kh[d];
                    scores[j] = bf16_round_f32(dot * scale);
                    if (scores[j] > max_s) max_s = scores[j];
                }
                float sum_exp = 0.0f;
                for (int j = j_start; j <= t; j++) {
                    scores[j] = expf(scores[j] - max_s);
                    sum_exp += scores[j];
                }
                for (int j = j_start; j <= t; j++) {
                    scores[j] = bf16_round_f32(scores[j] / sum_exp);
                }
                float g_val = gate_raw[(size_t)t * 16 + h];
                float head_gate = bf16_round_f32(1.0f / (1.0f + expf(-g_val)));
                float *out_qh = gated_attn + (size_t)t * 2048 + h * 128;
                for (int d = 0; d < 128; d++) {
                    float acc = 0.0f;
                    for (int j = j_start; j <= t; j++) {
                        const float *vh = v + (size_t)j * 512 + kv_h * 128;
                        acc += scores[j] * vh[d];
                    }
                    out_qh[d] = bf16_round_f32(bf16_round_f32(acc) * head_gate);
                }
            }
            free(scores);
        }
        linear_bf16_omp(gated_attn, o_proj_w, op_out_raw, T, LFM3_D, 2048);
        free(q); free(k); free(v); free(gate_raw); free(gated_attn);
    }

    /* Sandwich post_operator_norm + Residual 1 */
    const uint16_t *post_op_norm_w = (const uint16_t *)get_lfm3_record_ptr(trk, 1);
    rms_norm_bf16(op_out_raw, post_op_norm_w, op_normed, T, LFM3_D, 1e-5f);
    for (int i = 0; i < T * LFM3_D; i++) {
        h_mid[i] = bf16_round_f32(h_in[i] + op_normed[i]);
    }

    /* SwiGLU FFN + Sandwich post_ffn_norm + Residual 2 */
    const uint16_t *ffn_norm_w = (const uint16_t *)get_lfm3_record_ptr(trk, 3);
    const uint16_t *post_ffn_norm_w = (const uint16_t *)get_lfm3_record_ptr(trk, 4);

    float *ffn_in = (float *)malloc((size_t)T * LFM3_D * sizeof(float));
    float *w1_out = (float *)malloc((size_t)T * LFM3_FFN * sizeof(float));
    float *w3_out = (float *)malloc((size_t)T * LFM3_FFN * sizeof(float));
    float *ffn_raw = (float *)malloc((size_t)T * LFM3_D * sizeof(float));
    float *ffn_normed = (float *)malloc((size_t)T * LFM3_D * sizeof(float));

    rms_norm_bf16(h_mid, ffn_norm_w, ffn_in, T, LFM3_D, 1e-5f);

    if (g_lfm3.use_4bit_ffn && trk->ffn_4bit_mf.ptr && trk->ffn_4bit_mf.size >= 3 * LFM3_FFN_MAT_4BIT_BYTES) {
        /* 4-Bit Planar Group-64 SwiGLU FFN (`ffn.4bit.bin`: w1 [7680,2048], w3 [7680,2048], w2 [2048,7680]) */
        const uint8_t *base4 = (const uint8_t *)trk->ffn_4bit_mf.ptr;
        const uint8_t *w1_nib = base4;
        const uint8_t *w1_stp = w1_nib + (7680 * 32 * 32);
        const float *w1_scl = (const float *)(w1_stp + (7680 * 32));

        const uint8_t *w3_nib = base4 + LFM3_FFN_MAT_4BIT_BYTES;
        const uint8_t *w3_stp = w3_nib + (7680 * 32 * 32);
        const float *w3_scl = (const float *)(w3_stp + (7680 * 32));

        const uint8_t *w2_nib = base4 + 2 * LFM3_FFN_MAT_4BIT_BYTES;
        const uint8_t *w2_stp = w2_nib + (2048 * 120 * 32);
        const float *w2_scl = (const float *)(w2_stp + (2048 * 120));

        linear_4bit_group64_omp(ffn_in, w1_nib, w1_stp, w1_scl, w1_out, T, LFM3_FFN, LFM3_D);
        linear_4bit_group64_omp(ffn_in, w3_nib, w3_stp, w3_scl, w3_out, T, LFM3_FFN, LFM3_D);

        #pragma omp parallel for num_threads(n_thr) schedule(static)
        for (int i = 0; i < T * LFM3_FFN; i++) {
            float v1 = w1_out[i];
            float silu = bf16_round_f32(v1 / (1.0f + expf(-v1)));
            w1_out[i] = bf16_round_f32(silu * w3_out[i]);
        }
        linear_4bit_group64_omp(w1_out, w2_nib, w2_stp, w2_scl, ffn_raw, T, LFM3_D, LFM3_FFN);
    } else {
        const uint16_t *w1_w = (const uint16_t *)get_lfm3_record_ptr(trk, 10);
        const uint16_t *w3_w = (const uint16_t *)get_lfm3_record_ptr(trk, 11);
        const uint16_t *w2_w = (const uint16_t *)get_lfm3_record_ptr(trk, 12);

        linear_bf16_omp(ffn_in, w1_w, w1_out, T, LFM3_FFN, LFM3_D);
        linear_bf16_omp(ffn_in, w3_w, w3_out, T, LFM3_FFN, LFM3_D);

        #pragma omp parallel for num_threads(n_thr) schedule(static)
        for (int i = 0; i < T * LFM3_FFN; i++) {
            float v1 = w1_out[i];
            float silu = bf16_round_f32(v1 / (1.0f + expf(-v1)));
            w1_out[i] = bf16_round_f32(silu * w3_out[i]);
        }
        linear_bf16_omp(w1_out, w2_w, ffn_raw, T, LFM3_D, LFM3_FFN);
    }
    rms_norm_bf16(ffn_raw, post_ffn_norm_w, ffn_normed, T, LFM3_D, 1e-5f);

    for (int i = 0; i < T * LFM3_D; i++) {
        h_out[i] = bf16_round_f32(h_mid[i] + ffn_normed[i]);
    }

    free(op_in); free(op_out_raw); free(op_normed); free(h_mid);
    free(ffn_in); free(w1_out); free(w3_out); free(ffn_raw); free(ffn_normed);
}

/* Executes all 24 tri-hybrid LFM3-1.7B layers + final `embedding_norm` with optional clubbed segment isolation `seg_starts` */
LFM3_API float lfm3_forward_24_layers_clubbed(const int *token_ids, const int *seg_starts, int T, float *out_final_hidden)
{
    double t0 = now_ms();
    if (g_lfm3.s23_pin_requested) {
        lfm3_configure_s23_snapdragon8gen2(1);
    }

    const uint16_t *emb_w = (const uint16_t *)g_lfm3.emb_mf.ptr;
    const uint16_t *emb_norm_w = (const uint16_t *)g_lfm3.emb_norm_mf.ptr;

    float *cur = (float *)malloc((size_t)T * LFM3_D * sizeof(float));
    float *nxt = (float *)malloc((size_t)T * LFM3_D * sizeof(float));

    for (int t = 0; t < T; t++) {
        int tid = token_ids[t];
        const uint16_t *erow = emb_w + (size_t)tid * LFM3_D;
        float *dst = cur + (size_t)t * LFM3_D;
        for (int d = 0; d < LFM3_D; d++) dst[d] = bf16_to_f32(erow[d]);
    }

    for (int l = 0; l < LFM3_LAYERS; l++) {
        lfm3_dm_record_layer(l, T);
        execute_single_lfm3_layer_c(&g_lfm3.trunks[l], cur, nxt, T, seg_starts);
        float *tmp = cur;
        cur = nxt;
        nxt = tmp;
    }

    rms_norm_bf16(cur, emb_norm_w, out_final_hidden, T, LFM3_D, 1e-5f);

    free(cur);
    free(nxt);
    return (float)(now_ms() - t0);
}

LFM3_API float lfm3_forward_24_layers(const int *token_ids, int T, float *out_final_hidden)
{
    return lfm3_forward_24_layers_clubbed(token_ids, NULL, T, out_final_hidden);
}

/* Computes tied `128,000`-vocab `lm_head` logits `out_logits[128000]` and returns `top1_id`
 * directly from the `mmap`ped `embed_tokens.bf16.bin` with zero extra weight memory allocation! */
LFM3_API int lfm3_lm_head_logits(const float *h_last_2048, float *out_logits_128k, float *out_max_logit)
{
    if (!g_lfm3.initialized || !g_lfm3.emb_mf.ptr || !h_last_2048) return -1;
    const uint16_t *emb_w = (const uint16_t *)g_lfm3.emb_mf.ptr;
    const int vocab_sz = 128000;
    float *lg = out_logits_128k ? out_logits_128k : (float *)malloc((size_t)vocab_sz * sizeof(float));
    linear_bf16_omp(h_last_2048, emb_w, lg, 1, vocab_sz, LFM3_D);
    int best_id = 0;
    float best_v = lg[0];
    for (int v = 1; v < vocab_sz; v++) {
        if (lg[v] > best_v) {
            best_v = lg[v];
            best_id = v;
        }
    }
    if (out_max_logit) *out_max_logit = best_v;
    if (!out_logits_128k) free(lg);
    return best_id;
}
