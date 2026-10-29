#define _GNU_SOURCE
/*
 * Native C / OpenMP CPU Engine for NVIDIA Nemotron 3.5 Super VL (`libnemotron_cpu_engine.so`)
 * (`clover-ai-system/src/models/nemotron-3.5-super-vl/nemotron_cpu_engine.c`)
 *
 * Implements the native C/OpenMP + mmap execution engine for Nemotron 3.5 Super VL's
 * pruned `NEMOTRK1` trunks, `NEMO_OP1` composed `operator.bin` files, and `NEMO_EXPERTS_DIRECT_V1`
 * `experts.direct` containers (`512` non-gated ReLU^2 experts/layer, `11,010,048` B/expert),
 * integrated with Clover's SpecMD Policy 3 (Least-Stale + R4 Clean-Drop) SlotCache (`libslotcache.so`
 * or built-in C fallback) and 4-Tier Data-Movement Ledger.
 */

#include <dlfcn.h>
#include <fcntl.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

enum {
    NEMO_D = 4096,
    NEMO_EXPERTS = 512,
    NEMO_TOP_K = 22,
    NEMO_LATENT = 1024,
    NEMO_MOE_INTER = 2688,
    NEMO_SH_INTER = 5376,
    NEMO_MAMBA_IN = 18560,
    NEMO_MAMBA_CONV = 10240,
    NEMO_MAMBA_INNER = 8192,
    NEMO_MAMBA_HEADS = 128,
    NEMO_MAMBA_HDIM = 64,
    NEMO_MAMBA_NGROUPS = 8,
    NEMO_MAMBA_DSTATE = 128,
    NEMO_GQA_Q = 4096,
    NEMO_GQA_KV = 256,
    NEMO_GQA_HEADS = 32,
    NEMO_GQA_KV_HEADS = 2,
    NEMO_GQA_HDIM = 128
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
} MmapFile;

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

static void mmap_close(MmapFile *mf)
{
    if (mf->ptr && mf->size > 0) munmap(mf->ptr, mf->size);
    if (mf->fd >= 0) close(mf->fd);
    mf->ptr = NULL;
    mf->size = 0;
    mf->fd = -1;
}

enum {
    NEMO_HEADER_BYTES = 1168,
    NEMO_RECORD_BYTES = 72
};

typedef struct {
    uint32_t rec_id;
    uint32_t grp_id;
    uint32_t dtype_code;
    uint32_t rows;
    uint32_t cols;
    uint32_t numel;
    uint64_t offset;
    uint64_t length;
    uint32_t crc;
} NemoRecordDesc;

typedef struct {
    MmapFile index_mf;
    MmapFile group_mf[3];
    MmapFile operator_mf;
    uint32_t version;
    uint32_t layer_id;
    uint32_t record_count;
    uint32_t layer_kind;
    uint64_t group_lengths[3];
    NemoRecordDesc recs[32];
} NemoTrunkMapped;

static int load_nemo_trunk(const char *trunk_dir, NemoTrunkMapped *t)
{
    memset(t, 0, sizeof(*t));
    t->index_mf.fd = -1;
    t->operator_mf.fd = -1;
    for (int g = 0; g < 3; g++) t->group_mf[g].fd = -1;

    char path[1024];
    snprintf(path, sizeof(path), "%s/index.bin", trunk_dir);
    if (mmap_readonly(path, &t->index_mf) != 0 || t->index_mf.size < NEMO_HEADER_BYTES) return -1;
    const uint8_t *hdr = (const uint8_t *)t->index_mf.ptr;
    if (memcmp(hdr, "NEMOTRK1", 8) != 0) return -2;

    memcpy(&t->version, hdr + 8, 4);
    memcpy(&t->layer_id, hdr + 12, 4);
    memcpy(&t->record_count, hdr + 16, 4);
    memcpy(t->group_lengths, hdr + 24, 24);
    memcpy(&t->layer_kind, hdr + 48, 4);

    uint32_t nrec = t->record_count < 32 ? t->record_count : 32;
    for (uint32_t i = 0; i < nrec; i++) {
        const uint8_t *rp = hdr + NEMO_HEADER_BYTES + (size_t)i * NEMO_RECORD_BYTES;
        memcpy(&t->recs[i].rec_id, rp + 0, 4);
        memcpy(&t->recs[i].grp_id, rp + 4, 4);
        memcpy(&t->recs[i].dtype_code, rp + 8, 4);
        memcpy(&t->recs[i].rows, rp + 12, 4);
        memcpy(&t->recs[i].cols, rp + 16, 4);
        memcpy(&t->recs[i].numel, rp + 20, 4);
        memcpy(&t->recs[i].offset, rp + 24, 8);
        memcpy(&t->recs[i].length, rp + 32, 8);
        memcpy(&t->recs[i].crc, rp + 40, 4);
    }

    const char *gnames[3] = { "common.bin", "attention.bin", "routed-shared.bin" };
    for (int g = 0; g < 3; g++) {
        snprintf(path, sizeof(path), "%s/%s", trunk_dir, gnames[g]);
        if (access(path, R_OK) == 0) {
            if (mmap_readonly(path, &t->group_mf[g]) != 0) return -3;
        }
    }
    snprintf(path, sizeof(path), "%s/operator.bin", trunk_dir);
    if (access(path, R_OK) == 0) {
        if (mmap_readonly(path, &t->operator_mf) != 0) return -4;
    }
    return 0;
}

static const void *get_nemo_record_ptr(const NemoTrunkMapped *t, uint32_t rec_id)
{
    for (uint32_t i = 0; i < t->record_count && i < 32; i++) {
        if (t->recs[i].rec_id == rec_id) {
            if (t->recs[i].length == 0) return NULL;
            uint32_t g = t->recs[i].grp_id;
            if (g >= 3 || !t->group_mf[g].ptr) return NULL;
            return (const uint8_t *)t->group_mf[g].ptr + t->recs[i].offset;
        }
    }
    return NULL;
}

/* SpecMD Policy 3 (Least-Stale + R4 Clean-Drop) Slot Cache + Data-Movement Ledger */
static struct {
    int initialized;
    MmapFile emb_mf;
    MmapFile norm_f_mf;
    NemoTrunkMapped trunk0_mamba;
    NemoTrunkMapped trunk1_moe;
    MmapFile expert1_direct_mf;
    NemoTrunkMapped trunk7_gqa;

    unsigned long clock;
    int pool_keys[64];
    unsigned long pool_ts[64];
    float pool_heat[64];
    int host_keys[512];

    unsigned long total_requests;
    unsigned long tier1_pool_hits;
    unsigned long tier2_host_hits;
    unsigned long tier3_drive_fetches;
    unsigned long r4_clean_drops;
    unsigned long dirty_writebacks;
} g_nemo = { 0 };

static void nemo_slotcache_reset(void)
{
    g_nemo.clock = 0;
    for (int i = 0; i < 64; i++) {
        g_nemo.pool_keys[i] = -1;
        g_nemo.pool_ts[i] = 0;
        g_nemo.pool_heat[i] = 0.0f;
    }
    for (int i = 0; i < 512; i++) g_nemo.host_keys[i] = 0;
    g_nemo.total_requests = 0;
    g_nemo.tier1_pool_hits = 0;
    g_nemo.tier2_host_hits = 0;
    g_nemo.tier3_drive_fetches = 0;
    g_nemo.r4_clean_drops = 0;
    g_nemo.dirty_writebacks = 0;
}

static void nemo_slotcache_acquire(int exp_id, const uint8_t *protected_mask, float weight)
{
    g_nemo.clock++;
    g_nemo.total_requests++;
    for (int i = 0; i < 64; i++) {
        if (g_nemo.pool_keys[i] == exp_id) {
            g_nemo.tier1_pool_hits++;
            g_nemo.pool_ts[i] = g_nemo.clock;
            g_nemo.pool_heat[i] = 0.75f * g_nemo.pool_heat[i] + weight;
            return;
        }
    }
    if (g_nemo.host_keys[exp_id]) {
        g_nemo.tier2_host_hits++;
    } else {
        g_nemo.tier3_drive_fetches++;
        g_nemo.host_keys[exp_id] = 1;
    }
    /* Find free slot or Least-Stale victim */
    int slot = -1;
    for (int i = 0; i < 64; i++) {
        if (g_nemo.pool_keys[i] < 0) {
            slot = i;
            break;
        }
    }
    if (slot < 0) {
        int best_v = 0;
        double best_score = 1e30;
        for (int i = 0; i < 64; i++) {
            int k = g_nemo.pool_keys[i];
            if (k >= 0 && k < NEMO_EXPERTS && protected_mask[k]) continue;
            unsigned long stale = g_nemo.clock > g_nemo.pool_ts[i] ? (g_nemo.clock - g_nemo.pool_ts[i]) : 1UL;
            double sc = (double)g_nemo.pool_heat[i] / (double)(1UL + stale);
            if (sc < best_score) {
                best_score = sc;
                best_v = i;
            }
        }
        slot = best_v;
        int victim = g_nemo.pool_keys[slot];
        if (victim >= 0 && g_nemo.host_keys[victim]) {
            g_nemo.r4_clean_drops++;
        } else {
            g_nemo.dirty_writebacks++;
        }
    }
    g_nemo.pool_keys[slot] = exp_id;
    g_nemo.pool_ts[slot] = g_nemo.clock;
    g_nemo.pool_heat[slot] = weight;
}

/* OpenMP BF16 Linear with 8-Token Register Blocking:
 * Streams each weight row `w_row[in_dim]` ONCE across up to 8 tokens simultaneously,
 * cutting DDR5 memory bandwidth traffic by up to 8x! */
static void linear_bf16_omp(
    const float *x,
    const uint16_t *W_bf16,
    float *y,
    int T,
    int out_dim,
    int in_dim
)
{
    #pragma omp parallel for schedule(static)
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
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            float s4 = 0.0f, s5 = 0.0f, s6 = 0.0f, s7 = 0.0f;
            for (int k = 0; k < in_dim; k++) {
                float w = bf16_to_f32(w_row[k]);
                s0 += x0[k] * w;
                s1 += x1[k] * w;
                s2 += x2[k] * w;
                s3 += x3[k] * w;
                s4 += x4[k] * w;
                s5 += x5[k] * w;
                s6 += x6[k] * w;
                s7 += x7[k] * w;
            }
            y[(size_t)(t + 0) * out_dim + o] = bf16_round_f32(s0);
            y[(size_t)(t + 1) * out_dim + o] = bf16_round_f32(s1);
            y[(size_t)(t + 2) * out_dim + o] = bf16_round_f32(s2);
            y[(size_t)(t + 3) * out_dim + o] = bf16_round_f32(s3);
            y[(size_t)(t + 4) * out_dim + o] = bf16_round_f32(s4);
            y[(size_t)(t + 5) * out_dim + o] = bf16_round_f32(s5);
            y[(size_t)(t + 6) * out_dim + o] = bf16_round_f32(s6);
            y[(size_t)(t + 7) * out_dim + o] = bf16_round_f32(s7);
        }
        for (; t < T; t++) {
            const float *x_row = x + (size_t)t * (size_t)in_dim;
            float sum = 0.0f;
            for (int k = 0; k < in_dim; k++) {
                sum += x_row[k] * bf16_to_f32(w_row[k]);
            }
            y[(size_t)t * (size_t)out_dim + (size_t)o] = bf16_round_f32(sum);
        }
    }
}

static void rms_norm_bf16(
    const float *x,
    const uint16_t *w_bf16,
    float *y,
    int T,
    int dim,
    float eps
)
{
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++) {
        const float *xr = x + (size_t)t * dim;
        float *yr = y + (size_t)t * dim;
        double ss = 0.0;
        for (int d = 0; d < dim; d++) ss += (double)xr[d] * (double)xr[d];
        float inv = 1.0f / sqrtf((float)(ss / (double)dim) + eps);
        for (int d = 0; d < dim; d++) {
            float normed_bf16 = bf16_round_f32(xr[d] * inv);
            yr[d] = bf16_round_f32(normed_bf16 * bf16_to_f32(w_bf16[d]));
        }
    }
}

int nemo_engine_init(const char *pruned_dataset_dir)
{
    if (g_nemo.initialized) return 0;
    char path[1024];
    snprintf(path, sizeof(path), "%s/global-0/embeddings.bf16.bin", pruned_dataset_dir);
    if (mmap_readonly(path, &g_nemo.emb_mf) != 0) return -1;

    snprintf(path, sizeof(path), "%s/global-0/norm_f.bf16.bin", pruned_dataset_dir);
    if (mmap_readonly(path, &g_nemo.norm_f_mf) != 0) return -2;

    snprintf(path, sizeof(path), "%s/trunk-0", pruned_dataset_dir);
    if (load_nemo_trunk(path, &g_nemo.trunk0_mamba) != 0) return -3;

    snprintf(path, sizeof(path), "%s/trunk-1", pruned_dataset_dir);
    if (load_nemo_trunk(path, &g_nemo.trunk1_moe) != 0) return -4;

    snprintf(path, sizeof(path), "%s/expert-1/experts.direct", pruned_dataset_dir);
    if (mmap_readonly(path, &g_nemo.expert1_direct_mf) != 0) return -5;

    snprintf(path, sizeof(path), "%s/trunk-7", pruned_dataset_dir);
    if (load_nemo_trunk(path, &g_nemo.trunk7_gqa) != 0) return -6;

    nemo_slotcache_reset();
    g_nemo.initialized = 1;
    return 0;
}

/* Executes Stage L1: Pruned `trunk-0` + `operator.bin` (Mamba-2) with optional clubbed segment boundaries `seg_starts` */
void nemo_mamba_layer_forward_seg(const float *h_in, float *h_out, int T, const int *seg_starts)
{
    const uint8_t *op = (const uint8_t *)g_nemo.trunk0_mamba.operator_mf.ptr;
    uint32_t len_norm, len_in, len_cw, len_cb, len_dtb, len_A, len_D;
    memcpy(&len_norm, op + 16, 4);
    memcpy(&len_in, op + 20, 4);
    memcpy(&len_cw, op + 24, 4);
    memcpy(&len_cb, op + 28, 4);
    memcpy(&len_dtb, op + 32, 4);
    memcpy(&len_A, op + 36, 4);
    memcpy(&len_D, op + 40, 4);

    size_t cur = 64;
    const uint16_t *norm_w = (const uint16_t *)(op + cur); cur += len_norm;
    const uint16_t *in_proj_w = (const uint16_t *)(op + cur); cur += len_in;
    const uint16_t *conv_w = (const uint16_t *)(op + cur); cur += len_cw;
    const uint16_t *conv_b = (const uint16_t *)(op + cur); cur += len_cb;
    const uint16_t *dt_bias = (const uint16_t *)(op + cur); cur += len_dtb;
    const float *A_precomp = (const float *)(op + cur); cur += len_A;
    const uint16_t *D_vec = (const uint16_t *)(op + cur); cur += len_D;

    const uint16_t *mixer_norm_w = (const uint16_t *)get_nemo_record_ptr(&g_nemo.trunk0_mamba, 1);
    const uint16_t *out_proj_w = (const uint16_t *)get_nemo_record_ptr(&g_nemo.trunk0_mamba, 2);

    float *u = (float *)malloc((size_t)T * NEMO_D * sizeof(float));
    float *proj = (float *)malloc((size_t)T * NEMO_MAMBA_IN * sizeof(float));
    float *conv_act = (float *)malloc((size_t)T * NEMO_MAMBA_CONV * sizeof(float));
    float *y_ssm = (float *)malloc((size_t)T * NEMO_MAMBA_INNER * sizeof(float));
    float *scan_out = (float *)malloc((size_t)T * NEMO_MAMBA_INNER * sizeof(float));
    float *branch_out = (float *)malloc((size_t)T * NEMO_D * sizeof(float));

    rms_norm_bf16(h_in, norm_w, u, T, NEMO_D, 1e-5f);
    linear_bf16_omp(u, in_proj_w, proj, T, NEMO_MAMBA_IN, NEMO_D);

    /* Causal depthwise conv1d (width 4) + SiLU on channels [8192 .. 8192+10240), isolated per segment `seg_starts[t]` */
    #pragma omp parallel for schedule(static)
    for (int c = 0; c < NEMO_MAMBA_CONV; c++) {
        float w0 = bf16_to_f32(conv_w[c * 4 + 0]);
        float w1 = bf16_to_f32(conv_w[c * 4 + 1]);
        float w2 = bf16_to_f32(conv_w[c * 4 + 2]);
        float w3 = bf16_to_f32(conv_w[c * 4 + 3]);
        float bias = bf16_to_f32(conv_b[c]);
        for (int t = 0; t < T; t++) {
            int s0 = seg_starts ? seg_starts[t] : 0;
            float x0 = (t - 3 >= s0) ? proj[(size_t)(t - 3) * NEMO_MAMBA_IN + 8192 + c] : 0.0f;
            float x1 = (t - 2 >= s0) ? proj[(size_t)(t - 2) * NEMO_MAMBA_IN + 8192 + c] : 0.0f;
            float x2 = (t - 1 >= s0) ? proj[(size_t)(t - 1) * NEMO_MAMBA_IN + 8192 + c] : 0.0f;
            float x3 = proj[(size_t)t * NEMO_MAMBA_IN + 8192 + c];
            float v = bf16_round_f32(x0 * w0 + x1 * w1 + x2 * w2 + x3 * w3 + bias);
            float silu = v / (1.0f + expf(-v));
            conv_act[(size_t)t * NEMO_MAMBA_CONV + c] = bf16_round_f32(silu);
        }
    }

    /* Selective State Space recurrence across 128 heads, resetting state at each clubbed segment start */
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < NEMO_MAMBA_HEADS; h++) {
        int grp = h / 16;
        float A_h = A_precomp[h];
        float D_h = bf16_to_f32(D_vec[h]);
        float dt_b = bf16_to_f32(dt_bias[h]);
        float state[NEMO_MAMBA_HDIM * NEMO_MAMBA_DSTATE];
        memset(state, 0, sizeof(state));

        for (int t = 0; t < T; t++) {
            if (seg_starts && t == seg_starts[t]) {
                memset(state, 0, sizeof(state));
            }
            float dt_raw = proj[(size_t)t * NEMO_MAMBA_IN + 18432 + h] + dt_b;
            float dt_val = log1pf(expf(dt_raw));
            if (dt_val < 0.001f) dt_val = 0.001f;
            dt_val = bf16_round_f32(dt_val);
            float dA = expf(dt_val * A_h);

            const float *x_vec = conv_act + (size_t)t * NEMO_MAMBA_CONV + h * NEMO_MAMBA_HDIM;
            const float *B_vec = conv_act + (size_t)t * NEMO_MAMBA_CONV + 8192 + grp * NEMO_MAMBA_DSTATE;
            const float *C_vec = conv_act + (size_t)t * NEMO_MAMBA_CONV + 9216 + grp * NEMO_MAMBA_DSTATE;
            float *y_vec = y_ssm + (size_t)t * NEMO_MAMBA_INNER + h * NEMO_MAMBA_HDIM;

            for (int d = 0; d < NEMO_MAMBA_HDIM; d++) {
                float xd = x_vec[d];
                float acc = D_h * xd;
                float *st_row = state + d * NEMO_MAMBA_DSTATE;
                for (int s = 0; s < NEMO_MAMBA_DSTATE; s++) {
                    st_row[s] = st_row[s] * dA + (dt_val * B_vec[s]) * xd;
                    acc += st_row[s] * C_vec[s];
                }
                y_vec[d] = acc;
            }
        }
    }

    /* Group-RMSNormGated (8 groups of 1024 channels) */
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++) {
        const float *gate_row = proj + (size_t)t * NEMO_MAMBA_IN;
        const float *y_row = y_ssm + (size_t)t * NEMO_MAMBA_INNER;
        float *s_row = scan_out + (size_t)t * NEMO_MAMBA_INNER;
        for (int g = 0; g < 8; g++) {
            double ss = 0.0;
            for (int i = 0; i < 1024; i++) {
                int idx = g * 1024 + i;
                float gt = gate_row[idx];
                float silu_g = gt / (1.0f + expf(-gt));
                float gated = y_row[idx] * silu_g;
                s_row[idx] = gated;
                ss += (double)gated * (double)gated;
            }
            float inv = 1.0f / sqrtf((float)(ss / 1024.0) + 1e-5f);
            for (int i = 0; i < 1024; i++) {
                int idx = g * 1024 + i;
                s_row[idx] = bf16_round_f32((s_row[idx] * inv) * bf16_to_f32(mixer_norm_w[idx]));
            }
        }
    }

    linear_bf16_omp(scan_out, out_proj_w, branch_out, T, NEMO_D, NEMO_MAMBA_INNER);
    for (int i = 0; i < T * NEMO_D; i++) {
        h_out[i] = bf16_round_f32(h_in[i] + branch_out[i]);
    }

    free(u); free(proj); free(conv_act); free(y_ssm); free(scan_out); free(branch_out);
}

/* Executes Stage L2: Pruned `trunk-1` + `expert-1/experts.direct` (Latent ReLU^2 MoE with SpecMD R4 SlotCache) */
void nemo_latent_moe_layer_forward(const float *h_in, float *h_out, int T)
{
    const uint16_t *norm_w = (const uint16_t *)get_nemo_record_ptr(&g_nemo.trunk1_moe, 0);
    const float *gate_w_fp32 = (const float *)get_nemo_record_ptr(&g_nemo.trunk1_moe, 29);
    const float *gate_bias_fp32 = (const float *)get_nemo_record_ptr(&g_nemo.trunk1_moe, 30);
    const uint16_t *fc1_w = (const uint16_t *)get_nemo_record_ptr(&g_nemo.trunk1_moe, 31);
    const uint16_t *fc2_w = (const uint16_t *)get_nemo_record_ptr(&g_nemo.trunk1_moe, 32);
    const uint16_t *sh_up_w = (const uint16_t *)get_nemo_record_ptr(&g_nemo.trunk1_moe, 34);
    const uint16_t *sh_down_w = (const uint16_t *)get_nemo_record_ptr(&g_nemo.trunk1_moe, 36);

    float *u = (float *)malloc((size_t)T * NEMO_D * sizeof(float));
    float *latent_in = (float *)malloc((size_t)T * NEMO_LATENT * sizeof(float));
    float *latent_acc = (float *)calloc((size_t)T * NEMO_LATENT, sizeof(float));
    float *routed_4096 = (float *)malloc((size_t)T * NEMO_D * sizeof(float));
    float *sh_inter = (float *)malloc((size_t)T * NEMO_SH_INTER * sizeof(float));
    float *shared_4096 = (float *)malloc((size_t)T * NEMO_D * sizeof(float));

    rms_norm_bf16(h_in, norm_w, u, T, NEMO_D, 1e-5f);
    linear_bf16_omp(u, fc1_w, latent_in, T, NEMO_LATENT, NEMO_D);

    int *topk_idx = (int *)malloc((size_t)T * NEMO_TOP_K * sizeof(int));
    float *topk_w = (float *)malloc((size_t)T * NEMO_TOP_K * sizeof(float));
    uint8_t active_mask[NEMO_EXPERTS];
    float expert_weight_sum[NEMO_EXPERTS];
    memset(active_mask, 0, sizeof(active_mask));
    memset(expert_weight_sum, 0, sizeof(expert_weight_sum));

    /* Router: FP32 dot product + sigmoid + bias + top-22 */
    for (int t = 0; t < T; t++) {
        const float *u_row = u + (size_t)t * NEMO_D;
        float scores[NEMO_EXPERTS];
        float choice[NEMO_EXPERTS];
        for (int e = 0; e < NEMO_EXPERTS; e++) {
            const float *gw = gate_w_fp32 + (size_t)e * NEMO_D;
            float dot = 0.0f;
            for (int d = 0; d < NEMO_D; d++) dot += u_row[d] * gw[d];
            float s = 1.0f / (1.0f + expf(-dot));
            scores[e] = s;
            choice[e] = s + gate_bias_fp32[e];
        }
        float w_sum = 0.0f;
        for (int k = 0; k < NEMO_TOP_K; k++) {
            int best_e = 0;
            float best_val = -1e30f;
            for (int e = 0; e < NEMO_EXPERTS; e++) {
                if (choice[e] > best_val) {
                    best_val = choice[e];
                    best_e = e;
                }
            }
            choice[best_e] = -1e30f;
            topk_idx[t * NEMO_TOP_K + k] = best_e;
            topk_w[t * NEMO_TOP_K + k] = scores[best_e];
            w_sum += scores[best_e];
            active_mask[best_e] = 1;
        }
        for (int k = 0; k < NEMO_TOP_K; k++) {
            topk_w[t * NEMO_TOP_K + k] = (topk_w[t * NEMO_TOP_K + k] / (w_sum + 1e-20f)) * 5.0f;
            expert_weight_sum[topk_idx[t * NEMO_TOP_K + k]] += topk_w[t * NEMO_TOP_K + k];
        }
    }

    /* SpecMD Hit-First Intra-Layer Ordering on the SlotCache */
    for (int e = 0; e < NEMO_EXPERTS; e++) {
        if (!active_mask[e]) continue;
        int in_pool = 0;
        for (int s = 0; s < 64; s++) if (g_nemo.pool_keys[s] == e) { in_pool = 1; break; }
        if (in_pool) nemo_slotcache_acquire(e, active_mask, expert_weight_sum[e]);
    }
    for (int e = 0; e < NEMO_EXPERTS; e++) {
        if (!active_mask[e]) continue;
        int in_pool = 0;
        for (int s = 0; s < 64; s++) if (g_nemo.pool_keys[s] == e) { in_pool = 1; break; }
        if (!in_pool) nemo_slotcache_acquire(e, active_mask, expert_weight_sum[e]);
    }

    /* Execute only the active routed experts on-demand from `experts.direct`.
     * Clubbed-batch optimization: gather all `n_m` tokens routed to expert `e` and call
     * `linear_bf16_omp` ONCE per active expert `e` (streaming `up_w` and `down_w` once across
     * all `n_m` tokens with 8-token register blocking instead of re-reading `up_w`/`down_w` per token!). */
    const size_t up_elems = (size_t)NEMO_MOE_INTER * NEMO_LATENT;
    const size_t stride_bytes = (size_t)11010048;
    const uint8_t *direct_base = (const uint8_t *)g_nemo.expert1_direct_mf.ptr;

    int *m_toks = (int *)malloc((size_t)T * sizeof(int));
    float *m_rw = (float *)malloc((size_t)T * sizeof(float));
    float *g_in = (float *)malloc((size_t)T * NEMO_LATENT * sizeof(float));
    float *g_inter = (float *)malloc((size_t)T * NEMO_MOE_INTER * sizeof(float));
    float *g_out = (float *)malloc((size_t)T * NEMO_LATENT * sizeof(float));

    for (int e = 0; e < NEMO_EXPERTS; e++) {
        if (!active_mask[e]) continue;
        int n_m = 0;
        for (int t = 0; t < T; t++) {
            for (int k = 0; k < NEMO_TOP_K; k++) {
                if (topk_idx[t * NEMO_TOP_K + k] == e) {
                    m_toks[n_m] = t;
                    m_rw[n_m] = topk_w[t * NEMO_TOP_K + k];
                    memcpy(g_in + (size_t)n_m * NEMO_LATENT, latent_in + (size_t)t * NEMO_LATENT, NEMO_LATENT * sizeof(float));
                    n_m++;
                    break;
                }
            }
        }
        if (n_m == 0) continue;

        const uint16_t *up_w = (const uint16_t *)(direct_base + (size_t)e * stride_bytes);
        const uint16_t *down_w = up_w + up_elems;

        linear_bf16_omp(g_in, up_w, g_inter, n_m, NEMO_MOE_INTER, NEMO_LATENT);
        for (int i = 0; i < n_m * NEMO_MOE_INTER; i++) {
            float r = g_inter[i] > 0.0f ? g_inter[i] : 0.0f;
            g_inter[i] = bf16_round_f32(r * r);
        }
        linear_bf16_omp(g_inter, down_w, g_out, n_m, NEMO_LATENT, NEMO_MOE_INTER);

        for (int idx_m = 0; idx_m < n_m; idx_m++) {
            int t = m_toks[idx_m];
            float rw = m_rw[idx_m];
            float *lacc = latent_acc + (size_t)t * NEMO_LATENT;
            const float *go = g_out + (size_t)idx_m * NEMO_LATENT;
            for (int d = 0; d < NEMO_LATENT; d++) {
                lacc[d] += go[d] * rw;
            }
        }
    }

    free(m_toks); free(m_rw); free(g_in); free(g_inter); free(g_out);

    for (int i = 0; i < T * NEMO_LATENT; i++) latent_acc[i] = bf16_round_f32(latent_acc[i]);
    linear_bf16_omp(latent_acc, fc2_w, routed_4096, T, NEMO_D, NEMO_LATENT);

    /* Shared expert (`4096 -> 5376 -> 4096`, ReLU^2) */
    linear_bf16_omp(u, sh_up_w, sh_inter, T, NEMO_SH_INTER, NEMO_D);
    for (int i = 0; i < T * NEMO_SH_INTER; i++) {
        float r = sh_inter[i] > 0.0f ? sh_inter[i] : 0.0f;
        sh_inter[i] = bf16_round_f32(r * r);
    }
    linear_bf16_omp(sh_inter, sh_down_w, shared_4096, T, NEMO_D, NEMO_SH_INTER);

    for (int i = 0; i < T * NEMO_D; i++) {
        float comb = bf16_round_f32(routed_4096[i] + shared_4096[i]);
        h_out[i] = bf16_round_f32(h_in[i] + comb);
    }

    free(u); free(latent_in); free(latent_acc); free(routed_4096);
    free(sh_inter); free(shared_4096); free(topk_idx); free(topk_w);
}

/* Executes Stage L3: Pruned `trunk-7` + `operator.bin` (NoPE 16:1 GQA Attention) with segment isolation */
void nemo_gqa_layer_forward_seg(const float *h_in, float *h_out, int T, const int *seg_starts)
{
    const uint8_t *op = (const uint8_t *)g_nemo.trunk7_gqa.operator_mf.ptr;
    uint32_t len_norm, len_q, len_k, len_v;
    memcpy(&len_norm, op + 16, 4);
    memcpy(&len_q, op + 20, 4);
    memcpy(&len_k, op + 24, 4);
    memcpy(&len_v, op + 28, 4);

    size_t cur = 64;
    const uint16_t *norm_w = (const uint16_t *)(op + cur); cur += len_norm;
    const uint16_t *q_w = (const uint16_t *)(op + cur); cur += len_q;
    const uint16_t *k_w = (const uint16_t *)(op + cur); cur += len_k;
    const uint16_t *v_w = (const uint16_t *)(op + cur); cur += len_v;
    const uint16_t *o_w = (const uint16_t *)get_nemo_record_ptr(&g_nemo.trunk7_gqa, 1);

    float *u = (float *)malloc((size_t)T * NEMO_D * sizeof(float));
    float *q = (float *)malloc((size_t)T * NEMO_GQA_Q * sizeof(float));
    float *k = (float *)malloc((size_t)T * NEMO_GQA_KV * sizeof(float));
    float *v = (float *)malloc((size_t)T * NEMO_GQA_KV * sizeof(float));
    float *attn_out = (float *)malloc((size_t)T * NEMO_GQA_Q * sizeof(float));
    float *branch_out = (float *)malloc((size_t)T * NEMO_D * sizeof(float));

    rms_norm_bf16(h_in, norm_w, u, T, NEMO_D, 1e-5f);
    linear_bf16_omp(u, q_w, q, T, NEMO_GQA_Q, NEMO_D);
    linear_bf16_omp(u, k_w, k, T, NEMO_GQA_KV, NEMO_D);
    linear_bf16_omp(u, v_w, v, T, NEMO_GQA_KV, NEMO_D);

    const float scale = 1.0f / sqrtf((float)NEMO_GQA_HDIM);
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < NEMO_GQA_HEADS; h++) {
        int kv_h = h / 16;
        float *scores = (float *)malloc((size_t)T * sizeof(float));
        for (int t = 0; t < T; t++) {
            int j_start = seg_starts ? seg_starts[t] : 0;
            const float *qh = q + (size_t)t * NEMO_GQA_Q + h * NEMO_GQA_HDIM;
            float max_s = -1e30f;
            for (int j = j_start; j <= t; j++) {
                const float *kh = k + (size_t)j * NEMO_GQA_KV + kv_h * NEMO_GQA_HDIM;
                float dot = 0.0f;
                for (int d = 0; d < NEMO_GQA_HDIM; d++) dot += qh[d] * kh[d];
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
            float *out_h = attn_out + (size_t)t * NEMO_GQA_Q + h * NEMO_GQA_HDIM;
            for (int d = 0; d < NEMO_GQA_HDIM; d++) {
                float acc = 0.0f;
                for (int j = j_start; j <= t; j++) {
                    const float *vh = v + (size_t)j * NEMO_GQA_KV + kv_h * NEMO_GQA_HDIM;
                    acc += scores[j] * vh[d];
                }
                out_h[d] = bf16_round_f32(acc);
            }
        }
        free(scores);
    }

    linear_bf16_omp(attn_out, o_w, branch_out, T, NEMO_D, NEMO_GQA_Q);
    for (int i = 0; i < T * NEMO_D; i++) {
        h_out[i] = bf16_round_f32(h_in[i] + branch_out[i]);
    }

    free(u); free(q); free(k); free(v); free(attn_out); free(branch_out);
}

/* Executes full Nemotron C/OpenMP pipeline with optional clubbed segment isolation `seg_starts` */
float nemo_forward_pipeline_clubbed(const int *token_ids, const int *seg_starts, int T, float *out_final_hidden)
{
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    const uint16_t *emb_w = (const uint16_t *)g_nemo.emb_mf.ptr;
    const uint16_t *norm_f_w = (const uint16_t *)g_nemo.norm_f_mf.ptr;

    float *h0 = (float *)malloc((size_t)T * NEMO_D * sizeof(float));
    float *h1 = (float *)malloc((size_t)T * NEMO_D * sizeof(float));
    float *h2 = (float *)malloc((size_t)T * NEMO_D * sizeof(float));
    float *h3 = (float *)malloc((size_t)T * NEMO_D * sizeof(float));

    for (int t = 0; t < T; t++) {
        int tid = token_ids[t];
        const uint16_t *erow = emb_w + (size_t)tid * NEMO_D;
        float *dst = h0 + (size_t)t * NEMO_D;
        for (int d = 0; d < NEMO_D; d++) dst[d] = bf16_to_f32(erow[d]);
    }

    nemo_mamba_layer_forward_seg(h0, h1, T, seg_starts);
    nemo_latent_moe_layer_forward(h1, h2, T);
    nemo_gqa_layer_forward_seg(h2, h3, T, seg_starts);
    rms_norm_bf16(h3, norm_f_w, out_final_hidden, T, NEMO_D, 1e-5f);

    free(h0); free(h1); free(h2); free(h3);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    return (float)((t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6);
}

float nemo_forward_pipeline(const int *token_ids, int T, float *out_final_hidden)
{
    return nemo_forward_pipeline_clubbed(token_ids, NULL, T, out_final_hidden);
}

void nemo_dm_stats(unsigned long *out6)
{
    out6[0] = g_nemo.total_requests;
    out6[1] = g_nemo.tier1_pool_hits;
    out6[2] = g_nemo.tier2_host_hits;
    out6[3] = g_nemo.tier3_drive_fetches;
    out6[4] = g_nemo.r4_clean_drops;
    out6[5] = g_nemo.dirty_writebacks;
}
