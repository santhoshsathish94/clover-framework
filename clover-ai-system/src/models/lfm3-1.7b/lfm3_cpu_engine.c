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
    LFM3_D = 2048,
    LFM3_FFN = 7680,
    LFM3_LAYERS = 24,
    LFM3_HEADS = 16,
    LFM3_KV_HEADS = 4,
    LFM3_HDIM = 128,
    LFM3_CONV_K = 7,
    LFM3_SLIDING_W = 256,
    LFM3_HEADER_SIZE = 72,
    LFM3_RECORD_SIZE = 36
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
    Lfm3TrunkHeader hdr;
    Lfm3RecordDesc recs[20];
} Lfm3TrunkMapped;

static int load_lfm3_trunk(const char *trunk_dir, Lfm3TrunkMapped *t)
{
    memset(t, 0, sizeof(*t));
    t->index_mf.fd = -1;
    t->operator_mf.fd = -1;
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

static struct {
    int initialized;
    MmapFile emb_mf;
    MmapFile emb_norm_mf;
    MmapFile rope_mf;
    float inv_freq_full[64];
    float inv_freq_sliding[64];
    Lfm3TrunkMapped trunks[LFM3_LAYERS];
} g_lfm3 = { 0 };

static void linear_bf16_omp(
    const float *x,
    const uint16_t *W_bf16,
    float *y,
    int T,
    int out_dim,
    int in_dim
)
{
    /* Use 8 threads (1 physical CCD on Ryzen 9 7950X3D) + 8-token register blocking
       so each weight row is streamed once per 8 tokens without cross-CCD stalls! */
    #pragma omp parallel for num_threads(8) schedule(static)
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
    int n_rows,
    int dim,
    float eps
)
{
    #pragma omp parallel for num_threads(8) schedule(static)
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

int lfm3_engine_init(const char *pruned_dataset_dir)
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

        #pragma omp parallel for num_threads(8) schedule(static)
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
        #pragma omp parallel for num_threads(8) schedule(static)
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
        #pragma omp parallel for num_threads(8) schedule(static)
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
    const uint16_t *w1_w = (const uint16_t *)get_lfm3_record_ptr(trk, 10);
    const uint16_t *w3_w = (const uint16_t *)get_lfm3_record_ptr(trk, 11);
    const uint16_t *w2_w = (const uint16_t *)get_lfm3_record_ptr(trk, 12);

    float *ffn_in = (float *)malloc((size_t)T * LFM3_D * sizeof(float));
    float *w1_out = (float *)malloc((size_t)T * LFM3_FFN * sizeof(float));
    float *w3_out = (float *)malloc((size_t)T * LFM3_FFN * sizeof(float));
    float *ffn_raw = (float *)malloc((size_t)T * LFM3_D * sizeof(float));
    float *ffn_normed = (float *)malloc((size_t)T * LFM3_D * sizeof(float));

    rms_norm_bf16(h_mid, ffn_norm_w, ffn_in, T, LFM3_D, 1e-5f);
    linear_bf16_omp(ffn_in, w1_w, w1_out, T, LFM3_FFN, LFM3_D);
    linear_bf16_omp(ffn_in, w3_w, w3_out, T, LFM3_FFN, LFM3_D);

    #pragma omp parallel for num_threads(8) schedule(static)
    for (int i = 0; i < T * LFM3_FFN; i++) {
        float v1 = w1_out[i];
        float silu = bf16_round_f32(v1 / (1.0f + expf(-v1)));
        w1_out[i] = bf16_round_f32(silu * w3_out[i]);
    }
    linear_bf16_omp(w1_out, w2_w, ffn_raw, T, LFM3_D, LFM3_FFN);
    rms_norm_bf16(ffn_raw, post_ffn_norm_w, ffn_normed, T, LFM3_D, 1e-5f);

    for (int i = 0; i < T * LFM3_D; i++) {
        h_out[i] = bf16_round_f32(h_mid[i] + ffn_normed[i]);
    }

    free(op_in); free(op_out_raw); free(op_normed); free(h_mid);
    free(ffn_in); free(w1_out); free(w3_out); free(ffn_raw); free(ffn_normed);
}

/* Executes all 24 tri-hybrid LFM3-1.7B layers + final `embedding_norm` with optional clubbed segment isolation `seg_starts` */
float lfm3_forward_24_layers_clubbed(const int *token_ids, const int *seg_starts, int T, float *out_final_hidden)
{
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

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
        execute_single_lfm3_layer_c(&g_lfm3.trunks[l], cur, nxt, T, seg_starts);
        float *tmp = cur;
        cur = nxt;
        nxt = tmp;
    }

    rms_norm_bf16(cur, emb_norm_w, out_final_hidden, T, LFM3_D, 1e-5f);

    free(cur);
    free(nxt);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    return (float)((t1.tv_sec - t0.tv_sec) * 1000.0 + (t1.tv_nsec - t0.tv_nsec) / 1e6);
}

float lfm3_forward_24_layers(const int *token_ids, int T, float *out_final_hidden)
{
    return lfm3_forward_24_layers_clubbed(token_ids, NULL, T, out_final_hidden);
}
