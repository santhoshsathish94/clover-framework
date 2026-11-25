/*
 * Native C11 / OpenMP `HG4-16R` Engine for MoonshotAI `Kimi K3.1 Pro` (`D = 7168`, `V = 163,840`)
 * & `LithosAI/Kimi-K3.1-Pro-DSpark` 5-Layer Speculative Drafter (`libkimi_k31_hg4_engine.so`)
 * (`clover-ai-system/src/models/kimi-k3/kimi_k31_hg4_engine.c`)
 *
 * Implements:
 *   1. Canonical 16-Row `HG4-16R` & `HG8-16R` Kernels + Per-128-Block Bit-Exact `Int32-FWHT`
 *      (`Scalar == AVX2 == AVX-512` `memcmp == 0` under `-ffp-contract=off`)
 *   2. Register-Fused `SiTU-GLU` (`(4*tanh(g/4)*sigmoid(g)) * (25*tanh(u/25))`) & `SwiGLU`
 *   3. `Kimi Delta Attention (KDA)` (`96 x 128` causal delta-rule + 4-tap conv1d + `f_a/f_b` decay)
 *   4. `Gated DeepSeek Sparse Attention (Gated DSA)` + `5-Tensor Lightning Indexer` (`index_topk = 2048`)
 *   5. `Block-Snapshot Attention-Residual Aggregation` (`sa_res` / `mlp_res` / `output_attn_res`)
 *   6. All `93 / 93` `Kimi K3.1 Pro` Trunks (`Layer 0` `33,792` Dense `SiTU-GLU` + `92` Latent MoE Trunks)
 *   7. Full `5 / 5`-Layer `Kimi-K3.1-Pro-DSpark` (`7168`-dim `64Q/16KV` GQA + `14,336` SwiGLU +
 *      `163,840 x 256` Markov Head + Confidence Head) multi-token speculative drafter.
 */
#ifndef _WIN32
#define _GNU_SOURCE
#endif
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#include <cpuid.h>
#define K31_X86 1
#define K31_TGT_AVX2       __attribute__((target("avx2,fma,f16c")))
#define K31_TGT_AVX512VNNI __attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,avx512vnni,avx512vbmi,fma,f16c")))
#else
#define K31_X86 0
#define K31_TGT_AVX2
#define K31_TGT_AVX512VNNI
#endif

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#define K31_API __declspec(dllexport)
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#define K31_API __attribute__((visibility("default")))
#endif
#include "../hg_cuda_hook.h"

enum {
    K31_D            = 7168,
    K31_VOCAB        = 163840,
    K31_LATENT       = 3584,
    K31_EXP_INTER    = 3072,
    K31_SH_INTER     = 6144,
    K31_DENSE_INT    = 33792,
    K31_EXPERTS      = 896,
    K31_TOP_K        = 16,
    K31_MAX_LAYERS   = 93,
    K31_ATTN_RES_BLK = 12,
    K31_MAX_BLOCKS   = 8,
    K31_DS_LAYERS    = 5,
    K31_DS_INTER     = 14336,
    K31_PANEL_MAX    = 128
};

static inline float fp16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x03FFu;
    uint32_t u;
    if (exp == 0) {
        if (mant == 0) u = sign;
        else {
            exp = 1;
            while ((mant & 0x0400u) == 0) { mant <<= 1; exp--; }
            mant &= 0x03FFu;
            u = sign | ((exp + 112u) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        u = sign | 0x7F800000u | (mant << 13);
    } else {
        u = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static inline float situ_glu_scalar(float g, float u) {
    float act = (4.0f * tanhf(g * 0.25f)) / (1.0f + expf(-g));
    float cap = 25.0f * tanhf(u * 0.04f);
    return act * cap;
}

typedef struct {
    void  *ptr;
    size_t size;
#ifdef _WIN32
    HANDLE hFile, hMap;
#else
    int fd;
#endif
} k31_mmap_t;

static int k31_mmap_open(const char *path, k31_mmap_t *m) {
    memset(m, 0, sizeof(*m));
#ifdef _WIN32
    m->hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (m->hFile == INVALID_HANDLE_VALUE) return -1;
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(m->hFile, &sz)) { CloseHandle(m->hFile); return -1; }
    m->size = (size_t)sz.QuadPart;
    m->hMap = CreateFileMappingA(m->hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!m->hMap) { CloseHandle(m->hFile); return -1; }
    m->ptr = MapViewOfFile(m->hMap, FILE_MAP_READ, 0, 0, 0);
    if (!m->ptr) { CloseHandle(m->hMap); CloseHandle(m->hFile); return -1; }
    return 0;
#else
    m->fd = open(path, O_RDONLY);
    if (m->fd < 0) return -1;
    struct stat st;
    if (fstat(m->fd, &st) != 0) { close(m->fd); return -1; }
    m->size = (size_t)st.st_size;
    m->ptr = mmap(NULL, m->size, PROT_READ, MAP_PRIVATE, m->fd, 0);
    if (m->ptr == MAP_FAILED) { m->ptr = NULL; close(m->fd); return -1; }
#ifdef MADV_HUGEPAGE
    if (m->size >= (2u << 20)) madvise(m->ptr, m->size, MADV_HUGEPAGE);
#endif
    return 0;
#endif
}

static void k31_mmap_close(k31_mmap_t *m) {
    if (!m || !m->ptr) return;
#ifdef _WIN32
    UnmapViewOfFile(m->ptr);
    if (m->hMap) CloseHandle(m->hMap);
    if (m->hFile != INVALID_HANDLE_VALUE) CloseHandle(m->hFile);
#else
    munmap(m->ptr, m->size);
    if (m->fd >= 0) close(m->fd);
#endif
    memset(m, 0, sizeof(*m));
}

typedef struct {
    uint32_t R, K, bits;
    const float    *pre_diag;
    const uint16_t *scales_fp16;
    const uint8_t  *tiles;
} k31_mat_t;

static void parse_k31_mat(const uint8_t *blob, k31_mat_t *v) {
    uint32_t raw_k;
    memcpy(&v->R, blob + 0, 4);
    memcpy(&raw_k, blob + 4, 4);
    uint32_t top = raw_k >> 28;
    v->bits = (top == 8u) ? 8u : ((top == 2u) ? 2u : 4u);
    v->K = raw_k & 0x0FFFFFFFu;
    size_t nb = v->K / 128;
    size_t off = 8;
    v->pre_diag = (const float *)(blob + off);
    off += (size_t)v->K * 4;
    v->scales_fp16 = (const uint16_t *)(blob + off);
    off += ((size_t)v->R * nb) * 2;
    v->tiles = blob + off;
}

typedef struct {
    const float *qc, *kc, *vc; /* [12288 * 4] each */
    const float *dtb;          /* [12288] */
    const float *o_norm;       /* [128] */
    const float *A_neg;        /* [128] */
    k31_mat_t q, k, v, g, o, b, fa, fb;
} k31_kda_t;

typedef struct {
    const float *qa_ln;        /* [1536] */
    const float *kva_ln;       /* [512] */
    const float *ikn_w, *ikn_b;/* [128] */
    k31_mat_t qa, qb, kva, kvb, g, o;
    k31_mat_t idx_wp, idx_wk, idx_wqb;
} k31_dsa_t;

typedef struct {
    const float *gate_w;       /* [896 * 7168] */
    const float *gate_b;       /* [896] */
    const float *lat_norm;     /* [3584] */
    k31_mat_t lat_dn, lat_up, sh_w13, sh_dn;
} k31_moe_trunk_t;

typedef struct {
    k31_mmap_t     mf;          /* Hot HG4-16R mmap (`K31H` or legacy `K31E`) */
    k31_mmap_t     cold_mf;     /* Optional Cold HG2-16R mmap when `CLOVER_PIN_COLD_VRAM=1` */
    int            loaded;
    int            is_hybrid;   /* 1 = K31H (hot HG4 mmap + cold HG2 pread/VRAM), 0 = K31E */
    uint32_t       n_cached;    /* 896 when is_hybrid == 1 */
    uint32_t       n_hot;
    uint32_t       n_cold;
    uint32_t       exp_stride;  /* hot HG4 stride (17,043,464 B) */
    uint32_t       cold_stride; /* cold HG2 stride (8,785,928 B) */
    const int16_t *slot_map;    /* [896]: >=0 -> hot idx, <0 -> -1 - cold idx */
    const float   *pre_diag_up; /* [3584] */
    const uint8_t *exp_base;    /* hot HG4 base pointer */
    const uint8_t *cold_base;   /* cold HG2 base pointer when mmap'd for GPU VRAM pinning */
#ifdef _WIN32
    HANDLE         cold_fh;
#else
    int            cold_fd;
#endif
} k31_experts_t;

typedef struct {
    uint32_t     kind; /* 0 = L0 KDA+Dense, 1 = KDA+MoE, 2 = DSA+MoE */
    k31_mmap_t   mf;
    const float *in_ln, *post_ln;
    const float *sa_rn, *sa_rp, *mlp_rn, *mlp_rp;
    k31_kda_t       kda;
    k31_dsa_t       dsa;
    k31_mat_t       dense_w13, dense_dn;
    k31_moe_trunk_t moe;
    k31_experts_t   ex;
} k31_layer_t;

typedef struct {
    const float *in_ln, *post_ln, *qn, *kn;
    k31_mat_t q, k, v, o, w13, wdn;
} k31_ds_layer_t;

typedef struct {
    k31_mmap_t     mf;
    int            loaded;
    const float   *hid_norm, *final_norm, *conf_w, *conf_b;
    k31_mat_t      fc, mw1, mw2;
    k31_ds_layer_t layers[K31_DS_LAYERS];
    float          inv_freq[32];
    float          yarn_mscale;
} k31_dspark_t;

static struct {
    int ready;
    int isa_tier;
    int n_layers;
    char loaded_dir[512];

    uint8_t lut16_u8[16];
    int8_t  lut16_s8[16];
    float   lut16_s8_f[16];
    uint8_t lut4_u8[16];
    int8_t  lut4_s8[16];
    float   norm_w[K31_D];
    float   out_res_norm[K31_D];
    float   out_res_proj[K31_D];

    k31_mmap_t   emb_mf, lm_mf;
    k31_mat_t    emb_mat, lm_mat;
    k31_layer_t  layers[K31_MAX_LAYERS];
    k31_dspark_t dspark;

    int8_t  *scr_xq;
    float   *scr_xsc;
    int32_t *scr_xsum;
    uint8_t *cold_stage_buf; /* 8 * 8,785,928 B reusable NVMe burst buffer */
} g_k31;

static int detect_k31_isa(void) {
#if K31_X86
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
        __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512vnni") &&
        __builtin_cpu_supports("avx512vbmi") && __builtin_cpu_supports("f16c")) {
        return 3;
    }
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c")) {
        return 1;
    }
#endif
    return 0;
}

K31_API const char *kimi_k31_hg4_isa_name(void) {
    int t = g_k31.ready ? g_k31.isa_tier : detect_k31_isa();
    if (t == 3) return "AVX-512 VNNI+VBMI (Kimi K3.1 Pro HG4-16R + Int32-FWHT + KDA/DSA + 5L DSpark)";
    if (t == 1) return "AVX2+FMA (Kimi K3.1 Pro HG4-16R + abs/sign LUT16 + Int32-FWHT + KDA/DSA + 5L DSpark)";
    return "Scalar C11 (Kimi K3.1 Pro HG4-16R + Int32-FWHT + KDA/DSA + 5L DSpark)";
}

K31_API int kimi_k31_hg4_num_layers(void) {
    return g_k31.n_layers;
}

K31_API void kimi_k31_hg4_set_isa_tier(int tier) {
    int max_hw = detect_k31_isa();
    if (tier < 0) g_k31.isa_tier = max_hw;
    else if (tier == 0) g_k31.isa_tier = 0;
    else if (tier == 1 && max_hw >= 1) g_k31.isa_tier = 1;
    else if (tier == 3 && max_hw == 3) g_k31.isa_tier = 3;
}

static void fwht128_1block_scalar(float *v) {
    const float inv_sqrt128 = 0.08838834764831845f;
    for (int h = 1; h < 128; h <<= 1) {
        for (int j = 0; j < 128; j += (h << 1)) {
            for (int k = 0; k < h; k++) {
                float a = v[j + k], b = v[j + h + k];
                v[j + k] = a + b;
                v[j + h + k] = a - b;
            }
        }
    }
    for (int i = 0; i < 128; i++) v[i] *= inv_sqrt128;
}

static void rotate_and_quantize_128_scalar(
    const float *x, const float *pre_diag, int T, int K,
    int8_t *out_xq, float *out_xsc, int32_t *out_xsum, int use_omp
) {
    const int nb = K / 128;
    const float scale_c = 0.08838834764831845f / 127.0f;
    #pragma omp parallel for schedule(static) if (use_omp && T >= 4)
    for (int t = 0; t < T; t++) {
        const float *xt = x + (size_t)t * K;
        int8_t  *xqt = out_xq + (size_t)t * K;
        float   *xst = out_xsc + (size_t)t * nb;
        int32_t *xmt = out_xsum + (size_t)t * nb;
        for (int b = 0; b < nb; b++) {
            float prod[128];
            const float *xb = xt + b * 128;
            const float *pd = pre_diag + b * 128;
            float max_abs = 0.0f;
            for (int i = 0; i < 128; i++) {
                float p = xb[i] * pd[i];
                prod[i] = p;
                float ap = fabsf(p);
                if (ap > max_abs) max_abs = ap;
            }
            int8_t *qd = xqt + b * 128;
            if (max_abs < 1e-12f) {
                xst[b] = 0.0f; xmt[b] = 0; memset(qd, 0, 128);
                continue;
            }
            float in_sc = max_abs * (1.0f / 2048.0f);
            float inv_in = 2048.0f / max_abs;
            int32_t rot[128];
            for (int i = 0; i < 128; i++) {
                float v = prod[i] * inv_in;
                if (v > 2048.0f) v = 2048.0f;
                if (v < -2048.0f) v = -2048.0f;
                rot[i] = (int32_t)rint((double)v);
            }
            for (int h = 1; h < 128; h <<= 1) {
                for (int j = 0; j < 128; j += (h << 1)) {
                    for (int k = 0; k < h; k++) {
                        int32_t a = rot[j + k], bv = rot[j + h + k];
                        rot[j + k] = a + bv; rot[j + h + k] = a - bv;
                    }
                }
            }
            int32_t max_rot = 0;
            for (int i = 0; i < 128; i++) {
                int32_t ar = rot[i] < 0 ? -rot[i] : rot[i];
                if (ar > max_rot) max_rot = ar;
            }
            if (max_rot == 0) {
                xst[b] = 0.0f; xmt[b] = 0; memset(qd, 0, 128);
            } else {
                xst[b] = (in_sc * (float)max_rot) * scale_c;
                float inv_rot = 127.0f / (float)max_rot;
                int32_t ssum = 0;
                for (int i = 0; i < 128; i++) {
                    float qf = (float)rot[i] * inv_rot;
                    if (qf > 127.0f) qf = 127.0f;
                    if (qf < -127.0f) qf = -127.0f;
                    int32_t qi = (int32_t)rint((double)qf);
                    qd[i] = (int8_t)qi;
                    ssum += qi;
                }
                xmt[b] = ssum;
            }
        }
    }
}

#if K31_X86
K31_TGT_AVX2 static void rotate_and_quantize_128_avx2(
    const float *x, const float *pre_diag, int T, int K,
    int8_t *out_xq, float *out_xsc, int32_t *out_xsum, int use_omp
) {
    const int nb = K / 128;
    const float scale_c = 0.08838834764831845f / 127.0f;
    const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    const __m256 vmin_in = _mm256_set1_ps(-2048.0f), vmax_in = _mm256_set1_ps(2048.0f);
    const __m256 vmin_c  = _mm256_set1_ps(-127.0f),  vmax_c  = _mm256_set1_ps(127.0f);
    const __m256i bias80 = _mm256_set1_epi8((char)0x80);
    const __m256i zero   = _mm256_setzero_si256();

    #pragma omp parallel for schedule(static) if (use_omp && T >= 4)
    for (int t = 0; t < T; t++) {
        const float *xt = x + (size_t)t * K;
        int8_t  *xqt = out_xq + (size_t)t * K;
        float   *xst = out_xsc + (size_t)t * nb;
        int32_t *xmt = out_xsum + (size_t)t * nb;

        for (int b = 0; b < nb; b++) {
            float prod[128];
            int32_t rot[128];
            const float *xb = xt + b * 128;
            const float *pd = pre_diag + b * 128;
            __m256 vmax = _mm256_setzero_ps();
            for (int i = 0; i < 128; i += 8) {
                __m256 v = _mm256_mul_ps(_mm256_loadu_ps(xb + i), _mm256_loadu_ps(pd + i));
                _mm256_storeu_ps(prod + i, v);
                vmax = _mm256_max_ps(vmax, _mm256_and_ps(v, sign_mask));
            }
            __m128 m128 = _mm_max_ps(_mm256_castps256_ps128(vmax), _mm256_extractf128_ps(vmax, 1));
            m128 = _mm_max_ps(m128, _mm_movehl_ps(m128, m128));
            m128 = _mm_max_ss(m128, _mm_shuffle_ps(m128, m128, 1));
            float max_abs = _mm_cvtss_f32(m128);

            int8_t *qd = xqt + b * 128;
            if (max_abs < 1e-12f) {
                xst[b] = 0.0f; xmt[b] = 0; memset(qd, 0, 128);
                continue;
            }
            float in_sc = max_abs * (1.0f / 2048.0f);
            __m256 vinv_in = _mm256_set1_ps(2048.0f / max_abs);
            for (int i = 0; i < 128; i += 8) {
                __m256 vf = _mm256_min_ps(vmax_in, _mm256_max_ps(vmin_in, _mm256_mul_ps(_mm256_loadu_ps(prod + i), vinv_in)));
                __m256i vi = _mm256_cvtps_epi32(vf);
                __m256i a1 = _mm256_shuffle_epi32(vi, 0xA0), b1 = _mm256_shuffle_epi32(vi, 0xF5);
                vi = _mm256_blend_epi32(_mm256_add_epi32(a1, b1), _mm256_sub_epi32(a1, b1), 0xAA);
                __m256i a2 = _mm256_shuffle_epi32(vi, 0x44), b2 = _mm256_shuffle_epi32(vi, 0xEE);
                vi = _mm256_blend_epi32(_mm256_add_epi32(a2, b2), _mm256_sub_epi32(a2, b2), 0xCC);
                __m256i hi = _mm256_permute2x128_si256(vi, vi, 0x11), lo = _mm256_permute2x128_si256(vi, vi, 0x00);
                vi = _mm256_blend_epi32(_mm256_add_epi32(lo, hi), _mm256_sub_epi32(lo, hi), 0xF0);
                _mm256_storeu_si256((__m256i *)(rot + i), vi);
            }
            for (int h = 8; h < 128; h <<= 1) {
                for (int j = 0; j < 128; j += (h << 1)) {
                    for (int k = 0; k < h; k += 8) {
                        __m256i a = _mm256_loadu_si256((const __m256i *)(rot + j + k));
                        __m256i bv = _mm256_loadu_si256((const __m256i *)(rot + j + h + k));
                        _mm256_storeu_si256((__m256i *)(rot + j + k),     _mm256_add_epi32(a, bv));
                        _mm256_storeu_si256((__m256i *)(rot + j + h + k), _mm256_sub_epi32(a, bv));
                    }
                }
            }
            __m256i vimax = _mm256_setzero_si256();
            for (int i = 0; i < 128; i += 8) vimax = _mm256_max_epi32(vimax, _mm256_abs_epi32(_mm256_loadu_si256((const __m256i *)(rot + i))));
            __m128i mi = _mm_max_epi32(_mm256_castsi256_si128(vimax), _mm256_extracti128_si256(vimax, 1));
            mi = _mm_max_epi32(mi, _mm_shuffle_epi32(mi, 0x4E));
            mi = _mm_max_epi32(mi, _mm_shuffle_epi32(mi, 0xB1));
            int32_t max_rot = _mm_cvtsi128_si32(mi);

            if (max_rot == 0) {
                xst[b] = 0.0f; xmt[b] = 0; memset(qd, 0, 128);
            } else {
                xst[b] = (in_sc * (float)max_rot) * scale_c;
                __m256 vinv = _mm256_set1_ps(127.0f / (float)max_rot);
                int64_t tot_u8 = 0;
                for (int half = 0; half < 2; half++) {
                    const int32_t *rh = rot + half * 64;
                    #define K31_Q8(off) _mm256_cvtps_epi32(_mm256_min_ps(vmax_c, _mm256_max_ps(vmin_c, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i *)(rh + (off)))), vinv))))
                    __m256i i0 = K31_Q8(0),  i1 = K31_Q8(8),  i2 = K31_Q8(16), i3 = K31_Q8(24);
                    __m256i i4 = K31_Q8(32), i5 = K31_Q8(40), i6 = K31_Q8(48), i7 = K31_Q8(56);
                    #undef K31_Q8
                    __m256i b03 = _mm256_permute4x64_epi64(_mm256_packs_epi16(
                        _mm256_permute4x64_epi64(_mm256_packs_epi32(i0, i1), 0xD8),
                        _mm256_permute4x64_epi64(_mm256_packs_epi32(i2, i3), 0xD8)), 0xD8);
                    __m256i b47 = _mm256_permute4x64_epi64(_mm256_packs_epi16(
                        _mm256_permute4x64_epi64(_mm256_packs_epi32(i4, i5), 0xD8),
                        _mm256_permute4x64_epi64(_mm256_packs_epi32(i6, i7), 0xD8)), 0xD8);
                    _mm256_storeu_si256((__m256i *)(qd + half * 64 + 0),  b03);
                    _mm256_storeu_si256((__m256i *)(qd + half * 64 + 32), b47);
                    __m256i sad = _mm256_add_epi64(_mm256_sad_epu8(_mm256_xor_si256(b03, bias80), zero),
                                                   _mm256_sad_epu8(_mm256_xor_si256(b47, bias80), zero));
                    __m128i s128 = _mm_add_epi64(_mm256_castsi256_si128(sad), _mm256_extracti128_si256(sad, 1));
                    tot_u8 += _mm_cvtsi128_si64(s128) + _mm_cvtsi128_si64(_mm_srli_si128(s128, 8));
                }
                xmt[b] = (int32_t)(tot_u8 - 16384LL);
            }
        }
    }
}

/* `glu_mode`: 0 = none, 1 = SwiGLU (DSpark), 2 = SiTU-GLU (Kimi K3.1 Pro) */
K31_TGT_AVX512VNNI static void k31_group16_avx512(
    const k31_mat_t *M, const uint8_t *lut16_u8, const uint8_t *lut4_u8,
    const int8_t *X_q, const float *X_sc, const int32_t *X_sum,
    float *Y, int T, int rg, int glu_mode
) {
    const int R = (int)M->R, K = (int)M->K, nb = K / 128;
    const int out_R = glu_mode ? (R / 2) : R;
    const size_t bstride = (M->bits == 8) ? 2048u : ((M->bits == 2) ? 512u : 1024u);
    const uint8_t *wt = M->tiles + (size_t)rg * (size_t)nb * bstride;
    const uint16_t *wsc = M->scales_fp16 + (size_t)rg * (size_t)nb * 16;
    const __m512i vlut  = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)lut16_u8));
    const __m512i vlut4 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)lut4_u8));
    const __m512i m0f   = _mm512_set1_epi8(0x0F);
    const __m512i m03   = _mm512_set1_epi8(0x03);

    for (int t0 = 0; t0 < T; t0 += 8) {
        int tblk = (t0 + 8 <= T) ? 8 : (T - t0);
        __m512 vs[8];
        for (int ti = 0; ti < tblk; ti++) vs[ti] = _mm512_setzero_ps();

        for (int b = 0; b < nb; b++) {
            const uint8_t *wb = wt + (size_t)b * bstride;
            __m512 vws = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(wsc + b * 16)));
            __m512i acc[8];
            for (int ti = 0; ti < tblk; ti++) acc[ti] = _mm512_setzero_si512();

            if (M->bits == 8) {
                for (int s = 0; s < 32; s++) {
                    __m512i wu8 = _mm512_loadu_si512((const __m512i *)(wb + s * 64));
                    for (int ti = 0; ti < tblk; ti++) {
                        const int32_t *xp = (const int32_t *)(X_q + (size_t)(t0 + ti) * K + b * 128);
                        acc[ti] = _mm512_dpbusd_epi32(acc[ti], wu8, _mm512_set1_epi32(xp[s]));
                    }
                }
            } else if (M->bits == 2) {
                for (int s = 0; s < 8; s++) {
                    __m512i pk = _mm512_loadu_si512((const __m512i *)(wb + s * 64));
                    __m512i w0 = _mm512_permutexvar_epi8(_mm512_and_si512(pk, m03), vlut4);
                    __m512i w1 = _mm512_permutexvar_epi8(_mm512_and_si512(_mm512_srli_epi16(pk, 2), m03), vlut4);
                    __m512i w2 = _mm512_permutexvar_epi8(_mm512_and_si512(_mm512_srli_epi16(pk, 4), m03), vlut4);
                    __m512i w3 = _mm512_permutexvar_epi8(_mm512_and_si512(_mm512_srli_epi16(pk, 6), m03), vlut4);
                    for (int ti = 0; ti < tblk; ti++) {
                        const int32_t *xp = (const int32_t *)(X_q + (size_t)(t0 + ti) * K + b * 128);
                        acc[ti] = _mm512_dpbusd_epi32(acc[ti], w0, _mm512_set1_epi32(xp[4 * s + 0]));
                        acc[ti] = _mm512_dpbusd_epi32(acc[ti], w1, _mm512_set1_epi32(xp[4 * s + 1]));
                        acc[ti] = _mm512_dpbusd_epi32(acc[ti], w2, _mm512_set1_epi32(xp[4 * s + 2]));
                        acc[ti] = _mm512_dpbusd_epi32(acc[ti], w3, _mm512_set1_epi32(xp[4 * s + 3]));
                    }
                }
            } else {
                for (int s = 0; s < 16; s++) {
                    __m512i pk  = _mm512_loadu_si512((const __m512i *)(wb + s * 64));
                    __m512i wlo = _mm512_permutexvar_epi8(_mm512_and_si512(pk, m0f), vlut);
                    __m512i whi = _mm512_permutexvar_epi8(_mm512_and_si512(_mm512_srli_epi16(pk, 4), m0f), vlut);
                    for (int ti = 0; ti < tblk; ti++) {
                        const int32_t *xp = (const int32_t *)(X_q + (size_t)(t0 + ti) * K + b * 128);
                        acc[ti] = _mm512_dpbusd_epi32(acc[ti], wlo, _mm512_set1_epi32(xp[2 * s + 0]));
                        acc[ti] = _mm512_dpbusd_epi32(acc[ti], whi, _mm512_set1_epi32(xp[2 * s + 1]));
                    }
                }
            }
            for (int ti = 0; ti < tblk; ti++) {
                int ta = t0 + ti;
                __m512i d32 = _mm512_sub_epi32(acc[ti], _mm512_set1_epi32(128 * X_sum[ta * nb + b]));
                vs[ti] = _mm512_add_ps(vs[ti], _mm512_mul_ps(_mm512_cvtepi32_ps(d32), _mm512_mul_ps(vws, _mm512_set1_ps(X_sc[ta * nb + b]))));
            }
        }
        for (int ti = 0; ti < tblk; ti++) {
            float *yt = Y + (size_t)(t0 + ti) * out_R;
            if (glu_mode == 1) {
                float sv[16]; _mm512_storeu_ps(sv, vs[ti]);
                for (int j = 0; j < 8; j++) yt[rg * 8 + j] = (sv[j] / (1.0f + expf(-sv[j]))) * sv[j + 8];
            } else if (glu_mode == 2) {
                float sv[16]; _mm512_storeu_ps(sv, vs[ti]);
                for (int j = 0; j < 8; j++) yt[rg * 8 + j] = situ_glu_scalar(sv[j], sv[j + 8]);
            } else {
                _mm512_storeu_ps(yt + rg * 16, vs[ti]);
            }
        }
    }
}

K31_TGT_AVX2 static void k31_group16_avx2(
    const k31_mat_t *M, const int8_t *lut16_s8, const int8_t *lut4_s8,
    const int8_t *X_q, const float *X_sc,
    float *Y, int T, int rg, int glu_mode
) {
    const int R = (int)M->R, K = (int)M->K, nb = K / 128;
    const int out_R = glu_mode ? (R / 2) : R;
    const size_t bstride = (M->bits == 8) ? 2048u : ((M->bits == 2) ? 512u : 1024u);
    const uint8_t *wt = M->tiles + (size_t)rg * (size_t)nb * bstride;
    const uint16_t *wsc = M->scales_fp16 + (size_t)rg * (size_t)nb * 16;
    const __m256i vlut   = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)lut16_s8));
    const __m256i vlut4  = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)lut4_s8));
    const __m256i m0f    = _mm256_set1_epi8(0x0F);
    const __m256i m03    = _mm256_set1_epi8(0x03);
    const __m256i bias80 = _mm256_set1_epi8((char)0x80);
    const __m256i ones16 = _mm256_set1_epi16(1);

    for (int t0 = 0; t0 < T; t0 += 4) {
        int tblk = (t0 + 4 <= T) ? 4 : (T - t0);
        __m256 vs0[4], vs1[4];
        for (int ti = 0; ti < tblk; ti++) { vs0[ti] = _mm256_setzero_ps(); vs1[ti] = _mm256_setzero_ps(); }

        for (int b = 0; b < nb; b++) {
            const uint8_t *wb = wt + (size_t)b * bstride;
            __m256 vws0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(wsc + b * 16 + 0)));
            __m256 vws1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(wsc + b * 16 + 8)));
            __m256i a0[4], a1[4];
            for (int ti = 0; ti < tblk; ti++) { a0[ti] = _mm256_setzero_si256(); a1[ti] = _mm256_setzero_si256(); }

            if (M->bits == 8) {
                for (int s = 0; s < 32; s++) {
                    __m256i w0 = _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(wb + s * 64 + 0)),  bias80);
                    __m256i w1 = _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(wb + s * 64 + 32)), bias80);
                    __m256i aw0 = _mm256_abs_epi8(w0), aw1 = _mm256_abs_epi8(w1);
                    for (int ti = 0; ti < tblk; ti++) {
                        const int32_t *xp = (const int32_t *)(X_q + (size_t)(t0 + ti) * K + b * 128);
                        __m256i xb = _mm256_set1_epi32(xp[s]);
                        a0[ti] = _mm256_add_epi32(a0[ti], _mm256_madd_epi16(_mm256_maddubs_epi16(aw0, _mm256_sign_epi8(xb, w0)), ones16));
                        a1[ti] = _mm256_add_epi32(a1[ti], _mm256_madd_epi16(_mm256_maddubs_epi16(aw1, _mm256_sign_epi8(xb, w1)), ones16));
                    }
                }
            } else if (M->bits == 2) {
                for (int s = 0; s < 8; s++) {
                    __m256i pk0 = _mm256_loadu_si256((const __m256i *)(wb + s * 64 + 0));
                    __m256i pk1 = _mm256_loadu_si256((const __m256i *)(wb + s * 64 + 32));
                    __m256i w0_0 = _mm256_shuffle_epi8(vlut4, _mm256_and_si256(pk0, m03));
                    __m256i w0_1 = _mm256_shuffle_epi8(vlut4, _mm256_and_si256(_mm256_srli_epi16(pk0, 2), m03));
                    __m256i w0_2 = _mm256_shuffle_epi8(vlut4, _mm256_and_si256(_mm256_srli_epi16(pk0, 4), m03));
                    __m256i w0_3 = _mm256_shuffle_epi8(vlut4, _mm256_and_si256(_mm256_srli_epi16(pk0, 6), m03));
                    __m256i w1_0 = _mm256_shuffle_epi8(vlut4, _mm256_and_si256(pk1, m03));
                    __m256i w1_1 = _mm256_shuffle_epi8(vlut4, _mm256_and_si256(_mm256_srli_epi16(pk1, 2), m03));
                    __m256i w1_2 = _mm256_shuffle_epi8(vlut4, _mm256_and_si256(_mm256_srli_epi16(pk1, 4), m03));
                    __m256i w1_3 = _mm256_shuffle_epi8(vlut4, _mm256_and_si256(_mm256_srli_epi16(pk1, 6), m03));
                    __m256i aw0_0 = _mm256_abs_epi8(w0_0), aw0_1 = _mm256_abs_epi8(w0_1), aw0_2 = _mm256_abs_epi8(w0_2), aw0_3 = _mm256_abs_epi8(w0_3);
                    __m256i aw1_0 = _mm256_abs_epi8(w1_0), aw1_1 = _mm256_abs_epi8(w1_1), aw1_2 = _mm256_abs_epi8(w1_2), aw1_3 = _mm256_abs_epi8(w1_3);
                    for (int ti = 0; ti < tblk; ti++) {
                        const int32_t *xp = (const int32_t *)(X_q + (size_t)(t0 + ti) * K + b * 128);
                        __m256i x0 = _mm256_set1_epi32(xp[4 * s + 0]), x1 = _mm256_set1_epi32(xp[4 * s + 1]);
                        __m256i x2 = _mm256_set1_epi32(xp[4 * s + 2]), x3 = _mm256_set1_epi32(xp[4 * s + 3]);
                        __m256i s00 = _mm256_add_epi32(_mm256_madd_epi16(_mm256_maddubs_epi16(aw0_0, _mm256_sign_epi8(x0, w0_0)), ones16),
                                                       _mm256_madd_epi16(_mm256_maddubs_epi16(aw0_1, _mm256_sign_epi8(x1, w0_1)), ones16));
                        __m256i s01 = _mm256_add_epi32(_mm256_madd_epi16(_mm256_maddubs_epi16(aw0_2, _mm256_sign_epi8(x2, w0_2)), ones16),
                                                       _mm256_madd_epi16(_mm256_maddubs_epi16(aw0_3, _mm256_sign_epi8(x3, w0_3)), ones16));
                        __m256i s10 = _mm256_add_epi32(_mm256_madd_epi16(_mm256_maddubs_epi16(aw1_0, _mm256_sign_epi8(x0, w1_0)), ones16),
                                                       _mm256_madd_epi16(_mm256_maddubs_epi16(aw1_1, _mm256_sign_epi8(x1, w1_1)), ones16));
                        __m256i s11 = _mm256_add_epi32(_mm256_madd_epi16(_mm256_maddubs_epi16(aw1_2, _mm256_sign_epi8(x2, w1_2)), ones16),
                                                       _mm256_madd_epi16(_mm256_maddubs_epi16(aw1_3, _mm256_sign_epi8(x3, w1_3)), ones16));
                        a0[ti] = _mm256_add_epi32(a0[ti], _mm256_add_epi32(s00, s01));
                        a1[ti] = _mm256_add_epi32(a1[ti], _mm256_add_epi32(s10, s11));
                    }
                }
            } else {
                for (int s = 0; s < 16; s++) {
                    __m256i pk0 = _mm256_loadu_si256((const __m256i *)(wb + s * 64 + 0));
                    __m256i pk1 = _mm256_loadu_si256((const __m256i *)(wb + s * 64 + 32));
                    __m256i w0_lo = _mm256_shuffle_epi8(vlut, _mm256_and_si256(pk0, m0f));
                    __m256i w0_hi = _mm256_shuffle_epi8(vlut, _mm256_and_si256(_mm256_srli_epi16(pk0, 4), m0f));
                    __m256i w1_lo = _mm256_shuffle_epi8(vlut, _mm256_and_si256(pk1, m0f));
                    __m256i w1_hi = _mm256_shuffle_epi8(vlut, _mm256_and_si256(_mm256_srli_epi16(pk1, 4), m0f));
                    __m256i aw0_lo = _mm256_abs_epi8(w0_lo), aw0_hi = _mm256_abs_epi8(w0_hi);
                    __m256i aw1_lo = _mm256_abs_epi8(w1_lo), aw1_hi = _mm256_abs_epi8(w1_hi);
                    for (int ti = 0; ti < tblk; ti++) {
                        const int32_t *xp = (const int32_t *)(X_q + (size_t)(t0 + ti) * K + b * 128);
                        __m256i xb0 = _mm256_set1_epi32(xp[2 * s + 0]);
                        __m256i xb1 = _mm256_set1_epi32(xp[2 * s + 1]);
                        __m256i p00 = _mm256_maddubs_epi16(aw0_lo, _mm256_sign_epi8(xb0, w0_lo));
                        __m256i p01 = _mm256_maddubs_epi16(aw0_hi, _mm256_sign_epi8(xb1, w0_hi));
                        __m256i p10 = _mm256_maddubs_epi16(aw1_lo, _mm256_sign_epi8(xb0, w1_lo));
                        __m256i p11 = _mm256_maddubs_epi16(aw1_hi, _mm256_sign_epi8(xb1, w1_hi));
                        a0[ti] = _mm256_add_epi32(a0[ti], _mm256_add_epi32(_mm256_madd_epi16(p00, ones16), _mm256_madd_epi16(p01, ones16)));
                        a1[ti] = _mm256_add_epi32(a1[ti], _mm256_add_epi32(_mm256_madd_epi16(p10, ones16), _mm256_madd_epi16(p11, ones16)));
                    }
                }
            }
            for (int ti = 0; ti < tblk; ti++) {
                int ta = t0 + ti;
                __m256 vxsc = _mm256_set1_ps(X_sc[ta * nb + b]);
                vs0[ti] = _mm256_add_ps(vs0[ti], _mm256_mul_ps(_mm256_cvtepi32_ps(a0[ti]), _mm256_mul_ps(vws0, vxsc)));
                vs1[ti] = _mm256_add_ps(vs1[ti], _mm256_mul_ps(_mm256_cvtepi32_ps(a1[ti]), _mm256_mul_ps(vws1, vxsc)));
            }
        }
        for (int ti = 0; ti < tblk; ti++) {
            float *yt = Y + (size_t)(t0 + ti) * out_R;
            if (glu_mode) {
                float g8[8], u8[8];
                _mm256_storeu_ps(g8, vs0[ti]);
                _mm256_storeu_ps(u8, vs1[ti]);
                if (glu_mode == 1) {
                    for (int j = 0; j < 8; j++) yt[rg * 8 + j] = (g8[j] / (1.0f + expf(-g8[j]))) * u8[j];
                } else {
                    for (int j = 0; j < 8; j++) yt[rg * 8 + j] = situ_glu_scalar(g8[j], u8[j]);
                }
            } else {
                _mm256_storeu_ps(yt + rg * 16 + 0, vs0[ti]);
                _mm256_storeu_ps(yt + rg * 16 + 8, vs1[ti]);
            }
        }
    }
}
#endif

static void k31_group16_scalar(
    const k31_mat_t *M, const int8_t *lut16_s8, const int8_t *lut4_s8,
    const int8_t *X_q, const float *X_sc,
    float *Y, int T, int rg, int glu_mode
) {
    const int R = (int)M->R, K = (int)M->K, nb = K / 128;
    const int out_R = glu_mode ? (R / 2) : R;
    const size_t bstride = (M->bits == 8) ? 2048u : ((M->bits == 2) ? 512u : 1024u);
    const uint8_t *wt = M->tiles + (size_t)rg * (size_t)nb * bstride;
    const uint16_t *wsc = M->scales_fp16 + (size_t)rg * (size_t)nb * 16;

    for (int t = 0; t < T; t++) {
        float vs[16] = {0};
        const int8_t *xqt = X_q + (size_t)t * K;
        for (int b = 0; b < nb; b++) {
            const uint8_t *wb = wt + (size_t)b * bstride;
            const int8_t *xqb = xqt + b * 128;
            int32_t acc[16] = {0};
            if (M->bits == 8) {
                for (int s = 0; s < 32; s++) {
                    const uint8_t *ws = wb + s * 64;
                    for (int r = 0; r < 16; r++)
                        for (int j = 0; j < 4; j++)
                            acc[r] += ((int32_t)ws[r * 4 + j] - 128) * (int32_t)xqb[s * 4 + j];
                }
            } else if (M->bits == 2) {
                for (int s = 0; s < 8; s++) {
                    const uint8_t *ws = wb + s * 64;
                    for (int r = 0; r < 16; r++) {
                        for (int j = 0; j < 4; j++) {
                            uint8_t pk = ws[r * 4 + j];
                            acc[r] += (int32_t)lut4_s8[pk & 0x03u]         * (int32_t)xqb[s * 16 + 0  + j]
                                    + (int32_t)lut4_s8[(pk >> 2) & 0x03u]  * (int32_t)xqb[s * 16 + 4  + j]
                                    + (int32_t)lut4_s8[(pk >> 4) & 0x03u]  * (int32_t)xqb[s * 16 + 8  + j]
                                    + (int32_t)lut4_s8[(pk >> 6) & 0x03u]  * (int32_t)xqb[s * 16 + 12 + j];
                        }
                    }
                }
            } else {
                for (int s = 0; s < 16; s++) {
                    const uint8_t *ws = wb + s * 64;
                    for (int r = 0; r < 16; r++) {
                        for (int j = 0; j < 4; j++) {
                            uint8_t pk = ws[r * 4 + j];
                            acc[r] += (int32_t)lut16_s8[pk & 0x0F] * (int32_t)xqb[s * 8 + j]
                                    + (int32_t)lut16_s8[pk >> 4]   * (int32_t)xqb[s * 8 + 4 + j];
                        }
                    }
                }
            }
            float xsc = X_sc[t * nb + b];
            for (int r = 0; r < 16; r++) vs[r] = vs[r] + (float)acc[r] * (fp16_to_f32(wsc[b * 16 + r]) * xsc);
        }
        float *yt = Y + (size_t)t * out_R;
        if (glu_mode == 1) {
            for (int j = 0; j < 8; j++) yt[rg * 8 + j] = (vs[j] / (1.0f + expf(-vs[j]))) * vs[j + 8];
        } else if (glu_mode == 2) {
            for (int j = 0; j < 8; j++) yt[rg * 8 + j] = situ_glu_scalar(vs[j], vs[j + 8]);
        } else {
            for (int r = 0; r < 16; r++) yt[rg * 16 + r] = vs[r];
        }
    }
}

static inline void k31_rot_quant_dispatch(const float *X, const float *pd, int T, int K, int8_t *xq, float *xsc, int32_t *xsum, int use_omp) {
#if K31_X86
    if (g_k31.isa_tier >= 1) {
        rotate_and_quantize_128_avx2(X, pd, T, K, xq, xsc, xsum, use_omp);
        return;
    }
#endif
    rotate_and_quantize_128_scalar(X, pd, T, K, xq, xsc, xsum, use_omp);
}

static inline void k31_run_one_group16(const k31_mat_t *M, const int8_t *xq, const float *xsc, const int32_t *xsum, float *Y, int T, int rg, int glu_mode) {
#if K31_X86
    if (g_k31.isa_tier == 3) { k31_group16_avx512(M, g_k31.lut16_u8, g_k31.lut4_u8, xq, xsc, xsum, Y, T, rg, glu_mode); return; }
    if (g_k31.isa_tier >= 1) { k31_group16_avx2(M, g_k31.lut16_s8, g_k31.lut4_s8, xq, xsc, Y, T, rg, glu_mode); return; }
#endif
    k31_group16_scalar(M, g_k31.lut16_s8, g_k31.lut4_s8, xq, xsc, Y, T, rg, glu_mode);
}

static void k31_matmul_prequant(const k31_mat_t *M, const int8_t *xq, const float *xsc, const int32_t *xsum, float *Y, int T, int glu_mode, int use_omp) {
    if (use_omp && hg_cuda_dispatch_matmul(M->tiles, M->scales_fp16, M->R, M->K, M->bits, xq, xsc, Y, T, glu_mode)) {
        return;
    }
    const int n_rg = (int)(M->R / 16);
    #pragma omp parallel for schedule(static) if (use_omp)
    for (int rg = 0; rg < n_rg; rg++)
        k31_run_one_group16(M, xq, xsc, xsum, Y, T, rg, glu_mode);
}

static void k31_matmul(const k31_mat_t *M, const float *X, float *Y, int T, int glu_mode, int use_omp) {
    const int K = (int)M->K, nb = K / 128;
    int use_scr = (use_omp && T <= K31_PANEL_MAX && K <= 35840 && g_k31.scr_xq != NULL);
    int8_t  *xq   = use_scr ? g_k31.scr_xq   : (int8_t *)malloc((size_t)T * K);
    float   *xsc  = use_scr ? g_k31.scr_xsc  : (float *)malloc((size_t)T * nb * sizeof(float));
    int32_t *xsum = use_scr ? g_k31.scr_xsum : (int32_t *)malloc((size_t)T * nb * sizeof(int32_t));
    k31_rot_quant_dispatch(X, M->pre_diag, T, K, xq, xsc, xsum, use_omp);
    k31_matmul_prequant(M, xq, xsc, xsum, Y, T, glu_mode, use_omp);
    if (!use_scr) { free(xq); free(xsc); free(xsum); }
}

static void k31_embed_lookup_row(const k31_mat_t *E, int token_id, float *dst) {
    if (token_id < 0 || token_id >= (int)E->R) token_id = 0;
    const int K = (int)E->K, nb = K / 128;
    int rg = token_id / 16, lane = token_id % 16;
    const size_t bstride = (E->bits == 8) ? 2048u : 1024u;
    const uint8_t *wt = E->tiles + (size_t)rg * (size_t)nb * bstride;
    const uint16_t *wsc = E->scales_fp16 + (size_t)rg * (size_t)nb * 16;

    for (int b = 0; b < nb; b++) {
        float *blk = dst + b * 128;
        float sc = fp16_to_f32(wsc[b * 16 + lane]);
        const uint8_t *wb = wt + (size_t)b * bstride;
        if (E->bits == 8) {
            for (int s = 0; s < 32; s++) {
                const uint8_t *wr = wb + s * 64 + lane * 4;
                for (int j = 0; j < 4; j++) blk[s * 4 + j] = (float)((int)wr[j] - 128) * sc;
            }
        } else {
            for (int s = 0; s < 16; s++) {
                const uint8_t *wr = wb + s * 64 + lane * 4;
                for (int j = 0; j < 4; j++) {
                    uint8_t pk = wr[j];
                    blk[s * 8 + j]     = g_k31.lut16_s8_f[pk & 0x0F] * sc;
                    blk[s * 8 + 4 + j] = g_k31.lut16_s8_f[pk >> 4]   * sc;
                }
            }
        }
        fwht128_1block_scalar(blk);
        for (int k = 0; k < 128; k++) blk[k] *= E->pre_diag[b * 128 + k];
    }
}

static inline void rms_norm_row(const float *x, const float *w, float *y, int dim, float eps) {
    double ss = 0.0;
    for (int d = 0; d < dim; d++) ss += (double)x[d] * (double)x[d];
    float inv = 1.0f / sqrtf((float)(ss / (double)dim) + eps);
    for (int d = 0; d < dim; d++) y[d] = x[d] * inv * w[d];
}

static void rms_norm_batch(const float *x, const float *w, float *y, int rows, int dim, float eps) {
    #pragma omp parallel for schedule(static) if (rows >= 4)
    for (int r = 0; r < rows; r++) rms_norm_row(x + (size_t)r * dim, w, y + (size_t)r * dim, dim, eps);
}

/* Block-Snapshot Attention-Residual Mixing (`attn_res_block_size = 12`, `blocks [T, K31_MAX_BLOCKS, K31_D]`) */
static void k31_attn_res_mix(
    const float *blocks, int num_blocks, const float *prefix_sum,
    const float *res_norm_w, const float *res_proj_w,
    float *out_mixed, int T
) {
    if (num_blocks <= 0) {
        if (out_mixed != prefix_sum) {
            memcpy(out_mixed, prefix_sum, (size_t)T * K31_D * sizeof(float));
        }
        return;
    }
    const int S = num_blocks + 1;
    #pragma omp parallel for schedule(static) if (T >= 2)
    for (int t = 0; t < T; t++) {
        float scores[16];
        float mx = -1e30f;
        for (int s = 0; s < S; s++) {
            const float *src = (s < num_blocks)
                ? (blocks + ((size_t)t * K31_MAX_BLOCKS + (size_t)s) * K31_D)
                : (prefix_sum + (size_t)t * K31_D);
            double ss = 0.0;
            for (int d = 0; d < K31_D; d++) ss += (double)src[d] * (double)src[d];
            float inv = 1.0f / sqrtf((float)(ss / (double)K31_D) + 1e-5f);
            float dot = 0.0f;
            #pragma omp simd reduction(+:dot)
            for (int d = 0; d < K31_D; d++) dot += (src[d] * inv * res_norm_w[d]) * res_proj_w[d];
            scores[s] = dot;
            if (dot > mx) mx = dot;
        }
        float sw = 0.0f;
        for (int s = 0; s < S; s++) { scores[s] = expf(scores[s] - mx); sw += scores[s]; }
        for (int s = 0; s < S; s++) scores[s] /= (sw + 1e-12f);

        float *dst = out_mixed + (size_t)t * K31_D;
        float w_last = scores[num_blocks];
        const float *p_src = prefix_sum + (size_t)t * K31_D;
        for (int d = 0; d < K31_D; d++) dst[d] = w_last * p_src[d];
        for (int s = 0; s < num_blocks; s++) {
            const float *b_src = blocks + ((size_t)t * K31_MAX_BLOCKS + (size_t)s) * K31_D;
            float ws = scores[s];
            for (int d = 0; d < K31_D; d++) dst[d] += ws * b_src[d];
        }
    }
}

/* Kimi Delta Attention (`KDA`) Branch — Exact Householder / Delta-Rule Recurrence */
static void k31_run_kda(const k31_kda_t *kda, const float *u, float *attn_proj, int T) {
    float *q_raw = (float *)malloc((size_t)T * 12288 * sizeof(float));
    float *k_raw = (float *)malloc((size_t)T * 12288 * sizeof(float));
    float *v_raw = (float *)malloc((size_t)T * 12288 * sizeof(float));
    float *q_c   = (float *)malloc((size_t)T * 12288 * sizeof(float));
    float *k_c   = (float *)malloc((size_t)T * 12288 * sizeof(float));
    float *v_c   = (float *)malloc((size_t)T * 12288 * sizeof(float));
    float *b_raw = (float *)malloc((size_t)T * 96 * sizeof(float));
    float *fa_o  = (float *)malloc((size_t)T * 128 * sizeof(float));
    float *fb_o  = (float *)malloc((size_t)T * 12288 * sizeof(float));
    float *g_raw = (float *)malloc((size_t)T * 12288 * sizeof(float));
    float *a_out = (float *)malloc((size_t)T * 12288 * sizeof(float));

    k31_matmul(&kda->q,  u,    q_raw, T, 0, 1);
    k31_matmul(&kda->k,  u,    k_raw, T, 0, 1);
    k31_matmul(&kda->v,  u,    v_raw, T, 0, 1);
    k31_matmul(&kda->b,  u,    b_raw, T, 0, 1);
    k31_matmul(&kda->fa, u,    fa_o,  T, 0, 1);
    k31_matmul(&kda->fb, fa_o, fb_o,  T, 0, 1);
    k31_matmul(&kda->g,  u,    g_raw, T, 0, 1);

    #pragma omp parallel for schedule(static)
    for (int c = 0; c < 12288; c++) {
        const float *qw = kda->qc + c * 4, *kw = kda->kc + c * 4, *vw = kda->vc + c * 4;
        for (int t = 0; t < T; t++) {
            float sq = 0.0f, sk = 0.0f, sv = 0.0f;
            for (int tap = 0; tap < 4; tap++) {
                int ts = t - 3 + tap;
                if (ts >= 0) {
                    sq += q_raw[(size_t)ts * 12288 + c] * qw[tap];
                    sk += k_raw[(size_t)ts * 12288 + c] * kw[tap];
                    sv += v_raw[(size_t)ts * 12288 + c] * vw[tap];
                }
            }
            q_c[(size_t)t * 12288 + c] = sq / (1.0f + expf(-sq));
            k_c[(size_t)t * 12288 + c] = sk / (1.0f + expf(-sk));
            v_c[(size_t)t * 12288 + c] = sv / (1.0f + expf(-sv));
        }
    }

    const float inv_s128 = 1.0f / sqrtf(128.0f);
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < 96; h++) {
        float Sh[128 * 128];
        memset(Sh, 0, sizeof(Sh));
        const float b_a = -kda->A_neg[h]; /* A_neg[h] = -exp(A_log[h]), so b_a = exp(A_log[h]) > 0 */
        for (int t = 0; t < T; t++) {
            const float *qh = q_c + (size_t)t * 12288 + h * 128;
            const float *kh = k_c + (size_t)t * 12288 + h * 128;
            const float *vh = v_c + (size_t)t * 12288 + h * 128;
            const float *dr = fb_o + (size_t)t * 12288 + h * 128;
            double sq = 0.0, sk = 0.0;
            for (int d = 0; d < 128; d++) { sq += (double)qh[d] * qh[d]; sk += (double)kh[d] * kh[d]; }
            float iq = (1.0f / sqrtf((float)sq + 1e-6f)) * inv_s128;
            float ik = 1.0f / sqrtf((float)sk + 1e-6f);
            float beta = 1.0f / (1.0f + expf(-b_raw[t * 96 + h]));

            float qn[128], kn[128], dec[128], yh[128];
            for (int d = 0; d < 128; d++) {
                float g_arg = b_a * (dr[d] + kda->dtb[h * 128 + d]);
                float b_gate = -5.0f / (1.0f + expf(-g_arg));
                dec[d] = expf(b_gate);
                qn[d]  = qh[d] * iq;
                kn[d]  = kh[d] * ik;
            }

            for (int v = 0; v < 128; v++) {
                float *Srow = Sh + v * 128;
                float pred = 0.0f;
                #pragma omp simd reduction(+:pred)
                for (int k = 0; k < 128; k++) {
                    float s_dec = Srow[k] * dec[k];
                    Srow[k] = s_dec;
                    pred += s_dec * kn[k];
                }
                float vd = beta * (vh[v] - pred);
                float yv = 0.0f;
                #pragma omp simd reduction(+:yv)
                for (int k = 0; k < 128; k++) {
                    float s_new = Srow[k] + vd * kn[k];
                    Srow[k] = s_new;
                    yv += s_new * qn[k];
                }
                yh[v] = yv;
            }

            double ss = 0.0;
            for (int d = 0; d < 128; d++) ss += (double)yh[d] * yh[d];
            float inv = 1.0f / sqrtf((float)(ss / 128.0) + 1e-5f);
            float *dst = a_out + (size_t)t * 12288 + h * 128;
            const float *gr = g_raw + (size_t)t * 12288 + h * 128;
            for (int d = 0; d < 128; d++) {
                float gv = 1.0f / (1.0f + expf(-gr[d]));
                dst[d] = (yh[d] * inv * kda->o_norm[d]) * gv;
            }
        }
    }

    k31_matmul(&kda->o, a_out, attn_proj, T, 0, 1);
    free(q_raw); free(k_raw); free(v_raw); free(q_c); free(k_c); free(v_c);
    free(b_raw); free(fa_o); free(fb_o); free(g_raw); free(a_out);
}

/* Gated DeepSeek Sparse Attention (`Gated DSA`) + 5-Tensor Lightning Indexer */
static void k31_run_dsa(const k31_dsa_t *dsa, const float *u, float *attn_proj, float *out_idx_scores, int T) {
    float *q_lat  = (float *)malloc((size_t)T * 1536 * sizeof(float));
    float *q_all  = (float *)malloc((size_t)T * 18432 * sizeof(float));
    float *kv_all = (float *)malloc((size_t)T * 576 * sizeof(float));
    float *kv_lat = (float *)malloc((size_t)T * 512 * sizeof(float));
    float *kv_exp = (float *)malloc((size_t)T * 24576 * sizeof(float));
    float *g_raw  = (float *)malloc((size_t)T * 12288 * sizeof(float));
    float *idx_q  = (float *)malloc((size_t)T * 4096 * sizeof(float));
    float *idx_k  = (float *)malloc((size_t)T * 128 * sizeof(float));
    float *idx_w  = (float *)malloc((size_t)T * 32 * sizeof(float));
    float *idx_sc = (float *)malloc((size_t)T * T * sizeof(float));
    float *a_out  = (float *)malloc((size_t)T * 12288 * sizeof(float));

    k31_matmul(&dsa->qa, u, q_lat, T, 0, 1);
    rms_norm_batch(q_lat, dsa->qa_ln, q_lat, T, 1536, 1e-5f);
    k31_matmul(&dsa->qb, q_lat, q_all, T, 0, 1);

    k31_matmul(&dsa->kva, u, kv_all, T, 0, 1);
    for (int t = 0; t < T; t++) rms_norm_row(kv_all + (size_t)t * 576, dsa->kva_ln, kv_lat + (size_t)t * 512, 512, 1e-5f);
    k31_matmul(&dsa->kvb, kv_lat, kv_exp, T, 0, 1);
    k31_matmul(&dsa->g, u, g_raw, T, 0, 1);

    /* 5-Tensor Lightning Indexer */
    k31_matmul(&dsa->idx_wqb, q_lat, idx_q, T, 0, 1);
    k31_matmul(&dsa->idx_wk,  u,     idx_k, T, 0, 1);
    k31_matmul(&dsa->idx_wp,  u,     idx_w, T, 0, 1);

    for (int t = 0; t < T; t++) {
        float *kr = idx_k + (size_t)t * 128;
        double mean = 0.0, var = 0.0;
        for (int d = 0; d < 128; d++) mean += kr[d];
        mean /= 128.0;
        for (int d = 0; d < 128; d++) { double df = kr[d] - mean; var += df * df; }
        float inv = 1.0f / sqrtf((float)(var / 128.0) + 1e-5f);
        for (int d = 0; d < 128; d++) kr[d] = (float)((kr[d] - mean) * inv) * dsa->ikn_w[d] + dsa->ikn_b[d];
        for (int h = 0; h < 32; h++) {
            float v = idx_w[t * 32 + h];
            idx_w[t * 32 + h] = (v > 20.0f) ? v : log1pf(expf(v));
        }
    }

    const float idx_scale = 1.0f / sqrtf(128.0f);
    for (int t = 0; t < T; t++) {
        for (int s = 0; s < T; s++) {
            if (s > t) { idx_sc[t * T + s] = -1e9f; continue; }
            float sum = 0.0f;
            for (int h = 0; h < 32; h++) {
                const float *qh = idx_q + (size_t)t * 4096 + h * 128;
                const float *ks = idx_k + (size_t)s * 128;
                float dot = 0.0f;
                for (int d = 0; d < 128; d++) dot += qh[d] * ks[d];
                dot *= idx_scale;
                if (dot > 0.0f) sum += dot * idx_w[t * 32 + h];
            }
            idx_sc[t * T + s] = sum;
        }
    }
    if (out_idx_scores) memcpy(out_idx_scores, idx_sc, (size_t)T * T * sizeof(float));

    const float mla_scale = 1.0f / sqrtf(192.0f);
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < 96; h++) {
        float *scores = (float *)malloc((size_t)T * sizeof(float));
        for (int t = 0; t < T; t++) {
            const float *qn = q_all + (size_t)t * 18432 + h * 192;
            const float *qr = qn + 128;
            float mx = -1e30f;
            for (int s = 0; s <= t; s++) {
                const float *kn = kv_exp + (size_t)s * 24576 + h * 256;
                const float *kr = kv_all + (size_t)s * 576 + 512;
                float dot = 0.0f;
                for (int d = 0; d < 128; d++) dot += qn[d] * kn[d];
                for (int d = 0; d < 64;  d++) dot += qr[d] * kr[d];
                scores[s] = dot * mla_scale;
                if (scores[s] > mx) mx = scores[s];
            }
            float sw = 0.0f;
            for (int s = 0; s <= t; s++) { scores[s] = expf(scores[s] - mx); sw += scores[s]; }
            for (int s = 0; s <= t; s++) scores[s] /= (sw + 1e-12f);

            float yh[128] = {0};
            for (int s = 0; s <= t; s++) {
                const float *vs = kv_exp + (size_t)s * 24576 + h * 256 + 128;
                float ws = scores[s];
                for (int d = 0; d < 128; d++) yh[d] += ws * vs[d];
            }
            float *dst = a_out + (size_t)t * 12288 + h * 128;
            const float *gr = g_raw + (size_t)t * 12288 + h * 128;
            for (int d = 0; d < 128; d++) dst[d] = yh[d] * (1.0f / (1.0f + expf(-gr[d])));
        }
        free(scores);
    }

    k31_matmul(&dsa->o, a_out, attn_proj, T, 0, 1);
    free(q_lat); free(q_all); free(kv_all); free(kv_lat); free(kv_exp);
    free(g_raw); free(idx_q); free(idx_k); free(idx_w); free(idx_sc); free(a_out);
}

static int k31_pread_cold(const k31_experts_t *ex, int cold_idx, uint8_t *dst) {
    size_t len = (size_t)ex->cold_stride;
    uint64_t off = (uint64_t)cold_idx * (uint64_t)len;
#ifdef _WIN32
    if (ex->cold_fh == INVALID_HANDLE_VALUE) return -1;
    OVERLAPPED ov;
    memset(&ov, 0, sizeof(ov));
    ov.Offset = (DWORD)(off & 0xFFFFFFFFULL);
    ov.OffsetHigh = (DWORD)(off >> 32);
    DWORD got = 0;
    if (!ReadFile(ex->cold_fh, dst, (DWORD)len, &got, &ov) || got != (DWORD)len) return -1;
    return 0;
#else
    if (ex->cold_fd < 0) return -1;
    size_t done = 0;
    while (done < len) {
        ssize_t r = pread(ex->cold_fd, dst + done, len - done, (off_t)(off + done));
        if (r <= 0) return -1;
        done += (size_t)r;
    }
#ifdef POSIX_FADV_DONTNEED
    /* Release one-shot cold pages immediately so our DDR5-pinned trunks & hot experts are never evicted */
    posix_fadvise(ex->cold_fd, (off_t)off, (off_t)len, POSIX_FADV_DONTNEED);
#endif
    return 0;
#endif
}

static void k31_eval_expert_batch(
    const uint8_t *ep, int is_hg2, const float *pre_diag_up,
    const int8_t *x_xq, const float *x_xsc, const int32_t *x_xsum,
    const int *tok_ids, const float *tok_w, int T_e, float *lat_acc, int use_omp
) {
    if (T_e <= 0) return;
    const int nb_up = K31_LATENT / 128; /* 28 */
    k31_mat_t gu_m, dn_m;
    gu_m.R = 6144; gu_m.K = 3584; gu_m.bits = is_hg2 ? 2u : 4u;
    gu_m.pre_diag = pre_diag_up;
    gu_m.scales_fp16 = (const uint16_t *)ep;
    gu_m.tiles = ep + 344064u;
    parse_k31_mat(ep + (is_hg2 ? 5849088u : 11354112u), &dn_m);

    if (T_e == 1) {
        int t = tok_ids[0];
        float w = tok_w[0];
        float hid[K31_EXP_INTER], ey[K31_LATENT];
        k31_matmul_prequant(&gu_m, x_xq + (size_t)t * K31_LATENT, x_xsc + (size_t)t * nb_up, x_xsum + (size_t)t * nb_up, hid, 1, 2, use_omp);
        k31_matmul(&dn_m, hid, ey, 1, 0, use_omp);
        float *dst = lat_acc + (size_t)t * K31_LATENT;
        for (int d = 0; d < K31_LATENT; d++) dst[d] += ey[d] * w;
        return;
    }

    int8_t  *g_xq   = (int8_t *)malloc((size_t)T_e * K31_LATENT);
    float   *g_xsc  = (float *)malloc((size_t)T_e * nb_up * sizeof(float));
    int32_t *g_xsum = (int32_t *)malloc((size_t)T_e * nb_up * sizeof(int32_t));
    float   *hid_p  = (float *)malloc((size_t)T_e * K31_EXP_INTER * sizeof(float));
    float   *ey_p   = (float *)malloc((size_t)T_e * K31_LATENT * sizeof(float));

    for (int i = 0; i < T_e; i++) {
        int t = tok_ids[i];
        memcpy(g_xq   + (size_t)i * K31_LATENT, x_xq   + (size_t)t * K31_LATENT, K31_LATENT);
        memcpy(g_xsc  + (size_t)i * nb_up,      x_xsc  + (size_t)t * nb_up,      nb_up * sizeof(float));
        memcpy(g_xsum + (size_t)i * nb_up,      x_xsum + (size_t)t * nb_up,      nb_up * sizeof(int32_t));
    }

    k31_matmul_prequant(&gu_m, g_xq, g_xsc, g_xsum, hid_p, T_e, 2, use_omp);
    k31_matmul(&dn_m, hid_p, ey_p, T_e, 0, use_omp);

    for (int i = 0; i < T_e; i++) {
        float *dst = lat_acc + (size_t)tok_ids[i] * K31_LATENT;
        const float *ey = ey_p + (size_t)i * K31_LATENT;
        float w = tok_w[i];
        for (int d = 0; d < K31_LATENT; d++) dst[d] += ey[d] * w;
    }

    free(g_xq); free(g_xsc); free(g_xsum); free(hid_p); free(ey_p);
}

/* Latent `SiTU-GLU` MoE with Expert-Grouped Batching, Prompt-Seeded Topic LRU & Hybrid `K31H` */
static void k31_run_moe(const k31_moe_trunk_t *m, k31_experts_t *ex, const float *post, float *mlp_out, int T) {
    float *sh_hid = (float *)malloc((size_t)T * K31_SH_INTER * sizeof(float));
    k31_matmul(&m->sh_w13, post, sh_hid, T, 2, 1); /* Fused SiTU-GLU! */
    k31_matmul(&m->sh_dn, sh_hid, mlp_out, T, 0, 1);
    free(sh_hid);

    if (!ex->loaded || ex->n_cached == 0 || ex->n_cached > K31_EXPERTS) return;

    const int E = (int)ex->n_cached;
    const int top_k = (K31_TOP_K < E) ? K31_TOP_K : E;
    float *lat_in  = (float *)malloc((size_t)T * K31_LATENT * sizeof(float));
    float *lat_acc = (float *)calloc((size_t)T * K31_LATENT, sizeof(float));
    float *lat_nm  = (float *)malloc((size_t)T * K31_LATENT * sizeof(float));
    float *routed  = (float *)malloc((size_t)T * K31_D * sizeof(float));
    k31_matmul(&m->lat_dn, post, lat_in, T, 0, 1);

    const int nb_up = K31_LATENT / 128; /* 28 */
    int8_t  *x_xq   = (int8_t *)malloc((size_t)T * K31_LATENT);
    float   *x_xsc  = (float *)malloc((size_t)T * nb_up * sizeof(float));
    int32_t *x_xsum = (int32_t *)malloc((size_t)T * nb_up * sizeof(int32_t));
    k31_rot_quant_dispatch(lat_in, ex->pre_diag_up, T, K31_LATENT, x_xq, x_xsc, x_xsum, 1);

    int   *all_sel = (int *)malloc((size_t)T * top_k * sizeof(int));
    float *all_sw  = (float *)malloc((size_t)T * top_k * sizeof(float));

    if (T == 1) {
        float sc[896], ch[896];
        #pragma omp parallel for schedule(static)
        for (int e = 0; e < E; e++) {
            const float *gw = m->gate_w + (size_t)e * K31_D;
            float dot = 0.0f;
            #pragma omp simd reduction(+:dot)
            for (int d = 0; d < K31_D; d++) dot += post[d] * gw[d];
            float sv = 1.0f / (1.0f + expf(-dot));
            sc[e] = sv;
            ch[e] = sv + m->gate_b[e];
        }
        float wsum = 0.0f;
        for (int k = 0; k < top_k; k++) {
            int be = 0; float bv = -1e30f;
            for (int e = 0; e < E; e++) if (ch[e] > bv) { bv = ch[e]; be = e; }
            ch[be] = -1e30f;
            all_sel[k] = be; all_sw[k] = sc[be]; wsum += sc[be];
        }
        for (int k = 0; k < top_k; k++) all_sw[k] /= (wsum + 1e-12f);
    } else {
        #pragma omp parallel for schedule(static)
        for (int t = 0; t < T; t++) {
            const float *pt = post + (size_t)t * K31_D;
            float sc[896], ch[896];
            for (int e = 0; e < E; e++) {
                const float *gw = m->gate_w + (size_t)e * K31_D;
                float dot = 0.0f;
                #pragma omp simd reduction(+:dot)
                for (int d = 0; d < K31_D; d++) dot += pt[d] * gw[d];
                float sv = 1.0f / (1.0f + expf(-dot));
                sc[e] = sv;
                ch[e] = sv + m->gate_b[e];
            }
            float wsum = 0.0f;
            for (int k = 0; k < top_k; k++) {
                int be = 0; float bv = -1e30f;
                for (int e = 0; e < E; e++) if (ch[e] > bv) { bv = ch[e]; be = e; }
                ch[be] = -1e30f;
                all_sel[t * top_k + k] = be;
                all_sw[t * top_k + k]  = sc[be];
                wsum += sc[be];
            }
            for (int k = 0; k < top_k; k++) all_sw[t * top_k + k] /= (wsum + 1e-12f);
        }
    }

    /* Group routed (token, weight) pairs by unique expert ID (`0..E-1`) */
    int cnt[896] = {0};
    int off_e[897] = {0};
    for (int i = 0; i < T * top_k; i++) cnt[all_sel[i]]++;
    for (int e = 0; e < E; e++) off_e[e + 1] = off_e[e] + cnt[e];

    int   *grp_tok = (int *)malloc((size_t)(T * top_k) * sizeof(int));
    float *grp_w   = (float *)malloc((size_t)(T * top_k) * sizeof(float));
    int cur_off[896];
    memcpy(cur_off, off_e, (size_t)E * sizeof(int));
    for (int t = 0; t < T; t++) {
        for (int k = 0; k < top_k; k++) {
            int e = all_sel[t * top_k + k];
            int pos = cur_off[e]++;
            grp_tok[pos] = t;
            grp_w[pos]   = all_sw[t * top_k + k];
        }
    }

    int hot_active[896], n_hot_act = 0;
    int cold_active[896], n_cold_act = 0;
    for (int e = 0; e < E; e++) {
        if (cnt[e] == 0) continue;
        if (!ex->is_hybrid || ex->slot_map[e] >= 0) hot_active[n_hot_act++] = e;
        else cold_active[n_cold_act++] = e;
    }

    /* 1. Evaluate all active hot HG4-16R experts (GPU VRAM if available, else DDR5 CPU!) */
    int k31_hot_gpu_done = 0;
    if (n_hot_act > 0) {
        int n_hp = 0;
        for (int hi = 0; hi < n_hot_act; hi++) n_hp += cnt[hot_active[hi]];
        int   *hp_tok  = (int *)malloc((size_t)n_hp * sizeof(int));
        int   *hp_slot = (int *)malloc((size_t)n_hp * sizeof(int));
        float *hp_w    = (float *)malloc((size_t)n_hp * sizeof(float));
        int cur_hp = 0;
        for (int hi = 0; hi < n_hot_act; hi++) {
            int e = hot_active[hi];
            int slot = ex->is_hybrid ? (int)ex->slot_map[e] : e;
            for (int j = 0; j < cnt[e]; j++) {
                hp_tok[cur_hp]  = grp_tok[off_e[e] + j];
                hp_slot[cur_hp] = slot;
                hp_w[cur_hp]    = grp_w[off_e[e] + j];
                cur_hp++;
            }
        }
        k31_hot_gpu_done = hg_cuda_dispatch_moe_pairs(
            ex->exp_base, ex->is_hybrid ? ex->n_hot : ex->n_cached, ex->exp_stride,
            344064u, 11354112u, 6144u, K31_LATENT, K31_LATENT, K31_EXP_INTER, 4u, 2, 0,
            x_xq, x_xsc, T, hp_tok, hp_slot, hp_w, n_hp, lat_acc);
        free(hp_tok); free(hp_slot); free(hp_w);
    }
    if (!k31_hot_gpu_done) {
        for (int hi = 0; hi < n_hot_act; hi++) {
            int e = hot_active[hi];
            int slot = ex->is_hybrid ? (int)ex->slot_map[e] : e;
            const uint8_t *ep = ex->exp_base + (size_t)slot * ex->exp_stride;
            k31_eval_expert_batch(ep, 0, ex->pre_diag_up, x_xq, x_xsc, x_xsum,
                                  grp_tok + off_e[e], grp_w + off_e[e], cnt[e], lat_acc, 1);
        }
    }

    /* 2. Cold HG2-16R experts: dispatch to GPU VRAM when pinned (`cold_base != NULL`), else 16-lane `pread + GEMV` */
    if (n_cold_act > 0) {
        int k31_cold_gpu_done = 0;
        if (ex->cold_base != NULL) {
            int n_cp = 0;
            for (int ci = 0; ci < n_cold_act; ci++) n_cp += cnt[cold_active[ci]];
            int   *cp_tok  = (int *)malloc((size_t)n_cp * sizeof(int));
            int   *cp_slot = (int *)malloc((size_t)n_cp * sizeof(int));
            float *cp_w    = (float *)malloc((size_t)n_cp * sizeof(float));
            int cur_cp = 0;
            for (int ci = 0; ci < n_cold_act; ci++) {
                int e = cold_active[ci];
                int cslot = -1 - (int)ex->slot_map[e];
                for (int j = 0; j < cnt[e]; j++) {
                    cp_tok[cur_cp]  = grp_tok[off_e[e] + j];
                    cp_slot[cur_cp] = cslot;
                    cp_w[cur_cp]    = grp_w[off_e[e] + j];
                    cur_cp++;
                }
            }
            k31_cold_gpu_done = hg_cuda_dispatch_moe_pairs(
                ex->cold_base, ex->n_cold, ex->cold_stride,
                344064u, 5849088u, 6144u, K31_LATENT, K31_LATENT, K31_EXP_INTER, 2u, 2, 0,
                x_xq, x_xsc, T, cp_tok, cp_slot, cp_w, n_cp, lat_acc);
            free(cp_tok); free(cp_slot); free(cp_w);
        }

        if (!k31_cold_gpu_done) {
            const int nw = (n_cold_act < 16) ? n_cold_act : 16;
            /* Process in chunks of at most 32 experts so per-expert deterministic output buffer stays <= ~20 MB even at large T */
            for (int c_base = 0; c_base < n_cold_act; c_base += 32) {
                const int c_cnt = (c_base + 32 <= n_cold_act) ? 32 : (n_cold_act - c_base);
                float *lat_per_exp = (float *)calloc((size_t)c_cnt * (size_t)T * K31_LATENT, sizeof(float));

                #pragma omp parallel num_threads(nw)
                {
                    int tid = omp_get_thread_num();
                    uint8_t *my_stage = g_k31.cold_stage_buf + (size_t)tid * (size_t)ex->cold_stride;

                    #pragma omp for schedule(dynamic, 1)
                    for (int rel = 0; rel < c_cnt; rel++) {
                        int e = cold_active[c_base + rel];
                        int cslot = -1 - (int)ex->slot_map[e];
                        float *exp_lat = lat_per_exp + (size_t)rel * (size_t)T * K31_LATENT;
                        if (k31_pread_cold(ex, cslot, my_stage) == 0) {
                            k31_eval_expert_batch(my_stage, 1, ex->pre_diag_up, x_xq, x_xsc, x_xsum,
                                                  grp_tok + off_e[e], grp_w + off_e[e], cnt[e], exp_lat, 0);
                        }
                    }
                }

                /* Strictly ascending expert reduction (`rel = 0 .. c_cnt - 1`) guarantees 100% run-to-run bit-exact determinism */
                for (int rel = 0; rel < c_cnt; rel++) {
                    const float *src = lat_per_exp + (size_t)rel * (size_t)T * K31_LATENT;
                    for (int i = 0; i < T * K31_LATENT; i++) lat_acc[i] += src[i];
                }
                free(lat_per_exp);
            }
        }
    }

    rms_norm_batch(lat_acc, m->lat_norm, lat_nm, T, K31_LATENT, 1e-5f);
    k31_matmul(&m->lat_up, lat_nm, routed, T, 0, 1);
    for (int i = 0; i < T * K31_D; i++) mlp_out[i] += routed[i];

    free(all_sel); free(all_sw); free(grp_tok); free(grp_w);
    free(lat_in); free(lat_acc); free(lat_nm); free(routed);
    free(x_xq); free(x_xsc); free(x_xsum);
}

static void k31_forward_one_layer(
    k31_layer_t *lyr, int lid,
    float *prefix_sum, float *delta, int has_delta, float *blocks,
    float *out_delta, float *out_idx_scores, int T
) {
    const int is_write    = (lid % K31_ATTN_RES_BLK == 0) ? 1 : 0;
    const int write_idx   = lid / K31_ATTN_RES_BLK;
    const int prev_blocks = (lid + K31_ATTN_RES_BLK - 1) / K31_ATTN_RES_BLK;
    const int mlp_blocks  = prev_blocks + is_write;

    float *mixed     = (float *)malloc((size_t)T * K31_D * sizeof(float));
    float *u         = (float *)malloc((size_t)T * K31_D * sizeof(float));
    float *attn_proj = (float *)malloc((size_t)T * K31_D * sizeof(float));
    float *post      = (float *)malloc((size_t)T * K31_D * sizeof(float));

    /* 1. Pre-Attention Block AttnRes + input_layernorm */
    if (has_delta) {
        for (int i = 0; i < T * K31_D; i++) prefix_sum[i] += delta[i];
    }
    if (is_write && write_idx < K31_MAX_BLOCKS) {
        for (int t = 0; t < T; t++) {
            memcpy(blocks + ((size_t)t * K31_MAX_BLOCKS + (size_t)write_idx) * K31_D,
                   prefix_sum + (size_t)t * K31_D, K31_D * sizeof(float));
        }
    }
    k31_attn_res_mix(blocks, prev_blocks, prefix_sum, lyr->sa_rn, lyr->sa_rp, mixed, T);
    rms_norm_batch(mixed, lyr->in_ln, u, T, K31_D, 1e-5f);

    /* 2. Self-Attention (`KDA` or `Gated DSA`) */
    if (lyr->kind == 2) k31_run_dsa(&lyr->dsa, u, attn_proj, out_idx_scores, T);
    else                k31_run_kda(&lyr->kda, u, attn_proj, T);

    /* 3. Post-Attention Block AttnRes + post_attention_layernorm */
    if (is_write) {
        memcpy(prefix_sum, attn_proj, (size_t)T * K31_D * sizeof(float));
    } else {
        for (int i = 0; i < T * K31_D; i++) prefix_sum[i] += attn_proj[i];
    }
    k31_attn_res_mix(blocks, mlp_blocks, prefix_sum, lyr->mlp_rn, lyr->mlp_rp, mixed, T);
    rms_norm_batch(mixed, lyr->post_ln, post, T, K31_D, 1e-5f);

    /* 4. MLP / Latent SiTU-GLU MoE -> un-added `out_delta` */
    if (lyr->kind == 0) {
        float *dhid = (float *)malloc((size_t)T * K31_DENSE_INT * sizeof(float));
        k31_matmul(&lyr->dense_w13, post, dhid, T, 2, 1); /* Fused SiTU-GLU on 33,792 FFN! */
        k31_matmul(&lyr->dense_dn,  dhid, out_delta, T, 0, 1);
        free(dhid);
    } else {
        k31_run_moe(&lyr->moe, &lyr->ex, post, out_delta, T);
    }

    free(mixed); free(u); free(attn_proj); free(post);
}

static size_t parse_moe_tail_k31(const uint8_t *p, size_t off, k31_moe_trunk_t *m) {
    m->gate_w   = (const float *)(p + off); off += (size_t)K31_EXPERTS * K31_D * 4;
    m->gate_b   = (const float *)(p + off); off += K31_EXPERTS * 4;
    m->lat_norm = (const float *)(p + off); off += K31_LATENT * 4;
    uint32_t l1, l2, l3, l4;
    memcpy(&l1, p + off, 4); off += 4;
    memcpy(&l2, p + off, 4); off += 4;
    memcpy(&l3, p + off, 4); off += 4;
    memcpy(&l4, p + off, 4); off += 4;
    parse_k31_mat(p + off, &m->lat_dn); off += l1;
    parse_k31_mat(p + off, &m->lat_up); off += l2;
    parse_k31_mat(p + off, &m->sh_w13); off += l3;
    parse_k31_mat(p + off, &m->sh_dn);  off += l4;
    return off;
}

static void close_k31_experts(k31_experts_t *ex) {
    if (!ex || !ex->loaded) return;
    k31_mmap_close(&ex->mf);
    if (ex->cold_mf.ptr) k31_mmap_close(&ex->cold_mf);
#ifdef _WIN32
    if (ex->is_hybrid && ex->cold_fh != INVALID_HANDLE_VALUE) CloseHandle(ex->cold_fh);
#else
    if (ex->is_hybrid && ex->cold_fd >= 0) close(ex->cold_fd);
#endif
    memset(ex, 0, sizeof(*ex));
}

static int open_k31_experts(const char *hg4_dir, int layer_id, k31_experts_t *ex) {
    memset(ex, 0, sizeof(*ex));
#ifdef _WIN32
    ex->cold_fh = INVALID_HANDLE_VALUE;
#else
    ex->cold_fd = -1;
#endif
    char hot_path[1024], cold_path[1024];
    const char *cold_dir = getenv("CLOVER_KIMI_COLD_DIR");
    if (!cold_dir || !cold_dir[0]) cold_dir = "/mnt/nvme1/kimi-k31-hybrid";

    /* Prefer Hybrid `K31H` (`experts_hot_%02d.hg4` + `experts_cold_%02d.hg2`) when present */
    snprintf(hot_path, sizeof(hot_path), "%s/experts_hot_%02d.hg4", hg4_dir, layer_id);
    if (k31_mmap_open(hot_path, &ex->mf) != 0) {
        snprintf(hot_path, sizeof(hot_path), "%s/experts_hot_%02d.hg4", cold_dir, layer_id);
        if (k31_mmap_open(hot_path, &ex->mf) != 0) {
            /* Fall back to legacy `K31E` (`experts_%02d.hg4`) */
            snprintf(hot_path, sizeof(hot_path), "%s/experts_%02d.hg4", hg4_dir, layer_id);
            if (k31_mmap_open(hot_path, &ex->mf) != 0) return -1;
            const uint8_t *p = (const uint8_t *)ex->mf.ptr;
            if (memcmp(p, "K31E", 4) != 0) { k31_mmap_close(&ex->mf); return -2; }
            memcpy(&ex->n_cached, p + 4, 4);
            memcpy(&ex->exp_stride, p + 24, 4);
            ex->pre_diag_up = (const float *)(p + 28);
            ex->exp_base = p + 28 + K31_LATENT * 4;
            ex->loaded = 1;
            return 0;
        }
    }

    const uint8_t *p = (const uint8_t *)ex->mf.ptr;
    if (memcmp(p, "K31H", 4) != 0) { k31_mmap_close(&ex->mf); return -2; }
    memcpy(&ex->n_cached,    p + 4,  4);
    memcpy(&ex->n_hot,       p + 8,  4);
    memcpy(&ex->n_cold,      p + 12, 4);
    memcpy(&ex->exp_stride,  p + 16, 4);
    memcpy(&ex->cold_stride, p + 20, 4);
    if (ex->cold_stride == 0 || ex->cold_stride > 16u * 1024u * 1024u) {
        k31_mmap_close(&ex->mf);
        return -3;
    }
    ex->slot_map    = (const int16_t *)(p + 32);
    ex->pre_diag_up = (const float *)(p + 32 + 896 * 2);
    ex->exp_base    = p + 32 + 896 * 2 + K31_LATENT * 4;
    ex->is_hybrid   = 1;

#ifndef _WIN32
#ifdef MADV_WILLNEED
    if (ex->mf.size > 0) madvise(ex->mf.ptr, ex->mf.size, MADV_WILLNEED);
#endif
#endif

    snprintf(cold_path, sizeof(cold_path), "%s/experts_cold_%02d.hg2", cold_dir, layer_id);
#ifdef _WIN32
    ex->cold_fh = CreateFileA(cold_path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (ex->cold_fh == INVALID_HANDLE_VALUE) {
        snprintf(cold_path, sizeof(cold_path), "%s/experts_cold_%02d.hg2", hg4_dir, layer_id);
        ex->cold_fh = CreateFileA(cold_path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    }
    if (ex->n_cold > 0 && ex->cold_fh == INVALID_HANDLE_VALUE) {
        k31_mmap_close(&ex->mf);
        return -4;
    }
#else
    ex->cold_fd = open(cold_path, O_RDONLY);
    if (ex->cold_fd < 0) {
        snprintf(cold_path, sizeof(cold_path), "%s/experts_cold_%02d.hg2", hg4_dir, layer_id);
        ex->cold_fd = open(cold_path, O_RDONLY);
    }
    if (ex->n_cold > 0 && ex->cold_fd < 0) {
        k31_mmap_close(&ex->mf);
        return -4;
    }
#endif
    const char *pin_cold = getenv("CLOVER_PIN_COLD_VRAM");
    if (ex->n_cold > 0 && pin_cold && strcmp(pin_cold, "1") == 0) {
        if (k31_mmap_open(cold_path, &ex->cold_mf) == 0) {
            ex->cold_base = (const uint8_t *)ex->cold_mf.ptr;
        }
    }
    ex->loaded = 1;
    return 0;
}

static int open_k31_dspark(const char *path, k31_dspark_t *ds) {
    memset(ds, 0, sizeof(*ds));
    if (k31_mmap_open(path, &ds->mf) != 0) return -1;
    const uint8_t *p = (const uint8_t *)ds->mf.ptr;
    if (memcmp(p, "K3DS", 4) != 0) { k31_mmap_close(&ds->mf); return -2; }
    size_t off = 12;
    ds->hid_norm   = (const float *)(p + off); off += K31_D * 4;
    ds->final_norm = (const float *)(p + off); off += K31_D * 4;
    ds->conf_w     = (const float *)(p + off); off += 7424 * 4;
    ds->conf_b     = (const float *)(p + off); off += 4;
    uint32_t l_fc, l_mw1, l_mw2, l_ly[K31_DS_LAYERS];
    memcpy(&l_fc,  p + off, 4); off += 4;
    memcpy(&l_mw1, p + off, 4); off += 4;
    memcpy(&l_mw2, p + off, 4); off += 4;
    memcpy(l_ly,   p + off, sizeof(l_ly)); off += sizeof(l_ly);
    parse_k31_mat(p + off, &ds->fc);  off += l_fc;
    parse_k31_mat(p + off, &ds->mw1); off += l_mw1;
    parse_k31_mat(p + off, &ds->mw2); off += l_mw2;

    for (int l = 0; l < K31_DS_LAYERS; l++) {
        const uint8_t *lp = p + off;
        size_t lo = 0;
        k31_ds_layer_t *dl = &ds->layers[l];
        dl->in_ln   = (const float *)(lp + lo); lo += K31_D * 4;
        dl->post_ln = (const float *)(lp + lo); lo += K31_D * 4;
        dl->qn      = (const float *)(lp + lo); lo += 64 * 4;
        dl->kn      = (const float *)(lp + lo); lo += 64 * 4;
        uint32_t lq, lk, lv, lo_m, lw13, lwdn;
        memcpy(&lq,   lp + lo, 4); lo += 4;
        memcpy(&lk,   lp + lo, 4); lo += 4;
        memcpy(&lv,   lp + lo, 4); lo += 4;
        memcpy(&lo_m, lp + lo, 4); lo += 4;
        memcpy(&lw13, lp + lo, 4); lo += 4;
        memcpy(&lwdn, lp + lo, 4); lo += 4;
        parse_k31_mat(lp + lo, &dl->q);   lo += lq;
        parse_k31_mat(lp + lo, &dl->k);   lo += lk;
        parse_k31_mat(lp + lo, &dl->v);   lo += lv;
        parse_k31_mat(lp + lo, &dl->o);   lo += lo_m;
        parse_k31_mat(lp + lo, &dl->w13); lo += lw13;
        parse_k31_mat(lp + lo, &dl->wdn);
        off += l_ly[l];
    }
    /* YaRN RoPE (`head_dim=64, base=10000.0, factor=16.0, orig_max=65536, beta_fast=32, beta_slow=1`) */
    const float base = 10000.0f, factor = 16.0f, orig_max = 65536.0f;
    const float two_pi = 6.28318530717958647692f;
    float low  = floorf((64.0f * logf(orig_max / (32.0f * two_pi))) / (2.0f * logf(base)));
    float high = ceilf((64.0f * logf(orig_max / (1.0f  * two_pi))) / (2.0f * logf(base)));
    if (low < 0.0f) low = 0.0f;
    if (high > 63.0f) high = 63.0f;
    float denom = (high - low > 0.001f) ? (high - low) : 0.001f;
    for (int i = 0; i < 32; i++) {
        float pos_freq = powf(base, (float)(2 * i) / 64.0f);
        float inv_ext  = 1.0f / pos_freq;
        float inv_int  = 1.0f / (factor * pos_freq);
        float ramp     = ((float)i - low) / denom;
        if (ramp < 0.0f) ramp = 0.0f;
        if (ramp > 1.0f) ramp = 1.0f;
        float mask     = 1.0f - ramp;
        ds->inv_freq[i] = inv_int * (1.0f - mask) + inv_ext * mask;
    }
    ds->yarn_mscale = 0.1f * logf(factor) + 1.0f;
    ds->loaded = 1;
    return 0;
}

K31_API void kimi_k31_cache_reset(void);

K31_API int kimi_k31_hg4_init_range(const char *hg4_dir, int l_lo, int l_hi, int need_emb, int need_head) {
    kimi_k31_cache_reset();
    if (g_k31.ready) {
        if (g_k31.emb_mf.ptr) k31_mmap_close(&g_k31.emb_mf);
        if (g_k31.lm_mf.ptr)  k31_mmap_close(&g_k31.lm_mf);
        if (g_k31.dspark.loaded) k31_mmap_close(&g_k31.dspark.mf);
        for (int l = 0; l < g_k31.n_layers; l++) {
            if (g_k31.layers[l].mf.ptr) k31_mmap_close(&g_k31.layers[l].mf);
            close_k31_experts(&g_k31.layers[l].ex);
        }
        g_k31.ready = 0;
    }

    g_k31.isa_tier = detect_k31_isa();
    char path[1024];
    snprintf(path, sizeof(path), "%s/meta.bin", hg4_dir);
    FILE *mf = fopen(path, "rb");
    if (!mf) return -1;
    char magic[4];
    uint32_t hdr[4];
    if (fread(magic, 1, 4, mf) != 4 || memcmp(magic, "K31R", 4) != 0 ||
        fread(hdr, 4, 4, mf) != 4 ||
        fread(g_k31.lut16_u8, 1, 16, mf) != 16 ||
        fread(g_k31.norm_w, 4, K31_D, mf) != K31_D ||
        fread(g_k31.out_res_norm, 4, K31_D, mf) != K31_D ||
        fread(g_k31.out_res_proj, 4, K31_D, mf) != K31_D) { fclose(mf); return -2; }
    fclose(mf);

    g_k31.n_layers = (int)hdr[1];
    if (g_k31.n_layers <= 0 || g_k31.n_layers > K31_MAX_LAYERS) return -2;
    if (l_lo < 0) l_lo = 0;
    if (l_hi > g_k31.n_layers) l_hi = g_k31.n_layers;

    for (int i = 0; i < 16; i++) {
        int sv = (int)g_k31.lut16_u8[i] - 128;
        g_k31.lut16_s8[i] = (int8_t)sv;
        g_k31.lut16_s8_f[i] = (float)sv;
    }
    memset(g_k31.lut4_u8, 128, 16);
    memset(g_k31.lut4_s8, 0, 16);
    g_k31.lut4_s8[0] = -127; g_k31.lut4_s8[1] = -38; g_k31.lut4_s8[2] = 38; g_k31.lut4_s8[3] = 127;
    for (int i = 0; i < 4; i++) g_k31.lut4_u8[i] = (uint8_t)((int)g_k31.lut4_s8[i] + 128);

    if (need_emb) {
        snprintf(path, sizeof(path), "%s/embed_tokens.hg4", hg4_dir);
        if (k31_mmap_open(path, &g_k31.emb_mf) != 0) return -3;
        parse_k31_mat((const uint8_t *)g_k31.emb_mf.ptr, &g_k31.emb_mat);
    }
    if (need_head) {
        snprintf(path, sizeof(path), "%s/lm_head.hg4", hg4_dir);
        if (k31_mmap_open(path, &g_k31.lm_mf) != 0) return -4;
        parse_k31_mat((const uint8_t *)g_k31.lm_mf.ptr, &g_k31.lm_mat);
    }

    for (int l = 0; l < g_k31.n_layers; l++) {
        memset(&g_k31.layers[l], 0, sizeof(g_k31.layers[l]));
        if (l < l_lo || l >= l_hi) continue;
        k31_layer_t *lyr = &g_k31.layers[l];
        snprintf(path, sizeof(path), "%s/layer_%02d.hg4", hg4_dir, l);
        if (k31_mmap_open(path, &lyr->mf) != 0) return -(100 + l);
        const uint8_t *p = (const uint8_t *)lyr->mf.ptr;
        size_t off = 0;
        memcpy(&lyr->kind, p + off, 4); off += 4;
        lyr->in_ln   = (const float *)(p + off); off += K31_D * 4;
        lyr->post_ln = (const float *)(p + off); off += K31_D * 4;
        lyr->sa_rn   = (const float *)(p + off); off += K31_D * 4;
        lyr->sa_rp   = (const float *)(p + off); off += K31_D * 4;
        lyr->mlp_rn  = (const float *)(p + off); off += K31_D * 4;
        lyr->mlp_rp  = (const float *)(p + off); off += K31_D * 4;

        if (lyr->kind == 0 || lyr->kind == 1) {
            lyr->kda.qc     = (const float *)(p + off); off += 12288 * 4 * 4;
            lyr->kda.kc     = (const float *)(p + off); off += 12288 * 4 * 4;
            lyr->kda.vc     = (const float *)(p + off); off += 12288 * 4 * 4;
            lyr->kda.dtb    = (const float *)(p + off); off += 12288 * 4;
            lyr->kda.o_norm = (const float *)(p + off); off += 128 * 4;
            lyr->kda.A_neg  = (const float *)(p + off); off += 128 * 4;
            uint32_t ls[8];
            memcpy(ls, p + off, 32); off += 32;
            parse_k31_mat(p + off, &lyr->kda.q);  off += ls[0];
            parse_k31_mat(p + off, &lyr->kda.k);  off += ls[1];
            parse_k31_mat(p + off, &lyr->kda.v);  off += ls[2];
            parse_k31_mat(p + off, &lyr->kda.g);  off += ls[3];
            parse_k31_mat(p + off, &lyr->kda.o);  off += ls[4];
            parse_k31_mat(p + off, &lyr->kda.b);  off += ls[5];
            parse_k31_mat(p + off, &lyr->kda.fa); off += ls[6];
            parse_k31_mat(p + off, &lyr->kda.fb); off += ls[7];
        } else {
            lyr->dsa.qa_ln  = (const float *)(p + off); off += 1536 * 4;
            lyr->dsa.kva_ln = (const float *)(p + off); off += 512 * 4;
            lyr->dsa.ikn_w  = (const float *)(p + off); off += 128 * 4;
            lyr->dsa.ikn_b  = (const float *)(p + off); off += 128 * 4;
            uint32_t ls[9];
            memcpy(ls, p + off, 36); off += 36;
            parse_k31_mat(p + off, &lyr->dsa.qa);      off += ls[0];
            parse_k31_mat(p + off, &lyr->dsa.qb);      off += ls[1];
            parse_k31_mat(p + off, &lyr->dsa.kva);     off += ls[2];
            parse_k31_mat(p + off, &lyr->dsa.kvb);     off += ls[3];
            parse_k31_mat(p + off, &lyr->dsa.g);       off += ls[4];
            parse_k31_mat(p + off, &lyr->dsa.o);       off += ls[5];
            parse_k31_mat(p + off, &lyr->dsa.idx_wp);  off += ls[6];
            parse_k31_mat(p + off, &lyr->dsa.idx_wk);  off += ls[7];
            parse_k31_mat(p + off, &lyr->dsa.idx_wqb); off += ls[8];
        }

        if (lyr->kind == 0) {
            uint32_t lw13, lwdn;
            memcpy(&lw13, p + off, 4); off += 4;
            memcpy(&lwdn, p + off, 4); off += 4;
            parse_k31_mat(p + off, &lyr->dense_w13); off += lw13;
            parse_k31_mat(p + off, &lyr->dense_dn);
        } else {
            parse_moe_tail_k31(p, off, &lyr->moe);
            /* Stage setups (`n_layers <= 4`) allow missing non-stage expert files; full 93L requires all 92 MoE layers */
            int exp_rc = open_k31_experts(hg4_dir, l, &lyr->ex);
            if (exp_rc != 0 && g_k31.n_layers == K31_MAX_LAYERS) {
                return -(200 + l);
            }
        }
    }

    if (need_head) {
        snprintf(path, sizeof(path), "%s/dspark.hg4", hg4_dir);
        open_k31_dspark(path, &g_k31.dspark);
    }

    if (!g_k31.scr_xq) {
        g_k31.scr_xq   = (int8_t *)malloc((size_t)K31_PANEL_MAX * 35840);
        g_k31.scr_xsc  = (float *)malloc((size_t)K31_PANEL_MAX * (35840 / 128) * sizeof(float));
        g_k31.scr_xsum = (int32_t *)malloc((size_t)K31_PANEL_MAX * (35840 / 128) * sizeof(int32_t));
    }
    uint32_t max_cold_stride = 8785928u;
    for (int l = l_lo; l < l_hi; l++) {
        if (g_k31.layers[l].ex.cold_stride > max_cold_stride) max_cold_stride = g_k31.layers[l].ex.cold_stride;
    }
    if (g_k31.cold_stage_buf) free(g_k31.cold_stage_buf);
    g_k31.cold_stage_buf = (uint8_t *)malloc((size_t)16 * (size_t)max_cold_stride);
    snprintf(g_k31.loaded_dir, sizeof(g_k31.loaded_dir), "%s[%d:%d]", hg4_dir, l_lo, l_hi);
    g_k31.ready = 1;
    return 0;
}

K31_API int kimi_k31_hg4_init(const char *hg4_dir) {
    char key[1080];
    snprintf(key, sizeof(key), "%s[0:%d]", hg4_dir, K31_MAX_LAYERS);
    if (g_k31.ready && strcmp(g_k31.loaded_dir, key) == 0) return 0;
    return kimi_k31_hg4_init_range(hg4_dir, 0, K31_MAX_LAYERS, 1, 1);
}

/* Forward selected Kimi K3.1 Pro layers and optionally extract DSpark's 5 target features `[L7, L23, L67, L83, L91]` (`[T, 35840]`) */
K31_API float kimi_k31_hg4_forward_layers_ex(
    const int *token_ids, int T, const int *layer_ids, int n_sel_layers,
    float *out_final_hidden, float *out_dsa_indexer_scores, float *out_dspark_feats_35840
) {
    if (!g_k31.ready || T <= 0 || n_sel_layers <= 0) return -1.0f;
    double t0 = omp_get_wtime();

    float *prefix_sum = (float *)malloc((size_t)T * K31_D * sizeof(float));
    float *delta      = (float *)malloc((size_t)T * K31_D * sizeof(float));
    float *nxt_delta  = (float *)malloc((size_t)T * K31_D * sizeof(float));
    float *blocks     = (float *)calloc((size_t)T * K31_MAX_BLOCKS * K31_D, sizeof(float));
    float *agg_final  = (float *)malloc((size_t)T * K31_D * sizeof(float));

    #pragma omp parallel for schedule(static) if (T >= 4)
    for (int t = 0; t < T; t++) {
        k31_embed_lookup_row(&g_k31.emb_mat, token_ids[t], prefix_sum + (size_t)t * K31_D);
    }

    const int tgt_lids[5] = {7, 23, 67, 83, 91};
    int feat_captured[5] = {0, 0, 0, 0, 0};

    int has_delta = 0;
    int max_lid = 0;
    for (int i = 0; i < n_sel_layers; i++) {
        int lid = layer_ids[i];
        if (lid < 0 || lid >= g_k31.n_layers) continue;
        if (lid > max_lid) max_lid = lid;
        k31_forward_one_layer(&g_k31.layers[lid], lid, prefix_sum, delta, has_delta, blocks,
                              nxt_delta, (g_k31.layers[lid].kind == 2) ? out_dsa_indexer_scores : NULL, T);
        float *sw = delta; delta = nxt_delta; nxt_delta = sw;
        has_delta = 1;

        if (out_dspark_feats_35840) {
            for (int m = 0; m < 5; m++) {
                if (lid == tgt_lids[m]) {
                    feat_captured[m] = 1;
                    for (int t = 0; t < T; t++) {
                        float *dst = out_dspark_feats_35840 + ((size_t)t * 5 + (size_t)m) * K31_D;
                        const float *ps = prefix_sum + (size_t)t * K31_D;
                        const float *dl = delta      + (size_t)t * K31_D;
                        for (int d = 0; d < K31_D; d++) dst[d] = ps[d] + dl[d];
                    }
                }
            }
        }
    }

    if (has_delta) {
        for (int i = 0; i < T * K31_D; i++) prefix_sum[i] += delta[i];
    }

    if (out_dspark_feats_35840) {
        for (int m = 0; m < 5; m++) {
            if (!feat_captured[m]) {
                for (int t = 0; t < T; t++) {
                    memcpy(out_dspark_feats_35840 + ((size_t)t * 5 + (size_t)m) * K31_D,
                           prefix_sum + (size_t)t * K31_D, K31_D * sizeof(float));
                }
            }
        }
    }

    int final_blocks = (max_lid + 1 + K31_ATTN_RES_BLK - 1) / K31_ATTN_RES_BLK;
    if (final_blocks > K31_MAX_BLOCKS) final_blocks = K31_MAX_BLOCKS;
    k31_attn_res_mix(blocks, final_blocks, prefix_sum, g_k31.out_res_norm, g_k31.out_res_proj, agg_final, T);
    rms_norm_batch(agg_final, g_k31.norm_w, out_final_hidden, T, K31_D, 1e-5f);

    free(prefix_sum); free(delta); free(nxt_delta); free(blocks); free(agg_final);
    return (float)((omp_get_wtime() - t0) * 1000.0);
}

/* Forward a selected list of Kimi K3.1 Pro layer IDs (e.g. C1 stage `[0, 1, 3]` or full trunk chain `[0..92]`) */
K31_API float kimi_k31_hg4_forward_layers(
    const int *token_ids, int T, const int *layer_ids, int n_sel_layers,
    float *out_final_hidden, float *out_dsa_indexer_scores
) {
    return kimi_k31_hg4_forward_layers_ex(token_ids, T, layer_ids, n_sel_layers, out_final_hidden, out_dsa_indexer_scores, NULL);
}

/* Convenience C1 Stage Forward (`Layer 0 KDA+Dense -> Layer 1 KDA+MoE -> Layer 3 Gated-DSA+MoE`) */
K31_API float kimi_k31_hg4_forward_stage(const int *token_ids, int T, float *out_final_hidden) {
    int stage_lids[3] = {0, 1, 3};
    int cnt = (g_k31.n_layers >= 4) ? 3 : 1;
    return kimi_k31_hg4_forward_layers(token_ids, T, stage_lids, cnt, out_final_hidden, NULL);
}

K31_API float kimi_k31_hg4_all_logits(const float *hidden_t_7168, int T, float *out_logits_t_163k) {
    double t0 = omp_get_wtime();
    k31_matmul(&g_k31.lm_mat, hidden_t_7168, out_logits_t_163k, T, 0, 1);
    return (float)((omp_get_wtime() - t0) * 1000.0);
}

/* Full 5-Layer `Kimi-K3.1-Pro-DSpark` (`DFlashQwen3Model`) Stateless Forward:
 * Tokens with `token_ids[t] >= 0` up to the query block act as context (`K/V` projected from `rms_norm(fc(feats), hid_norm)`),
 * and the trailing query block (`[anchor_tok, -1, ..., -1]`) runs 5-layer non-causal Qwen3 GQA + sequential Markov head.
 */
K31_API float kimi_k31_dspark_forward(
    const int *token_ids, const float *target_feats_35840, int T,
    float *out_dspark_hidden_7168, int *out_draft_tokens, float *out_confidences
) {
    if (!g_k31.ready || !g_k31.dspark.loaded || T <= 0) return -1.0f;
    double t0 = omp_get_wtime();
    const k31_dspark_t *ds = &g_k31.dspark;

    /* Split `[0 .. ctx_len - 1]` (context) from `[ctx_len .. T - 1]` (query block: anchor + masks) */
    int first_mask = T;
    for (int t = 0; t < T; t++) {
        if (token_ids[t] < 0) { first_mask = t; break; }
    }
    int ctx_len = (first_mask < T) ? ((first_mask > 0) ? (first_mask - 1) : 0) : (T - 1);
    int q_len   = T - ctx_len;

    float *ctx_fc  = (ctx_len > 0) ? (float *)malloc((size_t)ctx_len * K31_D * sizeof(float)) : NULL;
    float *ctx_nm  = (ctx_len > 0) ? (float *)malloc((size_t)ctx_len * K31_D * sizeof(float)) : NULL;
    float *fc_tmp  = NULL;

    if (ctx_len > 0) {
        const float *f_in = target_feats_35840;
        if (!f_in) {
            fc_tmp = (float *)malloc((size_t)ctx_len * 5 * K31_D * sizeof(float));
            for (int t = 0; t < ctx_len; t++) {
                float e_row[K31_D];
                k31_embed_lookup_row(&g_k31.emb_mat, token_ids[t], e_row);
                for (int m = 0; m < 5; m++)
                    memcpy(fc_tmp + ((size_t)t * 5 + m) * K31_D, e_row, K31_D * sizeof(float));
            }
            f_in = fc_tmp;
        }
        k31_matmul(&ds->fc, f_in, ctx_fc, ctx_len, 0, 1);
        rms_norm_batch(ctx_fc, ds->hid_norm, ctx_nm, ctx_len, K31_D, 1e-5f);
        if (fc_tmp) free(fc_tmp);
    }

    float *cur = (float *)malloc((size_t)q_len * K31_D * sizeof(float));
    float *nxt = (float *)malloc((size_t)q_len * K31_D * sizeof(float));
    for (int qi = 0; qi < q_len; qi++) {
        int raw_id = token_ids[ctx_len + qi];
        int tid = (raw_id >= 0) ? raw_id : 163824;
        k31_embed_lookup_row(&g_k31.emb_mat, tid, cur + (size_t)qi * K31_D);
    }

    for (int l = 0; l < K31_DS_LAYERS; l++) {
        const k31_ds_layer_t *dl = &ds->layers[l];
        float *k_all = (float *)malloc((size_t)T * 1024 * sizeof(float));
        float *v_all = (float *)malloc((size_t)T * 1024 * sizeof(float));

        if (ctx_len > 0) {
            k31_matmul(&dl->k, ctx_nm, k_all, ctx_len, 0, 1);
            k31_matmul(&dl->v, ctx_nm, v_all, ctx_len, 0, 1);
            #pragma omp parallel for schedule(static) if (ctx_len >= 2)
            for (int t = 0; t < ctx_len; t++) {
                float cs[32], sn[32];
                for (int d = 0; d < 32; d++) {
                    float ang = (float)t * ds->inv_freq[d];
                    cs[d] = cosf(ang) * ds->yarn_mscale;
                    sn[d] = sinf(ang) * ds->yarn_mscale;
                }
                for (int kh = 0; kh < 16; kh++) {
                    float *khp = k_all + (size_t)t * 1024 + kh * 64;
                    rms_norm_row(khp, dl->kn, khp, 64, 1e-5f);
                    for (int d = 0; d < 32; d++) {
                        float a = khp[d], b = khp[32 + d];
                        khp[d] = a * cs[d] - b * sn[d];
                        khp[32 + d] = b * cs[d] + a * sn[d];
                    }
                }
            }
        }

        float *u  = (float *)malloc((size_t)q_len * K31_D * sizeof(float));
        float *q  = (float *)malloc((size_t)q_len * 4096 * sizeof(float));
        float *ao = (float *)malloc((size_t)q_len * 4096 * sizeof(float));
        float *bo = (float *)malloc((size_t)q_len * K31_D * sizeof(float));
        float *fh = (float *)malloc((size_t)q_len * K31_DS_INTER * sizeof(float));
        float *fo = (float *)malloc((size_t)q_len * K31_D * sizeof(float));

        float *kq = k_all + (size_t)ctx_len * 1024;
        float *vq = v_all + (size_t)ctx_len * 1024;
        rms_norm_batch(cur, dl->in_ln, u, q_len, K31_D, 1e-5f);
        k31_matmul(&dl->q, u, q,  q_len, 0, 1);
        k31_matmul(&dl->k, u, kq, q_len, 0, 1);
        k31_matmul(&dl->v, u, vq, q_len, 0, 1);

        for (int qi = 0; qi < q_len; qi++) {
            int gpos = ctx_len + qi;
            float cs[32], sn[32];
            for (int d = 0; d < 32; d++) {
                float ang = (float)gpos * ds->inv_freq[d];
                cs[d] = cosf(ang) * ds->yarn_mscale;
                sn[d] = sinf(ang) * ds->yarn_mscale;
            }
            for (int h = 0; h < 64; h++) {
                float *qh = q + (size_t)qi * 4096 + h * 64;
                rms_norm_row(qh, dl->qn, qh, 64, 1e-5f);
                for (int d = 0; d < 32; d++) {
                    float a = qh[d], b = qh[32 + d];
                    qh[d] = a * cs[d] - b * sn[d];
                    qh[32 + d] = b * cs[d] + a * sn[d];
                }
            }
            for (int kh = 0; kh < 16; kh++) {
                float *khp = kq + (size_t)qi * 1024 + kh * 64;
                rms_norm_row(khp, dl->kn, khp, 64, 1e-5f);
                for (int d = 0; d < 32; d++) {
                    float a = khp[d], b = khp[32 + d];
                    khp[d] = a * cs[d] - b * sn[d];
                    khp[32 + d] = b * cs[d] + a * sn[d];
                }
            }
        }

        const float scale = 1.0f / 8.0f;
        #pragma omp parallel for schedule(static)
        for (int h = 0; h < 64; h++) {
            int kv_h = h / 4;
            float *sc = (float *)malloc((size_t)T * sizeof(float));
            for (int qi = 0; qi < q_len; qi++) {
                const float *qh = q + (size_t)qi * 4096 + h * 64;
                float mx = -1e30f;
                for (int j = 0; j < T; j++) {
                    const float *kj = k_all + (size_t)j * 1024 + kv_h * 64;
                    float dot = 0.0f;
                    for (int d = 0; d < 64; d++) dot += qh[d] * kj[d];
                    sc[j] = dot * scale;
                    if (sc[j] > mx) mx = sc[j];
                }
                float sw = 0.0f;
                for (int j = 0; j < T; j++) { sc[j] = expf(sc[j] - mx); sw += sc[j]; }
                for (int j = 0; j < T; j++) sc[j] /= (sw + 1e-12f);

                float *yh = ao + (size_t)qi * 4096 + h * 64;
                memset(yh, 0, 64 * sizeof(float));
                for (int j = 0; j < T; j++) {
                    const float *vj = v_all + (size_t)j * 1024 + kv_h * 64;
                    float w = sc[j];
                    for (int d = 0; d < 64; d++) yh[d] += w * vj[d];
                }
            }
            free(sc);
        }

        k31_matmul(&dl->o, ao, bo, q_len, 0, 1);
        for (int i = 0; i < q_len * K31_D; i++) bo[i] += cur[i];
        rms_norm_batch(bo, dl->post_ln, u, q_len, K31_D, 1e-5f);
        k31_matmul(&dl->w13, u,  fh, q_len, 1, 1);
        k31_matmul(&dl->wdn, fh, fo, q_len, 0, 1);
        for (int i = 0; i < q_len * K31_D; i++) nxt[i] = bo[i] + fo[i];

        free(u); free(q); free(ao); free(bo); free(fh); free(fo); free(k_all); free(v_all);
        float *tmp = cur; cur = nxt; nxt = tmp;
    }

    if (ctx_len > 0 && out_dspark_hidden_7168) {
        memcpy(out_dspark_hidden_7168, ctx_nm, (size_t)ctx_len * K31_D * sizeof(float));
    }
    float *q_final = out_dspark_hidden_7168
        ? (out_dspark_hidden_7168 + (size_t)ctx_len * K31_D)
        : (float *)malloc((size_t)q_len * K31_D * sizeof(float));
    rms_norm_batch(cur, ds->final_norm, q_final, q_len, K31_D, 1e-5f);

    if (out_draft_tokens) {
        for (int t = 0; t < ctx_len; t++) {
            out_draft_tokens[t] = token_ids[t];
            if (out_confidences) out_confidences[t] = 1.0f;
        }
        float *base_lg = (float *)malloc((size_t)q_len * K31_VOCAB * sizeof(float));
        k31_matmul(&g_k31.lm_mat, q_final, base_lg, q_len, 0, 1);

        int prev_tok = (token_ids[ctx_len] >= 0) ? token_ids[ctx_len] : 0;
        float m_vec[256];
        float *m_bias = (float *)malloc((size_t)K31_VOCAB * sizeof(float));
        for (int qi = 0; qi < q_len; qi++) {
            k31_embed_lookup_row(&ds->mw1, prev_tok, m_vec);
            k31_matmul(&ds->mw2, m_vec, m_bias, 1, 0, 1);

            float *lt = base_lg + (size_t)qi * K31_VOCAB;
            int best_id = 0; float best_v = -1e30f;
            for (int v = 0; v < K31_VOCAB; v++) {
                float val = lt[v] + m_bias[v];
                if (val > best_v) { best_v = val; best_id = v; }
            }
            out_draft_tokens[ctx_len + qi] = best_id;
            if (out_confidences) {
                const float *ht = q_final + (size_t)qi * K31_D;
                float cdot = ds->conf_b[0];
                for (int d = 0; d < K31_D; d++) cdot += ht[d] * ds->conf_w[d];
                for (int d = 0; d < 256;   d++) cdot += m_vec[d] * ds->conf_w[K31_D + d];
                out_confidences[ctx_len + qi] = 1.0f / (1.0f + expf(-cdot));
            }
            prev_tok = best_id;
        }
        free(m_bias);
        free(base_lg);
    }

    if (!out_dspark_hidden_7168) free(q_final);
    if (ctx_fc) free(ctx_fc);
    if (ctx_nm) free(ctx_nm);
    free(cur); free(nxt);
    return (float)((omp_get_wtime() - t0) * 1000.0);
}

static int k31_argmax_f32(const float *v, int n) {
    int bi = 0; float bv = v[0];
    for (int i = 1; i < n; i++) if (v[i] > bv) { bv = v[i]; bi = i; }
    return bi;
}

/* ============================================================================================================
 * Stateful Incremental Cache (`KDA` 96x128x128 Delta State + 3-Tap Conv Ring, `DSA` MLA KV Cache, `DSpark` KV)
 * with Zero-IO ~1.5ms Causal Rollback on Speculative Draft Rejection
 * ============================================================================================================ */

typedef struct {
    float *S;         /* [96 * 128 * 128] = 6 MB */
    float *conv_q;    /* [3 * 12288] */
    float *conv_k;    /* [3 * 12288] */
    float *conv_v;    /* [3 * 12288] */
    float *S_bak;
    float *cq_bak, *ck_bak, *cv_bak;
    float *step_kn;   /* [8 * 12288] normalized step keys */
    float *step_vc;   /* [8 * 12288] post-conv step values */
    float *step_dec;  /* [8 * 12288] per-channel exp(b_gate) */
    float *step_beta; /* [8 * 96]    per-head sigmoid(beta) */
    float *step_qr;   /* [8 * 12288] */
    float *step_kr;   /* [8 * 12288] */
    float *step_vr;   /* [8 * 12288] */
} k31_kda_cache_t;

typedef struct {
    float *kv_exp;  /* [cap_seq * 24576] */
    float *k_rope;  /* [cap_seq * 64] */
} k31_dsa_cache_t;

static struct {
    int   alloc_cap;
    int   cur_len;
    int   ds_len;
    int   last_lcp;
    int   n_sel_cached;
    int   sel_lids[K31_MAX_LAYERS];
    int  *cached_ids;                 /* [alloc_cap] */
    float *feat_cache;                /* [alloc_cap * 5 * K31_D] */
    k31_kda_cache_t kda[K31_MAX_LAYERS];
    k31_dsa_cache_t dsa[K31_MAX_LAYERS];
    float *ds_k[K31_DS_LAYERS];       /* [alloc_cap * 1024] */
    float *ds_v[K31_DS_LAYERS];       /* [alloc_cap * 1024] */
} g_k31_cache;

K31_API int kimi_k31_last_lcp_tokens(void) {
    return g_k31_cache.last_lcp;
}

K31_API void kimi_k31_cache_reset(void) {
    g_k31_cache.cur_len = 0;
    g_k31_cache.ds_len = 0;
    g_k31_cache.last_lcp = 0;
    for (int l = 0; l < K31_MAX_LAYERS; l++) {
        if (g_k31_cache.kda[l].S) {
            memset(g_k31_cache.kda[l].S,      0, (size_t)96 * 128 * 128 * sizeof(float));
            memset(g_k31_cache.kda[l].conv_q, 0, (size_t)3 * 12288 * sizeof(float));
            memset(g_k31_cache.kda[l].conv_k, 0, (size_t)3 * 12288 * sizeof(float));
            memset(g_k31_cache.kda[l].conv_v, 0, (size_t)3 * 12288 * sizeof(float));
        }
    }
}

static void k31_cache_ensure_cap(int needed_seq, const int *layer_ids, int n_sel_layers) {
    if (n_sel_layers > K31_MAX_LAYERS) n_sel_layers = K31_MAX_LAYERS;
    int same_layers = (g_k31_cache.n_sel_cached == n_sel_layers) &&
                      (memcmp(g_k31_cache.sel_lids, layer_ids, (size_t)n_sel_layers * sizeof(int)) == 0);
    if (!same_layers) {
        g_k31_cache.n_sel_cached = n_sel_layers;
        memcpy(g_k31_cache.sel_lids, layer_ids, (size_t)n_sel_layers * sizeof(int));
        kimi_k31_cache_reset();
    }

    for (int i = 0; i < n_sel_layers; i++) {
        int lid = layer_ids[i];
        if (lid < 0 || lid >= g_k31.n_layers) continue;
        if (g_k31.layers[lid].kind != 2 && !g_k31_cache.kda[lid].S) {
            k31_kda_cache_t *kc = &g_k31_cache.kda[lid];
            kc->S         = (float *)calloc((size_t)96 * 128 * 128, sizeof(float));
            kc->conv_q    = (float *)calloc((size_t)3 * 12288, sizeof(float));
            kc->conv_k    = (float *)calloc((size_t)3 * 12288, sizeof(float));
            kc->conv_v    = (float *)calloc((size_t)3 * 12288, sizeof(float));
            kc->S_bak     = (float *)malloc((size_t)96 * 128 * 128 * sizeof(float));
            kc->cq_bak    = (float *)malloc((size_t)3 * 12288 * sizeof(float));
            kc->ck_bak    = (float *)malloc((size_t)3 * 12288 * sizeof(float));
            kc->cv_bak    = (float *)malloc((size_t)3 * 12288 * sizeof(float));
            kc->step_kn   = (float *)malloc((size_t)8 * 12288 * sizeof(float));
            kc->step_vc   = (float *)malloc((size_t)8 * 12288 * sizeof(float));
            kc->step_dec  = (float *)malloc((size_t)8 * 12288 * sizeof(float));
            kc->step_beta = (float *)malloc((size_t)8 * 96 * sizeof(float));
            kc->step_qr   = (float *)malloc((size_t)8 * 12288 * sizeof(float));
            kc->step_kr   = (float *)malloc((size_t)8 * 12288 * sizeof(float));
            kc->step_vr   = (float *)malloc((size_t)8 * 12288 * sizeof(float));
        }
    }

    if (needed_seq <= g_k31_cache.alloc_cap) return;
    int new_cap = g_k31_cache.alloc_cap ? g_k31_cache.alloc_cap : 1536;
    while (new_cap < needed_seq) new_cap *= 2;

    g_k31_cache.cached_ids = (int *)realloc(g_k31_cache.cached_ids, (size_t)new_cap * sizeof(int));
    g_k31_cache.feat_cache = (float *)realloc(g_k31_cache.feat_cache, (size_t)new_cap * 5 * K31_D * sizeof(float));
    for (int l = 0; l < g_k31.n_layers; l++) {
        if (g_k31.layers[l].kind == 2) {
            g_k31_cache.dsa[l].kv_exp = (float *)realloc(g_k31_cache.dsa[l].kv_exp, (size_t)new_cap * 24576 * sizeof(float));
            g_k31_cache.dsa[l].k_rope = (float *)realloc(g_k31_cache.dsa[l].k_rope, (size_t)new_cap * 64 * sizeof(float));
        }
    }
    if (g_k31.dspark.loaded) {
        for (int dl = 0; dl < K31_DS_LAYERS; dl++) {
            g_k31_cache.ds_k[dl] = (float *)realloc(g_k31_cache.ds_k[dl], (size_t)new_cap * 1024 * sizeof(float));
            g_k31_cache.ds_v[dl] = (float *)realloc(g_k31_cache.ds_v[dl], (size_t)new_cap * 1024 * sizeof(float));
        }
    }
    g_k31_cache.alloc_cap = new_cap;
}

static void k31_update_conv_ring(
    float *ring, const float *old_ring, const float *raw_step,
    int base_pos, int keep_n
) {
    for (int slot = 0; slot < 3; slot++) {
        int gpos = (base_pos + keep_n) - 3 + slot;
        float *dst = ring + (size_t)slot * 12288;
        if (gpos < 0) {
            memset(dst, 0, 12288 * sizeof(float));
        } else if (gpos < base_pos) {
            int old_slot = gpos - (base_pos - 3);
            memcpy(dst, old_ring + (size_t)old_slot * 12288, 12288 * sizeof(float));
        } else {
            memcpy(dst, raw_step + (size_t)(gpos - base_pos) * 12288, 12288 * sizeof(float));
        }
    }
}

static void k31_run_kda_cached(
    const k31_kda_t *kda, k31_kda_cache_t *kc, const float *u, float *attn_proj,
    int base_pos, int T, int save_for_rollback
) {
    float *q_raw = (float *)malloc((size_t)T * 12288 * sizeof(float));
    float *k_raw = (float *)malloc((size_t)T * 12288 * sizeof(float));
    float *v_raw = (float *)malloc((size_t)T * 12288 * sizeof(float));
    float *q_c   = (float *)malloc((size_t)T * 12288 * sizeof(float));
    float *k_c   = (float *)malloc((size_t)T * 12288 * sizeof(float));
    float *v_c   = (float *)malloc((size_t)T * 12288 * sizeof(float));
    float *b_raw = (float *)malloc((size_t)T * 96 * sizeof(float));
    float *fa_o  = (float *)malloc((size_t)T * 128 * sizeof(float));
    float *fb_o  = (float *)malloc((size_t)T * 12288 * sizeof(float));
    float *g_raw = (float *)malloc((size_t)T * 12288 * sizeof(float));
    float *a_out = (float *)malloc((size_t)T * 12288 * sizeof(float));

    if (save_for_rollback && T <= 8) {
        memcpy(kc->S_bak,  kc->S,      (size_t)96 * 128 * 128 * sizeof(float));
        memcpy(kc->cq_bak, kc->conv_q, (size_t)3 * 12288 * sizeof(float));
        memcpy(kc->ck_bak, kc->conv_k, (size_t)3 * 12288 * sizeof(float));
        memcpy(kc->cv_bak, kc->conv_v, (size_t)3 * 12288 * sizeof(float));
    }

    k31_matmul(&kda->q,  u,    q_raw, T, 0, 1);
    k31_matmul(&kda->k,  u,    k_raw, T, 0, 1);
    k31_matmul(&kda->v,  u,    v_raw, T, 0, 1);
    k31_matmul(&kda->b,  u,    b_raw, T, 0, 1);
    k31_matmul(&kda->fa, u,    fa_o,  T, 0, 1);
    k31_matmul(&kda->fb, fa_o, fb_o,  T, 0, 1);
    k31_matmul(&kda->g,  u,    g_raw, T, 0, 1);

    #pragma omp parallel for schedule(static)
    for (int c = 0; c < 12288; c++) {
        const float *qw = kda->qc + c * 4, *kw = kda->kc + c * 4, *vw = kda->vc + c * 4;
        for (int t = 0; t < T; t++) {
            float sq = 0.0f, sk = 0.0f, sv = 0.0f;
            for (int tap = 0; tap < 4; tap++) {
                int rel = t - 3 + tap;
                int gpos = base_pos + rel;
                if (gpos < 0) continue;
                if (rel >= 0) {
                    sq += q_raw[(size_t)rel * 12288 + c] * qw[tap];
                    sk += k_raw[(size_t)rel * 12288 + c] * kw[tap];
                    sv += v_raw[(size_t)rel * 12288 + c] * vw[tap];
                } else {
                    int slot = 3 + rel;
                    sq += kc->conv_q[(size_t)slot * 12288 + c] * qw[tap];
                    sk += kc->conv_k[(size_t)slot * 12288 + c] * kw[tap];
                    sv += kc->conv_v[(size_t)slot * 12288 + c] * vw[tap];
                }
            }
            q_c[(size_t)t * 12288 + c] = sq / (1.0f + expf(-sq));
            k_c[(size_t)t * 12288 + c] = sk / (1.0f + expf(-sk));
            v_c[(size_t)t * 12288 + c] = sv / (1.0f + expf(-sv));
        }
    }

    if (save_for_rollback && T <= 8) {
        memcpy(kc->step_qr, q_raw, (size_t)T * 12288 * sizeof(float));
        memcpy(kc->step_kr, k_raw, (size_t)T * 12288 * sizeof(float));
        memcpy(kc->step_vr, v_raw, (size_t)T * 12288 * sizeof(float));
        memcpy(kc->step_vc, v_c,   (size_t)T * 12288 * sizeof(float));
    }

    float *old_cq = (float *)malloc((size_t)3 * 12288 * sizeof(float));
    float *old_ck = (float *)malloc((size_t)3 * 12288 * sizeof(float));
    float *old_cv = (float *)malloc((size_t)3 * 12288 * sizeof(float));
    memcpy(old_cq, kc->conv_q, (size_t)3 * 12288 * sizeof(float));
    memcpy(old_ck, kc->conv_k, (size_t)3 * 12288 * sizeof(float));
    memcpy(old_cv, kc->conv_v, (size_t)3 * 12288 * sizeof(float));
    k31_update_conv_ring(kc->conv_q, old_cq, q_raw, base_pos, T);
    k31_update_conv_ring(kc->conv_k, old_ck, k_raw, base_pos, T);
    k31_update_conv_ring(kc->conv_v, old_cv, v_raw, base_pos, T);
    free(old_cq); free(old_ck); free(old_cv);

    const float inv_s128 = 1.0f / sqrtf(128.0f);
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < 96; h++) {
        float *Sh = kc->S + (size_t)h * 128 * 128; /* [V=128, K=128] */
        const float b_a = -kda->A_neg[h];
        for (int t = 0; t < T; t++) {
            const float *qh = q_c + (size_t)t * 12288 + h * 128;
            const float *kh = k_c + (size_t)t * 12288 + h * 128;
            const float *vh = v_c + (size_t)t * 12288 + h * 128;
            const float *dr = fb_o + (size_t)t * 12288 + h * 128;
            double sq = 0.0, sk = 0.0;
            for (int d = 0; d < 128; d++) { sq += (double)qh[d] * qh[d]; sk += (double)kh[d] * kh[d]; }
            float iq = (1.0f / sqrtf((float)sq + 1e-6f)) * inv_s128;
            float ik = 1.0f / sqrtf((float)sk + 1e-6f);
            float beta = 1.0f / (1.0f + expf(-b_raw[t * 96 + h]));

            float qn[128], kn[128], dec[128], yh[128];
            for (int d = 0; d < 128; d++) {
                float g_arg = b_a * (dr[d] + kda->dtb[h * 128 + d]);
                float b_gate = -5.0f / (1.0f + expf(-g_arg));
                dec[d] = expf(b_gate);
                qn[d]  = qh[d] * iq;
                kn[d]  = kh[d] * ik;
            }
            if (save_for_rollback && T <= 8) {
                memcpy(kc->step_kn  + (size_t)t * 12288 + h * 128, kn,  128 * sizeof(float));
                memcpy(kc->step_dec + (size_t)t * 12288 + h * 128, dec, 128 * sizeof(float));
                kc->step_beta[t * 96 + h] = beta;
            }

            /* Exact KDA Householder / Delta-Rule State Update:
             * S[v, k] *= dec[k]; vd = beta * (vh[v] - sum_k S[v, k]*kn[k]); S[v, k] += vd*kn[k]; yh[v] = sum_k S[v, k]*qn[k]
             */
            for (int v = 0; v < 128; v++) {
                float *Srow = Sh + v * 128;
                float pred = 0.0f;
                #pragma omp simd reduction(+:pred)
                for (int k = 0; k < 128; k++) {
                    float s_dec = Srow[k] * dec[k];
                    Srow[k] = s_dec;
                    pred += s_dec * kn[k];
                }
                float vd = beta * (vh[v] - pred);
                float yv = 0.0f;
                #pragma omp simd reduction(+:yv)
                for (int k = 0; k < 128; k++) {
                    float s_new = Srow[k] + vd * kn[k];
                    Srow[k] = s_new;
                    yv += s_new * qn[k];
                }
                yh[v] = yv;
            }

            double ss = 0.0;
            for (int d = 0; d < 128; d++) ss += (double)yh[d] * yh[d];
            float inv = 1.0f / sqrtf((float)(ss / 128.0) + 1e-5f);
            float *dst = a_out + (size_t)t * 12288 + h * 128;
            const float *gr = g_raw + (size_t)t * 12288 + h * 128;
            for (int d = 0; d < 128; d++) {
                float gv = 1.0f / (1.0f + expf(-gr[d]));
                dst[d] = (yh[d] * inv * kda->o_norm[d]) * gv;
            }
        }
    }

    k31_matmul(&kda->o, a_out, attn_proj, T, 0, 1);
    free(q_raw); free(k_raw); free(v_raw); free(q_c); free(k_c); free(v_c);
    free(b_raw); free(fa_o); free(fb_o); free(g_raw); free(a_out);
}

/* Zero-IO ~1.5ms KDA Delta-Rule state rollback when only `keep_n < T_step` speculative tokens were accepted */
static void k31_rollback_kda_keep(k31_kda_cache_t *kc, int base_pos, int keep_n) {
    memcpy(kc->S, kc->S_bak, (size_t)96 * 128 * 128 * sizeof(float));
    k31_update_conv_ring(kc->conv_q, kc->cq_bak, kc->step_qr, base_pos, keep_n);
    k31_update_conv_ring(kc->conv_k, kc->ck_bak, kc->step_kr, base_pos, keep_n);
    k31_update_conv_ring(kc->conv_v, kc->cv_bak, kc->step_vr, base_pos, keep_n);

    #pragma omp parallel for schedule(static)
    for (int h = 0; h < 96; h++) {
        float *Sh = kc->S + (size_t)h * 128 * 128;
        for (int t = 0; t < keep_n; t++) {
            const float *kn  = kc->step_kn  + (size_t)t * 12288 + h * 128;
            const float *vh  = kc->step_vc  + (size_t)t * 12288 + h * 128;
            const float *dec = kc->step_dec + (size_t)t * 12288 + h * 128;
            const float beta = kc->step_beta[t * 96 + h];
            for (int v = 0; v < 128; v++) {
                float *Srow = Sh + v * 128;
                float pred = 0.0f;
                #pragma omp simd reduction(+:pred)
                for (int k = 0; k < 128; k++) {
                    float s_dec = Srow[k] * dec[k];
                    Srow[k] = s_dec;
                    pred += s_dec * kn[k];
                }
                float vd = beta * (vh[v] - pred);
                #pragma omp simd
                for (int k = 0; k < 128; k++) Srow[k] += vd * kn[k];
            }
        }
    }
}

static void k31_run_dsa_cached(
    const k31_dsa_t *dsa, k31_dsa_cache_t *dc, const float *u, float *attn_proj,
    int base_pos, int T
) {
    float *q_lat  = (float *)malloc((size_t)T * 1536 * sizeof(float));
    float *q_all  = (float *)malloc((size_t)T * 18432 * sizeof(float));
    float *kv_all = (float *)malloc((size_t)T * 576 * sizeof(float));
    float *kv_lat = (float *)malloc((size_t)T * 512 * sizeof(float));
    float *g_raw  = (float *)malloc((size_t)T * 12288 * sizeof(float));
    float *a_out  = (float *)malloc((size_t)T * 12288 * sizeof(float));

    k31_matmul(&dsa->qa, u, q_lat, T, 0, 1);
    rms_norm_batch(q_lat, dsa->qa_ln, q_lat, T, 1536, 1e-5f);
    k31_matmul(&dsa->qb, q_lat, q_all, T, 0, 1);

    k31_matmul(&dsa->kva, u, kv_all, T, 0, 1);
    for (int t = 0; t < T; t++) {
        rms_norm_row(kv_all + (size_t)t * 576, dsa->kva_ln, kv_lat + (size_t)t * 512, 512, 1e-5f);
        memcpy(dc->k_rope + (size_t)(base_pos + t) * 64, kv_all + (size_t)t * 576 + 512, 64 * sizeof(float));
    }
    k31_matmul(&dsa->kvb, kv_lat, dc->kv_exp + (size_t)base_pos * 24576, T, 0, 1);
    k31_matmul(&dsa->g, u, g_raw, T, 0, 1);

    const float mla_scale = 1.0f / sqrtf(192.0f);
    const int tot_seq = base_pos + T;
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < 96; h++) {
        float *scores = (float *)malloc((size_t)tot_seq * sizeof(float));
        for (int t = 0; t < T; t++) {
            int g_end = base_pos + t;
            const float *qn = q_all + (size_t)t * 18432 + h * 192;
            const float *qr = qn + 128;
            float mx = -1e30f;
            for (int s = 0; s <= g_end; s++) {
                const float *kn = dc->kv_exp + (size_t)s * 24576 + h * 256;
                const float *kr = dc->k_rope + (size_t)s * 64;
                float dot = 0.0f;
                for (int d = 0; d < 128; d++) dot += qn[d] * kn[d];
                for (int d = 0; d < 64;  d++) dot += qr[d] * kr[d];
                scores[s] = dot * mla_scale;
                if (scores[s] > mx) mx = scores[s];
            }
            float sw = 0.0f;
            for (int s = 0; s <= g_end; s++) { scores[s] = expf(scores[s] - mx); sw += scores[s]; }
            for (int s = 0; s <= g_end; s++) scores[s] /= (sw + 1e-12f);

            float yh[128] = {0};
            for (int s = 0; s <= g_end; s++) {
                const float *vs = dc->kv_exp + (size_t)s * 24576 + h * 256 + 128;
                float ws = scores[s];
                for (int d = 0; d < 128; d++) yh[d] += ws * vs[d];
            }
            float *dst = a_out + (size_t)t * 12288 + h * 128;
            const float *gr = g_raw + (size_t)t * 12288 + h * 128;
            for (int d = 0; d < 128; d++) dst[d] = yh[d] * (1.0f / (1.0f + expf(-gr[d])));
        }
        free(scores);
    }

    k31_matmul(&dsa->o, a_out, attn_proj, T, 0, 1);
    free(q_lat); free(q_all); free(kv_all); free(kv_lat); free(g_raw); free(a_out);
}

static void k31_forward_cached_step(
    const int *token_ids, int base_pos, int T, const int *layer_ids, int n_sel_layers,
    int save_for_rollback, float *out_final_hidden, float *out_dspark_feats_35840
) {
    float *prefix_sum = (float *)malloc((size_t)T * K31_D * sizeof(float));
    float *delta      = (float *)malloc((size_t)T * K31_D * sizeof(float));
    float *nxt_delta  = (float *)malloc((size_t)T * K31_D * sizeof(float));
    float *blocks     = (float *)calloc((size_t)T * K31_MAX_BLOCKS * K31_D, sizeof(float));
    float *mixed      = (float *)malloc((size_t)T * K31_D * sizeof(float));
    float *u          = (float *)malloc((size_t)T * K31_D * sizeof(float));
    float *attn_proj  = (float *)malloc((size_t)T * K31_D * sizeof(float));
    float *post       = (float *)malloc((size_t)T * K31_D * sizeof(float));
    float *agg_final  = (float *)malloc((size_t)T * K31_D * sizeof(float));

    #pragma omp parallel for schedule(static) if (T >= 4)
    for (int t = 0; t < T; t++) {
        k31_embed_lookup_row(&g_k31.emb_mat, token_ids[t], prefix_sum + (size_t)t * K31_D);
    }

    const int tgt_lids[5] = {7, 23, 67, 83, 91};
    int feat_captured[5] = {0, 0, 0, 0, 0};

    int has_delta = 0;
    int max_lid = 0;
    for (int i = 0; i < n_sel_layers; i++) {
        int lid = layer_ids[i];
        if (lid < 0 || lid >= g_k31.n_layers) continue;
        if (lid > max_lid) max_lid = lid;
        k31_layer_t *lyr = &g_k31.layers[lid];
        const int is_write    = (lid % K31_ATTN_RES_BLK == 0) ? 1 : 0;
        const int write_idx   = lid / K31_ATTN_RES_BLK;
        const int prev_blocks = (lid + K31_ATTN_RES_BLK - 1) / K31_ATTN_RES_BLK;
        const int mlp_blocks  = prev_blocks + is_write;

        if (has_delta) {
            for (int idx = 0; idx < T * K31_D; idx++) prefix_sum[idx] += delta[idx];
        }
        if (is_write && write_idx < K31_MAX_BLOCKS) {
            for (int t = 0; t < T; t++) {
                memcpy(blocks + ((size_t)t * K31_MAX_BLOCKS + (size_t)write_idx) * K31_D,
                       prefix_sum + (size_t)t * K31_D, K31_D * sizeof(float));
            }
        }
        k31_attn_res_mix(blocks, prev_blocks, prefix_sum, lyr->sa_rn, lyr->sa_rp, mixed, T);
        rms_norm_batch(mixed, lyr->in_ln, u, T, K31_D, 1e-5f);

        if (lyr->kind == 2) k31_run_dsa_cached(&lyr->dsa, &g_k31_cache.dsa[lid], u, attn_proj, base_pos, T);
        else                k31_run_kda_cached(&lyr->kda, &g_k31_cache.kda[lid], u, attn_proj, base_pos, T, save_for_rollback);

        if (is_write) {
            memcpy(prefix_sum, attn_proj, (size_t)T * K31_D * sizeof(float));
        } else {
            for (int idx = 0; idx < T * K31_D; idx++) prefix_sum[idx] += attn_proj[idx];
        }
        k31_attn_res_mix(blocks, mlp_blocks, prefix_sum, lyr->mlp_rn, lyr->mlp_rp, mixed, T);
        rms_norm_batch(mixed, lyr->post_ln, post, T, K31_D, 1e-5f);

        if (lyr->kind == 0) {
            float *dhid = (float *)malloc((size_t)T * K31_DENSE_INT * sizeof(float));
            k31_matmul(&lyr->dense_w13, post, dhid, T, 2, 1);
            k31_matmul(&lyr->dense_dn,  dhid, nxt_delta, T, 0, 1);
            free(dhid);
        } else {
            k31_run_moe(&lyr->moe, &lyr->ex, post, nxt_delta, T);
        }

        float *sw = delta; delta = nxt_delta; nxt_delta = sw;
        has_delta = 1;

        if (out_dspark_feats_35840) {
            for (int m = 0; m < 5; m++) {
                if (lid == tgt_lids[m]) {
                    feat_captured[m] = 1;
                    for (int t = 0; t < T; t++) {
                        float *dst = out_dspark_feats_35840 + ((size_t)t * 5 + (size_t)m) * K31_D;
                        const float *ps = prefix_sum + (size_t)t * K31_D;
                        const float *dl = delta      + (size_t)t * K31_D;
                        for (int d = 0; d < K31_D; d++) dst[d] = ps[d] + dl[d];
                    }
                }
            }
        }
    }

    if (has_delta) {
        for (int idx = 0; idx < T * K31_D; idx++) prefix_sum[idx] += delta[idx];
    }
    if (out_dspark_feats_35840) {
        for (int m = 0; m < 5; m++) {
            if (!feat_captured[m]) {
                for (int t = 0; t < T; t++) {
                    memcpy(out_dspark_feats_35840 + ((size_t)t * 5 + (size_t)m) * K31_D,
                           prefix_sum + (size_t)t * K31_D, K31_D * sizeof(float));
                }
            }
        }
    }

    int final_blocks = (max_lid + 1 + K31_ATTN_RES_BLK - 1) / K31_ATTN_RES_BLK;
    if (final_blocks > K31_MAX_BLOCKS) final_blocks = K31_MAX_BLOCKS;
    k31_attn_res_mix(blocks, final_blocks, prefix_sum, g_k31.out_res_norm, g_k31.out_res_proj, agg_final, T);
    rms_norm_batch(agg_final, g_k31.norm_w, out_final_hidden, T, K31_D, 1e-5f);

    free(prefix_sum); free(delta); free(nxt_delta); free(blocks);
    free(mixed); free(u); free(attn_proj); free(post); free(agg_final);
}

/* Stateful DFlash context-KV insertion (`rms_norm(fc(feats), hid_norm)` -> per-layer K/V with YaRN RoPE) */
static void k31_dspark_commit_ctx_kv(const float *step_feats_35840, int base_pos, int T) {
    if (T <= 0) return;
    const k31_dspark_t *ds = &g_k31.dspark;
    float *fc_out = (float *)malloc((size_t)T * K31_D * sizeof(float));
    float *ctx_nm = (float *)malloc((size_t)T * K31_D * sizeof(float));

    k31_matmul(&ds->fc, step_feats_35840, fc_out, T, 0, 1);
    rms_norm_batch(fc_out, ds->hid_norm, ctx_nm, T, K31_D, 1e-5f);

    for (int l = 0; l < K31_DS_LAYERS; l++) {
        const k31_ds_layer_t *dl = &ds->layers[l];
        float *k_dst = g_k31_cache.ds_k[l] + (size_t)base_pos * 1024;
        float *v_dst = g_k31_cache.ds_v[l] + (size_t)base_pos * 1024;
        k31_matmul(&dl->k, ctx_nm, k_dst, T, 0, 1);
        k31_matmul(&dl->v, ctx_nm, v_dst, T, 0, 1);

        #pragma omp parallel for schedule(static) if (T >= 2)
        for (int t = 0; t < T; t++) {
            int gpos = base_pos + t;
            float cs[32], sn[32];
            for (int d = 0; d < 32; d++) {
                float ang = (float)gpos * ds->inv_freq[d];
                cs[d] = cosf(ang) * ds->yarn_mscale;
                sn[d] = sinf(ang) * ds->yarn_mscale;
            }
            for (int kh = 0; kh < 16; kh++) {
                float *khp = k_dst + (size_t)t * 1024 + kh * 64;
                rms_norm_row(khp, dl->kn, khp, 64, 1e-5f);
                for (int d = 0; d < 32; d++) {
                    float a = khp[d], b = khp[32 + d];
                    khp[d] = a * cs[d] - b * sn[d];
                    khp[32 + d] = b * cs[d] + a * sn[d];
                }
            }
        }
    }
    g_k31_cache.ds_len = base_pos + T;
    free(fc_out); free(ctx_nm);
}

/* Stateful DFlash 5-Layer Non-Causal Query-Block Draft (`[cur_tok, mask, ..., mask]` -> `K` drafts + confidences) */
static void k31_dspark_draft_query_block(
    int anchor_tok, int base_pos, int K, int *out_draft_tokens, float *out_confidences
) {
    const k31_dspark_t *ds = &g_k31.dspark;
    float *cur     = (float *)malloc((size_t)K * K31_D * sizeof(float));
    float *nxt     = (float *)malloc((size_t)K * K31_D * sizeof(float));
    float *final_h = (float *)malloc((size_t)K * K31_D * sizeof(float));

    k31_embed_lookup_row(&g_k31.emb_mat, anchor_tok, cur);
    for (int qi = 1; qi < K; qi++) {
        k31_embed_lookup_row(&g_k31.emb_mat, 163824, cur + (size_t)qi * K31_D);
    }

    const int tot_seq = base_pos + K;
    for (int l = 0; l < K31_DS_LAYERS; l++) {
        const k31_ds_layer_t *dl = &ds->layers[l];
        float *u  = (float *)malloc((size_t)K * K31_D * sizeof(float));
        float *q  = (float *)malloc((size_t)K * 4096 * sizeof(float));
        float *ao = (float *)malloc((size_t)K * 4096 * sizeof(float));
        float *bo = (float *)malloc((size_t)K * K31_D * sizeof(float));
        float *fh = (float *)malloc((size_t)K * K31_DS_INTER * sizeof(float));
        float *fo = (float *)malloc((size_t)K * K31_D * sizeof(float));

        float *k_dst = g_k31_cache.ds_k[l] + (size_t)base_pos * 1024;
        float *v_dst = g_k31_cache.ds_v[l] + (size_t)base_pos * 1024;

        rms_norm_batch(cur, dl->in_ln, u, K, K31_D, 1e-5f);
        k31_matmul(&dl->q, u, q,     K, 0, 1);
        k31_matmul(&dl->k, u, k_dst, K, 0, 1);
        k31_matmul(&dl->v, u, v_dst, K, 0, 1);

        for (int qi = 0; qi < K; qi++) {
            int gpos = base_pos + qi;
            float cs[32], sn[32];
            for (int d = 0; d < 32; d++) {
                float ang = (float)gpos * ds->inv_freq[d];
                cs[d] = cosf(ang) * ds->yarn_mscale;
                sn[d] = sinf(ang) * ds->yarn_mscale;
            }
            for (int h = 0; h < 64; h++) {
                float *qh = q + (size_t)qi * 4096 + h * 64;
                rms_norm_row(qh, dl->qn, qh, 64, 1e-5f);
                for (int d = 0; d < 32; d++) {
                    float a = qh[d], b = qh[32 + d];
                    qh[d] = a * cs[d] - b * sn[d];
                    qh[32 + d] = b * cs[d] + a * sn[d];
                }
            }
            for (int kh = 0; kh < 16; kh++) {
                float *khp = k_dst + (size_t)qi * 1024 + kh * 64;
                rms_norm_row(khp, dl->kn, khp, 64, 1e-5f);
                for (int d = 0; d < 32; d++) {
                    float a = khp[d], b = khp[32 + d];
                    khp[d] = a * cs[d] - b * sn[d];
                    khp[32 + d] = b * cs[d] + a * sn[d];
                }
            }
        }

        const float scale = 1.0f / 8.0f;
        const float *k_all = g_k31_cache.ds_k[l];
        const float *v_all = g_k31_cache.ds_v[l];

        #pragma omp parallel for schedule(static)
        for (int h = 0; h < 64; h++) {
            int kv_h = h / 4;
            float *sc = (float *)malloc((size_t)tot_seq * sizeof(float));
            for (int qi = 0; qi < K; qi++) {
                const float *qh = q + (size_t)qi * 4096 + h * 64;
                float mx = -1e30f;
                for (int j = 0; j < tot_seq; j++) {
                    const float *kj = k_all + (size_t)j * 1024 + kv_h * 64;
                    float dot = 0.0f;
                    for (int d = 0; d < 64; d++) dot += qh[d] * kj[d];
                    sc[j] = dot * scale;
                    if (sc[j] > mx) mx = sc[j];
                }
                float sw = 0.0f;
                for (int j = 0; j < tot_seq; j++) { sc[j] = expf(sc[j] - mx); sw += sc[j]; }
                for (int j = 0; j < tot_seq; j++) sc[j] /= (sw + 1e-12f);

                float *yh = ao + (size_t)qi * 4096 + h * 64;
                memset(yh, 0, 64 * sizeof(float));
                for (int j = 0; j < tot_seq; j++) {
                    const float *vj = v_all + (size_t)j * 1024 + kv_h * 64;
                    float w = sc[j];
                    for (int d = 0; d < 64; d++) yh[d] += w * vj[d];
                }
            }
            free(sc);
        }

        k31_matmul(&dl->o, ao, bo, K, 0, 1);
        for (int i = 0; i < K * K31_D; i++) bo[i] += cur[i];
        rms_norm_batch(bo, dl->post_ln, u, K, K31_D, 1e-5f);
        k31_matmul(&dl->w13, u,  fh, K, 1, 1);
        k31_matmul(&dl->wdn, fh, fo, K, 0, 1);
        for (int i = 0; i < K * K31_D; i++) nxt[i] = bo[i] + fo[i];

        free(u); free(q); free(ao); free(bo); free(fh); free(fo);
        float *tmp = cur; cur = nxt; nxt = tmp;
    }

    rms_norm_batch(cur, ds->final_norm, final_h, K, K31_D, 1e-5f);
    float *base_lg = (float *)malloc((size_t)K * K31_VOCAB * sizeof(float));
    k31_matmul(&g_k31.lm_mat, final_h, base_lg, K, 0, 1);

    int prev_tok = anchor_tok;
    float m_vec[256];
    float *m_bias = (float *)malloc((size_t)K31_VOCAB * sizeof(float));
    for (int qi = 0; qi < K; qi++) {
        k31_embed_lookup_row(&ds->mw1, prev_tok, m_vec);
        k31_matmul(&ds->mw2, m_vec, m_bias, 1, 0, 1);

        float *lt = base_lg + (size_t)qi * K31_VOCAB;
        int best_id = 0; float best_v = -1e30f;
        for (int v = 0; v < K31_VOCAB; v++) {
            float val = lt[v] + m_bias[v];
            if (val > best_v) { best_v = val; best_id = v; }
        }
        out_draft_tokens[qi] = best_id;
        if (out_confidences) {
            const float *ht = final_h + (size_t)qi * K31_D;
            float cdot = ds->conf_b[0];
            for (int d = 0; d < K31_D; d++) cdot += ht[d] * ds->conf_w[d];
            for (int d = 0; d < 256;   d++) cdot += m_vec[d] * ds->conf_w[K31_D + d];
            out_confidences[qi] = 1.0f / (1.0f + expf(-cdot));
        }
        prev_tok = best_id;
    }

    free(m_bias); free(base_lg); free(cur); free(nxt); free(final_h);
}

/* Reference stateless speculative generator preserved for `exact_greedy_match` unit verification */
K31_API int kimi_k31_generate_speculative_stateless(
    const int *prompt_ids, int prompt_len, const int *layer_ids, int n_sel_layers,
    int max_new_tokens, int eos_id, int draft_k,
    int *out_gen_ids, double *out_prefill_ms, double *out_decode_ms,
    int *out_accepted_drafts, int *out_total_drafts
) {
    if (!g_k31.ready || prompt_len <= 0 || n_sel_layers <= 0 || max_new_tokens <= 0) return -1;
    if (n_sel_layers > K31_MAX_LAYERS) n_sel_layers = K31_MAX_LAYERS;
    if (draft_k < 0) draft_k = 0;
    if (draft_k > 5) draft_k = 5;
    if (!g_k31.dspark.loaded) draft_k = 0;

    int max_tot = prompt_len + max_new_tokens + draft_k + 4;
    int   *seq   = (int *)malloc((size_t)max_tot * sizeof(int));
    float *h_buf = (float *)malloc((size_t)max_tot * K31_D * sizeof(float));
    float *f_buf = (float *)malloc((size_t)max_tot * 5 * K31_D * sizeof(float));
    float *lg_1  = (float *)malloc((size_t)(draft_k + 2) * K31_VOCAB * sizeof(float));
    memcpy(seq, prompt_ids, (size_t)prompt_len * sizeof(int));
    int cur_len = prompt_len;

    double t0 = omp_get_wtime();
    kimi_k31_hg4_forward_layers_ex(seq, cur_len, layer_ids, n_sel_layers, h_buf, NULL, (draft_k > 0) ? f_buf : NULL);
    k31_matmul(&g_k31.lm_mat, h_buf + (size_t)(cur_len - 1) * K31_D, lg_1, 1, 0, 1);
    *out_prefill_ms = (omp_get_wtime() - t0) * 1000.0;

    double td0 = omp_get_wtime();
    int gen_n = 0, acc_tot = 0, draft_tot = 0;
    int cur_tok = k31_argmax_f32(lg_1, K31_VOCAB);
    out_gen_ids[gen_n++] = cur_tok;

    if (draft_k == 0) {
        while (gen_n < max_new_tokens && cur_tok != eos_id) {
            seq[cur_len++] = cur_tok;
            kimi_k31_hg4_forward_layers_ex(seq, cur_len, layer_ids, n_sel_layers, h_buf, NULL, NULL);
            k31_matmul(&g_k31.lm_mat, h_buf + (size_t)(cur_len - 1) * K31_D, lg_1, 1, 0, 1);
            cur_tok = k31_argmax_f32(lg_1, K31_VOCAB);
            out_gen_ids[gen_n++] = cur_tok;
        }
    } else {
        int   *ds_ids  = (int *)malloc((size_t)max_tot * sizeof(int));
        int   *ds_out  = (int *)malloc((size_t)max_tot * sizeof(int));
        float *ds_conf = (float *)malloc((size_t)max_tot * sizeof(float));
        float *ds_h    = (float *)malloc((size_t)max_tot * K31_D * sizeof(float));

        while (gen_n < max_new_tokens && cur_tok != eos_id) {
            int rem = max_new_tokens - gen_n;
            int K = (draft_k < rem) ? draft_k : rem;

            memcpy(ds_ids, seq, (size_t)cur_len * sizeof(int));
            ds_ids[cur_len] = cur_tok;
            for (int j = 1; j < K; j++) ds_ids[cur_len + j] = -1;
            int ds_T = cur_len + K;
            kimi_k31_dspark_forward(ds_ids, f_buf, ds_T, ds_h, ds_out, ds_conf);

            int drafts[6];
            for (int j = 0; j < K; j++) drafts[j] = ds_out[cur_len + j];
            draft_tot += K;

            seq[cur_len] = cur_tok;
            for (int j = 0; j < K; j++) seq[cur_len + 1 + j] = drafts[j];
            int ver_T = cur_len + 1 + K;
            kimi_k31_hg4_forward_layers_ex(seq, ver_T, layer_ids, n_sel_layers, h_buf, NULL, f_buf);
            k31_matmul(&g_k31.lm_mat, h_buf + (size_t)cur_len * K31_D, lg_1, K + 1, 0, 1);

            int matched_all = 1;
            for (int j = 0; j < K; j++) {
                int gt = k31_argmax_f32(lg_1 + (size_t)j * K31_VOCAB, K31_VOCAB);
                if (drafts[j] == gt) {
                    acc_tot++;
                    out_gen_ids[gen_n++] = gt;
                    if (gt == eos_id || gen_n >= max_new_tokens) {
                        cur_len += (j + 1);
                        cur_tok = gt;
                        matched_all = 0;
                        break;
                    }
                } else {
                    cur_len += (j + 1);
                    cur_tok = gt;
                    out_gen_ids[gen_n++] = cur_tok;
                    matched_all = 0;
                    break;
                }
            }
            if (matched_all) {
                cur_len += (K + 1);
                cur_tok = k31_argmax_f32(lg_1 + (size_t)K * K31_VOCAB, K31_VOCAB);
                if (gen_n < max_new_tokens) {
                    out_gen_ids[gen_n++] = cur_tok;
                }
            }
        }
        free(ds_ids); free(ds_out); free(ds_conf); free(ds_h);
    }

    *out_decode_ms = (omp_get_wtime() - td0) * 1000.0;
    if (out_accepted_drafts) *out_accepted_drafts = acc_tot;
    if (out_total_drafts) *out_total_drafts = draft_tot;
    free(seq); free(h_buf); free(f_buf); free(lg_1);
    return gen_n;
}

/* Production Stateful Incremental + 5L-DSpark Speculative Generator (`O(1)` KDA + MLA KV + Zero-IO Rollback) */
K31_API int kimi_k31_generate_speculative(
    const int *prompt_ids, int prompt_len, const int *layer_ids, int n_sel_layers,
    int max_new_tokens, int eos_id, int draft_k,
    int *out_gen_ids, double *out_prefill_ms, double *out_decode_ms,
    int *out_accepted_drafts, int *out_total_drafts
) {
    if (!g_k31.ready || prompt_len <= 0 || n_sel_layers <= 0 || max_new_tokens <= 0) return -1;
    if (n_sel_layers > K31_MAX_LAYERS) n_sel_layers = K31_MAX_LAYERS;
    if (draft_k < 0) draft_k = 0;
    if (draft_k > 5) draft_k = 5;
    if (!g_k31.dspark.loaded) draft_k = 0;

    float conf_thresh = 0.0f;
    const char *ct_env = getenv("CLOVER_KIMI_CONF_THRESH");
    if (ct_env && ct_env[0]) conf_thresh = (float)atof(ct_env);

    int max_tot = prompt_len + max_new_tokens + draft_k + 8;
    k31_cache_ensure_cap(max_tot, layer_ids, n_sel_layers);

    /* Check if new prompt strictly extends `g_k31_cache.cached_ids[0..cur_len-1]` (at least 1 token fed) */
    int lcp = 0;
    if (g_k31_cache.cur_len > 0 && g_k31_cache.cur_len < prompt_len) {
        if (memcmp(g_k31_cache.cached_ids, prompt_ids, (size_t)g_k31_cache.cur_len * sizeof(int)) == 0) {
            lcp = g_k31_cache.cur_len;
        } else {
            kimi_k31_cache_reset();
        }
    } else if (g_k31_cache.cur_len > 0) {
        kimi_k31_cache_reset();
    }
    g_k31_cache.last_lcp = lcp;

    int feed_n = prompt_len - lcp;
    float *h_pf  = (float *)malloc((size_t)feed_n * K31_D * sizeof(float));
    float *lg_st = (float *)malloc((size_t)(draft_k + 2) * K31_VOCAB * sizeof(float));

    double t0 = omp_get_wtime();
    k31_forward_cached_step(
        prompt_ids + lcp, lcp, feed_n, layer_ids, n_sel_layers, 0,
        h_pf, g_k31.dspark.loaded ? (g_k31_cache.feat_cache + (size_t)lcp * 5 * K31_D) : NULL
    );
    memcpy(g_k31_cache.cached_ids + lcp, prompt_ids + lcp, (size_t)feed_n * sizeof(int));
    g_k31_cache.cur_len = prompt_len;
    k31_matmul(&g_k31.lm_mat, h_pf + (size_t)(feed_n - 1) * K31_D, lg_st, 1, 0, 1);
    *out_prefill_ms = (omp_get_wtime() - t0) * 1000.0;
    free(h_pf);

    double td0 = omp_get_wtime();
    int gen_n = 0, acc_tot = 0, draft_tot = 0;
    int cur_tok = k31_argmax_f32(lg_st, K31_VOCAB);
    out_gen_ids[gen_n++] = cur_tok;

    float *h_step = (float *)malloc((size_t)8 * K31_D * sizeof(float));

    if (draft_k == 0) {
        while (gen_n < max_new_tokens && cur_tok != eos_id) {
            int pos = g_k31_cache.cur_len;
            k31_forward_cached_step(
                &cur_tok, pos, 1, layer_ids, n_sel_layers, 0, h_step,
                g_k31.dspark.loaded ? (g_k31_cache.feat_cache + (size_t)pos * 5 * K31_D) : NULL
            );
            g_k31_cache.cached_ids[pos] = cur_tok;
            g_k31_cache.cur_len = pos + 1;
            k31_matmul(&g_k31.lm_mat, h_step, lg_st, 1, 0, 1);
            cur_tok = k31_argmax_f32(lg_st, K31_VOCAB);
            out_gen_ids[gen_n++] = cur_tok;
        }
    } else {
        while (gen_n < max_new_tokens && cur_tok != eos_id) {
            int rem = max_new_tokens - gen_n;
            int K_max = (draft_k < rem) ? draft_k : rem;
            int cur_len = g_k31_cache.cur_len;

            /* 1. Commit any un-cached accepted prefix (`ds_len .. cur_len - 1`) into DSpark's context KV cache */
            if (g_k31_cache.ds_len < cur_len) {
                int dn = cur_len - g_k31_cache.ds_len;
                k31_dspark_commit_ctx_kv(
                    g_k31_cache.feat_cache + (size_t)g_k31_cache.ds_len * 5 * K31_D,
                    g_k31_cache.ds_len, dn
                );
            }

            /* 2. Draft up to `K_max` tokens with 5L-DSpark Query Block (`[cur_tok, mask, ..., mask]`) */
            int drafts[8];
            float confs[8];
            k31_dspark_draft_query_block(cur_tok, cur_len, K_max, drafts, confs);

            /* Optional adaptive verification truncation using DSpark's trained confidence head */
            int K = 1;
            while (K < K_max && confs[K - 1] >= conf_thresh && drafts[K - 1] != eos_id) {
                K++;
            }
            draft_tot += K;

            /* 3. Batched C1 Verification on ONLY the `1 + K` new tokens (`[cur_tok, drafts[0..K-1]]`)! */
            int ver_ids[8];
            ver_ids[0] = cur_tok;
            for (int j = 0; j < K; j++) ver_ids[1 + j] = drafts[j];
            int ver_n = 1 + K;

            k31_forward_cached_step(
                ver_ids, cur_len, ver_n, layer_ids, n_sel_layers, 1,
                h_step, g_k31_cache.feat_cache + (size_t)cur_len * 5 * K31_D
            );
            k31_matmul(&g_k31.lm_mat, h_step, lg_st, ver_n, 0, 1);

            int keep_n = ver_n;
            int matched_all = 1;
            for (int j = 0; j < K; j++) {
                int gt = k31_argmax_f32(lg_st + (size_t)j * K31_VOCAB, K31_VOCAB);
                if (drafts[j] == gt) {
                    acc_tot++;
                    out_gen_ids[gen_n++] = gt;
                    if (gt == eos_id || gen_n >= max_new_tokens) {
                        keep_n = j + 1;
                        cur_tok = gt;
                        matched_all = 0;
                        break;
                    }
                } else {
                    keep_n = j + 1;
                    cur_tok = gt;
                    out_gen_ids[gen_n++] = cur_tok;
                    matched_all = 0;
                    break;
                }
            }
            if (matched_all) {
                keep_n = K + 1;
                cur_tok = k31_argmax_f32(lg_st + (size_t)K * K31_VOCAB, K31_VOCAB);
                if (gen_n < max_new_tokens) {
                    out_gen_ids[gen_n++] = cur_tok;
                }
            }

            /* 4. If `keep_n < ver_n`, replay ONLY the `keep_n` causal KDA updates in ~1.5ms (zero NVMe I/O!) */
            if (keep_n < ver_n) {
                for (int i = 0; i < n_sel_layers; i++) {
                    int lid = layer_ids[i];
                    if (lid >= 0 && lid < g_k31.n_layers && g_k31.layers[lid].kind != 2) {
                        k31_rollback_kda_keep(&g_k31_cache.kda[lid], cur_len, keep_n);
                    }
                }
            }
            memcpy(g_k31_cache.cached_ids + cur_len, ver_ids, (size_t)keep_n * sizeof(int));
            g_k31_cache.cur_len = cur_len + keep_n;
        }
    }

    *out_decode_ms = (omp_get_wtime() - td0) * 1000.0;
    if (out_accepted_drafts) *out_accepted_drafts = acc_tot;
    if (out_total_drafts) *out_total_drafts = draft_tot;
    free(h_step);
    free(lg_st);
    return gen_n;
}

/* ============================================================================================================
 * Multi-Node Distributed Pipeline-Parallel C ABI (`Node_s` owns `[l_lo, l_hi)` in VRAM; <1 MB TCP state hops)
 * ============================================================================================================ */

K31_API void kimi_k31_pipeline_embed(const int *token_ids, int T, float *out_prefix_sum) {
    #pragma omp parallel for schedule(static) if (T >= 4)
    for (int t = 0; t < T; t++) {
        k31_embed_lookup_row(&g_k31.emb_mat, token_ids[t], out_prefix_sum + (size_t)t * K31_D);
    }
}

K31_API void kimi_k31_pipeline_forward_range(
    int l_lo, int l_hi, int base_pos, int T, int save_for_rollback,
    float *io_prefix_sum, float *io_delta, int *io_has_delta,
    float *io_blocks, float *io_dspark_feats_35840, uint8_t *io_feat_mask_5
) {
    if (!g_k31.ready || T <= 0 || l_lo >= l_hi) return;
    if (l_lo < 0) l_lo = 0;
    if (l_hi > g_k31.n_layers) l_hi = g_k31.n_layers;
    if (l_lo >= l_hi) return;

    int range_lids[K31_MAX_LAYERS] = {0};
    int n_sel = 0;
    for (int l = l_lo; l < l_hi; l++) range_lids[n_sel++] = l;
    k31_cache_ensure_cap(base_pos + T + 16, range_lids, n_sel);

    float *nxt_delta = (float *)malloc((size_t)T * K31_D * sizeof(float));
    float *mixed     = (float *)malloc((size_t)T * K31_D * sizeof(float));
    float *u         = (float *)malloc((size_t)T * K31_D * sizeof(float));
    float *attn_proj = (float *)malloc((size_t)T * K31_D * sizeof(float));
    float *post      = (float *)malloc((size_t)T * K31_D * sizeof(float));

    const int tgt_lids[5] = {7, 23, 67, 83, 91};
    int has_delta = *io_has_delta;

    for (int lid = l_lo; lid < l_hi; lid++) {
        k31_layer_t *lyr = &g_k31.layers[lid];
        const int is_write    = (lid % K31_ATTN_RES_BLK == 0) ? 1 : 0;
        const int write_idx   = lid / K31_ATTN_RES_BLK;
        const int prev_blocks = (lid + K31_ATTN_RES_BLK - 1) / K31_ATTN_RES_BLK;
        const int mlp_blocks  = prev_blocks + is_write;

        if (has_delta) {
            for (int idx = 0; idx < T * K31_D; idx++) io_prefix_sum[idx] += io_delta[idx];
        }
        if (is_write && write_idx < K31_MAX_BLOCKS) {
            for (int t = 0; t < T; t++) {
                memcpy(io_blocks + ((size_t)t * K31_MAX_BLOCKS + (size_t)write_idx) * K31_D,
                       io_prefix_sum + (size_t)t * K31_D, K31_D * sizeof(float));
            }
        }
        k31_attn_res_mix(io_blocks, prev_blocks, io_prefix_sum, lyr->sa_rn, lyr->sa_rp, mixed, T);
        rms_norm_batch(mixed, lyr->in_ln, u, T, K31_D, 1e-5f);

        if (lyr->kind == 2) k31_run_dsa_cached(&lyr->dsa, &g_k31_cache.dsa[lid], u, attn_proj, base_pos, T);
        else                k31_run_kda_cached(&lyr->kda, &g_k31_cache.kda[lid], u, attn_proj, base_pos, T, save_for_rollback);

        if (is_write) {
            memcpy(io_prefix_sum, attn_proj, (size_t)T * K31_D * sizeof(float));
        } else {
            for (int idx = 0; idx < T * K31_D; idx++) io_prefix_sum[idx] += attn_proj[idx];
        }
        k31_attn_res_mix(io_blocks, mlp_blocks, io_prefix_sum, lyr->mlp_rn, lyr->mlp_rp, mixed, T);
        rms_norm_batch(mixed, lyr->post_ln, post, T, K31_D, 1e-5f);

        if (lyr->kind == 0) {
            float *dhid = (float *)malloc((size_t)T * K31_DENSE_INT * sizeof(float));
            k31_matmul(&lyr->dense_w13, post, dhid, T, 2, 1);
            k31_matmul(&lyr->dense_dn,  dhid, nxt_delta, T, 0, 1);
            free(dhid);
        } else {
            k31_run_moe(&lyr->moe, &lyr->ex, post, nxt_delta, T);
        }

        memcpy(io_delta, nxt_delta, (size_t)T * K31_D * sizeof(float));
        has_delta = 1;

        if (io_dspark_feats_35840 && io_feat_mask_5) {
            for (int m = 0; m < 5; m++) {
                if (lid == tgt_lids[m]) {
                    io_feat_mask_5[m] = 1;
                    for (int t = 0; t < T; t++) {
                        float *dst = io_dspark_feats_35840 + ((size_t)t * 5 + (size_t)m) * K31_D;
                        const float *ps = io_prefix_sum + (size_t)t * K31_D;
                        const float *dl = io_delta      + (size_t)t * K31_D;
                        for (int d = 0; d < K31_D; d++) dst[d] = ps[d] + dl[d];
                    }
                }
            }
        }
    }

    *io_has_delta = has_delta;
    free(nxt_delta); free(mixed); free(u); free(attn_proj); free(post);
}

K31_API void kimi_k31_pipeline_tail_logits(
    int max_lid, int T, int logits_last_only,
    float *io_prefix_sum, const float *delta, int has_delta,
    const float *blocks, float *io_dspark_feats_35840, const uint8_t *feat_mask_5,
    int *out_top1_tokens
) {
    if (has_delta) {
        for (int idx = 0; idx < T * K31_D; idx++) io_prefix_sum[idx] += delta[idx];
    }
    if (io_dspark_feats_35840 && feat_mask_5) {
        for (int m = 0; m < 5; m++) {
            if (!feat_mask_5[m]) {
                for (int t = 0; t < T; t++) {
                    memcpy(io_dspark_feats_35840 + ((size_t)t * 5 + (size_t)m) * K31_D,
                           io_prefix_sum + (size_t)t * K31_D, K31_D * sizeof(float));
                }
            }
        }
    }

    int final_blocks = (max_lid + 1 + K31_ATTN_RES_BLK - 1) / K31_ATTN_RES_BLK;
    if (final_blocks > K31_MAX_BLOCKS) final_blocks = K31_MAX_BLOCKS;

    int eval_T = logits_last_only ? 1 : T;
    int t_off  = logits_last_only ? (T - 1) : 0;
    float *agg_final = (float *)malloc((size_t)eval_T * K31_D * sizeof(float));
    float *fin_norm  = (float *)malloc((size_t)eval_T * K31_D * sizeof(float));
    float *lg        = (float *)malloc((size_t)eval_T * K31_VOCAB * sizeof(float));

    k31_attn_res_mix(
        blocks + (size_t)t_off * K31_MAX_BLOCKS * K31_D,
        final_blocks,
        io_prefix_sum + (size_t)t_off * K31_D,
        g_k31.out_res_norm, g_k31.out_res_proj, agg_final, eval_T
    );
    rms_norm_batch(agg_final, g_k31.norm_w, fin_norm, eval_T, K31_D, 1e-5f);
    k31_matmul(&g_k31.lm_mat, fin_norm, lg, eval_T, 0, 1);

    for (int i = 0; i < eval_T; i++) {
        out_top1_tokens[i] = k31_argmax_f32(lg + (size_t)i * K31_VOCAB, K31_VOCAB);
    }

    free(agg_final); free(fin_norm); free(lg);
}

K31_API void kimi_k31_pipeline_rollback_range(int l_lo, int l_hi, int base_pos, int keep_n) {
    if (l_lo < 0) l_lo = 0;
    if (l_hi > g_k31.n_layers) l_hi = g_k31.n_layers;
    for (int lid = l_lo; lid < l_hi; lid++) {
        if (g_k31.layers[lid].kind != 2 && g_k31_cache.kda[lid].S) {
            k31_rollback_kda_keep(&g_k31_cache.kda[lid], base_pos, keep_n);
        }
    }
}

K31_API int kimi_k31_pipeline_dspark_draft(
    int cur_tok, int cur_len, int K_max, float conf_thresh, int eos_id,
    const float *new_feats_35840, int n_new_feats, int *out_drafts
) {
    if (!g_k31.dspark.loaded || K_max <= 0) return 0;
    if (K_max > 5) K_max = 5;
    int need_cap = cur_len + K_max + 16;
    if (need_cap > g_k31_cache.alloc_cap) {
        int new_cap = g_k31_cache.alloc_cap ? g_k31_cache.alloc_cap : 1536;
        while (new_cap < need_cap) new_cap *= 2;
        for (int dl = 0; dl < K31_DS_LAYERS; dl++) {
            g_k31_cache.ds_k[dl] = (float *)realloc(g_k31_cache.ds_k[dl], (size_t)new_cap * 1024 * sizeof(float));
            g_k31_cache.ds_v[dl] = (float *)realloc(g_k31_cache.ds_v[dl], (size_t)new_cap * 1024 * sizeof(float));
        }
        g_k31_cache.alloc_cap = new_cap;
    }
    if (n_new_feats > 0 && new_feats_35840) {
        k31_dspark_commit_ctx_kv(new_feats_35840, cur_len - n_new_feats, n_new_feats);
    }
    float confs[8];
    k31_dspark_draft_query_block(cur_tok, cur_len, K_max, out_drafts, confs);
    int K = 1;
    while (K < K_max && confs[K - 1] >= conf_thresh && out_drafts[K - 1] != eos_id) {
        K++;
    }
    return K;
}

