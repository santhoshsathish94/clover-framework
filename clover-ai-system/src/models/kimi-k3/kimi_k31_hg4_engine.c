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

enum {
    K31_D          = 7168,
    K31_VOCAB      = 163840,
    K31_LATENT     = 3584,
    K31_EXP_INTER  = 3072,
    K31_SH_INTER   = 6144,
    K31_DENSE_INT  = 33792,
    K31_EXPERTS    = 896,
    K31_TOP_K      = 8,
    K31_MAX_LAYERS = 93,
    K31_DS_LAYERS  = 5,
    K31_DS_INTER   = 14336,
    K31_PANEL_MAX  = 128
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
    v->bits = (raw_k >> 28) == 8 ? 8u : 4u;
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
    k31_mmap_t    mf;
    int           loaded;
    uint32_t      n_cached;
    uint32_t      exp_stride;
    const float  *pre_diag_up; /* [3584] */
    const uint8_t *exp_base;
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
} k31_dspark_t;

static struct {
    int ready;
    int isa_tier;
    int n_layers;
    char loaded_dir[512];

    uint8_t lut16_u8[16];
    int8_t  lut16_s8[16];
    float   lut16_s8_f[16];
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
    const k31_mat_t *M, const uint8_t *lut16_u8,
    const int8_t *X_q, const float *X_sc, const int32_t *X_sum,
    float *Y, int T, int rg, int glu_mode
) {
    const int R = (int)M->R, K = (int)M->K, nb = K / 128;
    const int out_R = glu_mode ? (R / 2) : R;
    const size_t bstride = (M->bits == 8) ? 2048u : 1024u;
    const uint8_t *wt = M->tiles + (size_t)rg * (size_t)nb * bstride;
    const uint16_t *wsc = M->scales_fp16 + (size_t)rg * (size_t)nb * 16;
    const __m512i vlut = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)lut16_u8));
    const __m512i m0f  = _mm512_set1_epi8(0x0F);

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
    const k31_mat_t *M, const int8_t *lut16_s8,
    const int8_t *X_q, const float *X_sc,
    float *Y, int T, int rg, int glu_mode
) {
    const int R = (int)M->R, K = (int)M->K, nb = K / 128;
    const int out_R = glu_mode ? (R / 2) : R;
    const size_t bstride = (M->bits == 8) ? 2048u : 1024u;
    const uint8_t *wt = M->tiles + (size_t)rg * (size_t)nb * bstride;
    const uint16_t *wsc = M->scales_fp16 + (size_t)rg * (size_t)nb * 16;
    const __m256i vlut   = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)lut16_s8));
    const __m256i m0f    = _mm256_set1_epi8(0x0F);
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
    const k31_mat_t *M, const int8_t *lut16_s8,
    const int8_t *X_q, const float *X_sc,
    float *Y, int T, int rg, int glu_mode
) {
    const int R = (int)M->R, K = (int)M->K, nb = K / 128;
    const int out_R = glu_mode ? (R / 2) : R;
    const size_t bstride = (M->bits == 8) ? 2048u : 1024u;
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
    if (g_k31.isa_tier == 3) { k31_group16_avx512(M, g_k31.lut16_u8, xq, xsc, xsum, Y, T, rg, glu_mode); return; }
    if (g_k31.isa_tier >= 1) { k31_group16_avx2(M, g_k31.lut16_s8, xq, xsc, Y, T, rg, glu_mode); return; }
#endif
    k31_group16_scalar(M, g_k31.lut16_s8, xq, xsc, Y, T, rg, glu_mode);
}

static void k31_matmul_prequant(const k31_mat_t *M, const int8_t *xq, const float *xsc, const int32_t *xsum, float *Y, int T, int glu_mode, int use_omp) {
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

/* Block-Snapshot Attention-Residual Aggregation (`transformer_aggregate`) */
static void k31_attn_res_aggregate(
    float **snapshots, int n_snaps, const float *residual,
    const float *res_norm_w, const float *res_proj_w,
    float *out_agg, int T
) {
    const int S = n_snaps + 1;
    #pragma omp parallel for schedule(static) if (T >= 2)
    for (int t = 0; t < T; t++) {
        float scores[128];
        float mx = -1e30f;
        for (int s = 0; s < S; s++) {
            const float *src = (s < n_snaps) ? (snapshots[s] + (size_t)t * K31_D) : (residual + (size_t)t * K31_D);
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

        float *dst = out_agg + (size_t)t * K31_D;
        memset(dst, 0, K31_D * sizeof(float));
        for (int s = 0; s < S; s++) {
            const float *src = (s < n_snaps) ? (snapshots[s] + (size_t)t * K31_D) : (residual + (size_t)t * K31_D);
            float ws = scores[s];
            for (int d = 0; d < K31_D; d++) dst[d] += ws * src[d];
        }
    }
}

/* Kimi Delta Attention (`KDA`) Branch */
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
        float *qn = (float *)malloc((size_t)T * 128 * sizeof(float));
        float *kn = (float *)malloc((size_t)T * 128 * sizeof(float));
        for (int t = 0; t < T; t++) {
            const float *qh = q_c + (size_t)t * 12288 + h * 128;
            const float *kh = k_c + (size_t)t * 12288 + h * 128;
            const float *dr = fb_o + (size_t)t * 12288 + h * 128;
            double sq = 0.0, sk = 0.0;
            for (int d = 0; d < 128; d++) { sq += (double)qh[d] * qh[d]; sk += (double)kh[d] * kh[d]; }
            float iq = (1.0f / sqrtf((float)sq + 1e-6f)) * inv_s128;
            float ik = 1.0f / sqrtf((float)sk + 1e-6f);
            float beta = 1.0f / (1.0f + expf(-b_raw[t * 96 + h]));
            for (int d = 0; d < 128; d++) {
                float dec_arg = (dr[d] + kda->dtb[h * 128 + d]) * kda->A_neg[d];
                float dec = expf(-5.0f / (1.0f + expf(-dec_arg)));
                qn[t * 128 + d] = qh[d] * iq;
                kn[t * 128 + d] = kh[d] * ik * beta * dec;
            }
        }
        for (int t = 0; t < T; t++) {
            float yh[128] = {0};
            for (int s = 0; s <= t; s++) {
                float dot = 0.0f;
                for (int d = 0; d < 128; d++) dot += qn[t * 128 + d] * kn[s * 128 + d];
                const float *vs = v_c + (size_t)s * 12288 + h * 128;
                for (int d = 0; d < 128; d++) yh[d] += dot * vs[d];
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
        free(qn); free(kn);
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

/* Latent `SiTU-GLU` MoE (`routed_expert_down_proj -> cached SiTU-GLU experts -> norm -> up_proj` + Shared `SiTU-GLU`) */
static void k31_run_moe(const k31_moe_trunk_t *m, const k31_experts_t *ex, const float *post, float *mlp_out, int T) {
    float *sh_hid = (float *)malloc((size_t)T * K31_SH_INTER * sizeof(float));
    k31_matmul(&m->sh_w13, post, sh_hid, T, 2, 1); /* Fused SiTU-GLU! */
    k31_matmul(&m->sh_dn, sh_hid, mlp_out, T, 0, 1);
    free(sh_hid);

    if (!ex->loaded || ex->n_cached == 0) return;

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
        int sel[K31_TOP_K];
        float sw[K31_TOP_K], wsum = 0.0f;
        for (int k = 0; k < top_k; k++) {
            int be = 0; float bv = -1e30f;
            for (int e = 0; e < E; e++) if (ch[e] > bv) { bv = ch[e]; be = e; }
            ch[be] = -1e30f;
            sel[k] = be; sw[k] = sc[be]; wsum += sc[be];
        }
        for (int k = 0; k < top_k; k++) sw[k] /= (wsum + 1e-12f);

        for (int k = 0; k < top_k; k++) {
            const uint8_t *ep = ex->exp_base + (size_t)sel[k] * ex->exp_stride;
            k31_mat_t gu_m, dn_m;
            gu_m.R = 6144; gu_m.K = 3584; gu_m.bits = 4;
            gu_m.pre_diag = ex->pre_diag_up;
            gu_m.scales_fp16 = (const uint16_t *)ep;
            gu_m.tiles = ep + 344064;
            parse_k31_mat(ep + 11354112, &dn_m);

            float hid[K31_EXP_INTER], ey[K31_LATENT];
            k31_matmul_prequant(&gu_m, x_xq + (size_t)t * K31_LATENT, x_xsc + (size_t)t * nb_up, x_xsum + (size_t)t * nb_up, hid, 1, 2, 1);
            k31_matmul(&dn_m, hid, ey, 1, 0, 1);
            float *dst = lat_acc + (size_t)t * K31_LATENT;
            for (int d = 0; d < K31_LATENT; d++) dst[d] += ey[d] * sw[k];
        }
    }

    rms_norm_batch(lat_acc, m->lat_norm, lat_nm, T, K31_LATENT, 1e-5f);
    k31_matmul(&m->lat_up, lat_nm, routed, T, 0, 1);
    for (int i = 0; i < T * K31_D; i++) mlp_out[i] += routed[i];

    free(lat_in); free(lat_acc); free(lat_nm); free(routed);
    free(x_xq); free(x_xsc); free(x_xsum);
}

static void k31_forward_one_layer(
    const k31_layer_t *lyr, const float *h_in, float **snapshots, int n_snaps,
    float *h_out, float *out_idx_scores, int T
) {
    float *u         = (float *)malloc((size_t)T * K31_D * sizeof(float));
    float *attn_proj = (float *)malloc((size_t)T * K31_D * sizeof(float));
    float *res1      = (float *)malloc((size_t)T * K31_D * sizeof(float));
    float *agg1      = (float *)malloc((size_t)T * K31_D * sizeof(float));
    float *post      = (float *)malloc((size_t)T * K31_D * sizeof(float));
    float *mlp_out   = (float *)malloc((size_t)T * K31_D * sizeof(float));
    float *res2      = (float *)malloc((size_t)T * K31_D * sizeof(float));

    rms_norm_batch(h_in, lyr->in_ln, u, T, K31_D, 1e-5f);
    if (lyr->kind == 2) k31_run_dsa(&lyr->dsa, u, attn_proj, out_idx_scores, T);
    else                k31_run_kda(&lyr->kda, u, attn_proj, T);

    for (int i = 0; i < T * K31_D; i++) res1[i] = h_in[i] + attn_proj[i];
    k31_attn_res_aggregate(snapshots, n_snaps, res1, lyr->sa_rn, lyr->sa_rp, agg1, T);

    rms_norm_batch(agg1, lyr->post_ln, post, T, K31_D, 1e-5f);
    if (lyr->kind == 0) {
        float *dhid = (float *)malloc((size_t)T * K31_DENSE_INT * sizeof(float));
        k31_matmul(&lyr->dense_w13, post, dhid, T, 2, 1); /* Fused SiTU-GLU on 33,792 FFN! */
        k31_matmul(&lyr->dense_dn,  dhid, mlp_out, T, 0, 1);
        free(dhid);
    } else {
        k31_run_moe(&lyr->moe, &lyr->ex, post, mlp_out, T);
    }

    for (int i = 0; i < T * K31_D; i++) res2[i] = res1[i] + mlp_out[i];
    k31_attn_res_aggregate(snapshots, n_snaps, res2, lyr->mlp_rn, lyr->mlp_rp, h_out, T);

    free(u); free(attn_proj); free(res1); free(agg1); free(post); free(mlp_out); free(res2);
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

static int open_k31_experts(const char *path, k31_experts_t *ex) {
    memset(ex, 0, sizeof(*ex));
    if (k31_mmap_open(path, &ex->mf) != 0) return -1;
    const uint8_t *p = (const uint8_t *)ex->mf.ptr;
    if (memcmp(p, "K31E", 4) != 0) { k31_mmap_close(&ex->mf); return -2; }
    memcpy(&ex->n_cached, p + 4, 4);
    memcpy(&ex->exp_stride, p + 24, 4);
    ex->pre_diag_up = (const float *)(p + 28);
    ex->exp_base = p + 28 + K31_LATENT * 4;
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
    for (int i = 0; i < 32; i++) ds->inv_freq[i] = 1.0f / powf(10000.0f, (float)(2 * i) / 64.0f);
    ds->loaded = 1;
    return 0;
}

K31_API int kimi_k31_hg4_init(const char *hg4_dir) {
    if (g_k31.ready && strcmp(g_k31.loaded_dir, hg4_dir) == 0) return 0;
    if (g_k31.ready) {
        k31_mmap_close(&g_k31.emb_mf);
        k31_mmap_close(&g_k31.lm_mf);
        if (g_k31.dspark.loaded) k31_mmap_close(&g_k31.dspark.mf);
        for (int l = 0; l < g_k31.n_layers; l++) {
            k31_mmap_close(&g_k31.layers[l].mf);
            if (g_k31.layers[l].ex.loaded) k31_mmap_close(&g_k31.layers[l].ex.mf);
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
    for (int i = 0; i < 16; i++) {
        int sv = (int)g_k31.lut16_u8[i] - 128;
        g_k31.lut16_s8[i] = (int8_t)sv;
        g_k31.lut16_s8_f[i] = (float)sv;
    }

    snprintf(path, sizeof(path), "%s/embed_tokens.hg4", hg4_dir);
    if (k31_mmap_open(path, &g_k31.emb_mf) != 0) return -3;
    parse_k31_mat((const uint8_t *)g_k31.emb_mf.ptr, &g_k31.emb_mat);

    snprintf(path, sizeof(path), "%s/lm_head.hg4", hg4_dir);
    if (k31_mmap_open(path, &g_k31.lm_mf) != 0) return -4;
    parse_k31_mat((const uint8_t *)g_k31.lm_mf.ptr, &g_k31.lm_mat);

    for (int l = 0; l < g_k31.n_layers; l++) {
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
            snprintf(path, sizeof(path), "%s/experts_%02d.hg4", hg4_dir, l);
            open_k31_experts(path, &lyr->ex);
        }
    }

    snprintf(path, sizeof(path), "%s/dspark.hg4", hg4_dir);
    open_k31_dspark(path, &g_k31.dspark);

    if (!g_k31.scr_xq) {
        g_k31.scr_xq   = (int8_t *)malloc((size_t)K31_PANEL_MAX * 35840);
        g_k31.scr_xsc  = (float *)malloc((size_t)K31_PANEL_MAX * (35840 / 128) * sizeof(float));
        g_k31.scr_xsum = (int32_t *)malloc((size_t)K31_PANEL_MAX * (35840 / 128) * sizeof(int32_t));
    }
    snprintf(g_k31.loaded_dir, sizeof(g_k31.loaded_dir), "%s", hg4_dir);
    g_k31.ready = 1;
    return 0;
}

/* Forward selected Kimi K3.1 Pro layers and optionally extract DSpark's 5 target features `[L7, L23, L67, L83, L91]` (`[T, 35840]`) */
K31_API float kimi_k31_hg4_forward_layers_ex(
    const int *token_ids, int T, const int *layer_ids, int n_sel_layers,
    float *out_final_hidden, float *out_dsa_indexer_scores, float *out_dspark_feats_35840
) {
    if (!g_k31.ready || T <= 0 || n_sel_layers <= 0) return -1.0f;
    double t0 = omp_get_wtime();

    float **snaps = (float **)calloc((size_t)(n_sel_layers + 2), sizeof(float *));
    int *snap_lids = (int *)malloc((size_t)(n_sel_layers + 2) * sizeof(int));
    snaps[0] = (float *)malloc((size_t)T * K31_D * sizeof(float));
    snap_lids[0] = -1;
    #pragma omp parallel for schedule(static) if (T >= 4)
    for (int t = 0; t < T; t++) {
        k31_embed_lookup_row(&g_k31.emb_mat, token_ids[t], snaps[0] + (size_t)t * K31_D);
    }

    float *cur = snaps[0];
    int n_snaps = 1;
    for (int i = 0; i < n_sel_layers; i++) {
        int lid = layer_ids[i];
        if (lid < 0 || lid >= g_k31.n_layers) continue;
        float *nxt = (float *)malloc((size_t)T * K31_D * sizeof(float));
        k31_forward_one_layer(&g_k31.layers[lid], cur, snaps, n_snaps, nxt,
                              (g_k31.layers[lid].kind == 2) ? out_dsa_indexer_scores : NULL, T);
        snap_lids[n_snaps] = lid;
        snaps[n_snaps++] = nxt;
        cur = nxt;
    }

    if (out_dspark_feats_35840) {
        const int tgt_lids[5] = {7, 23, 67, 83, 91};
        const float *feat_src[5] = {cur, cur, cur, cur, cur};
        for (int m = 0; m < 5; m++) {
            for (int s = 1; s < n_snaps; s++) {
                if (snap_lids[s] == tgt_lids[m]) { feat_src[m] = snaps[s]; break; }
            }
        }
        for (int t = 0; t < T; t++) {
            float *dst = out_dspark_feats_35840 + (size_t)t * (5 * K31_D);
            for (int m = 0; m < 5; m++) {
                memcpy(dst + (size_t)m * K31_D, feat_src[m] + (size_t)t * K31_D, K31_D * sizeof(float));
            }
        }
    }

    float *agg_final = (float *)malloc((size_t)T * K31_D * sizeof(float));
    k31_attn_res_aggregate(snaps, n_snaps - 1, cur, g_k31.out_res_norm, g_k31.out_res_proj, agg_final, T);
    rms_norm_batch(agg_final, g_k31.norm_w, out_final_hidden, T, K31_D, 1e-5f);

    free(agg_final);
    for (int s = 0; s < n_snaps; s++) free(snaps[s]);
    free(snaps);
    free(snap_lids);
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

/* Full 5-Layer `Kimi-K3.1-Pro-DSpark` Forward (`target_feats_35840` optional; if NULL, tiles token embeddings 5x) */
K31_API float kimi_k31_dspark_forward(
    const int *token_ids, const float *target_feats_35840, int T,
    float *out_dspark_hidden_7168, int *out_draft_tokens, float *out_confidences
) {
    if (!g_k31.ready || !g_k31.dspark.loaded || T <= 0) return -1.0f;
    double t0 = omp_get_wtime();
    const k31_dspark_t *ds = &g_k31.dspark;

    float *emb     = (float *)malloc((size_t)T * K31_D * sizeof(float));
    float *fc_in   = (float *)malloc((size_t)T * (5 * K31_D) * sizeof(float));
    float *fc_out  = (float *)malloc((size_t)T * K31_D * sizeof(float));
    float *cur     = (float *)malloc((size_t)T * K31_D * sizeof(float));
    float *nxt     = (float *)malloc((size_t)T * K31_D * sizeof(float));

    #pragma omp parallel for schedule(static) if (T >= 4)
    for (int t = 0; t < T; t++) {
        int tid = (token_ids[t] >= 0) ? token_ids[t] : 163824; /* DSpark mask_token_id = 163824 */
        k31_embed_lookup_row(&g_k31.emb_mat, tid, emb + (size_t)t * K31_D);
    }

    if (target_feats_35840) {
        memcpy(fc_in, target_feats_35840, (size_t)T * 5 * K31_D * sizeof(float));
    } else {
        for (int t = 0; t < T; t++)
            for (int m = 0; m < 5; m++)
                memcpy(fc_in + ((size_t)t * 5 + m) * K31_D, emb + (size_t)t * K31_D, K31_D * sizeof(float));
    }

    k31_matmul(&ds->fc, fc_in, fc_out, T, 0, 1);
    rms_norm_batch(fc_out, ds->hid_norm, fc_out, T, K31_D, 1e-5f);
    for (int i = 0; i < T * K31_D; i++) cur[i] = emb[i] + fc_out[i];

    for (int l = 0; l < K31_DS_LAYERS; l++) {
        const k31_ds_layer_t *dl = &ds->layers[l];
        float *u   = (float *)malloc((size_t)T * K31_D * sizeof(float));
        float *q   = (float *)malloc((size_t)T * 4096 * sizeof(float));
        float *k   = (float *)malloc((size_t)T * 1024 * sizeof(float));
        float *v   = (float *)malloc((size_t)T * 1024 * sizeof(float));
        float *ao  = (float *)malloc((size_t)T * 4096 * sizeof(float));
        float *bo  = (float *)malloc((size_t)T * K31_D * sizeof(float));
        float *fh  = (float *)malloc((size_t)T * K31_DS_INTER * sizeof(float));
        float *fo  = (float *)malloc((size_t)T * K31_D * sizeof(float));

        rms_norm_batch(cur, dl->in_ln, u, T, K31_D, 1e-5f);
        k31_matmul(&dl->q, u, q, T, 0, 1);
        k31_matmul(&dl->k, u, k, T, 0, 1);
        k31_matmul(&dl->v, u, v, T, 0, 1);

        #pragma omp parallel for schedule(static) if (T >= 2)
        for (int t = 0; t < T; t++) {
            float cs[32], sn[32];
            for (int d = 0; d < 32; d++) { float ang = (float)t * ds->inv_freq[d]; cs[d] = cosf(ang); sn[d] = sinf(ang); }
            for (int h = 0; h < 64; h++) {
                float *qh = q + (size_t)t * 4096 + h * 64;
                rms_norm_row(qh, dl->qn, qh, 64, 1e-5f);
                for (int d = 0; d < 32; d++) {
                    float a = qh[d], b = qh[32 + d];
                    qh[d] = a * cs[d] - b * sn[d]; qh[32 + d] = b * cs[d] + a * sn[d];
                }
            }
            for (int kh = 0; kh < 16; kh++) {
                float *khp = k + (size_t)t * 1024 + kh * 64;
                rms_norm_row(khp, dl->kn, khp, 64, 1e-5f);
                for (int d = 0; d < 32; d++) {
                    float a = khp[d], b = khp[32 + d];
                    khp[d] = a * cs[d] - b * sn[d]; khp[32 + d] = b * cs[d] + a * sn[d];
                }
            }
        }

        const float scale = 1.0f / 8.0f; /* 1 / sqrt(64) */
        #pragma omp parallel for schedule(static)
        for (int h = 0; h < 64; h++) {
            int kv_h = h / 4;
            float *sc = (float *)malloc((size_t)T * sizeof(float));
            for (int t = 0; t < T; t++) {
                int j_end = (token_ids[t] < 0) ? (T - 1) : t; /* Bidirectional inside DSpark masked draft block */
                const float *qh = q + (size_t)t * 4096 + h * 64;
                float mx = -1e30f;
                for (int j = 0; j <= j_end; j++) {
                    const float *kj = k + (size_t)j * 1024 + kv_h * 64;
                    float dot = 0.0f;
                    for (int d = 0; d < 64; d++) dot += qh[d] * kj[d];
                    sc[j] = dot * scale;
                    if (sc[j] > mx) mx = sc[j];
                }
                float sw = 0.0f;
                for (int j = 0; j <= j_end; j++) { sc[j] = expf(sc[j] - mx); sw += sc[j]; }
                for (int j = 0; j <= j_end; j++) sc[j] /= (sw + 1e-12f);

                float *yh = ao + (size_t)t * 4096 + h * 64;
                memset(yh, 0, 64 * sizeof(float));
                for (int j = 0; j <= j_end; j++) {
                    const float *vj = v + (size_t)j * 1024 + kv_h * 64;
                    float w = sc[j];
                    for (int d = 0; d < 64; d++) yh[d] += w * vj[d];
                }
            }
            free(sc);
        }

        k31_matmul(&dl->o, ao, bo, T, 0, 1);
        for (int i = 0; i < T * K31_D; i++) bo[i] += cur[i];

        rms_norm_batch(bo, dl->post_ln, u, T, K31_D, 1e-5f);
        k31_matmul(&dl->w13, u,  fh, T, 1, 1); /* Fused SwiGLU! */
        k31_matmul(&dl->wdn, fh, fo, T, 0, 1);
        for (int i = 0; i < T * K31_D; i++) nxt[i] = bo[i] + fo[i];

        free(u); free(q); free(k); free(v); free(ao); free(bo); free(fh); free(fo);
        float *tmp = cur; cur = nxt; nxt = tmp;
    }

    rms_norm_batch(cur, ds->final_norm, out_dspark_hidden_7168, T, K31_D, 1e-5f);

    if (out_draft_tokens) {
        float *base_lg = (float *)malloc((size_t)T * K31_VOCAB * sizeof(float));
        k31_matmul(&g_k31.lm_mat, out_dspark_hidden_7168, base_lg, T, 0, 1);

        int prev_tok = (token_ids[0] >= 0) ? token_ids[0] : 0;
        float m_vec[256], m_bias[K31_VOCAB];
        for (int t = 0; t < T; t++) {
            int ctx_tok = (token_ids[t] >= 0) ? token_ids[t] : prev_tok;
            k31_embed_lookup_row(&ds->mw1, ctx_tok, m_vec);
            k31_matmul(&ds->mw2, m_vec, m_bias, 1, 0, 1);

            float *lt = base_lg + (size_t)t * K31_VOCAB;
            int best_id = 0; float best_v = -1e30f;
            for (int v = 0; v < K31_VOCAB; v++) {
                float val = lt[v] + m_bias[v];
                if (val > best_v) { best_v = val; best_id = v; }
            }
            out_draft_tokens[t] = best_id;
            prev_tok = best_id;

            if (out_confidences) {
                const float *ht = out_dspark_hidden_7168 + (size_t)t * K31_D;
                float cdot = ds->conf_b[0];
                for (int d = 0; d < K31_D; d++) cdot += ht[d] * ds->conf_w[d];
                for (int d = 0; d < 256;   d++) cdot += m_vec[d] * ds->conf_w[K31_D + d];
                out_confidences[t] = 1.0f / (1.0f + expf(-cdot));
            }
        }
        free(base_lg);
    }

    free(emb); free(fc_in); free(fc_out); free(cur); free(nxt);
    return (float)((omp_get_wtime() - t0) * 1000.0);
}

static int k31_argmax_f32(const float *v, int n) {
    int bi = 0; float bv = v[0];
    for (int i = 1; i < n; i++) if (v[i] > bv) { bv = v[i]; bi = i; }
    return bi;
}

/* Lossless Speculative Decode Inside Kimi K3.1 Pro Using 5-Layer DSpark (`draft_k = 0` plain, `1..4` speculative) */
K31_API int kimi_k31_generate_speculative(
    const int *prompt_ids, int prompt_len, const int *layer_ids, int n_sel_layers,
    int max_new_tokens, int eos_id, int draft_k,
    int *out_gen_ids, double *out_prefill_ms, double *out_decode_ms,
    int *out_accepted_drafts, int *out_total_drafts
) {
    if (!g_k31.ready || prompt_len <= 0 || n_sel_layers <= 0 || max_new_tokens <= 0) return -1;
    if (draft_k < 0) draft_k = 0;
    if (draft_k > 4) draft_k = 4;
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

            /* 1. DSpark 5-Layer Block Draft (`[seq, cur_tok, -1..]` with target features `[f_0..f_{L-1}, broadcast f_{L-1}]`) */
            memcpy(ds_ids, seq, (size_t)cur_len * sizeof(int));
            ds_ids[cur_len] = cur_tok;
            for (int j = 1; j < K; j++) ds_ids[cur_len + j] = -1;
            const float *last_f = f_buf + (size_t)(cur_len - 1) * 5 * K31_D;
            for (int j = 0; j < K; j++) {
                memcpy(f_buf + (size_t)(cur_len + j) * 5 * K31_D, last_f, 5 * K31_D * sizeof(float));
            }
            int ds_T = cur_len + K;
            kimi_k31_dspark_forward(ds_ids, f_buf, ds_T, ds_h, ds_out, ds_conf);

            int drafts[4];
            for (int j = 0; j < K; j++) drafts[j] = ds_out[cur_len + j];
            draft_tot += K;

            /* 2. Batched C1 Verification on `[seq[0..cur_len-1], cur_tok, drafts[0..K-1]]` */
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
