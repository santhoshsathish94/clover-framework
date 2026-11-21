/*
 * Native C11 / OpenMP `HG4/HG8-128` Engine for DiffuSpace `Clover-35B-A3B-Preview` (`libclover35b_hg4_engine.so`)
 * (`clover-ai-system/src/models/clover-35b-a3b/clover35b_hg4_engine.c`)
 *
 * First native C11 / SIMD runtime for `Clover-35B-A3B` (`D = 2048`, `V = 248,320`), supporting BOTH
 * Causal Commit mode (`preview = 0`) and Bidirectional Block-Diffusion Preview mode (`preview = 1`):
 *   - Layer 0: Bidirectional Gated DeltaNet (`BD-GDN`: causal DeltaNet `32 x 128 x 128` recurrence +
 *     bidirectional `bd_gdn_conv1d` & `bd_gdn_local_memory` preview) + SwiGLU MoE (`coreset_top_p = 0.9`)
 *   - Layer 3: Gated Full-Attention (`16Q / 2KV`, `head_dim = 256`, Partial RoPE `0.25`, `theta = 1e7`) + SwiGLU MoE
 *   - Register-Fused `[gate, up]` SwiGLU `HG4-128` Experts with Shared-Input `FWHT-128` & SwiGLU Equilibrium `pre_diag`
 *   - 1-Row Inverse-`FWHT-128` `248,320`-Vocab Embedding Lookup + Direct VNNI `HG8-128` `248,320`-Vocab `lm_head`.
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
#define C35_X86 1
#define C35_TGT_AVX2       __attribute__((target("avx2,fma,f16c")))
#define C35_TGT_AVX512VNNI __attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,avx512vnni,avx512vbmi,fma,f16c")))
#else
#define C35_X86 0
#define C35_TGT_AVX2
#define C35_TGT_AVX512VNNI
#endif

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#define C35_API __declspec(dllexport)
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#define C35_API __attribute__((visibility("default")))
#endif

enum {
    C35_D          = 2048,
    C35_VOCAB      = 248320,
    C35_MOE_INT    = 512,
    C35_TOP_K      = 8,
    C35_MAX_LAYERS = 40,
    C35_MAX_SEQ    = 4096,
    C35_PANEL_MAX  = 256
};

static inline uint16_t f32_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

static inline float bf16_to_f32(uint16_t b) {
    uint32_t u = ((uint32_t)b) << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

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

typedef struct {
    void  *ptr;
    size_t size;
#ifdef _WIN32
    HANDLE hFile, hMap;
#else
    int fd;
#endif
} c35_mmap_t;

static int c35_mmap_open(const char *path, c35_mmap_t *m) {
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

static void c35_mmap_close(c35_mmap_t *m) {
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
} c35_mat_t;

static void parse_c35_mat(const uint8_t *blob, c35_mat_t *v) {
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
    uint32_t     k_cached;
    const float *gate_w; /* [k_cached * 2048] */
    const float *sh_g;   /* [2048] */
    c35_mat_t    sh_w13; /* [1024, 2048] fused SwiGLU */
    c35_mat_t    sh_dn;  /* [2048, 512] */
} c35_moe_tail_t;

typedef struct {
    c35_mmap_t    mf;
    int           is_16r;
    uint32_t      k_exp;
    uint32_t      exp_stride;
    const float  *pre_diag_up; /* [2048] */
    const uint8_t *exp_base;
} c35_experts_t;

static void parse_c35_moe_tail(const uint8_t *p, c35_moe_tail_t *m) {
    uint32_t l_sh13, l_shdn;
    memcpy(&m->k_cached, p + 0, 4);
    memcpy(&l_sh13,      p + 4, 4);
    memcpy(&l_shdn,      p + 8, 4);
    size_t off = 12;
    m->gate_w = (const float *)(p + off); off += (size_t)m->k_cached * C35_D * 4;
    m->sh_g   = (const float *)(p + off); off += C35_D * 4;
    parse_c35_mat(p + off, &m->sh_w13);   off += l_sh13;
    parse_c35_mat(p + off, &m->sh_dn);
}

static int open_c35_experts(const char *path, c35_experts_t *ex) {
    if (c35_mmap_open(path, &ex->mf) != 0) return -1;
    const uint8_t *p = (const uint8_t *)ex->mf.ptr;
    if (memcmp(p, "C35F", 4) == 0) ex->is_16r = 1;
    else if (memcmp(p, "C35E", 4) == 0) ex->is_16r = 0;
    else return -2;
    memcpy(&ex->k_exp, p + 4, 4);
    memcpy(&ex->exp_stride, p + 24, 4);
    ex->pre_diag_up = (const float *)(p + 28);
    ex->exp_base = p + 28 + C35_D * 4;
    return 0;
}

typedef struct {
    uint32_t       kind; /* 0 = BD-GDN, 1 = Gated GQA */
    c35_mmap_t     mf;
    const float   *in_ln, *post_ln;
    /* BD-GDN fields (kind == 0) */
    const float   *conv_w, *conv_rw, *a_w, *b_w, *neg_exp_A, *dt_bias, *gdn_norm, *ls, *hb, *hc;
    c35_mat_t      qkv, z, out;
    /* Gated GQA fields (kind == 1) */
    const float   *qn, *kn;
    c35_mat_t      q, k, v, o;
    /* Shared + Routed MoE */
    c35_moe_tail_t moe;
    c35_experts_t  ex;
    /* Stateful recurrent / KV caches */
    float         *gdn_state;     /* [32 * 128 * 128] when kind == 0 */
    float         *conv_hist;     /* [3 * 8192] when kind == 0 */
    uint16_t      *k_cache_bf16;  /* [C35_MAX_SEQ * 512] when kind == 1 */
    uint16_t      *v_cache_bf16;  /* [C35_MAX_SEQ * 512] when kind == 1 */
} c35_layer_t;

static struct {
    int ready;
    int is_16r;
    int isa_tier;
    int n_layers;
    int cache_len;
    char loaded_dir[512];

    uint8_t lut16_u8[16];
    int8_t  lut16_s8[16];
    float   lut16_s8_f[16];
    float   norm_w[C35_D];
    float   inv_freq[32];

    c35_mmap_t  emb_mf, lm_mf;
    c35_mat_t   emb_mat, lm_mat;
    c35_layer_t layers[C35_MAX_LAYERS];

    /* Preallocated hot-path scratch buffers (zero malloc in T=1..256 decode/prefill) */
    int8_t  *scr_xq;
    float   *scr_xsc;
    int32_t *scr_xsum;
    float   *scr_moe_hid; /* [C35_TOP_K * C35_MOE_INT] for T=1 */
    float   *scr_moe_out; /* [C35_TOP_K * C35_D] for T=1 */
} g_c35;

static int detect_c35_isa(void) {
#if C35_X86
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

C35_API const char *clover35b_hg4_isa_name(void) {
    int t = g_c35.ready ? g_c35.isa_tier : detect_c35_isa();
    if (t == 3) return "AVX-512 VNNI+VBMI (HG4/HG8-16R + Shrinkage-LDLQ + Int32-FWHT + Native BD-GDN)";
    if (t == 1) return "AVX2+FMA (HG4/HG8-16R + abs/sign LUT16 + Int32-FWHT + Native BD-GDN)";
    return "Scalar C11 (HG4/HG8-16R + Int32-FWHT + Native BD-GDN)";
}

C35_API int clover35b_hg4_num_layers(void) {
    return g_c35.n_layers;
}

C35_API void clover35b_hg4_set_isa_tier(int tier) {
    int max_hw = detect_c35_isa();
    if (tier < 0) g_c35.isa_tier = max_hw;
    else if (tier == 0) g_c35.isa_tier = 0;
    else if (tier == 1 && max_hw >= 1) g_c35.isa_tier = 1;
    else if (tier == 3 && max_hw == 3) g_c35.isa_tier = 3;
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

#if C35_X86
C35_TGT_AVX2 static void rotate_and_quantize_128_avx2(
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
                    #define C35_Q8(off) _mm256_cvtps_epi32(_mm256_min_ps(vmax_c, _mm256_max_ps(vmin_c, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i *)(rh + (off)))), vinv))))
                    __m256i i0 = C35_Q8(0),  i1 = C35_Q8(8),  i2 = C35_Q8(16), i3 = C35_Q8(24);
                    __m256i i4 = C35_Q8(32), i5 = C35_Q8(40), i6 = C35_Q8(48), i7 = C35_Q8(56);
                    #undef C35_Q8
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

C35_TGT_AVX512VNNI static inline __m128i hsum4_zmm_epi32(__m512i z0, __m512i z1, __m512i z2, __m512i z3) {
    __m256i y0 = _mm256_add_epi32(_mm512_castsi512_si256(z0), _mm512_extracti32x8_epi32(z0, 1));
    __m256i y1 = _mm256_add_epi32(_mm512_castsi512_si256(z1), _mm512_extracti32x8_epi32(z1, 1));
    __m256i y2 = _mm256_add_epi32(_mm512_castsi512_si256(z2), _mm512_extracti32x8_epi32(z2, 1));
    __m256i y3 = _mm256_add_epi32(_mm512_castsi512_si256(z3), _mm512_extracti32x8_epi32(z3, 1));
    __m256i h01 = _mm256_hadd_epi32(y0, y1);
    __m256i h23 = _mm256_hadd_epi32(y2, y3);
    __m256i h0123 = _mm256_hadd_epi32(h01, h23);
    return _mm_add_epi32(_mm256_castsi256_si128(h0123), _mm256_extracti128_si256(h0123, 1));
}

C35_TGT_AVX512VNNI static void c35_tile4_avx512(
    const c35_mat_t *M, const uint8_t *lut16_u8,
    const int8_t *X_q, const float *X_sc, const int32_t *X_sum,
    float *Y, int T, int tile_idx, int fused_swiglu
) {
    const int R = (int)M->R, K = (int)M->K, nb = K / 128;
    const int out_R = fused_swiglu ? (R / 2) : R;
    const int o_base = tile_idx * 4;
    const size_t tstride = (M->bits == 8) ? 512u : 256u;
    const uint8_t *wt = M->tiles + (size_t)tile_idx * (size_t)nb * tstride;
    const uint16_t *wsc = M->scales_fp16 + (size_t)tile_idx * (size_t)nb * 4;
    const __m512i vlut = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)lut16_u8));
    const __m256i m0f  = _mm256_set1_epi8(0x0F);

    for (int t0 = 0; t0 < T; t0 += 32) {
        int tblk = (t0 + 32 <= T) ? 32 : (T - t0);
        __m128 vs[32];
        for (int ti = 0; ti < tblk; ti++) vs[ti] = _mm_setzero_ps();

        for (int b = 0; b < nb; b++) {
            const uint8_t *wb = wt + b * tstride;
            __m512i wu8[2][4];
            if (M->bits == 8) {
                for (int u = 0; u < 2; u++) {
                    const uint8_t *wu = wb + u * 256;
                    for (int ri = 0; ri < 4; ri++)
                        wu8[u][ri] = _mm512_loadu_si512((const __m512i *)(wu + ri * 64));
                }
            } else {
                for (int u = 0; u < 2; u++) {
                    const uint8_t *wu = wb + u * 128;
                    for (int ri = 0; ri < 4; ri++) {
                        __m256i pk = _mm256_loadu_si256((const __m256i *)(wu + ri * 32));
                        __m512i nib = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_and_si256(pk, m0f)),
                                                          _mm256_and_si256(_mm256_srli_epi16(pk, 4), m0f), 1);
                        wu8[u][ri] = _mm512_permutexvar_epi8(nib, vlut);
                    }
                }
            }
            __m128 vws = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)(wsc + b * 4)));
            int ti = 0;
            for (; ti + 4 <= tblk; ti += 4) {
                __m512i a[4][4];
                for (int m = 0; m < 4; m++)
                    for (int ri = 0; ri < 4; ri++)
                        a[m][ri] = _mm512_setzero_si512();
                for (int u = 0; u < 2; u++) {
                    for (int m = 0; m < 4; m++) {
                        __m512i xv = _mm512_loadu_si512((const __m512i *)(X_q + (size_t)(t0 + ti + m) * K + b * 128 + u * 64));
                        a[m][0] = _mm512_dpbusd_epi32(a[m][0], wu8[u][0], xv);
                        a[m][1] = _mm512_dpbusd_epi32(a[m][1], wu8[u][1], xv);
                        a[m][2] = _mm512_dpbusd_epi32(a[m][2], wu8[u][2], xv);
                        a[m][3] = _mm512_dpbusd_epi32(a[m][3], wu8[u][3], xv);
                    }
                }
                for (int m = 0; m < 4; m++) {
                    int t_abs = t0 + ti + m;
                    __m128i sums4 = _mm_sub_epi32(hsum4_zmm_epi32(a[m][0], a[m][1], a[m][2], a[m][3]),
                                                  _mm_set1_epi32(128 * X_sum[t_abs * nb + b]));
                    vs[ti + m] = _mm_fmadd_ps(_mm_cvtepi32_ps(sums4), _mm_mul_ps(vws, _mm_set1_ps(X_sc[t_abs * nb + b])), vs[ti + m]);
                }
            }
            for (; ti < tblk; ti++) {
                int t_abs = t0 + ti;
                __m512i a0 = _mm512_setzero_si512(), a1 = _mm512_setzero_si512();
                __m512i a2 = _mm512_setzero_si512(), a3 = _mm512_setzero_si512();
                for (int u = 0; u < 2; u++) {
                    __m512i xv = _mm512_loadu_si512((const __m512i *)(X_q + (size_t)t_abs * K + b * 128 + u * 64));
                    a0 = _mm512_dpbusd_epi32(a0, wu8[u][0], xv);
                    a1 = _mm512_dpbusd_epi32(a1, wu8[u][1], xv);
                    a2 = _mm512_dpbusd_epi32(a2, wu8[u][2], xv);
                    a3 = _mm512_dpbusd_epi32(a3, wu8[u][3], xv);
                }
                __m128i sums4 = _mm_sub_epi32(hsum4_zmm_epi32(a0, a1, a2, a3), _mm_set1_epi32(128 * X_sum[t_abs * nb + b]));
                vs[ti] = _mm_fmadd_ps(_mm_cvtepi32_ps(sums4), _mm_mul_ps(vws, _mm_set1_ps(X_sc[t_abs * nb + b])), vs[ti]);
            }
        }
        for (int ti = 0; ti < tblk; ti++) {
            float *yt = Y + (size_t)(t0 + ti) * out_R;
            if (fused_swiglu) {
                float s[4];
                _mm_storeu_ps(s, vs[ti]);
                yt[tile_idx * 2 + 0] = (s[0] / (1.0f + expf(-s[0]))) * s[2];
                yt[tile_idx * 2 + 1] = (s[1] / (1.0f + expf(-s[1]))) * s[3];
            } else {
                _mm_storeu_ps(yt + o_base, vs[ti]);
            }
        }
    }
}

C35_TGT_AVX2 static void c35_tile4_avx2(
    const c35_mat_t *M, const uint8_t *lut16_u8,
    const int8_t *X_q, const float *X_sc, const int32_t *X_sum,
    float *Y, int T, int tile_idx, int fused_swiglu
) {
    const int R = (int)M->R, K = (int)M->K, nb = K / 128;
    const int out_R = fused_swiglu ? (R / 2) : R;
    const int o_base = tile_idx * 4;
    const size_t tstride = (M->bits == 8) ? 512u : 256u;
    const uint8_t *wt = M->tiles + (size_t)tile_idx * (size_t)nb * tstride;
    const uint16_t *wsc = M->scales_fp16 + (size_t)tile_idx * (size_t)nb * 4;
    const __m256i vlut = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)lut16_u8));
    const __m256i m0f  = _mm256_set1_epi8(0x0F);
    const __m256i ones16 = _mm256_set1_epi16(1);

    for (int t0 = 0; t0 < T; t0 += 16) {
        int tblk = (t0 + 16 <= T) ? 16 : (T - t0);
        __m128 vs[16];
        for (int ti = 0; ti < tblk; ti++) vs[ti] = _mm_setzero_ps();
        for (int b = 0; b < nb; b++) {
            const uint8_t *wb = wt + b * tstride;
            __m128 vws = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)(wsc + b * 4)));
            for (int ti = 0; ti < tblk; ti++) {
                int t_abs = t0 + ti;
                __m256i acc[4] = {_mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256()};
                const int8_t *xq = X_q + (size_t)t_abs * K + b * 128;
                for (int u = 0; u < 2; u++) {
                    __m256i xlo = _mm256_loadu_si256((const __m256i *)(xq + u * 64 + 0));
                    __m256i xhi = _mm256_loadu_si256((const __m256i *)(xq + u * 64 + 32));
                    for (int ri = 0; ri < 4; ri++) {
                        __m256i wlo, whi;
                        if (M->bits == 8) {
                            const uint8_t *wu = wb + u * 256 + ri * 64;
                            wlo = _mm256_loadu_si256((const __m256i *)(wu + 0));
                            whi = _mm256_loadu_si256((const __m256i *)(wu + 32));
                        } else {
                            __m256i pk = _mm256_loadu_si256((const __m256i *)(wb + u * 128 + ri * 32));
                            wlo = _mm256_shuffle_epi8(vlut, _mm256_and_si256(pk, m0f));
                            whi = _mm256_shuffle_epi8(vlut, _mm256_and_si256(_mm256_srli_epi16(pk, 4), m0f));
                        }
                        __m256i p16 = _mm256_add_epi16(_mm256_maddubs_epi16(wlo, xlo), _mm256_maddubs_epi16(whi, xhi));
                        acc[ri] = _mm256_add_epi32(acc[ri], _mm256_madd_epi16(p16, ones16));
                    }
                }
                __m256i h01 = _mm256_hadd_epi32(acc[0], acc[1]);
                __m256i h23 = _mm256_hadd_epi32(acc[2], acc[3]);
                __m256i h0123 = _mm256_hadd_epi32(h01, h23);
                __m128i sums4 = _mm_sub_epi32(_mm_add_epi32(_mm256_castsi256_si128(h0123), _mm256_extracti128_si256(h0123, 1)),
                                              _mm_set1_epi32(128 * X_sum[t_abs * nb + b]));
                vs[ti] = _mm_add_ps(vs[ti], _mm_mul_ps(_mm_cvtepi32_ps(sums4), _mm_mul_ps(vws, _mm_set1_ps(X_sc[t_abs * nb + b]))));
            }
        }
        for (int ti = 0; ti < tblk; ti++) {
            float *yt = Y + (size_t)(t0 + ti) * out_R;
            if (fused_swiglu) {
                float s[4];
                _mm_storeu_ps(s, vs[ti]);
                yt[tile_idx * 2 + 0] = (s[0] / (1.0f + expf(-s[0]))) * s[2];
                yt[tile_idx * 2 + 1] = (s[1] / (1.0f + expf(-s[1]))) * s[3];
            } else {
                _mm_storeu_ps(yt + o_base, vs[ti]);
            }
        }
    }
}

/* Canonical 16-Row `HG4-16R` & `HG8-16R` Kernels (Zero Horizontal Reductions) */
C35_TGT_AVX512VNNI static void c35_group16_avx512(
    const c35_mat_t *M, const uint8_t *lut16_u8,
    const int8_t *X_q, const float *X_sc, const int32_t *X_sum,
    float *Y, int T, int rg, int fused_swiglu
) {
    const int R = (int)M->R, K = (int)M->K, nb = K / 128;
    const int out_R = fused_swiglu ? (R / 2) : R;
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
            if (fused_swiglu) {
                float sv[16];
                _mm512_storeu_ps(sv, vs[ti]);
                for (int j = 0; j < 8; j++) yt[rg * 8 + j] = (sv[j] / (1.0f + expf(-sv[j]))) * sv[j + 8];
            } else {
                _mm512_storeu_ps(yt + rg * 16, vs[ti]);
            }
        }
    }
}

C35_TGT_AVX2 static void c35_group16_avx2(
    const c35_mat_t *M, const int8_t *lut16_s8,
    const int8_t *X_q, const float *X_sc,
    float *Y, int T, int rg, int fused_swiglu
) {
    const int R = (int)M->R, K = (int)M->K, nb = K / 128;
    const int out_R = fused_swiglu ? (R / 2) : R;
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
            if (fused_swiglu) {
                float g8[8], u8[8];
                _mm256_storeu_ps(g8, vs0[ti]);
                _mm256_storeu_ps(u8, vs1[ti]);
                for (int j = 0; j < 8; j++) yt[rg * 8 + j] = (g8[j] / (1.0f + expf(-g8[j]))) * u8[j];
            } else {
                _mm256_storeu_ps(yt + rg * 16 + 0, vs0[ti]);
                _mm256_storeu_ps(yt + rg * 16 + 8, vs1[ti]);
            }
        }
    }
}
#endif

static void c35_group16_scalar(
    const c35_mat_t *M, const int8_t *lut16_s8,
    const int8_t *X_q, const float *X_sc,
    float *Y, int T, int rg, int fused_swiglu
) {
    const int R = (int)M->R, K = (int)M->K, nb = K / 128;
    const int out_R = fused_swiglu ? (R / 2) : R;
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
        if (fused_swiglu) {
            for (int j = 0; j < 8; j++) yt[rg * 8 + j] = (vs[j] / (1.0f + expf(-vs[j]))) * vs[j + 8];
        } else {
            for (int r = 0; r < 16; r++) yt[rg * 16 + r] = vs[r];
        }
    }
}

static inline void c35_rot_quant_dispatch(const float *X, const float *pd, int T, int K, int8_t *xq, float *xsc, int32_t *xsum, int use_omp) {
#if C35_X86
    if (g_c35.isa_tier >= 1) {
        rotate_and_quantize_128_avx2(X, pd, T, K, xq, xsc, xsum, use_omp);
        return;
    }
#endif
    rotate_and_quantize_128_scalar(X, pd, T, K, xq, xsc, xsum, use_omp);
}

static inline void c35_run_one_group16(const c35_mat_t *M, const int8_t *xq, const float *xsc, const int32_t *xsum, float *Y, int T, int rg, int fused_swiglu) {
#if C35_X86
    if (g_c35.isa_tier == 3) { c35_group16_avx512(M, g_c35.lut16_u8, xq, xsc, xsum, Y, T, rg, fused_swiglu); return; }
    if (g_c35.isa_tier >= 1) { c35_group16_avx2(M, g_c35.lut16_s8, xq, xsc, Y, T, rg, fused_swiglu); return; }
#endif
    c35_group16_scalar(M, g_c35.lut16_s8, xq, xsc, Y, T, rg, fused_swiglu);
}

static void c35_matmul_prequant(
    const c35_mat_t *M, const int8_t *xq, const float *xsc, const int32_t *xsum,
    float *Y, int T, int fused_swiglu, int use_omp
) {
    if (g_c35.is_16r) {
        const int n_rg = (int)(M->R / 16);
        #pragma omp parallel for schedule(static) if (use_omp)
        for (int rg = 0; rg < n_rg; rg++)
            c35_run_one_group16(M, xq, xsc, xsum, Y, T, rg, fused_swiglu);
        return;
    }
    const int n_tiles = (int)(M->R / 4);
#if C35_X86
    if (g_c35.isa_tier == 3) {
        #pragma omp parallel for schedule(static) if (use_omp)
        for (int tl = 0; tl < n_tiles; tl++)
            c35_tile4_avx512(M, g_c35.lut16_u8, xq, xsc, xsum, Y, T, tl, fused_swiglu);
        return;
    } else if (g_c35.isa_tier >= 1) {
        #pragma omp parallel for schedule(static) if (use_omp)
        for (int tl = 0; tl < n_tiles; tl++)
            c35_tile4_avx2(M, g_c35.lut16_u8, xq, xsc, xsum, Y, T, tl, fused_swiglu);
        return;
    }
#endif
    (void)xq; (void)xsc; (void)xsum;
}

static void c35_matmul(const c35_mat_t *M, const float *X, float *Y, int T, int fused_swiglu, int use_omp) {
    const int K = (int)M->K, nb = K / 128;
    int use_global_scr = (use_omp && T <= C35_PANEL_MAX && K <= 4096 && g_c35.scr_xq != NULL);
    int8_t  *xq   = use_global_scr ? g_c35.scr_xq   : (int8_t *)malloc((size_t)T * K);
    float   *xsc  = use_global_scr ? g_c35.scr_xsc  : (float *)malloc((size_t)T * nb * sizeof(float));
    int32_t *xsum = use_global_scr ? g_c35.scr_xsum : (int32_t *)malloc((size_t)T * nb * sizeof(int32_t));
    c35_rot_quant_dispatch(X, M->pre_diag, T, K, xq, xsc, xsum, use_omp);
    c35_matmul_prequant(M, xq, xsc, xsum, Y, T, fused_swiglu, use_omp);
    if (!use_global_scr) { free(xq); free(xsc); free(xsum); }
}

static void c35_embed_lookup_row(int token_id, float *dst_2048) {
    if (token_id < 0 || token_id >= C35_VOCAB) token_id = 0;
    const c35_mat_t *E = &g_c35.emb_mat;
    const int nb = C35_D / 128;

    if (g_c35.is_16r) {
        int rg = token_id / 16, lane = token_id % 16;
        const size_t bstride = (E->bits == 8) ? 2048u : 1024u;
        const uint8_t *wt = E->tiles + (size_t)rg * (size_t)nb * bstride;
        const uint16_t *wsc = E->scales_fp16 + (size_t)rg * (size_t)nb * 16;
        for (int b = 0; b < nb; b++) {
            float *blk = dst_2048 + b * 128;
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
                        blk[s * 8 + j]     = g_c35.lut16_s8_f[pk & 0x0F] * sc;
                        blk[s * 8 + 4 + j] = g_c35.lut16_s8_f[pk >> 4]   * sc;
                    }
                }
            }
            fwht128_1block_scalar(blk);
            for (int k = 0; k < 128; k++) blk[k] *= E->pre_diag[b * 128 + k];
        }
        return;
    }

    int tl = token_id / 4, ri = token_id % 4;
    const size_t tstride = (E->bits == 8) ? 512u : 256u;
    const uint8_t *wt = E->tiles + (size_t)tl * (size_t)nb * tstride;
    const uint16_t *wsc = E->scales_fp16 + (size_t)tl * (size_t)nb * 4;

    for (int b = 0; b < nb; b++) {
        float *blk = dst_2048 + b * 128;
        float sc = fp16_to_f32(wsc[b * 4 + ri]);
        const uint8_t *wb = wt + b * tstride;
        if (E->bits == 8) {
            for (int u = 0; u < 2; u++) {
                const uint8_t *wu = wb + u * 256 + ri * 64;
                for (int j = 0; j < 64; j++) blk[u * 64 + j] = (float)((int)wu[j] - 128) * sc;
            }
        } else {
            for (int u = 0; u < 2; u++) {
                const uint8_t *wu = wb + u * 128 + ri * 32;
                for (int j = 0; j < 32; j++) {
                    uint8_t pk = wu[j];
                    blk[u * 64 + j]      = g_c35.lut16_s8_f[pk & 0x0F] * sc;
                    blk[u * 64 + 32 + j] = g_c35.lut16_s8_f[pk >> 4]   * sc;
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
    for (int r = 0; r < rows; r++) {
        rms_norm_row(x + (size_t)r * dim, w, y + (size_t)r * dim, dim, eps);
    }
}

/* SwiGLU MoE (`top-8` + `coreset_routing_op` when `preview == 1`) + Gated Shared SwiGLU Expert */
static void c35_run_moe(
    const float *x, float *y_out, int T,
    const c35_moe_tail_t *mt, const c35_experts_t *ex, int preview
) {
    const int E = (int)mt->k_cached;
    const int top_k = (C35_TOP_K < E) ? C35_TOP_K : E;

    /* Fast zero-malloc 16-core path for T == 1 Causal Decode (`512` gu row-groups + `1024` dn row-groups across 16 cores!) */
    if (T == 1 && !preview && g_c35.is_16r && g_c35.scr_moe_hid != NULL) {
        float logits1[256];
        #pragma omp parallel for schedule(static) if (E >= 64)
        for (int e = 0; e < E; e++) {
            const float *gw = mt->gate_w + (size_t)e * C35_D;
            float s = 0.0f;
            #pragma omp simd reduction(+:s)
            for (int d = 0; d < C35_D; d++) s += x[d] * gw[d];
            logits1[e] = s;
        }
        float mx = logits1[0];
        for (int e = 1; e < E; e++) if (logits1[e] > mx) mx = logits1[e];
        float probs[256], psum = 0.0f;
        for (int e = 0; e < E; e++) { probs[e] = expf(logits1[e] - mx); psum += probs[e]; }
        for (int e = 0; e < E; e++) probs[e] /= (psum + 1e-12f);

        int sel_e[C35_TOP_K];
        float sel_w[C35_TOP_K], wsum = 0.0f;
        uint8_t picked[256] = {0};
        for (int k = 0; k < top_k; k++) {
            int best_e = 0; float best_v = -1e30f;
            for (int e = 0; e < E; e++) if (!picked[e] && logits1[e] > best_v) { best_v = logits1[e]; best_e = e; }
            picked[best_e] = 1;
            sel_e[k] = best_e; sel_w[k] = probs[best_e]; wsum += probs[best_e];
        }
        for (int k = 0; k < top_k; k++) sel_w[k] /= (wsum + 1e-12f);

        int8_t  x_xq[C35_D];
        float   x_xsc[C35_D / 128];
        int32_t x_xsum[C35_D / 128];
        c35_rot_quant_dispatch(x, ex->pre_diag_up, 1, C35_D, x_xq, x_xsc, x_xsum, 0);

        c35_mat_t gu_m[C35_TOP_K], dn_m[C35_TOP_K];
        for (int k = 0; k < top_k; k++) {
            const uint8_t *ep = ex->exp_base + (size_t)sel_e[k] * ex->exp_stride;
            gu_m[k].R = 1024; gu_m[k].K = 2048; gu_m[k].bits = 4;
            gu_m[k].pre_diag = ex->pre_diag_up;
            gu_m[k].scales_fp16 = (const uint16_t *)ep;
            gu_m[k].tiles = ep + 32768;
            parse_c35_mat(ep + 1081344, &dn_m[k]);
        }

        /* 1. Fused SwiGLU up/gate across all `top_k * 64 = 512` row-groups on 16 cores */
        #pragma omp parallel for schedule(static)
        for (int task = 0; task < top_k * 64; task++) {
            int k = task / 64, rg = task % 64;
            c35_run_one_group16(&gu_m[k], x_xq, x_xsc, x_xsum, g_c35.scr_moe_hid + k * C35_MOE_INT, 1, rg, 1);
        }

        /* 2. Rotate each expert's 512-dim hidden state & run `top_k * 128 = 1024` down_proj row-groups on 16 cores */
        int8_t  z_xq[C35_TOP_K * C35_MOE_INT];
        float   z_xsc[C35_TOP_K * 4];
        int32_t z_xsum[C35_TOP_K * 4];
        for (int k = 0; k < top_k; k++) {
            c35_rot_quant_dispatch(g_c35.scr_moe_hid + k * C35_MOE_INT, dn_m[k].pre_diag, 1, C35_MOE_INT,
                                   z_xq + k * C35_MOE_INT, z_xsc + k * 4, z_xsum + k * 4, 0);
        }
        #pragma omp parallel for schedule(static)
        for (int task = 0; task < top_k * 128; task++) {
            int k = task / 128, rg = task % 128;
            c35_run_one_group16(&dn_m[k], z_xq + k * C35_MOE_INT, z_xsc + k * 4, z_xsum + k * 4,
                                g_c35.scr_moe_out + k * C35_D, 1, rg, 0);
        }

        /* 3. Shared expert + weighted expert sum */
        float sh_hid[C35_MOE_INT], sh_out[C35_D];
        c35_matmul(&mt->sh_w13, x, sh_hid, 1, 1, 1);
        c35_matmul(&mt->sh_dn, sh_hid, sh_out, 1, 0, 1);
        float gdot = 0.0f;
        #pragma omp simd reduction(+:gdot)
        for (int d = 0; d < C35_D; d++) gdot += x[d] * mt->sh_g[d];
        float sh_sc = 1.0f / (1.0f + expf(-gdot));

        for (int d = 0; d < C35_D; d++) {
            float acc = sh_out[d] * sh_sc;
            for (int k = 0; k < top_k; k++) acc += g_c35.scr_moe_out[k * C35_D + d] * sel_w[k];
            y_out[d] = acc;
        }
        return;
    }

    float *logits = (float *)malloc((size_t)T * E * sizeof(float));
    #pragma omp parallel for schedule(static) if (T * E >= 256)
    for (int t = 0; t < T; t++) {
        const float *xt = x + (size_t)t * C35_D;
        for (int e = 0; e < E; e++) {
            const float *gw = mt->gate_w + (size_t)e * C35_D;
            float s = 0.0f;
            #pragma omp simd reduction(+:s)
            for (int d = 0; d < C35_D; d++) s += xt[d] * gw[d];
            logits[t * E + e] = s;
        }
    }

    if (preview) {
        float totals[256] = {0};
        for (int t = 0; t < T; t++) {
            float *lt = logits + t * E;
            float mx = lt[0];
            for (int e = 1; e < E; e++) if (lt[e] > mx) mx = lt[e];
            float probs[256], psum = 0.0f;
            for (int e = 0; e < E; e++) { probs[e] = expf(lt[e] - mx); psum += probs[e]; }
            for (int e = 0; e < E; e++) probs[e] /= psum;

            uint8_t picked[256] = {0};
            for (int k = 0; k < top_k; k++) {
                int best_e = 0; float best_v = -1e30f;
                for (int e = 0; e < E; e++) if (!picked[e] && lt[e] > best_v) { best_v = lt[e]; best_e = e; }
                picked[best_e] = 1;
                totals[best_e] += probs[best_e];
            }
        }
        float tsum = 0.0f;
        for (int e = 0; e < E; e++) tsum += totals[e];
        if (tsum < 1e-9f) tsum = 1e-9f;
        int order[256];
        for (int e = 0; e < E; e++) { totals[e] /= tsum; order[e] = e; }
        for (int i = 0; i < E - 1; i++) {
            for (int j = 0; j < E - 1 - i; j++) {
                if (totals[order[j]] < totals[order[j + 1]]) {
                    int tmp = order[j]; order[j] = order[j + 1]; order[j + 1] = tmp;
                }
            }
        }
        uint8_t allowed[256] = {0};
        float cum = 0.0f;
        for (int i = 0; i < E; i++) {
            if (i == 0 || cum < 0.9f) allowed[order[i]] = 1;
            cum += totals[order[i]];
        }
        for (int t = 0; t < T; t++) {
            for (int e = 0; e < E; e++) if (!allowed[e]) logits[t * E + e] = -1e30f;
        }
    }

    int   *topk_ids = (int *)malloc((size_t)T * top_k * sizeof(int));
    float *topk_w   = (float *)malloc((size_t)T * top_k * sizeof(float));
    uint8_t active_mask[256] = {0};

    for (int t = 0; t < T; t++) {
        const float *lt = logits + t * E;
        float mx = lt[0];
        for (int e = 1; e < E; e++) if (lt[e] > mx) mx = lt[e];
        float probs[256], psum = 0.0f;
        for (int e = 0; e < E; e++) { probs[e] = expf(lt[e] - mx); psum += probs[e]; }
        for (int e = 0; e < E; e++) probs[e] /= (psum + 1e-12f);

        uint8_t picked[256] = {0};
        float wsum = 0.0f;
        for (int k = 0; k < top_k; k++) {
            int best_e = 0; float best_v = -1e30f;
            for (int e = 0; e < E; e++) if (!picked[e] && lt[e] > best_v) { best_v = lt[e]; best_e = e; }
            picked[best_e] = 1;
            topk_ids[t * top_k + k] = best_e;
            topk_w[t * top_k + k] = probs[best_e];
            wsum += probs[best_e];
            active_mask[best_e] = 1;
        }
        for (int k = 0; k < top_k; k++) topk_w[t * top_k + k] /= (wsum + 1e-12f);
    }
    free(logits);

    memset(y_out, 0, (size_t)T * C35_D * sizeof(float));

    const int nb_up = C35_D / 128;
    int8_t  *x_xq   = (int8_t *)malloc((size_t)T * C35_D);
    float   *x_xsc  = (float *)malloc((size_t)T * nb_up * sizeof(float));
    int32_t *x_xsum = (int32_t *)malloc((size_t)T * nb_up * sizeof(int32_t));
    c35_rot_quant_dispatch(x, ex->pre_diag_up, T, C35_D, x_xq, x_xsc, x_xsum, 1);

    int active_list[256], n_act = 0;
    for (int e = 0; e < E; e++) if (active_mask[e]) active_list[n_act++] = e;

    float *exp_acc = (float *)calloc((size_t)n_act * T * C35_D, sizeof(float));
    #pragma omp parallel for schedule(dynamic, 1)
    for (int ai = 0; ai < n_act; ai++) {
        int e = active_list[ai];
        int   *m_toks = (int *)malloc((size_t)T * sizeof(int));
        float *m_rw   = (float *)malloc((size_t)T * sizeof(float));
        int8_t  *g_xq   = (int8_t *)malloc((size_t)T * C35_D);
        float   *g_xsc  = (float *)malloc((size_t)T * nb_up * sizeof(float));
        int32_t *g_xsum = (int32_t *)malloc((size_t)T * nb_up * sizeof(int32_t));
        int n_m = 0;
        for (int t = 0; t < T; t++) {
            for (int k = 0; k < top_k; k++) {
                if (topk_ids[t * top_k + k] == e) {
                    m_toks[n_m] = t;
                    m_rw[n_m] = topk_w[t * top_k + k];
                    memcpy(g_xq + (size_t)n_m * C35_D, x_xq + (size_t)t * C35_D, C35_D);
                    memcpy(g_xsc + (size_t)n_m * nb_up, x_xsc + (size_t)t * nb_up, nb_up * sizeof(float));
                    memcpy(g_xsum + (size_t)n_m * nb_up, x_xsum + (size_t)t * nb_up, nb_up * sizeof(int32_t));
                    n_m++;
                    break;
                }
            }
        }
        if (n_m > 0) {
            const uint8_t *ep = ex->exp_base + (size_t)e * ex->exp_stride;
            c35_mat_t gu_m, dn_m;
            gu_m.R = 1024; gu_m.K = 2048; gu_m.bits = 4;
            gu_m.pre_diag = ex->pre_diag_up;
            gu_m.scales_fp16 = (const uint16_t *)ep;
            gu_m.tiles = ep + 32768;
            parse_c35_mat(ep + 1081344, &dn_m);

            float *hid = (float *)malloc((size_t)n_m * C35_MOE_INT * sizeof(float));
            float *ey  = (float *)malloc((size_t)n_m * C35_D * sizeof(float));
            c35_matmul_prequant(&gu_m, g_xq, g_xsc, g_xsum, hid, n_m, 1, 0);
            c35_matmul(&dn_m, hid, ey, n_m, 0, 0);

            float *dst_a = exp_acc + (size_t)ai * T * C35_D;
            for (int im = 0; im < n_m; im++) {
                float *dt = dst_a + (size_t)m_toks[im] * C35_D;
                const float *src_e = ey + (size_t)im * C35_D;
                float rw = m_rw[im];
                for (int d = 0; d < C35_D; d++) dt[d] += src_e[d] * rw;
            }
            free(hid); free(ey);
        }
        free(m_toks); free(m_rw); free(g_xq); free(g_xsc); free(g_xsum);
    }
    for (int ai = 0; ai < n_act; ai++) {
        const float *src_a = exp_acc + (size_t)ai * T * C35_D;
        for (int i = 0; i < T * C35_D; i++) y_out[i] += src_a[i];
    }
    free(exp_acc); free(x_xq); free(x_xsc); free(x_xsum);

    float *sh_hid = (float *)malloc((size_t)T * C35_MOE_INT * sizeof(float));
    float *sh_out = (float *)malloc((size_t)T * C35_D * sizeof(float));
    c35_matmul(&mt->sh_w13, x, sh_hid, T, 1, 1);
    c35_matmul(&mt->sh_dn, sh_hid, sh_out, T, 0, 1);

    for (int t = 0; t < T; t++) {
        const float *xt = x + (size_t)t * C35_D;
        float gdot = 0.0f;
        #pragma omp simd reduction(+:gdot)
        for (int d = 0; d < C35_D; d++) gdot += xt[d] * mt->sh_g[d];
        float sh_scale = 1.0f / (1.0f + expf(-gdot));
        float *yt = y_out + (size_t)t * C35_D;
        const float *st = sh_out + (size_t)t * C35_D;
        for (int d = 0; d < C35_D; d++) yt[d] += st[d] * sh_scale;
    }

    free(sh_hid); free(sh_out); free(topk_ids); free(topk_w);
}

/* Generic Layer Execution: Bidirectional Gated DeltaNet (`BD-GDN`) + SwiGLU MoE (stateful or stateless) */
static void c35_forward_layer_bdgdn(c35_layer_t *lyr, const float *x, float *h_out, int pos_off, int T, int preview, int stateful) {
    float *u       = (float *)malloc((size_t)T * C35_D * sizeof(float));
    float *raw_qkv = (float *)malloc((size_t)T * 8192 * sizeof(float));
    float *mixed   = (float *)malloc((size_t)T * 8192 * sizeof(float));
    float *z       = (float *)malloc((size_t)T * 4096 * sizeof(float));
    float *y_attn  = (float *)malloc((size_t)T * 4096 * sizeof(float));
    float *h_attn  = (float *)malloc((size_t)T * C35_D * sizeof(float));
    float *u_mlp   = (float *)malloc((size_t)T * C35_D * sizeof(float));
    float *moe_out = (float *)malloc((size_t)T * C35_D * sizeof(float));

    rms_norm_batch(x, lyr->in_ln, u, T, C35_D, 1e-6f);
    c35_matmul(&lyr->qkv, u, raw_qkv, T, 0, 1);
    c35_matmul(&lyr->z,   u, z,       T, 0, 1);

    #pragma omp parallel for schedule(static)
    for (int c = 0; c < 8192; c++) {
        const float *cw  = lyr->conv_w  + c * 4;
        const float *crw = lyr->conv_rw + c * 3;
        for (int t = 0; t < T; t++) {
            float acc = 0.0f;
            for (int tap = 0; tap < 4; tap++) {
                int rel = t - 3 + tap;
                if (rel >= 0) {
                    acc += raw_qkv[(size_t)rel * 8192 + c] * cw[tap];
                } else if (stateful && !preview && (pos_off + rel) >= 0) {
                    /* Read from `conv_hist[3 + rel]` (`rel` in {-3, -2, -1} -> indices {0, 1, 2}) */
                    acc += lyr->conv_hist[(size_t)(3 + rel) * 8192 + c] * cw[tap];
                }
            }
            if (preview) {
                for (int dist = 1; dist <= 3 && t + dist < T; dist++)
                    acc += raw_qkv[(size_t)(t + dist) * 8192 + c] * crw[dist - 1];
            }
            mixed[(size_t)t * 8192 + c] = acc / (1.0f + expf(-acc));
        }
    }

    if (stateful && !preview && lyr->conv_hist) {
        if (T >= 3) {
            memcpy(lyr->conv_hist, raw_qkv + (size_t)(T - 3) * 8192, 3 * 8192 * sizeof(float));
        } else if (T == 2) {
            memmove(lyr->conv_hist, lyr->conv_hist + 2 * 8192, 8192 * sizeof(float));
            memcpy(lyr->conv_hist + 8192, raw_qkv, 2 * 8192 * sizeof(float));
        } else if (T == 1) {
            memmove(lyr->conv_hist, lyr->conv_hist + 8192, 2 * 8192 * sizeof(float));
            memcpy(lyr->conv_hist + 2 * 8192, raw_qkv, 8192 * sizeof(float));
        }
    }

    float *g_val    = (float *)malloc((size_t)T * 32 * sizeof(float));
    float *beta_val = (float *)malloc((size_t)T * 32 * sizeof(float));
    for (int t = 0; t < T; t++) {
        const float *ut = u + (size_t)t * C35_D;
        for (int h = 0; h < 32; h++) {
            const float *aw = lyr->a_w + (size_t)h * C35_D;
            const float *bw = lyr->b_w + (size_t)h * C35_D;
            float da = 0.0f, db = 0.0f;
            #pragma omp simd reduction(+:da,db)
            for (int d = 0; d < C35_D; d++) { da += ut[d] * aw[d]; db += ut[d] * bw[d]; }
            float sp = da + lyr->dt_bias[h];
            sp = (sp > 20.0f) ? sp : log1pf(expf(sp));
            g_val[t * 32 + h] = lyr->neg_exp_A[h] * sp;
            beta_val[t * 32 + h] = 1.0f / (1.0f + expf(-db));
        }
    }

    const float inv_sqrt128 = 1.0f / sqrtf(128.0f);
    #pragma omp parallel for schedule(static)
    for (int vh = 0; vh < 32; vh++) {
        int kh = vh / 2;
        if (!preview) {
            float local_st[128 * 128];
            float *st = (stateful && lyr->gdn_state) ? (lyr->gdn_state + (size_t)vh * 128 * 128) : local_st;
            if (!stateful || pos_off == 0) memset(st, 0, 128 * 128 * sizeof(float));

            for (int t = 0; t < T; t++) {
                const float *q_raw = mixed + (size_t)t * 8192 + kh * 128;
                const float *k_raw = mixed + (size_t)t * 8192 + 2048 + kh * 128;
                const float *v_raw = mixed + (size_t)t * 8192 + 4096 + vh * 128;
                float qf[128], kf[128];
                double sq = 0.0, sk = 0.0;
                for (int d = 0; d < 128; d++) { sq += (double)q_raw[d] * q_raw[d]; sk += (double)k_raw[d] * k_raw[d]; }
                float iq = (1.0f / sqrtf((float)sq + 1e-6f)) * inv_sqrt128;
                float ik = 1.0f / sqrtf((float)sk + 1e-6f);
                for (int d = 0; d < 128; d++) { qf[d] = q_raw[d] * iq; kf[d] = k_raw[d] * ik; }

                float dec = expf(g_val[t * 32 + vh]);
                float b_t = beta_val[t * 32 + vh];
                float *yt = y_attn + (size_t)t * 4096 + vh * 128;

                for (int vd = 0; vd < 128; vd++) {
                    float *srow = st + vd * 128;
                    float rem = 0.0f;
                    #pragma omp simd reduction(+:rem)
                    for (int kd = 0; kd < 128; kd++) {
                        srow[kd] *= dec;
                        rem += srow[kd] * kf[kd];
                    }
                    float delta = (v_raw[vd] - rem) * b_t;
                    float yv = 0.0f;
                    #pragma omp simd reduction(+:yv)
                    for (int kd = 0; kd < 128; kd++) {
                        srow[kd] += delta * kf[kd];
                        yv += srow[kd] * qf[kd];
                    }
                    yt[vd] = yv;
                }
            }
        } else {
            float *qn_all = (float *)malloc((size_t)T * 128 * sizeof(float));
            float *kn_all = (float *)malloc((size_t)T * 128 * sizeof(float));
            for (int t = 0; t < T; t++) {
                const float *q_raw = mixed + (size_t)t * 8192 + kh * 128;
                const float *k_raw = mixed + (size_t)t * 8192 + 2048 + kh * 128;
                double sq = 0.0, sk = 0.0;
                for (int d = 0; d < 128; d++) { sq += (double)q_raw[d] * q_raw[d]; sk += (double)k_raw[d] * k_raw[d]; }
                float iq = 1.0f / sqrtf((float)sq + 1e-6f);
                float ik = 1.0f / sqrtf((float)sk + 1e-6f);
                for (int d = 0; d < 128; d++) {
                    qn_all[t * 128 + d] = q_raw[d] * iq;
                    kn_all[t * 128 + d] = k_raw[d] * ik;
                }
            }
            float l_scale = lyr->ls[vh];
            float h_logit = lyr->hb[vh] + lyr->hc[vh] * logf((float)T / 32.0f);
            float *w_row = (float *)malloc((size_t)(T + 1) * sizeof(float));

            for (int t = 0; t < T; t++) {
                float mx = h_logit;
                for (int s = 0; s < T; s++) {
                    float dot = 0.0f;
                    for (int d = 0; d < 128; d++) dot += qn_all[t * 128 + d] * kn_all[s * 128 + d];
                    w_row[s] = dot * l_scale;
                    if (w_row[s] > mx) mx = w_row[s];
                }
                w_row[T] = h_logit;
                float sw = 0.0f;
                for (int s = 0; s <= T; s++) { w_row[s] = expf(w_row[s] - mx); sw += w_row[s]; }
                for (int s = 0; s < T; s++) w_row[s] /= sw;

                float *yt = y_attn + (size_t)t * 4096 + vh * 128;
                memset(yt, 0, 128 * sizeof(float));
                for (int s = 0; s < T; s++) {
                    const float *vs = mixed + (size_t)s * 8192 + 4096 + vh * 128;
                    float ws = w_row[s];
                    for (int d = 0; d < 128; d++) yt[d] += ws * vs[d];
                }
            }
            free(qn_all); free(kn_all); free(w_row);
        }

        for (int t = 0; t < T; t++) {
            float *yt = y_attn + (size_t)t * 4096 + vh * 128;
            const float *zt = z + (size_t)t * 4096 + vh * 128;
            double ss = 0.0;
            for (int d = 0; d < 128; d++) ss += (double)yt[d] * yt[d];
            float inv = 1.0f / sqrtf((float)(ss / 128.0) + 1e-6f);
            for (int d = 0; d < 128; d++) {
                float zg = zt[d] / (1.0f + expf(-zt[d]));
                yt[d] = (yt[d] * inv * lyr->gdn_norm[d]) * zg;
            }
        }
    }
    free(g_val); free(beta_val);

    c35_matmul(&lyr->out, y_attn, h_attn, T, 0, 1);
    for (int i = 0; i < T * C35_D; i++) h_attn[i] += x[i];

    rms_norm_batch(h_attn, lyr->post_ln, u_mlp, T, C35_D, 1e-6f);
    c35_run_moe(u_mlp, moe_out, T, &lyr->moe, &lyr->ex, preview);
    for (int i = 0; i < T * C35_D; i++) h_out[i] = h_attn[i] + moe_out[i];

    free(u); free(raw_qkv); free(mixed); free(z); free(y_attn); free(h_attn); free(u_mlp); free(moe_out);
}

/* Generic Layer Execution: Gated Full-Attention (`16Q/2KV`, `head_dim = 256`, Partial RoPE `0.25`) + SwiGLU MoE */
static void c35_forward_layer_gqa(c35_layer_t *lyr, const float *x, float *h_out, int pos_off, int T, int preview, int stateful) {
    float *u       = (float *)malloc((size_t)T * C35_D * sizeof(float));
    float *qg      = (float *)malloc((size_t)T * 8192 * sizeof(float));
    float *k       = (float *)malloc((size_t)T * 512 * sizeof(float));
    float *v       = (float *)malloc((size_t)T * 512 * sizeof(float));
    float *y_attn  = (float *)malloc((size_t)T * 4096 * sizeof(float));
    float *h_attn  = (float *)malloc((size_t)T * C35_D * sizeof(float));
    float *u_mlp   = (float *)malloc((size_t)T * C35_D * sizeof(float));
    float *moe_out = (float *)malloc((size_t)T * C35_D * sizeof(float));

    rms_norm_batch(x, lyr->in_ln, u, T, C35_D, 1e-6f);
    c35_matmul(&lyr->q, u, qg, T, 0, 1);
    c35_matmul(&lyr->k, u, k,  T, 0, 1);
    c35_matmul(&lyr->v, u, v,  T, 0, 1);

    #pragma omp parallel for schedule(static) if (T >= 4)
    for (int t = 0; t < T; t++) {
        int abs_pos = stateful ? (pos_off + t) : t;
        float cs[32], sn[32];
        for (int d = 0; d < 32; d++) {
            float ang = (float)abs_pos * g_c35.inv_freq[d];
            cs[d] = cosf(ang); sn[d] = sinf(ang);
        }
        for (int h = 0; h < 16; h++) {
            float *qh = qg + (size_t)t * 8192 + h * 512;
            rms_norm_row(qh, lyr->qn, qh, 256, 1e-6f);
            for (int d = 0; d < 32; d++) {
                float f1 = qh[d], f2 = qh[32 + d];
                qh[d]      = f1 * cs[d] - f2 * sn[d];
                qh[32 + d] = f2 * cs[d] + f1 * sn[d];
            }
        }
        for (int kh = 0; kh < 2; kh++) {
            float *kh_p = k + (size_t)t * 512 + kh * 256;
            rms_norm_row(kh_p, lyr->kn, kh_p, 256, 1e-6f);
            for (int d = 0; d < 32; d++) {
                float f1 = kh_p[d], f2 = kh_p[32 + d];
                kh_p[d]      = f1 * cs[d] - f2 * sn[d];
                kh_p[32 + d] = f2 * cs[d] + f1 * sn[d];
            }
        }
        if (stateful && !preview && lyr->k_cache_bf16) {
            uint16_t *kd = lyr->k_cache_bf16 + (size_t)abs_pos * 512;
            uint16_t *vd = lyr->v_cache_bf16 + (size_t)abs_pos * 512;
            for (int d = 0; d < 512; d++) {
                kd[d] = f32_to_bf16(k[(size_t)t * 512 + d]);
                vd[d] = f32_to_bf16(v[(size_t)t * 512 + d]);
            }
        }
    }

    const float scale = 1.0f / 16.0f;
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < 16; h++) {
        int kv_h = h / 8;
        if (stateful && !preview && lyr->k_cache_bf16) {
            for (int t = 0; t < T; t++) {
                int abs_pos = pos_off + t;
                const float *qh = qg + (size_t)t * 8192 + h * 512;
                const float *gh = qh + 256;
                float m_max = -1e30f, l_sum = 0.0f;
                float out_h[256] = {0};
                for (int j = 0; j <= abs_pos; j++) {
                    const uint16_t *kj = lyr->k_cache_bf16 + (size_t)j * 512 + kv_h * 256;
                    const uint16_t *vj = lyr->v_cache_bf16 + (size_t)j * 512 + kv_h * 256;
                    float dot = 0.0f;
                    for (int d = 0; d < 256; d++) dot += qh[d] * bf16_to_f32(kj[d]);
                    dot *= scale;
                    if (dot > m_max) {
                        float alpha = expf(m_max - dot);
                        l_sum = l_sum * alpha + 1.0f;
                        for (int d = 0; d < 256; d++) out_h[d] = out_h[d] * alpha + bf16_to_f32(vj[d]);
                        m_max = dot;
                    } else {
                        float w = expf(dot - m_max);
                        l_sum += w;
                        for (int d = 0; d < 256; d++) out_h[d] += w * bf16_to_f32(vj[d]);
                    }
                }
                float inv_l = 1.0f / (l_sum + 1e-12f);
                float *yh = y_attn + (size_t)t * 4096 + h * 256;
                for (int d = 0; d < 256; d++) yh[d] = (out_h[d] * inv_l) * (1.0f / (1.0f + expf(-gh[d])));
            }
        } else {
            float *scores = (float *)malloc((size_t)T * sizeof(float));
            for (int t = 0; t < T; t++) {
                int j_end = preview ? (T - 1) : t;
                const float *qh = qg + (size_t)t * 8192 + h * 512;
                const float *gh = qh + 256;
                float mx = -1e30f;
                for (int j = 0; j <= j_end; j++) {
                    const float *kj = k + (size_t)j * 512 + kv_h * 256;
                    float dot = 0.0f;
                    for (int d = 0; d < 256; d++) dot += qh[d] * kj[d];
                    scores[j] = dot * scale;
                    if (scores[j] > mx) mx = scores[j];
                }
                float sw = 0.0f;
                for (int j = 0; j <= j_end; j++) { scores[j] = expf(scores[j] - mx); sw += scores[j]; }
                for (int j = 0; j <= j_end; j++) scores[j] /= sw;

                float *yh = y_attn + (size_t)t * 4096 + h * 256;
                memset(yh, 0, 256 * sizeof(float));
                for (int j = 0; j <= j_end; j++) {
                    const float *vj = v + (size_t)j * 512 + kv_h * 256;
                    float sj = scores[j];
                    for (int d = 0; d < 256; d++) yh[d] += sj * vj[d];
                }
                for (int d = 0; d < 256; d++) yh[d] *= 1.0f / (1.0f + expf(-gh[d]));
            }
            free(scores);
        }
    }

    c35_matmul(&lyr->o, y_attn, h_attn, T, 0, 1);
    for (int i = 0; i < T * C35_D; i++) h_attn[i] += x[i];

    rms_norm_batch(h_attn, lyr->post_ln, u_mlp, T, C35_D, 1e-6f);
    c35_run_moe(u_mlp, moe_out, T, &lyr->moe, &lyr->ex, preview);
    for (int i = 0; i < T * C35_D; i++) h_out[i] = h_attn[i] + moe_out[i];

    free(u); free(qg); free(k); free(v); free(y_attn); free(h_attn); free(u_mlp); free(moe_out);
}

static void parse_bdgdn_layer_blob(const uint8_t *p, size_t off, c35_layer_t *lyr) {
    lyr->kind      = 0;
    lyr->in_ln     = (const float *)(p + off); off += C35_D * 4;
    lyr->post_ln   = (const float *)(p + off); off += C35_D * 4;
    lyr->conv_w    = (const float *)(p + off); off += 8192 * 4 * 4;
    lyr->conv_rw   = (const float *)(p + off); off += 8192 * 3 * 4;
    lyr->a_w       = (const float *)(p + off); off += 32 * C35_D * 4;
    lyr->b_w       = (const float *)(p + off); off += 32 * C35_D * 4;
    lyr->neg_exp_A = (const float *)(p + off); off += 32 * 4;
    lyr->dt_bias   = (const float *)(p + off); off += 32 * 4;
    lyr->gdn_norm  = (const float *)(p + off); off += 128 * 4;
    lyr->ls        = (const float *)(p + off); off += 32 * 4;
    lyr->hb        = (const float *)(p + off); off += 32 * 4;
    lyr->hc        = (const float *)(p + off); off += 32 * 4;
    uint32_t l_qkv, l_z, l_out, l_moe;
    memcpy(&l_qkv, p + off, 4); off += 4;
    memcpy(&l_z,   p + off, 4); off += 4;
    memcpy(&l_out, p + off, 4); off += 4;
    memcpy(&l_moe, p + off, 4); off += 4;
    parse_c35_mat(p + off, &lyr->qkv); off += l_qkv;
    parse_c35_mat(p + off, &lyr->z);   off += l_z;
    parse_c35_mat(p + off, &lyr->out); off += l_out;
    parse_c35_moe_tail(p + off, &lyr->moe);
}

static void parse_gqa_layer_blob(const uint8_t *p, size_t off, c35_layer_t *lyr) {
    lyr->kind    = 1;
    lyr->in_ln   = (const float *)(p + off); off += C35_D * 4;
    lyr->post_ln = (const float *)(p + off); off += C35_D * 4;
    lyr->qn      = (const float *)(p + off); off += 256 * 4;
    lyr->kn      = (const float *)(p + off); off += 256 * 4;
    uint32_t l_q, l_k, l_v, l_o, l_moe;
    memcpy(&l_q,   p + off, 4); off += 4;
    memcpy(&l_k,   p + off, 4); off += 4;
    memcpy(&l_v,   p + off, 4); off += 4;
    memcpy(&l_o,   p + off, 4); off += 4;
    memcpy(&l_moe, p + off, 4); off += 4;
    parse_c35_mat(p + off, &lyr->q); off += l_q;
    parse_c35_mat(p + off, &lyr->k); off += l_k;
    parse_c35_mat(p + off, &lyr->v); off += l_v;
    parse_c35_mat(p + off, &lyr->o); off += l_o;
    parse_c35_moe_tail(p + off, &lyr->moe);
}

C35_API void clover35b_hg4_cache_reset(void) {
    g_c35.cache_len = 0;
    for (int l = 0; l < g_c35.n_layers; l++) {
        if (g_c35.layers[l].gdn_state) memset(g_c35.layers[l].gdn_state, 0, 32 * 128 * 128 * sizeof(float));
        if (g_c35.layers[l].conv_hist) memset(g_c35.layers[l].conv_hist, 0, 3 * 8192 * sizeof(float));
    }
}

C35_API int clover35b_hg4_init(const char *hg4_dir) {
    if (g_c35.ready && strcmp(g_c35.loaded_dir, hg4_dir) == 0) return 0;
    if (g_c35.ready) {
        c35_mmap_close(&g_c35.emb_mf);
        c35_mmap_close(&g_c35.lm_mf);
        for (int l = 0; l < g_c35.n_layers; l++) {
            c35_mmap_close(&g_c35.layers[l].mf);
            c35_mmap_close(&g_c35.layers[l].ex.mf);
            free(g_c35.layers[l].gdn_state);    g_c35.layers[l].gdn_state = NULL;
            free(g_c35.layers[l].conv_hist);    g_c35.layers[l].conv_hist = NULL;
            free(g_c35.layers[l].k_cache_bf16); g_c35.layers[l].k_cache_bf16 = NULL;
            free(g_c35.layers[l].v_cache_bf16); g_c35.layers[l].v_cache_bf16 = NULL;
        }
        g_c35.ready = 0;
    }

    g_c35.isa_tier = detect_c35_isa();
    char path[1024];
    snprintf(path, sizeof(path), "%s/meta.bin", hg4_dir);
    FILE *mf = fopen(path, "rb");
    if (!mf) return -1;
    char magic[4];
    uint32_t hdr[3];
    if (fread(magic, 1, 4, mf) != 4 ||
        (memcmp(magic, "C35R", 4) != 0 && memcmp(magic, "C35M", 4) != 0) ||
        fread(hdr, 4, 3, mf) != 3 ||
        fread(g_c35.lut16_u8, 1, 16, mf) != 16 ||
        fread(g_c35.norm_w, 4, C35_D, mf) != C35_D ||
        fread(g_c35.inv_freq, 4, 32, mf) != 32) { fclose(mf); return -2; }
    fclose(mf);

    g_c35.is_16r = (memcmp(magic, "C35R", 4) == 0) ? 1 : 0;
    for (int i = 0; i < 16; i++) {
        int sv = (int)g_c35.lut16_u8[i] - 128;
        g_c35.lut16_s8[i] = (int8_t)sv;
        g_c35.lut16_s8_f[i] = (float)sv;
    }

    snprintf(path, sizeof(path), "%s/embed_tokens.hg4", hg4_dir);
    if (c35_mmap_open(path, &g_c35.emb_mf) != 0) return -3;
    parse_c35_mat((const uint8_t *)g_c35.emb_mf.ptr, &g_c35.emb_mat);

    snprintf(path, sizeof(path), "%s/lm_head.hg4", hg4_dir);
    if (c35_mmap_open(path, &g_c35.lm_mf) != 0) return -4;
    parse_c35_mat((const uint8_t *)g_c35.lm_mf.ptr, &g_c35.lm_mat);

    if (g_c35.is_16r) {
        g_c35.n_layers = (int)hdr[1];
        if (g_c35.n_layers <= 0 || g_c35.n_layers > C35_MAX_LAYERS) return -5;
        for (int l = 0; l < g_c35.n_layers; l++) {
            c35_layer_t *lyr = &g_c35.layers[l];
            snprintf(path, sizeof(path), "%s/layer_%02d.hg4", hg4_dir, l);
            if (c35_mmap_open(path, &lyr->mf) != 0) return -(100 + l);
            const uint8_t *p = (const uint8_t *)lyr->mf.ptr;
            uint32_t kind = 0;
            memcpy(&kind, p, 4);
            if (kind == 0) {
                parse_bdgdn_layer_blob(p, 4, lyr);
                lyr->gdn_state = (float *)calloc(32 * 128 * 128, sizeof(float));
                lyr->conv_hist = (float *)calloc(3 * 8192, sizeof(float));
            } else {
                parse_gqa_layer_blob(p, 4, lyr);
                lyr->k_cache_bf16 = (uint16_t *)calloc((size_t)C35_MAX_SEQ * 512, sizeof(uint16_t));
                lyr->v_cache_bf16 = (uint16_t *)calloc((size_t)C35_MAX_SEQ * 512, sizeof(uint16_t));
            }
            snprintf(path, sizeof(path), "%s/experts_%02d.hg4", hg4_dir, l);
            if (open_c35_experts(path, &lyr->ex) != 0) return -(200 + l);
        }
    } else {
        /* Legacy 2-layer precomputed C1 stage (`layer0_bdgdn.hg4` + `layer3_gqa.hg4`) */
        g_c35.n_layers = 2;
        snprintf(path, sizeof(path), "%s/layer0_bdgdn.hg4", hg4_dir);
        if (c35_mmap_open(path, &g_c35.layers[0].mf) != 0) return -5;
        parse_bdgdn_layer_blob((const uint8_t *)g_c35.layers[0].mf.ptr, 0, &g_c35.layers[0]);
        snprintf(path, sizeof(path), "%s/experts0.hg4", hg4_dir);
        if (open_c35_experts(path, &g_c35.layers[0].ex) != 0) return -6;

        snprintf(path, sizeof(path), "%s/layer3_gqa.hg4", hg4_dir);
        if (c35_mmap_open(path, &g_c35.layers[1].mf) != 0) return -7;
        parse_gqa_layer_blob((const uint8_t *)g_c35.layers[1].mf.ptr, 0, &g_c35.layers[1]);
        snprintf(path, sizeof(path), "%s/experts3.hg4", hg4_dir);
        if (open_c35_experts(path, &g_c35.layers[1].ex) != 0) return -8;
    }

    if (!g_c35.scr_xq) {
        g_c35.scr_xq      = (int8_t *)malloc((size_t)C35_PANEL_MAX * 4096);
        g_c35.scr_xsc     = (float *)malloc((size_t)C35_PANEL_MAX * 32 * sizeof(float));
        g_c35.scr_xsum    = (int32_t *)malloc((size_t)C35_PANEL_MAX * 32 * sizeof(int32_t));
        g_c35.scr_moe_hid = (float *)malloc((size_t)C35_TOP_K * C35_MOE_INT * sizeof(float));
        g_c35.scr_moe_out = (float *)malloc((size_t)C35_TOP_K * C35_D * sizeof(float));
    }
    snprintf(g_c35.loaded_dir, sizeof(g_c35.loaded_dir), "%s", hg4_dir);
    g_c35.cache_len = 0;
    g_c35.ready = 1;
    return 0;
}

C35_API float clover35b_hg4_forward_ids(const int *token_ids, int T, int preview, float *out_final_hidden) {
    if (!g_c35.ready || T <= 0) return -1.0f;
    double t0 = omp_get_wtime();
    float *cur = (float *)malloc((size_t)T * C35_D * sizeof(float));
    float *nxt = (float *)malloc((size_t)T * C35_D * sizeof(float));

    #pragma omp parallel for schedule(static) if (T >= 4)
    for (int t = 0; t < T; t++) {
        c35_embed_lookup_row(token_ids[t], cur + (size_t)t * C35_D);
    }
    for (int l = 0; l < g_c35.n_layers; l++) {
        c35_layer_t *lyr = &g_c35.layers[l];
        if (lyr->kind == 0) c35_forward_layer_bdgdn(lyr, cur, nxt, 0, T, preview, 0);
        else                c35_forward_layer_gqa(lyr, cur, nxt, 0, T, preview, 0);
        float *tmp = cur; cur = nxt; nxt = tmp;
    }
    rms_norm_batch(cur, g_c35.norm_w, out_final_hidden, T, C35_D, 1e-6f);

    free(cur); free(nxt);
    return (float)((omp_get_wtime() - t0) * 1000.0);
}

C35_API int clover35b_hg4_cache_feed(
    const int *token_ids, int n_tokens, float *out_last_hidden, float *out_last_logits
) {
    if (!g_c35.ready || n_tokens <= 0 || g_c35.cache_len + n_tokens > C35_MAX_SEQ) return -1;
    float h_last[C35_D];
    int done = 0;
    while (done < n_tokens) {
        int chunk = (n_tokens - done > C35_PANEL_MAX) ? C35_PANEL_MAX : (n_tokens - done);
        float *cur = (float *)malloc((size_t)chunk * C35_D * sizeof(float));
        float *nxt = (float *)malloc((size_t)chunk * C35_D * sizeof(float));
        #pragma omp parallel for schedule(static) if (chunk >= 4)
        for (int t = 0; t < chunk; t++) {
            c35_embed_lookup_row(token_ids[done + t], cur + (size_t)t * C35_D);
        }
        for (int l = 0; l < g_c35.n_layers; l++) {
            c35_layer_t *lyr = &g_c35.layers[l];
            if (lyr->kind == 0) c35_forward_layer_bdgdn(lyr, cur, nxt, g_c35.cache_len, chunk, 0, 1);
            else                c35_forward_layer_gqa(lyr, cur, nxt, g_c35.cache_len, chunk, 0, 1);
            float *tmp = cur; cur = nxt; nxt = tmp;
        }
        g_c35.cache_len += chunk;
        done += chunk;
        if (done == n_tokens) {
            rms_norm_row(cur + (size_t)(chunk - 1) * C35_D, g_c35.norm_w, h_last, C35_D, 1e-6f);
            if (out_last_hidden) memcpy(out_last_hidden, h_last, C35_D * sizeof(float));
            if (out_last_logits) c35_matmul(&g_c35.lm_mat, h_last, out_last_logits, 1, 0, 1);
        }
        free(cur); free(nxt);
    }
    return g_c35.cache_len;
}

C35_API int clover35b_hg4_generate(
    const int *prompt_ids, int prompt_len, int max_new_tokens, int eos_id,
    int *out_gen_ids, double *out_prefill_ms, double *out_decode_ms
) {
    if (!g_c35.ready || prompt_len <= 0 || prompt_len + max_new_tokens >= C35_MAX_SEQ) return -1;
    float *logits = (float *)malloc((size_t)C35_VOCAB * sizeof(float));
    clover35b_hg4_cache_reset();

    double t0 = omp_get_wtime();
    if (clover35b_hg4_cache_feed(prompt_ids, prompt_len, NULL, logits) < 0) {
        free(logits);
        return -1;
    }
    *out_prefill_ms = (omp_get_wtime() - t0) * 1000.0;

    double td0 = omp_get_wtime();
    int gen_n = 0;
    for (int step = 0; step < max_new_tokens; step++) {
        int best_id = 0;
        float best_v = logits[0];
        for (int v = 1; v < C35_VOCAB; v++) {
            if (logits[v] > best_v) { best_v = logits[v]; best_id = v; }
        }
        out_gen_ids[gen_n++] = best_id;
        if (best_id == eos_id || step + 1 >= max_new_tokens) break;
        clover35b_hg4_cache_feed(&best_id, 1, NULL, logits);
    }
    *out_decode_ms = (omp_get_wtime() - td0) * 1000.0;
    free(logits);
    return gen_n;
}

C35_API int clover35b_hg4_forward_logits(const int *token_ids, int seq_len, float *out_logits) {
    if (!g_c35.ready || seq_len <= 0 || seq_len > C35_MAX_SEQ) return -1;
    float *hid = (float *)malloc((size_t)seq_len * C35_D * sizeof(float));
    if (clover35b_hg4_forward_ids(token_ids, seq_len, 0, hid) < 0.0f) { free(hid); return -1; }
    c35_matmul(&g_c35.lm_mat, hid, out_logits, seq_len, 0, 1);
    free(hid);
    return 0;
}

C35_API float clover35b_hg4_all_logits(const float *hidden_t_2048, int T, float *out_logits_t_248k) {
    double t0 = omp_get_wtime();
    c35_matmul(&g_c35.lm_mat, hidden_t_2048, out_logits_t_248k, T, 0, 1);
    return (float)((omp_get_wtime() - t0) * 1000.0);
}

C35_API int clover35b_hg4_lm_head_top1(const float *h_last_2048, float *out_logits_248k) {
    float *lg = out_logits_248k ? out_logits_248k : (float *)malloc((size_t)C35_VOCAB * sizeof(float));
    c35_matmul(&g_c35.lm_mat, h_last_2048, lg, 1, 0, 1);
    int best_id = 0;
    float best_v = lg[0];
    for (int v = 1; v < C35_VOCAB; v++) {
        if (lg[v] > best_v) { best_v = lg[v]; best_id = v; }
    }
    if (!out_logits_248k) free(lg);
    return best_id;
}
