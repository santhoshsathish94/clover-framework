/*
 * Production Native C11 / OpenMP `HG4-128` Engine for `LiquidAI/LFM3-Omni-2B` (`853.88 MiB`)
 * Used across Clover AI Studio (Windows `.dll`, Linux `k3` `.so`, and Android `arm64-v8a` `.so`).
 *
 * Implements the Three-Way Optimal 4-Bit Arrangement (Faster + Smaller + More Accurate simultaneously):
 *   1. Column-Balanced Fast Walsh-Hadamard Incoherence (`FWHT-128`, < 0.5 us/token)
 *   2. 1-Cycle Register `LUT16` (`vpermb` / `vpshufb` / `vqtbl1q_s8`) 15-Point MSE Lloyd-Max Gaussian Codebook
 *   3. 4-Row Tile Interleaved SIMD Layout (`[R/4, K/128, 2, 4, 32]`) with:
 *      - Fused `[w1_r0, w1_r1, w3_r0, w3_r1]` SwiGLU register tile (computes SiLU(w1)*w3 before storing)
 *      - Fused `Q + K + V` (`[3072, 2048]`) single-rotation operator
 *      - Unified `128.91 MiB` Tied Embedding + LM-Head (1-row inverse FWHT-128 token lookup, 0 MB duplicate table)
 *   4. Single-binary runtime SIMD dispatch:
 *      - x86_64: `AVX-512 VNNI+VBMI` -> `AVX-VNNI` -> `AVX2+FMA` -> `Scalar`
 *      - ARM64 : `ARMv8.2-A NEON + DOTPROD (vqtbl1q_s8 + vdotq_s32)` -> `NEON` -> `Scalar`
 *   5. Stateful Prefix-Reusing BF16 KV + ShortConv Cache (`lfm3_hg4_cache_*`), Multimodal Continuous Embedding
 *      Prefill (`lfm3_hg4_cache_feed_embeddings`), and Full-Sequence C1 Hidden States (`lfm3_hg4_hidden_states`).
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
#define HG4_X86 1
#define HG4_ARM64 0
#define HG4_TGT_AVX2        __attribute__((target("avx2,fma,f16c")))
#define HG4_TGT_AVXVNNI     __attribute__((target("avx2,fma,f16c,avxvnni")))
#define HG4_TGT_AVX512VNNI  __attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,avx512vnni,avx512vbmi,fma,f16c")))
#elif defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#define HG4_X86 0
#define HG4_ARM64 1
#define HG4_TGT_AVX2
#define HG4_TGT_AVXVNNI
#define HG4_TGT_AVX512VNNI
#if defined(__ARM_FEATURE_DOTPROD)
#define HG4_HAS_ARM_DOTPROD 1
#else
#define HG4_HAS_ARM_DOTPROD 0
#endif
#else
#define HG4_X86 0
#define HG4_ARM64 0
#define HG4_TGT_AVX2
#define HG4_TGT_AVXVNNI
#define HG4_TGT_AVX512VNNI
#endif

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#define HG4_API __declspec(dllexport)
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#define HG4_API __attribute__((visibility("default")))
#endif
#include "../hg_cuda_hook.h"

#define HG4_LAYERS      24
#define HG4_D           2048
#define HG4_FFN         7680
#define HG4_VOCAB       128000
#define HG4_SLIDING_W   256
#define HG4_PANEL_MAX   512
#define HG4_DEFAULT_CTX 8192

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

typedef struct {
    void  *ptr;
    size_t size;
#ifdef _WIN32
    HANDLE hFile, hMap;
#else
    int fd;
#endif
} hg4_mmap_t;

static int hg4_mmap_open(const char *path, hg4_mmap_t *m) {
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

static void hg4_mmap_close(hg4_mmap_t *m) {
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
    uint32_t R, K;
    const float    *pre_diag;    /* [K] */
    const uint16_t *scales_fp16; /* HG4R: [R/16, K/128, 16]; HG4M: [R/4, K/128, 4] */
    const uint8_t  *tiles;       /* HG4R: [R/16, K/128, 16, 64]; HG4M: [R/4, K/128, 2, 4, 32] */
} hg4_mat_view_t;

static void parse_hg4_mat(const uint8_t *blob, hg4_mat_view_t *v) {
    memcpy(&v->R, blob + 0, 4);
    memcpy(&v->K, blob + 4, 4);
    size_t nb = v->K / 128;
    size_t off = 8;
    v->pre_diag = (const float *)(blob + off);
    off += (size_t)v->K * 4;
    v->scales_fp16 = (const uint16_t *)(blob + off);
    off += ((size_t)v->R * nb) * 2;
    v->tiles = blob + off;
}

typedef struct {
    hg4_mmap_t mf;
    uint32_t kind;
    const float *op_norm, *post_op_norm, *ffn_norm, *post_ffn_norm;
    const float *conv_w; /* [2048 * 7] when kind == 0 */
    hg4_mat_view_t in_proj, out_proj;
    const float *gate_w; /* [16 * 2048] when kind != 0 */
    const float *qn_w;   /* [128] */
    const float *kn_w;   /* [128] */
    hg4_mat_view_t qkv_proj, o_proj;
    hg4_mat_view_t w13_proj, w2_proj;
} hg4_layer_t;

static struct {
    int ready;
    int is_16r;     /* 1 = canonical HG4-16R ("HG4R"), 0 = legacy 4-row ("HG4M") */
    int isa_tier;   /* 0=Scalar, 1=AVX2, 2=AVX-VNNI, 3=AVX512-VNNI+VBMI, 4=ARM64-NEON-DOTPROD, 5=ARM64-NEON */
    int num_threads;
    int max_seq;
    int cache_len;
    char loaded_dir[512];

    uint8_t lut16_u8[16];
    int8_t  lut16_s8[16];
    float   lut16_s8_f[16];
    float   final_norm[HG4_D];
    float   inv_freq_full[64];
    float   inv_freq_sliding[64];
    uint32_t kinds[HG4_LAYERS];
    hg4_mmap_t emb_mf;
    hg4_mat_view_t emb_mat;
    hg4_layer_t layers[HG4_LAYERS];

    /* Compact BF16 KV + ShortConv caches for O(1) prefix truncation up to `max_seq` */
    uint16_t *k_cache_bf16[HG4_LAYERS];    /* [max_seq * 512] */
    uint16_t *v_cache_bf16[HG4_LAYERS];    /* [max_seq * 512] */
    uint16_t *conv_cache_bf16[HG4_LAYERS]; /* [max_seq * 2048] */

    /* Fixed bounded panel scratch (`HG4_PANEL_MAX = 512` tokens -> ~36 MiB total, independent of max_seq!) */
    float   *scr_a, *scr_b, *scr_op_in, *scr_op_out, *scr_norm, *scr_mid;
    float   *scr_bcx, *scr_qkv, *scr_attn, *scr_ffn_mid, *scr_gates;
    int8_t  *scr_xq;
    float   *scr_xsc;
    int32_t *scr_xsum;
} g_hg4;

static int detect_hg4_isa(void) {
#if HG4_X86
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
        __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512vnni") &&
        __builtin_cpu_supports("avx512vbmi") && __builtin_cpu_supports("f16c")) {
        return 3;
    }
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
    int has_vnni = 0;
    if (__get_cpuid_max(0, NULL) >= 7) {
        __cpuid_count(7, 1, eax, ebx, ecx, edx);
        has_vnni = (eax & (1u << 4)) != 0;
    }
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c")) {
        return has_vnni ? 2 : 1;
    }
#elif HG4_ARM64
#if HG4_HAS_ARM_DOTPROD
    return 4;
#else
    return 5;
#endif
#endif
    return 0;
}

HG4_API const char *lfm3_hg4_isa_name(void) {
    int t = g_hg4.ready ? g_hg4.isa_tier : detect_hg4_isa();
    if (t == 3) return "AVX-512 VNNI+VBMI (HG4-128/16R 16-Row + Bit-Exact Int32-FWHT + vpermb LUT16)";
    if (t == 2) return "AVX-VNNI+FMA (HG4-128/16R 16-Row + Bit-Exact Int32-FWHT + vpshufb LUT16)";
    if (t == 1) return "AVX2+FMA (HG4-128/16R 16-Row + Bit-Exact Int32-FWHT + abs/sign LUT16)";
    if (t == 4) return "ARM64 NEON+DOTPROD (HG4-128/16R 16-Row + Bit-Exact Int32-FWHT + vqtbl1q LUT16)";
    if (t == 5) return "ARM64 NEON (HG4-128/16R 16-Row + Bit-Exact Int32-FWHT)";
    return "Scalar C11 (HG4-128/16R + Bit-Exact Int32-FWHT)";
}

HG4_API void lfm3_hg4_set_isa_tier(int tier) {
    int max_hw = detect_hg4_isa();
    if (tier < 0) g_hg4.isa_tier = max_hw;
    else if (tier == 0) g_hg4.isa_tier = 0;
#if HG4_X86
    else if (tier == 1 && max_hw >= 1) g_hg4.isa_tier = 1;
    else if (tier == 2 && max_hw >= 2) {
        unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
        if (__get_cpuid_max(0, NULL) >= 7) __cpuid_count(7, 1, eax, ebx, ecx, edx);
        g_hg4.isa_tier = ((eax & (1u << 4)) != 0) ? 2 : 1;
    }
    else if (tier == 3 && max_hw == 3) g_hg4.isa_tier = 3;
#elif HG4_ARM64
    else if (tier == 4 || tier == 5) g_hg4.isa_tier = max_hw;
#endif
}

HG4_API void lfm3_hg4_set_num_threads(int n) {
    if (n > 0) {
        g_hg4.num_threads = n;
        omp_set_num_threads(n);
    }
}

HG4_API int lfm3_hg4_get_num_threads(void) {
    return g_hg4.num_threads > 0 ? g_hg4.num_threads : omp_get_max_threads();
}

static void fwht128_1block_scalar(float *v) {
    const float inv_sqrt128 = 0.08838834764831845f; /* 1 / sqrt(128) */
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
    int8_t *out_xq, float *out_xsc, int32_t *out_xsum
) {
    const int nb = K / 128;
    const float scale_c = 0.08838834764831845f / 127.0f;
    #pragma omp parallel for schedule(static) if (T >= 4)
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
                xst[b] = 0.0f;
                if (xmt) xmt[b] = 0;
                memset(qd, 0, 128);
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
                        rot[j + k] = a + bv;
                        rot[j + h + k] = a - bv;
                    }
                }
            }
            int32_t max_rot = 0;
            for (int i = 0; i < 128; i++) {
                int32_t ar = rot[i] < 0 ? -rot[i] : rot[i];
                if (ar > max_rot) max_rot = ar;
            }
            if (max_rot == 0) {
                xst[b] = 0.0f;
                if (xmt) xmt[b] = 0;
                memset(qd, 0, 128);
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
                if (xmt) xmt[b] = ssum;
            }
        }
    }
}

#if HG4_X86
HG4_TGT_AVX2 static void rotate_and_quantize_128_avx2(
    const float *x, const float *pre_diag, int T, int K,
    int8_t *out_xq, float *out_xsc, int32_t *out_xsum
) {
    const int nb = K / 128;
    const float scale_c = 0.08838834764831845f / 127.0f;
    const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    const __m256 vmin_in = _mm256_set1_ps(-2048.0f), vmax_in = _mm256_set1_ps(2048.0f);
    const __m256 vmin_c  = _mm256_set1_ps(-127.0f),  vmax_c  = _mm256_set1_ps(127.0f);
    const __m256i bias80 = _mm256_set1_epi8((char)0x80);
    const __m256i zero   = _mm256_setzero_si256();

    #pragma omp parallel for schedule(static) if (T >= 4)
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
                xst[b] = 0.0f; xmt[b] = 0;
                memset(qd, 0, 128);
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
            for (int i = 0; i < 128; i += 8) {
                vimax = _mm256_max_epi32(vimax, _mm256_abs_epi32(_mm256_loadu_si256((const __m256i *)(rot + i))));
            }
            __m128i mi = _mm_max_epi32(_mm256_castsi256_si128(vimax), _mm256_extracti128_si256(vimax, 1));
            mi = _mm_max_epi32(mi, _mm_shuffle_epi32(mi, 0x4E));
            mi = _mm_max_epi32(mi, _mm_shuffle_epi32(mi, 0xB1));
            int32_t max_rot = _mm_cvtsi128_si32(mi);

            if (max_rot == 0) {
                xst[b] = 0.0f; xmt[b] = 0;
                memset(qd, 0, 128);
            } else {
                xst[b] = (in_sc * (float)max_rot) * scale_c;
                __m256 vinv = _mm256_set1_ps(127.0f / (float)max_rot);
                int64_t tot_u8 = 0;
                for (int half = 0; half < 2; half++) {
                    const int32_t *rh = rot + half * 64;
                    #define Q8_BLK(off) _mm256_cvtps_epi32(_mm256_min_ps(vmax_c, _mm256_max_ps(vmin_c, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i *)(rh + (off)))), vinv))))
                    __m256i i0 = Q8_BLK(0),  i1 = Q8_BLK(8),  i2 = Q8_BLK(16), i3 = Q8_BLK(24);
                    __m256i i4 = Q8_BLK(32), i5 = Q8_BLK(40), i6 = Q8_BLK(48), i7 = Q8_BLK(56);
                    #undef Q8_BLK
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

HG4_TGT_AVX512VNNI static inline __m128i hsum4_zmm_epi32(__m512i z0, __m512i z1, __m512i z2, __m512i z3) {
    __m256i y0 = _mm256_add_epi32(_mm512_castsi512_si256(z0), _mm512_extracti32x8_epi32(z0, 1));
    __m256i y1 = _mm256_add_epi32(_mm512_castsi512_si256(z1), _mm512_extracti32x8_epi32(z1, 1));
    __m256i y2 = _mm256_add_epi32(_mm512_castsi512_si256(z2), _mm512_extracti32x8_epi32(z2, 1));
    __m256i y3 = _mm256_add_epi32(_mm512_castsi512_si256(z3), _mm512_extracti32x8_epi32(z3, 1));
    __m256i h01 = _mm256_hadd_epi32(y0, y1);
    __m256i h23 = _mm256_hadd_epi32(y2, y3);
    __m256i h0123 = _mm256_hadd_epi32(h01, h23);
    return _mm_add_epi32(_mm256_castsi256_si128(h0123), _mm256_extracti128_si256(h0123, 1));
}

HG4_TGT_AVX512VNNI static void hg4_tile4_avx512(
    const hg4_mat_view_t *M, const uint8_t *lut16_u8,
    const int8_t *X_q, const float *X_sc, const int32_t *X_sum,
    float *Y, int T, int tile_idx, int fused_swiglu
) {
    const int R = (int)M->R, K = (int)M->K, nb = K / 128;
    const int out_R = fused_swiglu ? (R / 2) : R;
    const int o_base = tile_idx * 4;
    const uint8_t *wt = M->tiles + (size_t)tile_idx * (size_t)nb * 256;
    const uint16_t *wsc = M->scales_fp16 + (size_t)tile_idx * (size_t)nb * 4;
    const __m512i vlut = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)lut16_u8));
    const __m256i m0f  = _mm256_set1_epi8(0x0F);

    if (T == 1) {
        __m128 vs = _mm_setzero_ps();
        for (int b = 0; b < nb; b++) {
            __m512i a0 = _mm512_setzero_si512(), a1 = _mm512_setzero_si512();
            __m512i a2 = _mm512_setzero_si512(), a3 = _mm512_setzero_si512();
            const uint8_t *wb = wt + b * 256;
            for (int u = 0; u < 2; u++) {
                __m512i xv = _mm512_loadu_si512((const __m512i *)(X_q + b * 128 + u * 64));
                const uint8_t *wu = wb + u * 128;
                #define ONE_DEC_R(ri, acc) do { \
                    __m256i pk = _mm256_loadu_si256((const __m256i *)(wu + (ri) * 32)); \
                    __m512i nib = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_and_si256(pk, m0f)), \
                                                     _mm256_and_si256(_mm256_srli_epi16(pk, 4), m0f), 1); \
                    acc = _mm512_dpbusd_epi32(acc, _mm512_permutexvar_epi8(nib, vlut), xv); \
                } while (0)
                ONE_DEC_R(0, a0); ONE_DEC_R(1, a1); ONE_DEC_R(2, a2); ONE_DEC_R(3, a3);
                #undef ONE_DEC_R
            }
            __m128i sums4 = _mm_sub_epi32(hsum4_zmm_epi32(a0, a1, a2, a3), _mm_set1_epi32(128 * X_sum[b]));
            __m128 vws = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)(wsc + b * 4)));
            vs = _mm_add_ps(vs, _mm_mul_ps(_mm_cvtepi32_ps(sums4), _mm_mul_ps(vws, _mm_set1_ps(X_sc[b]))));
        }
        if (fused_swiglu) {
            float s[4];
            _mm_storeu_ps(s, vs);
            Y[tile_idx * 2 + 0] = (s[0] / (1.0f + expf(-s[0]))) * s[2];
            Y[tile_idx * 2 + 1] = (s[1] / (1.0f + expf(-s[1]))) * s[3];
        } else {
            _mm_storeu_ps(Y + o_base, vs);
        }
        return;
    }

    for (int t0 = 0; t0 < T; t0 += 32) {
        int tblk = (t0 + 32 <= T) ? 32 : (T - t0);
        __m128 vs[32];
        for (int ti = 0; ti < tblk; ti++) vs[ti] = _mm_setzero_ps();

        for (int b = 0; b < nb; b++) {
            const uint8_t *wb = wt + b * 256;
            __m512i wu8[2][4];
            for (int u = 0; u < 2; u++) {
                const uint8_t *wu = wb + u * 128;
                for (int ri = 0; ri < 4; ri++) {
                    __m256i pk = _mm256_loadu_si256((const __m256i *)(wu + ri * 32));
                    __m512i nib = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_and_si256(pk, m0f)),
                                                      _mm256_and_si256(_mm256_srli_epi16(pk, 4), m0f), 1);
                    wu8[u][ri] = _mm512_permutexvar_epi8(nib, vlut);
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
                    vs[ti + m] = _mm_add_ps(vs[ti + m], _mm_mul_ps(_mm_cvtepi32_ps(sums4), _mm_mul_ps(vws, _mm_set1_ps(X_sc[t_abs * nb + b]))));
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

HG4_TGT_AVXVNNI static void hg4_tile4_avxvnni(
    const hg4_mat_view_t *M, const uint8_t *lut16_u8,
    const int8_t *X_q, const float *X_sc, const int32_t *X_sum,
    float *Y, int T, int tile_idx, int fused_swiglu
) {
    const int R = (int)M->R, K = (int)M->K, nb = K / 128;
    const int out_R = fused_swiglu ? (R / 2) : R;
    const int o_base = tile_idx * 4;
    const uint8_t *wt = M->tiles + (size_t)tile_idx * (size_t)nb * 256;
    const uint16_t *wsc = M->scales_fp16 + (size_t)tile_idx * (size_t)nb * 4;
    const __m256i vlut = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)lut16_u8));
    const __m256i m0f  = _mm256_set1_epi8(0x0F);

    for (int t0 = 0; t0 < T; t0 += 16) {
        int tblk = (t0 + 16 <= T) ? 16 : (T - t0);
        __m128 vs[16];
        for (int ti = 0; ti < tblk; ti++) vs[ti] = _mm_setzero_ps();
        for (int b = 0; b < nb; b++) {
            const uint8_t *wb = wt + b * 256;
            __m128 vws = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)(wsc + b * 4)));
            __m256i wlo[2][4], whi[2][4];
            for (int u = 0; u < 2; u++) {
                const uint8_t *wu = wb + u * 128;
                for (int ri = 0; ri < 4; ri++) {
                    __m256i pk = _mm256_loadu_si256((const __m256i *)(wu + ri * 32));
                    wlo[u][ri] = _mm256_shuffle_epi8(vlut, _mm256_and_si256(pk, m0f));
                    whi[u][ri] = _mm256_shuffle_epi8(vlut, _mm256_and_si256(_mm256_srli_epi16(pk, 4), m0f));
                }
            }
            for (int ti = 0; ti < tblk; ti++) {
                int t_abs = t0 + ti;
                __m256i acc[4] = {_mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256()};
                const int8_t *xq = X_q + (size_t)t_abs * K + b * 128;
                for (int u = 0; u < 2; u++) {
                    __m256i xlo = _mm256_loadu_si256((const __m256i *)(xq + u * 64 + 0));
                    __m256i xhi = _mm256_loadu_si256((const __m256i *)(xq + u * 64 + 32));
                    for (int ri = 0; ri < 4; ri++) {
                        acc[ri] = _mm256_dpbusd_avx_epi32(acc[ri], wlo[u][ri], xlo);
                        acc[ri] = _mm256_dpbusd_avx_epi32(acc[ri], whi[u][ri], xhi);
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

HG4_TGT_AVX2 static void hg4_tile4_avx2(
    const hg4_mat_view_t *M, const int8_t *lut16_s8,
    const int8_t *X_q, const float *X_sc,
    float *Y, int T, int tile_idx, int fused_swiglu
) {
    const int R = (int)M->R, K = (int)M->K, nb = K / 128;
    const int out_R = fused_swiglu ? (R / 2) : R;
    const int o_base = tile_idx * 4;
    const uint8_t *wt = M->tiles + (size_t)tile_idx * (size_t)nb * 256;
    const uint16_t *wsc = M->scales_fp16 + (size_t)tile_idx * (size_t)nb * 4;
    const __m256i vlut = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)lut16_s8));
    const __m256i m0f  = _mm256_set1_epi8(0x0F);
    const __m256i ones16 = _mm256_set1_epi16(1);

    for (int t0 = 0; t0 < T; t0 += 16) {
        int tblk = (t0 + 16 <= T) ? 16 : (T - t0);
        __m128 vs[16];
        for (int ti = 0; ti < tblk; ti++) vs[ti] = _mm_setzero_ps();
        for (int b = 0; b < nb; b++) {
            const uint8_t *wb = wt + b * 256;
            __m128 vws = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)(wsc + b * 4)));
            __m256i wlo[2][4], whi[2][4];
            for (int u = 0; u < 2; u++) {
                const uint8_t *wu = wb + u * 128;
                for (int ri = 0; ri < 4; ri++) {
                    __m256i pk = _mm256_loadu_si256((const __m256i *)(wu + ri * 32));
                    wlo[u][ri] = _mm256_shuffle_epi8(vlut, _mm256_and_si256(pk, m0f));
                    whi[u][ri] = _mm256_shuffle_epi8(vlut, _mm256_and_si256(_mm256_srli_epi16(pk, 4), m0f));
                }
            }
            for (int ti = 0; ti < tblk; ti++) {
                int t_abs = t0 + ti;
                __m256i acc[4] = {_mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256()};
                const int8_t *xq = X_q + (size_t)t_abs * K + b * 128;
                for (int u = 0; u < 2; u++) {
                    __m256i xlo = _mm256_loadu_si256((const __m256i *)(xq + u * 64 + 0));
                    __m256i xhi = _mm256_loadu_si256((const __m256i *)(xq + u * 64 + 32));
                    for (int ri = 0; ri < 4; ri++) {
                        __m256i p0 = _mm256_maddubs_epi16(_mm256_abs_epi8(wlo[u][ri]), _mm256_sign_epi8(xlo, wlo[u][ri]));
                        __m256i p1 = _mm256_maddubs_epi16(_mm256_abs_epi8(whi[u][ri]), _mm256_sign_epi8(xhi, whi[u][ri]));
                        acc[ri] = _mm256_add_epi32(acc[ri], _mm256_add_epi32(_mm256_madd_epi16(p0, ones16),
                                                                             _mm256_madd_epi16(p1, ones16)));
                    }
                }
                __m256i h01 = _mm256_hadd_epi32(acc[0], acc[1]);
                __m256i h23 = _mm256_hadd_epi32(acc[2], acc[3]);
                __m256i h0123 = _mm256_hadd_epi32(h01, h23);
                __m128i sums4 = _mm_add_epi32(_mm256_castsi256_si128(h0123), _mm256_extracti128_si256(h0123, 1));
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

/* =========================================================================
 * Canonical `HG4-16R` Kernels (16-Row Lane-Interleaved, Zero Horizontal Sum)
 * Memory layout per [16_rows, 128_cols] block (1024 bytes):
 *   16 steps (s = 0..15), each step = 64 bytes = 16 lanes (r = 0..15) x 4 bytes:
 *     low nibbles  (& 0x0F) = cols 8*s + 0..3 of row r
 *     high nibbles (>> 4)   = cols 8*s + 4..7 of row r
 *   For fused_swiglu = 1: lanes 0..7 = w1[0..7], lanes 8..15 = w3[0..7].
 * ========================================================================= */
HG4_TGT_AVX512VNNI static void hg4_group16_avx512(
    const hg4_mat_view_t *M, const uint8_t *lut16_u8,
    const int8_t *X_q, const float *X_sc, const int32_t *X_sum,
    float *Y, int T, int rg, int fused_swiglu
) {
    const int R = (int)M->R, K = (int)M->K, nb = K / 128;
    const int out_R = fused_swiglu ? (R / 2) : R;
    const uint8_t *wt = M->tiles + (size_t)rg * (size_t)nb * 1024;
    const uint16_t *wsc = M->scales_fp16 + (size_t)rg * (size_t)nb * 16;
    const __m512i vlut = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)lut16_u8));
    const __m512i m0f  = _mm512_set1_epi8(0x0F);

    if (T == 1) {
        __m512 vs = _mm512_setzero_ps();
        for (int b = 0; b < nb; b++) {
            __m512i acc = _mm512_setzero_si512();
            const uint8_t *wb = wt + (size_t)b * 1024;
            const int32_t *x32 = (const int32_t *)(X_q + b * 128);
            for (int s = 0; s < 16; s++) {
                __m512i pk  = _mm512_loadu_si512((const __m512i *)(wb + s * 64));
                __m512i wlo = _mm512_permutexvar_epi8(_mm512_and_si512(pk, m0f), vlut);
                __m512i whi = _mm512_permutexvar_epi8(_mm512_and_si512(_mm512_srli_epi16(pk, 4), m0f), vlut);
                acc = _mm512_dpbusd_epi32(acc, wlo, _mm512_set1_epi32(x32[2 * s + 0]));
                acc = _mm512_dpbusd_epi32(acc, whi, _mm512_set1_epi32(x32[2 * s + 1]));
            }
            acc = _mm512_sub_epi32(acc, _mm512_set1_epi32(128 * X_sum[b]));
            __m512 vws = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(wsc + b * 16)));
            vs = _mm512_add_ps(vs, _mm512_mul_ps(_mm512_cvtepi32_ps(acc), _mm512_mul_ps(vws, _mm512_set1_ps(X_sc[b]))));
        }
        if (fused_swiglu) {
            float sv[16];
            _mm512_storeu_ps(sv, vs);
            for (int j = 0; j < 8; j++) Y[rg * 8 + j] = (sv[j] / (1.0f + expf(-sv[j]))) * sv[j + 8];
        } else {
            _mm512_storeu_ps(Y + rg * 16, vs);
        }
        return;
    }

    for (int t0 = 0; t0 < T; t0 += 8) {
        int tblk = (t0 + 8 <= T) ? 8 : (T - t0);
        __m512 vs[8];
        for (int ti = 0; ti < tblk; ti++) vs[ti] = _mm512_setzero_ps();

        for (int b = 0; b < nb; b++) {
            const uint8_t *wb = wt + (size_t)b * 1024;
            __m512 vws = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(wsc + b * 16)));
            if (tblk == 8) {
                __m512i a0 = _mm512_setzero_si512(), a1 = _mm512_setzero_si512();
                __m512i a2 = _mm512_setzero_si512(), a3 = _mm512_setzero_si512();
                __m512i a4 = _mm512_setzero_si512(), a5 = _mm512_setzero_si512();
                __m512i a6 = _mm512_setzero_si512(), a7 = _mm512_setzero_si512();
                const int32_t *x0 = (const int32_t *)(X_q + (size_t)(t0 + 0) * K + b * 128);
                const int32_t *x1 = (const int32_t *)(X_q + (size_t)(t0 + 1) * K + b * 128);
                const int32_t *x2 = (const int32_t *)(X_q + (size_t)(t0 + 2) * K + b * 128);
                const int32_t *x3 = (const int32_t *)(X_q + (size_t)(t0 + 3) * K + b * 128);
                const int32_t *x4 = (const int32_t *)(X_q + (size_t)(t0 + 4) * K + b * 128);
                const int32_t *x5 = (const int32_t *)(X_q + (size_t)(t0 + 5) * K + b * 128);
                const int32_t *x6 = (const int32_t *)(X_q + (size_t)(t0 + 6) * K + b * 128);
                const int32_t *x7 = (const int32_t *)(X_q + (size_t)(t0 + 7) * K + b * 128);
                for (int s = 0; s < 16; s++) {
                    __m512i pk  = _mm512_loadu_si512((const __m512i *)(wb + s * 64));
                    __m512i wlo = _mm512_permutexvar_epi8(_mm512_and_si512(pk, m0f), vlut);
                    __m512i whi = _mm512_permutexvar_epi8(_mm512_and_si512(_mm512_srli_epi16(pk, 4), m0f), vlut);
                    #define ACC_TOK(ac, xp) do { \
                        ac = _mm512_dpbusd_epi32(ac, wlo, _mm512_set1_epi32((xp)[2 * s + 0])); \
                        ac = _mm512_dpbusd_epi32(ac, whi, _mm512_set1_epi32((xp)[2 * s + 1])); \
                    } while (0)
                    ACC_TOK(a0, x0); ACC_TOK(a1, x1); ACC_TOK(a2, x2); ACC_TOK(a3, x3);
                    ACC_TOK(a4, x4); ACC_TOK(a5, x5); ACC_TOK(a6, x6); ACC_TOK(a7, x7);
                    #undef ACC_TOK
                }
                #define FIN_TOK(ti, ac) do { \
                    int ta = t0 + (ti); \
                    __m512i d32 = _mm512_sub_epi32(ac, _mm512_set1_epi32(128 * X_sum[ta * nb + b])); \
                    vs[ti] = _mm512_add_ps(vs[ti], _mm512_mul_ps(_mm512_cvtepi32_ps(d32), _mm512_mul_ps(vws, _mm512_set1_ps(X_sc[ta * nb + b])))); \
                } while (0)
                FIN_TOK(0, a0); FIN_TOK(1, a1); FIN_TOK(2, a2); FIN_TOK(3, a3);
                FIN_TOK(4, a4); FIN_TOK(5, a5); FIN_TOK(6, a6); FIN_TOK(7, a7);
                #undef FIN_TOK
            } else {
                __m512i acc[8];
                for (int ti = 0; ti < tblk; ti++) acc[ti] = _mm512_setzero_si512();
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
                for (int ti = 0; ti < tblk; ti++) {
                    int ta = t0 + ti;
                    __m512i d32 = _mm512_sub_epi32(acc[ti], _mm512_set1_epi32(128 * X_sum[ta * nb + b]));
                    vs[ti] = _mm512_add_ps(vs[ti], _mm512_mul_ps(_mm512_cvtepi32_ps(d32), _mm512_mul_ps(vws, _mm512_set1_ps(X_sc[ta * nb + b]))));
                }
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

HG4_TGT_AVXVNNI static void hg4_group16_avxvnni(
    const hg4_mat_view_t *M, const uint8_t *lut16_u8,
    const int8_t *X_q, const float *X_sc, const int32_t *X_sum,
    float *Y, int T, int rg, int fused_swiglu
) {
    const int R = (int)M->R, K = (int)M->K, nb = K / 128;
    const int out_R = fused_swiglu ? (R / 2) : R;
    const uint8_t *wt = M->tiles + (size_t)rg * (size_t)nb * 1024;
    const uint16_t *wsc = M->scales_fp16 + (size_t)rg * (size_t)nb * 16;
    const __m256i vlut = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)lut16_u8));
    const __m256i m0f  = _mm256_set1_epi8(0x0F);

    for (int t0 = 0; t0 < T; t0 += 4) {
        int tblk = (t0 + 4 <= T) ? 4 : (T - t0);
        __m256 vs0[4], vs1[4];
        for (int ti = 0; ti < tblk; ti++) { vs0[ti] = _mm256_setzero_ps(); vs1[ti] = _mm256_setzero_ps(); }

        for (int b = 0; b < nb; b++) {
            const uint8_t *wb = wt + (size_t)b * 1024;
            __m256 vws0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(wsc + b * 16 + 0)));
            __m256 vws1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(wsc + b * 16 + 8)));
            __m256i a0[4], a1[4];
            for (int ti = 0; ti < tblk; ti++) { a0[ti] = _mm256_setzero_si256(); a1[ti] = _mm256_setzero_si256(); }

            for (int s = 0; s < 16; s++) {
                __m256i pk0 = _mm256_loadu_si256((const __m256i *)(wb + s * 64 + 0));
                __m256i pk1 = _mm256_loadu_si256((const __m256i *)(wb + s * 64 + 32));
                __m256i w0_lo = _mm256_shuffle_epi8(vlut, _mm256_and_si256(pk0, m0f));
                __m256i w0_hi = _mm256_shuffle_epi8(vlut, _mm256_and_si256(_mm256_srli_epi16(pk0, 4), m0f));
                __m256i w1_lo = _mm256_shuffle_epi8(vlut, _mm256_and_si256(pk1, m0f));
                __m256i w1_hi = _mm256_shuffle_epi8(vlut, _mm256_and_si256(_mm256_srli_epi16(pk1, 4), m0f));
                for (int ti = 0; ti < tblk; ti++) {
                    const int32_t *xp = (const int32_t *)(X_q + (size_t)(t0 + ti) * K + b * 128);
                    __m256i xb0 = _mm256_set1_epi32(xp[2 * s + 0]);
                    __m256i xb1 = _mm256_set1_epi32(xp[2 * s + 1]);
                    a0[ti] = _mm256_dpbusd_avx_epi32(a0[ti], w0_lo, xb0);
                    a0[ti] = _mm256_dpbusd_avx_epi32(a0[ti], w0_hi, xb1);
                    a1[ti] = _mm256_dpbusd_avx_epi32(a1[ti], w1_lo, xb0);
                    a1[ti] = _mm256_dpbusd_avx_epi32(a1[ti], w1_hi, xb1);
                }
            }
            for (int ti = 0; ti < tblk; ti++) {
                int ta = t0 + ti;
                __m256i sub = _mm256_set1_epi32(128 * X_sum[ta * nb + b]);
                __m256 vxsc = _mm256_set1_ps(X_sc[ta * nb + b]);
                vs0[ti] = _mm256_add_ps(vs0[ti], _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(a0[ti], sub)), _mm256_mul_ps(vws0, vxsc)));
                vs1[ti] = _mm256_add_ps(vs1[ti], _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(a1[ti], sub)), _mm256_mul_ps(vws1, vxsc)));
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

HG4_TGT_AVX2 static void hg4_group16_avx2(
    const hg4_mat_view_t *M, const int8_t *lut16_s8,
    const int8_t *X_q, const float *X_sc,
    float *Y, int T, int rg, int fused_swiglu
) {
    const int R = (int)M->R, K = (int)M->K, nb = K / 128;
    const int out_R = fused_swiglu ? (R / 2) : R;
    const uint8_t *wt = M->tiles + (size_t)rg * (size_t)nb * 1024;
    const uint16_t *wsc = M->scales_fp16 + (size_t)rg * (size_t)nb * 16;
    const __m256i vlut   = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)lut16_s8));
    const __m256i m0f    = _mm256_set1_epi8(0x0F);
    const __m256i ones16 = _mm256_set1_epi16(1);

    for (int t0 = 0; t0 < T; t0 += 4) {
        int tblk = (t0 + 4 <= T) ? 4 : (T - t0);
        __m256 vs0[4], vs1[4];
        for (int ti = 0; ti < tblk; ti++) { vs0[ti] = _mm256_setzero_ps(); vs1[ti] = _mm256_setzero_ps(); }

        for (int b = 0; b < nb; b++) {
            const uint8_t *wb = wt + (size_t)b * 1024;
            __m256 vws0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(wsc + b * 16 + 0)));
            __m256 vws1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(wsc + b * 16 + 8)));
            __m256i a0[4], a1[4];
            for (int ti = 0; ti < tblk; ti++) { a0[ti] = _mm256_setzero_si256(); a1[ti] = _mm256_setzero_si256(); }

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

#if HG4_ARM64
static void rotate_and_quantize_128_neon(
    const float *x, const float *pre_diag, int T, int K,
    int8_t *out_xq, float *out_xsc
) {
    const int nb = K / 128;
    const float scale_c = 0.08838834764831845f / 127.0f;
    const float32x4_t vmin_in = vdupq_n_f32(-2048.0f), vmax_in = vdupq_n_f32(2048.0f);
    const float32x4_t vmin_c  = vdupq_n_f32(-127.0f),  vmax_c  = vdupq_n_f32(127.0f);
    #pragma omp parallel for schedule(static) if (T >= 4)
    for (int t = 0; t < T; t++) {
        const float *xt = x + (size_t)t * K;
        int8_t *xqt = out_xq + (size_t)t * K;
        float  *xst = out_xsc + (size_t)t * nb;
        for (int b = 0; b < nb; b++) {
            float prod[128];
            int32_t rot[128];
            const float *xb = xt + b * 128;
            const float *pd = pre_diag + b * 128;
            float32x4_t vmax = vdupq_n_f32(0.0f);
            for (int i = 0; i < 128; i += 4) {
                float32x4_t v = vmulq_f32(vld1q_f32(xb + i), vld1q_f32(pd + i));
                vst1q_f32(prod + i, v);
                vmax = vmaxq_f32(vmax, vabsq_f32(v));
            }
            float max_abs = vmaxvq_f32(vmax);
            int8_t *qd = xqt + b * 128;
            if (max_abs < 1e-12f) {
                xst[b] = 0.0f;
                memset(qd, 0, 128);
                continue;
            }
            float in_sc = max_abs * (1.0f / 2048.0f);
            float32x4_t vinv_in = vdupq_n_f32(2048.0f / max_abs);
            for (int i = 0; i < 128; i += 4) {
                float32x4_t vf = vminq_f32(vmax_in, vmaxq_f32(vmin_in, vmulq_f32(vld1q_f32(prod + i), vinv_in)));
                vst1q_s32(rot + i, vcvtnq_s32_f32(vf));
            }
            for (int h = 1; h < 4; h <<= 1) {
                for (int j = 0; j < 128; j += (h << 1)) {
                    for (int k = 0; k < h; k++) {
                        int32_t a = rot[j + k], bv = rot[j + h + k];
                        rot[j + k] = a + bv; rot[j + h + k] = a - bv;
                    }
                }
            }
            for (int h = 4; h < 128; h <<= 1) {
                for (int j = 0; j < 128; j += (h << 1)) {
                    for (int k = 0; k < h; k += 4) {
                        int32x4_t a = vld1q_s32(rot + j + k);
                        int32x4_t bv = vld1q_s32(rot + j + h + k);
                        vst1q_s32(rot + j + k,     vaddq_s32(a, bv));
                        vst1q_s32(rot + j + h + k, vsubq_s32(a, bv));
                    }
                }
            }
            int32x4_t vimax = vdupq_n_s32(0);
            for (int i = 0; i < 128; i += 4) vimax = vmaxq_s32(vimax, vabsq_s32(vld1q_s32(rot + i)));
            int32_t max_rot = vmaxvq_s32(vimax);
            if (max_rot == 0) {
                xst[b] = 0.0f;
                memset(qd, 0, 128);
            } else {
                xst[b] = (in_sc * (float)max_rot) * scale_c;
                float32x4_t vinv = vdupq_n_f32(127.0f / (float)max_rot);
                for (int i = 0; i < 128; i += 16) {
                    int32x4_t i0 = vcvtnq_s32_f32(vminq_f32(vmax_c, vmaxq_f32(vmin_c, vmulq_f32(vcvtq_f32_s32(vld1q_s32(rot + i +  0)), vinv))));
                    int32x4_t i1 = vcvtnq_s32_f32(vminq_f32(vmax_c, vmaxq_f32(vmin_c, vmulq_f32(vcvtq_f32_s32(vld1q_s32(rot + i +  4)), vinv))));
                    int32x4_t i2 = vcvtnq_s32_f32(vminq_f32(vmax_c, vmaxq_f32(vmin_c, vmulq_f32(vcvtq_f32_s32(vld1q_s32(rot + i +  8)), vinv))));
                    int32x4_t i3 = vcvtnq_s32_f32(vminq_f32(vmax_c, vmaxq_f32(vmin_c, vmulq_f32(vcvtq_f32_s32(vld1q_s32(rot + i + 12)), vinv))));
                    int16x8_t p01 = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
                    int16x8_t p23 = vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3));
                    vst1q_s8(qd + i, vcombine_s8(vqmovn_s16(p01), vqmovn_s16(p23)));
                }
            }
        }
    }
}

static void hg4_group16_neon(
    const hg4_mat_view_t *M, const int8_t *lut16_s8,
    const int8_t *X_q, const float *X_sc,
    float *Y, int T, int rg, int fused_swiglu
) {
    const int R = (int)M->R, K = (int)M->K, nb = K / 128;
    const int out_R = fused_swiglu ? (R / 2) : R;
    const uint8_t *wt = M->tiles + (size_t)rg * (size_t)nb * 1024;
    const uint16_t *wsc = M->scales_fp16 + (size_t)rg * (size_t)nb * 16;
    const int8x16_t vlut = vld1q_s8(lut16_s8);
    const uint8x16_t m0f = vdupq_n_u8(0x0F);

    for (int t0 = 0; t0 < T; t0 += 4) {
        int tblk = (t0 + 4 <= T) ? 4 : (T - t0);
        float32x4_t vs[4][4];
        for (int ti = 0; ti < tblk; ti++)
            for (int q = 0; q < 4; q++) vs[ti][q] = vdupq_n_f32(0.0f);

        for (int b = 0; b < nb; b++) {
            const uint8_t *wb = wt + (size_t)b * 1024;
            float32x4_t vws[4];
            for (int q = 0; q < 4; q++) {
                float wsa[4] = {
                    fp16_to_f32(wsc[b * 16 + q * 4 + 0]), fp16_to_f32(wsc[b * 16 + q * 4 + 1]),
                    fp16_to_f32(wsc[b * 16 + q * 4 + 2]), fp16_to_f32(wsc[b * 16 + q * 4 + 3])
                };
                vws[q] = vld1q_f32(wsa);
            }
            int32x4_t acc[4][4];
            for (int ti = 0; ti < tblk; ti++)
                for (int q = 0; q < 4; q++) acc[ti][q] = vdupq_n_s32(0);

            for (int s = 0; s < 16; s++) {
                int8x16_t wlo[4], whi[4];
                for (int q = 0; q < 4; q++) {
                    uint8x16_t pk = vld1q_u8(wb + s * 64 + q * 16);
                    wlo[q] = vqtbl1q_s8(vlut, vandq_u8(pk, m0f));
                    whi[q] = vqtbl1q_s8(vlut, vshrq_n_u8(pk, 4));
                }
                for (int ti = 0; ti < tblk; ti++) {
                    const int32_t *xp = (const int32_t *)(X_q + (size_t)(t0 + ti) * K + b * 128);
                    int8x16_t xb0 = vreinterpretq_s8_s32(vdupq_n_s32(xp[2 * s + 0]));
                    int8x16_t xb1 = vreinterpretq_s8_s32(vdupq_n_s32(xp[2 * s + 1]));
                    for (int q = 0; q < 4; q++) {
#if HG4_HAS_ARM_DOTPROD
                        acc[ti][q] = vdotq_s32(acc[ti][q], wlo[q], xb0);
                        acc[ti][q] = vdotq_s32(acc[ti][q], whi[q], xb1);
#else
                        int16x8_t m0 = vpaddq_s16(vmull_s8(vget_low_s8(wlo[q]), vget_low_s8(xb0)), vmull_high_s8(wlo[q], xb0));
                        int16x8_t m1 = vpaddq_s16(vmull_s8(vget_low_s8(whi[q]), vget_low_s8(xb1)), vmull_high_s8(whi[q], xb1));
                        acc[ti][q] = vpadalq_s16(acc[ti][q], vpaddq_s16(m0, m1));
#endif
                    }
                }
            }
            for (int ti = 0; ti < tblk; ti++) {
                int ta = t0 + ti;
                float xsc = X_sc[ta * nb + b];
                for (int q = 0; q < 4; q++) {
                    vs[ti][q] = vaddq_f32(vs[ti][q], vmulq_f32(vcvtq_f32_s32(acc[ti][q]), vmulq_n_f32(vws[q], xsc)));
                }
            }
        }
        for (int ti = 0; ti < tblk; ti++) {
            float *yt = Y + (size_t)(t0 + ti) * out_R;
            if (fused_swiglu) {
                float sv[16];
                for (int q = 0; q < 4; q++) vst1q_f32(sv + q * 4, vs[ti][q]);
                for (int j = 0; j < 8; j++) yt[rg * 8 + j] = (sv[j] / (1.0f + expf(-sv[j]))) * sv[j + 8];
            } else {
                for (int q = 0; q < 4; q++) vst1q_f32(yt + rg * 16 + q * 4, vs[ti][q]);
            }
        }
    }
}

static void hg4_tile4_neon(
    const hg4_mat_view_t *M, const int8_t *lut16_s8,
    const int8_t *X_q, const float *X_sc,
    float *Y, int T, int tile_idx, int fused_swiglu
) {
    const int R = (int)M->R, K = (int)M->K, nb = K / 128;
    const int out_R = fused_swiglu ? (R / 2) : R;
    const int o_base = tile_idx * 4;
    const uint8_t *wt = M->tiles + (size_t)tile_idx * (size_t)nb * 256;
    const uint16_t *wsc = M->scales_fp16 + (size_t)tile_idx * (size_t)nb * 4;
    const int8x16_t vlut = vld1q_s8(lut16_s8);
    const uint8x16_t m0f = vdupq_n_u8(0x0F);

    for (int t0 = 0; t0 < T; t0 += 16) {
        int tblk = (t0 + 16 <= T) ? 16 : (T - t0);
        float32x4_t vs[16];
        for (int ti = 0; ti < tblk; ti++) vs[ti] = vdupq_n_f32(0.0f);

        for (int b = 0; b < nb; b++) {
            const uint8_t *wb = wt + b * 256;
            float ws_arr[4] = {
                fp16_to_f32(wsc[b * 4 + 0]), fp16_to_f32(wsc[b * 4 + 1]),
                fp16_to_f32(wsc[b * 4 + 2]), fp16_to_f32(wsc[b * 4 + 3])
            };
            float32x4_t vws = vld1q_f32(ws_arr);
            int8x16_t wlo0[2][4], wlo1[2][4], whi0[2][4], whi1[2][4];
            for (int u = 0; u < 2; u++) {
                const uint8_t *wu = wb + u * 128;
                for (int ri = 0; ri < 4; ri++) {
                    uint8x16_t pk0 = vld1q_u8(wu + ri * 32 + 0);
                    uint8x16_t pk1 = vld1q_u8(wu + ri * 32 + 16);
                    wlo0[u][ri] = vqtbl1q_s8(vlut, vandq_u8(pk0, m0f));
                    wlo1[u][ri] = vqtbl1q_s8(vlut, vandq_u8(pk1, m0f));
                    whi0[u][ri] = vqtbl1q_s8(vlut, vshrq_n_u8(pk0, 4));
                    whi1[u][ri] = vqtbl1q_s8(vlut, vshrq_n_u8(pk1, 4));
                }
            }
            for (int ti = 0; ti < tblk; ti++) {
                int t_abs = t0 + ti;
                const int8_t *xq = X_q + (size_t)t_abs * K + b * 128;
                int32_t sums[4] = {0, 0, 0, 0};
                for (int ri = 0; ri < 4; ri++) {
                    int32x4_t acc = vdupq_n_s32(0);
                    for (int u = 0; u < 2; u++) {
                        int8x16_t x0 = vld1q_s8(xq + u * 64 + 0);
                        int8x16_t x1 = vld1q_s8(xq + u * 64 + 16);
                        int8x16_t x2 = vld1q_s8(xq + u * 64 + 32);
                        int8x16_t x3 = vld1q_s8(xq + u * 64 + 48);
#if HG4_HAS_ARM_DOTPROD
                        acc = vdotq_s32(acc, wlo0[u][ri], x0);
                        acc = vdotq_s32(acc, wlo1[u][ri], x1);
                        acc = vdotq_s32(acc, whi0[u][ri], x2);
                        acc = vdotq_s32(acc, whi1[u][ri], x3);
#else
                        int16x8_t m0 = vpaddq_s16(vmull_s8(vget_low_s8(wlo0[u][ri]), vget_low_s8(x0)),
                                                  vmull_high_s8(wlo0[u][ri], x0));
                        int16x8_t m1 = vpaddq_s16(vmull_s8(vget_low_s8(wlo1[u][ri]), vget_low_s8(x1)),
                                                  vmull_high_s8(wlo1[u][ri], x1));
                        int16x8_t m2 = vpaddq_s16(vmull_s8(vget_low_s8(whi0[u][ri]), vget_low_s8(x2)),
                                                  vmull_high_s8(whi0[u][ri], x2));
                        int16x8_t m3 = vpaddq_s16(vmull_s8(vget_low_s8(whi1[u][ri]), vget_low_s8(x3)),
                                                  vmull_high_s8(whi1[u][ri], x3));
                        acc = vpadalq_s16(acc, vaddq_s16(vaddq_s16(m0, m1), vaddq_s16(m2, m3)));
#endif
                    }
                    sums[ri] = vaddvq_s32(acc);
                }
                int32x4_t vsums = vld1q_s32(sums);
                vs[ti] = vaddq_f32(vs[ti], vmulq_f32(vcvtq_f32_s32(vsums), vmulq_n_f32(vws, X_sc[t_abs * nb + b])));
            }
        }
        for (int ti = 0; ti < tblk; ti++) {
            float *yt = Y + (size_t)(t0 + ti) * out_R;
            if (fused_swiglu) {
                float s[4];
                vst1q_f32(s, vs[ti]);
                yt[tile_idx * 2 + 0] = (s[0] / (1.0f + expf(-s[0]))) * s[2];
                yt[tile_idx * 2 + 1] = (s[1] / (1.0f + expf(-s[1]))) * s[3];
            } else {
                vst1q_f32(yt + o_base, vs[ti]);
            }
        }
    }
}
#endif

static void hg4_group16_scalar(
    const hg4_mat_view_t *M, const int8_t *lut16_s8,
    const int8_t *X_q, const float *X_sc,
    float *Y, int T, int rg, int fused_swiglu
) {
    const int R = (int)M->R, K = (int)M->K, nb = K / 128;
    const int out_R = fused_swiglu ? (R / 2) : R;
    const uint8_t *wt = M->tiles + (size_t)rg * (size_t)nb * 1024;
    const uint16_t *wsc = M->scales_fp16 + (size_t)rg * (size_t)nb * 16;

    for (int t = 0; t < T; t++) {
        float vs[16] = {0};
        const int8_t *xqt = X_q + (size_t)t * K;
        for (int b = 0; b < nb; b++) {
            const uint8_t *wb = wt + (size_t)b * 1024;
            const int8_t *xqb = xqt + b * 128;
            int32_t acc[16] = {0};
            for (int s = 0; s < 16; s++) {
                const uint8_t *ws = wb + s * 64;
                for (int r = 0; r < 16; r++) {
                    const uint8_t *wr = ws + r * 4;
                    for (int j = 0; j < 4; j++) {
                        uint8_t pk = wr[j];
                        acc[r] += (int32_t)lut16_s8[pk & 0x0F] * (int32_t)xqb[s * 8 + j]
                                + (int32_t)lut16_s8[pk >> 4]   * (int32_t)xqb[s * 8 + 4 + j];
                    }
                }
            }
            float xsc = X_sc[t * nb + b];
            for (int r = 0; r < 16; r++) {
                float sc = fp16_to_f32(wsc[b * 16 + r]) * xsc;
                vs[r] = vs[r] + (float)acc[r] * sc;
            }
        }
        float *yt = Y + (size_t)t * out_R;
        if (fused_swiglu) {
            for (int j = 0; j < 8; j++) yt[rg * 8 + j] = (vs[j] / (1.0f + expf(-vs[j]))) * vs[j + 8];
        } else {
            for (int r = 0; r < 16; r++) yt[rg * 16 + r] = vs[r];
        }
    }
}

static void hg4_tile4_scalar(
    const hg4_mat_view_t *M, const int8_t *lut16_s8,
    const int8_t *X_q, const float *X_sc,
    float *Y, int T, int tl, int fused_swiglu
) {
    const int R = (int)M->R, K = (int)M->K, nb = K / 128;
    const int out_R = fused_swiglu ? (R / 2) : R;
    const uint8_t *wt = M->tiles + (size_t)tl * (size_t)nb * 256;
    const uint16_t *wsc = M->scales_fp16 + (size_t)tl * (size_t)nb * 4;
    for (int t = 0; t < T; t++) {
        float vs[4] = {0, 0, 0, 0};
        const int8_t *xqt = X_q + (size_t)t * K;
        for (int b = 0; b < nb; b++) {
            const uint8_t *wb = wt + b * 256;
            const int8_t *xqb = xqt + b * 128;
            float xsc = X_sc[t * nb + b];
            for (int ri = 0; ri < 4; ri++) {
                int32_t acc = 0;
                for (int u = 0; u < 2; u++) {
                    const uint8_t *wu = wb + u * 128 + ri * 32;
                    const int8_t *xqu = xqb + u * 64;
                    for (int j = 0; j < 32; j++) {
                        uint8_t pk = wu[j];
                        acc += (int32_t)lut16_s8[pk & 0x0F] * (int32_t)xqu[j]
                             + (int32_t)lut16_s8[pk >> 4]   * (int32_t)xqu[j + 32];
                    }
                }
                float sc = fp16_to_f32(wsc[b * 4 + ri]) * xsc;
                vs[ri] = vs[ri] + (float)acc * sc;
            }
        }
        float *yt = Y + (size_t)t * out_R;
        if (fused_swiglu) {
            yt[tl * 2 + 0] = (vs[0] / (1.0f + expf(-vs[0]))) * vs[2];
            yt[tl * 2 + 1] = (vs[1] / (1.0f + expf(-vs[1]))) * vs[3];
        } else {
            for (int ri = 0; ri < 4; ri++) yt[tl * 4 + ri] = vs[ri];
        }
    }
}

static void hg4_matmul(const hg4_mat_view_t *M, const float *X, float *Y, int T, int fused_swiglu) {
    const int K = (int)M->K;
    if (g_hg4.is_16r) {
        const int n_groups = (int)(M->R / 16);
#if HG4_X86
        if (g_hg4.isa_tier >= 1 && g_hg4.isa_tier <= 3) {
            rotate_and_quantize_128_avx2(X, M->pre_diag, T, K, g_hg4.scr_xq, g_hg4.scr_xsc, g_hg4.scr_xsum);
            if (hg_cuda_dispatch_matmul(M->tiles, M->scales_fp16, M->R, M->K, 4u, g_hg4.scr_xq, g_hg4.scr_xsc, Y, T, fused_swiglu ? 1 : 0)) {
                return;
            }
            if (g_hg4.isa_tier == 3) {
                #pragma omp parallel for schedule(static)
                for (int rg = 0; rg < n_groups; rg++)
                    hg4_group16_avx512(M, g_hg4.lut16_u8, g_hg4.scr_xq, g_hg4.scr_xsc, g_hg4.scr_xsum, Y, T, rg, fused_swiglu);
                return;
            } else if (g_hg4.isa_tier == 2) {
                #pragma omp parallel for schedule(static)
                for (int rg = 0; rg < n_groups; rg++)
                    hg4_group16_avxvnni(M, g_hg4.lut16_u8, g_hg4.scr_xq, g_hg4.scr_xsc, g_hg4.scr_xsum, Y, T, rg, fused_swiglu);
                return;
            } else {
                #pragma omp parallel for schedule(static)
                for (int rg = 0; rg < n_groups; rg++)
                    hg4_group16_avx2(M, g_hg4.lut16_s8, g_hg4.scr_xq, g_hg4.scr_xsc, Y, T, rg, fused_swiglu);
                return;
            }
        }
#elif HG4_ARM64
        if (g_hg4.isa_tier == 4 || g_hg4.isa_tier == 5) {
            rotate_and_quantize_128_neon(X, M->pre_diag, T, K, g_hg4.scr_xq, g_hg4.scr_xsc);
            #pragma omp parallel for schedule(static)
            for (int rg = 0; rg < n_groups; rg++)
                hg4_group16_neon(M, g_hg4.lut16_s8, g_hg4.scr_xq, g_hg4.scr_xsc, Y, T, rg, fused_swiglu);
            return;
        }
#endif
        rotate_and_quantize_128_scalar(X, M->pre_diag, T, K, g_hg4.scr_xq, g_hg4.scr_xsc, g_hg4.scr_xsum);
        #pragma omp parallel for schedule(static)
        for (int rg = 0; rg < n_groups; rg++)
            hg4_group16_scalar(M, g_hg4.lut16_s8, g_hg4.scr_xq, g_hg4.scr_xsc, Y, T, rg, fused_swiglu);
        return;
    }

    const int n_tiles = (int)(M->R / 4);
#if HG4_X86
    if (g_hg4.isa_tier >= 1 && g_hg4.isa_tier <= 3) {
        rotate_and_quantize_128_avx2(X, M->pre_diag, T, K, g_hg4.scr_xq, g_hg4.scr_xsc, g_hg4.scr_xsum);
        if (g_hg4.isa_tier == 3) {
            #pragma omp parallel for schedule(static)
            for (int tl = 0; tl < n_tiles; tl++)
                hg4_tile4_avx512(M, g_hg4.lut16_u8, g_hg4.scr_xq, g_hg4.scr_xsc, g_hg4.scr_xsum, Y, T, tl, fused_swiglu);
            return;
        } else if (g_hg4.isa_tier == 2) {
            #pragma omp parallel for schedule(static)
            for (int tl = 0; tl < n_tiles; tl++)
                hg4_tile4_avxvnni(M, g_hg4.lut16_u8, g_hg4.scr_xq, g_hg4.scr_xsc, g_hg4.scr_xsum, Y, T, tl, fused_swiglu);
            return;
        } else {
            #pragma omp parallel for schedule(static)
            for (int tl = 0; tl < n_tiles; tl++)
                hg4_tile4_avx2(M, g_hg4.lut16_s8, g_hg4.scr_xq, g_hg4.scr_xsc, Y, T, tl, fused_swiglu);
            return;
        }
    }
#elif HG4_ARM64
    if (g_hg4.isa_tier == 4 || g_hg4.isa_tier == 5) {
        rotate_and_quantize_128_neon(X, M->pre_diag, T, K, g_hg4.scr_xq, g_hg4.scr_xsc);
        #pragma omp parallel for schedule(static)
        for (int tl = 0; tl < n_tiles; tl++)
            hg4_tile4_neon(M, g_hg4.lut16_s8, g_hg4.scr_xq, g_hg4.scr_xsc, Y, T, tl, fused_swiglu);
        return;
    }
#endif
    rotate_and_quantize_128_scalar(X, M->pre_diag, T, K, g_hg4.scr_xq, g_hg4.scr_xsc, g_hg4.scr_xsum);
    #pragma omp parallel for schedule(static)
    for (int tl = 0; tl < n_tiles; tl++)
        hg4_tile4_scalar(M, g_hg4.lut16_s8, g_hg4.scr_xq, g_hg4.scr_xsc, Y, T, tl, fused_swiglu);
}

static void hg4_embed_lookup_row(int token_id, float *dst_2048) {
    if (token_id < 0 || token_id >= HG4_VOCAB) token_id = 0;
    const hg4_mat_view_t *E = &g_hg4.emb_mat;
    const int nb = HG4_D / 128;
    if (g_hg4.is_16r) {
        int rg = token_id / 16, lane = token_id % 16;
        const uint8_t *wt = E->tiles + (size_t)rg * (size_t)nb * 1024;
        const uint16_t *wsc = E->scales_fp16 + (size_t)rg * (size_t)nb * 16;
        for (int b = 0; b < nb; b++) {
            float *blk = dst_2048 + b * 128;
            float sc = fp16_to_f32(wsc[b * 16 + lane]);
            const uint8_t *wb = wt + (size_t)b * 1024;
            for (int s = 0; s < 16; s++) {
                const uint8_t *wr = wb + s * 64 + lane * 4;
                for (int j = 0; j < 4; j++) {
                    uint8_t pk = wr[j];
                    blk[s * 8 + j]     = g_hg4.lut16_s8_f[pk & 0x0F] * sc;
                    blk[s * 8 + 4 + j] = g_hg4.lut16_s8_f[pk >> 4]   * sc;
                }
            }
            fwht128_1block_scalar(blk);
            for (int k = 0; k < 128; k++) blk[k] *= E->pre_diag[b * 128 + k];
        }
        return;
    }
    int tl = token_id / 4, ri = token_id % 4;
    const uint8_t *wt = E->tiles + (size_t)tl * (size_t)nb * 256;
    const uint16_t *wsc = E->scales_fp16 + (size_t)tl * (size_t)nb * 4;

    for (int b = 0; b < nb; b++) {
        float *blk = dst_2048 + b * 128;
        float sc = fp16_to_f32(wsc[b * 4 + ri]);
        const uint8_t *wb = wt + b * 256;
        for (int u = 0; u < 2; u++) {
            const uint8_t *wu = wb + u * 128 + ri * 32;
            for (int j = 0; j < 32; j++) {
                uint8_t pk = wu[j];
                blk[u * 64 + j]      = g_hg4.lut16_s8_f[pk & 0x0F] * sc;
                blk[u * 64 + 32 + j] = g_hg4.lut16_s8_f[pk >> 4]   * sc;
            }
        }
        fwht128_1block_scalar(blk);
        for (int k = 0; k < 128; k++) blk[k] *= E->pre_diag[b * 128 + k];
    }
}

static inline void rms_norm_1row(const float *xr, const float *w, float *yr, int dim) {
    double ss = 0.0;
    for (int d = 0; d < dim; d++) ss += (double)xr[d] * (double)xr[d];
    float inv = 1.0f / sqrtf((float)(ss / (double)dim) + 1e-5f);
    for (int d = 0; d < dim; d++) yr[d] = xr[d] * inv * w[d];
}

static void rms_norm_f32(const float *x, const float *w, float *y, int rows, int dim) {
    if (rows == 1) {
        rms_norm_1row(x, w, y, dim);
        return;
    }
    #pragma omp parallel for schedule(static) if (rows >= 8)
    for (int r = 0; r < rows; r++) {
        rms_norm_1row(x + (size_t)r * dim, w, y + (size_t)r * dim, dim);
    }
}

static void execute_hg4_layer_step(int l, const float *h_in, float *h_out, int pos_off, int T) {
    const hg4_layer_t *lyr = &g_hg4.layers[l];
    rms_norm_f32(h_in, lyr->op_norm, g_hg4.scr_op_in, T, HG4_D);

    if (lyr->kind == 0) {
        hg4_matmul(&lyr->in_proj, g_hg4.scr_op_in, g_hg4.scr_bcx, T, 0);
        uint16_t *c_cache = g_hg4.conv_cache_bf16[l];
        #pragma omp parallel for schedule(static) if (T >= 4)
        for (int t = 0; t < T; t++) {
            uint16_t *dst_pre = c_cache + (size_t)(pos_off + t) * HG4_D;
            const float *bcxt = g_hg4.scr_bcx + (size_t)t * 6144;
            for (int c = 0; c < HG4_D; c++) dst_pre[c] = f32_to_bf16(bcxt[c] * bcxt[4096 + c]);
        }
        #pragma omp parallel for schedule(static)
        for (int c = 0; c < HG4_D; c++) {
            const float *wk = lyr->conv_w + c * 7;
            for (int t = 0; t < T; t++) {
                int p = pos_off + t;
                float acc = 0.0f;
                for (int k = 0; k < 7; k++) {
                    int ts = p - 6 + k;
                    if (ts >= 0) acc += bf16_to_f32(c_cache[(size_t)ts * HG4_D + c]) * wk[k];
                }
                g_hg4.scr_attn[(size_t)t * HG4_D + c] = g_hg4.scr_bcx[(size_t)t * 6144 + 2048 + c] * acc;
            }
        }
        hg4_matmul(&lyr->out_proj, g_hg4.scr_attn, g_hg4.scr_op_out, T, 0);
    } else {
        hg4_matmul(&lyr->qkv_proj, g_hg4.scr_op_in, g_hg4.scr_qkv, T, 0);
        float *gate_vals = g_hg4.scr_gates;
        #pragma omp parallel for schedule(static) if (T >= 4)
        for (int t = 0; t < T; t++) {
            const float *xrt = g_hg4.scr_op_in + (size_t)t * HG4_D;
            for (int h = 0; h < 16; h++) {
                const float *gw = lyr->gate_w + h * HG4_D;
                float s = 0.0f;
                #pragma omp simd reduction(+:s)
                for (int d = 0; d < HG4_D; d++) s += xrt[d] * gw[d];
                gate_vals[t * 16 + h] = 1.0f / (1.0f + expf(-s));
            }
        }
        const float *inv_f = (lyr->kind == 1) ? g_hg4.inv_freq_sliding : g_hg4.inv_freq_full;
        uint16_t *kc = g_hg4.k_cache_bf16[l], *vc = g_hg4.v_cache_bf16[l];

        #pragma omp parallel for schedule(static) if (T >= 4)
        for (int t = 0; t < T; t++) {
            float *q_t = g_hg4.scr_qkv + (size_t)t * 3072;
            float *k_t = q_t + 2048;
            float *v_t = q_t + 2560;
            for (int h = 0; h < 16; h++) rms_norm_1row(q_t + h * 128, lyr->qn_w, q_t + h * 128, 128);
            for (int h = 0; h < 4;  h++) rms_norm_1row(k_t + h * 128, lyr->kn_w, k_t + h * 128, 128);

            int pos = pos_off + t;
            float cs[64], sn[64];
            for (int d = 0; d < 64; d++) {
                float ang = (float)pos * inv_f[d];
                cs[d] = cosf(ang); sn[d] = sinf(ang);
            }
            for (int h = 0; h < 16; h++) {
                float *qh = q_t + h * 128;
                for (int d = 0; d < 64; d++) {
                    float x1 = qh[d], x2 = qh[d + 64];
                    qh[d] = x1 * cs[d] - x2 * sn[d];
                    qh[d + 64] = x2 * cs[d] + x1 * sn[d];
                }
            }
            for (int h = 0; h < 4; h++) {
                float *kh = k_t + h * 128;
                for (int d = 0; d < 64; d++) {
                    float x1 = kh[d], x2 = kh[d + 64];
                    kh[d] = x1 * cs[d] - x2 * sn[d];
                    kh[d + 64] = x2 * cs[d] + x1 * sn[d];
                }
            }
            uint16_t *kd = kc + (size_t)pos * 512;
            uint16_t *vd = vc + (size_t)pos * 512;
            for (int d = 0; d < 512; d++) {
                kd[d] = f32_to_bf16(k_t[d]);
                vd[d] = f32_to_bf16(v_t[d]);
            }
        }

        const float scale = 1.0f / sqrtf(128.0f);
        #pragma omp parallel for schedule(static)
        for (int h = 0; h < 16; h++) {
            int kv_h = h / 4;
            for (int t = 0; t < T; t++) {
                int pos = pos_off + t;
                int j0 = (lyr->kind == 1 && pos - HG4_SLIDING_W + 1 > 0) ? (pos - HG4_SLIDING_W + 1) : 0;
                const float *qh = g_hg4.scr_qkv + (size_t)t * 3072 + h * 128;
                float m_max = -1e30f, l_sum = 0.0f;
                float out_h[128] = {0};
                for (int j = j0; j <= pos; j++) {
                    const uint16_t *kj = kc + (size_t)j * 512 + kv_h * 128;
                    const uint16_t *vj = vc + (size_t)j * 512 + kv_h * 128;
                    float dot = 0.0f;
                    for (int d = 0; d < 128; d++) dot += qh[d] * bf16_to_f32(kj[d]);
                    dot *= scale;
                    if (dot > m_max) {
                        float alpha = expf(m_max - dot);
                        l_sum = l_sum * alpha + 1.0f;
                        for (int d = 0; d < 128; d++) out_h[d] = out_h[d] * alpha + bf16_to_f32(vj[d]);
                        m_max = dot;
                    } else {
                        float w = expf(dot - m_max);
                        l_sum += w;
                        for (int d = 0; d < 128; d++) out_h[d] += w * bf16_to_f32(vj[d]);
                    }
                }
                float inv_l = gate_vals[t * 16 + h] / (l_sum + 1e-12f);
                float *dst = g_hg4.scr_attn + (size_t)t * HG4_D + h * 128;
                for (int d = 0; d < 128; d++) dst[d] = out_h[d] * inv_l;
            }
        }
        hg4_matmul(&lyr->o_proj, g_hg4.scr_attn, g_hg4.scr_op_out, T, 0);
    }

    rms_norm_f32(g_hg4.scr_op_out, lyr->post_op_norm, g_hg4.scr_norm, T, HG4_D);
    #pragma omp parallel for schedule(static) if (T >= 4)
    for (int i = 0; i < T * HG4_D; i++) g_hg4.scr_mid[i] = h_in[i] + g_hg4.scr_norm[i];

    rms_norm_f32(g_hg4.scr_mid, lyr->ffn_norm, g_hg4.scr_op_in, T, HG4_D);
    hg4_matmul(&lyr->w13_proj, g_hg4.scr_op_in, g_hg4.scr_ffn_mid, T, 1);
    hg4_matmul(&lyr->w2_proj,  g_hg4.scr_ffn_mid, g_hg4.scr_op_out, T, 0);
    rms_norm_f32(g_hg4.scr_op_out, lyr->post_ffn_norm, g_hg4.scr_norm, T, HG4_D);
    #pragma omp parallel for schedule(static) if (T >= 4)
    for (int i = 0; i < T * HG4_D; i++) h_out[i] = g_hg4.scr_mid[i] + g_hg4.scr_norm[i];
}

HG4_API int lfm3_hg4_cache_init(int max_tokens) {
    if (!g_hg4.ready) return -1;
    if (max_tokens < 512) max_tokens = 512;
    if (max_tokens == g_hg4.max_seq && g_hg4.conv_cache_bf16[0] != NULL) {
        g_hg4.cache_len = 0;
        return 0;
    }
    for (int l = 0; l < HG4_LAYERS; l++) {
        free(g_hg4.k_cache_bf16[l]);    g_hg4.k_cache_bf16[l] = NULL;
        free(g_hg4.v_cache_bf16[l]);    g_hg4.v_cache_bf16[l] = NULL;
        free(g_hg4.conv_cache_bf16[l]); g_hg4.conv_cache_bf16[l] = NULL;
        if (g_hg4.layers[l].kind == 0) {
            g_hg4.conv_cache_bf16[l] = (uint16_t *)calloc((size_t)max_tokens * HG4_D, sizeof(uint16_t));
            if (!g_hg4.conv_cache_bf16[l]) return -2;
        } else {
            g_hg4.k_cache_bf16[l] = (uint16_t *)calloc((size_t)max_tokens * 512, sizeof(uint16_t));
            g_hg4.v_cache_bf16[l] = (uint16_t *)calloc((size_t)max_tokens * 512, sizeof(uint16_t));
            if (!g_hg4.k_cache_bf16[l] || !g_hg4.v_cache_bf16[l]) return -2;
        }
    }
    g_hg4.max_seq = max_tokens;
    g_hg4.cache_len = 0;
    return 0;
}

HG4_API void lfm3_hg4_cache_truncate(int keep_tokens) {
    if (keep_tokens < 0) keep_tokens = 0;
    if (keep_tokens <= g_hg4.cache_len) g_hg4.cache_len = keep_tokens;
}

HG4_API int lfm3_hg4_cache_get_len(void) {
    return g_hg4.cache_len;
}

HG4_API int lfm3_hg4_init(const char *hg4_dir) {
    if (g_hg4.ready && strcmp(g_hg4.loaded_dir, hg4_dir) == 0) return 0;
    if (g_hg4.ready) {
        hg4_mmap_close(&g_hg4.emb_mf);
        for (int l = 0; l < HG4_LAYERS; l++) {
            hg4_mmap_close(&g_hg4.layers[l].mf);
            free(g_hg4.k_cache_bf16[l]);    g_hg4.k_cache_bf16[l] = NULL;
            free(g_hg4.v_cache_bf16[l]);    g_hg4.v_cache_bf16[l] = NULL;
            free(g_hg4.conv_cache_bf16[l]); g_hg4.conv_cache_bf16[l] = NULL;
        }
        g_hg4.ready = 0;
        g_hg4.max_seq = 0;
    }

    g_hg4.isa_tier = detect_hg4_isa();
    char path[512];
    snprintf(path, sizeof(path), "%s/meta.bin", hg4_dir);
    FILE *mf = fopen(path, "rb");
    if (!mf) return -1;
    char magic[4];
    uint32_t hdr[4];
    if (fread(magic, 1, 4, mf) != 4 ||
        (memcmp(magic, "HG4R", 4) != 0 && memcmp(magic, "HG4M", 4) != 0) ||
        fread(hdr, 4, 4, mf) != 4 ||
        fread(g_hg4.lut16_u8, 1, 16, mf) != 16 ||
        fread(g_hg4.final_norm, 4, HG4_D, mf) != HG4_D ||
        fread(g_hg4.inv_freq_full, 4, 64, mf) != 64 ||
        fread(g_hg4.inv_freq_sliding, 4, 64, mf) != 64 ||
        fread(g_hg4.kinds, 4, HG4_LAYERS, mf) != HG4_LAYERS) { fclose(mf); return -2; }
    fclose(mf);
    g_hg4.is_16r = (memcmp(magic, "HG4R", 4) == 0) ? 1 : 0;
    for (int i = 0; i < 16; i++) {
        int sv = (int)g_hg4.lut16_u8[i] - 128;
        g_hg4.lut16_s8[i] = (int8_t)sv;
        g_hg4.lut16_s8_f[i] = (float)sv;
    }

    snprintf(path, sizeof(path), "%s/embed_lm_head.hg4", hg4_dir);
    if (hg4_mmap_open(path, &g_hg4.emb_mf) != 0) return -3;
    parse_hg4_mat((const uint8_t *)g_hg4.emb_mf.ptr, &g_hg4.emb_mat);

    for (int l = 0; l < HG4_LAYERS; l++) {
        snprintf(path, sizeof(path), "%s/layer_%02d.hg4", hg4_dir, l);
        hg4_layer_t *lyr = &g_hg4.layers[l];
        if (hg4_mmap_open(path, &lyr->mf) != 0) return -(10 + l);
        const uint8_t *p = (const uint8_t *)lyr->mf.ptr;
        size_t off = 0;
        memcpy(&lyr->kind, p + off, 4); off += 4;
        lyr->op_norm       = (const float *)(p + off); off += HG4_D * 4;
        lyr->post_op_norm  = (const float *)(p + off); off += HG4_D * 4;
        lyr->ffn_norm      = (const float *)(p + off); off += HG4_D * 4;
        lyr->post_ffn_norm = (const float *)(p + off); off += HG4_D * 4;
        if (lyr->kind == 0) {
            lyr->conv_w = (const float *)(p + off); off += HG4_D * 7 * 4;
            uint32_t l_in, l_out;
            memcpy(&l_in,  p + off, 4); off += 4;
            memcpy(&l_out, p + off, 4); off += 4;
            parse_hg4_mat(p + off, &lyr->in_proj);  off += l_in;
            parse_hg4_mat(p + off, &lyr->out_proj); off += l_out;
        } else {
            lyr->gate_w = (const float *)(p + off); off += 16 * HG4_D * 4;
            lyr->qn_w   = (const float *)(p + off); off += 128 * 4;
            lyr->kn_w   = (const float *)(p + off); off += 128 * 4;
            uint32_t l_qkv, l_o;
            memcpy(&l_qkv, p + off, 4); off += 4;
            memcpy(&l_o,   p + off, 4); off += 4;
            parse_hg4_mat(p + off, &lyr->qkv_proj); off += l_qkv;
            parse_hg4_mat(p + off, &lyr->o_proj);   off += l_o;
        }
        uint32_t l_w13, l_w2;
        memcpy(&l_w13, p + off, 4); off += 4;
        memcpy(&l_w2,  p + off, 4); off += 4;
        parse_hg4_mat(p + off, &lyr->w13_proj); off += l_w13;
        parse_hg4_mat(p + off, &lyr->w2_proj);  off += l_w2;
    }

    if (!g_hg4.scr_a) {
        size_t mt = HG4_PANEL_MAX;
        g_hg4.scr_a       = (float *)malloc(mt * HG4_D * 4);
        g_hg4.scr_b       = (float *)malloc(mt * HG4_D * 4);
        g_hg4.scr_op_in   = (float *)malloc(mt * HG4_D * 4);
        g_hg4.scr_op_out  = (float *)malloc(mt * HG4_D * 4);
        g_hg4.scr_norm    = (float *)malloc(mt * HG4_D * 4);
        g_hg4.scr_mid     = (float *)malloc(mt * HG4_D * 4);
        g_hg4.scr_bcx     = (float *)malloc(mt * 6144 * 4);
        g_hg4.scr_qkv     = (float *)malloc(mt * 3072 * 4);
        g_hg4.scr_attn    = (float *)malloc(mt * HG4_D * 4);
        g_hg4.scr_ffn_mid = (float *)malloc(mt * HG4_FFN * 4);
        g_hg4.scr_gates   = (float *)malloc(mt * 16 * 4);
        g_hg4.scr_xq      = (int8_t *)malloc(mt * HG4_FFN);
        g_hg4.scr_xsc     = (float *)malloc(mt * (HG4_FFN / 128) * 4);
        g_hg4.scr_xsum    = (int32_t *)malloc(mt * (HG4_FFN / 128) * 4);
    }
    snprintf(g_hg4.loaded_dir, sizeof(g_hg4.loaded_dir), "%s", hg4_dir);
    g_hg4.ready = 1;
    return lfm3_hg4_cache_init(HG4_DEFAULT_CTX);
}

/* Feed `n_tokens` token IDs starting at `g_hg4.cache_len` (chunked in panels of <= 512) */
HG4_API int lfm3_hg4_cache_feed(
    const int *token_ids, int n_tokens,
    float *out_last_hidden_2048, float *out_last_logits_128k
) {
    if (!g_hg4.ready || n_tokens <= 0 || g_hg4.cache_len + n_tokens > g_hg4.max_seq) return -1;
    float h_last[HG4_D];
    int done = 0;
    while (done < n_tokens) {
        int chunk = (n_tokens - done > HG4_PANEL_MAX) ? HG4_PANEL_MAX : (n_tokens - done);
        float *cur = g_hg4.scr_a, *nxt = g_hg4.scr_b;
        #pragma omp parallel for schedule(static) if (chunk >= 8)
        for (int t = 0; t < chunk; t++) {
            hg4_embed_lookup_row(token_ids[done + t], cur + (size_t)t * HG4_D);
        }
        for (int l = 0; l < HG4_LAYERS; l++) {
            execute_hg4_layer_step(l, cur, nxt, g_hg4.cache_len, chunk);
            float *tmp = cur; cur = nxt; nxt = tmp;
        }
        g_hg4.cache_len += chunk;
        done += chunk;
        if (done == n_tokens) {
            rms_norm_f32(cur + (size_t)(chunk - 1) * HG4_D, g_hg4.final_norm, h_last, 1, HG4_D);
            if (out_last_hidden_2048) memcpy(out_last_hidden_2048, h_last, HG4_D * sizeof(float));
            if (out_last_logits_128k) hg4_matmul(&g_hg4.emb_mat, h_last, out_last_logits_128k, 1, 0);
        }
    }
    return g_hg4.cache_len;
}

/* Feed continuous [n_tokens, 2048] multimodal / HybridLink embeddings starting at `g_hg4.cache_len` */
HG4_API int lfm3_hg4_cache_feed_embeddings(
    const float *embeds_t_2048, int n_tokens,
    float *out_last_hidden_2048, float *out_last_logits_128k
) {
    if (!g_hg4.ready || n_tokens <= 0 || g_hg4.cache_len + n_tokens > g_hg4.max_seq) return -1;
    float h_last[HG4_D];
    int done = 0;
    while (done < n_tokens) {
        int chunk = (n_tokens - done > HG4_PANEL_MAX) ? HG4_PANEL_MAX : (n_tokens - done);
        float *cur = g_hg4.scr_a, *nxt = g_hg4.scr_b;
        memcpy(cur, embeds_t_2048 + (size_t)done * HG4_D, (size_t)chunk * HG4_D * sizeof(float));
        for (int l = 0; l < HG4_LAYERS; l++) {
            execute_hg4_layer_step(l, cur, nxt, g_hg4.cache_len, chunk);
            float *tmp = cur; cur = nxt; nxt = tmp;
        }
        g_hg4.cache_len += chunk;
        done += chunk;
        if (done == n_tokens) {
            rms_norm_f32(cur + (size_t)(chunk - 1) * HG4_D, g_hg4.final_norm, h_last, 1, HG4_D);
            if (out_last_hidden_2048) memcpy(out_last_hidden_2048, h_last, HG4_D * sizeof(float));
            if (out_last_logits_128k) hg4_matmul(&g_hg4.emb_mat, h_last, out_last_logits_128k, 1, 0);
        }
    }
    return g_hg4.cache_len;
}

/* Compute post-final_norm [n_tokens, 2048] hidden states for C1->C2->C3 (`c1_infer_states`) with prefix reuse! */
HG4_API int lfm3_hg4_hidden_states(
    const int *token_ids, int n_tokens, int start_pos, float *out_all_hidden_t_2048
) {
    if (!g_hg4.ready || n_tokens <= 0 || start_pos < 0 || start_pos + n_tokens > g_hg4.max_seq) return -1;
    g_hg4.cache_len = start_pos;
    int done = 0;
    while (done < n_tokens) {
        int chunk = (n_tokens - done > HG4_PANEL_MAX) ? HG4_PANEL_MAX : (n_tokens - done);
        float *cur = g_hg4.scr_a, *nxt = g_hg4.scr_b;
        #pragma omp parallel for schedule(static) if (chunk >= 8)
        for (int t = 0; t < chunk; t++) {
            hg4_embed_lookup_row(token_ids[done + t], cur + (size_t)t * HG4_D);
        }
        for (int l = 0; l < HG4_LAYERS; l++) {
            execute_hg4_layer_step(l, cur, nxt, g_hg4.cache_len, chunk);
            float *tmp = cur; cur = nxt; nxt = tmp;
        }
        rms_norm_f32(cur, g_hg4.final_norm, out_all_hidden_t_2048 + (size_t)done * HG4_D, chunk, HG4_D);
        g_hg4.cache_len += chunk;
        done += chunk;
    }
    return g_hg4.cache_len;
}

HG4_API int lfm3_hg4_generate(
    const int *prompt_ids, int prompt_len, int max_new_tokens, int eos_id,
    int *out_gen_ids, float *out_first_logits_128k, double *out_prefill_ms, double *out_decode_ms
) {
    if (!g_hg4.ready || prompt_len <= 0 || prompt_len + max_new_tokens >= g_hg4.max_seq) return -1;
    float *logits = (float *)malloc(HG4_VOCAB * sizeof(float));
    g_hg4.cache_len = 0;

    double t0 = omp_get_wtime();
    if (lfm3_hg4_cache_feed(prompt_ids, prompt_len, NULL, logits) < 0) {
        free(logits);
        return -1;
    }
    if (out_first_logits_128k) memcpy(out_first_logits_128k, logits, HG4_VOCAB * sizeof(float));
    *out_prefill_ms = (omp_get_wtime() - t0) * 1000.0;

    double td0 = omp_get_wtime();
    int gen_n = 0;
    for (int step = 0; step < max_new_tokens; step++) {
        int best_id = 0;
        float best_v = logits[0];
        for (int v = 1; v < HG4_VOCAB; v++) {
            if (logits[v] > best_v) { best_v = logits[v]; best_id = v; }
        }
        out_gen_ids[gen_n++] = best_id;
        if (best_id == eos_id || step + 1 >= max_new_tokens) break;
        lfm3_hg4_cache_feed(&best_id, 1, NULL, logits);
    }
    *out_decode_ms = (omp_get_wtime() - td0) * 1000.0;
    free(logits);
    return gen_n;
}

HG4_API int lfm3_hg4_forward_logits(const int *token_ids, int seq_len, float *out_logits) {
    if (!g_hg4.ready || seq_len <= 0 || seq_len > g_hg4.max_seq) return -1;
    g_hg4.cache_len = 0;
    int done = 0;
    while (done < seq_len) {
        int chunk = (seq_len - done > HG4_PANEL_MAX) ? HG4_PANEL_MAX : (seq_len - done);
        float *cur = g_hg4.scr_a, *nxt = g_hg4.scr_b;
        #pragma omp parallel for schedule(static) if (chunk >= 8)
        for (int t = 0; t < chunk; t++) {
            hg4_embed_lookup_row(token_ids[done + t], cur + (size_t)t * HG4_D);
        }
        for (int l = 0; l < HG4_LAYERS; l++) {
            execute_hg4_layer_step(l, cur, nxt, g_hg4.cache_len, chunk);
            float *tmp = cur; cur = nxt; nxt = tmp;
        }
        rms_norm_f32(cur, g_hg4.final_norm, g_hg4.scr_norm, chunk, HG4_D);
        hg4_matmul(&g_hg4.emb_mat, g_hg4.scr_norm, out_logits + (size_t)done * HG4_VOCAB, chunk, 0);
        g_hg4.cache_len += chunk;
        done += chunk;
    }
    return 0;
}

HG4_API int lfm3_hg4_test_w13(const float *x_in, int T, float *y_out) {
    if (!g_hg4.ready || T <= 0 || T > HG4_PANEL_MAX) return -1;
    hg4_matmul(&g_hg4.layers[0].w13_proj, x_in, y_out, T, 1);
    return 0;
}

