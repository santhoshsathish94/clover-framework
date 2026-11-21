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
    C35_D       = 2048,
    C35_VOCAB   = 248320,
    C35_MOE_INT = 512,
    C35_TOP_K   = 8
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
    off += ((size_t)(v->R / 4) * nb * 4) * 2;
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
    if (memcmp(p, "C35E", 4) != 0) return -2;
    memcpy(&ex->k_exp, p + 4, 4);
    memcpy(&ex->exp_stride, p + 24, 4);
    ex->pre_diag_up = (const float *)(p + 28);
    ex->exp_base = p + 28 + C35_D * 4;
    return 0;
}

static struct {
    int ready;
    int isa_tier;
    uint8_t lut16_u8[16];
    float   lut16_s8_f[16];
    float   norm_w[C35_D];
    float   inv_freq[32];

    c35_mmap_t emb_mf, lm_mf, l0_mf, l3_mf;
    c35_mat_t  emb_mat, lm_mat;

    /* Layer 0: BD-GDN + MoE */
    const float *l0_in_ln, *l0_post_ln, *l0_conv_w, *l0_conv_rw;
    const float *l0_a_w, *l0_b_w, *l0_neg_exp_A, *l0_dt_bias, *l0_gdn_norm;
    const float *l0_ls, *l0_hb, *l0_hc;
    c35_mat_t      l0_qkv, l0_z, l0_out;
    c35_moe_tail_t l0_moe;
    c35_experts_t  ex0;

    /* Layer 3: Gated GQA + MoE */
    const float   *l3_in_ln, *l3_post_ln, *l3_qn, *l3_kn;
    c35_mat_t      l3_q, l3_k, l3_v, l3_o;
    c35_moe_tail_t l3_moe;
    c35_experts_t  ex3;
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
    int t = detect_c35_isa();
    if (t == 3) return "AVX-512 VNNI+VBMI (HG4/HG8-128 + Fused SwiGLU MoE + Native BD-GDN)";
    if (t == 1) return "AVX2+FMA (HG4/HG8-128 + Fused SwiGLU MoE + Native BD-GDN)";
    return "Scalar C11 (HG4/HG8-128)";
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

#if C35_X86
C35_TGT_AVX2 static void rotate_and_quantize_128_avx2(
    const float *x, const float *pre_diag, int T, int K,
    int8_t *out_xq, float *out_xsc, int32_t *out_xsum, int use_omp
) {
    const int nb = K / 128;
    const __m256 vnorm = _mm256_set1_ps(0.08838834764831845f);
    const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    const __m256 vmin_c = _mm256_set1_ps(-127.0f), vmax_c = _mm256_set1_ps(127.0f);
    const __m256i bias80 = _mm256_set1_epi8((char)0x80);
    const __m256i zero = _mm256_setzero_si256();

    #pragma omp parallel for schedule(static) if (use_omp && T >= 4)
    for (int t = 0; t < T; t++) {
        const float *xt = x + (size_t)t * K;
        int8_t  *xqt = out_xq + (size_t)t * K;
        float   *xst = out_xsc + (size_t)t * nb;
        int32_t *xmt = out_xsum + (size_t)t * nb;

        for (int b = 0; b < nb; b++) {
            float buf[128];
            const float *xb = xt + b * 128;
            const float *pd = pre_diag + b * 128;
            for (int i = 0; i < 128; i += 8) {
                __m256 v = _mm256_mul_ps(_mm256_loadu_ps(xb + i), _mm256_loadu_ps(pd + i));
                __m256 a1 = _mm256_shuffle_ps(v, v, 0xA0), b1 = _mm256_shuffle_ps(v, v, 0xF5);
                v = _mm256_blend_ps(_mm256_add_ps(a1, b1), _mm256_sub_ps(a1, b1), 0xAA);
                __m256 a2 = _mm256_shuffle_ps(v, v, 0x44), b2 = _mm256_shuffle_ps(v, v, 0xEE);
                v = _mm256_blend_ps(_mm256_add_ps(a2, b2), _mm256_sub_ps(a2, b2), 0xCC);
                __m256 hi = _mm256_permute2f128_ps(v, v, 0x11), lo = _mm256_permute2f128_ps(v, v, 0x00);
                v = _mm256_blend_ps(_mm256_add_ps(lo, hi), _mm256_sub_ps(lo, hi), 0xF0);
                _mm256_storeu_ps(buf + i, v);
            }
            for (int h = 8; h < 128; h <<= 1) {
                for (int j = 0; j < 128; j += (h << 1)) {
                    for (int k = 0; k < h; k += 8) {
                        __m256 a = _mm256_loadu_ps(buf + j + k);
                        __m256 bv = _mm256_loadu_ps(buf + j + h + k);
                        _mm256_storeu_ps(buf + j + k,     _mm256_add_ps(a, bv));
                        _mm256_storeu_ps(buf + j + h + k, _mm256_sub_ps(a, bv));
                    }
                }
            }
            __m256 vmax = _mm256_setzero_ps();
            for (int i = 0; i < 128; i += 8) {
                __m256 v = _mm256_mul_ps(_mm256_loadu_ps(buf + i), vnorm);
                _mm256_storeu_ps(buf + i, v);
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
            } else {
                xst[b] = max_abs / 127.0f;
                __m256 vinv = _mm256_set1_ps(127.0f / max_abs);
                int64_t tot_u8 = 0;
                for (int half = 0; half < 2; half++) {
                    const float *bh = buf + half * 64;
                    __m256i i0 = _mm256_cvtps_epi32(_mm256_min_ps(vmax_c, _mm256_max_ps(vmin_c, _mm256_mul_ps(_mm256_loadu_ps(bh + 0),  vinv))));
                    __m256i i1 = _mm256_cvtps_epi32(_mm256_min_ps(vmax_c, _mm256_max_ps(vmin_c, _mm256_mul_ps(_mm256_loadu_ps(bh + 8),  vinv))));
                    __m256i i2 = _mm256_cvtps_epi32(_mm256_min_ps(vmax_c, _mm256_max_ps(vmin_c, _mm256_mul_ps(_mm256_loadu_ps(bh + 16), vinv))));
                    __m256i i3 = _mm256_cvtps_epi32(_mm256_min_ps(vmax_c, _mm256_max_ps(vmin_c, _mm256_mul_ps(_mm256_loadu_ps(bh + 24), vinv))));
                    __m256i b03 = _mm256_permute4x64_epi64(_mm256_packs_epi16(
                        _mm256_permute4x64_epi64(_mm256_packs_epi32(i0, i1), 0xD8),
                        _mm256_permute4x64_epi64(_mm256_packs_epi32(i2, i3), 0xD8)), 0xD8);

                    __m256i i4 = _mm256_cvtps_epi32(_mm256_min_ps(vmax_c, _mm256_max_ps(vmin_c, _mm256_mul_ps(_mm256_loadu_ps(bh + 32), vinv))));
                    __m256i i5 = _mm256_cvtps_epi32(_mm256_min_ps(vmax_c, _mm256_max_ps(vmin_c, _mm256_mul_ps(_mm256_loadu_ps(bh + 40), vinv))));
                    __m256i i6 = _mm256_cvtps_epi32(_mm256_min_ps(vmax_c, _mm256_max_ps(vmin_c, _mm256_mul_ps(_mm256_loadu_ps(bh + 48), vinv))));
                    __m256i i7 = _mm256_cvtps_epi32(_mm256_min_ps(vmax_c, _mm256_max_ps(vmin_c, _mm256_mul_ps(_mm256_loadu_ps(bh + 56), vinv))));
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
#endif

static void c35_matmul_prequant(
    const c35_mat_t *M, const int8_t *xq, const float *xsc, const int32_t *xsum,
    float *Y, int T, int fused_swiglu, int use_omp
) {
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
#if C35_X86
    if (g_c35.isa_tier >= 1) {
        int8_t  *xq   = (int8_t *)malloc((size_t)T * K);
        float   *xsc  = (float *)malloc((size_t)T * nb * sizeof(float));
        int32_t *xsum = (int32_t *)malloc((size_t)T * nb * sizeof(int32_t));
        rotate_and_quantize_128_avx2(X, M->pre_diag, T, K, xq, xsc, xsum, use_omp);
        c35_matmul_prequant(M, xq, xsc, xsum, Y, T, fused_swiglu, use_omp);
        free(xq); free(xsc); free(xsum);
        return;
    }
#endif
    (void)X; (void)Y;
}

static void c35_embed_lookup_row(int token_id, float *dst_2048) {
    if (token_id < 0 || token_id >= C35_VOCAB) token_id = 0;
    const c35_mat_t *E = &g_c35.emb_mat;
    const int nb = C35_D / 128;
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
    float *logits = (float *)malloc((size_t)T * E * sizeof(float));

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
        /* Exact `coreset_routing_op(top_p = 0.9)` */
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
                for (int e = 0; e < E; e++) {
                    if (!picked[e] && lt[e] > best_v) { best_v = lt[e]; best_e = e; }
                }
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
            for (int e = 0; e < E; e++) {
                if (!picked[e] && lt[e] > best_v) { best_v = lt[e]; best_e = e; }
            }
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

#if C35_X86
    if (g_c35.isa_tier >= 1) {
        const int nb_up = C35_D / 128; /* 16 */
        int8_t  *x_xq   = (int8_t *)malloc((size_t)T * C35_D);
        float   *x_xsc  = (float *)malloc((size_t)T * nb_up * sizeof(float));
        int32_t *x_xsum = (int32_t *)malloc((size_t)T * nb_up * sizeof(int32_t));
        /* Rotate & quantize `x` ONCE across all active SwiGLU experts! */
        rotate_and_quantize_128_avx2(x, ex->pre_diag_up, T, C35_D, x_xq, x_xsc, x_xsum, 1);

        int active_list[256], n_act = 0;
        for (int e = 0; e < E; e++) if (active_mask[e]) active_list[n_act++] = e;

        float *exp_acc = (float *)calloc((size_t)n_act * T * C35_D, sizeof(float));
        #pragma omp parallel for schedule(dynamic, 1)
        for (int ai = 0; ai < n_act; ai++) {
            int e = active_list[ai];
            int   m_toks[256];
            float m_rw[256];
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
                c35_matmul_prequant(&gu_m, g_xq, g_xsc, g_xsum, hid, n_m, 1, 0); /* Fused SwiGLU! */
                c35_matmul(&dn_m, hid, ey, n_m, 0, 0);

                float *dst_a = exp_acc + (size_t)ai * T * C35_D;
                for (int im = 0; im < n_m; im++) {
                    float *dt = dst_a + (size_t)m_toks[im] * C35_D;
                    const float * src_e = ey + (size_t)im * C35_D;
                    float rw = m_rw[im];
                    for (int d = 0; d < C35_D; d++) dt[d] += src_e[d] * rw;
                }
                free(hid); free(ey);
            }
            free(g_xq); free(g_xsc); free(g_xsum);
        }
        for (int ai = 0; ai < n_act; ai++) {
            const float *src_a = exp_acc + (size_t)ai * T * C35_D;
            for (int i = 0; i < T * C35_D; i++) y_out[i] += src_a[i];
        }
        free(exp_acc);
        free(x_xq); free(x_xsc); free(x_xsum);
    }
#endif

    /* Gated Shared SwiGLU Expert */
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

/* Layer 0: Bidirectional Gated DeltaNet (`BD-GDN`) + SwiGLU MoE */
static void c35_forward_layer0_bdgdn(const float *x, float *h_out, int T, int preview) {
    float *u       = (float *)malloc((size_t)T * C35_D * sizeof(float));
    float *raw_qkv = (float *)malloc((size_t)T * 8192 * sizeof(float));
    float *mixed   = (float *)malloc((size_t)T * 8192 * sizeof(float));
    float *z       = (float *)malloc((size_t)T * 4096 * sizeof(float));
    float *y_attn  = (float *)malloc((size_t)T * 4096 * sizeof(float));
    float *h_attn  = (float *)malloc((size_t)T * C35_D * sizeof(float));
    float *u_mlp   = (float *)malloc((size_t)T * C35_D * sizeof(float));
    float *moe_out = (float *)malloc((size_t)T * C35_D * sizeof(float));

    rms_norm_batch(x, g_c35.l0_in_ln, u, T, C35_D, 1e-6f);
    c35_matmul(&g_c35.l0_qkv, u, raw_qkv, T, 0, 1);
    c35_matmul(&g_c35.l0_z,   u, z,       T, 0, 1);

    #pragma omp parallel for schedule(static)
    for (int c = 0; c < 8192; c++) {
        const float *cw = g_c35.l0_conv_w + c * 4;
        const float *crw = g_c35.l0_conv_rw + c * 3;
        for (int t = 0; t < T; t++) {
            float acc = 0.0f;
            for (int tap = 0; tap < 4; tap++) {
                int ts = t - 3 + tap;
                if (ts >= 0) acc += raw_qkv[(size_t)ts * 8192 + c] * cw[tap];
            }
            if (preview) {
                for (int dist = 1; dist <= 3 && t + dist < T; dist++) {
                    acc += raw_qkv[(size_t)(t + dist) * 8192 + c] * crw[dist - 1];
                }
            }
            mixed[(size_t)t * 8192 + c] = acc / (1.0f + expf(-acc));
        }
    }

    float *g_val    = (float *)malloc((size_t)T * 32 * sizeof(float));
    float *beta_val = (float *)malloc((size_t)T * 32 * sizeof(float));
    for (int t = 0; t < T; t++) {
        const float *ut = u + (size_t)t * C35_D;
        for (int h = 0; h < 32; h++) {
            const float *aw = g_c35.l0_a_w + (size_t)h * C35_D;
            const float *bw = g_c35.l0_b_w + (size_t)h * C35_D;
            float da = 0.0f, db = 0.0f;
            #pragma omp simd reduction(+:da,db)
            for (int d = 0; d < C35_D; d++) { da += ut[d] * aw[d]; db += ut[d] * bw[d]; }
            float sp = da + g_c35.l0_dt_bias[h];
            sp = (sp > 20.0f) ? sp : log1pf(expf(sp));
            g_val[t * 32 + h] = g_c35.l0_neg_exp_A[h] * sp;
            beta_val[t * 32 + h] = 1.0f / (1.0f + expf(-db));
        }
    }

    const float inv_sqrt128 = 1.0f / sqrtf(128.0f);
    #pragma omp parallel for schedule(static)
    for (int vh = 0; vh < 32; vh++) {
        int kh = vh / 2;
        if (!preview) {
            /* Causal DeltaNet Commit (`32 x 128 x 128` fast in-register/L2 recurrence) */
            float st[128 * 128];
            memset(st, 0, sizeof(st));
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
            /* Bidirectional `bd_gdn_local_memory` Preview */
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
            float l_scale = g_c35.l0_ls[vh];
            float h_logit = g_c35.l0_hb[vh] + g_c35.l0_hc[vh] * logf((float)T / 32.0f);
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

        /* Per-head RMSNorm(128, zero_centered=False) * SiLU(z) */
        for (int t = 0; t < T; t++) {
            float *yt = y_attn + (size_t)t * 4096 + vh * 128;
            const float *zt = z + (size_t)t * 4096 + vh * 128;
            double ss = 0.0;
            for (int d = 0; d < 128; d++) ss += (double)yt[d] * yt[d];
            float inv = 1.0f / sqrtf((float)(ss / 128.0) + 1e-6f);
            for (int d = 0; d < 128; d++) {
                float zg = zt[d] / (1.0f + expf(-zt[d]));
                yt[d] = (yt[d] * inv * g_c35.l0_gdn_norm[d]) * zg;
            }
        }
    }
    free(g_val); free(beta_val);

    c35_matmul(&g_c35.l0_out, y_attn, h_attn, T, 0, 1);
    for (int i = 0; i < T * C35_D; i++) h_attn[i] += x[i];

    rms_norm_batch(h_attn, g_c35.l0_post_ln, u_mlp, T, C35_D, 1e-6f);
    c35_run_moe(u_mlp, moe_out, T, &g_c35.l0_moe, &g_c35.ex0, preview);
    for (int i = 0; i < T * C35_D; i++) h_out[i] = h_attn[i] + moe_out[i];

    free(u); free(raw_qkv); free(mixed); free(z); free(y_attn); free(h_attn); free(u_mlp); free(moe_out);
}

/* Layer 3: Gated Full-Attention (`16Q/2KV`, `head_dim = 256`, Partial RoPE `0.25`) + SwiGLU MoE */
static void c35_forward_layer3_gqa(const float *x, float *h_out, int T, int preview) {
    float *u       = (float *)malloc((size_t)T * C35_D * sizeof(float));
    float *qg      = (float *)malloc((size_t)T * 8192 * sizeof(float)); /* [T, 16, 512] = [256 q | 256 gate] */
    float *k       = (float *)malloc((size_t)T * 512 * sizeof(float));  /* [T, 2, 256] */
    float *v       = (float *)malloc((size_t)T * 512 * sizeof(float));  /* [T, 2, 256] */
    float *y_attn  = (float *)malloc((size_t)T * 4096 * sizeof(float));
    float *h_attn  = (float *)malloc((size_t)T * C35_D * sizeof(float));
    float *u_mlp   = (float *)malloc((size_t)T * C35_D * sizeof(float));
    float *moe_out = (float *)malloc((size_t)T * C35_D * sizeof(float));

    rms_norm_batch(x, g_c35.l3_in_ln, u, T, C35_D, 1e-6f);
    c35_matmul(&g_c35.l3_q, u, qg, T, 0, 1);
    c35_matmul(&g_c35.l3_k, u, k,  T, 0, 1);
    c35_matmul(&g_c35.l3_v, u, v,  T, 0, 1);

    #pragma omp parallel for schedule(static) if (T >= 4)
    for (int t = 0; t < T; t++) {
        float cs[32], sn[32];
        for (int d = 0; d < 32; d++) {
            float ang = (float)t * g_c35.inv_freq[d];
            cs[d] = cosf(ang); sn[d] = sinf(ang);
        }
        for (int h = 0; h < 16; h++) {
            float *qh = qg + (size_t)t * 8192 + h * 512;
            rms_norm_row(qh, g_c35.l3_qn, qh, 256, 1e-6f);
            for (int d = 0; d < 32; d++) {
                float f1 = qh[d], f2 = qh[32 + d];
                qh[d]      = f1 * cs[d] - f2 * sn[d];
                qh[32 + d] = f2 * cs[d] + f1 * sn[d];
            }
        }
        for (int kh = 0; kh < 2; kh++) {
            float *kh_p = k + (size_t)t * 512 + kh * 256;
            rms_norm_row(kh_p, g_c35.l3_kn, kh_p, 256, 1e-6f);
            for (int d = 0; d < 32; d++) {
                float f1 = kh_p[d], f2 = kh_p[32 + d];
                kh_p[d]      = f1 * cs[d] - f2 * sn[d];
                kh_p[32 + d] = f2 * cs[d] + f1 * sn[d];
            }
        }
    }

    const float scale = 1.0f / 16.0f; /* 1 / sqrt(256) */
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < 16; h++) {
        int kv_h = h / 8;
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
            for (int d = 0; d < 256; d++) {
                yh[d] *= 1.0f / (1.0f + expf(-gh[d]));
            }
        }
        free(scores);
    }

    c35_matmul(&g_c35.l3_o, y_attn, h_attn, T, 0, 1);
    for (int i = 0; i < T * C35_D; i++) h_attn[i] += x[i];

    rms_norm_batch(h_attn, g_c35.l3_post_ln, u_mlp, T, C35_D, 1e-6f);
    c35_run_moe(u_mlp, moe_out, T, &g_c35.l3_moe, &g_c35.ex3, preview);
    for (int i = 0; i < T * C35_D; i++) h_out[i] = h_attn[i] + moe_out[i];

    free(u); free(qg); free(k); free(v); free(y_attn); free(h_attn); free(u_mlp); free(moe_out);
}

C35_API int clover35b_hg4_init(const char *hg4_dir) {
    if (g_c35.ready) return 0;
    g_c35.isa_tier = detect_c35_isa();

    char path[1024];
    snprintf(path, sizeof(path), "%s/meta.bin", hg4_dir);
    FILE *mf = fopen(path, "rb");
    if (!mf) return -1;
    char magic[4];
    uint32_t hdr[3];
    if (fread(magic, 1, 4, mf) != 4 || memcmp(magic, "C35M", 4) != 0 ||
        fread(hdr, 4, 3, mf) != 3 ||
        fread(g_c35.lut16_u8, 1, 16, mf) != 16 ||
        fread(g_c35.norm_w, 4, C35_D, mf) != C35_D ||
        fread(g_c35.inv_freq, 4, 32, mf) != 32) { fclose(mf); return -2; }
    fclose(mf);
    for (int i = 0; i < 16; i++) g_c35.lut16_s8_f[i] = (float)((int)g_c35.lut16_u8[i] - 128);

    snprintf(path, sizeof(path), "%s/embed_tokens.hg4", hg4_dir);
    if (c35_mmap_open(path, &g_c35.emb_mf) != 0) return -3;
    parse_c35_mat((const uint8_t *)g_c35.emb_mf.ptr, &g_c35.emb_mat);

    snprintf(path, sizeof(path), "%s/lm_head.hg4", hg4_dir);
    if (c35_mmap_open(path, &g_c35.lm_mf) != 0) return -4;
    parse_c35_mat((const uint8_t *)g_c35.lm_mf.ptr, &g_c35.lm_mat);

    snprintf(path, sizeof(path), "%s/layer0_bdgdn.hg4", hg4_dir);
    if (c35_mmap_open(path, &g_c35.l0_mf) != 0) return -5;
    {
        const uint8_t *p = (const uint8_t *)g_c35.l0_mf.ptr;
        size_t off = 0;
        g_c35.l0_in_ln     = (const float *)(p + off); off += C35_D * 4;
        g_c35.l0_post_ln   = (const float *)(p + off); off += C35_D * 4;
        g_c35.l0_conv_w    = (const float *)(p + off); off += 8192 * 4 * 4;
        g_c35.l0_conv_rw   = (const float *)(p + off); off += 8192 * 3 * 4;
        g_c35.l0_a_w       = (const float *)(p + off); off += 32 * C35_D * 4;
        g_c35.l0_b_w       = (const float *)(p + off); off += 32 * C35_D * 4;
        g_c35.l0_neg_exp_A = (const float *)(p + off); off += 32 * 4;
        g_c35.l0_dt_bias   = (const float *)(p + off); off += 32 * 4;
        g_c35.l0_gdn_norm  = (const float *)(p + off); off += 128 * 4;
        g_c35.l0_ls        = (const float *)(p + off); off += 32 * 4;
        g_c35.l0_hb        = (const float *)(p + off); off += 32 * 4;
        g_c35.l0_hc        = (const float *)(p + off); off += 32 * 4;
        uint32_t l_qkv, l_z, l_out, l_moe;
        memcpy(&l_qkv, p + off, 4); off += 4;
        memcpy(&l_z,   p + off, 4); off += 4;
        memcpy(&l_out, p + off, 4); off += 4;
        memcpy(&l_moe, p + off, 4); off += 4;
        parse_c35_mat(p + off, &g_c35.l0_qkv); off += l_qkv;
        parse_c35_mat(p + off, &g_c35.l0_z);   off += l_z;
        parse_c35_mat(p + off, &g_c35.l0_out); off += l_out;
        parse_c35_moe_tail(p + off, &g_c35.l0_moe);
    }

    snprintf(path, sizeof(path), "%s/experts0.hg4", hg4_dir);
    if (open_c35_experts(path, &g_c35.ex0) != 0) return -6;

    snprintf(path, sizeof(path), "%s/layer3_gqa.hg4", hg4_dir);
    if (c35_mmap_open(path, &g_c35.l3_mf) != 0) return -7;
    {
        const uint8_t *p = (const uint8_t *)g_c35.l3_mf.ptr;
        size_t off = 0;
        g_c35.l3_in_ln   = (const float *)(p + off); off += C35_D * 4;
        g_c35.l3_post_ln = (const float *)(p + off); off += C35_D * 4;
        g_c35.l3_qn      = (const float *)(p + off); off += 256 * 4;
        g_c35.l3_kn      = (const float *)(p + off); off += 256 * 4;
        uint32_t l_q, l_k, l_v, l_o, l_moe;
        memcpy(&l_q,   p + off, 4); off += 4;
        memcpy(&l_k,   p + off, 4); off += 4;
        memcpy(&l_v,   p + off, 4); off += 4;
        memcpy(&l_o,   p + off, 4); off += 4;
        memcpy(&l_moe, p + off, 4); off += 4;
        parse_c35_mat(p + off, &g_c35.l3_q); off += l_q;
        parse_c35_mat(p + off, &g_c35.l3_k); off += l_k;
        parse_c35_mat(p + off, &g_c35.l3_v); off += l_v;
        parse_c35_mat(p + off, &g_c35.l3_o); off += l_o;
        parse_c35_moe_tail(p + off, &g_c35.l3_moe);
    }

    snprintf(path, sizeof(path), "%s/experts3.hg4", hg4_dir);
    if (open_c35_experts(path, &g_c35.ex3) != 0) return -8;

    g_c35.ready = 1;
    return 0;
}

C35_API float clover35b_hg4_forward_ids(const int *token_ids, int T, int preview, float *out_final_hidden) {
    if (!g_c35.ready || T <= 0) return -1.0f;
    double t0 = omp_get_wtime();
    float *h0 = (float *)malloc((size_t)T * C35_D * sizeof(float));
    float *h1 = (float *)malloc((size_t)T * C35_D * sizeof(float));
    float *h3 = (float *)malloc((size_t)T * C35_D * sizeof(float));

    #pragma omp parallel for schedule(static) if (T >= 4)
    for (int t = 0; t < T; t++) {
        c35_embed_lookup_row(token_ids[t], h0 + (size_t)t * C35_D);
    }
    c35_forward_layer0_bdgdn(h0, h1, T, preview);
    c35_forward_layer3_gqa(h1, h3, T, preview);
    rms_norm_batch(h3, g_c35.norm_w, out_final_hidden, T, C35_D, 1e-6f);

    free(h0); free(h1); free(h3);
    return (float)((omp_get_wtime() - t0) * 1000.0);
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
