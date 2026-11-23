/*
 * Ultra-Fast OpenMP + AVX2/FMA/F16C C11 Packer for Canonical `HG4-16R` & `HG8-16R`
 * (`hg4_fast_packer.c` -> `libhg4_fast_packer.so`)
 *
 * Accelerates FWHT-128 weight rotation, 5-point Lloyd-Max FP16 scale search, 128-step
 * Cholesky LDLQ error-feedback quantization, optional dequantization, and 16-row nibble
 * interleaving by ~60-100x over Python `torch.bucketize` loops across 16 CPU cores.
 */
#ifndef _WIN32
#define _GNU_SOURCE
#include <errno.h>
#endif
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define FP_X86 1
#define FP_AVX2 __attribute__((target("avx2,fma,f16c")))
#else
#define FP_X86 0
#define FP_AVX2
#endif

static const float LUT16_S8_F[16] = {
    -127.0f, -96.0f, -75.0f, -58.0f, -44.0f, -31.0f, -18.0f, -6.0f,
       6.0f,  18.0f,  31.0f,  44.0f,  58.0f,  75.0f,  96.0f, 127.0f
};

static const float CODE_MIDS[15] = {
    -111.5f, -85.5f, -66.5f, -51.0f, -37.5f, -24.5f, -12.0f, 0.0f,
      12.0f,  24.5f,  37.5f,  51.0f,  66.5f,  85.5f, 111.5f
};

static inline uint16_t f32_to_fp16_guarded(float f) {
    float af = fabsf(f);
    if (af > 0.0f && af < 6.11e-5f) f = (f < 0.0f) ? -6.11e-5f : 6.11e-5f;
    if (af > 65504.0f) f = (f < 0.0f) ? -65504.0f : 65504.0f;
#if FP_X86 && defined(__F16C__)
    return (uint16_t)_mm_cvtsi128_si32(_mm_cvtps_ph(_mm_set_ss(f), _MM_FROUND_TO_NEAREST_INT));
#else
    uint32_t u;
    memcpy(&u, &f, 4);
    uint32_t sign = (u >> 16) & 0x8000u;
    int32_t exp = ((int32_t)((u >> 23) & 0xFFu)) - 127 + 15;
    uint32_t mant = u & 0x007FFFFFu;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7BFFu);
    uint32_t round_bit = (mant >> 12) & 1u;
    uint32_t sticky = (mant & 0xFFFu) != 0;
    uint32_t h_mant = mant >> 13;
    if (round_bit && (sticky || (h_mant & 1u))) {
        h_mant++;
        if (h_mant >= 0x400u) { h_mant = 0; exp++; if (exp >= 31) return (uint16_t)(sign | 0x7BFFu); }
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | h_mant);
#endif
}

static inline float fp16_to_f32_c(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x03FFu;
    uint32_t u = (exp == 0) ? sign : (sign | ((exp + 112u) << 23) | (mant << 13));
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static inline uint8_t lut16_nearest_idx(float x) {
    /* Exact 4-comparison binary tree over CODE_MIDS[0..14] */
    int idx = 0;
    if (x > CODE_MIDS[idx + 7]) idx += 8;
    if (x > CODE_MIDS[idx + 3]) idx += 4;
    if (x > CODE_MIDS[idx + 1]) idx += 2;
    if (x > CODE_MIDS[idx + 0]) idx += 1;
    return (uint8_t)idx;
}

FP_AVX2 static inline void fwht128_inplace(float *buf) {
    const float inv_sqrt128 = 0.08838834764831845f;
#if FP_X86
    const __m256 vnorm = _mm256_set1_ps(inv_sqrt128);
    for (int i = 0; i < 128; i += 8) {
        __m256 v = _mm256_loadu_ps(buf + i);
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
                __m256 b = _mm256_loadu_ps(buf + j + h + k);
                _mm256_storeu_ps(buf + j + k,     _mm256_add_ps(a, b));
                _mm256_storeu_ps(buf + j + h + k, _mm256_sub_ps(a, b));
            }
        }
    }
    for (int i = 0; i < 128; i += 8) {
        _mm256_storeu_ps(buf + i, _mm256_mul_ps(_mm256_loadu_ps(buf + i), vnorm));
    }
#else
    for (int h = 1; h < 128; h <<= 1) {
        for (int j = 0; j < 128; j += (h << 1)) {
            for (int k = 0; k < h; k++) {
                float a = buf[j + k], b = buf[j + h + k];
                buf[j + k] = a + b; buf[j + h + k] = a - b;
            }
        }
    }
    for (int i = 0; i < 128; i++) buf[i] *= inv_sqrt128;
#endif
}

FP_AVX2 void hg4_pack_16r_f32(
    const float *W,              /* [R, K] */
    const float *pre_diag,       /* [K] */
    const float *F_ldlq,         /* [K/128, 128, 128] or NULL */
    int R, int K,
    uint16_t *out_sc_16r,        /* [R/16, K/128, 16] */
    uint8_t  *out_pk_16r,        /* [R/16, K/128, 16, 16, 4] = R*K/2 bytes */
    float    *out_dequant        /* [R, K] or NULL */
) {
    const int nb = K / 128;
    const int n_rg = R / 16;
    float *inv_pd = (float *)malloc((size_t)K * sizeof(float));
    for (int k = 0; k < K; k++) inv_pd[k] = 1.0f / pre_diag[k];

    static const float MULTS[5] = {0.86f, 0.93f, 1.00f, 1.07f, 1.14f};

    #pragma omp parallel for schedule(static)
    for (int rg = 0; rg < n_rg; rg++) {
        uint8_t idx_blk[16][128];
        float wr[128], wq[128];

        for (int b = 0; b < nb; b++) {
            const float *ipd = inv_pd + b * 128;
            const float *pd  = pre_diag + b * 128;
            const float *Fb  = F_ldlq ? (F_ldlq + (size_t)b * 128 * 128) : NULL;
            uint16_t *sc_dst = out_sc_16r + ((size_t)rg * nb + b) * 16;

            for (int r = 0; r < 16; r++) {
                int row = rg * 16 + r;
                const float *wsrc = W + (size_t)row * K + b * 128;
                for (int i = 0; i < 128; i++) wr[i] = wsrc[i] * ipd[i];
                fwht128_inplace(wr);

                float ss = 0.0f;
                for (int i = 0; i < 128; i++) ss += wr[i] * wr[i];
                float std_b = sqrtf(ss * (1.0f / 128.0f) + 1e-12f) * (2.7326f / 127.0f);

                uint16_t best_h = f32_to_fp16_guarded(std_b);
                float best_mse = 1e30f;
                for (int m = 0; m < 5; m++) {
                    uint16_t ch = f32_to_fp16_guarded(std_b * MULTS[m]);
                    float csc = fp16_to_f32_c(ch);
                    if (csc < 1e-12f) csc = 1e-12f;
                    float icsc = 1.0f / csc;
                    float num = 0.0f, den = 0.0f;
                    for (int i = 0; i < 128; i++) {
                        float qv = LUT16_S8_F[lut16_nearest_idx(wr[i] * icsc)];
                        num += wr[i] * qv;
                        den += qv * qv;
                    }
                    uint16_t oh = f32_to_fp16_guarded(num / (den > 1e-12f ? den : 1e-12f));
                    float osc = fp16_to_f32_c(oh);
                    float mse = 0.0f;
                    float iosc = 1.0f / (osc > 1e-12f ? osc : 1e-12f);
                    for (int i = 0; i < 128; i++) {
                        float qv = LUT16_S8_F[lut16_nearest_idx(wr[i] * iosc)];
                        float d = wr[i] - qv * osc;
                        mse += d * d;
                    }
                    if (mse < best_mse) { best_mse = mse; best_h = oh; }
                }

                sc_dst[r] = best_h;
                float sc_f = fp16_to_f32_c(best_h);
                float inv_sc = (sc_f > 1e-12f) ? (1.0f / sc_f) : 0.0f;

                if (Fb) {
                    float clip = 1.05f * 127.0f * sc_f;
                    for (int k = 0; k < 128; k++) {
                        float wk = wr[k];
                        uint8_t ik = lut16_nearest_idx(wk * inv_sc);
                        idx_blk[r][k] = ik;
                        float qk = LUT16_S8_F[ik] * sc_f;
                        wq[k] = qk;
                        if (k + 1 < 128) {
                            float err = wk - qk;
                            if (err > clip) err = clip;
                            if (err < -clip) err = -clip;
                            const float *frow = Fb + k * 128;
                            int j = k + 1;
#if FP_X86
                            __m256 verr = _mm256_set1_ps(err);
                            for (; j + 8 <= 128; j += 8) {
                                __m256 wv = _mm256_loadu_ps(wr + j);
                                __m256 fv = _mm256_loadu_ps(frow + j);
                                _mm256_storeu_ps(wr + j, _mm256_fmadd_ps(verr, fv, wv));
                            }
#endif
                            for (; j < 128; j++) wr[j] += err * frow[j];
                        }
                    }
                } else {
                    for (int k = 0; k < 128; k++) {
                        uint8_t ik = lut16_nearest_idx(wr[k] * inv_sc);
                        idx_blk[r][k] = ik;
                        wq[k] = LUT16_S8_F[ik] * sc_f;
                    }
                }

                if (out_dequant) {
                    fwht128_inplace(wq);
                    float *dq_dst = out_dequant + (size_t)row * K + b * 128;
                    for (int i = 0; i < 128; i++) dq_dst[i] = wq[i] * pd[i];
                }
            }

            uint8_t *pk_dst = out_pk_16r + ((size_t)rg * nb + b) * 1024;
            for (int s = 0; s < 16; s++) {
                uint8_t *sd = pk_dst + s * 64;
                int c0 = s * 8;
                for (int r = 0; r < 16; r++) {
                    sd[r * 4 + 0] = (uint8_t)(idx_blk[r][c0 + 0] | (idx_blk[r][c0 + 4] << 4));
                    sd[r * 4 + 1] = (uint8_t)(idx_blk[r][c0 + 1] | (idx_blk[r][c0 + 5] << 4));
                    sd[r * 4 + 2] = (uint8_t)(idx_blk[r][c0 + 2] | (idx_blk[r][c0 + 6] << 4));
                    sd[r * 4 + 3] = (uint8_t)(idx_blk[r][c0 + 3] | (idx_blk[r][c0 + 7] << 4));
                }
            }
        }
    }
    free(inv_pd);
}

/* Batch MoE wrapper: packs `n_exp` experts `[n_exp, R, K]` with per-expert `pre_diag [n_exp, K]` (or shared `[1, K]`)
 * and per-expert `F_ldlq [n_exp, K/128, 128, 128]` in one OpenMP parallel loop across `n_exp * (R/16)`! */
FP_AVX2 void hg4_pack_experts_16r_f32(
    const float *W_all,          /* [n_exp, R, K] */
    const float *pre_diag_all,   /* [n_exp, K] if per_exp_pd else [K] */
    int          per_exp_pd,
    const float *F_ldlq_all,     /* [n_exp, K/128, 128, 128] or NULL */
    int n_exp, int R, int K,
    uint16_t *out_sc_all,        /* [n_exp, R/16, K/128, 16] */
    uint8_t  *out_pk_all,        /* [n_exp, R*K/2] */
    float    *out_dequant_all    /* [n_exp, R, K] or NULL */
) {
    const int nb = K / 128;
    const int n_rg = R / 16;
    const int total_work = n_exp * n_rg;
    static const float MULTS[5] = {0.86f, 0.93f, 1.00f, 1.07f, 1.14f};

    #pragma omp parallel for schedule(static)
    for (int task = 0; task < total_work; task++) {
        int e = task / n_rg;
        int rg = task % n_rg;
        const float *We = W_all + (size_t)e * R * K;
        const float *pde = per_exp_pd ? (pre_diag_all + (size_t)e * K) : pre_diag_all;
        const float *Fe = F_ldlq_all ? (F_ldlq_all + (size_t)e * nb * 128 * 128) : NULL;
        uint16_t *sce = out_sc_all + (size_t)e * R * nb;
        uint8_t  *pke = out_pk_all + (size_t)e * ((size_t)R * K / 2);
        float    *dqe = out_dequant_all ? (out_dequant_all + (size_t)e * R * K) : NULL;

        uint8_t idx_blk[16][128];
        float wr[128], wq[128], ipd[128];

        for (int b = 0; b < nb; b++) {
            const float *pd = pde + b * 128;
            for (int i = 0; i < 128; i++) ipd[i] = 1.0f / pd[i];
            const float *Fb = Fe ? (Fe + (size_t)b * 128 * 128) : NULL;
            uint16_t *sc_dst = sce + ((size_t)rg * nb + b) * 16;

            for (int r = 0; r < 16; r++) {
                int row = rg * 16 + r;
                const float *wsrc = We + (size_t)row * K + b * 128;
                for (int i = 0; i < 128; i++) wr[i] = wsrc[i] * ipd[i];
                fwht128_inplace(wr);

                float ss = 0.0f;
                for (int i = 0; i < 128; i++) ss += wr[i] * wr[i];
                float std_b = sqrtf(ss * (1.0f / 128.0f) + 1e-12f) * (2.7326f / 127.0f);

                uint16_t best_h = f32_to_fp16_guarded(std_b);
                float best_mse = 1e30f;
                for (int m = 0; m < 5; m++) {
                    uint16_t ch = f32_to_fp16_guarded(std_b * MULTS[m]);
                    float csc = fp16_to_f32_c(ch);
                    if (csc < 1e-12f) csc = 1e-12f;
                    float icsc = 1.0f / csc;
                    float num = 0.0f, den = 0.0f;
                    for (int i = 0; i < 128; i++) {
                        float qv = LUT16_S8_F[lut16_nearest_idx(wr[i] * icsc)];
                        num += wr[i] * qv;
                        den += qv * qv;
                    }
                    uint16_t oh = f32_to_fp16_guarded(num / (den > 1e-12f ? den : 1e-12f));
                    float osc = fp16_to_f32_c(oh);
                    float mse = 0.0f;
                    float iosc = 1.0f / (osc > 1e-12f ? osc : 1e-12f);
                    for (int i = 0; i < 128; i++) {
                        float qv = LUT16_S8_F[lut16_nearest_idx(wr[i] * iosc)];
                        float d = wr[i] - qv * osc;
                        mse += d * d;
                    }
                    if (mse < best_mse) { best_mse = mse; best_h = oh; }
                }

                sc_dst[r] = best_h;
                float sc_f = fp16_to_f32_c(best_h);
                float inv_sc = (sc_f > 1e-12f) ? (1.0f / sc_f) : 0.0f;

                if (Fb) {
                    float clip = 1.05f * 127.0f * sc_f;
                    for (int k = 0; k < 128; k++) {
                        float wk = wr[k];
                        uint8_t ik = lut16_nearest_idx(wk * inv_sc);
                        idx_blk[r][k] = ik;
                        float qk = LUT16_S8_F[ik] * sc_f;
                        wq[k] = qk;
                        if (k + 1 < 128) {
                            float err = wk - qk;
                            if (err > clip) err = clip;
                            if (err < -clip) err = -clip;
                            const float *frow = Fb + k * 128;
                            int j = k + 1;
#if FP_X86
                            __m256 verr = _mm256_set1_ps(err);
                            for (; j + 8 <= 128; j += 8) {
                                _mm256_storeu_ps(wr + j, _mm256_fmadd_ps(verr, _mm256_loadu_ps(frow + j), _mm256_loadu_ps(wr + j)));
                            }
#endif
                            for (; j < 128; j++) wr[j] += err * frow[j];
                        }
                    }
                } else {
                    for (int k = 0; k < 128; k++) {
                        uint8_t ik = lut16_nearest_idx(wr[k] * inv_sc);
                        idx_blk[r][k] = ik;
                        wq[k] = LUT16_S8_F[ik] * sc_f;
                    }
                }

                if (dqe) {
                    fwht128_inplace(wq);
                    float *dq_dst = dqe + (size_t)row * K + b * 128;
                    for (int i = 0; i < 128; i++) dq_dst[i] = wq[i] * pd[i];
                }
            }

            uint8_t *pk_dst = pke + ((size_t)rg * nb + b) * 1024;
            for (int s = 0; s < 16; s++) {
                uint8_t *sd = pk_dst + s * 64;
                int c0 = s * 8;
                for (int r = 0; r < 16; r++) {
                    sd[r * 4 + 0] = (uint8_t)(idx_blk[r][c0 + 0] | (idx_blk[r][c0 + 4] << 4));
                    sd[r * 4 + 1] = (uint8_t)(idx_blk[r][c0 + 1] | (idx_blk[r][c0 + 5] << 4));
                    sd[r * 4 + 2] = (uint8_t)(idx_blk[r][c0 + 2] | (idx_blk[r][c0 + 6] << 4));
                    sd[r * 4 + 3] = (uint8_t)(idx_blk[r][c0 + 3] | (idx_blk[r][c0 + 7] << 4));
                }
            }
        }
    }
}

/* ============================================================================================================
 * Canonical `HG2-16R` (`2.25 bits/weight`) 4-Level Gaussian Lloyd-Max Hadamard Packer + C11 MXFP4 Dequantizer
 * ============================================================================================================ */

static const float LUT4_S8_F[4] = { -127.0f, -38.0f, 38.0f, 127.0f };
static const float CODE4_MIDS[3] = { -82.5f, 0.0f, 82.5f };

static inline uint8_t lut4_nearest_idx(float x) {
    int idx = (x > CODE4_MIDS[1]) ? 2 : 0;
    if (x > CODE4_MIDS[idx]) idx += 1;
    return (uint8_t)idx;
}

/* Packs `n_exp` cold routed experts `[n_exp, R, K]` into `HG2-16R` (`R*K/4` bytes per expert, 512 B per 16x128 tile). */
FP_AVX2 void hg2_pack_experts_16r_f32(
    const float *W_all,          /* [n_exp, R, K] */
    const float *pre_diag_all,   /* [n_exp, K] if per_exp_pd else [K] */
    int          per_exp_pd,
    int n_exp, int R, int K,
    uint16_t *out_sc_all,        /* [n_exp, R/16, K/128, 16] */
    uint8_t  *out_pk_all         /* [n_exp, R*K/4] */
) {
    const int nb = K / 128;
    const int n_rg = R / 16;
    const int total_work = n_exp * n_rg;
    static const float MULTS[5] = {0.86f, 0.93f, 1.00f, 1.07f, 1.14f};

    #pragma omp parallel for schedule(static)
    for (int task = 0; task < total_work; task++) {
        int e = task / n_rg;
        int rg = task % n_rg;
        const float *We = W_all + (size_t)e * R * K;
        const float *pde = per_exp_pd ? (pre_diag_all + (size_t)e * K) : pre_diag_all;
        uint16_t *sce = out_sc_all + (size_t)e * R * nb;
        uint8_t  *pke = out_pk_all + (size_t)e * ((size_t)R * K / 4);

        uint8_t idx_blk[16][128];
        float wr[128], ipd[128];

        for (int b = 0; b < nb; b++) {
            const float *pd = pde + b * 128;
            for (int i = 0; i < 128; i++) ipd[i] = 1.0f / pd[i];
            uint16_t *sc_dst = sce + ((size_t)rg * nb + b) * 16;

            for (int r = 0; r < 16; r++) {
                int row = rg * 16 + r;
                const float *wsrc = We + (size_t)row * K + b * 128;
                for (int i = 0; i < 128; i++) wr[i] = wsrc[i] * ipd[i];
                fwht128_inplace(wr);

                float ss = 0.0f;
                for (int i = 0; i < 128; i++) ss += wr[i] * wr[i];
                float std_b = sqrtf(ss * (1.0f / 128.0f) + 1e-12f) * (1.5104f / 127.0f);

                uint16_t best_h = f32_to_fp16_guarded(std_b);
                float best_mse = 1e30f;
                for (int m = 0; m < 5; m++) {
                    uint16_t ch = f32_to_fp16_guarded(std_b * MULTS[m]);
                    float csc = fp16_to_f32_c(ch);
                    if (csc < 1e-12f) csc = 1e-12f;
                    float icsc = 1.0f / csc;
                    float num = 0.0f, den = 0.0f;
                    for (int i = 0; i < 128; i++) {
                        float qv = LUT4_S8_F[lut4_nearest_idx(wr[i] * icsc)];
                        num += wr[i] * qv;
                        den += qv * qv;
                    }
                    uint16_t oh = f32_to_fp16_guarded(num / (den > 1e-12f ? den : 1e-12f));
                    float osc = fp16_to_f32_c(oh);
                    float mse = 0.0f;
                    float iosc = 1.0f / (osc > 1e-12f ? osc : 1e-12f);
                    for (int i = 0; i < 128; i++) {
                        float qv = LUT4_S8_F[lut4_nearest_idx(wr[i] * iosc)];
                        float d = wr[i] - qv * osc;
                        mse += d * d;
                    }
                    if (mse < best_mse) { best_mse = mse; best_h = oh; }
                }

                sc_dst[r] = best_h;
                float sc_f = fp16_to_f32_c(best_h);
                float inv_sc = (sc_f > 1e-12f) ? (1.0f / sc_f) : 0.0f;
                for (int k = 0; k < 128; k++) idx_blk[r][k] = lut4_nearest_idx(wr[k] * inv_sc);
            }

            /* Pack 16x128 2-bit indices into 512 bytes (8 steps x 64 bytes; 4 dpbusd groups per byte!) */
            uint8_t *pk_dst = pke + ((size_t)rg * nb + b) * 512;
            for (int s = 0; s < 8; s++) {
                uint8_t *sd = pk_dst + s * 64;
                int c0 = s * 16;
                for (int r = 0; r < 16; r++) {
                    for (int j = 0; j < 4; j++) {
                        sd[r * 4 + j] = (uint8_t)(
                            idx_blk[r][c0 + j] |
                            (idx_blk[r][c0 + 4 + j] << 2) |
                            (idx_blk[r][c0 + 8 + j] << 4) |
                            (idx_blk[r][c0 + 12 + j] << 6)
                        );
                    }
                }
            }
        }
    }
}

/* OpenMP C11 OCP MXFP4 (`E2M1` + `E8M0` per 32) dequantizer -> float32 `[rows, cols]` */
void mxfp4_dequant_f32(
    const uint8_t *packed_u8,    /* [rows, cols / 2] */
    const uint8_t *scale_u8,     /* [rows, cols / 32] */
    int rows, int cols,
    float *out_f32               /* [rows, cols] */
) {
    static const float E2M1_LUT[16] = {
         0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    const int n_grp = cols / 32;
    const int half_cols = cols / 2;

    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; r++) {
        const uint8_t *pr = packed_u8 + (size_t)r * half_cols;
        const uint8_t *sr = scale_u8  + (size_t)r * n_grp;
        float *dst = out_f32 + (size_t)r * cols;
        for (int g = 0; g < n_grp; g++) {
            uint8_t se = sr[g];
            float sc;
            if (se == 255) {
                sc = 0.0f;
            } else {
                uint32_t bits = (uint32_t)se << 23;
                memcpy(&sc, &bits, 4);
            }
            const uint8_t *pg = pr + g * 16;
            float *dg = dst + g * 32;
            for (int j = 0; j < 16; j++) {
                uint8_t b = pg[j];
                dg[2 * j + 0] = E2M1_LUT[b & 0x0Fu] * sc;
                dg[2 * j + 1] = E2M1_LUT[b >> 4]    * sc;
            }
        }
    }
}

#ifndef _WIN32
#include <stdio.h>
#include <unistd.h>

static void dequant_mxfp4_serial(const uint8_t *packed_u8, const uint8_t *scale_u8, int rows, int cols, float *out_f32) {
    static const float E2M1_LUT[16] = {
         0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    const int n_grp = cols / 32;
    const int half_cols = cols / 2;
    for (int r = 0; r < rows; r++) {
        const uint8_t *pr = packed_u8 + (size_t)r * half_cols;
        const uint8_t *sr = scale_u8  + (size_t)r * n_grp;
        float *dst = out_f32 + (size_t)r * cols;
        for (int g = 0; g < n_grp; g++) {
            uint8_t se = sr[g];
            float sc = 0.0f;
            if (se != 255) { uint32_t bits = (uint32_t)se << 23; memcpy(&sc, &bits, 4); }
            const uint8_t *pg = pr + g * 16;
            float *dg = dst + g * 32;
            for (int j = 0; j < 16; j++) {
                uint8_t b = pg[j];
                dg[2 * j + 0] = E2M1_LUT[b & 0x0Fu] * sc;
                dg[2 * j + 1] = E2M1_LUT[b >> 4]    * sc;
            }
        }
    }
}

FP_AVX2 static void pack_single_mat_16r(
    const float *W, const float *pre_diag, int R, int K, int bits,
    uint16_t *out_sc, uint8_t *out_pk
) {
    const int nb = K / 128;
    const int n_rg = R / 16;
    static const float MULTS[5] = {0.86f, 0.93f, 1.00f, 1.07f, 1.14f};
    float ipd[3584];
    for (int k = 0; k < K; k++) ipd[k] = 1.0f / pre_diag[k];

    uint8_t idx_blk[16][128];
    float wr[128];

    for (int rg = 0; rg < n_rg; rg++) {
        for (int b = 0; b < nb; b++) {
            const float *ipdb = ipd + b * 128;
            uint16_t *sc_dst = out_sc + ((size_t)rg * nb + b) * 16;
            for (int r = 0; r < 16; r++) {
                const float *wsrc = W + (size_t)(rg * 16 + r) * K + b * 128;
                for (int i = 0; i < 128; i++) wr[i] = wsrc[i] * ipdb[i];
                fwht128_inplace(wr);

                float ss = 0.0f;
                for (int i = 0; i < 128; i++) ss += wr[i] * wr[i];
                float c_max = (bits == 2) ? 1.5104f : 2.7326f;
                float std_b = sqrtf(ss * (1.0f / 128.0f) + 1e-12f) * (c_max / 127.0f);
                uint16_t best_h = f32_to_fp16_guarded(std_b);
                float best_mse = 1e30f;

                for (int m = 0; m < 5; m++) {
                    uint16_t ch = f32_to_fp16_guarded(std_b * MULTS[m]);
                    float csc = fp16_to_f32_c(ch);
                    if (csc < 1e-12f) csc = 1e-12f;
                    float icsc = 1.0f / csc, num = 0.0f, den = 0.0f;
                    if (bits == 2) {
                        for (int i = 0; i < 128; i++) { float qv = LUT4_S8_F[lut4_nearest_idx(wr[i] * icsc)]; num += wr[i] * qv; den += qv * qv; }
                    } else {
                        for (int i = 0; i < 128; i++) { float qv = LUT16_S8_F[lut16_nearest_idx(wr[i] * icsc)]; num += wr[i] * qv; den += qv * qv; }
                    }
                    uint16_t oh = f32_to_fp16_guarded(num / (den > 1e-12f ? den : 1e-12f));
                    float osc = fp16_to_f32_c(oh);
                    float iosc = 1.0f / (osc > 1e-12f ? osc : 1e-12f), mse = 0.0f;
                    if (bits == 2) {
                        for (int i = 0; i < 128; i++) { float qv = LUT4_S8_F[lut4_nearest_idx(wr[i] * iosc)]; float d = wr[i] - qv * osc; mse += d * d; }
                    } else {
                        for (int i = 0; i < 128; i++) { float qv = LUT16_S8_F[lut16_nearest_idx(wr[i] * iosc)]; float d = wr[i] - qv * osc; mse += d * d; }
                    }
                    if (mse < best_mse) { best_mse = mse; best_h = oh; }
                }

                sc_dst[r] = best_h;
                float sc_f = fp16_to_f32_c(best_h);
                float inv_sc = (sc_f > 1e-12f) ? (1.0f / sc_f) : 0.0f;
                if (bits == 2) {
                    for (int k = 0; k < 128; k++) idx_blk[r][k] = lut4_nearest_idx(wr[k] * inv_sc);
                } else {
                    for (int k = 0; k < 128; k++) idx_blk[r][k] = lut16_nearest_idx(wr[k] * inv_sc);
                }
            }

            if (bits == 2) {
                uint8_t *pk_dst = out_pk + ((size_t)rg * nb + b) * 512;
                for (int s = 0; s < 8; s++) {
                    uint8_t *sd = pk_dst + s * 64;
                    int c0 = s * 16;
                    for (int r = 0; r < 16; r++)
                        for (int j = 0; j < 4; j++)
                            sd[r * 4 + j] = (uint8_t)(idx_blk[r][c0 + j] | (idx_blk[r][c0 + 4 + j] << 2) | (idx_blk[r][c0 + 8 + j] << 4) | (idx_blk[r][c0 + 12 + j] << 6));
                }
            } else {
                uint8_t *pk_dst = out_pk + ((size_t)rg * nb + b) * 1024;
                for (int s = 0; s < 16; s++) {
                    uint8_t *sd = pk_dst + s * 64;
                    int c0 = s * 8;
                    for (int r = 0; r < 16; r++) {
                        sd[r * 4 + 0] = (uint8_t)(idx_blk[r][c0 + 0] | (idx_blk[r][c0 + 4] << 4));
                        sd[r * 4 + 1] = (uint8_t)(idx_blk[r][c0 + 1] | (idx_blk[r][c0 + 5] << 4));
                        sd[r * 4 + 2] = (uint8_t)(idx_blk[r][c0 + 2] | (idx_blk[r][c0 + 6] << 4));
                        sd[r * 4 + 3] = (uint8_t)(idx_blk[r][c0 + 3] | (idx_blk[r][c0 + 7] << 4));
                    }
                }
            }
        }
    }
}

static int pwrite_all(int fd, const void *buf, size_t len, off_t off) {
    const uint8_t *p = (const uint8_t *)buf;
    size_t done = 0;
    while (done < len) {
        ssize_t w = pwrite(fd, p + done, len - done, off + (off_t)done);
        if (w < 0) {
            if (errno == EINTR) continue;
            return -1;
        }
        if (w == 0) return -1;
        done += (size_t)w;
    }
    return 0;
}

/* Fused all-in-C11 896-expert layer packer reading directly from mmap'd safetensor shard (~4.5s/layer on 16 cores!) */
FP_AVX2 int k31_pack_layer_from_safetensor_mmap(
    const uint8_t *shard_base,
    const uint64_t *exp_offsets_6, /* [896 * 6]: w1p, w1s, w3p, w3s, w2p, w2s byte offsets from shard_base */
    const int16_t *slot_map,       /* [896]: >=0 -> hot slot 0..n_hot-1, <0 -> -1 - cold slot */
    int n_hot, int n_cold,
    const float *signs_up,         /* [3584] */
    const float *signs_dn_all,     /* [896 * 3072] */
    int hot_fd, int cold_fd
) {
    if (n_hot <= 0 || n_cold < 0 || n_hot + n_cold != 896) return -3;
    const size_t hot_hdr_len = 32 + 896 * 2 + 3584 * 4;
    const size_t hot_stride  = 17043464u;
    const size_t cold_stride = 8785928u;

    /* Pass 1: Parallel reduction over hot experts to compute shared `pre_diag_up [3584]` */
    double col_sq[3584];
    memset(col_sq, 0, sizeof(col_sq));

    #pragma omp parallel
    {
        double local_sq[3584];
        memset(local_sq, 0, sizeof(local_sq));
        float *w1 = (float *)malloc((size_t)3072 * 3584 * sizeof(float));
        float *w3 = (float *)malloc((size_t)3072 * 3584 * sizeof(float));

        #pragma omp for schedule(dynamic, 1)
        for (int e = 0; e < 896; e++) {
            if (slot_map[e] < 0) continue;
            const uint64_t *offs = exp_offsets_6 + (size_t)e * 6;
            dequant_mxfp4_serial(shard_base + offs[0], shard_base + offs[1], 3072, 3584, w1);
            dequant_mxfp4_serial(shard_base + offs[2], shard_base + offs[3], 3072, 3584, w3);
            for (int r = 0; r < 3072; r++) {
                const float *r1 = w1 + (size_t)r * 3584;
                const float *r3 = w3 + (size_t)r * 3584;
                for (int c = 0; c < 3584; c++) {
                    local_sq[c] += (double)r1[c] * r1[c] + (double)r3[c] * r3[c];
                }
            }
        }
        #pragma omp critical
        {
            for (int c = 0; c < 3584; c++) col_sq[c] += local_sq[c];
        }
        free(w1); free(w3);
    }

    float pre_diag_up[3584];
    double denom = 2.0 * (double)n_hot * 3072.0;
    for (int c = 0; c < 3584; c++) {
        float rms = sqrtf((float)(col_sq[c] / denom) + 1e-12f);
        pre_diag_up[c] = sqrtf(rms) * signs_up[c];
    }

    /* Write K31H header + slot_map + pre_diag_up to hot_fd */
    uint8_t hdr[32];
    uint32_t u_vals[7] = { 896u, (uint32_t)n_hot, (uint32_t)n_cold, (uint32_t)hot_stride, (uint32_t)cold_stride, 0u, 0u };
    memcpy(hdr, "K31H", 4);
    memcpy(hdr + 4, u_vals, 28);
    if (pwrite_all(hot_fd, hdr, 32, 0) != 0) return -1;
    if (pwrite_all(hot_fd, slot_map, 896 * 2, 32) != 0) return -1;
    if (pwrite_all(hot_fd, pre_diag_up, 3584 * 4, 32 + 896 * 2) != 0) return -1;

    /* Pass 2: 16-core OpenMP parallel dequant + interleave + HG4/HG2 pack + thread-safe pwrite! */
    int any_err = 0;
    #pragma omp parallel
    {
        float   *w13  = (float *)malloc((size_t)6144 * 3584 * sizeof(float));
        float   *w2   = (float *)malloc((size_t)3584 * 3072 * sizeof(float));
        uint8_t *obuf = (uint8_t *)malloc(hot_stride);

        #pragma omp for schedule(dynamic, 1)
        for (int e = 0; e < 896; e++) {
            int cur_err = 0;
            #pragma omp atomic read
            cur_err = any_err;
            if (cur_err) continue;
            const uint64_t *offs = exp_offsets_6 + (size_t)e * 6;
            /* Dequantize w1 into first half of w2-temp? Let's dequantize row-groups directly into interleaved w13! */
            /* w1 is [3072, 3584] = 384 groups of 8 rows; w3 is [3072, 3584] = 384 groups of 8 rows */
            for (int g8 = 0; g8 < 384; g8++) {
                dequant_mxfp4_serial(
                    shard_base + offs[0] + (size_t)(g8 * 8) * 1792,
                    shard_base + offs[1] + (size_t)(g8 * 8) * 112,
                    8, 3584, w13 + (size_t)(g8 * 16 + 0) * 3584
                );
                dequant_mxfp4_serial(
                    shard_base + offs[2] + (size_t)(g8 * 8) * 1792,
                    shard_base + offs[3] + (size_t)(g8 * 8) * 112,
                    8, 3584, w13 + (size_t)(g8 * 16 + 8) * 3584
                );
            }
            dequant_mxfp4_serial(shard_base + offs[4], shard_base + offs[5], 3584, 3072, w2);

            /* Compute `glu_eq_pre_diag` for `w2` (`[3584, 3072]`) */
            float pd_dn[3072];
            const float *sdn = signs_dn_all + (size_t)e * 3072;
            double dn_sq[3072];
            memset(dn_sq, 0, sizeof(dn_sq));
            for (int r = 0; r < 3584; r++) {
                const float *wr2 = w2 + (size_t)r * 3072;
                for (int c = 0; c < 3072; c++) dn_sq[c] += (double)wr2[c] * wr2[c];
            }
            for (int k = 0; k < 3072; k++) {
                int g8 = k / 8, rem = k % 8;
                const float *gw_row = w13 + (size_t)(g8 * 16 + rem) * 3584;
                const float *uw_row = w13 + (size_t)(g8 * 16 + 8 + rem) * 3584;
                double gsq = 0.0, usq = 0.0;
                for (int c = 0; c < 3584; c++) {
                    gsq += (double)gw_row[c] * gw_row[c];
                    usq += (double)uw_row[c] * uw_row[c];
                }
                float dn_crms = sqrtf((float)(dn_sq[k] / 3584.0) + 1e-12f);
                float gu_rrms = sqrtf((float)(gsq / 3584.0) + 1e-12f) * sqrtf((float)(usq / 3584.0) + 1e-12f);
                if (gu_rrms < 1e-12f) gu_rrms = 1e-12f;
                pd_dn[k] = sqrtf(dn_crms / gu_rrms) * sdn[k];
            }

            if (slot_map[e] >= 0) {
                int hslot = (int)slot_map[e];
                uint16_t *sc_u = (uint16_t *)obuf;
                uint8_t  *pk_u = obuf + 344064u;
                uint32_t dn_hdr[2] = { 3584u, 3072u | (4u << 28) };
                memcpy(obuf + 11354112u, dn_hdr, 8);
                memcpy(obuf + 11354120u, pd_dn, 3072 * 4);
                uint16_t *sc_d = (uint16_t *)(obuf + 11366408u);
                uint8_t  *pk_d = obuf + 11538440u;

                pack_single_mat_16r(w13, pre_diag_up, 6144, 3584, 4, sc_u, pk_u);
                pack_single_mat_16r(w2,  pd_dn,       3584, 3072, 4, sc_d, pk_d);

                off_t foff = (off_t)(hot_hdr_len + (size_t)hslot * hot_stride);
                if (pwrite_all(hot_fd, obuf, hot_stride, foff) != 0) {
                    #pragma omp atomic write
                    any_err = 1;
                }
            } else {
                int cslot = -1 - (int)slot_map[e];
                uint16_t *sc_u = (uint16_t *)obuf;
                uint8_t  *pk_u = obuf + 344064u;
                uint32_t dn_hdr[2] = { 3584u, 3072u | (2u << 28) };
                memcpy(obuf + 5849088u, dn_hdr, 8);
                memcpy(obuf + 5849096u, pd_dn, 3072 * 4);
                uint16_t *sc_d = (uint16_t *)(obuf + 5861384u);
                uint8_t  *pk_d = obuf + 6033416u;

                pack_single_mat_16r(w13, pre_diag_up, 6144, 3584, 2, sc_u, pk_u);
                pack_single_mat_16r(w2,  pd_dn,       3584, 3072, 2, sc_d, pk_d);

                off_t foff = (off_t)((size_t)cslot * cold_stride);
                if (pwrite_all(cold_fd, obuf, cold_stride, foff) != 0) {
                    #pragma omp atomic write
                    any_err = 1;
                }
            }
        }
        free(w13); free(w2); free(obuf);
    }
    return any_err ? -2 : 0;
}
#endif


