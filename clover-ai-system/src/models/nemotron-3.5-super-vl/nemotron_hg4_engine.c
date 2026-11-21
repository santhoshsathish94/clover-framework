/*
 * Native C11 / OpenMP `HG4-128` Engine for NVIDIA Nemotron 3.5 Super VL (`120B-A12B` MoE, `~3.36 GiB`)
 * (`clover-ai-system/src/models/nemotron-3.5-super-vl/nemotron_hg4_engine.c`)
 *
 * Drop-in `nemo_*` C ABI replacement for `nemotron_cpu_engine.c` executing 100% of Nemotron 3.5's
 * trunks, embeddings, `131,072`-vocab `lm_head`, `512` routed `ReLU^2` LatentMoE experts, NoPE 16:1 GQA,
 * Mamba-2 SSM, and 2-layer Multi-Token Prediction (`MTP`) speculative drafter in `HG4-128` (`4.14 bpw`):
 *   1. Column-Balanced `FWHT-128` Incoherence + 15-Point Lloyd-Max Gaussian `LUT16` (`vpermb` / `vpshufb` / `vqtbl1q`)
 *   2. Shared-Input Incoherence across all 512 routed MoE `up_proj` matrices (`latent_in` rotated ONCE per token!)
 *   3. L1-Resident Single-Barrier Parallel Expert Execution (`up_proj -> ReLU^2 -> down_proj` per core when `T <= 2`)
 *   4. Fused `Q+K+V` (`[4608, 4096]`) NoPE 16:1 GQA + 1-Row Inverse-`FWHT-128` `131K` Token Embedding Lookup
 *   5. SpecMD Policy 3 (`Least-Stale + R4 Clean-Drop`) Expert SlotCache & Data-Movement Ledger preserved intact.
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
#include <time.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#include <cpuid.h>
#define NHG4_X86 1
#define NHG4_ARM64 0
#define NHG4_TGT_AVX2       __attribute__((target("avx2,fma,f16c")))
#define NHG4_TGT_AVXVNNI    __attribute__((target("avx2,fma,f16c,avxvnni")))
#define NHG4_TGT_AVX512VNNI __attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,avx512vnni,avx512vbmi,fma,f16c")))
#elif defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#define NHG4_X86 0
#define NHG4_ARM64 1
#define NHG4_TGT_AVX2
#define NHG4_TGT_AVXVNNI
#define NHG4_TGT_AVX512VNNI
#else
#define NHG4_X86 0
#define NHG4_ARM64 0
#define NHG4_TGT_AVX2
#define NHG4_TGT_AVXVNNI
#define NHG4_TGT_AVX512VNNI
#endif

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#define NEMO_API __declspec(dllexport)
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#define NEMO_API __attribute__((visibility("default")))
#endif

enum {
    NEMO_D            = 4096,
    NEMO_VOCAB        = 131072,
    NEMO_EXPERTS      = 512,
    NEMO_TOP_K        = 22,
    NEMO_LATENT       = 1024,
    NEMO_MOE_INTER    = 2688,
    NEMO_SH_INTER     = 5376,
    NEMO_MAMBA_IN     = 18560,
    NEMO_MAMBA_CONV   = 10240,
    NEMO_MAMBA_INNER  = 8192,
    NEMO_MAMBA_HEADS  = 128,
    NEMO_MAMBA_HDIM   = 64,
    NEMO_MAMBA_DSTATE = 128,
    NEMO_GQA_Q        = 4096,
    NEMO_GQA_KV       = 256,
    NEMO_GQA_QKV      = 4608,
    NEMO_GQA_HEADS    = 32,
    NEMO_GQA_HDIM     = 128,
    NEMO_MAX_K        = 8192,
    NEMO_MAX_LAYERS   = 96,
    NEMO_MAX_SEQ      = 4096,
    NEMO_PANEL_MAX    = 256
};

static inline uint16_t f32_to_bf16_u16(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

static inline float bf16_u16_to_f32(uint16_t b) {
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

static inline float bf16_round_f32(float x) {
    uint32_t u;
    memcpy(&u, &x, 4);
    if ((u & 0x7FFFFFFFu) <= 0x7F800000u) {
        u += 0x7FFFu + ((u >> 16) & 1u);
        u &= 0xFFFF0000u;
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
} nhg4_mmap_t;

static int nhg4_mmap_open(const char *path, nhg4_mmap_t *m) {
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

static void nhg4_mmap_close(nhg4_mmap_t *m) {
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
    const float    *pre_diag;    /* [K] */
    const uint16_t *scales_fp16; /* HG4P: [R/16, K/128, 16]; HG4N: [R/4, K/128, 4] */
    const uint8_t  *tiles;       /* HG4P: [R/16, K/128, ...]; HG4N: [R/4, K/128, ...] */
} nhg4_mat_t;

static void parse_nhg4_mat(const uint8_t *blob, nhg4_mat_t *v) {
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
    const float *norm_w;
    const float *cw, *cb, *dtb, *A, *D, *mix_norm;
    nhg4_mat_t   zxbc_proj, dt_proj, out_proj;
} nhg4_mamba_t;

typedef struct {
    const float *norm_w;     /* [4096] */
    nhg4_mat_t   q_proj;     /* [4096, 4096] */
    nhg4_mat_t   k_proj;     /* [256, 4096] */
    nhg4_mat_t   v_proj;     /* [256, 4096] */
    nhg4_mat_t   o_proj;     /* [4096, 4096] */
} nhg4_gqa_t;

typedef struct {
    const float *norm_w;     /* [4096] */
    const float *gate_w;     /* [512 * 4096] */
    const float *gate_b;     /* [512] */
    nhg4_mat_t   fc1;        /* [1024, 4096] */
    nhg4_mat_t   fc2;        /* [4096, 1024] */
    nhg4_mat_t   sh_up;      /* [5376, 4096] */
    nhg4_mat_t   sh_dn;      /* [4096, 5376] */
} nhg4_moe_trunk_t;

typedef struct {
    nhg4_mmap_t  mf;
    int          is_16r;
    uint32_t     n_exp;
    uint32_t     exp_stride;
    const float *pre_diag_up; /* [1024] */
    const uint8_t *exp_base;  /* [512 * exp_stride] */
} nhg4_experts_t;

static void parse_mamba_blob(const uint8_t *p, size_t off, nhg4_mamba_t *m) {
    m->norm_w   = (const float *)(p + off); off += NEMO_D * 4;
    m->cw       = (const float *)(p + off); off += NEMO_MAMBA_CONV * 4 * 4;
    m->cb       = (const float *)(p + off); off += NEMO_MAMBA_CONV * 4;
    m->dtb      = (const float *)(p + off); off += NEMO_MAMBA_HEADS * 4;
    m->A        = (const float *)(p + off); off += NEMO_MAMBA_HEADS * 4;
    m->D        = (const float *)(p + off); off += NEMO_MAMBA_HEADS * 4;
    m->mix_norm = (const float *)(p + off); off += NEMO_MAMBA_INNER * 4;
    uint32_t l_zxbc, l_dt, l_out;
    memcpy(&l_zxbc, p + off, 4); off += 4;
    memcpy(&l_dt,   p + off, 4); off += 4;
    memcpy(&l_out,  p + off, 4); off += 4;
    parse_nhg4_mat(p + off, &m->zxbc_proj); off += l_zxbc;
    parse_nhg4_mat(p + off, &m->dt_proj);   off += l_dt;
    parse_nhg4_mat(p + off, &m->out_proj);
}

static void parse_gqa_blob_off(const uint8_t *p, size_t off, nhg4_gqa_t *g) {
    g->norm_w = (const float *)(p + off); off += NEMO_D * 4;
    uint32_t l_q, l_k, l_v, l_o;
    memcpy(&l_q, p + off, 4); off += 4;
    memcpy(&l_k, p + off, 4); off += 4;
    memcpy(&l_v, p + off, 4); off += 4;
    memcpy(&l_o, p + off, 4); off += 4;
    parse_nhg4_mat(p + off, &g->q_proj); off += l_q;
    parse_nhg4_mat(p + off, &g->k_proj); off += l_k;
    parse_nhg4_mat(p + off, &g->v_proj); off += l_v;
    parse_nhg4_mat(p + off, &g->o_proj);
}

static void parse_moe_trunk_blob_off(const uint8_t *p, size_t off, nhg4_moe_trunk_t *m) {
    m->norm_w = (const float *)(p + off); off += NEMO_D * 4;
    m->gate_w = (const float *)(p + off); off += (size_t)NEMO_EXPERTS * NEMO_D * 4;
    m->gate_b = (const float *)(p + off); off += NEMO_EXPERTS * 4;
    uint32_t l1, l2, l3, l4;
    memcpy(&l1, p + off, 4); off += 4;
    memcpy(&l2, p + off, 4); off += 4;
    memcpy(&l3, p + off, 4); off += 4;
    memcpy(&l4, p + off, 4); off += 4;
    parse_nhg4_mat(p + off, &m->fc1);   off += l1;
    parse_nhg4_mat(p + off, &m->fc2);   off += l2;
    parse_nhg4_mat(p + off, &m->sh_up); off += l3;
    parse_nhg4_mat(p + off, &m->sh_dn);
}

static int open_experts_hg4(const char *path, nhg4_experts_t *ex) {
    if (nhg4_mmap_open(path, &ex->mf) != 0) return -1;
    const uint8_t *p = (const uint8_t *)ex->mf.ptr;
    if (memcmp(p, "HG4F", 4) == 0) ex->is_16r = 1;
    else if (memcmp(p, "HG4E", 4) == 0) ex->is_16r = 0;
    else return -2;
    memcpy(&ex->n_exp, p + 4, 4);
    memcpy(&ex->exp_stride, p + 24, 4);
    ex->pre_diag_up = (const float *)(p + 28);
    ex->exp_base = p + 28 + NEMO_LATENT * 4;
    return 0;
}

typedef struct {
    uint32_t         kind; /* 0 = Mamba-2, 1 = LatentMoE, 2 = NoPE GQA */
    nhg4_mmap_t      mf;
    nhg4_mamba_t     mamba;
    nhg4_moe_trunk_t moe;
    nhg4_experts_t   experts;
    nhg4_gqa_t       gqa;
    /* Stateful caches */
    float           *ssm_state;    /* [128 * 64 * 128] when kind == 0 */
    float           *conv_hist;    /* [3 * 10240] when kind == 0 */
    float           *ssm_snaps;    /* [4 * 128 * 64 * 128] lazy speculative rollback snapshots */
    float           *conv_snaps;   /* [4 * 3 * 10240] lazy speculative rollback snapshots */
    uint16_t        *k_cache_bf16; /* [NEMO_MAX_SEQ * 256] when kind == 2 */
    uint16_t        *v_cache_bf16; /* [NEMO_MAX_SEQ * 256] when kind == 2 */
} nhg4_layer_t;

static struct {
    int initialized;
    int is_16r;
    int isa_tier;
    int n_layers;
    int cache_len;
    int madvise_active;
    char loaded_dir[512];

    uint8_t lut16_u8[16];
    int8_t  lut16_s8[16];
    float   lut16_s8_f[16];
    float   norm_f[NEMO_D];

    nhg4_mmap_t  emb_mf, lm_mf, mtp_mf;
    nhg4_mat_t   emb_mat, lm_mat;
    nhg4_layer_t layers[NEMO_MAX_LAYERS];

    /* Legacy 3-layer stage aliases (`layers[0]`, `layers[1]`, `layers[2]`) */
    int idx_mamba0, idx_moe1, idx_gqa7;

    /* 2-Layer MTP Drafter */
    int has_mtp;
    int mtp_cache_len;
    int spec_save_snaps;
    const float *mtp_enorm, *mtp_hnorm, *mtp_final_ln;
    nhg4_mat_t       mtp_eh_proj;
    nhg4_gqa_t       mtp0_gqa;
    nhg4_moe_trunk_t mtp1_moe;
    nhg4_experts_t   experts_mtp1;
    uint16_t        *mtp_k_cache_bf16;
    uint16_t        *mtp_v_cache_bf16;

    /* Preallocated hot-path scratch buffers */
    int8_t  *scr_xq;
    float   *scr_xsc;
    int32_t *scr_xsum;
    float   *scr_moe_inter; /* [NEMO_TOP_K * NEMO_MOE_INTER] */
    float   *scr_moe_out;   /* [NEMO_TOP_K * NEMO_LATENT] */

    /* SpecMD Policy 3 SlotCache + Data-Movement Ledger */
    unsigned long clock;
    int pool_keys[64];
    unsigned long pool_ts[64];
    float pool_heat[64];
    int host_keys[512];
    unsigned long total_requests, tier1_pool_hits, tier2_host_hits, tier3_drive_fetches, r4_clean_drops, dirty_writebacks;
    unsigned long long bytes_tier0_reg, bytes_tier1_l3_vcache, bytes_tier2_ddr5, bytes_saved_q4_and_broadcast;
    unsigned long madvise_willneed_calls, madvise_dontneed_calls;
} g_nhg4;

static int detect_nhg4_isa(void) {
#if NHG4_X86
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
#endif
    return 0;
}

NEMO_API const char *nemo_hg4_isa_name(void) {
    int t = g_nhg4.initialized ? g_nhg4.isa_tier : detect_nhg4_isa();
    if (t == 3) return "AVX-512 VNNI+VBMI (HG4/HG8-16R + Shrinkage-LDLQ + Int32-FWHT + 512-MoE)";
    if (t == 2) return "AVX-VNNI+FMA (HG4/HG8-16R + Int32-FWHT + 512-MoE)";
    if (t == 1) return "AVX2+FMA (HG4/HG8-16R + abs/sign LUT16 + Int32-FWHT + 512-MoE)";
    return "Scalar C11 (HG4/HG8-16R + Int32-FWHT + 512-MoE)";
}

NEMO_API int nemo_num_layers(void) {
    return g_nhg4.n_layers;
}

NEMO_API void nemo_set_isa_tier(int tier) {
    int max_hw = detect_nhg4_isa();
    if (tier < 0) g_nhg4.isa_tier = max_hw;
    else if (tier == 0) g_nhg4.isa_tier = 0;
#if NHG4_X86
    else if (tier == 1 && max_hw >= 1) g_nhg4.isa_tier = 1;
    else if (tier == 3 && max_hw == 3) g_nhg4.isa_tier = 3;
#endif
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

#if NHG4_X86
NHG4_TGT_AVX2 static void rotate_and_quantize_128_avx2(
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
                    #define NHG4_Q8(off) _mm256_cvtps_epi32(_mm256_min_ps(vmax_c, _mm256_max_ps(vmin_c, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_loadu_si256((const __m256i *)(rh + (off)))), vinv))))
                    __m256i i0 = NHG4_Q8(0),  i1 = NHG4_Q8(8),  i2 = NHG4_Q8(16), i3 = NHG4_Q8(24);
                    __m256i i4 = NHG4_Q8(32), i5 = NHG4_Q8(40), i6 = NHG4_Q8(48), i7 = NHG4_Q8(56);
                    #undef NHG4_Q8
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

NHG4_TGT_AVX512VNNI static inline __m128i hsum4_zmm_epi32(__m512i z0, __m512i z1, __m512i z2, __m512i z3) {
    __m256i y0 = _mm256_add_epi32(_mm512_castsi512_si256(z0), _mm512_extracti32x8_epi32(z0, 1));
    __m256i y1 = _mm256_add_epi32(_mm512_castsi512_si256(z1), _mm512_extracti32x8_epi32(z1, 1));
    __m256i y2 = _mm256_add_epi32(_mm512_castsi512_si256(z2), _mm512_extracti32x8_epi32(z2, 1));
    __m256i y3 = _mm256_add_epi32(_mm512_castsi512_si256(z3), _mm512_extracti32x8_epi32(z3, 1));
    __m256i h01 = _mm256_hadd_epi32(y0, y1);
    __m256i h23 = _mm256_hadd_epi32(y2, y3);
    __m256i h0123 = _mm256_hadd_epi32(h01, h23);
    return _mm_add_epi32(_mm256_castsi256_si128(h0123), _mm256_extracti128_si256(h0123, 1));
}

NHG4_TGT_AVX512VNNI static void nhg4_tile4_avx512(
    const nhg4_mat_t *M, const uint8_t *lut16_u8,
    const int8_t *X_q, const float *X_sc, const int32_t *X_sum,
    float *Y, int T, int tile_idx, int apply_relu2
) {
    const int R = (int)M->R, K = (int)M->K, nb = K / 128;
    const int o_base = tile_idx * 4;
    const uint8_t *wt = M->tiles + (size_t)tile_idx * (size_t)nb * 256;
    const uint16_t *wsc = M->scales_fp16 + (size_t)tile_idx * (size_t)nb * 4;
    const __m512i vlut = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)lut16_u8));
    const __m256i m0f  = _mm256_set1_epi8(0x0F);

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
            float s[4];
            _mm_storeu_ps(s, vs[ti]);
            float *yt = Y + (size_t)(t0 + ti) * R + o_base;
            if (apply_relu2) {
                for (int r = 0; r < 4; r++) {
                    float v = bf16_round_f32(s[r]);
                    v = v > 0.0f ? v : 0.0f;
                    yt[r] = bf16_round_f32(v * v);
                }
            } else {
                for (int r = 0; r < 4; r++) yt[r] = bf16_round_f32(s[r]);
            }
        }
    }
}

/* Direct VNNI HG8-128 4-Row Tile Kernel (Zero vpermb / Zero Nibble Shift!) */
NHG4_TGT_AVX512VNNI static void nhg8_tile4_avx512(
    const nhg4_mat_t *M,
    const int8_t *X_q, const float *X_sc, const int32_t *X_sum,
    float *Y, int T, int tile_idx, int apply_relu2
) {
    const int R = (int)M->R, K = (int)M->K, nb = K / 128;
    const int o_base = tile_idx * 4;
    const uint8_t *wt = M->tiles + (size_t)tile_idx * (size_t)nb * 512; /* 2 * 4 * 64 = 512 B */
    const uint16_t *wsc = M->scales_fp16 + (size_t)tile_idx * (size_t)nb * 4;

    for (int t0 = 0; t0 < T; t0 += 32) {
        int tblk = (t0 + 32 <= T) ? 32 : (T - t0);
        __m128 vs[32];
        for (int ti = 0; ti < tblk; ti++) vs[ti] = _mm_setzero_ps();

        for (int b = 0; b < nb; b++) {
            const uint8_t *wb = wt + b * 512;
            __m512i wu8[2][4];
            for (int u = 0; u < 2; u++) {
                const uint8_t *wu = wb + u * 256;
                for (int ri = 0; ri < 4; ri++)
                    wu8[u][ri] = _mm512_loadu_si512((const __m512i *)(wu + ri * 64));
            }
            __m128 vws = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)(wsc + b * 4)));
            for (int ti = 0; ti < tblk; ti++) {
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
            float s[4];
            _mm_storeu_ps(s, vs[ti]);
            float *yt = Y + (size_t)(t0 + ti) * R + o_base;
            if (apply_relu2) {
                for (int r = 0; r < 4; r++) {
                    float v = bf16_round_f32(s[r]);
                    v = v > 0.0f ? v : 0.0f;
                    yt[r] = bf16_round_f32(v * v);
                }
            } else {
                for (int r = 0; r < 4; r++) yt[r] = bf16_round_f32(s[r]);
            }
        }
    }
}

NHG4_TGT_AVX2 static void nhg8_tile4_avx2(
    const nhg4_mat_t *M,
    const int8_t *X_q, const float *X_sc, const int32_t *X_sum,
    float *Y, int T, int tile_idx, int apply_relu2
) {
    const int R = (int)M->R, K = (int)M->K, nb = K / 128;
    const int o_base = tile_idx * 4;
    const uint8_t *wt = M->tiles + (size_t)tile_idx * (size_t)nb * 512;
    const uint16_t *wsc = M->scales_fp16 + (size_t)tile_idx * (size_t)nb * 4;
    const __m256i ones16 = _mm256_set1_epi16(1);

    for (int t0 = 0; t0 < T; t0 += 16) {
        int tblk = (t0 + 16 <= T) ? 16 : (T - t0);
        __m128 vs[16];
        for (int ti = 0; ti < tblk; ti++) vs[ti] = _mm_setzero_ps();
        for (int b = 0; b < nb; b++) {
            const uint8_t *wb = wt + b * 512;
            __m128 vws = _mm_cvtph_ps(_mm_loadl_epi64((const __m128i *)(wsc + b * 4)));
            for (int ti = 0; ti < tblk; ti++) {
                int t_abs = t0 + ti;
                __m256i acc[4] = {_mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256(), _mm256_setzero_si256()};
                const int8_t *xq = X_q + (size_t)t_abs * K + b * 128;
                for (int u = 0; u < 2; u++) {
                    __m256i xlo = _mm256_loadu_si256((const __m256i *)(xq + u * 64 + 0));
                    __m256i xhi = _mm256_loadu_si256((const __m256i *)(xq + u * 64 + 32));
                    const uint8_t *wu = wb + u * 256;
                    for (int ri = 0; ri < 4; ri++) {
                        __m256i wlo = _mm256_loadu_si256((const __m256i *)(wu + ri * 64 + 0));
                        __m256i whi = _mm256_loadu_si256((const __m256i *)(wu + ri * 64 + 32));
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
            float s[4];
            _mm_storeu_ps(s, vs[ti]);
            float *yt = Y + (size_t)(t0 + ti) * R + o_base;
            if (apply_relu2) {
                for (int r = 0; r < 4; r++) {
                    float v = bf16_round_f32(s[r]);
                    v = v > 0.0f ? v : 0.0f;
                    yt[r] = bf16_round_f32(v * v);
                }
            } else {
                for (int r = 0; r < 4; r++) yt[r] = bf16_round_f32(s[r]);
            }
        }
    }
}

NHG4_TGT_AVX2 static void nhg4_tile4_avx2(
    const nhg4_mat_t *M, const uint8_t *lut16_u8,
    const int8_t *X_q, const float *X_sc, const int32_t *X_sum,
    float *Y, int T, int tile_idx, int apply_relu2
) {
    const int R = (int)M->R, K = (int)M->K, nb = K / 128;
    const int o_base = tile_idx * 4;
    const uint8_t *wt = M->tiles + (size_t)tile_idx * (size_t)nb * 256;
    const uint16_t *wsc = M->scales_fp16 + (size_t)tile_idx * (size_t)nb * 4;
    const __m256i vlut = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)lut16_u8));
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
                        __m256i p16 = _mm256_add_epi16(_mm256_maddubs_epi16(wlo[u][ri], xlo),
                                                       _mm256_maddubs_epi16(whi[u][ri], xhi));
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
            float s[4];
            _mm_storeu_ps(s, vs[ti]);
            float *yt = Y + (size_t)(t0 + ti) * R + o_base;
            if (apply_relu2) {
                for (int r = 0; r < 4; r++) {
                    float v = bf16_round_f32(s[r]);
                    v = v > 0.0f ? v : 0.0f;
                    yt[r] = bf16_round_f32(v * v);
                }
            } else {
                for (int r = 0; r < 4; r++) yt[r] = bf16_round_f32(s[r]);
            }
        }
    }
}

/* Canonical 16-Row `HG4-16R` & `HG8-16R` Kernels for Nemotron 3.5 */
NHG4_TGT_AVX512VNNI static void nhg4_group16_avx512(
    const nhg4_mat_t *M, const uint8_t *lut16_u8,
    const int8_t *X_q, const float *X_sc, const int32_t *X_sum,
    float *Y, int T, int rg, int apply_relu2
) {
    const int R = (int)M->R, K = (int)M->K, nb = K / 128;
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
            float sv[16];
            _mm512_storeu_ps(sv, vs[ti]);
            float *yt = Y + (size_t)(t0 + ti) * R + rg * 16;
            if (apply_relu2) {
                for (int r = 0; r < 16; r++) {
                    float v = bf16_round_f32(sv[r]);
                    v = v > 0.0f ? v : 0.0f;
                    yt[r] = bf16_round_f32(v * v);
                }
            } else {
                for (int r = 0; r < 16; r++) yt[r] = bf16_round_f32(sv[r]);
            }
        }
    }
}

NHG4_TGT_AVX2 static void nhg4_group16_avx2(
    const nhg4_mat_t *M, const int8_t *lut16_s8,
    const int8_t *X_q, const float *X_sc,
    float *Y, int T, int rg, int apply_relu2
) {
    const int R = (int)M->R, K = (int)M->K, nb = K / 128;
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
            float sv[16];
            _mm256_storeu_ps(sv + 0, vs0[ti]);
            _mm256_storeu_ps(sv + 8, vs1[ti]);
            float *yt = Y + (size_t)(t0 + ti) * R + rg * 16;
            if (apply_relu2) {
                for (int r = 0; r < 16; r++) {
                    float v = bf16_round_f32(sv[r]);
                    v = v > 0.0f ? v : 0.0f;
                    yt[r] = bf16_round_f32(v * v);
                }
            } else {
                for (int r = 0; r < 16; r++) yt[r] = bf16_round_f32(sv[r]);
            }
        }
    }
}
#endif

static void nhg4_group16_scalar(
    const nhg4_mat_t *M, const int8_t *lut16_s8,
    const int8_t *X_q, const float *X_sc,
    float *Y, int T, int rg, int apply_relu2
) {
    const int R = (int)M->R, K = (int)M->K, nb = K / 128;
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
        float *yt = Y + (size_t)t * R + rg * 16;
        if (apply_relu2) {
            for (int r = 0; r < 16; r++) {
                float v = bf16_round_f32(vs[r]);
                v = v > 0.0f ? v : 0.0f;
                yt[r] = bf16_round_f32(v * v);
            }
        } else {
            for (int r = 0; r < 16; r++) yt[r] = bf16_round_f32(vs[r]);
        }
    }
}

static inline void nhg4_rot_quant_dispatch(const float *X, const float *pd, int T, int K, int8_t *xq, float *xsc, int32_t *xsum, int use_omp) {
#if NHG4_X86
    if (g_nhg4.isa_tier >= 1) {
        rotate_and_quantize_128_avx2(X, pd, T, K, xq, xsc, xsum, use_omp);
        return;
    }
#endif
    rotate_and_quantize_128_scalar(X, pd, T, K, xq, xsc, xsum, use_omp);
}

static inline void nhg4_run_one_group16(const nhg4_mat_t *M, const int8_t *xq, const float *xsc, const int32_t *xsum, float *Y, int T, int rg, int apply_relu2) {
#if NHG4_X86
    if (g_nhg4.isa_tier == 3) { nhg4_group16_avx512(M, g_nhg4.lut16_u8, xq, xsc, xsum, Y, T, rg, apply_relu2); return; }
    if (g_nhg4.isa_tier >= 1) { nhg4_group16_avx2(M, g_nhg4.lut16_s8, xq, xsc, Y, T, rg, apply_relu2); return; }
#endif
    nhg4_group16_scalar(M, g_nhg4.lut16_s8, xq, xsc, Y, T, rg, apply_relu2);
}

static void nhg4_matmul_prequant(
    const nhg4_mat_t *M,
    const int8_t *xq, const float *xsc, const int32_t *xsum,
    float *Y, int T, int apply_relu2, int use_omp
) {
    const int n_tiles = (int)(M->R / 4);
    const int nb = (int)(M->K / 128);
    size_t tile_b = (M->bits == 8) ? 512u : 256u;
    size_t hg4_bytes = (size_t)n_tiles * (size_t)nb * (tile_b + 8);
    size_t bf16_equiv = (size_t)M->R * (size_t)M->K * 2ULL;
    if (use_omp) {
        g_nhg4.bytes_tier2_ddr5 += (unsigned long long)hg4_bytes;
        g_nhg4.bytes_tier1_l3_vcache += (unsigned long long)((size_t)T * (size_t)(M->K + M->R) * 4ULL);
        g_nhg4.bytes_tier0_reg += (unsigned long long)((size_t)T * (size_t)M->R * (size_t)M->K);
        g_nhg4.bytes_saved_q4_and_broadcast += (unsigned long long)((bf16_equiv - hg4_bytes) + (size_t)(T > 1 ? (T - 1) : 0) * hg4_bytes);
    }
    if (g_nhg4.is_16r) {
        const int n_rg = (int)(M->R / 16);
        #pragma omp parallel for schedule(static) if (use_omp)
        for (int rg = 0; rg < n_rg; rg++)
            nhg4_run_one_group16(M, xq, xsc, xsum, Y, T, rg, apply_relu2);
        return;
    }
#if NHG4_X86
    if (g_nhg4.isa_tier == 3) {
        if (M->bits == 8) {
            #pragma omp parallel for schedule(static) if (use_omp)
            for (int tl = 0; tl < n_tiles; tl++)
                nhg8_tile4_avx512(M, xq, xsc, xsum, Y, T, tl, apply_relu2);
        } else {
            #pragma omp parallel for schedule(static) if (use_omp)
            for (int tl = 0; tl < n_tiles; tl++)
                nhg4_tile4_avx512(M, g_nhg4.lut16_u8, xq, xsc, xsum, Y, T, tl, apply_relu2);
        }
        return;
    } else if (g_nhg4.isa_tier >= 1) {
        if (M->bits == 8) {
            #pragma omp parallel for schedule(static) if (use_omp)
            for (int tl = 0; tl < n_tiles; tl++)
                nhg8_tile4_avx2(M, xq, xsc, xsum, Y, T, tl, apply_relu2);
        } else {
            #pragma omp parallel for schedule(static) if (use_omp)
            for (int tl = 0; tl < n_tiles; tl++)
                nhg4_tile4_avx2(M, g_nhg4.lut16_u8, xq, xsc, xsum, Y, T, tl, apply_relu2);
        }
        return;
    }
#endif
    (void)xq; (void)xsc; (void)xsum;
}

static void nhg4_matmul(const nhg4_mat_t *M, const float *X, float *Y, int T, int apply_relu2, int use_omp) {
    const int K = (int)M->K, nb = K / 128;
    int use_scr = (use_omp && T <= NEMO_PANEL_MAX && K <= NEMO_MAX_K && g_nhg4.scr_xq != NULL);
    int8_t  *xq   = use_scr ? g_nhg4.scr_xq   : (int8_t *)malloc((size_t)T * K);
    float   *xsc  = use_scr ? g_nhg4.scr_xsc  : (float *)malloc((size_t)T * nb * sizeof(float));
    int32_t *xsum = use_scr ? g_nhg4.scr_xsum : (int32_t *)malloc((size_t)T * nb * sizeof(int32_t));
    nhg4_rot_quant_dispatch(X, M->pre_diag, T, K, xq, xsc, xsum, use_omp);
    nhg4_matmul_prequant(M, xq, xsc, xsum, Y, T, apply_relu2, use_omp);
    if (!use_scr) { free(xq); free(xsc); free(xsum); }
}

static void nhg4_embed_lookup_row(int token_id, float *dst_4096) {
    if (token_id < 0 || token_id >= NEMO_VOCAB) token_id = 0;
    const nhg4_mat_t *E = &g_nhg4.emb_mat;
    const int nb = NEMO_D / 128; /* 32 */
    if (g_nhg4.is_16r) {
        int rg = token_id / 16, lane = token_id % 16;
        const uint8_t *wt = E->tiles + (size_t)rg * (size_t)nb * 1024;
        const uint16_t *wsc = E->scales_fp16 + (size_t)rg * (size_t)nb * 16;
        for (int b = 0; b < nb; b++) {
            float *blk = dst_4096 + b * 128;
            float sc = fp16_to_f32(wsc[b * 16 + lane]);
            const uint8_t *wb = wt + (size_t)b * 1024;
            for (int s = 0; s < 16; s++) {
                const uint8_t *wr = wb + s * 64 + lane * 4;
                for (int j = 0; j < 4; j++) {
                    uint8_t pk = wr[j];
                    blk[s * 8 + j]     = g_nhg4.lut16_s8_f[pk & 0x0F] * sc;
                    blk[s * 8 + 4 + j] = g_nhg4.lut16_s8_f[pk >> 4]   * sc;
                }
            }
            fwht128_1block_scalar(blk);
            for (int k = 0; k < 128; k++) blk[k] = bf16_round_f32(blk[k] * E->pre_diag[b * 128 + k]);
        }
        return;
    }
    int tl = token_id / 4, ri = token_id % 4;
    const uint8_t *wt = E->tiles + (size_t)tl * (size_t)nb * 256;
    const uint16_t *wsc = E->scales_fp16 + (size_t)tl * (size_t)nb * 4;

    for (int b = 0; b < nb; b++) {
        float *blk = dst_4096 + b * 128;
        float sc = fp16_to_f32(wsc[b * 4 + ri]);
        const uint8_t *wb = wt + b * 256;
        for (int u = 0; u < 2; u++) {
            const uint8_t *wu = wb + u * 128 + ri * 32;
            for (int j = 0; j < 32; j++) {
                uint8_t pk = wu[j];
                blk[u * 64 + j]      = g_nhg4.lut16_s8_f[pk & 0x0F] * sc;
                blk[u * 64 + 32 + j] = g_nhg4.lut16_s8_f[pk >> 4]   * sc;
            }
        }
        fwht128_1block_scalar(blk);
        for (int k = 0; k < 128; k++) blk[k] = bf16_round_f32(blk[k] * E->pre_diag[b * 128 + k]);
    }
}

static void rms_norm_f32_bf16round(const float *x, const float *w, float *y, int T, int dim, float eps) {
    #pragma omp parallel for schedule(static) if (T >= 4)
    for (int t = 0; t < T; t++) {
        const float *xr = x + (size_t)t * dim;
        float *yr = y + (size_t)t * dim;
        double ss = 0.0;
        for (int d = 0; d < dim; d++) ss += (double)xr[d] * (double)xr[d];
        float inv = 1.0f / sqrtf((float)(ss / (double)dim) + eps);
        for (int d = 0; d < dim; d++) {
            float normed = bf16_round_f32(xr[d] * inv);
            yr[d] = bf16_round_f32(normed * w[d]);
        }
    }
}

/* SpecMD SlotCache bookkeeping */
NEMO_API void nemo_set_use_4bit_experts(int enable) { (void)enable; }
NEMO_API void nemo_set_madvise_active(int enable) { g_nhg4.madvise_active = enable ? 1 : 0; }

NEMO_API void nemo_dm_reset(void) {
    g_nhg4.clock = 0;
    for (int i = 0; i < 64; i++) { g_nhg4.pool_keys[i] = -1; g_nhg4.pool_ts[i] = 0; g_nhg4.pool_heat[i] = 0.0f; }
    memset(g_nhg4.host_keys, 0, sizeof(g_nhg4.host_keys));
    g_nhg4.total_requests = g_nhg4.tier1_pool_hits = g_nhg4.tier2_host_hits = 0;
    g_nhg4.tier3_drive_fetches = g_nhg4.r4_clean_drops = g_nhg4.dirty_writebacks = 0;
    g_nhg4.bytes_tier0_reg = g_nhg4.bytes_tier1_l3_vcache = g_nhg4.bytes_tier2_ddr5 = g_nhg4.bytes_saved_q4_and_broadcast = 0;
    g_nhg4.madvise_willneed_calls = g_nhg4.madvise_dontneed_calls = 0;
}

static void nhg4_slotcache_acquire(int exp_id, const uint8_t *protected_mask, float weight) {
    g_nhg4.clock++;
    g_nhg4.total_requests++;
    for (int i = 0; i < 64; i++) {
        if (g_nhg4.pool_keys[i] == exp_id) {
            g_nhg4.tier1_pool_hits++;
            g_nhg4.pool_ts[i] = g_nhg4.clock;
            g_nhg4.pool_heat[i] = 0.75f * g_nhg4.pool_heat[i] + weight;
            return;
        }
    }
    if (g_nhg4.host_keys[exp_id]) g_nhg4.tier2_host_hits++;
    else { g_nhg4.tier3_drive_fetches++; g_nhg4.host_keys[exp_id] = 1; }
    int slot = -1;
    for (int i = 0; i < 64; i++) if (g_nhg4.pool_keys[i] < 0) { slot = i; break; }
    if (slot < 0) {
        int best_v = 0;
        double best_sc = 1e30;
        for (int i = 0; i < 64; i++) {
            int k = g_nhg4.pool_keys[i];
            if (k >= 0 && k < NEMO_EXPERTS && protected_mask[k]) continue;
            unsigned long stale = g_nhg4.clock > g_nhg4.pool_ts[i] ? (g_nhg4.clock - g_nhg4.pool_ts[i]) : 1UL;
            double sc = (double)g_nhg4.pool_heat[i] / (double)(1UL + stale);
            if (sc < best_sc) { best_sc = sc; best_v = i; }
        }
        slot = best_v;
        if (g_nhg4.pool_keys[slot] >= 0 && g_nhg4.host_keys[g_nhg4.pool_keys[slot]]) g_nhg4.r4_clean_drops++;
        else g_nhg4.dirty_writebacks++;
    }
    g_nhg4.pool_keys[slot] = exp_id;
    g_nhg4.pool_ts[slot] = g_nhg4.clock;
    g_nhg4.pool_heat[slot] = weight;
}

NEMO_API void nemo_dm_stats(unsigned long *out6) {
    if (!out6) return;
    out6[0] = g_nhg4.total_requests;
    out6[1] = g_nhg4.tier1_pool_hits;
    out6[2] = g_nhg4.tier2_host_hits;
    out6[3] = g_nhg4.tier3_drive_fetches;
    out6[4] = g_nhg4.r4_clean_drops;
    out6[5] = g_nhg4.dirty_writebacks;
}

NEMO_API void nemo_dm_extended_stats(unsigned long long *out8) {
    if (!out8) return;
    out8[0] = g_nhg4.bytes_tier0_reg;
    out8[1] = g_nhg4.bytes_tier1_l3_vcache;
    out8[2] = g_nhg4.bytes_tier2_ddr5;
    out8[3] = g_nhg4.bytes_saved_q4_and_broadcast;
    out8[4] = g_nhg4.madvise_willneed_calls;
    out8[5] = g_nhg4.madvise_dontneed_calls;
    out8[6] = 0;
    out8[7] = 0;
}

/* Generic Mamba-2 Layer Execution (supports both stateless segmented batch & stateful O(1) recurrent decode) */
static void nhg4_mamba_exec(nhg4_layer_t *lyr, const float *h_in, float *h_out, int pos_off, int T, const int *seg_starts, int stateful) {
    const nhg4_mamba_t *m = &lyr->mamba;
    const int ZXBC_DIM = 18432;
    float *u          = (float *)malloc((size_t)T * NEMO_D * sizeof(float));
    float *zxbc       = (float *)malloc((size_t)T * ZXBC_DIM * sizeof(float));
    float *dt_proj    = (float *)malloc((size_t)T * NEMO_MAMBA_HEADS * sizeof(float));
    float *conv_act   = (float *)malloc((size_t)T * NEMO_MAMBA_CONV * sizeof(float));
    float *y_ssm      = (float *)malloc((size_t)T * NEMO_MAMBA_INNER * sizeof(float));
    float *scan_out   = (float *)malloc((size_t)T * NEMO_MAMBA_INNER * sizeof(float));
    float *branch_out = (float *)malloc((size_t)T * NEMO_D * sizeof(float));

    rms_norm_f32_bf16round(h_in, m->norm_w, u, T, NEMO_D, 1e-5f);
    nhg4_matmul(&m->zxbc_proj, u, zxbc, T, 0, 1);
    nhg4_matmul(&m->dt_proj,   u, dt_proj, T, 0, 1);

    #pragma omp parallel for schedule(static)
    for (int c = 0; c < NEMO_MAMBA_CONV; c++) {
        const float *cw = m->cw + c * 4;
        float w0 = cw[0], w1 = cw[1], w2 = cw[2], w3 = cw[3];
        float bias = m->cb[c];
        for (int t = 0; t < T; t++) {
            int s0 = seg_starts ? seg_starts[t] : 0;
            float x_taps[4];
            for (int tap = 0; tap < 4; tap++) {
                int rel = t - 3 + tap;
                if (rel >= (stateful ? 0 : s0)) {
                    x_taps[tap] = zxbc[(size_t)rel * ZXBC_DIM + 8192 + c];
                } else if (stateful && lyr->conv_hist && (pos_off + rel) >= 0) {
                    x_taps[tap] = lyr->conv_hist[(size_t)(3 + rel) * NEMO_MAMBA_CONV + c];
                } else {
                    x_taps[tap] = 0.0f;
                }
            }
            float v = bf16_round_f32(x_taps[0] * w0 + x_taps[1] * w1 + x_taps[2] * w2 + x_taps[3] * w3 + bias);
            float silu = v / (1.0f + expf(-v));
            conv_act[(size_t)t * NEMO_MAMBA_CONV + c] = bf16_round_f32(silu);
        }
    }

    if (stateful && lyr->conv_hist) {
        int n_sn = (g_nhg4.spec_save_snaps > 0 && lyr->conv_snaps) ? g_nhg4.spec_save_snaps : 0;
        for (int s = 0; s < n_sn && s < T && s < 4; s++) {
            int L = s + 1;
            float *cs_dst = lyr->conv_snaps + (size_t)s * 3 * NEMO_MAMBA_CONV;
            for (int k = 0; k < 3; k++) {
                int rel = L - 3 + k;
                const float *src = (rel >= 0)
                    ? (zxbc + (size_t)rel * ZXBC_DIM + 8192)
                    : (lyr->conv_hist + (size_t)(k + L) * NEMO_MAMBA_CONV);
                memcpy(cs_dst + (size_t)k * NEMO_MAMBA_CONV, src, NEMO_MAMBA_CONV * sizeof(float));
            }
        }
        for (int k = 0; k < 3; k++) {
            int rel = T - 3 + k;
            if (rel >= 0) {
                const float *src = zxbc + (size_t)rel * ZXBC_DIM + 8192;
                memcpy(lyr->conv_hist + (size_t)k * NEMO_MAMBA_CONV, src, NEMO_MAMBA_CONV * sizeof(float));
            } else if (k + T < 3) {
                memmove(lyr->conv_hist + (size_t)k * NEMO_MAMBA_CONV,
                        lyr->conv_hist + (size_t)(k + T) * NEMO_MAMBA_CONV,
                        NEMO_MAMBA_CONV * sizeof(float));
            }
        }
    }

    const int save_ssm_n = (stateful && g_nhg4.spec_save_snaps > 0 && lyr->ssm_snaps) ? g_nhg4.spec_save_snaps : 0;
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < NEMO_MAMBA_HEADS; h++) {
        int grp = h / 16;
        float A_h = m->A[h];
        float D_h = m->D[h];
        float dt_b = m->dtb[h];
        float local_st[NEMO_MAMBA_HDIM * NEMO_MAMBA_DSTATE];
        float *state = (stateful && lyr->ssm_state) ? (lyr->ssm_state + (size_t)h * NEMO_MAMBA_HDIM * NEMO_MAMBA_DSTATE) : local_st;
        if (!stateful || pos_off == 0) memset(state, 0, NEMO_MAMBA_HDIM * NEMO_MAMBA_DSTATE * sizeof(float));

        for (int t = 0; t < T; t++) {
            if (!stateful && seg_starts && t == seg_starts[t]) memset(state, 0, NEMO_MAMBA_HDIM * NEMO_MAMBA_DSTATE * sizeof(float));
            float dt_raw = dt_proj[(size_t)t * NEMO_MAMBA_HEADS + h] + dt_b;
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
                #pragma omp simd reduction(+:acc)
                for (int s = 0; s < NEMO_MAMBA_DSTATE; s++) {
                    st_row[s] = st_row[s] * dA + (dt_val * B_vec[s]) * xd;
                    acc += st_row[s] * C_vec[s];
                }
                y_vec[d] = acc;
            }
            if (t < save_ssm_n && t < 4) {
                memcpy(lyr->ssm_snaps + ((size_t)t * NEMO_MAMBA_HEADS + h) * (NEMO_MAMBA_HDIM * NEMO_MAMBA_DSTATE),
                       state, NEMO_MAMBA_HDIM * NEMO_MAMBA_DSTATE * sizeof(float));
            }
        }
    }

    #pragma omp parallel for schedule(static) if (T >= 2)
    for (int t = 0; t < T; t++) {
        const float *gate_row = zxbc + (size_t)t * ZXBC_DIM;
        const float *y_row = y_ssm + (size_t)t * NEMO_MAMBA_INNER;
        float *s_row = scan_out + (size_t)t * NEMO_MAMBA_INNER;
        for (int g = 0; g < 8; g++) {
            double ss = 0.0;
            for (int i = 0; i < 1024; i++) {
                int idx = g * 1024 + i;
                float gt = gate_row[idx];
                float gated = y_row[idx] * (gt / (1.0f + expf(-gt)));
                s_row[idx] = gated;
                ss += (double)gated * (double)gated;
            }
            float inv = 1.0f / sqrtf((float)(ss / 1024.0) + 1e-5f);
            for (int i = 0; i < 1024; i++) {
                int idx = g * 1024 + i;
                s_row[idx] = bf16_round_f32((s_row[idx] * inv) * m->mix_norm[idx]);
            }
        }
    }

    nhg4_matmul(&m->out_proj, scan_out, branch_out, T, 0, 1);
    for (int i = 0; i < T * NEMO_D; i++) h_out[i] = bf16_round_f32(h_in[i] + branch_out[i]);

    free(u); free(zxbc); free(dt_proj); free(conv_act); free(y_ssm); free(scan_out); free(branch_out);
}

NEMO_API void nemo_mamba_layer_forward_seg(const float *h_in, float *h_out, int T, const int *seg_starts) {
    nhg4_mamba_exec(&g_nhg4.layers[g_nhg4.idx_mamba0], h_in, h_out, 0, T, seg_starts, 0);
}

/* Stage L2: 512-Expert Latent ReLU^2 MoE with Shared-Input `FWHT-128` & 16-Core T=1 Row-Group Parallelism */
static void nhg4_latent_moe_forward(
    const nhg4_moe_trunk_t *tr, const nhg4_experts_t *ex,
    const float *h_in, float *h_out, int T
) {
    /* Fast zero-malloc 16-core path for T == 1 Decode (`22 * 168 = 3696` up tasks + `22 * 64 = 1408` down tasks!) */
    if (T == 1 && g_nhg4.is_16r && g_nhg4.scr_moe_inter != NULL) {
        float u[NEMO_D], latent_in[NEMO_LATENT], latent_acc[NEMO_LATENT];
        float routed_4096[NEMO_D], sh_inter[NEMO_SH_INTER], shared_4096[NEMO_D];
        rms_norm_f32_bf16round(h_in, tr->norm_w, u, 1, NEMO_D, 1e-5f);
        nhg4_matmul(&tr->fc1, u, latent_in, 1, 0, 1);

        float scores[NEMO_EXPERTS], choice[NEMO_EXPERTS];
        #pragma omp parallel for schedule(static)
        for (int e = 0; e < NEMO_EXPERTS; e++) {
            const float *gw = tr->gate_w + (size_t)e * NEMO_D;
            float dot = 0.0f;
            #pragma omp simd reduction(+:dot)
            for (int d = 0; d < NEMO_D; d++) dot += u[d] * gw[d];
            float sc = 1.0f / (1.0f + expf(-dot));
            scores[e] = sc;
            choice[e] = sc + tr->gate_b[e];
        }

        int sel_e[NEMO_TOP_K];
        float sel_w[NEMO_TOP_K], w_sum = 0.0f;
        for (int k = 0; k < NEMO_TOP_K; k++) {
            int best_e = 0; float best_val = -1e30f;
            for (int e = 0; e < NEMO_EXPERTS; e++) if (choice[e] > best_val) { best_val = choice[e]; best_e = e; }
            choice[best_e] = -1e30f;
            sel_e[k] = best_e; sel_w[k] = scores[best_e]; w_sum += scores[best_e];
        }
        for (int k = 0; k < NEMO_TOP_K; k++) sel_w[k] = (sel_w[k] / (w_sum + 1e-20f)) * 5.0f;
        /* Sort selected experts by ascending ID so T==1 and T>=2 speculative batches share identical FP32 sum order */
        for (int i = 1; i < NEMO_TOP_K; i++) {
            int ke = sel_e[i]; float kw = sel_w[i]; int j = i - 1;
            while (j >= 0 && sel_e[j] > ke) { sel_e[j + 1] = sel_e[j]; sel_w[j + 1] = sel_w[j]; j--; }
            sel_e[j + 1] = ke; sel_w[j + 1] = kw;
        }

        int8_t  lat_xq[NEMO_LATENT];
        float   lat_xsc[8];
        int32_t lat_xsum[8];
        nhg4_rot_quant_dispatch(latent_in, ex->pre_diag_up, 1, NEMO_LATENT, lat_xq, lat_xsc, lat_xsum, 0);

        nhg4_mat_t up_m[NEMO_TOP_K], dn_m[NEMO_TOP_K];
        for (int k = 0; k < NEMO_TOP_K; k++) {
            const uint8_t *ep = ex->exp_base + (size_t)sel_e[k] * ex->exp_stride;
            up_m[k].R = NEMO_MOE_INTER; up_m[k].K = NEMO_LATENT; up_m[k].bits = 4;
            up_m[k].pre_diag = ex->pre_diag_up;
            up_m[k].scales_fp16 = (const uint16_t *)ep;
            up_m[k].tiles = ep + 43008;
            parse_nhg4_mat(ep + 1419264, &dn_m[k]);
        }

        const int up_rg = NEMO_MOE_INTER / 16; /* 168 */
        #pragma omp parallel for schedule(static)
        for (int task = 0; task < NEMO_TOP_K * up_rg; task++) {
            int k = task / up_rg, rg = task % up_rg;
            nhg4_run_one_group16(&up_m[k], lat_xq, lat_xsc, lat_xsum, g_nhg4.scr_moe_inter + (size_t)k * NEMO_MOE_INTER, 1, rg, 1);
        }

        int8_t  z_xq[NEMO_TOP_K * NEMO_MOE_INTER];
        float   z_xsc[NEMO_TOP_K * 21];
        int32_t z_xsum[NEMO_TOP_K * 21];
        #pragma omp parallel for schedule(static)
        for (int k = 0; k < NEMO_TOP_K; k++) {
            nhg4_rot_quant_dispatch(g_nhg4.scr_moe_inter + (size_t)k * NEMO_MOE_INTER, dn_m[k].pre_diag, 1, NEMO_MOE_INTER,
                                    z_xq + (size_t)k * NEMO_MOE_INTER, z_xsc + k * 21, z_xsum + k * 21, 0);
        }

        const int dn_rg = NEMO_LATENT / 16; /* 64 */
        #pragma omp parallel for schedule(static)
        for (int task = 0; task < NEMO_TOP_K * dn_rg; task++) {
            int k = task / dn_rg, rg = task % dn_rg;
            nhg4_run_one_group16(&dn_m[k], z_xq + (size_t)k * NEMO_MOE_INTER, z_xsc + k * 21, z_xsum + k * 21,
                                 g_nhg4.scr_moe_out + (size_t)k * NEMO_LATENT, 1, rg, 0);
        }

        for (int d = 0; d < NEMO_LATENT; d++) {
            float acc = 0.0f;
            for (int k = 0; k < NEMO_TOP_K; k++) acc += g_nhg4.scr_moe_out[(size_t)k * NEMO_LATENT + d] * sel_w[k];
            latent_acc[d] = bf16_round_f32(acc);
        }
        nhg4_matmul(&tr->fc2, latent_acc, routed_4096, 1, 0, 1);
        nhg4_matmul(&tr->sh_up, u, sh_inter, 1, 1, 1);
        nhg4_matmul(&tr->sh_dn, sh_inter, shared_4096, 1, 0, 1);
        for (int d = 0; d < NEMO_D; d++) {
            float comb = bf16_round_f32(routed_4096[d] + shared_4096[d]);
            h_out[d] = bf16_round_f32(h_in[d] + comb);
        }
        g_nhg4.bytes_tier2_ddr5 += (unsigned long long)NEMO_TOP_K * ex->exp_stride;
        return;
    }

    float *u           = (float *)malloc((size_t)T * NEMO_D * sizeof(float));
    float *latent_in   = (float *)malloc((size_t)T * NEMO_LATENT * sizeof(float));
    float *latent_acc  = (float *)calloc((size_t)T * NEMO_LATENT, sizeof(float));
    float *routed_4096 = (float *)malloc((size_t)T * NEMO_D * sizeof(float));
    float *sh_inter    = (float *)malloc((size_t)T * NEMO_SH_INTER * sizeof(float));
    float *shared_4096 = (float *)malloc((size_t)T * NEMO_D * sizeof(float));

    rms_norm_f32_bf16round(h_in, tr->norm_w, u, T, NEMO_D, 1e-5f);
    nhg4_matmul(&tr->fc1, u, latent_in, T, 0, 1);

    int   *topk_idx = (int *)malloc((size_t)T * NEMO_TOP_K * sizeof(int));
    float *topk_w   = (float *)malloc((size_t)T * NEMO_TOP_K * sizeof(float));
    uint8_t active_mask[NEMO_EXPERTS];
    float   expert_weight_sum[NEMO_EXPERTS];
    memset(active_mask, 0, sizeof(active_mask));
    memset(expert_weight_sum, 0, sizeof(expert_weight_sum));

    float *all_scores = (float *)malloc((size_t)T * NEMO_EXPERTS * sizeof(float));
    #pragma omp parallel for schedule(static)
    for (int e = 0; e < NEMO_EXPERTS; e++) {
        const float *gw = tr->gate_w + (size_t)e * NEMO_D;
        for (int t = 0; t < T; t++) {
            const float *u_row = u + (size_t)t * NEMO_D;
            float dot = 0.0f;
            #pragma omp simd reduction(+:dot)
            for (int d = 0; d < NEMO_D; d++) dot += u_row[d] * gw[d];
            all_scores[(size_t)t * NEMO_EXPERTS + e] = 1.0f / (1.0f + expf(-dot));
        }
    }

    for (int t = 0; t < T; t++) {
        float choice[NEMO_EXPERTS];
        const float *sc_t = all_scores + (size_t)t * NEMO_EXPERTS;
        for (int e = 0; e < NEMO_EXPERTS; e++) choice[e] = sc_t[e] + tr->gate_b[e];
        float w_sum = 0.0f;
        for (int k = 0; k < NEMO_TOP_K; k++) {
            int best_e = 0;
            float best_val = -1e30f;
            for (int e = 0; e < NEMO_EXPERTS; e++) {
                if (choice[e] > best_val) { best_val = choice[e]; best_e = e; }
            }
            choice[best_e] = -1e30f;
            topk_idx[t * NEMO_TOP_K + k] = best_e;
            topk_w[t * NEMO_TOP_K + k] = sc_t[best_e];
            w_sum += sc_t[best_e];
            active_mask[best_e] = 1;
        }
        for (int k = 0; k < NEMO_TOP_K; k++) {
            topk_w[t * NEMO_TOP_K + k] = (topk_w[t * NEMO_TOP_K + k] / (w_sum + 1e-20f)) * 5.0f;
            expert_weight_sum[topk_idx[t * NEMO_TOP_K + k]] += topk_w[t * NEMO_TOP_K + k];
        }
    }
    free(all_scores);

    int active_list[NEMO_EXPERTS];
    int n_active = 0;
    for (int e = 0; e < NEMO_EXPERTS; e++) {
        if (!active_mask[e]) continue;
        nhg4_slotcache_acquire(e, active_mask, expert_weight_sum[e]);
        active_list[n_active++] = e;
    }

    const int nb_up = NEMO_LATENT / 128;
    int8_t  *lat_xq   = (int8_t *)malloc((size_t)T * NEMO_LATENT);
    float   *lat_xsc  = (float *)malloc((size_t)T * nb_up * sizeof(float));
    int32_t *lat_xsum = (int32_t *)malloc((size_t)T * nb_up * sizeof(int32_t));
    nhg4_rot_quant_dispatch(latent_in, ex->pre_diag_up, T, NEMO_LATENT, lat_xq, lat_xsc, lat_xsum, 1);

    float *exp_outs = (float *)calloc((size_t)n_active * T * NEMO_LATENT, sizeof(float));
    #pragma omp parallel for schedule(dynamic, 1)
    for (int ai = 0; ai < n_active; ai++) {
        int e = active_list[ai];
        int     s_toks[8];
        float   s_rw[8];
        int8_t  s_xq[8 * NEMO_LATENT];
        float   s_xsc[8 * 8];
        int32_t s_xsum[8 * 8];
        int   *m_toks = (T <= 8) ? s_toks : (int *)malloc((size_t)T * sizeof(int));
        float *m_rw   = (T <= 8) ? s_rw   : (float *)malloc((size_t)T * sizeof(float));
        int8_t  *g_xq   = (T <= 8) ? s_xq   : (int8_t *)malloc((size_t)T * NEMO_LATENT);
        float   *g_xsc  = (T <= 8) ? s_xsc  : (float *)malloc((size_t)T * 8 * sizeof(float));
        int32_t *g_xsum = (T <= 8) ? s_xsum : (int32_t *)malloc((size_t)T * 8 * sizeof(int32_t));
        int n_m = 0;
        for (int t = 0; t < T; t++) {
            for (int k = 0; k < NEMO_TOP_K; k++) {
                if (topk_idx[t * NEMO_TOP_K + k] == e) {
                    m_toks[n_m] = t;
                    m_rw[n_m] = topk_w[t * NEMO_TOP_K + k];
                    memcpy(g_xq + (size_t)n_m * NEMO_LATENT, lat_xq + (size_t)t * NEMO_LATENT, NEMO_LATENT);
                    memcpy(g_xsc + (size_t)n_m * 8, lat_xsc + (size_t)t * 8, 8 * sizeof(float));
                    memcpy(g_xsum + (size_t)n_m * 8, lat_xsum + (size_t)t * 8, 8 * sizeof(int32_t));
                    n_m++;
                    break;
                }
            }
        }
        if (n_m > 0) {
            const uint8_t *ep = ex->exp_base + (size_t)e * ex->exp_stride;
            nhg4_mat_t up_m, dn_m;
            up_m.R = NEMO_MOE_INTER; up_m.K = NEMO_LATENT; up_m.bits = 4;
            up_m.pre_diag = ex->pre_diag_up;
            up_m.scales_fp16 = (const uint16_t *)ep;
            up_m.tiles = ep + 43008;
            parse_nhg4_mat(ep + 1419264, &dn_m);

            float s_inter[8 * NEMO_MOE_INTER];
            float s_out[8 * NEMO_LATENT];
            float *g_inter = (n_m <= 8) ? s_inter : (float *)malloc((size_t)n_m * NEMO_MOE_INTER * sizeof(float));
            float *g_out   = (n_m <= 8) ? s_out   : (float *)malloc((size_t)n_m * NEMO_LATENT * sizeof(float));
            nhg4_matmul_prequant(&up_m, g_xq, g_xsc, g_xsum, g_inter, n_m, 1, 0);
            nhg4_matmul(&dn_m, g_inter, g_out, n_m, 0, 0);

            float *dst_ai = exp_outs + (size_t)ai * T * NEMO_LATENT;
            for (int im = 0; im < n_m; im++) {
                float *dt = dst_ai + (size_t)m_toks[im] * NEMO_LATENT;
                const float *go = g_out + (size_t)im * NEMO_LATENT;
                float rw = m_rw[im];
                for (int d = 0; d < NEMO_LATENT; d++) dt[d] += go[d] * rw;
            }
            if (n_m > 8) { free(g_inter); free(g_out); }
        }
        if (T > 8) { free(m_toks); free(m_rw); free(g_xq); free(g_xsc); free(g_xsum); }
    }
    for (int ai = 0; ai < n_active; ai++) {
        const float *src_ai = exp_outs + (size_t)ai * T * NEMO_LATENT;
        for (int i = 0; i < T * NEMO_LATENT; i++) latent_acc[i] += src_ai[i];
    }
    free(exp_outs); free(lat_xq); free(lat_xsc); free(lat_xsum);
    g_nhg4.bytes_tier2_ddr5 += (unsigned long long)n_active * ex->exp_stride;

    for (int i = 0; i < T * NEMO_LATENT; i++) latent_acc[i] = bf16_round_f32(latent_acc[i]);
    nhg4_matmul(&tr->fc2, latent_acc, routed_4096, T, 0, 1);

    nhg4_matmul(&tr->sh_up, u, sh_inter, T, 1, 1);
    nhg4_matmul(&tr->sh_dn, sh_inter, shared_4096, T, 0, 1);

    for (int i = 0; i < T * NEMO_D; i++) {
        float comb = bf16_round_f32(routed_4096[i] + shared_4096[i]);
        h_out[i] = bf16_round_f32(h_in[i] + comb);
    }

    free(u); free(latent_in); free(latent_acc); free(routed_4096);
    free(sh_inter); free(shared_4096); free(topk_idx); free(topk_w);
}

NEMO_API void nemo_latent_moe_layer_forward(const float *h_in, float *h_out, int T) {
    nhg4_layer_t *lyr = &g_nhg4.layers[g_nhg4.idx_moe1];
    nhg4_latent_moe_forward(&lyr->moe, &lyr->experts, h_in, h_out, T);
}

/* Stage L3: NoPE 16:1 GQA Attention (supports both stateless segmented batch & stateful BF16 KV cache) */
static void nhg4_gqa_exec(const nhg4_gqa_t *g, uint16_t *kc, uint16_t *vc, const float *h_in, float *h_out, int pos_off, int T, const int *seg_starts, int stateful) {
    float *u          = (float *)malloc((size_t)T * NEMO_D * sizeof(float));
    float *q          = (float *)malloc((size_t)T * NEMO_GQA_Q * sizeof(float));
    float *k          = (float *)malloc((size_t)T * NEMO_GQA_KV * sizeof(float));
    float *v          = (float *)malloc((size_t)T * NEMO_GQA_KV * sizeof(float));
    float *attn_out   = (float *)malloc((size_t)T * NEMO_GQA_Q * sizeof(float));
    float *branch_out = (float *)malloc((size_t)T * NEMO_D * sizeof(float));

    rms_norm_f32_bf16round(h_in, g->norm_w, u, T, NEMO_D, 1e-5f);
    nhg4_matmul(&g->q_proj, u, q, T, 0, 1);
    nhg4_matmul(&g->k_proj, u, k, T, 0, 1);
    nhg4_matmul(&g->v_proj, u, v, T, 0, 1);

    if (stateful && kc && vc) {
        for (int t = 0; t < T; t++) {
            uint16_t *kd = kc + (size_t)(pos_off + t) * NEMO_GQA_KV;
            uint16_t *vd = vc + (size_t)(pos_off + t) * NEMO_GQA_KV;
            for (int d = 0; d < NEMO_GQA_KV; d++) {
                kd[d] = f32_to_bf16_u16(k[(size_t)t * NEMO_GQA_KV + d]);
                vd[d] = f32_to_bf16_u16(v[(size_t)t * NEMO_GQA_KV + d]);
            }
        }
    }

    const float scale = 1.0f / sqrtf((float)NEMO_GQA_HDIM);
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < NEMO_GQA_HEADS; h++) {
        int kv_h = h / 16;
        if (stateful && kc && vc) {
            for (int t = 0; t < T; t++) {
                int abs_pos = pos_off + t;
                const float *qh = q + (size_t)t * NEMO_GQA_Q + h * NEMO_GQA_HDIM;
                float m_max = -1e30f, l_sum = 0.0f;
                float out_h[NEMO_GQA_HDIM] = {0};
                for (int j = 0; j <= abs_pos; j++) {
                    const uint16_t *kj = kc + (size_t)j * NEMO_GQA_KV + kv_h * NEMO_GQA_HDIM;
                    const uint16_t *vj = vc + (size_t)j * NEMO_GQA_KV + kv_h * NEMO_GQA_HDIM;
                    float dot = 0.0f;
                    for (int d = 0; d < NEMO_GQA_HDIM; d++) dot += qh[d] * bf16_u16_to_f32(kj[d]);
                    dot = bf16_round_f32(dot * scale);
                    if (dot > m_max) {
                        float alpha = expf(m_max - dot);
                        l_sum = l_sum * alpha + 1.0f;
                        for (int d = 0; d < NEMO_GQA_HDIM; d++) out_h[d] = out_h[d] * alpha + bf16_u16_to_f32(vj[d]);
                        m_max = dot;
                    } else {
                        float w = expf(dot - m_max);
                        l_sum += w;
                        for (int d = 0; d < NEMO_GQA_HDIM; d++) out_h[d] += w * bf16_u16_to_f32(vj[d]);
                    }
                }
                float inv_l = 1.0f / (l_sum + 1e-12f);
                float *dst = attn_out + (size_t)t * NEMO_GQA_Q + h * NEMO_GQA_HDIM;
                for (int d = 0; d < NEMO_GQA_HDIM; d++) dst[d] = bf16_round_f32(out_h[d] * inv_l);
            }
        } else {
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
                for (int j = j_start; j <= t; j++) { scores[j] = expf(scores[j] - max_s); sum_exp += scores[j]; }
                for (int j = j_start; j <= t; j++) scores[j] = bf16_round_f32(scores[j] / sum_exp);

                float *out_h = attn_out + (size_t)t * NEMO_GQA_Q + h * NEMO_GQA_HDIM;
                memset(out_h, 0, NEMO_GQA_HDIM * sizeof(float));
                for (int j = j_start; j <= t; j++) {
                    const float *vh = v + (size_t)j * NEMO_GQA_KV + kv_h * NEMO_GQA_HDIM;
                    float sj = scores[j];
                    for (int d = 0; d < NEMO_GQA_HDIM; d++) out_h[d] += sj * vh[d];
                }
                for (int d = 0; d < NEMO_GQA_HDIM; d++) out_h[d] = bf16_round_f32(out_h[d]);
            }
            free(scores);
        }
    }

    nhg4_matmul(&g->o_proj, attn_out, branch_out, T, 0, 1);
    for (int i = 0; i < T * NEMO_D; i++) h_out[i] = bf16_round_f32(h_in[i] + branch_out[i]);

    free(u); free(q); free(k); free(v); free(attn_out); free(branch_out);
}

NEMO_API void nemo_gqa_layer_forward_seg(const float *h_in, float *h_out, int T, const int *seg_starts) {
    nhg4_layer_t *lyr = &g_nhg4.layers[g_nhg4.idx_gqa7];
    nhg4_gqa_exec(&lyr->gqa, NULL, NULL, h_in, h_out, 0, T, seg_starts, 0);
}

NEMO_API void nemo_cache_reset(void) {
    g_nhg4.cache_len = 0;
    g_nhg4.mtp_cache_len = 0;
    g_nhg4.spec_save_snaps = 0;
    for (int l = 0; l < g_nhg4.n_layers; l++) {
        if (g_nhg4.layers[l].ssm_state) memset(g_nhg4.layers[l].ssm_state, 0, (size_t)NEMO_MAMBA_HEADS * NEMO_MAMBA_HDIM * NEMO_MAMBA_DSTATE * sizeof(float));
        if (g_nhg4.layers[l].conv_hist) memset(g_nhg4.layers[l].conv_hist, 0, 3 * NEMO_MAMBA_CONV * sizeof(float));
    }
}

NEMO_API int nemo_engine_init(const char *hg4_dir) {
    if (g_nhg4.initialized && strcmp(g_nhg4.loaded_dir, hg4_dir) == 0) return 0;
    if (g_nhg4.initialized) {
        nhg4_mmap_close(&g_nhg4.emb_mf);
        nhg4_mmap_close(&g_nhg4.lm_mf);
        nhg4_mmap_close(&g_nhg4.mtp_mf);
        nhg4_mmap_close(&g_nhg4.experts_mtp1.mf);
        free(g_nhg4.mtp_k_cache_bf16); g_nhg4.mtp_k_cache_bf16 = NULL;
        free(g_nhg4.mtp_v_cache_bf16); g_nhg4.mtp_v_cache_bf16 = NULL;
        for (int l = 0; l < g_nhg4.n_layers; l++) {
            nhg4_mmap_close(&g_nhg4.layers[l].mf);
            nhg4_mmap_close(&g_nhg4.layers[l].experts.mf);
            free(g_nhg4.layers[l].ssm_state);    g_nhg4.layers[l].ssm_state = NULL;
            free(g_nhg4.layers[l].conv_hist);    g_nhg4.layers[l].conv_hist = NULL;
            free(g_nhg4.layers[l].ssm_snaps);    g_nhg4.layers[l].ssm_snaps = NULL;
            free(g_nhg4.layers[l].conv_snaps);   g_nhg4.layers[l].conv_snaps = NULL;
            free(g_nhg4.layers[l].k_cache_bf16); g_nhg4.layers[l].k_cache_bf16 = NULL;
            free(g_nhg4.layers[l].v_cache_bf16); g_nhg4.layers[l].v_cache_bf16 = NULL;
        }
        g_nhg4.has_mtp = 0;
        g_nhg4.initialized = 0;
    }

    g_nhg4.isa_tier = detect_nhg4_isa();
    g_nhg4.madvise_active = 1;

    char path[1024];
    snprintf(path, sizeof(path), "%s/meta.bin", hg4_dir);
    FILE *mf = fopen(path, "rb");
    if (!mf) return -1;
    char magic[4];
    if (fread(magic, 1, 4, mf) != 4) { fclose(mf); return -2; }

    if (memcmp(magic, "HG4P", 4) == 0) {
        uint32_t hdr[4];
        if (fread(hdr, 4, 4, mf) != 4 ||
            fread(g_nhg4.lut16_u8, 1, 16, mf) != 16 ||
            fread(g_nhg4.norm_f, 4, NEMO_D, mf) != NEMO_D) { fclose(mf); return -2; }
        g_nhg4.is_16r = 1;
        g_nhg4.n_layers = (int)hdr[1];
    } else if (memcmp(magic, "HG4N", 4) == 0) {
        uint32_t hdr[3];
        if (fread(hdr, 4, 3, mf) != 3 ||
            fread(g_nhg4.lut16_u8, 1, 16, mf) != 16 ||
            fread(g_nhg4.norm_f, 4, NEMO_D, mf) != NEMO_D) { fclose(mf); return -2; }
        g_nhg4.is_16r = 0;
        g_nhg4.n_layers = 3;
    } else {
        fclose(mf);
        return -2;
    }
    fclose(mf);

    for (int i = 0; i < 16; i++) {
        int sv = (int)g_nhg4.lut16_u8[i] - 128;
        g_nhg4.lut16_s8[i] = (int8_t)sv;
        g_nhg4.lut16_s8_f[i] = (float)sv;
    }

    snprintf(path, sizeof(path), "%s/embeddings.hg4", hg4_dir);
    if (nhg4_mmap_open(path, &g_nhg4.emb_mf) != 0) return -3;
    parse_nhg4_mat((const uint8_t *)g_nhg4.emb_mf.ptr, &g_nhg4.emb_mat);

    snprintf(path, sizeof(path), "%s/lm_head.hg4", hg4_dir);
    if (nhg4_mmap_open(path, &g_nhg4.lm_mf) != 0) return -4;
    parse_nhg4_mat((const uint8_t *)g_nhg4.lm_mf.ptr, &g_nhg4.lm_mat);

    g_nhg4.idx_mamba0 = 0; g_nhg4.idx_moe1 = 1; g_nhg4.idx_gqa7 = 2;

    if (g_nhg4.is_16r) {
        for (int l = 0; l < g_nhg4.n_layers; l++) {
            nhg4_layer_t *lyr = &g_nhg4.layers[l];
            snprintf(path, sizeof(path), "%s/layer_%02d.hg4", hg4_dir, l);
            if (nhg4_mmap_open(path, &lyr->mf) != 0) return -(100 + l);
            const uint8_t *p = (const uint8_t *)lyr->mf.ptr;
            memcpy(&lyr->kind, p, 4);
            if (lyr->kind == 0) {
                parse_mamba_blob(p, 4, &lyr->mamba);
                lyr->ssm_state = (float *)calloc((size_t)NEMO_MAMBA_HEADS * NEMO_MAMBA_HDIM * NEMO_MAMBA_DSTATE, sizeof(float));
                lyr->conv_hist = (float *)calloc(3 * NEMO_MAMBA_CONV, sizeof(float));
            } else if (lyr->kind == 1) {
                parse_moe_trunk_blob_off(p, 4, &lyr->moe);
                snprintf(path, sizeof(path), "%s/experts_%02d.hg4", hg4_dir, l);
                if (open_experts_hg4(path, &lyr->experts) != 0) return -(200 + l);
            } else {
                parse_gqa_blob_off(p, 4, &lyr->gqa);
                lyr->k_cache_bf16 = (uint16_t *)calloc((size_t)NEMO_MAX_SEQ * NEMO_GQA_KV, sizeof(uint16_t));
                lyr->v_cache_bf16 = (uint16_t *)calloc((size_t)NEMO_MAX_SEQ * NEMO_GQA_KV, sizeof(uint16_t));
                if (g_nhg4.idx_gqa7 == 2 && l >= 2) g_nhg4.idx_gqa7 = l;
            }
        }
    } else {
        snprintf(path, sizeof(path), "%s/mamba_trunk0.hg4", hg4_dir);
        if (nhg4_mmap_open(path, &g_nhg4.layers[0].mf) != 0) return -5;
        g_nhg4.layers[0].kind = 0;
        parse_mamba_blob((const uint8_t *)g_nhg4.layers[0].mf.ptr, 0, &g_nhg4.layers[0].mamba);

        snprintf(path, sizeof(path), "%s/moe_trunk1.hg4", hg4_dir);
        if (nhg4_mmap_open(path, &g_nhg4.layers[1].mf) != 0) return -6;
        g_nhg4.layers[1].kind = 1;
        parse_moe_trunk_blob_off((const uint8_t *)g_nhg4.layers[1].mf.ptr, 0, &g_nhg4.layers[1].moe);
        snprintf(path, sizeof(path), "%s/experts_1.hg4", hg4_dir);
        if (open_experts_hg4(path, &g_nhg4.layers[1].experts) != 0) return -7;

        snprintf(path, sizeof(path), "%s/gqa_trunk7.hg4", hg4_dir);
        if (nhg4_mmap_open(path, &g_nhg4.layers[2].mf) != 0) return -8;
        g_nhg4.layers[2].kind = 2;
        parse_gqa_blob_off((const uint8_t *)g_nhg4.layers[2].mf.ptr, 0, &g_nhg4.layers[2].gqa);
    }

    snprintf(path, sizeof(path), "%s/mtp_trunks.hg4", hg4_dir);
    char path_me[1024];
    snprintf(path_me, sizeof(path_me), "%s/experts_mtp_1.hg4", hg4_dir);
    if (nhg4_mmap_open(path, &g_nhg4.mtp_mf) == 0 && open_experts_hg4(path_me, &g_nhg4.experts_mtp1) == 0) {
        const uint8_t *p = (const uint8_t *)g_nhg4.mtp_mf.ptr;
        size_t off = 0;
        g_nhg4.mtp_enorm    = (const float *)(p + off); off += NEMO_D * 4;
        g_nhg4.mtp_hnorm    = (const float *)(p + off); off += NEMO_D * 4;
        g_nhg4.mtp_final_ln = (const float *)(p + off); off += NEMO_D * 4;
        uint32_t l_eh, l_g0, l_m1;
        memcpy(&l_eh, p + off, 4); off += 4;
        memcpy(&l_g0, p + off, 4); off += 4;
        memcpy(&l_m1, p + off, 4); off += 4;
        parse_nhg4_mat(p + off, &g_nhg4.mtp_eh_proj); off += l_eh;
        parse_gqa_blob_off(p, off, &g_nhg4.mtp0_gqa); off += l_g0;
        parse_moe_trunk_blob_off(p, off, &g_nhg4.mtp1_moe);
        g_nhg4.mtp_k_cache_bf16 = (uint16_t *)calloc((size_t)NEMO_MAX_SEQ * NEMO_GQA_KV, sizeof(uint16_t));
        g_nhg4.mtp_v_cache_bf16 = (uint16_t *)calloc((size_t)NEMO_MAX_SEQ * NEMO_GQA_KV, sizeof(uint16_t));
        g_nhg4.has_mtp = 1;
    }

    if (!g_nhg4.scr_xq) {
        g_nhg4.scr_xq        = (int8_t *)malloc((size_t)NEMO_PANEL_MAX * NEMO_MAX_K);
        g_nhg4.scr_xsc       = (float *)malloc((size_t)NEMO_PANEL_MAX * (NEMO_MAX_K / 128) * sizeof(float));
        g_nhg4.scr_xsum      = (int32_t *)malloc((size_t)NEMO_PANEL_MAX * (NEMO_MAX_K / 128) * sizeof(int32_t));
        g_nhg4.scr_moe_inter = (float *)malloc((size_t)NEMO_TOP_K * NEMO_MOE_INTER * sizeof(float));
        g_nhg4.scr_moe_out   = (float *)malloc((size_t)NEMO_TOP_K * NEMO_LATENT * sizeof(float));
    }
    snprintf(g_nhg4.loaded_dir, sizeof(g_nhg4.loaded_dir), "%s", hg4_dir);
    g_nhg4.cache_len = 0;
    nemo_dm_reset();
    g_nhg4.initialized = 1;
    return 0;
}

NEMO_API float nemo_forward_pipeline_clubbed(const int *token_ids, const int *seg_starts, int T, float *out_final_hidden) {
    double t0 = omp_get_wtime();
    float *cur = (float *)malloc((size_t)T * NEMO_D * sizeof(float));
    float *nxt = (float *)malloc((size_t)T * NEMO_D * sizeof(float));

    #pragma omp parallel for schedule(static) if (T >= 4)
    for (int t = 0; t < T; t++) {
        nhg4_embed_lookup_row(token_ids[t], cur + (size_t)t * NEMO_D);
    }
    for (int l = 0; l < g_nhg4.n_layers; l++) {
        nhg4_layer_t *lyr = &g_nhg4.layers[l];
        if (lyr->kind == 0)      nhg4_mamba_exec(lyr, cur, nxt, 0, T, seg_starts, 0);
        else if (lyr->kind == 1) nhg4_latent_moe_forward(&lyr->moe, &lyr->experts, cur, nxt, T);
        else                     nhg4_gqa_exec(&lyr->gqa, NULL, NULL, cur, nxt, 0, T, seg_starts, 0);
        float *tmp = cur; cur = nxt; nxt = tmp;
    }
    rms_norm_f32_bf16round(cur, g_nhg4.norm_f, out_final_hidden, T, NEMO_D, 1e-5f);

    free(cur); free(nxt);
    return (float)((omp_get_wtime() - t0) * 1000.0);
}

NEMO_API float nemo_forward_pipeline(const int *token_ids, int T, float *out_final_hidden) {
    return nemo_forward_pipeline_clubbed(token_ids, NULL, T, out_final_hidden);
}

NEMO_API float nemo_forward_embeddings(const float *in_embeds, int T, float *out_final_hidden) {
    double t0 = omp_get_wtime();
    float *cur = (float *)malloc((size_t)T * NEMO_D * sizeof(float));
    float *nxt = (float *)malloc((size_t)T * NEMO_D * sizeof(float));
    memcpy(cur, in_embeds, (size_t)T * NEMO_D * sizeof(float));

    for (int l = 0; l < g_nhg4.n_layers; l++) {
        nhg4_layer_t *lyr = &g_nhg4.layers[l];
        if (lyr->kind == 0)      nhg4_mamba_exec(lyr, cur, nxt, 0, T, NULL, 0);
        else if (lyr->kind == 1) nhg4_latent_moe_forward(&lyr->moe, &lyr->experts, cur, nxt, T);
        else                     nhg4_gqa_exec(&lyr->gqa, NULL, NULL, cur, nxt, 0, T, NULL, 0);
        float *tmp = cur; cur = nxt; nxt = tmp;
    }
    rms_norm_f32_bf16round(cur, g_nhg4.norm_f, out_final_hidden, T, NEMO_D, 1e-5f);

    free(cur); free(nxt);
    return (float)((omp_get_wtime() - t0) * 1000.0);
}

NEMO_API int nemo_cache_feed(const int *token_ids, int n_tokens, float *out_last_hidden, float *out_last_logits) {
    if (!g_nhg4.initialized || n_tokens <= 0 || g_nhg4.cache_len + n_tokens > NEMO_MAX_SEQ) return -1;
    float h_last[NEMO_D];
    int done = 0;
    while (done < n_tokens) {
        int chunk = (n_tokens - done > NEMO_PANEL_MAX) ? NEMO_PANEL_MAX : (n_tokens - done);
        float *cur = (float *)malloc((size_t)chunk * NEMO_D * sizeof(float));
        float *nxt = (float *)malloc((size_t)chunk * NEMO_D * sizeof(float));
        #pragma omp parallel for schedule(static) if (chunk >= 4)
        for (int t = 0; t < chunk; t++) nhg4_embed_lookup_row(token_ids[done + t], cur + (size_t)t * NEMO_D);

        for (int l = 0; l < g_nhg4.n_layers; l++) {
            nhg4_layer_t *lyr = &g_nhg4.layers[l];
            if (lyr->kind == 0)      nhg4_mamba_exec(lyr, cur, nxt, g_nhg4.cache_len, chunk, NULL, 1);
            else if (lyr->kind == 1) nhg4_latent_moe_forward(&lyr->moe, &lyr->experts, cur, nxt, chunk);
            else                     nhg4_gqa_exec(&lyr->gqa, lyr->k_cache_bf16, lyr->v_cache_bf16, cur, nxt, g_nhg4.cache_len, chunk, NULL, 1);
            float *tmp = cur; cur = nxt; nxt = tmp;
        }
        g_nhg4.cache_len += chunk;
        done += chunk;
        if (done == n_tokens) {
            rms_norm_f32_bf16round(cur + (size_t)(chunk - 1) * NEMO_D, g_nhg4.norm_f, h_last, 1, NEMO_D, 1e-5f);
            if (out_last_hidden) memcpy(out_last_hidden, h_last, NEMO_D * sizeof(float));
            if (out_last_logits) nhg4_matmul(&g_nhg4.lm_mat, h_last, out_last_logits, 1, 0, 1);
        }
        free(cur); free(nxt);
    }
    return g_nhg4.cache_len;
}

NEMO_API int nemo_generate_speculative(
    const int *prompt_ids, int prompt_len, int max_new_tokens, int eos_id, int draft_k,
    int *out_gen_ids, double *out_prefill_ms, double *out_decode_ms,
    int *out_accepted_drafts, int *out_total_drafts
);

NEMO_API int nemo_generate(
    const int *prompt_ids, int prompt_len, int max_new_tokens, int eos_id,
    int *out_gen_ids, double *out_prefill_ms, double *out_decode_ms
) {
    if (!g_nhg4.initialized || prompt_len <= 0 || prompt_len + max_new_tokens >= NEMO_MAX_SEQ) return -1;
    if (g_nhg4.has_mtp && getenv("CLOVER_NEMO_NO_MTP") == NULL && max_new_tokens >= 2 &&
        prompt_len + max_new_tokens + 4 < NEMO_MAX_SEQ) {
        return nemo_generate_speculative(prompt_ids, prompt_len, max_new_tokens, eos_id, 2,
                                         out_gen_ids, out_prefill_ms, out_decode_ms, NULL, NULL);
    }
    float *logits = (float *)malloc((size_t)NEMO_VOCAB * sizeof(float));
    nemo_cache_reset();

    double t0 = omp_get_wtime();
    if (nemo_cache_feed(prompt_ids, prompt_len, NULL, logits) < 0) { free(logits); return -1; }
    *out_prefill_ms = (omp_get_wtime() - t0) * 1000.0;

    double td0 = omp_get_wtime();
    int gen_n = 0;
    for (int step = 0; step < max_new_tokens; step++) {
        int best_id = 0; float best_v = logits[0];
        for (int v = 1; v < NEMO_VOCAB; v++) if (logits[v] > best_v) { best_v = logits[v]; best_id = v; }
        out_gen_ids[gen_n++] = best_id;
        if (best_id == eos_id || step + 1 >= max_new_tokens) break;
        nemo_cache_feed(&best_id, 1, NULL, logits);
    }
    *out_decode_ms = (omp_get_wtime() - td0) * 1000.0;
    free(logits);
    return gen_n;
}

NEMO_API int nemo_lm_head_logits(const float *h_last_4096, float *out_logits_131k, float *out_max_logit) {
    if (!g_nhg4.initialized || !h_last_4096) return -1;
    float *lg = out_logits_131k ? out_logits_131k : (float *)malloc((size_t)NEMO_VOCAB * sizeof(float));
    nhg4_matmul(&g_nhg4.lm_mat, h_last_4096, lg, 1, 0, 1);
    int best_id = 0;
    float best_v = lg[0];
    for (int v = 1; v < NEMO_VOCAB; v++) {
        if (lg[v] > best_v) { best_v = lg[v]; best_id = v; }
    }
    if (out_max_logit) *out_max_logit = best_v;
    if (!out_logits_131k) free(lg);
    return best_id;
}

NEMO_API float nemo_all_logits(const float *hidden_t_4096, int T, float *out_logits_t_131k) {
    double t0 = omp_get_wtime();
    nhg4_matmul(&g_nhg4.lm_mat, hidden_t_4096, out_logits_t_131k, T, 0, 1);
    return (float)((omp_get_wtime() - t0) * 1000.0);
}

NEMO_API float nemo_mtp_draft_forward(
    const int *next_token_ids, const float *backbone_hidden, int T, float *out_mtp_final_hidden
) {
    if (!g_nhg4.has_mtp) return -1.0f;
    double t0 = omp_get_wtime();
    float *e_raw    = (float *)malloc((size_t)T * NEMO_D * sizeof(float));
    float *e_normed = (float *)malloc((size_t)T * NEMO_D * sizeof(float));
    float *h_normed = (float *)malloc((size_t)T * NEMO_D * sizeof(float));
    float *eh_cat   = (float *)malloc((size_t)T * (2 * NEMO_D) * sizeof(float));
    float *mtp_in   = (float *)malloc((size_t)T * NEMO_D * sizeof(float));
    float *mtp_h0   = (float *)malloc((size_t)T * NEMO_D * sizeof(float));
    float *mtp_h1   = (float *)malloc((size_t)T * NEMO_D * sizeof(float));

    #pragma omp parallel for schedule(static) if (T >= 4)
    for (int t = 0; t < T; t++) nhg4_embed_lookup_row(next_token_ids[t], e_raw + (size_t)t * NEMO_D);
    rms_norm_f32_bf16round(e_raw, g_nhg4.mtp_enorm, e_normed, T, NEMO_D, 1e-5f);
    rms_norm_f32_bf16round(backbone_hidden, g_nhg4.mtp_hnorm, h_normed, T, NEMO_D, 1e-5f);

    for (int t = 0; t < T; t++) {
        memcpy(eh_cat + (size_t)t * (2 * NEMO_D), e_normed + (size_t)t * NEMO_D, NEMO_D * sizeof(float));
        memcpy(eh_cat + (size_t)t * (2 * NEMO_D) + NEMO_D, h_normed + (size_t)t * NEMO_D, NEMO_D * sizeof(float));
    }

    nhg4_matmul(&g_nhg4.mtp_eh_proj, eh_cat, mtp_in, T, 0, 1);
    nhg4_gqa_exec(&g_nhg4.mtp0_gqa, NULL, NULL, mtp_in, mtp_h0, 0, T, NULL, 0);
    nhg4_latent_moe_forward(&g_nhg4.mtp1_moe, &g_nhg4.experts_mtp1, mtp_h0, mtp_h1, T);
    rms_norm_f32_bf16round(mtp_h1, g_nhg4.mtp_final_ln, out_mtp_final_hidden, T, NEMO_D, 1e-5f);

    free(e_raw); free(e_normed); free(h_normed); free(eh_cat);
    free(mtp_in); free(mtp_h0); free(mtp_h1);
    return (float)((omp_get_wtime() - t0) * 1000.0);
}

static void nemo_mtp_feed_stateful(const int *next_token_ids, const float *backbone_hidden, int pos_off, int T, float *out_mtp_final_hidden) {
    float *e_raw    = (float *)malloc((size_t)T * NEMO_D * sizeof(float));
    float *e_normed = (float *)malloc((size_t)T * NEMO_D * sizeof(float));
    float *h_normed = (float *)malloc((size_t)T * NEMO_D * sizeof(float));
    float *eh_cat   = (float *)malloc((size_t)T * (2 * NEMO_D) * sizeof(float));
    float *mtp_in   = (float *)malloc((size_t)T * NEMO_D * sizeof(float));
    float *mtp_h0   = (float *)malloc((size_t)T * NEMO_D * sizeof(float));
    float *mtp_h1   = (float *)malloc((size_t)T * NEMO_D * sizeof(float));

    #pragma omp parallel for schedule(static) if (T >= 4)
    for (int t = 0; t < T; t++) nhg4_embed_lookup_row(next_token_ids[t], e_raw + (size_t)t * NEMO_D);
    rms_norm_f32_bf16round(e_raw, g_nhg4.mtp_enorm, e_normed, T, NEMO_D, 1e-5f);
    rms_norm_f32_bf16round(backbone_hidden, g_nhg4.mtp_hnorm, h_normed, T, NEMO_D, 1e-5f);

    for (int t = 0; t < T; t++) {
        memcpy(eh_cat + (size_t)t * (2 * NEMO_D), e_normed + (size_t)t * NEMO_D, NEMO_D * sizeof(float));
        memcpy(eh_cat + (size_t)t * (2 * NEMO_D) + NEMO_D, h_normed + (size_t)t * NEMO_D, NEMO_D * sizeof(float));
    }

    nhg4_matmul(&g_nhg4.mtp_eh_proj, eh_cat, mtp_in, T, 0, 1);
    nhg4_gqa_exec(&g_nhg4.mtp0_gqa, g_nhg4.mtp_k_cache_bf16, g_nhg4.mtp_v_cache_bf16, mtp_in, mtp_h0, pos_off, T, NULL, 1);
    nhg4_latent_moe_forward(&g_nhg4.mtp1_moe, &g_nhg4.experts_mtp1, mtp_h0, mtp_h1, T);
    if (out_mtp_final_hidden) {
        rms_norm_f32_bf16round(mtp_h1, g_nhg4.mtp_final_ln, out_mtp_final_hidden, T, NEMO_D, 1e-5f);
    }

    free(e_raw); free(e_normed); free(h_normed); free(eh_cat);
    free(mtp_in); free(mtp_h0); free(mtp_h1);
}

static int argmax_f32_vec(const float *v, int n) {
    int bi = 0; float bv = v[0];
    for (int i = 1; i < n; i++) if (v[i] > bv) { bv = v[i]; bi = i; }
    return bi;
}

static void nemo_ensure_spec_snaps(void) {
    for (int l = 0; l < g_nhg4.n_layers; l++) {
        nhg4_layer_t *lyr = &g_nhg4.layers[l];
        if (lyr->kind == 0 && !lyr->ssm_snaps) {
            lyr->ssm_snaps  = (float *)malloc((size_t)4 * NEMO_MAMBA_HEADS * NEMO_MAMBA_HDIM * NEMO_MAMBA_DSTATE * sizeof(float));
            lyr->conv_snaps = (float *)malloc((size_t)4 * 3 * NEMO_MAMBA_CONV * sizeof(float));
        }
    }
}

static void nemo_restore_mamba_snap(int snap_idx) {
    for (int l = 0; l < g_nhg4.n_layers; l++) {
        nhg4_layer_t *lyr = &g_nhg4.layers[l];
        if (lyr->kind != 0 || !lyr->ssm_snaps || !lyr->conv_snaps) continue;
        memcpy(lyr->conv_hist, lyr->conv_snaps + (size_t)snap_idx * 3 * NEMO_MAMBA_CONV, 3 * NEMO_MAMBA_CONV * sizeof(float));
        const float *src_ssm = lyr->ssm_snaps + (size_t)snap_idx * NEMO_MAMBA_HEADS * NEMO_MAMBA_HDIM * NEMO_MAMBA_DSTATE;
        #pragma omp parallel for schedule(static)
        for (int h = 0; h < NEMO_MAMBA_HEADS; h++) {
            size_t off = (size_t)h * NEMO_MAMBA_HDIM * NEMO_MAMBA_DSTATE;
            memcpy(lyr->ssm_state + off, src_ssm + off, (size_t)NEMO_MAMBA_HDIM * NEMO_MAMBA_DSTATE * sizeof(float));
        }
    }
}

/* Lossless Speculative Decode Inside C1 Using Native 2-Layer MTP (`draft_k = 1..3`) */
NEMO_API int nemo_generate_speculative(
    const int *prompt_ids, int prompt_len, int max_new_tokens, int eos_id, int draft_k,
    int *out_gen_ids, double *out_prefill_ms, double *out_decode_ms,
    int *out_accepted_drafts, int *out_total_drafts
) {
    if (!g_nhg4.initialized || !g_nhg4.has_mtp || draft_k <= 0) {
        if (out_accepted_drafts) *out_accepted_drafts = 0;
        if (out_total_drafts) *out_total_drafts = 0;
        return nemo_generate(prompt_ids, prompt_len, max_new_tokens, eos_id, out_gen_ids, out_prefill_ms, out_decode_ms);
    }
    if (draft_k > 3) draft_k = 3;
    if (prompt_len <= 0 || prompt_len + max_new_tokens + draft_k + 2 >= NEMO_MAX_SEQ) return -1;

    nemo_ensure_spec_snaps();
    nemo_cache_reset();

    double t0 = omp_get_wtime();
    float *cur_p = (float *)malloc((size_t)prompt_len * NEMO_D * sizeof(float));
    float *nxt_p = (float *)malloc((size_t)prompt_len * NEMO_D * sizeof(float));
    float *h_prompt = (float *)malloc((size_t)prompt_len * NEMO_D * sizeof(float));
    #pragma omp parallel for schedule(static) if (prompt_len >= 4)
    for (int t = 0; t < prompt_len; t++) nhg4_embed_lookup_row(prompt_ids[t], cur_p + (size_t)t * NEMO_D);

    g_nhg4.spec_save_snaps = 0;
    for (int l = 0; l < g_nhg4.n_layers; l++) {
        nhg4_layer_t *lyr = &g_nhg4.layers[l];
        if (lyr->kind == 0)      nhg4_mamba_exec(lyr, cur_p, nxt_p, 0, prompt_len, NULL, 1);
        else if (lyr->kind == 1) nhg4_latent_moe_forward(&lyr->moe, &lyr->experts, cur_p, nxt_p, prompt_len);
        else                     nhg4_gqa_exec(&lyr->gqa, lyr->k_cache_bf16, lyr->v_cache_bf16, cur_p, nxt_p, 0, prompt_len, NULL, 1);
        float *tmp = cur_p; cur_p = nxt_p; nxt_p = tmp;
    }
    g_nhg4.cache_len = prompt_len;
    rms_norm_f32_bf16round(cur_p, g_nhg4.norm_f, h_prompt, prompt_len, NEMO_D, 1e-5f);

    /* Warm MTP causal KV cache on prompt[1..P-1] paired with h_prompt[0..P-2] */
    if (prompt_len >= 2) {
        nemo_mtp_feed_stateful(prompt_ids + 1, h_prompt, 0, prompt_len - 1, NULL);
        g_nhg4.mtp_cache_len = prompt_len - 1;
    }

    float *logits_1 = (float *)malloc((size_t)NEMO_VOCAB * sizeof(float));
    float h_prev[NEMO_D];
    memcpy(h_prev, h_prompt + (size_t)(prompt_len - 1) * NEMO_D, NEMO_D * sizeof(float));
    nhg4_matmul(&g_nhg4.lm_mat, h_prev, logits_1, 1, 0, 1);
    free(cur_p); free(nxt_p); free(h_prompt);
    *out_prefill_ms = (omp_get_wtime() - t0) * 1000.0;

    double td0 = omp_get_wtime();
    int gen_n = 0, acc_tot = 0, draft_tot = 0;
    int cur_tok = argmax_f32_vec(logits_1, NEMO_VOCAB);
    out_gen_ids[gen_n++] = cur_tok;

    float *ver_cur = (float *)malloc((size_t)(draft_k + 1) * NEMO_D * sizeof(float));
    float *ver_nxt = (float *)malloc((size_t)(draft_k + 1) * NEMO_D * sizeof(float));
    float *ver_h   = (float *)malloc((size_t)(draft_k + 1) * NEMO_D * sizeof(float));
    float *ver_lg  = (float *)malloc((size_t)(draft_k + 1) * NEMO_VOCAB * sizeof(float));

    while (gen_n < max_new_tokens && cur_tok != eos_id) {
        int rem = max_new_tokens - gen_n;
        int K = (draft_k < rem) ? draft_k : rem;

        /* 1. Autoregressive K-step MTP draft from (cur_tok, h_prev) */
        int drafts[4];
        float mtp_h[NEMO_D];
        int d_in = cur_tok;
        const float *h_cond = h_prev;
        int mtp_base = g_nhg4.mtp_cache_len;
        for (int j = 0; j < K; j++) {
            nemo_mtp_feed_stateful(&d_in, h_cond, mtp_base + j, 1, mtp_h);
            nhg4_matmul(&g_nhg4.lm_mat, mtp_h, logits_1, 1, 0, 1);
            drafts[j] = argmax_f32_vec(logits_1, NEMO_VOCAB);
            d_in = drafts[j];
            h_cond = mtp_h;
        }
        draft_tot += K;

        /* 2. Batched 88-layer backbone verification on batch = [cur_tok, drafts[0..K-1]] (B = K + 1) */
        int batch[5];
        batch[0] = cur_tok;
        for (int j = 0; j < K; j++) batch[j + 1] = drafts[j];
        int B = K + 1;
        int base_pos = g_nhg4.cache_len;

        for (int t = 0; t < B; t++) nhg4_embed_lookup_row(batch[t], ver_cur + (size_t)t * NEMO_D);
        g_nhg4.spec_save_snaps = K;
        for (int l = 0; l < g_nhg4.n_layers; l++) {
            nhg4_layer_t *lyr = &g_nhg4.layers[l];
            if (lyr->kind == 0)      nhg4_mamba_exec(lyr, ver_cur, ver_nxt, base_pos, B, NULL, 1);
            else if (lyr->kind == 1) nhg4_latent_moe_forward(&lyr->moe, &lyr->experts, ver_cur, ver_nxt, B);
            else                     nhg4_gqa_exec(&lyr->gqa, lyr->k_cache_bf16, lyr->v_cache_bf16, ver_cur, ver_nxt, base_pos, B, NULL, 1);
            float *tmp = ver_cur; ver_cur = ver_nxt; ver_nxt = tmp;
        }
        g_nhg4.spec_save_snaps = 0;
        rms_norm_f32_bf16round(ver_cur, g_nhg4.norm_f, ver_h, B, NEMO_D, 1e-5f);
        nhg4_matmul(&g_nhg4.lm_mat, ver_h, ver_lg, B, 0, 1);

        /* 3. Exact greedy verification & Mamba/GQA state reconciliation */
        int matched_all = 1;
        for (int j = 0; j < K; j++) {
            int gt = argmax_f32_vec(ver_lg + (size_t)j * NEMO_VOCAB, NEMO_VOCAB);
            if (drafts[j] == gt) {
                acc_tot++;
                out_gen_ids[gen_n++] = gt;
                if (gt == eos_id || gen_n >= max_new_tokens) {
                    nemo_restore_mamba_snap(j);
                    g_nhg4.cache_len = base_pos + j + 1;
                    cur_tok = gt;
                    matched_all = 0;
                    break;
                }
            } else {
                /* First draft mismatch at j: rollback Mamba/GQA to state after batch[0..j] and emit true C1 token gt! */
                nemo_restore_mamba_snap(j);
                g_nhg4.cache_len = base_pos + j + 1;
                /* Re-align MTP KV cache on true backbone states ver_h[0..j-1] if j >= 1 */
                if (j >= 1) {
                    nemo_mtp_feed_stateful(batch + 1, ver_h, mtp_base + 1, j, NULL);
                }
                g_nhg4.mtp_cache_len = mtp_base + 1 + j;
                cur_tok = gt;
                memcpy(h_prev, ver_h + (size_t)j * NEMO_D, NEMO_D * sizeof(float));
                out_gen_ids[gen_n++] = cur_tok;
                matched_all = 0;
                break;
            }
        }

        if (matched_all) {
            /* All K drafts accepted! State is already at base_pos + K + 1; emit C1 bonus token if budget remains! */
            g_nhg4.cache_len = base_pos + B;
            nemo_mtp_feed_stateful(batch + 1, ver_h, mtp_base + 1, K, NULL);
            g_nhg4.mtp_cache_len = mtp_base + 1 + K;
            int bonus = argmax_f32_vec(ver_lg + (size_t)K * NEMO_VOCAB, NEMO_VOCAB);
            cur_tok = bonus;
            memcpy(h_prev, ver_h + (size_t)K * NEMO_D, NEMO_D * sizeof(float));
            if (gen_n < max_new_tokens) {
                out_gen_ids[gen_n++] = cur_tok;
            }
        }
    }

    *out_decode_ms = (omp_get_wtime() - td0) * 1000.0;
    if (out_accepted_drafts) *out_accepted_drafts = acc_tot;
    if (out_total_drafts) *out_total_drafts = draft_tot;
    free(logits_1); free(ver_cur); free(ver_nxt); free(ver_h); free(ver_lg);
    return gen_n;
}
