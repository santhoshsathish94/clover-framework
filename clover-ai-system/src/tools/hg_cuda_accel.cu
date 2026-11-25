/*
 * Drop-in CUDA (`sm_89`) Matrix-Multiply & MoE VRAM Cache Accelerator (`libhg_cuda_accel.so`)
 * for Clover's Native C11 `HG4-16R` / `HG8-16R` / `HG2-16R` Inference Engines.
 *
 * Partitioned across GPUs via `CLOVER_CUDA_GPUS` (e.g. "0" for Clover-35B, "1,2,3" for Nemotron-120B,
 * "4,5,6,7" for Kimi-K3.1-Pro) so all 3 models remain resident in 8x RTX 4090 VRAM simultaneously!
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

static void release_host_pages(const void *p, size_t len) {
    if (!p || len < 65536) return;
    uintptr_t a = (uintptr_t)p & ~(uintptr_t)4095u;
    size_t span = len + ((uintptr_t)p - a);
    madvise((void *)a, span & ~(size_t)4095u, MADV_DONTNEED);
}

extern "C" {
int hg_cuda_try_matmul(
    const uint8_t *h_tiles, const uint16_t *h_scales,
    uint32_t R, uint32_t K, uint32_t bits,
    const int8_t *h_xq, const float *h_xsc,
    float *h_Y, int T, int act_mode
);

int hg_cuda_try_moe_pairs(
    const uint8_t *h_exp_base, uint32_t n_exp_total, size_t exp_stride,
    size_t up_tiles_off, size_t dn_blob_off,
    uint32_t R_up, uint32_t K_up, uint32_t R_dn, uint32_t K_dn,
    uint32_t bits, int act_mode, int bf16_dn_round,
    const int8_t *h_xq, const float *h_xsc, int T,
    const int *h_pair_tok, const int *h_pair_slot, const float *h_pair_w, int n_pairs,
    float *h_out_add /* [T * R_dn] accumulated onto existing h_out_add */
);
}

__constant__ int8_t c_lut16_acc[16];

__device__ __forceinline__ float d_fp16_f32(uint16_t h) {
    return __half2float(*reinterpret_cast<const __half *>(&h));
}

__device__ __forceinline__ float d_bf16_round(float f) {
    uint32_t u = __float_as_uint(f);
    u += 0x7FFFu + ((u >> 16) & 1u);
    u &= 0xFFFF0000u;
    return __uint_as_float(u);
}

__device__ __forceinline__ int d_lut16_lo(uint32_t pk, int my_lut_u8) {
    int b0 = __shfl_sync(0xffffffffu, my_lut_u8, (int)((pk >>  0) & 0x0Fu), 16);
    int b1 = __shfl_sync(0xffffffffu, my_lut_u8, (int)((pk >>  8) & 0x0Fu), 16);
    int b2 = __shfl_sync(0xffffffffu, my_lut_u8, (int)((pk >> 16) & 0x0Fu), 16);
    int b3 = __shfl_sync(0xffffffffu, my_lut_u8, (int)((pk >> 24) & 0x0Fu), 16);
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

__device__ __forceinline__ int d_lut16_hi(uint32_t pk, int my_lut_u8) {
    int b0 = __shfl_sync(0xffffffffu, my_lut_u8, (int)((pk >>  4) & 0x0Fu), 16);
    int b1 = __shfl_sync(0xffffffffu, my_lut_u8, (int)((pk >> 12) & 0x0Fu), 16);
    int b2 = __shfl_sync(0xffffffffu, my_lut_u8, (int)((pk >> 20) & 0x0Fu), 16);
    int b3 = __shfl_sync(0xffffffffu, my_lut_u8, (int)((pk >> 28) & 0x0Fu), 16);
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

__device__ __forceinline__ int d_lut4_crumb(uint32_t pk, int shift, int lut4_pk) {
    int i0 = (pk >> (shift +  0)) & 0x03u;
    int i1 = (pk >> (shift +  8)) & 0x03u;
    int i2 = (pk >> (shift + 16)) & 0x03u;
    int i3 = (pk >> (shift + 24)) & 0x03u;
    return ((lut4_pk >> (i0 * 8)) & 0xFF)
         | (((lut4_pk >> (i1 * 8)) & 0xFF) << 8)
         | (((lut4_pk >> (i2 * 8)) & 0xFF) << 16)
         | (((lut4_pk >> (i3 * 8)) & 0xFF) << 24);
}

/*
 * `act_mode`:
 *   0 = FP32 linear
 *   1 = SwiGLU  (out_R = R/2)
 *   2 = SiTU-GLU (out_R = R/2)
 *   3 = BF16-rounded ReLU^2 (Nemotron)
 *   4 = BF16-rounded linear (Nemotron)
 */
__global__ void k_hg_accel_16r(
    const uint8_t * __restrict__ tiles,
    const uint16_t * __restrict__ scales_fp16,
    int R, int K, int bits,
    const int8_t * __restrict__ X_q,
    const float * __restrict__ X_sc,
    float * __restrict__ Y,
    int T, int act_mode
) {
    int rg = blockIdx.x;
    int t_base = blockIdx.y * 4;
    int n_rg = R >> 4;
    if (rg >= n_rg || t_base >= T) return;

    int tblk = (t_base + 4 <= T) ? 4 : (T - t_base);
    int warp_id = threadIdx.x >> 5; /* 0..7 */
    int lane    = threadIdx.x & 31;
    int r       = lane & 15;
    int sub     = lane >> 4;
    int nb      = K >> 7;
    int out_R   = (act_mode == 1 || act_mode == 2) ? (R >> 1) : R;
    size_t bstride = (bits == 8) ? 2048u : ((bits == 2) ? 512u : 1024u);

    const uint8_t  *wt  = tiles + (size_t)rg * (size_t)nb * bstride;
    const uint16_t *wsc = scales_fp16 + (size_t)rg * (size_t)nb * 16;
    int my_lut_u8 = (int)(uint8_t)c_lut16_acc[r];
    const int lut4_pk = 0x7F26DA81;

    float vs[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int b = warp_id; b < nb; b += 8) {
        const uint32_t *wb32 = reinterpret_cast<const uint32_t *>(wt + (size_t)b * bstride);
        float ws = d_fp16_f32(__ldg(&wsc[b * 16 + r]));
        int acc[4] = {0, 0, 0, 0};

        if (bits == 4) {
            #pragma unroll
            for (int it = 0; it < 8; it++) {
                uint32_t pk = __ldg(&wb32[it * 32 + lane]);
                int wlo = d_lut16_lo(pk, my_lut_u8);
                int whi = d_lut16_hi(pk, my_lut_u8);
                int s = (it << 1) + sub;
                for (int ti = 0; ti < tblk; ti++) {
                    const int *xp = reinterpret_cast<const int *>(X_q + (size_t)(t_base + ti) * K + b * 128);
                    acc[ti] = __dp4a(wlo, __ldg(&xp[2 * s + 0]), acc[ti]);
                    acc[ti] = __dp4a(whi, __ldg(&xp[2 * s + 1]), acc[ti]);
                }
            }
        } else if (bits == 8) {
            #pragma unroll
            for (int it = 0; it < 16; it++) {
                uint32_t wu = __ldg(&wb32[it * 32 + lane]) ^ 0x80808080u;
                int s = (it << 1) + sub;
                for (int ti = 0; ti < tblk; ti++) {
                    const int *xp = reinterpret_cast<const int *>(X_q + (size_t)(t_base + ti) * K + b * 128);
                    acc[ti] = __dp4a((int)wu, __ldg(&xp[s]), acc[ti]);
                }
            }
        } else {
            #pragma unroll
            for (int it = 0; it < 4; it++) {
                uint32_t pk = __ldg(&wb32[it * 32 + lane]);
                int w0 = d_lut4_crumb(pk, 0, lut4_pk);
                int w1 = d_lut4_crumb(pk, 2, lut4_pk);
                int w2 = d_lut4_crumb(pk, 4, lut4_pk);
                int w3 = d_lut4_crumb(pk, 6, lut4_pk);
                int s = (it << 1) + sub;
                for (int ti = 0; ti < tblk; ti++) {
                    const int *xp = reinterpret_cast<const int *>(X_q + (size_t)(t_base + ti) * K + b * 128);
                    acc[ti] = __dp4a(w0, __ldg(&xp[4 * s + 0]), acc[ti]);
                    acc[ti] = __dp4a(w1, __ldg(&xp[4 * s + 1]), acc[ti]);
                    acc[ti] = __dp4a(w2, __ldg(&xp[4 * s + 2]), acc[ti]);
                    acc[ti] = __dp4a(w3, __ldg(&xp[4 * s + 3]), acc[ti]);
                }
            }
        }

        for (int ti = 0; ti < tblk; ti++) {
            int sum32 = acc[ti] + __shfl_xor_sync(0xffffffffu, acc[ti], 16);
            if (sub == 0) vs[ti] += (float)sum32 * (ws * X_sc[(t_base + ti) * nb + b]);
        }
    }

    __shared__ float smem[8][4][16];
    if (sub == 0) {
        for (int ti = 0; ti < tblk; ti++) smem[warp_id][ti][r] = vs[ti];
    }
    __syncthreads();

    if (warp_id == 0 && sub == 0) {
        for (int ti = 0; ti < tblk; ti++) {
            smem[0][ti][r] = (smem[0][ti][r] + smem[1][ti][r] + smem[2][ti][r] + smem[3][ti][r]) +
                             (smem[4][ti][r] + smem[5][ti][r] + smem[6][ti][r] + smem[7][ti][r]);
        }
        __syncwarp(0x0000ffffu);
        for (int ti = 0; ti < tblk; ti++) {
            float *yt = Y + (size_t)(t_base + ti) * out_R;
            if (act_mode == 1) {
                if (r < 8) {
                    float g = smem[0][ti][r], u = smem[0][ti][r + 8];
                    yt[rg * 8 + r] = (g / (1.0f + expf(-g))) * u;
                }
            } else if (act_mode == 2) {
                if (r < 8) {
                    float g = smem[0][ti][r], u = smem[0][ti][r + 8];
                    yt[rg * 8 + r] = ((4.0f * tanhf(g * 0.25f)) / (1.0f + expf(-g))) * (25.0f * tanhf(u * 0.04f));
                }
            } else if (act_mode == 3) {
                float v = d_bf16_round(smem[0][ti][r]);
                v = v > 0.0f ? v : 0.0f;
                yt[rg * 16 + r] = d_bf16_round(v * v);
            } else if (act_mode == 4) {
                yt[rg * 16 + r] = d_bf16_round(smem[0][ti][r]);
            } else {
                yt[rg * 16 + r] = smem[0][ti][r];
            }
        }
    }
}

#define HT_CAP 131072u /* power-of-2 table holding up to ~90K matrix pointers */

__global__ void k_moe_pairs_up(
    const uint8_t * __restrict__ exp_base, size_t exp_stride, size_t up_tiles_off,
    int R_up, int K_up, int bits, int act_mode,
    const int8_t * __restrict__ X_q, const float * __restrict__ X_sc,
    const int * __restrict__ pair_tok, const int * __restrict__ pair_slot,
    float * __restrict__ out_hid, int n_pairs
) {
    int rg = blockIdx.x, p = blockIdx.y;
    if (rg >= (R_up >> 4) || p >= n_pairs) return;
    int tok  = pair_tok[p];
    int slot = pair_slot[p];
    int nb   = K_up >> 7;
    int out_R = (act_mode == 1 || act_mode == 2) ? (R_up >> 1) : R_up;
    size_t bstride = (bits == 2) ? 512u : 1024u;

    const uint8_t  *ep  = exp_base + (size_t)slot * exp_stride;
    const uint16_t *wsc = reinterpret_cast<const uint16_t *>(ep) + (size_t)rg * nb * 16;
    const uint8_t  *wt  = ep + up_tiles_off + (size_t)rg * nb * bstride;

    int warp_id = threadIdx.x >> 5; /* 0..3 */
    int lane    = threadIdx.x & 31;
    int r       = lane & 15;
    int sub     = lane >> 4;
    int my_lut_u8 = (int)(uint8_t)c_lut16_acc[r];
    const int lut4_pk = 0x7F26DA81;
    const int *xp_base = reinterpret_cast<const int *>(X_q + (size_t)tok * K_up);

    float vs = 0.0f;
    for (int b = warp_id; b < nb; b += 4) {
        const uint32_t *wb32 = reinterpret_cast<const uint32_t *>(wt + (size_t)b * bstride);
        float ws = d_fp16_f32(__ldg(&wsc[b * 16 + r]));
        int acc = 0;
        if (bits == 4) {
            #pragma unroll
            for (int it = 0; it < 8; it++) {
                uint32_t pk = __ldg(&wb32[it * 32 + lane]);
                int s = (it << 1) + sub;
                acc = __dp4a(d_lut16_lo(pk, my_lut_u8), __ldg(&xp_base[b * 32 + 2 * s + 0]), acc);
                acc = __dp4a(d_lut16_hi(pk, my_lut_u8), __ldg(&xp_base[b * 32 + 2 * s + 1]), acc);
            }
        } else {
            #pragma unroll
            for (int it = 0; it < 4; it++) {
                uint32_t pk = __ldg(&wb32[it * 32 + lane]);
                int s = (it << 1) + sub;
                acc = __dp4a(d_lut4_crumb(pk, 0, lut4_pk), __ldg(&xp_base[b * 32 + 4 * s + 0]), acc);
                acc = __dp4a(d_lut4_crumb(pk, 2, lut4_pk), __ldg(&xp_base[b * 32 + 4 * s + 1]), acc);
                acc = __dp4a(d_lut4_crumb(pk, 4, lut4_pk), __ldg(&xp_base[b * 32 + 4 * s + 2]), acc);
                acc = __dp4a(d_lut4_crumb(pk, 6, lut4_pk), __ldg(&xp_base[b * 32 + 4 * s + 3]), acc);
            }
        }
        int sum32 = acc + __shfl_xor_sync(0xffffffffu, acc, 16);
        if (sub == 0) vs += (float)sum32 * (ws * X_sc[tok * nb + b]);
    }
    __shared__ float sm[4][16];
    if (sub == 0) sm[warp_id][r] = vs;
    __syncthreads();
    if (warp_id == 0 && sub == 0) {
        float tot = (sm[0][r] + sm[1][r]) + (sm[2][r] + sm[3][r]);
        sm[0][r] = tot;
        __syncwarp(0x0000ffffu);
        float *yt = out_hid + (size_t)p * out_R;
        if (act_mode == 1) {
            if (r < 8) { float g = sm[0][r], u = sm[0][r + 8]; yt[rg * 8 + r] = (g / (1.0f + expf(-g))) * u; }
        } else if (act_mode == 2) {
            if (r < 8) {
                float g = sm[0][r], u = sm[0][r + 8];
                yt[rg * 8 + r] = ((4.0f * tanhf(g * 0.25f)) / (1.0f + expf(-g))) * (25.0f * tanhf(u * 0.04f));
            }
        } else {
            float v = d_bf16_round(sm[0][r]);
            v = v > 0.0f ? v : 0.0f;
            yt[rg * 16 + r] = d_bf16_round(v * v);
        }
    }
}

__global__ void k_moe_pairs_rot_quant(
    const uint8_t * __restrict__ exp_base, size_t exp_stride, size_t dn_blob_off,
    const float * __restrict__ hid, const int * __restrict__ pair_slot,
    int K_dn, int n_pairs, int8_t * __restrict__ Z_q, float * __restrict__ Z_sc
) {
    int b = blockIdx.x, p = blockIdx.y, tid = threadIdx.x;
    int nb = K_dn >> 7;
    if (b >= nb || p >= n_pairs) return;
    int slot = pair_slot[p];
    const float *pre_diag = reinterpret_cast<const float *>(exp_base + (size_t)slot * exp_stride + dn_blob_off + 8u);

    __shared__ float s_abs[128];
    __shared__ int32_t s_rot[128];
    __shared__ int32_t s_rmax[128];

    float val = hid[(size_t)p * K_dn + b * 128 + tid] * pre_diag[b * 128 + tid];
    s_abs[tid] = fabsf(val);
    __syncthreads();
    for (int st = 64; st > 0; st >>= 1) {
        if (tid < st && s_abs[tid + st] > s_abs[tid]) s_abs[tid] = s_abs[tid + st];
        __syncthreads();
    }
    float max_abs = s_abs[0];
    if (max_abs < 1e-12f) {
        if (tid == 0) Z_sc[p * nb + b] = 0.0f;
        Z_q[(size_t)p * K_dn + b * 128 + tid] = 0;
        return;
    }
    float in_sc = max_abs * (1.0f / 2048.0f);
    s_rot[tid] = (int32_t)rintf(fminf(2048.0f, fmaxf(-2048.0f, val * (2048.0f / max_abs))));
    __syncthreads();
    for (int h = 1; h < 128; h <<= 1) {
        int grp = tid / h, k = tid & (h - 1);
        int i0 = grp * (h << 1) + k, i1 = i0 + h;
        int32_t a = (tid < 64) ? s_rot[i0] : 0;
        int32_t bv = (tid < 64) ? s_rot[i1] : 0;
        __syncthreads();
        if (tid < 64) { s_rot[i0] = a + bv; s_rot[i1] = a - bv; }
        __syncthreads();
    }
    int32_t rv = s_rot[tid];
    s_rmax[tid] = rv < 0 ? -rv : rv;
    __syncthreads();
    for (int st = 64; st > 0; st >>= 1) {
        if (tid < st && s_rmax[tid + st] > s_rmax[tid]) s_rmax[tid] = s_rmax[tid + st];
        __syncthreads();
    }
    int32_t max_rot = s_rmax[0];
    if (max_rot == 0) {
        if (tid == 0) Z_sc[p * nb + b] = 0.0f;
        Z_q[(size_t)p * K_dn + b * 128 + tid] = 0;
    } else {
        if (tid == 0) Z_sc[p * nb + b] = (in_sc * (float)max_rot) * (0.08838834764831845f / 127.0f);
        Z_q[(size_t)p * K_dn + b * 128 + tid] = (int8_t)rintf(fminf(127.0f, fmaxf(-127.0f, (float)rv * (127.0f / (float)max_rot))));
    }
}

__global__ void k_moe_pairs_down(
    const uint8_t * __restrict__ exp_base, size_t exp_stride, size_t dn_blob_off,
    int R_dn, int K_dn, int bits, int bf16_dn_round,
    const int8_t * __restrict__ Z_q, const float * __restrict__ Z_sc,
    const int * __restrict__ pair_tok, const int * __restrict__ pair_slot, const float * __restrict__ pair_w,
    float * __restrict__ out_sum, int n_pairs
) {
    int rg = blockIdx.x, p = blockIdx.y;
    if (rg >= (R_dn >> 4) || p >= n_pairs) return;
    int tok  = pair_tok[p];
    int slot = pair_slot[p];
    int nb   = K_dn >> 7;
    size_t bstride = (bits == 2) ? 512u : 1024u;

    const uint8_t  *dn  = exp_base + (size_t)slot * exp_stride + dn_blob_off;
    const uint16_t *wsc = reinterpret_cast<const uint16_t *>(dn + 8u + (size_t)K_dn * 4u) + (size_t)rg * nb * 16;
    const uint8_t  *wt  = dn + 8u + (size_t)K_dn * 4u + (size_t)R_dn * nb * 2u + (size_t)rg * nb * bstride;

    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;
    int r       = lane & 15;
    int sub     = lane >> 4;
    int my_lut_u8 = (int)(uint8_t)c_lut16_acc[r];
    const int lut4_pk = 0x7F26DA81;
    const int *zp_base = reinterpret_cast<const int *>(Z_q + (size_t)p * K_dn);

    float vs = 0.0f;
    for (int b = warp_id; b < nb; b += 4) {
        const uint32_t *wb32 = reinterpret_cast<const uint32_t *>(wt + (size_t)b * bstride);
        float ws = d_fp16_f32(__ldg(&wsc[b * 16 + r]));
        int acc = 0;
        if (bits == 4) {
            #pragma unroll
            for (int it = 0; it < 8; it++) {
                uint32_t pk = __ldg(&wb32[it * 32 + lane]);
                int s = (it << 1) + sub;
                acc = __dp4a(d_lut16_lo(pk, my_lut_u8), __ldg(&zp_base[b * 32 + 2 * s + 0]), acc);
                acc = __dp4a(d_lut16_hi(pk, my_lut_u8), __ldg(&zp_base[b * 32 + 2 * s + 1]), acc);
            }
        } else {
            #pragma unroll
            for (int it = 0; it < 4; it++) {
                uint32_t pk = __ldg(&wb32[it * 32 + lane]);
                int s = (it << 1) + sub;
                acc = __dp4a(d_lut4_crumb(pk, 0, lut4_pk), __ldg(&zp_base[b * 32 + 4 * s + 0]), acc);
                acc = __dp4a(d_lut4_crumb(pk, 2, lut4_pk), __ldg(&zp_base[b * 32 + 4 * s + 1]), acc);
                acc = __dp4a(d_lut4_crumb(pk, 4, lut4_pk), __ldg(&zp_base[b * 32 + 4 * s + 2]), acc);
                acc = __dp4a(d_lut4_crumb(pk, 6, lut4_pk), __ldg(&zp_base[b * 32 + 4 * s + 3]), acc);
            }
        }
        int sum32 = acc + __shfl_xor_sync(0xffffffffu, acc, 16);
        if (sub == 0) vs += (float)sum32 * (ws * Z_sc[p * nb + b]);
    }
    __shared__ float sm[4][16];
    if (sub == 0) sm[warp_id][r] = vs;
    __syncthreads();
    if (warp_id == 0 && sub == 0) {
        float tot = (sm[0][r] + sm[1][r]) + (sm[2][r] + sm[3][r]);
        if (bf16_dn_round) tot = d_bf16_round(tot);
        atomicAdd(&out_sum[(size_t)tok * R_dn + rg * 16 + r], tot * pair_w[p]);
    }
}

typedef struct {
    const uint8_t *h_tiles;
    uint8_t  *d_tiles;
    uint16_t *d_scales;
    int       dev;
    int       status; /* 0 = empty, 1 = resident in VRAM, -1 = skipped (VRAM full) */
} cache_slot_t;

static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static int g_inited = 0;
static int g_enabled = 0;
static int g_gpus[8];
static int g_n_gpus = 0;
static int g_rr_gpu = 0;
static cache_slot_t g_ht[HT_CAP];

static int8_t *g_dxq[8], *g_dzq[8];
static float  *g_dxsc[8], *g_dzsc[8], *g_dhid[8], *g_dy[8], *g_dpw[8];
static int    *g_dptok[8], *g_dpslot[8];
static size_t  g_dxq_cap[8], g_dxsc_cap[8], g_dy_cap[8];
static size_t  g_dzq_cap[8], g_dzsc_cap[8], g_dhid_cap[8], g_dpair_cap[8];
static cudaStream_t g_st[8];

static void ensure_init_locked(void) {
    if (g_inited) return;
    g_inited = 1;
    const char *env = getenv("CLOVER_CUDA_GPUS");
    if (env && (strcmp(env, "none") == 0 || strcmp(env, "off") == 0 || strcmp(env, "cpu") == 0)) {
        g_enabled = 0;
        return;
    }
    static const int8_t lut16[16] = {-127,-96,-75,-58,-44,-31,-18,-6,6,18,31,44,58,75,96,127};
    if (!env || !env[0] || strcmp(env, "auto") == 0 || strcmp(env, "all") == 0) {
        int ndev = 0;
        if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev <= 0) {
            cudaGetLastError();
            g_enabled = 0;
            return;
        }
        if (ndev > 8) ndev = 8;
        for (int id = 0; id < ndev; id++) {
            if (cudaSetDevice(id) == cudaSuccess) {
                cudaMemcpyToSymbol(c_lut16_acc, lut16, 16);
                cudaStreamCreate(&g_st[id]);
                g_gpus[g_n_gpus++] = id;
            }
        }
        g_enabled = (g_n_gpus > 0);
        return;
    }
    char buf[128];
    snprintf(buf, sizeof(buf), "%s", env);
    char *tok = strtok(buf, ",");
    while (tok && g_n_gpus < 8) {
        int id = atoi(tok);
        if (cudaSetDevice(id) == cudaSuccess) {
            cudaMemcpyToSymbol(c_lut16_acc, lut16, 16);
            cudaStreamCreate(&g_st[id]);
            g_gpus[g_n_gpus++] = id;
        }
        tok = strtok(NULL, ",");
    }
    g_enabled = (g_n_gpus > 0);
}

static inline uint32_t ptr_hash(const void *p) {
    uint64_t x = (uint64_t)(uintptr_t)p;
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    return (uint32_t)x & (HT_CAP - 1u);
}

extern "C" int hg_cuda_try_matmul(
    const uint8_t *h_tiles, const uint16_t *h_scales,
    uint32_t R, uint32_t K, uint32_t bits,
    const int8_t *h_xq, const float *h_xsc,
    float *h_Y, int T, int act_mode
) {
    /* Only offload trunk / head / large shared matrices (>= 1 MiB or T >= 4) so tiny single-expert CPU L2 calls stay zero-overhead */
    size_t nb = K / 128u;
    size_t bstride = (bits == 8u) ? 2048u : ((bits == 2u) ? 512u : 1024u);
    size_t tiles_b = (size_t)(R / 16u) * nb * bstride;
    size_t scales_b = (size_t)R * nb * 2u;
    if (T <= 0 || (R & 15u) != 0 || (K & 127u) != 0) return 0;
    if (tiles_b < (size_t)128 * 1024 && T < 4) return 0;

    pthread_mutex_lock(&g_mu);
    ensure_init_locked();
    if (!g_enabled) { pthread_mutex_unlock(&g_mu); return 0; }

    uint32_t idx = ptr_hash(h_tiles);
    cache_slot_t *slot = NULL;
    for (uint32_t probe = 0; probe < 1024u; probe++) {
        cache_slot_t *s = &g_ht[(idx + probe) & (HT_CAP - 1u)];
        if (s->status == 0) { slot = s; break; }
        if (s->h_tiles == h_tiles) { slot = s; break; }
    }
    if (!slot || slot->status == -1) { pthread_mutex_unlock(&g_mu); return 0; }

    if (slot->status == 0) {
        int placed = 0;
        for (int a = 0; a < g_n_gpus; a++) {
            int dev = g_gpus[(g_rr_gpu + a) % g_n_gpus];
            cudaSetDevice(dev);
            size_t free_b = 0, tot_b = 0;
            cudaMemGetInfo(&free_b, &tot_b);
            if (free_b > tiles_b + scales_b + (size_t)640 * 1024 * 1024) {
                uint8_t *dt = NULL; uint16_t *ds = NULL;
                if (cudaMalloc(&dt, tiles_b) == cudaSuccess && cudaMalloc(&ds, scales_b) == cudaSuccess) {
                    cudaMemcpy(dt, h_tiles, tiles_b, cudaMemcpyHostToDevice);
                    cudaMemcpy(ds, h_scales, scales_b, cudaMemcpyHostToDevice);
                    if (R < 65536u && tiles_b >= (size_t)16 * 1024 * 1024) {
                        release_host_pages(h_tiles, tiles_b);
                        release_host_pages(h_scales, scales_b);
                    }
                    slot->h_tiles = h_tiles;
                    slot->d_tiles = dt;
                    slot->d_scales = ds;
                    slot->dev = dev;
                    slot->status = 1;
                    g_rr_gpu = (g_rr_gpu + a + 1) % g_n_gpus;
                    placed = 1;
                    break;
                } else {
                    if (dt) cudaFree(dt);
                    if (ds) cudaFree(ds);
                    cudaGetLastError();
                }
            }
        }
        if (!placed) {
            slot->h_tiles = h_tiles;
            slot->status = -1;
            pthread_mutex_unlock(&g_mu);
            return 0;
        }
    }

    int dev = slot->dev;
    cudaSetDevice(dev);
    cudaStream_t st = g_st[dev];
    size_t xq_b  = (size_t)T * K;
    size_t xsc_b = (size_t)T * nb * sizeof(float);
    int out_R = (act_mode == 1 || act_mode == 2) ? (int)(R / 2u) : (int)R;
    size_t y_b   = (size_t)T * (size_t)out_R * sizeof(float);

    if (xq_b > g_dxq_cap[dev]) { if (g_dxq[dev]) cudaFree(g_dxq[dev]); cudaMalloc(&g_dxq[dev], xq_b * 2); g_dxq_cap[dev] = xq_b * 2; }
    if (xsc_b > g_dxsc_cap[dev]) { if (g_dxsc[dev]) cudaFree(g_dxsc[dev]); cudaMalloc(&g_dxsc[dev], xsc_b * 2); g_dxsc_cap[dev] = xsc_b * 2; }
    if (y_b > g_dy_cap[dev]) { if (g_dy[dev]) cudaFree(g_dy[dev]); cudaMalloc(&g_dy[dev], y_b * 2); g_dy_cap[dev] = y_b * 2; }

    cudaMemcpyAsync(g_dxq[dev],  h_xq,  xq_b,  cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(g_dxsc[dev], h_xsc, xsc_b, cudaMemcpyHostToDevice, st);
    k_hg_accel_16r<<<dim3(R >> 4, (T + 3) / 4), 256, 0, st>>>(
        slot->d_tiles, slot->d_scales, (int)R, (int)K, (int)bits,
        g_dxq[dev], g_dxsc[dev], g_dy[dev], T, act_mode
    );
    cudaMemcpyAsync(h_Y, g_dy[dev], y_b, cudaMemcpyDeviceToHost, st);
    cudaStreamSynchronize(st);
    pthread_mutex_unlock(&g_mu);
    return 1;
}

extern "C" int hg_cuda_try_moe_pairs(
    const uint8_t *h_exp_base, uint32_t n_exp_total, size_t exp_stride,
    size_t up_tiles_off, size_t dn_blob_off,
    uint32_t R_up, uint32_t K_up, uint32_t R_dn, uint32_t K_dn,
    uint32_t bits, int act_mode, int bf16_dn_round,
    const int8_t *h_xq, const float *h_xsc, int T,
    const int *h_pair_tok, const int *h_pair_slot, const float *h_pair_w, int n_pairs,
    float *h_out_add
) {
    if (T <= 0 || n_pairs <= 0) return 1;
    size_t total_exp_b = (size_t)n_exp_total * exp_stride;

    pthread_mutex_lock(&g_mu);
    ensure_init_locked();
    if (!g_enabled) { pthread_mutex_unlock(&g_mu); return 0; }

    uint32_t idx = ptr_hash(h_exp_base);
    cache_slot_t *slot = NULL;
    for (uint32_t probe = 0; probe < 1024u; probe++) {
        cache_slot_t *s = &g_ht[(idx + probe) & (HT_CAP - 1u)];
        if (s->status == 0) { slot = s; break; }
        if (s->h_tiles == h_exp_base) { slot = s; break; }
    }
    if (!slot || slot->status == -1) { pthread_mutex_unlock(&g_mu); return 0; }

    if (slot->status == 0) {
        int placed = 0;
        for (int a = 0; a < g_n_gpus; a++) {
            int dev = g_gpus[(g_rr_gpu + a) % g_n_gpus];
            cudaSetDevice(dev);
            size_t free_b = 0, tot_b = 0;
            cudaMemGetInfo(&free_b, &tot_b);
            if (free_b > total_exp_b + (size_t)640 * 1024 * 1024) {
                uint8_t *dt = NULL;
                if (cudaMalloc(&dt, total_exp_b) == cudaSuccess) {
                    cudaMemcpy(dt, h_exp_base, total_exp_b, cudaMemcpyHostToDevice);
                    release_host_pages(h_exp_base, total_exp_b);
                    slot->h_tiles = h_exp_base;
                    slot->d_tiles = dt;
                    slot->d_scales = NULL;
                    slot->dev = dev;
                    slot->status = 1;
                    g_rr_gpu = (g_rr_gpu + a + 1) % g_n_gpus;
                    placed = 1;
                    break;
                } else {
                    cudaGetLastError();
                }
            }
        }
        if (!placed) {
            slot->h_tiles = h_exp_base;
            slot->status = -1;
            pthread_mutex_unlock(&g_mu);
            return 0;
        }
    }

    int dev = slot->dev;
    cudaSetDevice(dev);
    cudaStream_t st = g_st[dev];
    size_t nb_up = K_up / 128u, nb_dn = K_dn / 128u;
    size_t xq_b   = (size_t)T * K_up;
    size_t xsc_b  = (size_t)T * nb_up * sizeof(float);
    size_t hid_b  = (size_t)n_pairs * K_dn * sizeof(float);
    size_t zq_b   = (size_t)n_pairs * K_dn;
    size_t zsc_b  = (size_t)n_pairs * nb_dn * sizeof(float);
    size_t out_b  = (size_t)T * R_dn * sizeof(float);
    size_t pc_cap = (size_t)n_pairs;

    if (xq_b  > g_dxq_cap[dev])  { if (g_dxq[dev])  cudaFree(g_dxq[dev]);  cudaMalloc(&g_dxq[dev],  xq_b * 2);  g_dxq_cap[dev]  = xq_b * 2; }
    if (xsc_b > g_dxsc_cap[dev]) { if (g_dxsc[dev]) cudaFree(g_dxsc[dev]); cudaMalloc(&g_dxsc[dev], xsc_b * 2); g_dxsc_cap[dev] = xsc_b * 2; }
    if (hid_b > g_dhid_cap[dev]) { if (g_dhid[dev]) cudaFree(g_dhid[dev]); cudaMalloc(&g_dhid[dev], hid_b * 2); g_dhid_cap[dev] = hid_b * 2; }
    if (zq_b  > g_dzq_cap[dev])  { if (g_dzq[dev])  cudaFree(g_dzq[dev]);  cudaMalloc(&g_dzq[dev],  zq_b * 2);  g_dzq_cap[dev]  = zq_b * 2; }
    if (zsc_b > g_dzsc_cap[dev]) { if (g_dzsc[dev]) cudaFree(g_dzsc[dev]); cudaMalloc(&g_dzsc[dev], zsc_b * 2); g_dzsc_cap[dev] = zsc_b * 2; }
    if (out_b > g_dy_cap[dev])   { if (g_dy[dev])   cudaFree(g_dy[dev]);   cudaMalloc(&g_dy[dev],   out_b * 2); g_dy_cap[dev]   = out_b * 2; }
    if (pc_cap > g_dpair_cap[dev]) {
        if (g_dptok[dev])  cudaFree(g_dptok[dev]);
        if (g_dpslot[dev]) cudaFree(g_dpslot[dev]);
        if (g_dpw[dev])    cudaFree(g_dpw[dev]);
        size_t nc = pc_cap * 2 + 64;
        cudaMalloc(&g_dptok[dev],  nc * sizeof(int));
        cudaMalloc(&g_dpslot[dev], nc * sizeof(int));
        cudaMalloc(&g_dpw[dev],    nc * sizeof(float));
        g_dpair_cap[dev] = nc;
    }

    cudaMemcpyAsync(g_dxq[dev],    h_xq,        xq_b,  cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(g_dxsc[dev],   h_xsc,       xsc_b, cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(g_dptok[dev],  h_pair_tok,  n_pairs * sizeof(int),   cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(g_dpslot[dev], h_pair_slot, n_pairs * sizeof(int),   cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(g_dpw[dev],    h_pair_w,    n_pairs * sizeof(float), cudaMemcpyHostToDevice, st);
    cudaMemcpyAsync(g_dy[dev],     h_out_add,   out_b, cudaMemcpyHostToDevice, st);

    k_moe_pairs_up<<<dim3(R_up >> 4, n_pairs), 128, 0, st>>>(
        slot->d_tiles, exp_stride, up_tiles_off, (int)R_up, (int)K_up, (int)bits, act_mode,
        g_dxq[dev], g_dxsc[dev], g_dptok[dev], g_dpslot[dev], g_dhid[dev], n_pairs
    );
    k_moe_pairs_rot_quant<<<dim3(nb_dn, n_pairs), 128, 0, st>>>(
        slot->d_tiles, exp_stride, dn_blob_off, g_dhid[dev], g_dpslot[dev],
        (int)K_dn, n_pairs, g_dzq[dev], g_dzsc[dev]
    );
    k_moe_pairs_down<<<dim3(R_dn >> 4, n_pairs), 128, 0, st>>>(
        slot->d_tiles, exp_stride, dn_blob_off, (int)R_dn, (int)K_dn, (int)bits, bf16_dn_round,
        g_dzq[dev], g_dzsc[dev], g_dptok[dev], g_dpslot[dev], g_dpw[dev], g_dy[dev], n_pairs
    );

    cudaMemcpyAsync(h_out_add, g_dy[dev], out_b, cudaMemcpyDeviceToHost, st);
    cudaStreamSynchronize(st);
    pthread_mutex_unlock(&g_mu);
    return 1;
}

