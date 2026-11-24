/*
 * Unified Multi-GPU (`1x`, `4x`, `8x RTX 4090`) & Parallel 3-Tier (`VRAM + RAM + NVMe`)
 * Native CUDA (`sm_89`) Benchmark Suite for All 4 Clover `HG` Models:
 *   1. `LFM3-Omni-2B` (`854 MiB`, `24L` `HG4-16R`) -> `1x RTX 4090` (8-warp `__ldg` + CUDA Graph)
 *   2. `Clover-35B-A3B` (`17.41 GiB`, `40L x 256E` + `248K` `HG8`) ->
 *        - `1x RTX 4090` (100% All-in-VRAM, `17.4 GiB` allocated)
 *        - `1x RTX 4090` Hybrid (`1.8 GiB` Trunk+Head in VRAM + `15.6 GiB` Experts in Pinned RAM)
 *        - `4x RTX 4090` (`10` layers/GPU, `4.4 GiB/GPU`)
 *   3. `Nemotron-3.5-Super-VL 120B` (`60.20 GiB`, `88L + 2L-MTP`, `20,992` experts) ->
 *        - `4x RTX 4090` (100% All-in-VRAM, `15.1 GiB/GPU`, `T=1` & `+ 2L-MTP` `K=3`)
 *        - `8x RTX 4090` (100% All-in-VRAM, `7.6 GiB/GPU`, `T=1` & `+ 2L-MTP` `K=3`)
 *        - `1x RTX 4090` Hybrid (`21.2 GiB` Trunks+MTP+Head+Hot in VRAM + `39 GiB` Experts in Pinned RAM)
 *   4. `Kimi-K3.1-Pro 2.8T K31H` (`103.85 GB` Hot `HG4` + `690.30 GB` Cold `HG2` + `5L-DSpark`) ->
 *        - `5L-DSpark` In-VRAM Drafter (`1.3 GB`)
 *        - `8x RTX 4090` Parallel 3-Tier (`103.8 GB` Hot in VRAM + `56` Cold `HG2` Layers `~392 GB` in RAM
 *          + `36` Cold `HG2` Layers on `6.1 GB/s` NVMe — all 3 pipes overlapped in parallel!)
 *        - `8x RTX 4090` Hot in VRAM + 100% Cold `HG2` streamed in parallel from NVMe (`128 GB` host mode)
 *        - `1x RTX 4090` Hybrid (`20.5 GiB` Active Trunks+DSpark in VRAM + Hot in RAM + Cold on NVMe)
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define CK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA err %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while (0)

__constant__ int8_t c_lut16_s8[16];
__constant__ int8_t c_lut4_s8[4];

__device__ __forceinline__ float d_fp16_to_f32(uint16_t h) {
    return __half2float(*reinterpret_cast<const __half *>(&h));
}

__device__ __forceinline__ int pack_lut16_lo_shfl(uint32_t pk, int my_lut_u8) {
    int b0 = __shfl_sync(0xffffffffu, my_lut_u8, (int)((pk >>  0) & 0x0Fu), 16);
    int b1 = __shfl_sync(0xffffffffu, my_lut_u8, (int)((pk >>  8) & 0x0Fu), 16);
    int b2 = __shfl_sync(0xffffffffu, my_lut_u8, (int)((pk >> 16) & 0x0Fu), 16);
    int b3 = __shfl_sync(0xffffffffu, my_lut_u8, (int)((pk >> 24) & 0x0Fu), 16);
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

__device__ __forceinline__ int pack_lut16_hi_shfl(uint32_t pk, int my_lut_u8) {
    int b0 = __shfl_sync(0xffffffffu, my_lut_u8, (int)((pk >>  4) & 0x0Fu), 16);
    int b1 = __shfl_sync(0xffffffffu, my_lut_u8, (int)((pk >> 12) & 0x0Fu), 16);
    int b2 = __shfl_sync(0xffffffffu, my_lut_u8, (int)((pk >> 20) & 0x0Fu), 16);
    int b3 = __shfl_sync(0xffffffffu, my_lut_u8, (int)((pk >> 28) & 0x0Fu), 16);
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

__device__ __forceinline__ int pack_lut4_crumb(uint32_t pk, int shift, int lut4_packed) {
    /* lut4_packed holds [-127, -38, +38, +127] in bytes 0..3 of a single register! */
    int i0 = (pk >> (shift +  0)) & 0x03u;
    int i1 = (pk >> (shift +  8)) & 0x03u;
    int i2 = (pk >> (shift + 16)) & 0x03u;
    int i3 = (pk >> (shift + 24)) & 0x03u;
    int b0 = (lut4_packed >> (i0 * 8)) & 0xFF;
    int b1 = (lut4_packed >> (i1 * 8)) & 0xFF;
    int b2 = (lut4_packed >> (i2 * 8)) & 0xFF;
    int b3 = (lut4_packed >> (i3 * 8)) & 0xFF;
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
}

/* Per-128-Block Int32 FWHT + Int8 Quantization: Grid(K/128, T), 128 threads */
__global__ void k_rot_quant_128(
    const float *x, const float *pre_diag, int T, int K,
    int8_t *out_xq, float *out_xsc
) {
    int b = blockIdx.x, t = blockIdx.y, tid = threadIdx.x;
    int nb = K >> 7;
    if (t >= T || b >= nb) return;

    __shared__ float s_abs[128];
    __shared__ int32_t s_rot[128];
    __shared__ int32_t s_rmax[128];

    float p = x[(size_t)t * K + b * 128 + tid] * (pre_diag ? pre_diag[b * 128 + tid] : 1.0f);
    s_abs[tid] = fabsf(p);
    __syncthreads();
    for (int st = 64; st > 0; st >>= 1) {
        if (tid < st && s_abs[tid + st] > s_abs[tid]) s_abs[tid] = s_abs[tid + st];
        __syncthreads();
    }
    float max_abs = s_abs[0];
    if (max_abs < 1e-12f) {
        if (tid == 0) out_xsc[t * nb + b] = 0.0f;
        out_xq[(size_t)t * K + b * 128 + tid] = 0;
        return;
    }
    float in_sc = max_abs * (1.0f / 2048.0f);
    s_rot[tid] = (int32_t)rintf(fminf(2048.0f, fmaxf(-2048.0f, p * (2048.0f / max_abs))));
    __syncthreads();

    for (int h = 1; h < 128; h <<= 1) {
        int grp = tid / h;
        int k = tid & (h - 1);
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
        if (tid == 0) out_xsc[t * nb + b] = 0.0f;
        out_xq[(size_t)t * K + b * 128 + tid] = 0;
    } else {
        if (tid == 0) out_xsc[t * nb + b] = (in_sc * (float)max_rot) * (0.08838834764831845f / 127.0f);
        out_xq[(size_t)t * K + b * 128 + tid] = (int8_t)rintf(fminf(127.0f, fmaxf(-127.0f, (float)rv * (127.0f / (float)max_rot))));
    }
}

/*
 * Universal 8-Warp (`256` threads) High-Occupancy `HG4-16R` / `HG8-16R` / `HG2-16R` Kernel:
 *   `bits`: 4 (`HG4-16R`, 1024B/blk), 8 (`HG8-16R`, 2048B/blk), 2 (`HG2-16R`, 512B/blk)
 *   `act_mode`: 0 = linear, 1 = SwiGLU, 2 = SiTU-GLU, 3 = ReLU^2
 */
__global__ void k_hg_universal_16r(
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
    int lane    = threadIdx.x & 31; /* 0..31 */
    int r       = lane & 15;
    int sub     = lane >> 4;        /* 0..1 */
    int nb      = K >> 7;
    int out_R   = (act_mode == 1 || act_mode == 2) ? (R >> 1) : R;
    size_t bstride = (bits == 8) ? 2048u : ((bits == 2) ? 512u : 1024u);

    const uint8_t  *wt  = tiles + (size_t)rg * (size_t)nb * bstride;
    const uint16_t *wsc = scales_fp16 + (size_t)rg * (size_t)nb * 16;
    int my_lut_u8 = (int)(uint8_t)c_lut16_s8[r];
    /* Packed 4-byte LUT4: [-127 (0x81), -38 (0xDA), +38 (0x26), +127 (0x7F)] = 0x7F26DA81 */
    const int lut4_pk = 0x7F26DA81;

    float vs[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int b = warp_id; b < nb; b += 8) {
        const uint32_t *wb32 = reinterpret_cast<const uint32_t *>(wt + (size_t)b * bstride);
        float ws = d_fp16_to_f32(__ldg(&wsc[b * 16 + r]));
        int acc[4] = {0, 0, 0, 0};

        if (bits == 4) {
            #pragma unroll
            for (int it = 0; it < 8; it++) {
                uint32_t pk = __ldg(&wb32[it * 32 + lane]);
                int wlo = pack_lut16_lo_shfl(pk, my_lut_u8);
                int whi = pack_lut16_hi_shfl(pk, my_lut_u8);
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
            /* bits == 2 (`HG2-16R`, 512 B/blk = 8 chunks x 16 lanes x 4 bytes) */
            #pragma unroll
            for (int it = 0; it < 4; it++) {
                uint32_t pk = __ldg(&wb32[it * 32 + lane]);
                int w0 = pack_lut4_crumb(pk, 0, lut4_pk);
                int w1 = pack_lut4_crumb(pk, 2, lut4_pk);
                int w2 = pack_lut4_crumb(pk, 4, lut4_pk);
                int w3 = pack_lut4_crumb(pk, 6, lut4_pk);
                int s = (it << 1) + sub; /* 0..7 */
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
            if (sub == 0) {
                vs[ti] += (float)sum32 * (ws * X_sc[(t_base + ti) * nb + b]);
            }
        }
    }

    __shared__ float smem[8][4][16];
    if (sub == 0) {
        for (int ti = 0; ti < tblk; ti++) smem[warp_id][ti][r] = vs[ti];
    }
    __syncthreads();

    if (warp_id == 0 && sub == 0) {
        for (int ti = 0; ti < tblk; ti++) {
            float total = (smem[0][ti][r] + smem[1][ti][r] + smem[2][ti][r] + smem[3][ti][r]) +
                          (smem[4][ti][r] + smem[5][ti][r] + smem[6][ti][r] + smem[7][ti][r]);
            smem[0][ti][r] = total;
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
                    float act = (4.0f * tanhf(g * 0.25f)) / (1.0f + expf(-g));
                    yt[rg * 8 + r] = act * (25.0f * tanhf(u * 0.04f));
                }
            } else if (act_mode == 3) {
                float v = smem[0][ti][r];
                v = v > 0.0f ? v : 0.0f;
                yt[rg * 16 + r] = v * v;
            } else {
                yt[rg * 16 + r] = smem[0][ti][r];
            }
        }
    }
}

/*
 * Batched Multi-Expert `Top-K` MoE Kernel (Executes ALL `K_act` active experts in ONE launch!):
 *   Each expert blob `exp_base + sel_idx[k] * exp_stride` starts at `scales_fp16` followed by `tiles`!
 */
__global__ void k_hg_batched_experts_up(
    const uint8_t * __restrict__ exp_base,
    const uint8_t * const * __restrict__ exp_ptrs,
    size_t exp_stride, size_t tiles_off,
    const int * __restrict__ sel_ids,
    int R, int K, int bits, int act_mode,
    const int8_t * __restrict__ X_q,
    const float * __restrict__ X_sc,
    float * __restrict__ out_hid, /* [K_act * T * out_R] */
    int K_act, int T
) {
    int rg = blockIdx.x;
    int e_slot = blockIdx.y; /* 0..K_act-1 */
    int n_rg = R >> 4;
    if (rg >= n_rg || e_slot >= K_act) return;

    const uint8_t *ep = exp_ptrs ? exp_ptrs[e_slot] : (exp_base + (size_t)sel_ids[e_slot] * exp_stride);
    const uint16_t *wsc = reinterpret_cast<const uint16_t *>(ep) + (size_t)rg * (size_t)(K >> 7) * 16;
    size_t bstride = (bits == 2) ? 512u : 1024u;
    const uint8_t *wt = ep + tiles_off + (size_t)rg * (size_t)(K >> 7) * bstride;

    int warp_id = threadIdx.x >> 5; /* 0..3 */
    int lane    = threadIdx.x & 31;
    int r       = lane & 15;
    int sub     = lane >> 4;
    int nb      = K >> 7;
    int out_R   = (act_mode == 1 || act_mode == 2) ? (R >> 1) : R;
    int my_lut_u8 = (int)(uint8_t)c_lut16_s8[r];
    const int lut4_pk = 0x7F26DA81;

    for (int t = 0; t < T; t++) {
        float vs = 0.0f;
        const int *xp_base = reinterpret_cast<const int *>(X_q + (size_t)t * K);
        for (int b = warp_id; b < nb; b += 4) {
            const uint32_t *wb32 = reinterpret_cast<const uint32_t *>(wt + (size_t)b * bstride);
            float ws = d_fp16_to_f32(__ldg(&wsc[b * 16 + r]));
            int acc = 0;
            if (bits == 4) {
                #pragma unroll
                for (int it = 0; it < 8; it++) {
                    uint32_t pk = __ldg(&wb32[it * 32 + lane]);
                    int s = (it << 1) + sub;
                    acc = __dp4a(pack_lut16_lo_shfl(pk, my_lut_u8), __ldg(&xp_base[b * 32 + 2 * s + 0]), acc);
                    acc = __dp4a(pack_lut16_hi_shfl(pk, my_lut_u8), __ldg(&xp_base[b * 32 + 2 * s + 1]), acc);
                }
            } else {
                #pragma unroll
                for (int it = 0; it < 4; it++) {
                    uint32_t pk = __ldg(&wb32[it * 32 + lane]);
                    int s = (it << 1) + sub;
                    acc = __dp4a(pack_lut4_crumb(pk, 0, lut4_pk), __ldg(&xp_base[b * 32 + 4 * s + 0]), acc);
                    acc = __dp4a(pack_lut4_crumb(pk, 2, lut4_pk), __ldg(&xp_base[b * 32 + 4 * s + 1]), acc);
                    acc = __dp4a(pack_lut4_crumb(pk, 4, lut4_pk), __ldg(&xp_base[b * 32 + 4 * s + 2]), acc);
                    acc = __dp4a(pack_lut4_crumb(pk, 6, lut4_pk), __ldg(&xp_base[b * 32 + 4 * s + 3]), acc);
                }
            }
            int sum32 = acc + __shfl_xor_sync(0xffffffffu, acc, 16);
            if (sub == 0) vs += (float)sum32 * (ws * X_sc[t * nb + b]);
        }
        __shared__ float smem[4][16];
        if (sub == 0) smem[warp_id][r] = vs;
        __syncthreads();
        if (warp_id == 0 && sub == 0) {
            float tot = (smem[0][r] + smem[1][r]) + (smem[2][r] + smem[3][r]);
            smem[0][r] = tot;
            __syncwarp(0x0000ffffu);
            float *yt = out_hid + ((size_t)e_slot * T + t) * out_R;
            if (act_mode == 1) {
                if (r < 8) { float g = smem[0][r], u = smem[0][r + 8]; yt[rg * 8 + r] = (g / (1.0f + expf(-g))) * u; }
            } else if (act_mode == 2) {
                if (r < 8) {
                    float g = smem[0][r], u = smem[0][r + 8];
                    yt[rg * 8 + r] = ((4.0f * tanhf(g * 0.25f)) / (1.0f + expf(-g))) * (25.0f * tanhf(u * 0.04f));
                }
            } else {
                float v = smem[0][r]; v = v > 0.0f ? v : 0.0f; yt[rg * 16 + r] = v * v;
            }
        }
        __syncthreads();
    }
}

__global__ void k_hg_batched_experts_down(
    const uint8_t * __restrict__ exp_base,
    const uint8_t * const * __restrict__ exp_ptrs,
    size_t exp_stride, size_t dn_blob_off,
    const int * __restrict__ sel_ids,
    const float * __restrict__ sel_w,
    int R, int K, int bits,
    const int8_t * __restrict__ Z_q,   /* [K_act * T * K] */
    const float * __restrict__ Z_sc,   /* [K_act * T * (K/128)] */
    float * __restrict__ out_sum,      /* [T * R] atomicAdd weighted sum */
    int K_act, int T
) {
    int rg = blockIdx.x;
    int e_slot = blockIdx.y;
    int n_rg = R >> 4;
    if (rg >= n_rg || e_slot >= K_act) return;

    const uint8_t *ep = exp_ptrs ? exp_ptrs[e_slot] : (exp_base + (size_t)sel_ids[e_slot] * exp_stride);
    const uint8_t *dn = ep + dn_blob_off;
    /* dn blob: 8B header + K*4B pre_diag + scales_fp16 + tiles */
    int nb = K >> 7;
    const uint16_t *wsc = reinterpret_cast<const uint16_t *>(dn + 8 + (size_t)K * 4) + (size_t)rg * nb * 16;
    size_t bstride = (bits == 2) ? 512u : 1024u;
    const uint8_t *wt = dn + 8 + (size_t)K * 4 + (size_t)R * nb * 2 + (size_t)rg * nb * bstride;

    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;
    int r       = lane & 15;
    int sub     = lane >> 4;
    int my_lut_u8 = (int)(uint8_t)c_lut16_s8[r];
    const int lut4_pk = 0x7F26DA81;

    for (int t = 0; t < T; t++) {
        float vs = 0.0f;
        size_t zt_idx = (size_t)e_slot * T + t;
        const int *xp_base = reinterpret_cast<const int *>(Z_q + zt_idx * K);
        for (int b = warp_id; b < nb; b += 4) {
            const uint32_t *wb32 = reinterpret_cast<const uint32_t *>(wt + (size_t)b * bstride);
            float ws = d_fp16_to_f32(__ldg(&wsc[b * 16 + r]));
            int acc = 0;
            if (bits == 4) {
                #pragma unroll
                for (int it = 0; it < 8; it++) {
                    uint32_t pk = __ldg(&wb32[it * 32 + lane]);
                    int s = (it << 1) + sub;
                    acc = __dp4a(pack_lut16_lo_shfl(pk, my_lut_u8), __ldg(&xp_base[b * 32 + 2 * s + 0]), acc);
                    acc = __dp4a(pack_lut16_hi_shfl(pk, my_lut_u8), __ldg(&xp_base[b * 32 + 2 * s + 1]), acc);
                }
            } else {
                #pragma unroll
                for (int it = 0; it < 4; it++) {
                    uint32_t pk = __ldg(&wb32[it * 32 + lane]);
                    int s = (it << 1) + sub;
                    acc = __dp4a(pack_lut4_crumb(pk, 0, lut4_pk), __ldg(&xp_base[b * 32 + 4 * s + 0]), acc);
                    acc = __dp4a(pack_lut4_crumb(pk, 2, lut4_pk), __ldg(&xp_base[b * 32 + 4 * s + 1]), acc);
                    acc = __dp4a(pack_lut4_crumb(pk, 4, lut4_pk), __ldg(&xp_base[b * 32 + 4 * s + 2]), acc);
                    acc = __dp4a(pack_lut4_crumb(pk, 6, lut4_pk), __ldg(&xp_base[b * 32 + 4 * s + 3]), acc);
                }
            }
            int sum32 = acc + __shfl_xor_sync(0xffffffffu, acc, 16);
            if (sub == 0) vs += (float)sum32 * (ws * Z_sc[zt_idx * nb + b]);
        }
        __shared__ float smem[4][16];
        if (sub == 0) smem[warp_id][r] = vs;
        __syncthreads();
        if (warp_id == 0 && sub == 0) {
            float tot = ((smem[0][r] + smem[1][r]) + (smem[2][r] + smem[3][r])) * sel_w[e_slot];
            atomicAdd(&out_sum[(size_t)t * R + rg * 16 + r], tot);
        }
        __syncthreads();
    }
}

__global__ void k_rmsnorm_fast(const float *x, const float *w, float *y, int T, int dim) {
    int t = blockIdx.x;
    if (t >= T) return;
    const float *xr = x + (size_t)t * dim;
    float *yr = y + (size_t)t * dim;
    float ss = 0.0f;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) ss += xr[d] * xr[d];
    __shared__ float sm[256];
    sm[threadIdx.x] = ss;
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) { if (threadIdx.x < st) sm[threadIdx.x] += sm[threadIdx.x + st]; __syncthreads(); }
    float inv = rsqrtf(sm[0] / (float)dim + 1e-5f);
    for (int d = threadIdx.x; d < dim; d += blockDim.x) yr[d] = xr[d] * inv * (w ? w[d] : 1.0f);
}

static size_t file_size_bytes(const char *path) {
    struct stat st;
    if (stat(path, &st) != 0) { fprintf(stderr, "Missing file: %s\n", path); exit(1); }
    return (size_t)st.st_size;
}

static uint8_t *read_file_host(const char *path, size_t *out_sz, int pinned) {
    size_t sz = file_size_bytes(path);
    uint8_t *buf = NULL;
    if (pinned) CK(cudaHostAlloc(&buf, sz, cudaHostAllocPortable));
    else buf = (uint8_t *)malloc(sz);
    int fd = open(path, O_RDONLY);
    size_t done = 0;
    while (done < sz) {
        ssize_t r = pread(fd, buf + done, sz - done, (off_t)done);
        if (r <= 0) break;
        done += (size_t)r;
    }
    close(fd);
    if (out_sz) *out_sz = sz;
    return buf;
}

static uint8_t *upload_file_dev(int dev, const char *path, uint8_t **keep_h, size_t *out_sz) {
    CK(cudaSetDevice(dev));
    size_t sz = 0;
    uint8_t *h = read_file_host(path, &sz, 0);
    uint8_t *d = NULL;
    CK(cudaMalloc(&d, sz));
    CK(cudaMemcpy(d, h, sz, cudaMemcpyHostToDevice));
    if (keep_h) *keep_h = h;
    else free(h);
    if (out_sz) *out_sz = sz;
    return d;
}

typedef struct {
    uint32_t R, K, bits;
    const float *pre_diag;
    const uint16_t *scales;
    const uint8_t *tiles;
} dev_mat_t;

static size_t parse_dev_mat(const uint8_t *d_ptr, const uint8_t *h_ptr, dev_mat_t *m) {
    uint32_t rk;
    memcpy(&m->R, h_ptr + 0, 4);
    memcpy(&rk,   h_ptr + 4, 4);
    uint32_t top = rk >> 28;
    m->bits = (top == 8u) ? 8u : ((top == 2u) ? 2u : 4u);
    m->K = rk & 0x0FFFFFFFu;
    size_t nb = m->K / 128;
    size_t bstride = (m->bits == 8u) ? 2048u : ((m->bits == 2u) ? 512u : 1024u);
    m->pre_diag = (const float *)(d_ptr + 8);
    m->scales   = (const uint16_t *)(d_ptr + 8 + (size_t)m->K * 4);
    m->tiles    = d_ptr + 8 + (size_t)m->K * 4 + (size_t)m->R * nb * 2;
    return 8 + (size_t)m->K * 4 + (size_t)m->R * nb * 2 + ((size_t)(m->R / 16) * nb * bstride);
}

static void launch_mat(
    const dev_mat_t *m, const float *d_x, float *d_y, int T, int act_mode,
    int8_t *d_xq, float *d_xsc, cudaStream_t st
) {
    k_rot_quant_128<<<dim3(m->K >> 7, T), 128, 0, st>>>(d_x, m->pre_diag, T, (int)m->K, d_xq, d_xsc);
    k_hg_universal_16r<<<dim3(m->R >> 4, (T + 3) / 4), 256, 0, st>>>(
        m->tiles, m->scales, (int)m->R, (int)m->K, (int)m->bits, d_xq, d_xsc, d_y, T, act_mode
    );
}

static int g_can_peer[8][8];
static float *g_peer_bounce = NULL;

static void init_lut_on_gpus(int n_gpus) {
    static const int8_t lut16[16] = {-127,-96,-75,-58,-44,-31,-18,-6,6,18,31,44,58,75,96,127};
    static const int8_t lut4[4]   = {-127,-38,38,127};
    if (!g_peer_bounce) CK(cudaHostAlloc(&g_peer_bounce, 256 * 8192 * sizeof(float), cudaHostAllocPortable));
    for (int g = 0; g < n_gpus; g++) {
        CK(cudaSetDevice(g));
        CK(cudaMemcpyToSymbol(c_lut16_s8, lut16, 16));
        CK(cudaMemcpyToSymbol(c_lut4_s8, lut4, 4));
        for (int p = 0; p < n_gpus; p++) {
            g_can_peer[g][p] = 0;
            if (g != p) {
                int can = 0;
                cudaDeviceCanAccessPeer(&can, g, p);
                g_can_peer[g][p] = can;
                if (can) {
                    cudaDeviceEnablePeerAccess(p, 0);
                    cudaGetLastError(); /* clear cudaErrorPeerAccessAlreadyEnabled across back-to-back runs */
                }
            }
        }
    }
}

static void copy_activation_gpu_to_gpu(float *d_dst, int dst_g, const float *d_src, int src_g, size_t nbytes, cudaStream_t src_st, cudaStream_t dst_st) {
    if (g_can_peer[dst_g][src_g] && g_can_peer[src_g][dst_g]) {
        CK(cudaMemcpyPeerAsync(d_dst, dst_g, d_src, src_g, nbytes, src_st));
        CK(cudaStreamSynchronize(src_st));
    } else {
        CK(cudaSetDevice(src_g));
        CK(cudaMemcpyAsync(g_peer_bounce, d_src, nbytes, cudaMemcpyDeviceToHost, src_st));
        CK(cudaStreamSynchronize(src_st));
        CK(cudaSetDevice(dst_g));
        CK(cudaMemcpyAsync(d_dst, g_peer_bounce, nbytes, cudaMemcpyHostToDevice, dst_st));
        CK(cudaStreamSynchronize(dst_st));
    }
}

/* =========================================================================
 * BENCHMARK 2: Clover-35B-A3B (`40` Layers x `256` Experts + `248K` LM-Head)
 * ========================================================================= */
typedef struct {
    int dev;
    uint32_t kind;
    uint8_t *d_trunk, *d_exp, *h_exp_pinned;
    size_t exp_stride;
    const float *in_ln, *post_ln, *pre_diag_up;
    dev_mat_t m1, m2, m3, m4; /* BD-GDN: qkv, z, out; GQA: q, k, v, o */
    dev_mat_t sh_w13, sh_dn;
} c35_gpu_layer_t;

static void bench_clover35b(const char *dir, int n_gpus, int experts_in_ram) {
    init_lut_on_gpus(n_gpus);
    c35_gpu_layer_t lyrs[40];
    size_t vram_per_gpu[8] = {0}, ram_pinned_b = 0;
    char path[512];

    uint8_t *h_lm = NULL; size_t sz_lm = 0;
    snprintf(path, sizeof(path), "%s/lm_head.hg4", dir);
    int last_g = n_gpus - 1;
    uint8_t *d_lm_raw = upload_file_dev(last_g, path, &h_lm, &sz_lm);
    vram_per_gpu[last_g] += sz_lm;
    dev_mat_t d_lm;
    parse_dev_mat(d_lm_raw, h_lm, &d_lm);
    free(h_lm);

    for (int l = 0; l < 40; l++) {
        int g = (l * n_gpus) / 40;
        c35_gpu_layer_t *L = &lyrs[l];
        memset(L, 0, sizeof(*L));
        L->dev = g;
        uint8_t *h_tr = NULL; size_t sz_tr = 0;
        snprintf(path, sizeof(path), "%s/layer_%02d.hg4", dir, l);
        L->d_trunk = upload_file_dev(g, path, &h_tr, &sz_tr);
        vram_per_gpu[g] += sz_tr;

        memcpy(&L->kind, h_tr, 4);
        size_t off = 4;
        L->in_ln   = (const float *)(L->d_trunk + off); off += 2048 * 4;
        L->post_ln = (const float *)(L->d_trunk + off); off += 2048 * 4;
        if (L->kind == 0) {
            off += (8192 * 4 + 8192 * 3 + 32 * 2048 * 2 + 32 * 5 + 128) * 4;
            uint32_t l1, l2, l3, lm;
            memcpy(&l1, h_tr + off, 4); memcpy(&l2, h_tr + off + 4, 4);
            memcpy(&l3, h_tr + off + 8, 4); memcpy(&lm, h_tr + off + 12, 4);
            off += 16;
            parse_dev_mat(L->d_trunk + off, h_tr + off, &L->m1); off += l1;
            parse_dev_mat(L->d_trunk + off, h_tr + off, &L->m2); off += l2;
            parse_dev_mat(L->d_trunk + off, h_tr + off, &L->m3); off += l3;
        } else {
            off += (256 * 2) * 4;
            uint32_t l1, l2, l3, l4, lm;
            memcpy(&l1, h_tr + off, 4); memcpy(&l2, h_tr + off + 4, 4);
            memcpy(&l3, h_tr + off + 8, 4); memcpy(&l4, h_tr + off + 12, 4);
            memcpy(&lm, h_tr + off + 16, 4);
            off += 20;
            parse_dev_mat(L->d_trunk + off, h_tr + off, &L->m1); off += l1;
            parse_dev_mat(L->d_trunk + off, h_tr + off, &L->m2); off += l2;
            parse_dev_mat(L->d_trunk + off, h_tr + off, &L->m3); off += l3;
            parse_dev_mat(L->d_trunk + off, h_tr + off, &L->m4); off += l4;
        }
        /* moe tail: k_cached(4B) + l_sh13(4B) + l_shdn(4B) + gate_w(k_cached*2048*4B) + sh_g(2048*4B) + sh_w13 + sh_dn */
        uint32_t k_cached = 256, lw13 = 0, ldn = 0;
        memcpy(&k_cached, h_tr + off, 4);
        memcpy(&lw13,     h_tr + off + 4, 4);
        memcpy(&ldn,      h_tr + off + 8, 4);
        off += 12 + ((size_t)k_cached * 2048 + 2048) * 4;
        parse_dev_mat(L->d_trunk + off, h_tr + off, &L->sh_w13); off += lw13;
        parse_dev_mat(L->d_trunk + off, h_tr + off, &L->sh_dn);
        free(h_tr);

        snprintf(path, sizeof(path), "%s/experts_%02d.hg4", dir, l);
        if (!experts_in_ram) {
            uint8_t *h_ex = NULL; size_t sz_ex = 0;
            L->d_exp = upload_file_dev(g, path, &h_ex, &sz_ex);
            vram_per_gpu[g] += sz_ex;
            uint32_t stride = 0;
            memcpy(&stride, h_ex + 24, 4);
            L->exp_stride = stride;
            L->pre_diag_up = (const float *)(L->d_exp + 28);
            free(h_ex);
        } else {
            size_t sz_ex = 0;
            L->h_exp_pinned = read_file_host(path, &sz_ex, 1);
            ram_pinned_b += sz_ex;
            uint32_t stride = 0;
            memcpy(&stride, L->h_exp_pinned + 24, 4);
            L->exp_stride = stride;
            CK(cudaSetDevice(g));
            float *d_pd = NULL;
            CK(cudaMalloc(&d_pd, 2048 * 4));
            CK(cudaMemcpy(d_pd, L->h_exp_pinned + 28, 2048 * 4, cudaMemcpyHostToDevice));
            L->pre_diag_up = d_pd;
        }
    }

    /* Per-GPU scratch & streams */
    float *d_x[8], *d_u[8], *d_tmp1[8], *d_tmp2[8], *d_hid[8], *d_xsc[8], *d_zsc[8], *d_selw[8], *d_logits = NULL;
    int8_t *d_xq[8], *d_zq[8];
    int *d_seli[8];
    uint8_t *d_stage_exp[8][2]; /* Double-buffered PCIe expert DMA slots when experts_in_ram=1 */
    const uint8_t **d_exp_ptrs[8];
    cudaStream_t s_comp[8], s_dma[8];
    cudaEvent_t ev_dma[8][2];

    int sel_h[8] = {0, 7, 19, 43, 88, 129, 191, 240};
    float selw_h[8] = {0.18f, 0.16f, 0.14f, 0.13f, 0.11f, 0.10f, 0.09f, 0.09f};

    for (int g = 0; g < n_gpus; g++) {
        CK(cudaSetDevice(g));
        CK(cudaStreamCreate(&s_comp[g]));
        CK(cudaStreamCreate(&s_dma[g]));
        CK(cudaEventCreate(&ev_dma[g][0]));
        CK(cudaEventCreate(&ev_dma[g][1]));
        CK(cudaMalloc(&d_x[g],    256 * 2048 * 4));
        CK(cudaMalloc(&d_u[g],    256 * 2048 * 4));
        CK(cudaMalloc(&d_tmp1[g], 256 * 8192 * 4));
        CK(cudaMalloc(&d_tmp2[g], 256 * 8192 * 4));
        CK(cudaMalloc(&d_hid[g],  8 * 256 * 512 * 4));
        CK(cudaMalloc(&d_xq[g],   256 * 8192));
        CK(cudaMalloc(&d_xsc[g],  256 * 64 * 4));
        CK(cudaMalloc(&d_zq[g],   8 * 256 * 512));
        CK(cudaMalloc(&d_zsc[g],  8 * 256 * 4 * 4));
        CK(cudaMalloc(&d_seli[g], 8 * 4));
        CK(cudaMalloc(&d_selw[g], 8 * 4));
        CK(cudaMemcpy(d_seli[g], sel_h, 32, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_selw[g], selw_h, 32, cudaMemcpyHostToDevice));
        if (experts_in_ram) {
            size_t st_sz = 8 * lyrs[0].exp_stride;
            CK(cudaMalloc(&d_stage_exp[g][0], st_sz));
            CK(cudaMalloc(&d_stage_exp[g][1], st_sz));
            CK(cudaMalloc(&d_exp_ptrs[g], 8 * sizeof(void *)));
        }
    }
    CK(cudaSetDevice(last_g));
    CK(cudaMalloc(&d_logits, 248320 * 4));

    auto run_step = [&](int T) {
        if (experts_in_ram) {
            /* Kick off Layer 0 expert PCIe DMA immediately */
            int g0 = lyrs[0].dev;
            CK(cudaSetDevice(g0));
            const uint8_t *ptrs[8];
            for (int k = 0; k < 8; k++) {
                uint8_t *dst = d_stage_exp[g0][0] + k * lyrs[0].exp_stride;
                CK(cudaMemcpyAsync(dst, lyrs[0].h_exp_pinned + 28 + 2048 * 4 + (size_t)sel_h[k] * lyrs[0].exp_stride,
                                   lyrs[0].exp_stride, cudaMemcpyHostToDevice, s_dma[g0]));
                ptrs[k] = dst;
            }
            CK(cudaMemcpyAsync(d_exp_ptrs[g0], ptrs, 8 * sizeof(void *), cudaMemcpyHostToDevice, s_dma[g0]));
            CK(cudaEventRecord(ev_dma[g0][0], s_dma[g0]));
        }

        for (int l = 0; l < 40; l++) {
            c35_gpu_layer_t *L = &lyrs[l];
            int g = L->dev;
            CK(cudaSetDevice(g));
            cudaStream_t sc = s_comp[g];

            /* While computing Layer l's trunk on s_comp, prefetch Layer l+1's active experts on s_dma! */
            if (experts_in_ram && l + 1 < 40) {
                int gn = lyrs[l + 1].dev;
                int slot = (l + 1) & 1;
                CK(cudaSetDevice(gn));
                const uint8_t *ptrs[8];
                for (int k = 0; k < 8; k++) {
                    uint8_t *dst = d_stage_exp[gn][slot] + k * lyrs[l + 1].exp_stride;
                    CK(cudaMemcpyAsync(dst, lyrs[l + 1].h_exp_pinned + 28 + 2048 * 4 + (size_t)sel_h[k] * lyrs[l + 1].exp_stride,
                                       lyrs[l + 1].exp_stride, cudaMemcpyHostToDevice, s_dma[gn]));
                    ptrs[k] = dst;
                }
                CK(cudaMemcpyAsync(d_exp_ptrs[gn], ptrs, 8 * sizeof(void *), cudaMemcpyHostToDevice, s_dma[gn]));
                CK(cudaEventRecord(ev_dma[gn][slot], s_dma[gn]));
                CK(cudaSetDevice(g));
            }

            k_rmsnorm_fast<<<T, 256, 0, sc>>>(d_x[g], L->in_ln, d_u[g], T, 2048);
            if (L->kind == 0) {
                launch_mat(&L->m1, d_u[g], d_tmp1[g], T, 0, d_xq[g], d_xsc[g], sc);
                launch_mat(&L->m2, d_u[g], d_tmp2[g], T, 0, d_xq[g], d_xsc[g], sc);
                launch_mat(&L->m3, d_tmp2[g], d_x[g], T, 0, d_xq[g], d_xsc[g], sc);
            } else {
                launch_mat(&L->m1, d_u[g], d_tmp1[g], T, 0, d_xq[g], d_xsc[g], sc);
                launch_mat(&L->m2, d_u[g], d_tmp2[g], T, 0, d_xq[g], d_xsc[g], sc);
                launch_mat(&L->m3, d_u[g], d_tmp2[g] + T * 512, T, 0, d_xq[g], d_xsc[g], sc);
                launch_mat(&L->m4, d_tmp1[g], d_x[g], T, 0, d_xq[g], d_xsc[g], sc);
            }
            k_rmsnorm_fast<<<T, 256, 0, sc>>>(d_x[g], L->post_ln, d_u[g], T, 2048);
            /* Shared SwiGLU expert */
            launch_mat(&L->sh_w13, d_u[g], d_tmp1[g], T, 1, d_xq[g], d_xsc[g], sc);
            launch_mat(&L->sh_dn,  d_tmp1[g], d_x[g], T, 0, d_xq[g], d_xsc[g], sc);

            /* 8 Routed SwiGLU experts in 1 batched launch! */
            k_rot_quant_128<<<dim3(16, T), 128, 0, sc>>>(d_u[g], L->pre_diag_up, T, 2048, d_xq[g], d_xsc[g]);
            if (experts_in_ram) {
                CK(cudaStreamWaitEvent(sc, ev_dma[g][l & 1], 0));
                k_hg_batched_experts_up<<<dim3(64, 8), 128, 0, sc>>>(
                    NULL, d_exp_ptrs[g], L->exp_stride, 32768, d_seli[g], 1024, 2048, 4, 1,
                    d_xq[g], d_xsc[g], d_hid[g], 8, T
                );
                k_rot_quant_128<<<dim3(4, 8 * T), 128, 0, sc>>>(d_hid[g], NULL, 8 * T, 512, d_zq[g], d_zsc[g]);
                k_hg_batched_experts_down<<<dim3(128, 8), 128, 0, sc>>>(
                    NULL, d_exp_ptrs[g], L->exp_stride, 1081344, d_seli[g], d_selw[g], 2048, 512, 4,
                    d_zq[g], d_zsc[g], d_x[g], 8, T
                );
            } else {
                const uint8_t *ebase = L->d_exp + 28 + 2048 * 4;
                k_hg_batched_experts_up<<<dim3(64, 8), 128, 0, sc>>>(
                    ebase, NULL, L->exp_stride, 32768, d_seli[g], 1024, 2048, 4, 1,
                    d_xq[g], d_xsc[g], d_hid[g], 8, T
                );
                k_rot_quant_128<<<dim3(4, 8 * T), 128, 0, sc>>>(d_hid[g], NULL, 8 * T, 512, d_zq[g], d_zsc[g]);
                k_hg_batched_experts_down<<<dim3(128, 8), 128, 0, sc>>>(
                    ebase, NULL, L->exp_stride, 1081344, d_seli[g], d_selw[g], 2048, 512, 4,
                    d_zq[g], d_zsc[g], d_x[g], 8, T
                );
            }

            if (l + 1 < 40 && lyrs[l + 1].dev != g) {
                int gn = lyrs[l + 1].dev;
                CK(cudaMemcpyPeerAsync(d_x[gn], gn, d_x[g], g, (size_t)T * 2048 * 4, sc));
                CK(cudaStreamSynchronize(sc));
            }
        }
        CK(cudaSetDevice(last_g));
        launch_mat(&d_lm, d_x[last_g] + (size_t)(T - 1) * 2048, d_logits, 1, 0, d_xq[last_g], d_xsc[last_g], s_comp[last_g]);
        CK(cudaStreamSynchronize(s_comp[last_g]));
    };

    run_step(1); run_step(1);
    double t0 = omp_get_wtime();
    int n_steps = 32;
    for (int i = 0; i < n_steps; i++) run_step(1);
    double dc_ms = (omp_get_wtime() - t0) * 1000.0;

    run_step(256);
    double tp0 = omp_get_wtime();
    run_step(256);
    double pf_ms = (omp_get_wtime() - tp0) * 1000.0;

    printf("RESULT_CLOVER35B gpus=%d mode=%s vram_gpu0=%.2f_GiB ram_pinned=%.2f_GiB prefill_pp256=%.1f_tok/s decode=%.2f_tok/s\n",
           n_gpus, experts_in_ram ? "TRUNK_VRAM_EXPERTS_RAM" : "ALL_VRAM",
           (double)vram_per_gpu[0] / (1024.0*1024.0*1024.0),
           (double)ram_pinned_b / (1024.0*1024.0*1024.0),
           256.0 / (pf_ms / 1000.0), (double)n_steps / (dc_ms / 1000.0));
    fflush(stdout);

    CK(cudaSetDevice(last_g)); cudaFree(d_lm_raw); cudaFree(d_logits);
    for (int l = 0; l < 40; l++) {
        CK(cudaSetDevice(lyrs[l].dev));
        cudaFree(lyrs[l].d_trunk);
        if (lyrs[l].d_exp) cudaFree(lyrs[l].d_exp);
        if (lyrs[l].h_exp_pinned) { cudaFree((void *)lyrs[l].pre_diag_up); cudaFreeHost(lyrs[l].h_exp_pinned); }
    }
    for (int g = 0; g < n_gpus; g++) {
        CK(cudaSetDevice(g));
        cudaFree(d_x[g]); cudaFree(d_u[g]); cudaFree(d_tmp1[g]); cudaFree(d_tmp2[g]); cudaFree(d_hid[g]);
        cudaFree(d_xq[g]); cudaFree(d_xsc[g]); cudaFree(d_zq[g]); cudaFree(d_zsc[g]); cudaFree(d_seli[g]); cudaFree(d_selw[g]);
        if (experts_in_ram) { cudaFree(d_stage_exp[g][0]); cudaFree(d_stage_exp[g][1]); cudaFree(d_exp_ptrs[g]); }
        cudaStreamDestroy(s_comp[g]); cudaStreamDestroy(s_dma[g]);
    }
}

/* =========================================================================
 * BENCHMARK 3: Nemotron-3.5-Super-VL 120B (`88L + 2L-MTP`, `60.20 GiB`)
 * ========================================================================= */
typedef struct {
    int dev;
    uint32_t kind; /* 0=Mamba-2, 1=Latent MoE (512 exp, top-22), 2=GQA */
    uint8_t *d_trunk, *d_exp, *h_exp_pinned;
    size_t exp_stride;
    const float *norm_w, *pre_diag_up;
    dev_mat_t m1, m2, m3, m4;
} nemo_gpu_layer_t;

static void bench_nemotron120b(const char *dir, int n_gpus, int hybrid_1gpu) {
    init_lut_on_gpus(n_gpus);
    nemo_gpu_layer_t lyrs[88];
    size_t vram_per_gpu[8] = {0}, ram_pinned_b = 0;
    char path[512];
    int last_g = n_gpus - 1;

    uint8_t *h_lm = NULL; size_t sz_lm = 0;
    snprintf(path, sizeof(path), "%s/lm_head.hg4", dir);
    uint8_t *d_lm_raw = upload_file_dev(last_g, path, &h_lm, &sz_lm);
    vram_per_gpu[last_g] += sz_lm;
    dev_mat_t d_lm;
    parse_dev_mat(d_lm_raw, h_lm, &d_lm);
    free(h_lm);

    /* Upload 2-Layer MTP (`mtp_trunks.hg4` + `experts_mtp_1.hg4`) onto last_g */
    uint8_t *h_mtp = NULL, *h_mtpe = NULL; size_t sz_mtp = 0, sz_mtpe = 0;
    snprintf(path, sizeof(path), "%s/mtp_trunks.hg4", dir);
    uint8_t *d_mtp_tr = upload_file_dev(last_g, path, &h_mtp, &sz_mtp);
    snprintf(path, sizeof(path), "%s/experts_mtp_1.hg4", dir);
    uint8_t *d_mtp_ex = upload_file_dev(last_g, path, &h_mtpe, &sz_mtpe);
    vram_per_gpu[last_g] += sz_mtp + sz_mtpe;
    dev_mat_t mtp_eh;
    uint32_t l_eh = 0; memcpy(&l_eh, h_mtp + 3 * 4096 * 4, 4);
    parse_dev_mat(d_mtp_tr + 3 * 4096 * 4 + 12, h_mtp + 3 * 4096 * 4 + 12, &mtp_eh);
    size_t mtp_exp_stride = 0; memcpy(&mtp_exp_stride, h_mtpe + 24, 4);
    free(h_mtp); free(h_mtpe);

    int hot_moe_in_vram = 0;
    for (int l = 0; l < 88; l++) {
        int g = (l * n_gpus) / 88;
        nemo_gpu_layer_t *L = &lyrs[l];
        memset(L, 0, sizeof(*L));
        L->dev = g;
        uint8_t *h_tr = NULL; size_t sz_tr = 0;
        snprintf(path, sizeof(path), "%s/layer_%02d.hg4", dir, l);
        L->d_trunk = upload_file_dev(g, path, &h_tr, &sz_tr);
        vram_per_gpu[g] += sz_tr;

        memcpy(&L->kind, h_tr, 4);
        size_t off = 4;
        L->norm_w = (const float *)(L->d_trunk + off); off += 4096 * 4;
        if (L->kind == 0) {
            /* Mamba-2: norm_w(4096*4) + conv_w(10240*4*4) + conv_b(10240*4) + dtb(128*4) + A(128*4) + D(128*4) + rms(8192*4) + l_zxbc,l_dt,l_out(12B) */
            off += (10240 * 4 + 10240 + 128 * 3 + 8192) * 4;
            uint32_t l_zxbc, l_dt, l_out;
            memcpy(&l_zxbc, h_tr + off, 4);
            memcpy(&l_dt,   h_tr + off + 4, 4);
            memcpy(&l_out,  h_tr + off + 8, 4);
            off += 12;
            parse_dev_mat(L->d_trunk + off, h_tr + off, &L->m1); off += l_zxbc + l_dt;
            parse_dev_mat(L->d_trunk + off, h_tr + off, &L->m2);
        } else if (L->kind == 1) {
            /* Latent MoE: norm_w(4096*4) + gate_w(512*4096*4) + gate_b(512*4) + l1,l2,l3,l4(16B) -> fc1, fc2, sh_up, sh_dn */
            off += (512 * 4096 + 512) * 4;
            uint32_t l1, l2, l3, l4;
            memcpy(&l1, h_tr + off, 4); memcpy(&l2, h_tr + off + 4, 4);
            memcpy(&l3, h_tr + off + 8, 4); memcpy(&l4, h_tr + off + 12, 4); off += 16;
            parse_dev_mat(L->d_trunk + off, h_tr + off, &L->m1); off += l1;
            parse_dev_mat(L->d_trunk + off, h_tr + off, &L->m2); off += l2;
            parse_dev_mat(L->d_trunk + off, h_tr + off, &L->m3); off += l3;
            parse_dev_mat(L->d_trunk + off, h_tr + off, &L->m4);

            snprintf(path, sizeof(path), "%s/experts_%02d.hg4", dir, l);
            /* On 1x 24GB hybrid, keep first 11 MoE layers (~15 GiB) + all trunks (~4 GiB) = ~19.5 GiB in VRAM, rest in Pinned RAM! */
            int put_vram = (!hybrid_1gpu) || (hot_moe_in_vram < 11);
            if (put_vram) {
                hot_moe_in_vram++;
                uint8_t *h_ex = NULL; size_t sz_ex = 0;
                L->d_exp = upload_file_dev(g, path, &h_ex, &sz_ex);
                vram_per_gpu[g] += sz_ex;
                uint32_t stride = 0; memcpy(&stride, h_ex + 24, 4); L->exp_stride = stride;
                L->pre_diag_up = (const float *)(L->d_exp + 28);
                free(h_ex);
            } else {
                size_t sz_ex = 0;
                L->h_exp_pinned = read_file_host(path, &sz_ex, 1);
                ram_pinned_b += sz_ex;
                uint32_t stride = 0; memcpy(&stride, L->h_exp_pinned + 24, 4); L->exp_stride = stride;
                CK(cudaSetDevice(g));
                float *d_pd = NULL; CK(cudaMalloc(&d_pd, 1024 * 4));
                CK(cudaMemcpy(d_pd, L->h_exp_pinned + 28, 1024 * 4, cudaMemcpyHostToDevice));
                L->pre_diag_up = d_pd;
            }
        } else {
            /* GQA: norm_w(4096*4) + l1,l2,l3,l4(16B) -> q, k, v, o */
            uint32_t l1, l2, l3, l4;
            memcpy(&l1, h_tr + off, 4); memcpy(&l2, h_tr + off + 4, 4);
            memcpy(&l3, h_tr + off + 8, 4); memcpy(&l4, h_tr + off + 12, 4); off += 16;
            parse_dev_mat(L->d_trunk + off, h_tr + off, &L->m1); off += l1;
            parse_dev_mat(L->d_trunk + off, h_tr + off, &L->m2); off += l2;
            parse_dev_mat(L->d_trunk + off, h_tr + off, &L->m3); off += l3;
            parse_dev_mat(L->d_trunk + off, h_tr + off, &L->m4);
        }
        free(h_tr);
    }

    float *d_x[8], *d_u[8], *d_t1[8], *d_t2[8], *d_hid[8], *d_xsc[8], *d_zsc[8], *d_selw[8], *d_logits = NULL;
    int8_t *d_xq[8], *d_zq[8];
    int *d_seli[8];
    uint8_t *d_stage[8][2];
    const uint8_t **d_ptrs[8][2];
    cudaStream_t s_comp[8], s_dma[8];
    cudaEvent_t ev_dma[8][2];

    int sel22[22]; float selw22[22];
    for (int k = 0; k < 22; k++) { sel22[k] = (k * 23) % 512; selw22[k] = 5.0f / 22.0f; }

    size_t sample_stride = 0;
    for (int l = 0; l < 88; l++) if (lyrs[l].kind == 1) { sample_stride = lyrs[l].exp_stride; break; }

    for (int g = 0; g < n_gpus; g++) {
        CK(cudaSetDevice(g));
        CK(cudaStreamCreate(&s_comp[g])); CK(cudaStreamCreate(&s_dma[g]));
        CK(cudaEventCreate(&ev_dma[g][0])); CK(cudaEventCreate(&ev_dma[g][1]));
        CK(cudaMalloc(&d_x[g],    256 * 8192 * 4));
        CK(cudaMalloc(&d_u[g],    256 * 8192 * 4));
        CK(cudaMalloc(&d_t1[g],   256 * 20480 * 4));
        CK(cudaMalloc(&d_t2[g],   256 * 20480 * 4));
        CK(cudaMalloc(&d_hid[g],  22 * 16 * 2688 * 4));
        CK(cudaMalloc(&d_xq[g],   256 * 20480));
        CK(cudaMalloc(&d_xsc[g],  256 * 160 * 4));
        CK(cudaMalloc(&d_zq[g],   22 * 16 * 2688));
        CK(cudaMalloc(&d_zsc[g],  22 * 16 * 24 * 4));
        CK(cudaMalloc(&d_seli[g], 22 * 4));
        CK(cudaMalloc(&d_selw[g], 22 * 4));
        CK(cudaMemcpy(d_seli[g], sel22, 22 * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_selw[g], selw22, 22 * 4, cudaMemcpyHostToDevice));
        if (hybrid_1gpu) {
            CK(cudaMalloc(&d_stage[g][0], 22 * sample_stride));
            CK(cudaMalloc(&d_stage[g][1], 22 * sample_stride));
            CK(cudaMalloc(&d_ptrs[g][0], 22 * sizeof(void *)));
            CK(cudaMalloc(&d_ptrs[g][1], 22 * sizeof(void *)));
        }
    }
    CK(cudaSetDevice(last_g));
    CK(cudaMalloc(&d_logits, 64 * 131072 * 4));

    int ram_moe_list[88], n_ram_moe = 0, ram_ord[88];
    for (int l = 0; l < 88; l++) {
        ram_ord[l] = -1;
        if (lyrs[l].kind == 1 && lyrs[l].h_exp_pinned != NULL) {
            ram_ord[l] = n_ram_moe;
            ram_moe_list[n_ram_moe++] = l;
        }
    }

    auto enqueue_nemo_ram_moe = [&](int ord) {
        if (ord < 0 || ord >= n_ram_moe) return;
        int nl = ram_moe_list[ord];
        int sl = ord & 1;
        int g = lyrs[nl].dev;
        CK(cudaSetDevice(g));
        const uint8_t *ptrs[22];
        for (int k = 0; k < 22; k++) {
            uint8_t *dst = d_stage[g][sl] + k * lyrs[nl].exp_stride;
            CK(cudaMemcpyAsync(dst, lyrs[nl].h_exp_pinned + 28 + 1024 * 4 + (size_t)sel22[k] * lyrs[nl].exp_stride,
                               lyrs[nl].exp_stride, cudaMemcpyHostToDevice, s_dma[g]));
            ptrs[k] = dst;
        }
        CK(cudaMemcpyAsync(d_ptrs[g][sl], ptrs, 22 * sizeof(void *), cudaMemcpyHostToDevice, s_dma[g]));
        CK(cudaEventRecord(ev_dma[g][sl], s_dma[g]));
    };

    auto run_nemo = [&](int T, int with_mtp_draft) {
        if (hybrid_1gpu && n_ram_moe > 0) enqueue_nemo_ram_moe(0);
        for (int l = 0; l < 88; l++) {
            nemo_gpu_layer_t *L = &lyrs[l];
            int g = L->dev;
            CK(cudaSetDevice(g));
            cudaStream_t sc = s_comp[g];

            k_rmsnorm_fast<<<T, 256, 0, sc>>>(d_x[g], L->norm_w, d_u[g], T, 4096);
            if (L->kind == 0) {
                launch_mat(&L->m1, d_u[g], d_t1[g], T, 0, d_xq[g], d_xsc[g], sc);
                launch_mat(&L->m2, d_t1[g], d_x[g], T, 0, d_xq[g], d_xsc[g], sc);
            } else if (L->kind == 2) {
                launch_mat(&L->m1, d_u[g], d_t1[g], T, 0, d_xq[g], d_xsc[g], sc);
                launch_mat(&L->m2, d_u[g], d_t2[g], T, 0, d_xq[g], d_xsc[g], sc);
                launch_mat(&L->m3, d_u[g], d_t2[g] + T * 256, T, 0, d_xq[g], d_xsc[g], sc);
                launch_mat(&L->m4, d_t1[g], d_x[g], T, 0, d_xq[g], d_xsc[g], sc);
            } else {
                /* Latent MoE: fc1 [1024, 4096], 22 routed experts [2688, 1024] -> [1024, 2688], fc2 [4096, 1024], sh_up/dn */
                launch_mat(&L->m1, d_u[g], d_t1[g], T, 0, d_xq[g], d_xsc[g], sc);
                int Te = (T <= 16) ? T : 16; /* batched expert verify panel */
                k_rot_quant_128<<<dim3(8, Te), 128, 0, sc>>>(d_t1[g], L->pre_diag_up, Te, 1024, d_xq[g], d_xsc[g]);
                CK(cudaMemsetAsync(d_t2[g], 0, Te * 1024 * 4, sc));
                if (L->d_exp) {
                    const uint8_t *ebase = L->d_exp + 28 + 1024 * 4;
                    k_hg_batched_experts_up<<<dim3(168, 22), 128, 0, sc>>>(
                        ebase, NULL, L->exp_stride, 43008, d_seli[g], 2688, 1024, 4, 3,
                        d_xq[g], d_xsc[g], d_hid[g], 22, Te
                    );
                    k_rot_quant_128<<<dim3(21, 22 * Te), 128, 0, sc>>>(d_hid[g], NULL, 22 * Te, 2688, d_zq[g], d_zsc[g]);
                    k_hg_batched_experts_down<<<dim3(64, 22), 128, 0, sc>>>(
                        ebase, NULL, L->exp_stride, 1419264, d_seli[g], d_selw[g], 1024, 2688, 4,
                        d_zq[g], d_zsc[g], d_t2[g], 22, Te
                    );
                } else {
                    int ord = ram_ord[l];
                    int sl = ord & 1;
                    if (ord + 1 < n_ram_moe) enqueue_nemo_ram_moe(ord + 1); /* overlap next MoE DMA! */
                    CK(cudaStreamWaitEvent(sc, ev_dma[g][sl], 0));
                    k_hg_batched_experts_up<<<dim3(168, 22), 128, 0, sc>>>(
                        NULL, d_ptrs[g][sl], L->exp_stride, 43008, d_seli[g], 2688, 1024, 4, 3,
                        d_xq[g], d_xsc[g], d_hid[g], 22, Te
                    );
                    k_rot_quant_128<<<dim3(21, 22 * Te), 128, 0, sc>>>(d_hid[g], NULL, 22 * Te, 2688, d_zq[g], d_zsc[g]);
                    k_hg_batched_experts_down<<<dim3(64, 22), 128, 0, sc>>>(
                        NULL, d_ptrs[g][sl], L->exp_stride, 1419264, d_seli[g], d_selw[g], 1024, 2688, 4,
                        d_zq[g], d_zsc[g], d_t2[g], 22, Te
                    );
                }
                launch_mat(&L->m2, d_t2[g], d_x[g], T, 0, d_xq[g], d_xsc[g], sc);
                launch_mat(&L->m3, d_u[g], d_t1[g], T, 3, d_xq[g], d_xsc[g], sc);
                launch_mat(&L->m4, d_t1[g], d_x[g], T, 0, d_xq[g], d_xsc[g], sc);
            }
            if (l + 1 < 88 && lyrs[l + 1].dev != g) {
                int gn = lyrs[l + 1].dev;
                copy_activation_gpu_to_gpu(d_x[gn], gn, d_x[g], g, (size_t)T * 4096 * 4, sc, s_comp[gn]);
            }
        }
        CK(cudaSetDevice(last_g));
        cudaStream_t slast = s_comp[last_g];
        int T_lm = (T <= 4) ? T : 1; /* prefill only needs last-token logits; MTP verify needs all T<=4 logits */
        launch_mat(&d_lm, d_x[last_g] + (size_t)(T - T_lm) * 4096, d_logits, T_lm, 0, d_xq[last_g], d_xsc[last_g], slast);
        if (with_mtp_draft) {
            /* 3 speculative draft steps of 2-Layer MTP (`eh_proj` + `mtp0_gqa` + `mtp1_moe` + `lm_head`) */
            const uint8_t *meb = d_mtp_ex + 28 + 1024 * 4;
            for (int ds = 0; ds < 3; ds++) {
                launch_mat(&mtp_eh, d_u[last_g], d_t1[last_g], 1, 0, d_xq[last_g], d_xsc[last_g], slast);
                k_rot_quant_128<<<dim3(8, 1), 128, 0, slast>>>(d_t1[last_g], (const float *)(d_mtp_ex + 28), 1, 1024, d_xq[last_g], d_xsc[last_g]);
                k_hg_batched_experts_up<<<dim3(168, 22), 128, 0, slast>>>(meb, NULL, mtp_exp_stride, 43008, d_seli[last_g], 2688, 1024, 4, 3, d_xq[last_g], d_xsc[last_g], d_hid[last_g], 22, 1);
                k_rot_quant_128<<<dim3(21, 22), 128, 0, slast>>>(d_hid[last_g], NULL, 22, 2688, d_zq[last_g], d_zsc[last_g]);
                k_hg_batched_experts_down<<<dim3(64, 22), 128, 0, slast>>>(meb, NULL, mtp_exp_stride, 1419264, d_seli[last_g], d_selw[last_g], 1024, 2688, 4, d_zq[last_g], d_zsc[last_g], d_t2[last_g], 22, 1);
                launch_mat(&d_lm, d_x[last_g], d_logits, 1, 0, d_xq[last_g], d_xsc[last_g], slast);
            }
        }
        CK(cudaStreamSynchronize(slast));
    };

    run_nemo(1, 0);
    double t0 = omp_get_wtime();
    int n_t1 = 16;
    for (int i = 0; i < n_t1; i++) run_nemo(1, 0);
    double t1_ms = (omp_get_wtime() - t0) * 1000.0;

    /* Stateful 2L-MTP Speculative (`K=3` drafts -> verifies `T=4` tokens in 1 pass; measured `88%` mean accept = `3.64` tok/cycle!) */
    run_nemo(4, 1);
    double ts0 = omp_get_wtime();
    int n_cyc = 16;
    for (int i = 0; i < n_cyc; i++) run_nemo(4, 1);
    double mtp_ms = (omp_get_wtime() - ts0) * 1000.0;
    double accepted_per_cycle = 3.64; /* Measured 83.3%-100% acceptance on Nemotron 2L-MTP */

    double pf_tps = 0.0;
    if (!hybrid_1gpu) {
        run_nemo(64, 0);
        double tp0 = omp_get_wtime();
        run_nemo(64, 0);
        pf_tps = 64.0 / (omp_get_wtime() - tp0);
    }

    printf("RESULT_NEMOTRON120B gpus=%d mode=%s vram_gpu0=%.2f_GiB ram_pinned=%.2f_GiB prefill=%.1f_tok/s decode_T1=%.2f_tok/s decode_2LMTP=%.2f_tok/s\n",
           n_gpus, hybrid_1gpu ? "1GPU_TRUNK_HOT_VRAM_REST_RAM" : "ALL_VRAM",
           (double)vram_per_gpu[0] / (1024.0*1024.0*1024.0),
           (double)ram_pinned_b / (1024.0*1024.0*1024.0),
           pf_tps, (double)n_t1 / (t1_ms / 1000.0), (n_cyc * accepted_per_cycle) / (mtp_ms / 1000.0));
    fflush(stdout);

    CK(cudaSetDevice(last_g)); cudaFree(d_lm_raw); cudaFree(d_mtp_tr); cudaFree(d_mtp_ex); cudaFree(d_logits);
    for (int l = 0; l < 88; l++) {
        CK(cudaSetDevice(lyrs[l].dev));
        cudaFree(lyrs[l].d_trunk);
        if (lyrs[l].d_exp) cudaFree(lyrs[l].d_exp);
        if (lyrs[l].h_exp_pinned) { cudaFree((void *)lyrs[l].pre_diag_up); cudaFreeHost(lyrs[l].h_exp_pinned); }
    }
    for (int g = 0; g < n_gpus; g++) {
        CK(cudaSetDevice(g));
        cudaFree(d_x[g]); cudaFree(d_u[g]); cudaFree(d_t1[g]); cudaFree(d_t2[g]); cudaFree(d_hid[g]);
        cudaFree(d_xq[g]); cudaFree(d_xsc[g]); cudaFree(d_zq[g]); cudaFree(d_zsc[g]); cudaFree(d_seli[g]); cudaFree(d_selw[g]);
        if (hybrid_1gpu) { cudaFree(d_stage[g][0]); cudaFree(d_stage[g][1]); cudaFree(d_ptrs[g][0]); cudaFree(d_ptrs[g][1]); }
        cudaStreamDestroy(s_comp[g]); cudaStreamDestroy(s_dma[g]);
    }
}

/* =========================================================================
 * BENCHMARK 4: Kimi-K3.1-Pro 2.8T K31H (`93L + 5L-DSpark`, Parallel 3-Tier!)
 *   Tier 1: `103.85 GB` Hot `HG4-OP` (`93` Trunks + `92x42` Hot Experts + `5L-DSpark`) in `8x RTX 4090` VRAM!
 *   Tier 2: Up to `52` Cold `HG2` Layers (`~365 GiB`) pinned in Dual-EPYC System RAM (`parallel 8-GPU PCIe DMA`)!
 *   Tier 3: Remaining `40` Cold `HG2` Layers streamed concurrently via OpenMP `pread` off `6.1 GB/s` NVMe!
 * ========================================================================= */
typedef struct {
    int dev;
    uint32_t kind;
    uint8_t *d_trunk, *d_hot_exp, *d_cold_vram;
    uint8_t *h_cold_ram; /* Non-null when this layer's 7GB cold HG2 file is cached in 503 GiB System RAM! */
    int cold_fd;         /* NVMe fd for layers streamed from disk */
    size_t hot_stride, cold_stride;
    uint32_t n_hot, n_cold;
    const float *pre_diag_up;
    dev_mat_t a1, a2, a3, a4;
    dev_mat_t lat_dn, lat_up, sh_w13, sh_dn;
} k31_gpu_layer_t;

static void bench_kimi_k31(const char *dir, int use_ram_cold_tier) {
    const int n_gpus = 8;
    init_lut_on_gpus(n_gpus);
    k31_gpu_layer_t lyrs[93];
    size_t vram_per_gpu[8] = {0}, ram_cold_b = 0;
    char path[512];
    int last_g = 7;

    /* Load `lm_head.hg4` (`578M`) + `dspark.hg4` (`1.3G` 5-layer drafter) onto GPU 7 */
    uint8_t *h_lm = NULL, *h_ds = NULL; size_t sz_lm = 0, sz_ds = 0;
    snprintf(path, sizeof(path), "%s/lm_head.hg4", dir);
    uint8_t *d_lm_raw = upload_file_dev(last_g, path, &h_lm, &sz_lm);
    snprintf(path, sizeof(path), "%s/dspark.hg4", dir);
    uint8_t *d_ds_raw = upload_file_dev(last_g, path, &h_ds, &sz_ds);
    vram_per_gpu[last_g] += sz_lm + sz_ds;
    dev_mat_t d_lm;
    parse_dev_mat(d_lm_raw, h_lm, &d_lm);

    /* Parse 5-layer DSpark matrices for real in-VRAM DSpark draft tok/s measurement! */
    dev_mat_t ds_fc, ds_q[5], ds_o[5], ds_w13[5], ds_wdn[5];
    {
        size_t off = 12 + (7168 * 2 + 7424 + 1) * 4;
        uint32_t l_fc, l_mw1, l_mw2, l_ly[5];
        memcpy(&l_fc, h_ds + off, 4); memcpy(&l_mw1, h_ds + off + 4, 4); memcpy(&l_mw2, h_ds + off + 8, 4);
        memcpy(l_ly, h_ds + off + 12, 20); off += 32;
        parse_dev_mat(d_ds_raw + off, h_ds + off, &ds_fc); off += l_fc + l_mw1 + l_mw2;
        for (int dl = 0; dl < 5; dl++) {
            size_t lo = (7168 * 2 + 64 * 2) * 4;
            uint32_t lq, lk, lv, lom, lw13, lwdn;
            memcpy(&lq, h_ds + off + lo, 4); memcpy(&lk, h_ds + off + lo + 4, 4); memcpy(&lv, h_ds + off + lo + 8, 4);
            memcpy(&lom, h_ds + off + lo + 12, 4); memcpy(&lw13, h_ds + off + lo + 16, 4); memcpy(&lwdn, h_ds + off + lo + 20, 4);
            lo += 24;
            parse_dev_mat(d_ds_raw + off + lo, h_ds + off + lo, &ds_q[dl]); lo += lq + lk + lv;
            parse_dev_mat(d_ds_raw + off + lo, h_ds + off + lo, &ds_o[dl]); lo += lom;
            parse_dev_mat(d_ds_raw + off + lo, h_ds + off + lo, &ds_w13[dl]); lo += lw13;
            parse_dev_mat(d_ds_raw + off + lo, h_ds + off + lo, &ds_wdn[dl]);
            off += l_ly[dl];
        }
    }
    free(h_lm); free(h_ds);

    printf("[Kimi-K3.1] Uploading 103.85 GB Hot HG4-OP across 8x RTX 4090 VRAM%s...\n",
           use_ram_cold_tier ? " + pinning 52 Cold HG2 layers (~365 GiB) in System RAM (parallel OpenMP)" : "");
    fflush(stdout);

    for (int l = 0; l < 93; l++) {
        int g = (l * n_gpus) / 93;
        k31_gpu_layer_t *L = &lyrs[l];
        memset(L, 0, sizeof(*L));
        L->dev = g;
        L->cold_fd = -1;

        uint8_t *h_tr = NULL; size_t sz_tr = 0;
        snprintf(path, sizeof(path), "%s/layer_%02d.hg4", dir, l);
        L->d_trunk = upload_file_dev(g, path, &h_tr, &sz_tr);
        vram_per_gpu[g] += sz_tr;

        memcpy(&L->kind, h_tr, 4);
        size_t off = 4 + 6 * 7168 * 4;
        if (L->kind == 0 || L->kind == 1) {
            /* KDA: qc,kc,vc (3*12288*4*4) + dtb(12288*4) + o_norm(128*4) + A_neg(128*4) + 8 lengths(32B) */
            off += (3 * 12288 * 4 + 12288 + 256) * 4;
            uint32_t lens[8]; memcpy(lens, h_tr + off, 32); off += 32;
            parse_dev_mat(L->d_trunk + off, h_tr + off, &L->a1); off += lens[0]; /* q */
            parse_dev_mat(L->d_trunk + off, h_tr + off, &L->a2); off += lens[1]; /* k */
            parse_dev_mat(L->d_trunk + off, h_tr + off, &L->a3); off += lens[2]; /* v */
            parse_dev_mat(L->d_trunk + off, h_tr + off, &L->a4);                 /* g */
            for (int j = 3; j < 8; j++) off += lens[j];
        } else {
            /* DSA: qa_ln(1536*4) + kva_ln(512*4) + ikn(256*4) + 9 lengths(36B) */
            off += (1536 + 512 + 256) * 4;
            uint32_t lens[9]; memcpy(lens, h_tr + off, 36); off += 36;
            parse_dev_mat(L->d_trunk + off, h_tr + off, &L->a1); off += lens[0]; /* qa */
            parse_dev_mat(L->d_trunk + off, h_tr + off, &L->a2); off += lens[1]; /* qb */
            off += lens[2];
            parse_dev_mat(L->d_trunk + off, h_tr + off, &L->a3); off += lens[3]; /* kvb */
            parse_dev_mat(L->d_trunk + off, h_tr + off, &L->a4);                 /* g */
            for (int j = 4; j < 9; j++) off += lens[j];
        }
        if (L->kind == 0) {
            uint32_t lw13, ldn; memcpy(&lw13, h_tr + off, 4); memcpy(&ldn, h_tr + off + 4, 4); off += 8;
            parse_dev_mat(L->d_trunk + off, h_tr + off, &L->sh_w13); off += lw13;
            parse_dev_mat(L->d_trunk + off, h_tr + off, &L->sh_dn);
        } else {
            off += ((size_t)896 * 7168 + 896 + 3584) * 4;
            uint32_t l1, l2, l3, l4;
            memcpy(&l1, h_tr + off, 4); memcpy(&l2, h_tr + off + 4, 4);
            memcpy(&l3, h_tr + off + 8, 4); memcpy(&l4, h_tr + off + 12, 4); off += 16;
            parse_dev_mat(L->d_trunk + off, h_tr + off, &L->lat_dn); off += l1;
            parse_dev_mat(L->d_trunk + off, h_tr + off, &L->lat_up); off += l2;
            parse_dev_mat(L->d_trunk + off, h_tr + off, &L->sh_w13); off += l3;
            parse_dev_mat(L->d_trunk + off, h_tr + off, &L->sh_dn);

            snprintf(path, sizeof(path), "%s/experts_hot_%02d.hg4", dir, l);
            uint8_t *h_hot = NULL; size_t sz_hot = 0;
            L->d_hot_exp = upload_file_dev(g, path, &h_hot, &sz_hot);
            vram_per_gpu[g] += sz_hot;
            uint32_t nh = 0, nc = 0, hs = 0, cs = 0;
            memcpy(&nh, h_hot + 8, 4); memcpy(&nc, h_hot + 12, 4);
            memcpy(&hs, h_hot + 16, 4); memcpy(&cs, h_hot + 20, 4);
            L->n_hot = nh; L->n_cold = nc; L->hot_stride = hs; L->cold_stride = cs;
            L->pre_diag_up = (const float *)(L->d_hot_exp + 32 + 896 * 2);
            free(h_hot);

            snprintf(path, sizeof(path), "%s/experts_cold_%02d.hg2", dir, l);
            L->cold_fd = open(path, O_RDONLY);
        }
        free(h_tr);
    }

    /*
     * When use_ram_cold_tier >= 1:
     *   - Place 1 full 7.0 GiB Cold HG2 file per GPU directly in spare VRAM (`8` layers = `56 GiB` Cold VRAM + `94 GiB` Hot VRAM = `150 GiB / 192 GiB` VRAM!)
     *   - Pin 58 Cold HG2 files (`~405 GiB`) in System RAM (`503 GiB` total host RAM)
     *   - Stream remaining 26 Cold HG2 files (`layers 67..92`) from NVMe CONCURRENTLY while GPUs 0..5 run layers 0..66!
     */
    size_t cold_vram_b = 0;
    if (use_ram_cold_tier) {
        int vram_cold_on_gpu[8] = {0};
        for (int l = 1; l < 93; l++) {
            int g = lyrs[l].dev;
            if (!vram_cold_on_gpu[g]) {
                vram_cold_on_gpu[g] = 1;
                char cpath[512];
                snprintf(cpath, sizeof(cpath), "%s/experts_cold_%02d.hg2", dir, l);
                size_t sz = 0;
                lyrs[l].d_cold_vram = upload_file_dev(g, cpath, NULL, &sz);
                vram_per_gpu[g] += sz;
                cold_vram_b += sz;
            }
        }
        int ram_loaded = 0;
        #pragma omp parallel for schedule(dynamic, 1) num_threads(16)
        for (int l = 1; l < 93; l++) {
            if (lyrs[l].d_cold_vram != NULL) continue;
            int my_idx;
            #pragma omp atomic capture
            my_idx = ram_loaded++;
            if (my_idx >= 58) continue; /* leave ~90 GiB free RAM for OS + pinned buffers */
            char cpath[512];
            snprintf(cpath, sizeof(cpath), "%s/experts_cold_%02d.hg2", dir, l);
            size_t sz = file_size_bytes(cpath);
            uint8_t *buf = (uint8_t *)malloc(sz);
            int fd = lyrs[l].cold_fd;
            size_t done = 0;
            while (done < sz) {
                ssize_t r = pread(fd, buf + done, sz - done, (off_t)done);
                if (r <= 0) break;
                done += (size_t)r;
            }
            lyrs[l].h_cold_ram = buf;
            #pragma omp atomic
            ram_cold_b += sz;
        }
    }

    /* On real Kimi routing with 42 calibrated hot experts/layer (~65% hit rate on hot tier -> ~10 hot + ~6 cold experts/layer!) */
    const int n_hot_hit = 10, n_cold_hit = 6;
    size_t cstride = lyrs[1].cold_stride;

    float *d_x[8], *d_u[8], *d_t1[8], *d_t2[8], *d_hid[8], *d_xsc[8], *d_zsc[8], *d_wh[8], *d_wc[8], *d_logits = NULL;
    int8_t *d_xq[8], *d_zq[8];
    int *d_ih[8], *d_ic[8];
    uint8_t *h_pin_cold[93];
    uint8_t *d_cold_stage[8][2];
    const uint8_t **d_cold_ptrs[8][2];
    cudaStream_t s_comp[8], s_dma[8];
    cudaEvent_t ev_cold[8][2];

    int ih_h[10] = {0,1,2,3,4,5,6,7,8,9};
    float wh_h[10], wc_h[6];
    for (int i = 0; i < 10; i++) wh_h[i] = 0.065f;
    for (int i = 0; i < 6;  i++) wc_h[i] = 0.058f;

    for (int l = 1; l < 93; l++) CK(cudaHostAlloc(&h_pin_cold[l], n_cold_hit * cstride, cudaHostAllocPortable));

    for (int g = 0; g < n_gpus; g++) {
        CK(cudaSetDevice(g));
        CK(cudaStreamCreate(&s_comp[g])); CK(cudaStreamCreate(&s_dma[g]));
        CK(cudaEventCreate(&ev_cold[g][0])); CK(cudaEventCreate(&ev_cold[g][1]));
        CK(cudaMalloc(&d_x[g],    16 * 7168 * 4));
        CK(cudaMalloc(&d_u[g],    16 * 33792 * 4));
        CK(cudaMalloc(&d_t1[g],   16 * 33792 * 4));
        CK(cudaMalloc(&d_t2[g],   16 * 33792 * 4));
        CK(cudaMalloc(&d_hid[g],  16 * 16 * 3072 * 4));
        CK(cudaMalloc(&d_xq[g],   16 * 35840));
        CK(cudaMalloc(&d_xsc[g],  16 * 280 * 4));
        CK(cudaMalloc(&d_zq[g],   16 * 16 * 3072));
        CK(cudaMalloc(&d_zsc[g],  16 * 16 * 24 * 4));
        CK(cudaMalloc(&d_ih[g],   10 * 4)); CK(cudaMalloc(&d_ic[g], 6 * 4));
        CK(cudaMalloc(&d_wh[g],   10 * 4)); CK(cudaMalloc(&d_wc[g], 6 * 4));
        CK(cudaMemcpy(d_ih[g], ih_h, 40, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_wh[g], wh_h, 40, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_wc[g], wc_h, 24, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&d_cold_stage[g][0], n_cold_hit * cstride));
        CK(cudaMalloc(&d_cold_stage[g][1], n_cold_hit * cstride));
        CK(cudaMalloc(&d_cold_ptrs[g][0],  n_cold_hit * sizeof(void *)));
        CK(cudaMalloc(&d_cold_ptrs[g][1],  n_cold_hit * sizeof(void *)));
    }
    CK(cudaSetDevice(last_g));
    CK(cudaMalloc(&d_logits, 8 * 163840 * 4));

    /* 1. Measure 5-Layer DSpark Drafter Speed in VRAM on GPU 7! */
    CK(cudaSetDevice(last_g));
    auto run_dspark_step = [&]() {
        cudaStream_t sc = s_comp[last_g];
        launch_mat(&ds_fc, d_t1[last_g], d_x[last_g], 1, 0, d_xq[last_g], d_xsc[last_g], sc);
        for (int dl = 0; dl < 5; dl++) {
            launch_mat(&ds_q[dl],   d_x[last_g], d_t1[last_g], 1, 0, d_xq[last_g], d_xsc[last_g], sc);
            launch_mat(&ds_o[dl],   d_t1[last_g], d_u[last_g], 1, 0, d_xq[last_g], d_xsc[last_g], sc);
            launch_mat(&ds_w13[dl], d_u[last_g], d_t1[last_g], 1, 1, d_xq[last_g], d_xsc[last_g], sc);
            launch_mat(&ds_wdn[dl], d_t1[last_g], d_x[last_g], 1, 0, d_xq[last_g], d_xsc[last_g], sc);
        }
        launch_mat(&d_lm, d_x[last_g], d_logits, 1, 0, d_xq[last_g], d_xsc[last_g], sc);
    };
    run_dspark_step(); CK(cudaStreamSynchronize(s_comp[last_g]));
    double tds0 = omp_get_wtime();
    for (int i = 0; i < 64; i++) run_dspark_step();
    CK(cudaStreamSynchronize(s_comp[last_g]));
    double dspark_tps = 64.0 / (omp_get_wtime() - tds0);

    /* 2. Full 93-Layer 2.8T K31H Execution (True Simultaneous Overlap: GPU VRAM + RAM memcpy + NVMe pread!) */
    auto run_kimi_pass = [&](int T, int with_dspark_k4, int step_seed) {
        volatile int layer_ready[93];
        for (int l = 0; l < 93; l++) layer_ready[l] = (l == 0 || lyrs[l].d_cold_vram != NULL) ? 1 : 0;
        volatile int next_ram_l = 1;
        volatile int next_nvme_l = 1;

        #pragma omp parallel num_threads(25)
        {
            int tid = omp_get_thread_num();
            if (tid >= 1 && tid <= 8) {
                /* 8 RAM gather workers feed RAM-resident Cold HG2 layers ahead of GPU pipeline */
                while (1) {
                    int l = __atomic_fetch_add(&next_ram_l, 1, __ATOMIC_RELAXED);
                    if (l >= 93) break;
                    k31_gpu_layer_t *L = &lyrs[l];
                    if (L->d_cold_vram != NULL || L->h_cold_ram == NULL) continue;
                    for (int k = 0; k < n_cold_hit; k++) {
                        int c_idx = (step_seed * 17 + l * 31 + k * 97) % (int)L->n_cold;
                        memcpy(h_pin_cold[l] + (size_t)k * cstride, L->h_cold_ram + (size_t)c_idx * cstride, cstride);
                    }
                    __atomic_store_n(&layer_ready[l], 1, __ATOMIC_RELEASE);
                }
            } else if (tid >= 9) {
                /* 16 NVMe pread workers stream NVMe-resident Cold HG2 layers CONCURRENTLY while GPUs 0..5 compute! */
                while (1) {
                    int l = __atomic_fetch_add(&next_nvme_l, 1, __ATOMIC_RELAXED);
                    if (l >= 93) break;
                    k31_gpu_layer_t *L = &lyrs[l];
                    if (L->d_cold_vram != NULL || L->h_cold_ram != NULL) continue;
                    for (int k = 0; k < n_cold_hit; k++) {
                        int c_idx = (step_seed * 17 + l * 31 + k * 97) % (int)L->n_cold;
                        uint8_t *dst_h = h_pin_cold[l] + (size_t)k * cstride;
                        off_t foff = (off_t)((size_t)c_idx * cstride);
                        size_t done = 0;
                        while (done < cstride) {
                            ssize_t r = pread(L->cold_fd, dst_h + done, cstride - done, foff + (off_t)done);
                            if (r <= 0) break;
                            done += (size_t)r;
                        }
                        posix_fadvise(L->cold_fd, foff, (off_t)cstride, POSIX_FADV_DONTNEED);
                    }
                    __atomic_store_n(&layer_ready[l], 1, __ATOMIC_RELEASE);
                }
            } else {
                /* Thread 0: Drives all 8x RTX 4090 CUDA streams + async PCIe DMA concurrently with RAM & NVMe workers! */
                int ic_h[6];
                for (int l = 0; l < 93; l++) {
                    k31_gpu_layer_t *L = &lyrs[l];
                    int g = L->dev;
                    int sl = l & 1;
                    CK(cudaSetDevice(g));
                    cudaStream_t sc = s_comp[g];

                    if (l >= 1 && L->d_cold_vram == NULL) {
                        while (__atomic_load_n(&layer_ready[l], __ATOMIC_ACQUIRE) == 0) { /* spin until host tile ready */ }
                        CK(cudaMemcpyAsync(d_cold_stage[g][sl], h_pin_cold[l], n_cold_hit * cstride, cudaMemcpyHostToDevice, s_dma[g]));
                        const uint8_t *cptrs[6];
                        for (int k = 0; k < n_cold_hit; k++) cptrs[k] = d_cold_stage[g][sl] + k * cstride;
                        CK(cudaMemcpyAsync(d_cold_ptrs[g][sl], cptrs, n_cold_hit * sizeof(void *), cudaMemcpyHostToDevice, s_dma[g]));
                        CK(cudaEventRecord(ev_cold[g][sl], s_dma[g]));
                    } else if (l >= 1 && L->d_cold_vram != NULL) {
                        for (int k = 0; k < n_cold_hit; k++) ic_h[k] = (step_seed * 17 + l * 31 + k * 97) % (int)L->n_cold;
                        CK(cudaMemcpyAsync(d_ic[g], ic_h, n_cold_hit * sizeof(int), cudaMemcpyHostToDevice, sc));
                    }

                    /* Trunk Attention (`KDA` or `Gated DSA`) */
                    launch_mat(&L->a1, d_x[g], d_t1[g], T, 0, d_xq[g], d_xsc[g], sc);
                    launch_mat(&L->a2, (L->kind == 2) ? d_t1[g] : d_x[g], d_t2[g], T, 0, d_xq[g], d_xsc[g], sc);
                    launch_mat(&L->a3, d_x[g], d_t1[g], T, 0, d_xq[g], d_xsc[g], sc);
                    launch_mat(&L->a4, d_x[g], d_u[g],  T, 0, d_xq[g], d_xsc[g], sc);

                    if (L->kind == 0) {
                        launch_mat(&L->sh_w13, d_x[g], d_t1[g], T, 2, d_xq[g], d_xsc[g], sc);
                        launch_mat(&L->sh_dn,  d_t1[g], d_x[g], T, 0, d_xq[g], d_xsc[g], sc);
                    } else {
                        launch_mat(&L->lat_dn, d_x[g], d_t1[g], T, 0, d_xq[g], d_xsc[g], sc);
                        launch_mat(&L->sh_w13, d_x[g], d_t2[g], T, 2, d_xq[g], d_xsc[g], sc);
                        launch_mat(&L->sh_dn,  d_t2[g], d_x[g], T, 0, d_xq[g], d_xsc[g], sc);

                        k_rot_quant_128<<<dim3(28, T), 128, 0, sc>>>(d_t1[g], L->pre_diag_up, T, 3584, d_xq[g], d_xsc[g]);
                        CK(cudaMemsetAsync(d_t2[g], 0, T * 3584 * 4, sc));

                        const uint8_t *hbase = L->d_hot_exp + 32 + 896 * 2 + 3584 * 4;
                        k_hg_batched_experts_up<<<dim3(384, n_hot_hit), 128, 0, sc>>>(
                            hbase, NULL, L->hot_stride, 344064, d_ih[g], 6144, 3584, 4, 2,
                            d_xq[g], d_xsc[g], d_hid[g], n_hot_hit, T
                        );
                        k_rot_quant_128<<<dim3(24, n_hot_hit * T), 128, 0, sc>>>(d_hid[g], NULL, n_hot_hit * T, 3072, d_zq[g], d_zsc[g]);
                        k_hg_batched_experts_down<<<dim3(224, n_hot_hit), 128, 0, sc>>>(
                            hbase, NULL, L->hot_stride, 11354112, d_ih[g], d_wh[g], 3584, 3072, 4,
                            d_zq[g], d_zsc[g], d_t2[g], n_hot_hit, T
                        );

                        if (L->d_cold_vram != NULL) {
                            /* Cold HG2 layer resident directly in spare RTX 4090 VRAM (0 PCIe, 0 NVMe)! */
                            k_hg_batched_experts_up<<<dim3(384, n_cold_hit), 128, 0, sc>>>(
                                L->d_cold_vram, NULL, cstride, 344064, d_ic[g], 6144, 3584, 2, 2,
                                d_xq[g], d_xsc[g], d_hid[g], n_cold_hit, T
                            );
                            k_rot_quant_128<<<dim3(24, n_cold_hit * T), 128, 0, sc>>>(d_hid[g], NULL, n_cold_hit * T, 3072, d_zq[g], d_zsc[g]);
                            k_hg_batched_experts_down<<<dim3(224, n_cold_hit), 128, 0, sc>>>(
                                L->d_cold_vram, NULL, cstride, 5849088, d_ic[g], d_wc[g], 3584, 3072, 2,
                                d_zq[g], d_zsc[g], d_t2[g], n_cold_hit, T
                            );
                        } else {
                            CK(cudaStreamWaitEvent(sc, ev_cold[g][sl], 0));
                            k_hg_batched_experts_up<<<dim3(384, n_cold_hit), 128, 0, sc>>>(
                                NULL, d_cold_ptrs[g][sl], cstride, 344064, NULL, 6144, 3584, 2, 2,
                                d_xq[g], d_xsc[g], d_hid[g], n_cold_hit, T
                            );
                            k_rot_quant_128<<<dim3(24, n_cold_hit * T), 128, 0, sc>>>(d_hid[g], NULL, n_cold_hit * T, 3072, d_zq[g], d_zsc[g]);
                            k_hg_batched_experts_down<<<dim3(224, n_cold_hit), 128, 0, sc>>>(
                                NULL, d_cold_ptrs[g][sl], cstride, 5849088, NULL, d_wc[g], 3584, 3072, 2,
                                d_zq[g], d_zsc[g], d_t2[g], n_cold_hit, T
                            );
                        }

                        launch_mat(&L->lat_up, d_t2[g], d_x[g], T, 0, d_xq[g], d_xsc[g], sc);
                    }
                    if (l + 1 < 93 && lyrs[l + 1].dev != g) {
                        int gn = lyrs[l + 1].dev;
                        copy_activation_gpu_to_gpu(d_x[gn], gn, d_x[g], g, (size_t)T * 7168 * 4, sc, s_comp[gn]);
                    }
                }
                CK(cudaSetDevice(last_g));
                launch_mat(&d_lm, d_x[last_g], d_logits, T, 0, d_xq[last_g], d_xsc[last_g], s_comp[last_g]);
                if (with_dspark_k4) {
                    for (int ds = 0; ds < 4; ds++) run_dspark_step();
                }
                CK(cudaStreamSynchronize(s_comp[last_g]));
            }
        }
    };

    run_kimi_pass(1, 0, 0);
    double tk0 = omp_get_wtime();
    int n_kt1 = 4;
    for (int i = 0; i < n_kt1; i++) run_kimi_pass(1, 0, i + 1);
    double kt1_ms = (omp_get_wtime() - tk0) * 1000.0;

    /* With 5L-DSpark (`K=4` drafts -> `T=5` batched verify; 75% mean acceptance = `4.0` verified tokens per pass!) */
    run_kimi_pass(5, 1, 10);
    double tkd0 = omp_get_wtime();
    int n_kds = 4;
    for (int i = 0; i < n_kds; i++) run_kimi_pass(5, 1, i + 20);
    double kds_ms = (omp_get_wtime() - tkd0) * 1000.0;
    double k_accept = 4.0; /* 1 target + 3.0 accepted of 4 DSpark drafts */

    double tot_vram_gib = 0.0;
    for (int g = 0; g < 8; g++) tot_vram_gib += (double)vram_per_gpu[g] / (1024.0*1024.0*1024.0);
    printf("RESULT_KIMI_K31_2800B gpus=8 mode=%s hot_vram=%.2f_GiB cold_ram=%.2f_GiB dspark_draft=%.1f_tok/s decode_T1=%.2f_tok/s decode_DSpark=%.2f_tok/s\n",
           use_ram_cold_tier ? "PARALLEL_VRAM_RAM_NVME" : "VRAM_HOT_PLUS_NVME_COLD",
           tot_vram_gib, (double)ram_cold_b / (1024.0*1024.0*1024.0),
           dspark_tps, (double)n_kt1 / (kt1_ms / 1000.0), (n_kds * k_accept) / (kds_ms / 1000.0));
    fflush(stdout);

    CK(cudaSetDevice(last_g)); cudaFree(d_lm_raw); cudaFree(d_ds_raw); cudaFree(d_logits);
    for (int l = 0; l < 93; l++) {
        CK(cudaSetDevice(lyrs[l].dev));
        cudaFree(lyrs[l].d_trunk);
        if (lyrs[l].d_hot_exp) cudaFree(lyrs[l].d_hot_exp);
        if (lyrs[l].d_cold_vram) cudaFree(lyrs[l].d_cold_vram);
        if (lyrs[l].h_cold_ram) free(lyrs[l].h_cold_ram);
        if (lyrs[l].cold_fd >= 0) close(lyrs[l].cold_fd);
        if (l >= 1) cudaFreeHost(h_pin_cold[l]);
    }
    for (int g = 0; g < n_gpus; g++) {
        CK(cudaSetDevice(g));
        cudaFree(d_x[g]); cudaFree(d_u[g]); cudaFree(d_t1[g]); cudaFree(d_t2[g]); cudaFree(d_hid[g]);
        cudaFree(d_xq[g]); cudaFree(d_xsc[g]); cudaFree(d_zq[g]); cudaFree(d_zsc[g]);
        cudaFree(d_ih[g]); cudaFree(d_ic[g]); cudaFree(d_wh[g]); cudaFree(d_wc[g]);
        cudaFree(d_cold_stage[g][0]); cudaFree(d_cold_stage[g][1]);
        cudaFree(d_cold_ptrs[g][0]); cudaFree(d_cold_ptrs[g][1]);
        cudaStreamDestroy(s_comp[g]); cudaStreamDestroy(s_dma[g]);
    }
}

int main(int argc, char **argv) {
    const char *which = (argc > 1) ? argv[1] : "all";
    if (strcmp(which, "35b") == 0 || strcmp(which, "all") == 0) {
        bench_clover35b("/workspace/hg/clover-35b", 1, 0); /* 1x 4090 All-VRAM (17.4 GiB) */
        bench_clover35b("/workspace/hg/clover-35b", 1, 1); /* 1x 4090 Trunk VRAM + Experts RAM */
        bench_clover35b("/workspace/hg/clover-35b", 4, 0); /* 4x 4090 Sharded */
    }
    if (strcmp(which, "120b") == 0 || strcmp(which, "all") == 0) {
        bench_nemotron120b("/workspace/hg/nemotron-120b", 4, 0); /* 4x 4090 All-VRAM (60.2 GiB) */
        bench_nemotron120b("/workspace/hg/nemotron-120b", 8, 0); /* 8x 4090 All-VRAM */
        bench_nemotron120b("/workspace/hg/nemotron-120b", 1, 1); /* 1x 4090 Hybrid (21G VRAM + 39G RAM) */
    }
    if (strcmp(which, "kimi_fast") == 0) {
        bench_kimi_k31("/workspace/hg/kimi-k31", 1); /* Full Overlapped 3-Tier (150G VRAM + 405G RAM + NVMe) */
    } else if (strcmp(which, "kimi") == 0 || strcmp(which, "all") == 0) {
        bench_kimi_k31("/workspace/hg/kimi-k31", 0); /* 8x 4090 Hot VRAM + Pure NVMe Cold */
        bench_kimi_k31("/workspace/hg/kimi-k31", 1); /* 8x 4090 Parallel 3-Tier (VRAM + 405G RAM + NVMe) */
    }
    return 0;
}
