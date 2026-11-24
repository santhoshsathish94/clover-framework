/*
 * Full 24-Layer Zero-PCIe Native CUDA (`sm_89` Ada / RTX 4090) Engine & Benchmark
 * for `LiquidAI/LFM3-Omni-2B` in Canonical `HG4-16R` Format (`853.88 MiB`).
 *
 * Uploads `embed_lm_head.hg4` + `layer_00..23.hg4` directly into GPU VRAM (`854 MiB`),
 * executes `Int32 FWHT-128`, `16-Row` `__dp4a` `LUT16` GEMV/GEMM, `ShortConv-7`,
 * `GQA (16Q/4KV + QKNorm + RoPE + Gate)`, `RMSNorm`, and `128K` ArgMax 100% on GPU
 * with zero per-layer host synchronization, and verifies exact token parity vs C11 CPU.
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HG4_LAYERS 24
#define HG4_D      2048
#define HG4_FFN    7680
#define HG4_VOCAB  128000
#define HG4_SLIDE  256
#define MAX_CTX    4096
#define MAX_PANEL  256

#define CK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while (0)

__constant__ int8_t c_lut16_s8[16];
__constant__ float  c_lut16_f32[16];
__constant__ float  c_inv_freq_full[64];
__constant__ float  c_inv_freq_slide[64];

typedef struct {
    uint32_t R, K;
    const float    *pre_diag;
    const uint16_t *scales_fp16;
    const uint8_t  *tiles;
} d_mat_t;

static void parse_d_mat(const uint8_t *d_blob, const uint8_t *h_blob, d_mat_t *m) {
    memcpy(&m->R, h_blob + 0, 4);
    memcpy(&m->K, h_blob + 4, 4);
    size_t nb = m->K / 128;
    size_t off = 8;
    m->pre_diag = (const float *)(d_blob + off);
    off += (size_t)m->K * 4;
    m->scales_fp16 = (const uint16_t *)(d_blob + off);
    off += ((size_t)m->R * nb) * 2;
    m->tiles = d_blob + off;
}

typedef struct {
    uint8_t *d_raw;
    uint32_t kind;
    const float *op_norm, *post_op_norm, *ffn_norm, *post_ffn_norm;
    const float *conv_w;
    d_mat_t in_proj, out_proj;
    const float *gate_w, *qn_w, *kn_w;
    d_mat_t qkv_proj, o_proj;
    d_mat_t w13_proj, w2_proj;
} d_layer_t;

__device__ __forceinline__ float d_bf16_to_f32(uint16_t b) {
    return __uint_as_float(((uint32_t)b) << 16);
}

__device__ __forceinline__ uint16_t d_f32_to_bf16(float f) {
    uint32_t u = __float_as_uint(f);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

__device__ __forceinline__ float d_fp16_to_f32(uint16_t h) {
    return __half2float(*reinterpret_cast<const __half *>(&h));
}

/* 1-Row Inverse-FWHT-128 Token Embedding Lookup: 1 block per token, 128 threads */
__global__ void k_embed_lookup(
    const int *token_ids, d_mat_t emb, float *out_h, int T
) {
    int t = blockIdx.x;
    int tid = threadIdx.x; /* 0..127 */
    if (t >= T) return;
    int tok = token_ids[t];
    if (tok < 0 || tok >= HG4_VOCAB) tok = 0;
    int rg = tok >> 4, lane = tok & 15;
    const int nb = HG4_D / 128; /* 16 */
    const uint8_t *wt = emb.tiles + (size_t)rg * (size_t)nb * 1024;
    const uint16_t *wsc = emb.scales_fp16 + (size_t)rg * (size_t)nb * 16;

    __shared__ float s_blk[128];
    for (int b = 0; b < nb; b++) {
        float sc = d_fp16_to_f32(wsc[b * 16 + lane]);
        int s = tid >> 3;       /* 0..15 */
        int rem = tid & 7;      /* 0..7 */
        int j = rem & 3;        /* 0..3 */
        uint8_t pk = wt[(size_t)b * 1024 + s * 64 + lane * 4 + j];
        uint8_t nib = (rem < 4) ? (pk & 0x0F) : (pk >> 4);
        s_blk[tid] = c_lut16_f32[nib] * sc;
        __syncthreads();

        for (int h = 1; h < 128; h <<= 1) {
            int grp = tid / h;
            int k = tid & (h - 1);
            int idx0 = grp * (h << 1) + k;
            int idx1 = idx0 + h;
            float a = (tid < 64) ? s_blk[idx0] : 0.0f;
            float bv = (tid < 64) ? s_blk[idx1] : 0.0f;
            __syncthreads();
            if (tid < 64) {
                s_blk[idx0] = a + bv;
                s_blk[idx1] = a - bv;
            }
            __syncthreads();
        }
        out_h[(size_t)t * HG4_D + b * 128 + tid] =
            s_blk[tid] * 0.08838834764831845f * emb.pre_diag[b * 128 + tid];
        __syncthreads();
    }
}

/* Exact RMSNorm: 1 block per row, 256 threads */
__global__ void k_rmsnorm(const float *x, const float *w, float *y, int T, int dim) {
    int t = blockIdx.x;
    if (t >= T) return;
    const float *xr = x + (size_t)t * dim;
    float *yr = y + (size_t)t * dim;
    double ss = 0.0;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        double v = (double)xr[d];
        ss += v * v;
    }
    __shared__ double s_ss[256];
    s_ss[threadIdx.x] = ss;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) s_ss[threadIdx.x] += s_ss[threadIdx.x + stride];
        __syncthreads();
    }
    float inv = rsqrtf((float)(s_ss[0] / (double)dim) + 1e-5f);
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        yr[d] = xr[d] * inv * w[d];
    }
}

__global__ void k_residual_add(const float *a, const float *b, float *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

/* Per-128-Block Bit-Exact Int32 FWHT + Int8 Quantization: Grid(nb, T), 128 threads */
__global__ void k_rotate_quant_128(
    const float *x, const float *pre_diag, int T, int K,
    int8_t *out_xq, float *out_xsc
) {
    int b = blockIdx.x;
    int t = blockIdx.y;
    int tid = threadIdx.x; /* 0..127 */
    int nb = K >> 7;
    if (t >= T || b >= nb) return;

    __shared__ float s_abs[128];
    __shared__ int32_t s_rot[128];
    __shared__ int32_t s_rmax[128];

    float p = x[(size_t)t * K + b * 128 + tid] * pre_diag[b * 128 + tid];
    s_abs[tid] = fabsf(p);
    __syncthreads();
    for (int st = 64; st > 0; st >>= 1) {
        if (tid < st) {
            float o = s_abs[tid + st];
            if (o > s_abs[tid]) s_abs[tid] = o;
        }
        __syncthreads();
    }
    float max_abs = s_abs[0];
    if (max_abs < 1e-12f) {
        if (tid == 0) out_xsc[t * nb + b] = 0.0f;
        out_xq[(size_t)t * K + b * 128 + tid] = 0;
        return;
    }
    float in_sc = max_abs * (1.0f / 2048.0f);
    float v = fminf(2048.0f, fmaxf(-2048.0f, p * (2048.0f / max_abs)));
    s_rot[tid] = (int32_t)rintf(v);
    __syncthreads();

    for (int h = 1; h < 128; h <<= 1) {
        int grp = tid / h;
        int k = tid & (h - 1);
        int i0 = grp * (h << 1) + k;
        int i1 = i0 + h;
        int32_t a = (tid < 64) ? s_rot[i0] : 0;
        int32_t bv = (tid < 64) ? s_rot[i1] : 0;
        __syncthreads();
        if (tid < 64) {
            s_rot[i0] = a + bv;
            s_rot[i1] = a - bv;
        }
        __syncthreads();
    }

    int32_t rv = s_rot[tid];
    s_rmax[tid] = rv < 0 ? -rv : rv;
    __syncthreads();
    for (int st = 64; st > 0; st >>= 1) {
        if (tid < st) {
            int32_t o = s_rmax[tid + st];
            if (o > s_rmax[tid]) s_rmax[tid] = o;
        }
        __syncthreads();
    }
    int32_t max_rot = s_rmax[0];
    if (max_rot == 0) {
        if (tid == 0) out_xsc[t * nb + b] = 0.0f;
        out_xq[(size_t)t * K + b * 128 + tid] = 0;
    } else {
        if (tid == 0) out_xsc[t * nb + b] = (in_sc * (float)max_rot) * (0.08838834764831845f / 127.0f);
        float qf = fminf(127.0f, fmaxf(-127.0f, (float)rv * (127.0f / (float)max_rot)));
        out_xq[(size_t)t * K + b * 128 + tid] = (int8_t)rintf(qf);
    }
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

/*
 * High-Bandwidth Coalesced `HG4-16R` CUDA GEMV/GEMM (`__dp4a`):
 *   Grid = (n_rg, (T + 3) / 4), Block = 128 threads (4 warps split across `nb` K-blocks).
 *   Within each warp (`lane = 0..31`):
 *     `r = lane & 15` (row 0..15 in rg), `sub = lane >> 4` (0 or 1).
 *     Reads `((const uint32_t *)wb)[it * 32 + lane]` -> 128-byte coalesced GDDR6X read!
 */
__global__ void k_hg4_16r_matmul(
    d_mat_t M, const int8_t *X_q, const float *X_sc,
    float *Y, int T, int fused_swiglu
) {
    int rg = blockIdx.x;
    int t_base = blockIdx.y * 4;
    int n_rg = M.R >> 4;
    if (rg >= n_rg || t_base >= T) return;

    int tblk = (t_base + 4 <= T) ? 4 : (T - t_base);
    int warp_id = threadIdx.x >> 5;   /* 0..3 */
    int lane    = threadIdx.x & 31;   /* 0..31 */
    int r       = lane & 15;          /* 0..15 */
    int sub     = lane >> 4;          /* 0..1 */
    int nb      = M.K >> 7;
    int out_R   = fused_swiglu ? (M.R >> 1) : M.R;

    const uint8_t  *wt  = M.tiles + (size_t)rg * (size_t)nb * 1024;
    const uint16_t *wsc = M.scales_fp16 + (size_t)rg * (size_t)nb * 16;
    int my_lut_u8 = (int)(uint8_t)c_lut16_s8[r]; /* Broadcast/loaded once per warp! */

    float vs[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int b = warp_id; b < nb; b += 4) {
        const uint32_t *wb32 = reinterpret_cast<const uint32_t *>(wt + (size_t)b * 1024);
        float ws = d_fp16_to_f32(wsc[b * 16 + r]);
        int acc[4] = {0, 0, 0, 0};

        #pragma unroll
        for (int it = 0; it < 8; it++) {
            uint32_t pk = wb32[it * 32 + lane];
            int wlo = pack_lut16_lo_shfl(pk, my_lut_u8);
            int whi = pack_lut16_hi_shfl(pk, my_lut_u8);
            int s = (it << 1) + sub; /* 0..15 */
            for (int ti = 0; ti < tblk; ti++) {
                const int *xp = reinterpret_cast<const int *>(X_q + (size_t)(t_base + ti) * M.K + b * 128);
                acc[ti] = __dp4a(wlo, xp[2 * s + 0], acc[ti]);
                acc[ti] = __dp4a(whi, xp[2 * s + 1], acc[ti]);
            }
        }
        for (int ti = 0; ti < tblk; ti++) {
            int sum32 = acc[ti] + __shfl_xor_sync(0xffffffffu, acc[ti], 16);
            if (sub == 0) {
                float sc = ws * X_sc[(t_base + ti) * nb + b];
                vs[ti] = __fadd_rn(vs[ti], __fmul_rn((float)sum32, sc));
            }
        }
    }

    __shared__ float smem[4][4][16];
    if (sub == 0) {
        for (int ti = 0; ti < tblk; ti++) smem[warp_id][ti][r] = vs[ti];
    }
    __syncthreads();

    if (warp_id == 0 && sub == 0) {
        for (int ti = 0; ti < tblk; ti++) {
            float total = __fadd_rn(__fadd_rn(smem[0][ti][r], smem[1][ti][r]),
                                    __fadd_rn(smem[2][ti][r], smem[3][ti][r]));
            smem[0][ti][r] = total;
        }
        __syncwarp(0x0000ffffu);
        for (int ti = 0; ti < tblk; ti++) {
            float *yt = Y + (size_t)(t_base + ti) * out_R;
            if (fused_swiglu) {
                if (r < 8) {
                    float g = smem[0][ti][r];
                    float u = smem[0][ti][r + 8];
                    yt[rg * 8 + r] = (g / (1.0f + expf(-g))) * u;
                }
            } else {
                yt[rg * 16 + r] = smem[0][ti][r];
            }
        }
    }
}

/* ShortConv-7 pre-cache & 7-tap causal depthwise conv (reads pos_off from device ptr if non-null) */
__global__ void k_shortconv_pre(const float *bcx, uint16_t *c_cache, int pos_off, const int *d_pos, int T) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T * HG4_D) return;
    int p0 = d_pos ? (*d_pos) : pos_off;
    int t = idx / HG4_D, c = idx % HG4_D;
    const float *row = bcx + (size_t)t * 6144;
    c_cache[(size_t)(p0 + t) * HG4_D + c] = d_f32_to_bf16(row[c] * row[4096 + c]);
}

__global__ void k_shortconv_tap7(
    const float *bcx, const uint16_t *c_cache, const float *conv_w,
    float *out_attn, int pos_off, const int *d_pos, int T
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T * HG4_D) return;
    int p0 = d_pos ? (*d_pos) : pos_off;
    int t = idx / HG4_D, c = idx % HG4_D;
    int p = p0 + t;
    const float *wk = conv_w + c * 7;
    float acc = 0.0f;
    #pragma unroll
    for (int k = 0; k < 7; k++) {
        int ts = p - 6 + k;
        if (ts >= 0) acc += d_bf16_to_f32(c_cache[(size_t)ts * HG4_D + c]) * wk[k];
    }
    out_attn[(size_t)t * HG4_D + c] = bcx[(size_t)t * 6144 + 2048 + c] * acc;
}

/* GQA 16-head sigmoid gate + QK RMSNorm + RoPE + BF16 KV store + Attention */
__global__ void k_gqa_gates(const float *op_in, const float *gate_w, float *gates, int T) {
    int h = blockIdx.x; /* 0..15 */
    int t = blockIdx.y; /* 0..T-1 */
    if (h >= 16 || t >= T) return;
    const float *xr = op_in + (size_t)t * HG4_D;
    const float *gw = gate_w + h * HG4_D;
    float s = 0.0f;
    for (int d = threadIdx.x; d < HG4_D; d += blockDim.x) s += xr[d] * gw[d];
    __shared__ float sm[128];
    sm[threadIdx.x] = s;
    __syncthreads();
    for (int st = 64; st > 0; st >>= 1) {
        if (threadIdx.x < st) sm[threadIdx.x] += sm[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) gates[t * 16 + h] = 1.0f / (1.0f + expf(-sm[0]));
}

__global__ void k_gqa_qknorm_rope_kv(
    float *qkv, const float *qn_w, const float *kn_w,
    uint16_t *k_cache, uint16_t *v_cache, int kind, int pos_off, const int *d_pos, int T
) {
    int h_idx = blockIdx.x; /* 0..19 (0..15 = Q heads, 16..19 = K heads) */
    int t = blockIdx.y;
    int tid = threadIdx.x;  /* 0..127 */
    if (h_idx >= 20 || t >= T) return;
    int p0 = d_pos ? (*d_pos) : pos_off;

    float *row_qkv = qkv + (size_t)t * 3072;
    float *vec = (h_idx < 16) ? (row_qkv + h_idx * 128) : (row_qkv + 2048 + (h_idx - 16) * 128);
    const float *nw = (h_idx < 16) ? qn_w : kn_w;

    __shared__ double s_ss[128];
    __shared__ float s_v[128];
    float val = vec[tid];
    s_ss[tid] = (double)val * (double)val;
    __syncthreads();
    for (int st = 64; st > 0; st >>= 1) {
        if (tid < st) s_ss[tid] += s_ss[tid + st];
        __syncthreads();
    }
    float inv = rsqrtf((float)(s_ss[0] / 128.0) + 1e-5f);
    s_v[tid] = val * inv * nw[tid];
    __syncthreads();

    int pos = p0 + t;
    const float *inv_f = (kind == 1) ? c_inv_freq_slide : c_inv_freq_full;
    if (tid < 64) {
        float ang = (float)pos * inv_f[tid];
        float cs = cosf(ang), sn = sinf(ang);
        float x1 = s_v[tid], x2 = s_v[tid + 64];
        vec[tid]      = x1 * cs - x2 * sn;
        vec[tid + 64] = x2 * cs + x1 * sn;
    }
    __syncthreads();
    if (h_idx >= 16) {
        int kh = h_idx - 16;
        k_cache[(size_t)pos * 512 + kh * 128 + tid] = d_f32_to_bf16(vec[tid]);
        v_cache[(size_t)pos * 512 + kh * 128 + tid] = d_f32_to_bf16(row_qkv[2560 + kh * 128 + tid]);
    }
}

__global__ void k_gqa_attn(
    const float *qkv, const uint16_t *k_cache, const uint16_t *v_cache,
    const float *gates, float *out_attn, int kind, int pos_off, const int *d_pos, int T
) {
    int h = blockIdx.x; /* 0..15 */
    int t = blockIdx.y; /* 0..T-1 */
    int d = threadIdx.x;/* 0..127 */
    if (h >= 16 || t >= T) return;
    int p0 = d_pos ? (*d_pos) : pos_off;

    int kv_h = h >> 2;
    int pos = p0 + t;
    int j0 = (kind == 1 && pos - HG4_SLIDE + 1 > 0) ? (pos - HG4_SLIDE + 1) : 0;
    float qd = qkv[(size_t)t * 3072 + h * 128 + d];
    const float scale = 0.08838834764831845f; /* 1 / sqrt(128) */

    __shared__ float s_dot[128];
    float m_max = -1e30f, l_sum = 0.0f, acc_o = 0.0f;

    for (int j = j0; j <= pos; j++) {
        float kd = d_bf16_to_f32(k_cache[(size_t)j * 512 + kv_h * 128 + d]);
        s_dot[d] = qd * kd;
        __syncthreads();
        for (int st = 64; st > 0; st >>= 1) {
            if (d < st) s_dot[d] += s_dot[d + st];
            __syncthreads();
        }
        float score = s_dot[0] * scale;
        float vd = d_bf16_to_f32(v_cache[(size_t)j * 512 + kv_h * 128 + d]);
        if (score > m_max) {
            float alpha = expf(m_max - score);
            l_sum = l_sum * alpha + 1.0f;
            acc_o = acc_o * alpha + vd;
            m_max = score;
        } else {
            float w = expf(score - m_max);
            l_sum += w;
            acc_o += w * vd;
        }
        __syncthreads();
    }
    out_attn[(size_t)t * HG4_D + h * 128 + d] = acc_o * (gates[t * 16 + h] / (l_sum + 1e-12f));
}

__global__ void k_argmax_128k(const float *logits, int *out_id) {
    __shared__ float s_val[256];
    __shared__ int   s_idx[256];
    int tid = threadIdx.x;
    float best_v = -1e30f;
    int best_i = 0;
    for (int i = tid; i < HG4_VOCAB; i += 256) {
        float v = logits[i];
        if (v > best_v) { best_v = v; best_i = i; }
    }
    s_val[tid] = best_v;
    s_idx[tid] = best_i;
    __syncthreads();
    for (int st = 128; st > 0; st >>= 1) {
        if (tid < st) {
            if (s_val[tid + st] > s_val[tid] ||
                (s_val[tid + st] == s_val[tid] && s_idx[tid + st] < s_idx[tid])) {
                s_val[tid] = s_val[tid + st];
                s_idx[tid] = s_idx[tid + st];
            }
        }
        __syncthreads();
    }
    if (tid == 0) *out_id = s_idx[0];
}

static uint8_t *upload_file_to_gpu(const char *path, uint8_t **out_host, size_t *out_sz) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    size_t sz = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *h = (uint8_t *)malloc(sz);
    if (fread(h, 1, sz, f) != sz) { fprintf(stderr, "Short read %s\n", path); exit(1); }
    fclose(f);
    uint8_t *d = NULL;
    CK(cudaMalloc(&d, sz));
    CK(cudaMemcpy(d, h, sz, cudaMemcpyHostToDevice));
    *out_host = h;
    *out_sz = sz;
    return d;
}

__global__ void k_inc_pos(int *d_pos) {
    (*d_pos)++;
}

static void launch_hg4_mm_s(
    d_mat_t M, const float *d_X, float *d_Y, int T, int fused_swiglu,
    int8_t *d_xq, float *d_xsc, cudaStream_t st
) {
    dim3 rq_grid(M.K >> 7, T);
    k_rotate_quant_128<<<rq_grid, 128, 0, st>>>(d_X, M.pre_diag, T, (int)M.K, d_xq, d_xsc);
    dim3 mm_grid(M.R >> 4, (T + 3) / 4);
    k_hg4_16r_matmul<<<mm_grid, 128, 0, st>>>(M, d_xq, d_xsc, d_Y, T, fused_swiglu);
}

extern "C" int lfm3_hg4_init(const char *dir);
extern "C" int lfm3_hg4_generate(
    const int *prompt_ids, int prompt_len, int max_new_tokens, int eos_id,
    int *out_ids, float *out_first_logits, double *out_prefill_ms, double *out_decode_ms
);

int main(int argc, char **argv) {
    const char *dir = (argc > 1) ? argv[1] : "/workspace/hg/omni-2b";
    char path[512];
    snprintf(path, sizeof(path), "%s/meta.bin", dir);
    FILE *mf = fopen(path, "rb");
    if (!mf) { fprintf(stderr, "Missing %s\n", path); return 1; }
    char magic[4]; uint32_t hdr[4];
    uint8_t lut16_u8[16]; int8_t lut16_s8[16]; float lut16_f32[16];
    float final_norm[HG4_D], inv_full[64], inv_slide[64]; uint32_t kinds[HG4_LAYERS];
    if (fread(magic, 1, 4, mf) != 4 || fread(hdr, 4, 4, mf) != 4 ||
        fread(lut16_u8, 1, 16, mf) != 16 || fread(final_norm, 4, HG4_D, mf) != HG4_D ||
        fread(inv_full, 4, 64, mf) != 64 || fread(inv_slide, 4, 64, mf) != 64 ||
        fread(kinds, 4, HG4_LAYERS, mf) != HG4_LAYERS) return 2;
    fclose(mf);
    for (int i = 0; i < 16; i++) {
        lut16_s8[i] = (int8_t)((int)lut16_u8[i] - 128);
        lut16_f32[i] = (float)lut16_s8[i];
    }
    CK(cudaMemcpyToSymbol(c_lut16_s8, lut16_s8, 16));
    CK(cudaMemcpyToSymbol(c_lut16_f32, lut16_f32, 64));
    CK(cudaMemcpyToSymbol(c_inv_freq_full, inv_full, 256));
    CK(cudaMemcpyToSymbol(c_inv_freq_slide, inv_slide, 256));

    float *d_final_norm = NULL;
    CK(cudaMalloc(&d_final_norm, HG4_D * 4));
    CK(cudaMemcpy(d_final_norm, final_norm, HG4_D * 4, cudaMemcpyHostToDevice));

    uint8_t *h_emb = NULL; size_t sz_emb = 0;
    snprintf(path, sizeof(path), "%s/embed_lm_head.hg4", dir);
    uint8_t *d_emb_raw = upload_file_to_gpu(path, &h_emb, &sz_emb);
    d_mat_t d_emb;
    parse_d_mat(d_emb_raw, h_emb, &d_emb);
    free(h_emb);
    size_t total_vram_b = sz_emb;

    d_layer_t lyrs[HG4_LAYERS];
    uint16_t *d_kcache[HG4_LAYERS], *d_vcache[HG4_LAYERS], *d_ccache[HG4_LAYERS];
    for (int l = 0; l < HG4_LAYERS; l++) {
        snprintf(path, sizeof(path), "%s/layer_%02d.hg4", dir, l);
        uint8_t *h_raw = NULL; size_t sz = 0;
        uint8_t *d_raw = upload_file_to_gpu(path, &h_raw, &sz);
        total_vram_b += sz;
        d_layer_t *L = &lyrs[l];
        memset(L, 0, sizeof(*L));
        L->d_raw = d_raw;
        size_t off = 0;
        memcpy(&L->kind, h_raw + off, 4); off += 4;
        L->op_norm       = (const float *)(d_raw + off); off += HG4_D * 4;
        L->post_op_norm  = (const float *)(d_raw + off); off += HG4_D * 4;
        L->ffn_norm      = (const float *)(d_raw + off); off += HG4_D * 4;
        L->post_ffn_norm = (const float *)(d_raw + off); off += HG4_D * 4;
        if (L->kind == 0) {
            L->conv_w = (const float *)(d_raw + off); off += HG4_D * 7 * 4;
            uint32_t li, lo;
            memcpy(&li, h_raw + off, 4); off += 4;
            memcpy(&lo, h_raw + off, 4); off += 4;
            parse_d_mat(d_raw + off, h_raw + off, &L->in_proj);  off += li;
            parse_d_mat(d_raw + off, h_raw + off, &L->out_proj); off += lo;
            CK(cudaMalloc(&d_ccache[l], (size_t)MAX_CTX * HG4_D * 2));
            d_kcache[l] = d_vcache[l] = NULL;
        } else {
            L->gate_w = (const float *)(d_raw + off); off += 16 * HG4_D * 4;
            L->qn_w   = (const float *)(d_raw + off); off += 128 * 4;
            L->kn_w   = (const float *)(d_raw + off); off += 128 * 4;
            uint32_t lq, lo;
            memcpy(&lq, h_raw + off, 4); off += 4;
            memcpy(&lo, h_raw + off, 4); off += 4;
            parse_d_mat(d_raw + off, h_raw + off, &L->qkv_proj); off += lq;
            parse_d_mat(d_raw + off, h_raw + off, &L->o_proj);   off += lo;
            CK(cudaMalloc(&d_kcache[l], (size_t)MAX_CTX * 512 * 2));
            CK(cudaMalloc(&d_vcache[l], (size_t)MAX_CTX * 512 * 2));
            d_ccache[l] = NULL;
        }
        uint32_t lw13, lw2;
        memcpy(&lw13, h_raw + off, 4); off += 4;
        memcpy(&lw2,  h_raw + off, 4); off += 4;
        parse_d_mat(d_raw + off, h_raw + off, &L->w13_proj); off += lw13;
        parse_d_mat(d_raw + off, h_raw + off, &L->w2_proj);  off += lw2;
        free(h_raw);
    }

    float *d_a, *d_b, *d_op_in, *d_op_out, *d_norm, *d_mid, *d_bcx, *d_qkv, *d_attn, *d_ffn, *d_gates, *d_logits, *d_xsc;
    int8_t *d_xq;
    int *d_ids, *d_next, *d_pos;
    CK(cudaMalloc(&d_a,      MAX_PANEL * HG4_D * 4));
    CK(cudaMalloc(&d_b,      MAX_PANEL * HG4_D * 4));
    CK(cudaMalloc(&d_op_in,  MAX_PANEL * HG4_D * 4));
    CK(cudaMalloc(&d_op_out, MAX_PANEL * HG4_D * 4));
    CK(cudaMalloc(&d_norm,   MAX_PANEL * HG4_D * 4));
    CK(cudaMalloc(&d_mid,    MAX_PANEL * HG4_D * 4));
    CK(cudaMalloc(&d_bcx,    MAX_PANEL * 6144 * 4));
    CK(cudaMalloc(&d_qkv,    MAX_PANEL * 3072 * 4));
    CK(cudaMalloc(&d_attn,   MAX_PANEL * HG4_D * 4));
    CK(cudaMalloc(&d_ffn,    MAX_PANEL * HG4_FFN * 4));
    CK(cudaMalloc(&d_gates,  MAX_PANEL * 16 * 4));
    CK(cudaMalloc(&d_logits, HG4_VOCAB * 4));
    CK(cudaMalloc(&d_xq,     MAX_PANEL * HG4_FFN));
    CK(cudaMalloc(&d_xsc,    MAX_PANEL * (HG4_FFN / 128) * 4));
    CK(cudaMalloc(&d_ids,    MAX_PANEL * 4));
    CK(cudaMalloc(&d_next,   4));
    CK(cudaMalloc(&d_pos,    4));

    cudaStream_t st;
    CK(cudaStreamCreate(&st));

    auto enqueue_step = [&](const int *d_tok_in, int T, int pos_off, const int *d_pos_ptr) {
        k_embed_lookup<<<T, 128, 0, st>>>(d_tok_in, d_emb, d_a, T);
        float *cur = d_a, *nxt = d_b;
        for (int l = 0; l < HG4_LAYERS; l++) {
            const d_layer_t *L = &lyrs[l];
            k_rmsnorm<<<T, 256, 0, st>>>(cur, L->op_norm, d_op_in, T, HG4_D);
            if (L->kind == 0) {
                launch_hg4_mm_s(L->in_proj, d_op_in, d_bcx, T, 0, d_xq, d_xsc, st);
                int tot = T * HG4_D;
                k_shortconv_pre<<<(tot + 255) / 256, 256, 0, st>>>(d_bcx, d_ccache[l], pos_off, d_pos_ptr, T);
                k_shortconv_tap7<<<(tot + 255) / 256, 256, 0, st>>>(d_bcx, d_ccache[l], L->conv_w, d_attn, pos_off, d_pos_ptr, T);
                launch_hg4_mm_s(L->out_proj, d_attn, d_op_out, T, 0, d_xq, d_xsc, st);
            } else {
                launch_hg4_mm_s(L->qkv_proj, d_op_in, d_qkv, T, 0, d_xq, d_xsc, st);
                k_gqa_gates<<<dim3(16, T), 128, 0, st>>>(d_op_in, L->gate_w, d_gates, T);
                k_gqa_qknorm_rope_kv<<<dim3(20, T), 128, 0, st>>>(d_qkv, L->qn_w, L->kn_w, d_kcache[l], d_vcache[l], L->kind, pos_off, d_pos_ptr, T);
                k_gqa_attn<<<dim3(16, T), 128, 0, st>>>(d_qkv, d_kcache[l], d_vcache[l], d_gates, d_attn, L->kind, pos_off, d_pos_ptr, T);
                launch_hg4_mm_s(L->o_proj, d_attn, d_op_out, T, 0, d_xq, d_xsc, st);
            }
            k_rmsnorm<<<T, 256, 0, st>>>(d_op_out, L->post_op_norm, d_norm, T, HG4_D);
            int tot = T * HG4_D;
            k_residual_add<<<(tot + 255) / 256, 256, 0, st>>>(cur, d_norm, d_mid, tot);
            k_rmsnorm<<<T, 256, 0, st>>>(d_mid, L->ffn_norm, d_op_in, T, HG4_D);
            launch_hg4_mm_s(L->w13_proj, d_op_in, d_ffn, T, 1, d_xq, d_xsc, st);
            launch_hg4_mm_s(L->w2_proj,  d_ffn, d_op_out, T, 0, d_xq, d_xsc, st);
            k_rmsnorm<<<T, 256, 0, st>>>(d_op_out, L->post_ffn_norm, d_norm, T, HG4_D);
            k_residual_add<<<(tot + 255) / 256, 256, 0, st>>>(d_mid, d_norm, nxt, tot);
            float *tmp = cur; cur = nxt; nxt = tmp;
        }
        k_rmsnorm<<<1, 256, 0, st>>>(cur + (size_t)(T - 1) * HG4_D, d_final_norm, d_op_in, 1, HG4_D);
        launch_hg4_mm_s(d_emb, d_op_in, d_logits, 1, 0, d_xq, d_xsc, st);
        k_argmax_128k<<<1, 256, 0, st>>>(d_logits, d_next);
        if (d_pos_ptr) k_inc_pos<<<1, 1, 0, st>>>(d_pos);
    };

    auto run_prefill = [&](const int *h_ids, int T, int pos_off) -> int {
        CK(cudaMemcpyAsync(d_ids, h_ids, T * 4, cudaMemcpyHostToDevice, st));
        enqueue_step(d_ids, T, pos_off, NULL);
        int next_id = 0;
        CK(cudaMemcpyAsync(&next_id, d_next, 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        return next_id;
    };

    /* Capture entire 24-layer + 128K LM-head T=1 decode step into ONE CUDA Graph! */
    int init_pos = 44;
    CK(cudaMemcpy(d_pos, &init_pos, 4, cudaMemcpyHostToDevice));
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
    enqueue_step(d_next, 1, 0, d_pos);
    CK(cudaStreamEndCapture(st, &graph));
    CK(cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0));

    static const int prompt_ids[44] = {
        124894,124899,23630,207,3582,437,267,11037,16200,22,124900,207,
        124899,5922,207,22497,267,55911,21516,1446,355,91520,7946,34,
        1006,17,9495,9993,309,4468,612,91422,1184,229,654,309,
        229,1163,22,124900,207,124899,63514,207
    };

    for (int l = 0; l < HG4_LAYERS; l++) if (d_ccache[l]) CK(cudaMemsetAsync(d_ccache[l], 0, MAX_CTX * HG4_D * 2, st));
    run_prefill(prompt_ids, 44, 0);

    cudaEvent_t ev0, ev1, ev2;
    CK(cudaEventCreate(&ev0)); CK(cudaEventCreate(&ev1)); CK(cudaEventCreate(&ev2));

    int cuda_gen[128];
    int n_gen = 0;
    float pf_ms = 0.0f, dc_ms = 0.0f;

    for (int rep = 0; rep < 3; rep++) {
        for (int l = 0; l < HG4_LAYERS; l++) if (d_ccache[l]) CK(cudaMemsetAsync(d_ccache[l], 0, 256 * HG4_D * 2, st));
        CK(cudaEventRecord(ev0, st));
        CK(cudaMemcpyAsync(d_ids, prompt_ids, 44 * 4, cudaMemcpyHostToDevice, st));
        enqueue_step(d_ids, 44, 0, NULL);
        CK(cudaEventRecord(ev1, st));

        int p44 = 44;
        CK(cudaMemcpyAsync(d_pos, &p44, 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(&cuda_gen[0], d_next, 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        n_gen = 1;
        while (n_gen < 110 && cuda_gen[n_gen - 1] != 124900) {
            CK(cudaGraphLaunch(graph_exec, st));
            CK(cudaMemcpyAsync(&cuda_gen[n_gen], d_next, 4, cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            n_gen++;
        }
        CK(cudaEventRecord(ev2, st));
        CK(cudaEventSynchronize(ev2));
        CK(cudaEventElapsedTime(&pf_ms, ev0, ev1));
        CK(cudaEventElapsedTime(&dc_ms, ev1, ev2));
        printf("RTX4090_OMNI_HG4_GRAPH rep=%d vram=%.1f MiB gen=%d prefill=%.1f tok/s (%.2f ms) decode=%.1f tok/s (%.2f ms)\n",
               rep + 1, (double)total_vram_b / (1024.0 * 1024.0), n_gen,
               44.0f / (pf_ms / 1000.0f), pf_ms,
               (float)(n_gen - 1) / (dc_ms / 1000.0f), dc_ms);
    }

    int p256[256];
    for (int i = 0; i < 256; i++) p256[i] = prompt_ids[i % 44];
    run_prefill(p256, 256, 0);
    CK(cudaEventRecord(ev0, st));
    CK(cudaMemcpyAsync(d_ids, p256, 256 * 4, cudaMemcpyHostToDevice, st));
    enqueue_step(d_ids, 256, 0, NULL);
    CK(cudaEventRecord(ev1, st));
    CK(cudaEventSynchronize(ev1));
    float pf256_ms = 0.0f;
    CK(cudaEventElapsedTime(&pf256_ms, ev0, ev1));
    printf("RTX4090_OMNI_HG4_PP256: %.1f tok/s (%.2f ms)\n", 256.0f / (pf256_ms / 1000.0f), pf256_ms);

    /* Cross-check token-for-token parity against C11 CPU engine! */
    if (lfm3_hg4_init(dir) == 0) {
        int cpu_gen[128];
        double cpf = 0, cdc = 0;
        int cn = lfm3_hg4_generate(prompt_ids, 44, 110, 124900, cpu_gen, NULL, &cpf, &cdc);
        int match = (cn == n_gen);
        int same_cnt = 0;
        int min_n = cn < n_gen ? cn : n_gen;
        for (int i = 0; i < min_n; i++) {
            if (cpu_gen[i] == cuda_gen[i]) same_cnt++;
            else match = 0;
        }
        printf("PARITY_VS_C11_CPU: exact_all=%d matching_tokens=%d/%d\n", match, same_cnt, n_gen);
    }
    return 0;
}
