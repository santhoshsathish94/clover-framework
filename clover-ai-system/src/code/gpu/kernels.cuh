// Kernels and helpers shared by the GPU layer programs (layer0.cu, layer1.cu). See layer0.cu for the arithmetic they follow.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <ctime>
#include <vector>
#include <cuda_runtime.h>
#if defined(INT8_PLANES) || defined(FAST_ROUTER)
#include <cublas_v2.h>
#define CUB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "%s:%d: cublas error %d\n", __FILE__, __LINE__, (int)s_); exit(1); } } while (0)
#endif

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

enum { W = 7168, ROWS = 12288, HEAD = 128, HEADS = 96, DENSE = 33792, QKV_PREFIX = 64 + 24 + W * 4 + 1024, QKV_COMP = W + 20 };

// ---------------------------------------------------------------- host: reading the trunk files
static std::vector<unsigned char> slurp(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<unsigned char> v((size_t)n);
    if (fread(v.data(), 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "cannot read %s\n", path); exit(1); }
    fclose(f);
    return v;
}
static uint32_t u32(const unsigned char *p) { return p[0] | (p[1] << 8) | (p[2] << 16) | ((uint32_t)p[3] << 24); }
static uint64_t u64(const unsigned char *p) { return u32(p) | ((uint64_t)u32(p + 4) << 32); }

struct Record { uint32_t id, group, kind, rows, cols; uint64_t offset, length; };

// ---------------------------------------------------------------- device helpers
// exp and tanh through double, rounded once to float: the CPU's libm gives the correctly rounded float result in practice, CUDA's own expf and tanhf are 2 ulp
__device__ __forceinline__ float expf_cr_ref(float v) { return (float)exp((double)v); }
__device__ __forceinline__ float tanhf_cr_ref(float v) { return (float)tanh((double)v); }

// expf_cr and tanhf_cr below return exactly the bits of the two reference functions above (checked for all 2^32 float inputs by testmath.cu), but compute on the
// float pipe, which is 64 times faster than the double pipe of this card: the value is evaluated as a float-float pair (about 2^-38 relative error), and if that pair
// lies too close to a rounding boundary of the float result to be sure of the reference's rounding, the reference function itself is called.
__device__ __forceinline__ void fm_two_sum(float a, float b, float &s, float &e)
{ s = __fadd_rn(a, b); const float bb = __fsub_rn(s, a); e = __fadd_rn(__fsub_rn(a, __fsub_rn(s, bb)), __fsub_rn(b, bb)); }
#define FM_INV_L 0x1.7154760000000p+5f   /* 32 / ln 2 */
#define FM_LH 0x1.62e4300000000p-6f      /* ln 2 / 32 as three floats */
#define FM_LL -0x1.05c6100000000p-34f
#define FM_LL2 -0x1.950d880000000p-59f
#define FM_THIRD_H 0x1.5555560000000p-2f
#define FM_THIRD_L -0x1.5555560000000p-27f
#define FM_TC2 0x1.1111120000000p-3f
#define FM_TC3 -0x1.ba1ba20000000p-5f
#define FM_TC4 0x1.664f480000000p-6f
#define FM_TC5 -0x1.226e360000000p-7f
#define FM_TC6 0x1.d6d3d00000000p-9f
#define FM_EC3 0x1.5555560000000p-3f
#define FM_EC4 0x1.5555560000000p-5f
#define FM_EC5 0x1.1111120000000p-7f
#define FM_EC6 0x1.6c16c20000000p-10f
#ifndef FM_TOL
#define FM_TOL 0x1p-9f                   /* a pair closer than this fraction of half a float step to a rounding boundary is left to the reference */
#endif
// 2^(j/32) as (high, low) floats, j = 0..31
__device__ const float fm_tab[64] = {
    0x1.0000000000000p+0f, 0x0.0p+0f, 0x1.059b0e0000000p+0f, -0x1.9d4f520000000p-25f, 0x1.0b55860000000p+0f, 0x1.9f31220000000p-25f,
    0x1.11301e0000000p+0f, -0x1.fdb4960000000p-25f, 0x1.172b840000000p+0f, -0x1.c157420000000p-27f, 0x1.1d48740000000p+0f, -0x1.d2e8ca0000000p-25f,
    0x1.2387a60000000p+0f, 0x1.ceac480000000p-25f, 0x1.29e9e00000000p+0f, -0x1.5c04240000000p-25f, 0x1.306fe00000000p+0f, 0x1.4636e20000000p-25f,
    0x1.371a740000000p+0f, -0x1.18aac60000000p-25f, 0x1.3dea640000000p+0f, 0x1.8246840000000p-25f, 0x1.44e0860000000p+0f, 0x1.8624b40000000p-30f,
    0x1.4bfdae0000000p+0f, -0x1.593abc0000000p-25f, 0x1.5342b60000000p+0f, -0x1.2c56100000000p-25f, 0x1.5ab07e0000000p+0f, -0x1.5bd5ec0000000p-27f,
    0x1.6247ec0000000p+0f, -0x1.f8b5500000000p-25f, 0x1.6a09e60000000p+0f, 0x1.9fcef40000000p-26f, 0x1.71f75e0000000p+0f, 0x1.1d8bee0000000p-25f,
    0x1.7a11480000000p+0f, -0x1.829fd00000000p-25f, 0x1.82589a0000000p+0f, -0x1.accc7c0000000p-26f, 0x1.8ace540000000p+0f, 0x1.15506e0000000p-27f,
    0x1.93737c0000000p+0f, -0x1.e647440000000p-25f, 0x1.9c49180000000p+0f, 0x1.51f8480000000p-27f, 0x1.a5503c0000000p+0f, -0x1.b83b540000000p-25f,
    0x1.ae89fa0000000p+0f, -0x1.a94b140000000p-26f, 0x1.b7f7700000000p+0f, -0x1.a094380000000p-25f, 0x1.c199be0000000p+0f, -0x1.3d56b20000000p-27f,
    0x1.cb720e0000000p+0f, -0x1.8837cc0000000p-27f, 0x1.d5818e0000000p+0f, -0x1.822dbc0000000p-27f, 0x1.dfc9740000000p+0f, -0x1.908c940000000p-25f,
    0x1.ea4afa0000000p+0f, 0x1.52486c0000000p-27f, 0x1.f507660000000p+0f, -0x1.246eb00000000p-26f,
};

// exp(x) = (s + lo) * 2^k for -70 < x < 88: x = n ln2/32 + r, 2^(n/32) from the table, exp(r) - 1 by a short series
__device__ __forceinline__ void fm_exp_df(float x, float &s, float &lo, int &k)
{
    const float nf = rintf(__fmul_rn(x, FM_INV_L));
    const int n = (int)nf;
    const float p = __fmul_rn(nf, FM_LH), pe = __fmaf_rn(nf, FM_LH, -p);
    const float q = __fmul_rn(nf, FM_LL), qe = __fmaf_rn(nf, FM_LL, -q);
    float ra, rae, s1, e1, rh, rl;
    fm_two_sum(x, -p, ra, rae);
    fm_two_sum(ra, -q, s1, e1);
    fm_two_sum(s1, __fsub_rn(__fadd_rn(__fadd_rn(rae, e1), __fadd_rn(-pe, -qe)), __fmul_rn(nf, FM_LL2)), rh, rl);   // r = rh + rl, |r| <= ln2/64
    const float sqh = __fmul_rn(rh, rh), sql = __fmaf_rn(rh, rh, -sqh);
    float t = __fmaf_rn(rh, FM_EC6, FM_EC5);
    t = __fmaf_rn(rh, t, FM_EC4); t = __fmaf_rn(rh, t, FM_EC3);
    const float c3 = __fmul_rn(sqh, __fmul_rn(rh, t));                               // r^3/6 + r^4/24 + r^5/120 + r^6/720
    float a, ea, ph, pl;
    fm_two_sum(rh, __fmul_rn(sqh, 0.5f), a, ea);
    fm_two_sum(a, __fadd_rn(ea, __fadd_rn(__fmul_rn(sql, 0.5f), __fadd_rn(c3, __fmul_rn(rl, __fadd_rn(1.0f, rh))))), ph, pl);   // exp(r) - 1 = ph + pl
    const int j = n & 31;
    k = n >> 5;
    const float th = fm_tab[2 * j], tl = fm_tab[2 * j + 1];
    const float mh = __fmul_rn(th, ph), ml = __fmaf_rn(th, ph, -mh);
    float s0, es;
    fm_two_sum(th, mh, s0, es);
    s = s0;
    lo = __fadd_rn(__fadd_rn(es, tl), __fadd_rn(ml, __fadd_rn(__fmul_rn(tl, ph), __fmul_rn(th, pl))));
}

// the float nearest s + lo, unless s + lo is within FM_TOL of half a float step of a rounding boundary (or the float is a power of two)
__device__ __forceinline__ bool fm_round(float s, float lo, float &c)
{
    c = __fadd_rn(s, lo);
    const float rem = __fadd_rn(__fsub_rn(s, c), lo);
    const int bits = __float_as_int(c);
    const float half = __int_as_float((((bits >> 23) & 255) - 24) << 23);
    return (bits & 0x7fffff) != 0 && fabsf(rem) < half * (1.0f - FM_TOL);
}

__device__ __forceinline__ bool fm_expf_try(float x, float &out)
{
    if (x > -70.0f && x < 88.0f) {
        float s, lo, c; int k;
        fm_exp_df(x, s, lo, k);
        if (fm_round(s, lo, c)) { out = __int_as_float(__float_as_int(c) + k * 8388608); return true; }
    }
    return false;
}
__device__ __forceinline__ float expf_cr(float x) { float r; return fm_expf_try(x, r) ? r : expf_cr_ref(x); }

__device__ __forceinline__ bool fm_tanhf_try(float z, float &out)
{
    const float a = fabsf(z);
    float s, lo, c;
    if (a < 0x1p-13f) { out = z; return true; }   // tanh z = z (1 - z^2/3 ..) with z^2/3 below a quarter of a float step
    if (a < 0.125f) {                           // tanh z = z (1 + w), w = -z^2/3 + c2 z^4 + ... + c6 z^12
        const float z2h = __fmul_rn(a, a), z2l = __fmaf_rn(a, a, -z2h);
        float t = __fmaf_rn(z2h, FM_TC6, FM_TC5);
        t = __fmaf_rn(z2h, t, FM_TC4); t = __fmaf_rn(z2h, t, FM_TC3); t = __fmaf_rn(z2h, t, FM_TC2);
        const float tail = __fmul_rn(__fmul_rn(z2h, z2h), t);
        const float mh = __fmul_rn(z2h, FM_THIRD_H), ml = __fadd_rn(__fmaf_rn(z2h, FM_THIRD_H, -mh), __fadd_rn(__fmul_rn(z2h, FM_THIRD_L), __fmul_rn(z2l, FM_THIRD_H)));
        const float wh = -mh, wl = __fsub_rn(tail, ml);
        const float ph = __fmul_rn(a, wh), pl = __fadd_rn(__fmaf_rn(a, wh, -ph), __fmul_rn(a, wl));
        float es; fm_two_sum(a, ph, s, es);
        lo = __fadd_rn(es, pl);
        if (fm_round(s, lo, c)) { out = copysignf(c, z); return true; }
    } else if (a < 20.0f) {                     // tanh a = (E - 1) / (E + 1), E = exp(2a)
        int k; fm_exp_df(__fadd_rn(a, a), s, lo, k);
        const float sc = __int_as_float((127 + k) << 23), eh = __fmul_rn(s, sc), el = __fmul_rn(lo, sc);
        float nh, ne, dh, de;
        fm_two_sum(eh, -1.0f, nh, ne); fm_two_sum(eh, 1.0f, dh, de);
        const float nl = __fadd_rn(ne, el), dl = __fadd_rn(de, el);
        const float q1 = __fdiv_rn(nh, dh);
        const float rr = __fsub_rn(__fadd_rn(__fmaf_rn(-q1, dh, nh), nl), __fmul_rn(q1, dl));
        if (fm_round(q1, __fdiv_rn(rr, dh), c)) { out = copysignf(c, z); return true; }
    } else if (a <= 3.0e38f) { out = copysignf(1.0f, z); return true; }
    return false;
}
__device__ __forceinline__ float tanhf_cr(float z) { float r; return fm_tanhf_try(z, r) ? r : tanhf_cr_ref(z); }
__device__ __forceinline__ float sigmoidf_(float v) { return 1.0f / (1.0f + expf_cr(-v)); }
__device__ __forceinline__ float f32at(const unsigned char *p) { return *(const float *)p; }

__device__ double block_sum(double v)
{
    __shared__ double warp[32];
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) warp[threadIdx.x >> 5] = v;
    __syncthreads();
    double t = threadIdx.x < (blockDim.x + 31) / 32 ? warp[threadIdx.x] : 0.0;
    if (threadIdx.x < 32) for (int o = 16; o > 0; o >>= 1) t += __shfl_down_sync(0xffffffffu, t, o);
    __syncthreads();
    return t;   // valid in thread 0 only
}

// out[row] = (gain * in[row]) * inverse_rms(in row); one block per row; in place is fine
__global__ void normalize_rows(float *out, const float *in, const unsigned char *gains, int width)
{
    const float *x = in + (size_t)blockIdx.x * width;
    float *y = out + (size_t)blockIdx.x * width;
    double s = 0.0;
    for (int c = threadIdx.x; c < width; c += blockDim.x) s += (double)x[c] * (double)x[c];
    s = block_sum(s);
    __shared__ float inverse;
    if (threadIdx.x == 0) inverse = (float)(1.0 / sqrt(s / (double)width + (double)1e-5f));
    __syncthreads();
    for (int c = threadIdx.x; c < width; c += blockDim.x) y[c] = (f32at(gains + (size_t)c * 4) * x[c]) * inverse;
}

// C[n * ldc + r] = scale[r] * sum_k A[n * K + k] * (signed code)[r * code_stride + k]
#define BM 128
#define BN 128
#define BK 8
__global__ void __launch_bounds__(256) gemm_i8(const float *__restrict__ A, int N, int K, const unsigned char *__restrict__ codes, size_t code_stride,
    const unsigned char *__restrict__ scales, size_t scale_stride, int R, float *__restrict__ C, int ldc)
{
    __shared__ float As[BK][BM + 4], Bs[BK][BN + 4];
    const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
    const int n0 = blockIdx.y * BM, r0 = blockIdx.x * BN;
    const int lrow = threadIdx.x >> 1, lk = (threadIdx.x & 1) * 4;
    float acc[8][8];
    #pragma unroll
    for (int i = 0; i < 8; i++) for (int j = 0; j < 8; j++) acc[i][j] = 0.0f;
    for (int k0 = 0; k0 < K; k0 += BK) {
        float4 av = make_float4(0.f, 0.f, 0.f, 0.f);
        if (n0 + lrow < N) av = *(const float4 *)(A + (size_t)(n0 + lrow) * K + k0 + lk);
        As[lk + 0][lrow] = av.x; As[lk + 1][lrow] = av.y; As[lk + 2][lrow] = av.z; As[lk + 3][lrow] = av.w;
        uint32_t w = 0;
        if (r0 + lrow < R) w = *(const uint32_t *)(codes + (size_t)(r0 + lrow) * code_stride + k0 + lk);
        Bs[lk + 0][lrow] = (float)(int8_t)(w & 0xff); Bs[lk + 1][lrow] = (float)(int8_t)((w >> 8) & 0xff);
        Bs[lk + 2][lrow] = (float)(int8_t)((w >> 16) & 0xff); Bs[lk + 3][lrow] = (float)(int8_t)(w >> 24);
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            float a[8], b[8];
            #pragma unroll
            for (int i = 0; i < 8; i++) a[i] = As[k][ty + 16 * i];
            #pragma unroll
            for (int j = 0; j < 8; j++) b[j] = Bs[k][tx + 16 * j];
            #pragma unroll
            for (int i = 0; i < 8; i++)
                #pragma unroll
                for (int j = 0; j < 8; j++) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        const int r = r0 + tx + 16 * j;
        if (r >= R) continue;
        const float scale = f32at(scales + (size_t)r * scale_stride);
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int n = n0 + ty + 16 * i;
            if (n < N) C[(size_t)n * ldc + r] = acc[i][j] * scale;
        }
    }
}

// Same product, same order of additions per output (so the same bits as gemm_i8), built to spend its time on the multiply-adds:
// a 128 x 128 tile, 32 columns of k per step (a full 32-byte sector of each weight row), the next step's loads in registers while this step
// computes, and the shared memory read in 16-byte pieces: 4 shared loads per 64 multiply-adds instead of 16. K must be a multiple of 32.
#define V2_BK 32
__global__ void __launch_bounds__(256, 2) gemm_i8_v2(const float *__restrict__ A, int N, int K, const unsigned char *__restrict__ codes, size_t code_stride,
    const unsigned char *__restrict__ scales, size_t scale_stride, int R, float *__restrict__ C, int ldc)
{
    __shared__ float As[V2_BK][BM + 4], Bs[V2_BK][BN + 4];
    const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
    const int n0 = blockIdx.y * BM, r0 = blockIdx.x * BN;
    const int lrow = threadIdx.x >> 1, lk = (threadIdx.x & 1) * 16;
    const bool a_ok = n0 + lrow < N, b_ok = r0 + lrow < R;
    const float *a_src = A + (size_t)(n0 + lrow) * K + lk;
    const unsigned char *b_src = codes + (size_t)(r0 + lrow) * code_stride + lk;
    float acc[8][8];
    #pragma unroll
    for (int i = 0; i < 8; i++) for (int j = 0; j < 8; j++) acc[i][j] = 0.0f;
    float4 av[4]; uint32_t wv[4];
    #pragma unroll
    for (int q = 0; q < 4; q++) { av[q] = a_ok ? *(const float4 *)(a_src + q * 4) : make_float4(0.f, 0.f, 0.f, 0.f); wv[q] = b_ok ? *(const uint32_t *)(b_src + q * 4) : 0u; }
    for (int k0 = 0; k0 < K; k0 += V2_BK) {
        #pragma unroll
        for (int q = 0; q < 4; q++) {
            As[lk + q * 4 + 0][lrow] = av[q].x; As[lk + q * 4 + 1][lrow] = av[q].y; As[lk + q * 4 + 2][lrow] = av[q].z; As[lk + q * 4 + 3][lrow] = av[q].w;
            Bs[lk + q * 4 + 0][lrow] = (float)(int8_t)(wv[q] & 0xff); Bs[lk + q * 4 + 1][lrow] = (float)(int8_t)((wv[q] >> 8) & 0xff);
            Bs[lk + q * 4 + 2][lrow] = (float)(int8_t)((wv[q] >> 16) & 0xff); Bs[lk + q * 4 + 3][lrow] = (float)(int8_t)(wv[q] >> 24);
        }
        __syncthreads();
        if (k0 + V2_BK < K) {
            #pragma unroll
            for (int q = 0; q < 4; q++) { av[q] = a_ok ? *(const float4 *)(a_src + k0 + V2_BK + q * 4) : make_float4(0.f, 0.f, 0.f, 0.f); wv[q] = b_ok ? *(const uint32_t *)(b_src + k0 + V2_BK + q * 4) : 0u; }
        }
        #pragma unroll 4
        for (int kk = 0; kk < V2_BK; kk++) {
            const float4 a0 = *(const float4 *)&As[kk][ty * 4], a1 = *(const float4 *)&As[kk][ty * 4 + 64];
            const float4 b0 = *(const float4 *)&Bs[kk][tx * 4], b1 = *(const float4 *)&Bs[kk][tx * 4 + 64];
            const float a[8] = { a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w }, b[8] = { b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w };
            #pragma unroll
            for (int i = 0; i < 8; i++)
                #pragma unroll
                for (int j = 0; j < 8; j++) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        const int r = r0 + tx * 4 + (j & 3) + (j >> 2) * 64;
        if (r >= R) continue;
        const float scale = f32at(scales + (size_t)r * scale_stride);
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const int n = n0 + ty * 4 + (i & 3) + (i >> 2) * 64;
            if (n < N) C[(size_t)n * ldc + r] = acc[i][j] * scale;
        }
    }
}

// the 3-tap causal convolution over positions, then x * sigmoid(x); raw[comp] and out[comp] are N x ROWS
__global__ void conv_silu(float *out, const float *raw, const unsigned char *op, int comp, int N)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)N * ROWS) return;
    const int n = (int)(idx / ROWS), row = (int)(idx % ROWS);
    const unsigned char *taps = op + QKV_PREFIX + ((size_t)row * 3 + comp) * QKV_COMP + 4;
    const float r0 = raw[idx];
    float h2 = n >= 1 ? raw[idx - ROWS] : 0.0f, h1 = n >= 2 ? raw[idx - 2 * (size_t)ROWS] : 0.0f, h0 = n >= 3 ? raw[idx - 3 * (size_t)ROWS] : 0.0f;
    float v = f32at(taps) * r0;
    v = v + f32at(taps + 4) * h0;
    v = v + f32at(taps + 8) * h1;
    v = v + f32at(taps + 12) * h2;
    out[idx] = v * sigmoidf_(v);
}

// divide each head's 128 values by their L2 norm (queries and keys)
__global__ void l2_heads(float *x)
{
    float *b = x + (size_t)blockIdx.x * HEAD;
    double s = (double)b[threadIdx.x] * (double)b[threadIdx.x];
    s = block_sum(s);
    __shared__ float inverse;
    if (threadIdx.x == 0) inverse = (float)(1.0 / sqrt(s + (double)1e-6f));
    __syncthreads();
    b[threadIdx.x] *= inverse;
}

__global__ void beta_sigmoid(float *beta, size_t count)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) beta[i] = sigmoidf_(beta[i]);
}

__global__ void decay_transform(float *decay, const unsigned char *exponent, const unsigned char *bias, int N)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)N * ROWS) return;
    const int col = (int)(idx % ROWS), head = col / HEAD;
    const float value = f32at(exponent + (size_t)head * 4) * (decay[idx] + f32at(bias + (size_t)col * 4));
    decay[idx] = expf_cr(-5.0f * sigmoidf_(value));
}

// the recurrent state of one head, positions in order; thread = one column of the 128 x 128 state, kept in registers for the whole sequence
// (the same operations in the same order per element as the version that kept the state in global memory); the next position's inputs are
// loaded before this position's long sequential sums, and the per-position vectors are double-buffered so there is one barrier per position
__global__ void __launch_bounds__(HEAD) recurrence(float *state, const float *q, const float *k, const float *v, const float *decay, const float *beta, float *out, int N)
{
    const int head = blockIdx.x, col = threadIdx.x;
    float *Sg = state + (size_t)head * HEAD * HEAD;
    __shared__ __align__(16) float key[2][HEAD], kbeta[2][HEAD], sq[2][HEAD], dec[2][HEAD];
    const float query_scale = 1.0f / sqrtf(128.0f);
    float S[HEAD];
    #pragma unroll
    for (int row = 0; row < HEAD; row++) S[row] = Sg[row * HEAD + col];
    size_t base = (size_t)head * HEAD;
    float kr = 0.f, dr = 0.f, qr = 0.f, vr = 0.f, br = 0.f;
    if (N > 0) { kr = k[base + col]; dr = decay[base + col]; qr = q[base + col]; vr = v[base + col]; br = beta[head]; }
    for (int n = 0; n < N; n++) {
        const int bf = n & 1;
        key[bf][col] = kr; dec[bf][col] = dr; kbeta[bf][col] = kr * br; sq[bf][col] = qr * query_scale;
        const float value = vr;
        const size_t cur = base;
        base += ROWS;
        if (n + 1 < N) { kr = k[base + col]; dr = decay[base + col]; qr = q[base + col]; vr = v[base + col]; br = beta[(size_t)(n + 1) * HEADS + head]; }
        __syncthreads();
        float prediction = 0.0f;
        #pragma unroll
        for (int r4 = 0; r4 < HEAD / 4; r4++) {
            const float4 dv = *(const float4 *)&dec[bf][r4 * 4], kv = *(const float4 *)&key[bf][r4 * 4];
            const float s0 = dv.x * S[r4 * 4 + 0]; prediction = prediction + s0 * kv.x;
            const float s1 = dv.y * S[r4 * 4 + 1]; prediction = prediction + s1 * kv.y;
            const float s2 = dv.z * S[r4 * 4 + 2]; prediction = prediction + s2 * kv.z;
            const float s3 = dv.w * S[r4 * 4 + 3]; prediction = prediction + s3 * kv.w;
            S[r4 * 4 + 0] = s0; S[r4 * 4 + 1] = s1; S[r4 * 4 + 2] = s2; S[r4 * 4 + 3] = s3;
        }
        const float delta = value - prediction;
        float o = 0.0f;
        #pragma unroll
        for (int r4 = 0; r4 < HEAD / 4; r4++) {
            const float4 bv = *(const float4 *)&kbeta[bf][r4 * 4], qv = *(const float4 *)&sq[bf][r4 * 4];
            const float s0 = S[r4 * 4 + 0] + bv.x * delta; o = o + s0 * qv.x; S[r4 * 4 + 0] = s0;
            const float s1 = S[r4 * 4 + 1] + bv.y * delta; o = o + s1 * qv.y; S[r4 * 4 + 1] = s1;
            const float s2 = S[r4 * 4 + 2] + bv.z * delta; o = o + s2 * qv.z; S[r4 * 4 + 2] = s2;
            const float s3 = S[r4 * 4 + 3] + bv.w * delta; o = o + s3 * qv.w; S[r4 * 4 + 3] = s3;
        }
        out[cur + col] = o;
    }
    #pragma unroll
    for (int row = 0; row < HEAD; row++) Sg[row * HEAD + col] = S[row];
}

__global__ void gate_mul(float *gate, const float *attention, size_t count)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) gate[i] = attention[i] * sigmoidf_(gate[i]);
}

// mix the layer's input (snapshot) with the attention output by a softmax over two scores; one block per position
__global__ void aggregate(float *agg, const float *snap, const float *res, const unsigned char *score_vector)
{
    const float *src[2] = { snap + (size_t)blockIdx.x * W, res + (size_t)blockIdx.x * W };
    float scores[2];
    __shared__ float inv[2], shared_score[2];
    for (int s = 0; s < 2; s++) {
        double q = 0.0;
        for (int c = threadIdx.x; c < W; c += blockDim.x) q += (double)src[s][c] * (double)src[s][c];
        q = block_sum(q);
        if (threadIdx.x == 0) inv[s] = (float)(1.0 / sqrt(q / (double)W + (double)1e-5f));
        __syncthreads();
        double sc = 0.0;
        for (int c = threadIdx.x; c < W; c += blockDim.x) sc += (double)(src[s][c] * inv[s]) * (double)f32at(score_vector + (size_t)c * 4);
        sc = block_sum(sc);
        if (threadIdx.x == 0) shared_score[s] = (float)sc;
        __syncthreads();
        scores[s] = shared_score[s];
    }
    const float mx = scores[1] > scores[0] ? scores[1] : scores[0];
    const float e0 = expf_cr(scores[0] - mx), e1 = expf_cr(scores[1] - mx);
    const double total = 0.0 + (double)e0 + (double)e1;
    const float w0 = (float)((double)e0 / total), w1 = (float)((double)e1 / total);
    for (int c = threadIdx.x; c < W; c += blockDim.x) {
        float value = 0.0f;
        value = value + w0 * src[0][c];
        value = value + w1 * src[1][c];
        agg[(size_t)blockIdx.x * W + c] = value;
    }
}

__global__ void dense_activation(float *gate, const float *up, size_t count)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float g = gate[i];
    const float activated = (4.0f * tanhf_cr(g / 4.0f)) * sigmoidf_(g);
    const float capped = 25.0f * tanhf_cr(up[i] / 25.0f);
    gate[i] = activated * capped;
}

__global__ void add_into(float *a, const float *b, size_t count)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) a[i] = a[i] + b[i];
}

// ---------------------------------------------------------------- the layer
struct Trunk {
    unsigned char *group[3], *op;
    Record rec[64];
    unsigned char *index_valid;
};
static const unsigned char *rec_data(const Trunk &L, unsigned id) { return L.group[L.rec[id].group] + L.rec[id].offset; }

static void gemm(const float *A, int N, int K, const Trunk &L, unsigned id, float *C);
static int blocks(size_t n) { return (int)((n + 255) / 256); }

#ifdef INT8_PLANES
// -DINT8_PLANES: the product on the integer tensor cores. The weights are exact 8-bit integers; each token's activations are scaled by the token's
// largest value and cut into 3 signed 8-bit digits (23 bits); one exact integer product per digit (cuBLAS, int32 sums), the digits recombined in
// double and the row scale applied. Not the float kernel's sums: the activations are rounded to 23 bits relative to the token's largest value.
__global__ void quantize_planes(int8_t *X8, double *tscale, const float *X, int N, int K)
{
    const int n = blockIdx.x, tid = threadIdx.x;
    const float *x = X + (size_t)n * K;
    float m = 0.0f;
    for (int c = tid; c < K; c += blockDim.x) m = fmaxf(m, fabsf(x[c]));
    __shared__ float sh[32], smax;
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, o));
    if ((tid & 31) == 0) sh[tid >> 5] = m;
    __syncthreads();
    m = tid < blockDim.x / 32 ? sh[tid] : 0.0f;
    if (tid < 32) for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, o));
    if (tid == 0) smax = m;
    __syncthreads();
    const double qmax = 8355711.0;   // the largest value whose balanced digits all fit in 8 bits
    const double inv = smax > 0.0f ? qmax / (double)smax : 0.0;
    if (tid == 0) tscale[n] = smax > 0.0f ? (double)smax / qmax : 0.0;
    for (int c = tid; c < K; c += blockDim.x) {
        long long t = llrint((double)x[c] * inv);
        for (int p = 0; p < 3; p++) {
            int d;
            if (p < 2) { d = (int)(((t + 128) & 255) - 128); t = (t - d) >> 8; } else d = (int)t;
            X8[((size_t)p * N + n) * K + c] = (int8_t)d;
        }
    }
}
__global__ void combine_planes(float *out, const int *C32, const double *tscale, const unsigned char *scales, size_t scale_stride, int N, int R)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)N * R) return;
    const int n = (int)(i / R), r = (int)(i % R);
    long long acc = 0;
    for (int p = 0; p < 3; p++) acc += (long long)C32[((size_t)p * N + n) * R + r] << (8 * p);
    out[i] = (float)((double)acc * tscale[n] * (double)f32at(scales + (size_t)r * scale_stride));
}
struct Planes { cublasHandle_t h; int8_t *x8; int *c32; double *ts; size_t x8_bytes, c32_bytes, ts_bytes; int ready; };
static Planes g_planes_dev[16];   // per device: the handle and scratch belong to the card that was current when they were made
static size_t g_planes_budget = 0;
static void grow(void **p, size_t *have, size_t want) { if (*have < want) { if (*p) cudaFree(*p); CHECK(cudaMalloc(p, want)); *have = want; } }
static void gemm_planes(const float *A, int N, int K, const unsigned char *codes, size_t code_stride, const unsigned char *scales, size_t scale_stride, int R, float *C)
{
    if (K % 4 || R % 4 || code_stride % 4) { fprintf(stderr, "integer product needs K, R and the row stride to be multiples of 4\n"); exit(1); }
    int dev_ = 0; CHECK(cudaGetDevice(&dev_));
    if (dev_ >= 16) { fprintf(stderr, "gemm_planes: device index %d not supported\n", dev_); exit(1); }
    Planes &g_planes = g_planes_dev[dev_];
    if (!g_planes.ready) { CUB(cublasCreate(&g_planes.h)); g_planes.ready = 1; }
    if (!g_planes_budget) { const char *e = getenv("MODEL_GPU_PLANES_MB"); g_planes_budget = (size_t)(e && *e ? atoi(e) : 384) << 20; }   // scratch cap, MB
    const size_t per_row = (size_t)3 * K + (size_t)12 * R + 8;
    const int nc = (int)std::min((size_t)N, std::max((size_t)64, g_planes_budget / per_row));
    grow((void **)&g_planes.x8, &g_planes.x8_bytes, (size_t)3 * nc * K);
    grow((void **)&g_planes.c32, &g_planes.c32_bytes, (size_t)3 * nc * R * 4);
    grow((void **)&g_planes.ts, &g_planes.ts_bytes, (size_t)nc * 8);
    const int alpha = 1, beta = 0;
    for (int n0 = 0; n0 < N; n0 += nc) {   // each row's digits and sums are its own, so the rows can go through in pieces with the same bits
        const int m = N - n0 < nc ? N - n0 : nc;
        quantize_planes<<<m, 256>>>(g_planes.x8, g_planes.ts, A + (size_t)n0 * K, m, K);
        CUB(cublasGemmEx(g_planes.h, CUBLAS_OP_T, CUBLAS_OP_N, R, 3 * m, K, &alpha, codes, CUDA_R_8I, (int)code_stride, g_planes.x8, CUDA_R_8I, K,
                         &beta, g_planes.c32, CUDA_R_32I, R, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        combine_planes<<<blocks((size_t)m * R), 256>>>(C + (size_t)n0 * R, g_planes.c32, g_planes.ts, scales, scale_stride, m, R);
    }
}
#endif

// one 8-bit weight matrix times N rows of input: C[n * ldc + r]
static void gemm_raw(const float *A, int N, int K, const unsigned char *codes, size_t code_stride, const unsigned char *scales, size_t scale_stride, int R, float *C, int ldc)
{
#ifdef INT8_PLANES
    if (ldc == R) { gemm_planes(A, N, K, codes, code_stride, scales, scale_stride, R, C); return; }
#endif
    dim3 grid((R + BN - 1) / BN, (N + BM - 1) / BM);
    gemm_i8_v2<<<grid, 256>>>(A, N, K, codes, code_stride, scales, scale_stride, R, C, ldc);
}
static void gemm(const float *A, int N, int K, const Trunk &L, unsigned id, float *C)
{
    const Record &r = L.rec[id];
    if ((unsigned)K != r.cols || (r.offset & 3) || (r.length & 3)) { fprintf(stderr, "record %u does not fit\n", id); exit(1); }
    const unsigned char *d = rec_data(L, id);
    gemm_raw(A, N, K, d + (size_t)r.rows * 4, r.cols, d, 4, r.rows, C, r.rows);
}

