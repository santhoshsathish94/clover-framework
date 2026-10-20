// Kernels and helpers shared by the GPU layer programs (layer0.cu, layer1.cu). See layer0.cu for the arithmetic they follow.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <vector>
#include <cuda_runtime.h>

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
__device__ __forceinline__ float expf_cr(float v) { return (float)exp((double)v); }
__device__ __forceinline__ float tanhf_cr(float v) { return (float)tanh((double)v); }
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

// the recurrent state of one head, positions in order; thread = one column of the 128 x 128 state
__global__ void recurrence(float *state, const float *q, const float *k, const float *v, const float *decay, const float *beta, float *out, int N)
{
    const int head = blockIdx.x, col = threadIdx.x;
    float *S = state + (size_t)head * HEAD * HEAD;
    __shared__ float key[HEAD], kbeta[HEAD], sq[HEAD], dec[HEAD];
    const float query_scale = 1.0f / sqrtf(128.0f);
    for (int n = 0; n < N; n++) {
        const size_t base = (size_t)n * ROWS + (size_t)head * HEAD;
        const float b = beta[(size_t)n * HEADS + head];
        key[col] = k[base + col]; dec[col] = decay[base + col];
        kbeta[col] = k[base + col] * b; sq[col] = q[base + col] * query_scale;
        __syncthreads();
        const float value = v[base + col];
        float prediction = 0.0f;
        for (int row = 0; row < HEAD; row++) {
            const float st = dec[row] * S[row * HEAD + col];
            S[row * HEAD + col] = st;
            prediction = prediction + st * key[row];
        }
        float o = 0.0f;
        for (int row = 0; row < HEAD; row++) {
            const float st = S[row * HEAD + col] + kbeta[row] * (value - prediction);
            S[row * HEAD + col] = st;
            o = o + st * sq[row];
        }
        out[base + col] = o;
        __syncthreads();
    }
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

static void gemm(const float *A, int N, int K, const Trunk &L, unsigned id, float *C)
{
    const Record &r = L.rec[id];
    if ((unsigned)K != r.cols || (r.offset & 3) || (r.length & 3)) { fprintf(stderr, "record %u does not fit\n", id); exit(1); }
    const unsigned char *d = rec_data(L, id);
    dim3 grid((r.rows + BN - 1) / BN, (N + BM - 1) / BM);
    gemm_i8<<<grid, 256>>>(A, N, K, d + (size_t)r.rows * 4, r.cols, d, 4, r.rows, C, r.rows);
}
static int blocks(size_t n) { return (int)((n + 255) / 256); }

