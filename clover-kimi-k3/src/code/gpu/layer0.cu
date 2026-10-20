// Layer 0 of Kimi K3 on one GPU, for a window of tokens at once, compared with the CPU engine's layer 0.
//
//   layer0-gpu TRUNK0_DIR EMBED_F32 REF_F32 [REPEATS]
//
// TRUNK0_DIR  dataset/trunk-0 (index.bin, common.bin, kda.bin, dense.bin, operator.bin), copied to the card as it is
// EMBED_F32   the layer 0 input rows (N x 7168 floats), written by the CPU engine: PF_SAVE_ROWS=base gives base.embed
// REF_F32     the CPU engine's layer 0 output rows (base.layer0)
//
// The arithmetic follows src/code/cpu/layer0.c: weights are 8-bit codes (the code as a signed integer) with one float scale per row,
// the sum is taken first and the scale applied after. Matrix sums run in a different order from the CPU's 16-lane order, so the
// matrix products are not expected to be bit-identical; everything else (norms in double, convolution, recurrence) keeps the CPU's
// order and is built without fused multiply-add (--fmad=false), as the CPU engine is (-ffp-contract=off).
//
// build: nvcc -O3 --fmad=false -arch=sm_86 -o layer0-gpu layer0.cu
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
__device__ __forceinline__ float sigmoidf_(float v) { return 1.0f / (1.0f + expf(-v)); }
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
                for (int j = 0; j < 8; j++) acc[i][j] += a[i] * b[j];
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
    decay[idx] = expf(-5.0f * sigmoidf_(value));
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
    const float e0 = expf(scores[0] - mx), e1 = expf(scores[1] - mx);
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
    const float activated = (4.0f * tanhf(g / 4.0f)) * sigmoidf_(g);
    const float capped = 25.0f * tanhf(up[i] / 25.0f);
    gate[i] = activated * capped;
}

__global__ void add_into(float *a, const float *b, size_t count)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) a[i] = a[i] + b[i];
}

// ---------------------------------------------------------------- the layer
struct Layer0 {
    unsigned char *group[3], *op;
    Record rec[64];
    unsigned char *index_valid;
};
static const unsigned char *rec_data(const Layer0 &L, unsigned id) { return L.group[L.rec[id].group] + L.rec[id].offset; }

static void gemm(const float *A, int N, int K, const Layer0 &L, unsigned id, float *C)
{
    const Record &r = L.rec[id];
    if ((unsigned)K != r.cols || (r.offset & 3) || (r.length & 3)) { fprintf(stderr, "record %u does not fit\n", id); exit(1); }
    const unsigned char *d = rec_data(L, id);
    dim3 grid((r.rows + BN - 1) / BN, (N + BM - 1) / BM);
    gemm_i8<<<grid, 256>>>(A, N, K, d + (size_t)r.rows * 4, r.cols, d, 4, r.rows, C, r.rows);
}
static int blocks(size_t n) { return (int)((n + 255) / 256); }

int main(int argc, char **argv)
{
    if (argc < 4) { fprintf(stderr, "usage: layer0-gpu TRUNK0_DIR EMBED_F32 REF_F32 [REPEATS]\n"); return 2; }
    const int repeats = argc > 4 ? atoi(argv[4]) : 3;
    char path[4096];
    Layer0 L; memset(&L, 0, sizeof L);

    snprintf(path, sizeof path, "%s/index.bin", argv[1]);
    std::vector<unsigned char> index = slurp(path);
    const char *names[3] = { "common.bin", "kda.bin", "dense.bin" };
    size_t group_bytes[3];
    for (int g = 0; g < 3; g++) group_bytes[g] = (size_t)u64(index.data() + 24 + g * 8);
    const unsigned count = u32(index.data() + 16);
    if (index.size() < 1168 + (size_t)count * 72 || memcmp(index.data(), "K3TRK001", 8)) { fprintf(stderr, "index.bin is not a trunk index\n"); return 1; }
    for (unsigned p = 0; p < count; p++) {
        const unsigned char *e = index.data() + 1168 + (size_t)p * 72;
        Record r = { u32(e), u32(e + 4), u32(e + 8), u32(e + 12), u32(e + 16), u64(e + 24), u64(e + 32) };
        if (r.id >= 64 || r.group > 2) { fprintf(stderr, "bad record\n"); return 1; }
        L.rec[r.id] = r;
    }

    CHECK(cudaSetDevice(0));
    cudaDeviceProp prop; CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("device 0: %s, %.1f GiB\n", prop.name, prop.totalGlobalMem / 1073741824.0);

    cudaEvent_t t0, t1; CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));
    struct timespec ts0, ts1; clock_gettime(CLOCK_MONOTONIC, &ts0);
    size_t uploaded = 0;
    for (int g = 0; g < 3; g++) {
        snprintf(path, sizeof path, "%s/%s", argv[1], names[g]);
        std::vector<unsigned char> bytes = slurp(path);
        if (bytes.size() != group_bytes[g]) { fprintf(stderr, "%s has the wrong size\n", path); return 1; }
        CHECK(cudaMalloc(&L.group[g], bytes.size()));
        CHECK(cudaMemcpy(L.group[g], bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
        uploaded += bytes.size();
    }
    snprintf(path, sizeof path, "%s/operator.bin", argv[1]);
    std::vector<unsigned char> opbytes = slurp(path);
    const size_t op_expect = QKV_PREFIX + (size_t)ROWS * 3 * QKV_COMP;
    if (opbytes.size() != op_expect || memcmp(opbytes.data(), "K3QKV001", 8)) { fprintf(stderr, "operator.bin is not what the engine reads\n"); return 1; }
    CHECK(cudaMalloc(&L.op, opbytes.size()));
    CHECK(cudaMemcpy(L.op, opbytes.data(), opbytes.size(), cudaMemcpyHostToDevice));
    uploaded += opbytes.size();
    clock_gettime(CLOCK_MONOTONIC, &ts1);
    printf("weights on the card: %.2f GB in %.2f s (read from disk and copied)\n", uploaded / 1e9, (ts1.tv_sec - ts0.tv_sec) + 1e-9 * (ts1.tv_nsec - ts0.tv_nsec));

    std::vector<unsigned char> embed = slurp(argv[2]), ref = slurp(argv[3]);
    if (embed.size() % (W * 4) || ref.size() != embed.size()) { fprintf(stderr, "embedding and reference files do not fit\n"); return 1; }
    const int N = (int)(embed.size() / (W * 4));
    printf("%d tokens\n", N);

    float *X, *Xn, *raw[3], *qkv[3], *beta, *dhid, *decay, *gate, *attn, *res, *agg, *post, *dg, *du, *dout, *state;
    auto alloc = [&](float **p, size_t floats) { CHECK(cudaMalloc((void **)p, floats * 4)); };
    alloc(&X, (size_t)N * W); alloc(&Xn, (size_t)N * W);
    for (int c = 0; c < 3; c++) { alloc(&raw[c], (size_t)N * ROWS); alloc(&qkv[c], (size_t)N * ROWS); }
    alloc(&beta, (size_t)N * HEADS); alloc(&dhid, (size_t)N * HEAD); alloc(&decay, (size_t)N * ROWS);
    alloc(&gate, (size_t)N * ROWS); alloc(&attn, (size_t)N * ROWS); alloc(&res, (size_t)N * W); alloc(&agg, (size_t)N * W); alloc(&post, (size_t)N * W);
    alloc(&dg, (size_t)N * DENSE); alloc(&du, (size_t)N * DENSE); alloc(&dout, (size_t)N * W); alloc(&state, (size_t)HEADS * HEAD * HEAD);
    CHECK(cudaMemcpy(X, embed.data(), embed.size(), cudaMemcpyHostToDevice));

    const char *stage_name[] = { "input norm", "q/k/v matrices", "convolution + l2", "decay + beta", "recurrence", "gate + output matrix", "aggregate", "dense (3 matrices)" };
    float stage_ms[8];
    std::vector<float> out((size_t)N * W);
    for (int rep = 0; rep < repeats; rep++) {
        CHECK(cudaMemset(state, 0, (size_t)HEADS * HEAD * HEAD * 4));
        CHECK(cudaDeviceSynchronize());
        struct timespec a, b; clock_gettime(CLOCK_MONOTONIC, &a);
        int s = 0;
        auto mark = [&](void) { CHECK(cudaEventRecord(t1)); CHECK(cudaEventSynchronize(t1)); float ms; CHECK(cudaEventElapsedTime(&ms, t0, t1)); stage_ms[s++] = ms; CHECK(cudaEventRecord(t0)); };
        CHECK(cudaEventRecord(t0));
        normalize_rows<<<N, 256>>>(Xn, X, L.op + 88, W);                                                          mark();
        for (int c = 0; c < 3; c++) {   // each (row, component) record is [scale, 4 taps, 7168 codes]; rows of one component are 3 records apart
            dim3 grid((ROWS + BN - 1) / BN, (N + BM - 1) / BM);
            gemm_i8<<<grid, 256>>>(Xn, N, W, L.op + QKV_PREFIX + (size_t)c * QKV_COMP + 20, (size_t)3 * QKV_COMP, L.op + QKV_PREFIX + (size_t)c * QKV_COMP, (size_t)3 * QKV_COMP, ROWS, raw[c], ROWS);
        }                                                                                                         mark();
        for (int c = 0; c < 3; c++) conv_silu<<<blocks((size_t)N * ROWS), 256>>>(qkv[c], raw[c], L.op, c, N);
        l2_heads<<<N * HEADS, HEAD>>>(qkv[0]); l2_heads<<<N * HEADS, HEAD>>>(qkv[1]);                             mark();
        gemm(Xn, N, W, L, 11, beta); beta_sigmoid<<<blocks((size_t)N * HEADS), 256>>>(beta, (size_t)N * HEADS);
        gemm(Xn, N, W, L, 12, dhid); gemm(dhid, N, HEAD, L, 13, decay);
        decay_transform<<<blocks((size_t)N * ROWS), 256>>>(decay, rec_data(L, 39), rec_data(L, 18), N);           mark();
        recurrence<<<HEADS, HEAD>>>(state, qkv[0], qkv[1], qkv[2], decay, beta, attn, N);                          mark();
        gemm(Xn, N, W, L, 6, gate);
        normalize_rows<<<N * HEADS, HEAD>>>(attn, attn, rec_data(L, 19), HEAD);
        gate_mul<<<blocks((size_t)N * ROWS), 256>>>(gate, attn, (size_t)N * ROWS);
        gemm(gate, N, ROWS, L, 7, res);                                                                            mark();
        aggregate<<<N, 256>>>(agg, X, res, rec_data(L, 38));                                                       mark();
        normalize_rows<<<N, 256>>>(post, agg, rec_data(L, 5), W);
        gemm(post, N, W, L, 26, dg); gemm(post, N, W, L, 27, du);
        dense_activation<<<blocks((size_t)N * DENSE), 256>>>(dg, du, (size_t)N * DENSE);
        gemm(dg, N, DENSE, L, 28, dout);
        add_into<<<blocks((size_t)N * W), 256>>>(res, dout, (size_t)N * W);                                        mark();
        CHECK(cudaGetLastError());
        clock_gettime(CLOCK_MONOTONIC, &b);
        double total_ms = 1000.0 * ((b.tv_sec - a.tv_sec) + 1e-9 * (b.tv_nsec - a.tv_nsec));
        printf("run %d: layer 0 for %d tokens in %.1f ms (", rep + 1, N, total_ms);
        for (int i = 0; i < 8; i++) printf("%s%s %.1f", i ? ", " : "", stage_name[i], stage_ms[i]);
        printf(" ms)\n");
    }
    CHECK(cudaMemcpy(out.data(), res, out.size() * 4, cudaMemcpyDeviceToHost));

    // against the CPU engine's layer 0 output
    const float *r = (const float *)ref.data();
    size_t same = 0, nonfinite = 0; double maxabs = 0, sumd2 = 0, sumr2 = 0, worst_pos_err = 0; double min_cos = 1.0; int worst_pos = 0;
    for (int n = 0; n < N; n++) {
        double dot = 0, a2 = 0, b2 = 0, pd2 = 0;
        for (int c = 0; c < W; c++) {
            const float x = out[(size_t)n * W + c], y = r[(size_t)n * W + c];
            if (!std::isfinite(x)) nonfinite++;
            same += memcmp(&x, &y, 4) == 0;
            const double d = (double)x - (double)y;
            if (fabs(d) > maxabs) maxabs = fabs(d);
            sumd2 += d * d; sumr2 += (double)y * y; pd2 += d * d; dot += (double)x * y; a2 += (double)x * x; b2 += (double)y * y;
        }
        const double cs = dot / (sqrt(a2) * sqrt(b2) + 1e-300);
        if (cs < min_cos) min_cos = cs;
        const double pe = sqrt(pd2 / (b2 + 1e-300));
        if (pe > worst_pos_err) { worst_pos_err = pe; worst_pos = n; }
    }
    printf("against the CPU engine's layer 0 output (%d x %d values): bit-identical %zu (%.2f%%), non-finite %zu, max abs difference %.3g, relative RMS difference %.3g, "
           "lowest cosine of a position %.9f, worst position %d (relative RMS %.3g)\n", N, W, same, 100.0 * same / ((double)N * W), nonfinite, maxabs, sqrt(sumd2 / sumr2),
           min_cos, worst_pos, worst_pos_err);
    return 0;
}
