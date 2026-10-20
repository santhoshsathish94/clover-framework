// The whole model's layers on a GPU: a layer runner for layer 0 and for layers 1 to 92 (KDA or MLA attention, then the mixture of experts), shared by the
// single-card program (model-gpu.cu) and the eight-card pipeline. Follows src/code/cpu/layer0.c and layer-kda.c / layer-mla.c, the sequence of one position
// done for all N positions of a window at once. Options as in the layer programs: -DFAST_ROUTER, -DEXPERT_V2, -DINT8_PLANES.
#pragma once
#include "moe.cuh"
#include <algorithm>
#include <string>
#include <fcntl.h>
#include <unistd.h>

static double seconds(const struct timespec &a, const struct timespec &b) { return (b.tv_sec - a.tv_sec) + 1e-9 * (b.tv_nsec - a.tv_nsec); }
enum { MLA_Q = 18432, MLA_QLAT = 1536, MLA_KVLAT = 576, MLA_KV = 512, MLA_EXP = 24576, LAST_LAYER_INDEX = 92 };
static bool layer_is_mla(int l) { return l % 4 == 3 || l == LAST_LAYER_INDEX; }
static int snapshots_in(int l) { return 1 + (l - 1) / 12; }

// ---------------------------------------------------------------- more kernels
struct Srcs { const float *p[9]; };

// the weighted mix of the layer's snapshots and its residual (S sources), one block per position, exactly the CPU's order
__global__ void aggregate_multi(float *agg, Srcs s, int S, const unsigned char *score_vector)
{
    __shared__ float inv[9], sc[9];
    for (int k = 0; k < S; k++) {
        const float *x = s.p[k] + (size_t)blockIdx.x * W;
        double q = 0.0;
        for (int c = threadIdx.x; c < W; c += blockDim.x) q += (double)x[c] * (double)x[c];
        q = block_sum(q);
        if (threadIdx.x == 0) inv[k] = (float)(1.0 / sqrt(q / (double)W + (double)1e-5f));
        __syncthreads();
        double d = 0.0;
        for (int c = threadIdx.x; c < W; c += blockDim.x) d += (double)(x[c] * inv[k]) * (double)f32at(score_vector + (size_t)c * 4);
        d = block_sum(d);
        if (threadIdx.x == 0) sc[k] = (float)d;
        __syncthreads();
    }
    __shared__ float weights[9];
    if (threadIdx.x == 0) {
        float mx = sc[0];
        for (int k = 1; k < S; k++) if (sc[k] > mx) mx = sc[k];
        float e[9]; double total = 0.0;
        for (int k = 0; k < S; k++) { e[k] = expf_cr(sc[k] - mx); total += (double)e[k]; }
        for (int k = 0; k < S; k++) weights[k] = (float)((double)e[k] / total);
    }
    __syncthreads();
    for (int c = threadIdx.x; c < W; c += blockDim.x) {
        float v = 0.0f;
        for (int k = 0; k < S; k++) v = v + weights[k] * s.p[k][(size_t)blockIdx.x * W + c];
        agg[(size_t)blockIdx.x * W + c] = v;
    }
}

// normalise the first `width` values of each row of `stride` values, in place
__global__ void normalize_prefix(float *x, const unsigned char *gains, int width, int stride)
{
    float *r = x + (size_t)blockIdx.x * stride;
    double s = 0.0;
    for (int c = threadIdx.x; c < width; c += blockDim.x) s += (double)r[c] * (double)r[c];
    s = block_sum(s);
    __shared__ float inverse;
    if (threadIdx.x == 0) inverse = (float)(1.0 / sqrt(s / (double)width + (double)1e-5f));
    __syncthreads();
    for (int c = threadIdx.x; c < width; c += blockDim.x) r[c] = (f32at(gains + (size_t)c * 4) * r[c]) * inverse;
}
__global__ void copy_prefix(float *dst, const float *src, int width, int src_stride, size_t rows)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * width) return;
    dst[i] = src[(i / width) * src_stride + i % width];
}

// MLA attention for all positions: one block per (position, head). Scores in double summed in coordinate order, as the CPU does; the values are
// added in position order. No rotation is applied (the CPU engine caches the 64 positional values as they are).
__global__ void mla_attention(float *out, const float *q, const float *kv, const float *kvlat, int N)
{
    extern __shared__ float sc[];
    __shared__ float red[32], smax;
    __shared__ double stotal;
    const int n = blockIdx.x, h = blockIdx.y, tid = threadIdx.x, count = n + 1;
    const float *qq = q + (size_t)n * MLA_Q + h * 192;
    const float scale = 1.0f / sqrtf(192.0f);
    float m = -INFINITY;
    for (int p = tid; p < count; p += blockDim.x) {
        const float *k = kv + (size_t)p * MLA_EXP + h * 256, *rope = kvlat + (size_t)p * MLA_KVLAT + MLA_KV;
        double s = 0.0;
        for (int c = 0; c < 128; c++) s += (double)qq[c] * (double)k[c];
        for (int c = 0; c < 64; c++) s += (double)qq[128 + c] * (double)rope[c];
        const float v = (float)s * scale;
        sc[p] = v;
        m = fmaxf(m, v);
    }
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, o));
    if ((tid & 31) == 0) red[tid >> 5] = m;
    __syncthreads();
    if (tid == 0) { float mm = red[0]; for (int i = 1; i < (int)(blockDim.x / 32); i++) mm = fmaxf(mm, red[i]); smax = mm; }
    __syncthreads();
    for (int p = tid; p < count; p += blockDim.x) sc[p] = expf_cr(sc[p] - smax);
    __syncthreads();
    if (tid == 0) { double t = 0.0; for (int p = 0; p < count; p++) t += (double)sc[p]; stotal = t; }
    __syncthreads();
    for (int p = tid; p < count; p += blockDim.x) sc[p] = (float)((double)sc[p] / stotal);
    __syncthreads();
    if (tid < 128) {
        float o = 0.0f;
        for (int p = 0; p < count; p++) o = o + sc[p] * kv[(size_t)p * MLA_EXP + h * 256 + 128 + tid];
        out[(size_t)n * ROWS + h * 128 + tid] = o;
    }
}

// ---------------------------------------------------------------- one layer's trunk on the card
struct Layer {
    int L = 0; bool mla = false;
    Trunk T;
    Record mr[26];                       // MLA operator records by slot
    const unsigned char *input_gains = nullptr;
    float *d_mapvals = nullptr; uint16_t *d_refs = nullptr; uint32_t *d_ref_base = nullptr;   // this layer's expert tables
    size_t bytes = 0;
};
static const unsigned char *mla_rec(const Layer &y, int slot) { return y.T.op + y.mr[slot].offset; }

static void layer_free(Layer &y)
{
    for (int g = 0; g < 3; g++) if (y.T.group[g]) cudaFree(y.T.group[g]);
    if (y.T.op) cudaFree(y.T.op);
    if (y.d_mapvals) cudaFree(y.d_mapvals);
    if (y.d_refs) cudaFree(y.d_refs);
    if (y.d_ref_base) cudaFree(y.d_ref_base);
    y = Layer();
}

// the trunk files of layer L (L = 0: common, kda, dense; L >= 1: common, attention, routed-shared) copied to the card as they are
static void layer_load_trunk(const char *dataset, int L, Layer &y)
{
    char path[4096];
    y = Layer(); y.L = L; y.mla = L >= 1 && layer_is_mla(L);
    snprintf(path, sizeof path, "%s/trunk-%d/index.bin", dataset, L);
    std::vector<unsigned char> index = slurp(path);
    if (index.size() < 1168 || memcmp(index.data(), "K3TRK001", 8) || (int)u32(index.data() + 12) != L) { fprintf(stderr, "%s is not the index of layer %d\n", path, L); exit(1); }
    const unsigned count = u32(index.data() + 16);
    for (unsigned p = 0; p < count; p++) {
        const unsigned char *e = index.data() + 1168 + (size_t)p * 72;
        Record r = { u32(e), u32(e + 4), u32(e + 8), u32(e + 12), u32(e + 16), u64(e + 24), u64(e + 32) };
        if (r.id >= 64 || r.group > 2) { fprintf(stderr, "bad record in %s\n", path); exit(1); }
        y.T.rec[r.id] = r;
    }
    static const char *names0[3] = { "common.bin", "kda.bin", "dense.bin" }, *names1[3] = { "common.bin", "attention.bin", "routed-shared.bin" };
    const char **names = L == 0 ? names0 : names1;
    for (int g = 0; g < 3; g++) {
        const size_t want = (size_t)u64(index.data() + 24 + g * 8);
        if (!want) continue;
        snprintf(path, sizeof path, "%s/trunk-%d/%s", dataset, L, names[g]);
        std::vector<unsigned char> bytes = slurp(path);
        if (bytes.size() != want) { fprintf(stderr, "%s has the wrong size\n", path); exit(1); }
        CHECK(cudaMalloc(&y.T.group[g], bytes.size()));
        CHECK(cudaMemcpy(y.T.group[g], bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
        y.bytes += bytes.size();
    }
    snprintf(path, sizeof path, "%s/trunk-%d/operator.bin", dataset, L);
    std::vector<unsigned char> op = slurp(path);
    if (!y.mla) {
        if (op.size() != (size_t)QKV_PREFIX + (size_t)ROWS * 3 * QKV_COMP || memcmp(op.data(), "K3QKV001", 8) || (int)u32(op.data() + 12) != L) { fprintf(stderr, "%s is not a KDA operator file of layer %d\n", path, L); exit(1); }
    } else {
        if (op.size() != 56251936 || memcmp(op.data(), "K3MLA001", 8) || (int)u32(op.data() + 12) != L) { fprintf(stderr, "%s is not an MLA operator file of layer %d\n", path, L); exit(1); }
        for (int i = 0; i < 7; i++) {
            const unsigned char *e = op.data() + 64 + i * 32;
            const unsigned slot = u32(e);
            if (slot >= 26) { fprintf(stderr, "bad MLA record\n"); exit(1); }
            y.mr[slot] = Record{ slot, 1, u32(e + 4), u32(e + 8), u32(e + 12), u64(e + 16), u64(e + 24) };
        }
    }
    CHECK(cudaMalloc(&y.T.op, op.size()));
    CHECK(cudaMemcpy(y.T.op, op.data(), op.size(), cudaMemcpyHostToDevice));
    y.bytes += op.size();
    y.input_gains = y.mla ? y.T.op + y.mr[4].offset : y.T.op + 88;
}

static unsigned json_count(const std::vector<unsigned char> &text, const char *key, int *found)
{
    std::string s((const char *)text.data(), text.size());
    const size_t at = s.find(std::string("\"") + key + "\"");
    if (at == std::string::npos) { *found = 0; return 0; }
    *found = 1;
    return (unsigned)strtoul(s.c_str() + s.find(':', at) + 1, NULL, 10);
}
static uint16_t u16le(const unsigned char *p) { return (uint16_t)(p[0] | (p[1] << 8)); }

// the decoding tables of layer L's experts (small), on the card
static void layer_load_expert_tables(const char *dataset, int L, Layer &y)
{
    char path[4096];
    snprintf(path, sizeof path, "%s/expert-%d/constants.bin", dataset, L); std::vector<unsigned char> constants = slurp(path);
    snprintf(path, sizeof path, "%s/expert-%d/maps.json", dataset, L); std::vector<unsigned char> json = slurp(path);
    int f1, f2, f3;
    const unsigned map_count = json_count(json, "unique_maps", &f1), template_count = json_count(json, "unique_templates", &f2), ref_count = json_count(json, "template_refs", &f3);
    if (!f1 || !f2 || !f3) { fprintf(stderr, "maps.json of layer %d lacks a count\n", L); exit(1); }
    const size_t other = (size_t)map_count * 16 + ((size_t)template_count + 1) * 2 + (size_t)ref_count * 2;
    if (constants.size() <= other || (constants.size() - other) % 2) { fprintf(stderr, "constants.bin of layer %d does not fit\n", L); exit(1); }
    const unsigned palette_count = (unsigned)((constants.size() - other) / 2);
    std::vector<float> palette(palette_count);
    for (unsigned i = 0; i < palette_count; i++) { uint32_t bits = (uint32_t)u16le(constants.data() + i * 2) << 16; memcpy(&palette[i], &bits, 4); }
    const unsigned char *maps = constants.data() + palette_count * 2, *offsets = maps + map_count * 16, *refs_p = offsets + (template_count + 1) * 2;
    std::vector<float> mapvals((size_t)map_count * 16);
    for (unsigned m = 0; m < map_count; m++) for (unsigned n = 0; n < 16; n++) mapvals[m * 16 + n] = maps[m * 16 + n] < palette_count ? palette[maps[m * 16 + n]] : 0.0f;
    snprintf(path, sizeof path, "%s/expert-%d/experts.bin", dataset, L);
    FILE *ef = fopen(path, "rb");
    if (!ef) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    std::vector<unsigned char> eindex(32 + 2688 * 2);
    if (fread(eindex.data(), 1, eindex.size(), ef) != eindex.size() || memcmp(eindex.data(), "K3MAPS01", 8)) { fprintf(stderr, "%s is not an expert index\n", path); exit(1); }
    fclose(ef);
    std::vector<uint32_t> ref_base(2688);
    for (unsigned i = 0; i < 2688; i++) { const unsigned t = u16le(eindex.data() + 32 + i * 2); if (t >= template_count) { fprintf(stderr, "bad template in layer %d\n", L); exit(1); } ref_base[i] = u16le(offsets + t * 2); }
    std::vector<uint16_t> refs(ref_count);
    for (unsigned i = 0; i < ref_count; i++) refs[i] = u16le(refs_p + i * 2);
    CHECK(cudaMalloc(&y.d_mapvals, mapvals.size() * 4)); CHECK(cudaMemcpy(y.d_mapvals, mapvals.data(), mapvals.size() * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMalloc(&y.d_refs, refs.size() * 2)); CHECK(cudaMemcpy(y.d_refs, refs.data(), refs.size() * 2, cudaMemcpyHostToDevice));
    CHECK(cudaMalloc(&y.d_ref_base, ref_base.size() * 4)); CHECK(cudaMemcpy(y.d_ref_base, ref_base.data(), ref_base.size() * 4, cudaMemcpyHostToDevice));
}

// ---------------------------------------------------------------- the buffers of a window on one card
struct Work {
    int N = 0, device = 0, SLOTS = 0;
    float *snap[8], *cur = nullptr, *nxt = nullptr;
    float *Xn, *Agg, *post, *post_t, *raw[3], *qkv[3], *beta, *dhid, *decay, *gate, *attn, *state;
    float *mq, *mqlat, *mkv, *mkvn, *mexp;                         // MLA
    float *dg = nullptr, *du = nullptr, *dout = nullptr;           // layer 0's dense MLP
    float *scores, *choices, *weights, *latent, *G, *U, *D, *mixture, *lnorm, *routed, *sg, *su, *so;
    int *selected, *bad, *d_slot_token, *d_slot_of, *d_start, *d_count;
    unsigned char *d_experts = nullptr;
#ifdef FAST_ROUTER
    float *approx, *cand_score; int *cand, *ncand, *overflow; cublasHandle_t cb;
#endif
    std::vector<int> h_selected, h_slot_token, h_slot_of, h_start, h_count;
};

static void work_alloc(Work &w, int device, int N, bool dense, bool experts)
{
    w.device = device; w.N = N; w.SLOTS = N * TOP;
    CHECK(cudaSetDevice(device));
    auto a = [&](float **p, size_t floats) { CHECK(cudaMalloc((void **)p, floats * 4)); };
    for (int i = 0; i < 8; i++) a(&w.snap[i], (size_t)N * W);
    a(&w.cur, (size_t)N * W); a(&w.nxt, (size_t)N * W);
    a(&w.Xn, (size_t)N * W); a(&w.Agg, (size_t)N * W); a(&w.post, (size_t)N * W); a(&w.post_t, (size_t)N * W);
    for (int c = 0; c < 3; c++) { a(&w.raw[c], (size_t)N * ROWS); a(&w.qkv[c], (size_t)N * ROWS); }
    a(&w.beta, (size_t)N * HEADS); a(&w.dhid, (size_t)N * HEAD); a(&w.decay, (size_t)N * ROWS); a(&w.gate, (size_t)N * ROWS); a(&w.attn, (size_t)N * ROWS);
    a(&w.state, (size_t)HEADS * HEAD * HEAD);
    a(&w.mq, (size_t)N * MLA_Q); a(&w.mqlat, (size_t)N * MLA_QLAT); a(&w.mkv, (size_t)N * MLA_KVLAT); a(&w.mkvn, (size_t)N * MLA_KV); a(&w.mexp, (size_t)N * MLA_EXP);
    if (dense) { a(&w.dg, (size_t)N * DENSE); a(&w.du, (size_t)N * DENSE); a(&w.dout, (size_t)N * W); }
    a(&w.scores, (size_t)N * EXPERTS); a(&w.choices, (size_t)N * EXPERTS); a(&w.weights, (size_t)N * TOP); a(&w.latent, (size_t)N * LATENT);
    a(&w.G, (size_t)w.SLOTS * EXPERT); a(&w.U, (size_t)w.SLOTS * EXPERT); a(&w.D, (size_t)w.SLOTS * LATENT);
    a(&w.mixture, (size_t)N * LATENT); a(&w.lnorm, (size_t)N * LATENT); a(&w.routed, (size_t)N * W);
    a(&w.sg, (size_t)N * SHARED); a(&w.su, (size_t)N * SHARED); a(&w.so, (size_t)N * W);
    CHECK(cudaMalloc(&w.selected, (size_t)N * TOP * 4)); CHECK(cudaMalloc(&w.bad, 4));
    CHECK(cudaMalloc(&w.d_slot_token, (size_t)w.SLOTS * 4)); CHECK(cudaMalloc(&w.d_slot_of, (size_t)N * TOP * 4));
    CHECK(cudaMalloc(&w.d_start, EXPERTS * 4)); CHECK(cudaMalloc(&w.d_count, EXPERTS * 4));
    if (experts) CHECK(cudaMalloc(&w.d_experts, (size_t)EXPERTS * EXPERT_RAW));
#ifdef FAST_ROUTER
    CHECK(cudaMalloc(&w.approx, (size_t)N * EXPERTS * 4)); CHECK(cudaMalloc(&w.cand_score, (size_t)N * CAND * 4));
    CHECK(cudaMalloc(&w.cand, (size_t)N * CAND * 4)); CHECK(cudaMalloc(&w.ncand, (size_t)N * 4)); CHECK(cudaMalloc(&w.overflow, 4));
    CUB(cublasCreate(&w.cb));
#endif
    w.h_selected.resize((size_t)N * TOP); w.h_slot_token.resize(w.SLOTS); w.h_slot_of.resize((size_t)N * TOP); w.h_start.resize(EXPERTS); w.h_count.resize(EXPERTS);
}

// ---------------------------------------------------------------- the pieces of a layer
// KDA attention from the normalised rows w.Xn into `res` (N x W): q/k/v with the convolution, decay, the recurrent state, the gate and the output matrix
static void kda_attention(Work &w, const Layer &y, float *res)
{
    const int N = w.N; const Trunk &T = y.T;
    CHECK(cudaMemset(w.state, 0, (size_t)HEADS * HEAD * HEAD * 4));
    for (int c = 0; c < 3; c++)
        gemm_raw(w.Xn, N, W, T.op + QKV_PREFIX + (size_t)c * QKV_COMP + 20, (size_t)3 * QKV_COMP, T.op + QKV_PREFIX + (size_t)c * QKV_COMP, (size_t)3 * QKV_COMP, ROWS, w.raw[c], ROWS);
    for (int c = 0; c < 3; c++) conv_silu<<<blocks((size_t)N * ROWS), 256>>>(w.qkv[c], w.raw[c], T.op, c, N);
    l2_heads<<<N * HEADS, HEAD>>>(w.qkv[0]); l2_heads<<<N * HEADS, HEAD>>>(w.qkv[1]);
    gemm(w.Xn, N, W, T, 11, w.beta); beta_sigmoid<<<blocks((size_t)N * HEADS), 256>>>(w.beta, (size_t)N * HEADS);
    gemm(w.Xn, N, W, T, 12, w.dhid); gemm(w.dhid, N, HEAD, T, 13, w.decay);
    decay_transform<<<blocks((size_t)N * ROWS), 256>>>(w.decay, rec_data(T, 39), rec_data(T, 18), N);
    recurrence<<<HEADS, HEAD>>>(w.state, w.qkv[0], w.qkv[1], w.qkv[2], w.decay, w.beta, w.attn, N);
    gemm(w.Xn, N, W, T, 6, w.gate);
    normalize_rows<<<N * HEADS, HEAD>>>(w.attn, w.attn, rec_data(T, 19), HEAD);
    gate_mul<<<blocks((size_t)N * ROWS), 256>>>(w.gate, w.attn, (size_t)N * ROWS);
    gemm(w.gate, N, ROWS, T, 7, res);
}

static void mla_project(const float *A, int N, const Layer &y, int slot, float *C)
{
    const Record &r = y.mr[slot];
    const unsigned char *d = mla_rec(y, slot);
    gemm_raw(A, N, (int)r.cols, d + (size_t)r.rows * 4, r.cols, d, 4, (int)r.rows, C, (int)r.rows);
}

// MLA attention from w.Xn into `res`
static void mla_block(Work &w, const Layer &y, float *res)
{
    const int N = w.N; const Trunk &T = y.T;
    mla_project(w.Xn, N, y, 20, w.mqlat);
    normalize_rows<<<N, 256>>>(w.mqlat, w.mqlat, mla_rec(y, 21), MLA_QLAT);
    mla_project(w.mqlat, N, y, 22, w.mq);
    mla_project(w.Xn, N, y, 23, w.mkv);
    normalize_prefix<<<N, 256>>>(w.mkv, mla_rec(y, 24), MLA_KV, MLA_KVLAT);
    copy_prefix<<<blocks((size_t)N * MLA_KV), 256>>>(w.mkvn, w.mkv, MLA_KV, MLA_KVLAT, (size_t)N);
    mla_project(w.mkvn, N, y, 25, w.mexp);
    const size_t shared = (size_t)N * 4;
    if (shared > 48 * 1024) CHECK(cudaFuncSetAttribute(mla_attention, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared));
    mla_attention<<<dim3(N, HEADS), 128, shared>>>(w.attn, w.mq, w.mexp, w.mkv, N);
    gemm(w.Xn, N, W, T, 6, w.gate);
    gate_mul<<<blocks((size_t)N * ROWS), 256>>>(w.gate, w.attn, (size_t)N * ROWS);
    gemm(w.gate, N, ROWS, T, 7, res);
}

static void aggregate_with(Work &w, const float *const *snaps, int nsnap, const float *last, const unsigned char *vec)
{
    Srcs s; for (int i = 0; i < nsnap; i++) s.p[i] = snaps[i];
    s.p[nsnap] = last;
    aggregate_multi<<<w.N, 256>>>(w.Agg, s, nsnap + 1, vec);
}

// the router with every score summed in double (selected, weights, h_selected)
static void route_all_double(Work &w, const Layer &y)
{
    const int N = w.N; const Trunk &T = y.T;
    CHECK(cudaMemset(w.bad, 0, 4));
    transpose_rows<<<blocks((size_t)N * W), 256>>>(w.post_t, w.post, N, W);
    router_scores<<<dim3((N + 127) / 128, EXPERTS), 128>>>(w.scores, w.choices, rec_data(T, 29), rec_data(T, 30), w.post_t, N);
    top16<<<blocks(N), 256>>>(w.scores, w.choices, w.selected, w.weights, w.bad, N);
    CHECK(cudaMemcpy(w.h_selected.data(), w.selected, w.h_selected.size() * 4, cudaMemcpyDeviceToHost));
    { int h_bad = 0; CHECK(cudaMemcpy(&h_bad, w.bad, 4, cudaMemcpyDeviceToHost)); if (h_bad) { fprintf(stderr, "layer %d: non-finite router choice for %d tokens\n", y.L, h_bad); exit(1); } }
}

// the mixture of experts for layer y.L from w.post, added into `res`; experts of this layer must be in w.d_experts
static void moe_block(Work &w, const Layer &y, float *res)
{
    const int N = w.N, SLOTS = w.SLOTS; const Trunk &T = y.T;
#ifdef FAST_ROUTER
    { const float one_f = 1.0f, zero_f = 0.0f;
      CUB(cublasSgemm(w.cb, CUBLAS_OP_T, CUBLAS_OP_N, EXPERTS, N, W, &one_f, (const float *)rec_data(T, 29), W, w.post, W, &zero_f, w.approx, EXPERTS)); }
    CHECK(cudaMemset(w.overflow, 0, 4));
    router_candidates<<<blocks(N), 256>>>(w.approx, rec_data(T, 30), w.cand, w.ncand, w.overflow, N);
    router_exact<<<blocks((size_t)N * CAND), 256>>>(w.cand_score, w.cand, w.ncand, rec_data(T, 29), w.post, N);
    top16_candidates<<<blocks(N), 256>>>(w.cand_score, w.cand, w.ncand, rec_data(T, 30), w.selected, w.weights, N);
    CHECK(cudaMemcpy(w.h_selected.data(), w.selected, w.h_selected.size() * 4, cudaMemcpyDeviceToHost));
    { int h_over = 0; CHECK(cudaMemcpy(&h_over, w.overflow, 4, cudaMemcpyDeviceToHost));
      if (h_over) { fprintf(stderr, "layer %d: %d tokens have more than %d router candidates, this layer is routed by the all-double router\n", y.L, h_over, (int)CAND); route_all_double(w, y); } }
#else
    route_all_double(w, y);
#endif
    std::fill(w.h_count.begin(), w.h_count.end(), 0);
    for (int i = 0; i < N * TOP; i++) w.h_count[w.h_selected[i]]++;
    {   // MODEL_GPU_ROUTES=dir: write each layer's chosen experts (N x 16 ints) to dir/routes-L.bin, to compare two runs
        const char *dir = getenv("MODEL_GPU_ROUTES");
        if (dir && *dir) {
            char name[4096]; snprintf(name, sizeof name, "%s/routes-%d.bin", dir, y.L);
            FILE *f = fopen(name, "wb");
            if (f) { fwrite(w.h_selected.data(), 4, w.h_selected.size(), f); fclose(f); }
        }
    }
    { int acc = 0; for (int e = 0; e < EXPERTS; e++) { w.h_start[e] = acc; acc += w.h_count[e]; } }
    { std::vector<int> fill(EXPERTS, 0);
      for (int n = 0; n < N; n++) for (int r = 0; r < TOP; r++) { const int e = w.h_selected[n * TOP + r]; const int slot = w.h_start[e] + fill[e]++; w.h_slot_token[slot] = n; w.h_slot_of[n * TOP + r] = slot; } }
    CHECK(cudaMemcpy(w.d_slot_token, w.h_slot_token.data(), SLOTS * 4, cudaMemcpyHostToDevice)); CHECK(cudaMemcpy(w.d_slot_of, w.h_slot_of.data(), (size_t)N * TOP * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(w.d_start, w.h_start.data(), EXPERTS * 4, cudaMemcpyHostToDevice)); CHECK(cudaMemcpy(w.d_count, w.h_count.data(), EXPERTS * 4, cudaMemcpyHostToDevice));
    gemm(w.post, N, W, T, 31, w.latent);
    EXPERT_LAUNCH(48, w.d_experts, y.d_mapvals, y.d_refs, y.d_ref_base, 0, w.latent, LATENT, w.d_slot_token, w.d_start, w.d_count, w.G, EXPERT);
    EXPERT_LAUNCH(48, w.d_experts, y.d_mapvals, y.d_refs, y.d_ref_base, 1, w.latent, LATENT, w.d_slot_token, w.d_start, w.d_count, w.U, EXPERT);
    dense_activation<<<blocks((size_t)SLOTS * EXPERT), 256>>>(w.G, w.U, (size_t)SLOTS * EXPERT);
    EXPERT_LAUNCH(56, w.d_experts, y.d_mapvals, y.d_refs, y.d_ref_base, 2, w.G, EXPERT, w.d_slot_token, w.d_start, w.d_count, w.D, LATENT);
    mix_ranks<<<blocks((size_t)N * LATENT), 256>>>(w.mixture, w.D, w.d_slot_of, w.weights, N);
    normalize_rows<<<N, 256>>>(w.lnorm, w.mixture, rec_data(T, 33), LATENT);
    gemm(w.lnorm, N, LATENT, T, 32, w.routed);
    gemm(w.post, N, W, T, 34, w.sg); gemm(w.post, N, W, T, 35, w.su);
    dense_activation<<<blocks((size_t)N * SHARED), 256>>>(w.sg, w.su, (size_t)N * SHARED);
    gemm(w.sg, N, SHARED, T, 36, w.so);
    final_sum<<<blocks((size_t)N * W), 256>>>(res, w.routed, w.so, (size_t)N * W);
}

// layer 0: attention, aggregate with the input as the only snapshot, the dense MLP
static void run_layer0(Work &w, const Layer &y, const float *in, float *out)
{
    const int N = w.N; const Trunk &T = y.T;
    normalize_rows<<<N, 256>>>(w.Xn, in, y.input_gains, W);
    kda_attention(w, y, out);
    const float *sn[1] = { in };
    aggregate_with(w, sn, 1, out, rec_data(T, 38));
    normalize_rows<<<N, 256>>>(w.post, w.Agg, rec_data(T, 5), W);
    gemm(w.post, N, W, T, 26, w.dg); gemm(w.post, N, W, T, 27, w.du);
    dense_activation<<<blocks((size_t)N * DENSE), 256>>>(w.dg, w.du, (size_t)N * DENSE);
    gemm(w.dg, N, DENSE, T, 28, w.dout);
    add_into<<<blocks((size_t)N * W), 256>>>(out, w.dout, (size_t)N * W);
}

// layers 1..92. snaps[0..nsnap_in) are the snapshots the layer receives; for a layer number divisible by 12 its input becomes one more (the caller keeps it).
static void run_layer(Work &w, const Layer &y, const float *in, const float *const *snaps, int nsnap_in, float *out)
{
    const int N = w.N, L = y.L; const Trunk &T = y.T;
    aggregate_with(w, snaps, nsnap_in, in, rec_data(T, 37));
    normalize_rows<<<N, 256>>>(w.Xn, w.Agg, y.input_gains, W);
    if (y.mla) mla_block(w, y, out); else kda_attention(w, y, out);
    if (L % 12 != 0) add_into<<<blocks((size_t)N * W), 256>>>(out, in, (size_t)N * W);
    const float *sn[9]; int ns = nsnap_in;
    for (int i = 0; i < nsnap_in; i++) sn[i] = snaps[i];
    if (L % 12 == 0) sn[ns++] = in;
    aggregate_with(w, sn, ns, out, rec_data(T, 38));
    normalize_rows<<<N, 256>>>(w.post, w.Agg, rec_data(T, 5), W);
    moe_block(w, y, out);
}

// the experts file of layer L, to the card's expert buffer in chunks through a pinned buffer
static double upload_experts(const char *dataset, int L, unsigned char *dev, unsigned char *pinned, size_t chunk)
{
    char path[4096];
    snprintf(path, sizeof path, "%s/expert-%d/experts.direct", dataset, L);
    const size_t total = (size_t)EXPERTS * EXPERT_RAW;
    int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    struct timespec a, b; clock_gettime(CLOCK_MONOTONIC, &a);
    for (size_t off = 0; off < total; off += chunk) {
        const size_t n = total - off < chunk ? total - off : chunk;
        size_t got = 0;
        while (got < n) { ssize_t r = pread(fd, pinned + got, n - got, (off_t)(off + got)); if (r <= 0) { fprintf(stderr, "read of %s failed\n", path); exit(1); } got += (size_t)r; }
        CHECK(cudaMemcpy(dev + off, pinned, n, cudaMemcpyHostToDevice));
    }
    close(fd);
    clock_gettime(CLOCK_MONOTONIC, &b);
    return (b.tv_sec - a.tv_sec) + 1e-9 * (b.tv_nsec - a.tv_nsec);
}

// how a window of rows differs from the reference rows
static void compare_rows(const char *what, const float *dev, const float *ref, int N)
{
    std::vector<float> h((size_t)N * W);
    CHECK(cudaMemcpy(h.data(), dev, h.size() * 4, cudaMemcpyDeviceToHost));
    size_t same = 0, nonfinite = 0; double maxabs = 0, sumd2 = 0, sumr2 = 0, min_cos = 1.0;
    for (int n = 0; n < N; n++) {
        double dot = 0, a2 = 0, b2 = 0;
        for (int c = 0; c < W; c++) {
            const float x = h[(size_t)n * W + c], y = ref[(size_t)n * W + c];
            if (!std::isfinite(x)) nonfinite++;
            same += memcmp(&x, &y, 4) == 0;
            const double d = (double)x - (double)y;
            if (fabs(d) > maxabs) maxabs = fabs(d);
            sumd2 += d * d; sumr2 += (double)y * y; dot += (double)x * y; a2 += (double)x * x; b2 += (double)y * y;
        }
        const double cs = dot / (sqrt(a2) * sqrt(b2) + 1e-300);
        if (cs < min_cos) min_cos = cs;
    }
    printf("   %s: bit-identical %.2f%%, non-finite %zu, max abs difference %.3g, relative RMS difference %.3g, lowest cosine of a position %.9f\n", what,
           100.0 * same / ((double)N * W), nonfinite, maxabs, sqrt(sumd2 / (sumr2 + 1e-300)), min_cos);
}
