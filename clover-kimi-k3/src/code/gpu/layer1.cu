// Layer 1 of Kimi K3 on one GPU: the trunk and all 896 experts of the layer in video memory, for a window of tokens at once,
// compared with the CPU engine's layer 1.
//
//   layer1-gpu DATASET_DIR EMBED_F32 LAYER0_OUT_F32 REF_F32 [REPEATS]
//
// DATASET_DIR  the dataset folder (trunk-1/, expert-1/)
// EMBED_F32    layer 0's input rows (the embeddings), N x 7168 floats: PF_SAVE_ROWS=base gives base.embed
// LAYER0_OUT   layer 0's output rows = layer 1's input: base.layer0
// REF_F32      the CPU engine's layer 1 output rows: base.layer1
//
// Follows src/code/cpu/layer-kda.c for a KDA layer (layer 1): aggregate of the embedding and the input (score vector 37), the same attention as
// layer 0, aggregate of the embedding and the residual (score vector 38), then the mixture of experts: router (scores in double, top 16 in index
// order), latent projection, 16 experts per token (4-bit codes with one map per group of 32, three matrices), the weighted sum in rank order,
// the latent norm and projection back, the shared expert, and the residual sum.
// Differences from the CPU: matrix sums in float with fused multiply-add and a different order (the CPU sums the experts in double).
//
// build: nvcc -O3 --fmad=false -arch=sm_86 -o layer1-gpu layer1.cu
#include "kernels.cuh"
#include <algorithm>
#include <string>
#include <fcntl.h>
#include <unistd.h>

enum { LATENT = 3584, EXPERT = 3072, SHARED = 6144, EXPERT_RAW = 17547264, EXP_BLOCK01 = 121856, EXP_BLOCK2 = 104448, EXPERTS = 896, TOP = 16 };

// ---------------------------------------------------------------- the router
__global__ void transpose_rows(float *out, const float *in, int N, int width)   // out[c * N + n] = in[n * width + c]
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)N * width) return;
    const int n = (int)(i / width), c = (int)(i % width);
    out[(size_t)c * N + n] = in[i];
}

// score of expert e for token n: sigmoid((float) sum over c of (double)w[e][c] * (double)x[n][c]), summed in index order like the CPU
__global__ void router_scores(float *scores, float *choices, const unsigned char *router, const unsigned char *bias, const float *post_t, int N)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x, e = blockIdx.y;
    if (n >= N) return;
    const float *w = (const float *)router + (size_t)e * W;
    double s = 0.0;
    for (int c = 0; c < W; c++) s += (double)w[c] * (double)post_t[(size_t)c * N + n];
    const float sc = sigmoidf_((float)s);
    scores[(size_t)n * EXPERTS + e] = sc;
    choices[(size_t)n * EXPERTS + e] = sc + f32at(bias + (size_t)e * 4);
}

// the 16 best choices in index order (a tie goes to the lower expert), and their weights
__global__ void top16(const float *scores, const float *choices, int *selected, float *weights, int *bad, int N)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const float *sc = scores + (size_t)n * EXPERTS, *ch = choices + (size_t)n * EXPERTS;
    float best[TOP]; int sel[TOP]; int count = 0;
    for (int e = 0; e < EXPERTS; e++) {
        const float choice = ch[e];
        if (!isfinite(choice)) { atomicAdd(bad, 1); return; }
        if (count == TOP && !(choice > best[TOP - 1])) continue;
        int pos = count < TOP ? count : TOP - 1;
        while (pos && choice > best[pos - 1]) { best[pos] = best[pos - 1]; sel[pos] = sel[pos - 1]; pos--; }
        best[pos] = choice; sel[pos] = e;
        if (count < TOP) count++;
    }
    double total = 0.0;
    for (int r = 0; r < TOP; r++) total += (double)sc[sel[r]];
    const float inverse = (float)(1.0 / (total + 1e-20));
    for (int r = 0; r < TOP; r++) { selected[n * TOP + r] = sel[r]; weights[n * TOP + r] = sc[sel[r]] * inverse; }
}

// ---------------------------------------------------------------- the experts
// One block = 64 rows of one matrix of one expert, for every token routed to that expert, 32 tokens at a time.
// matrix 0 (gate) and 1 (up): input = the token's latent (3584), output 3072 rows. matrix 2 (down): input = that slot's activation (3072), output 3584 rows.
// A row holds 4-bit codes (low nibble = even column) and one selector byte per 32 columns; the selector picks a 16-value map through the template of the matrix.
__global__ void __launch_bounds__(256) expert_proj(const unsigned char *direct, const float *mapvals, const uint16_t *refs, const uint32_t *ref_base, int matrix,
    const float *X, int x_stride, const int *slot_token, const int *expert_start, const int *expert_count, float *O, int o_stride)
{
    const int e = blockIdx.y, rb = blockIdx.x;
    const int count = expert_count[e];
    if (!count) return;
    const int start = expert_start[e];
    const int K = matrix == 2 ? EXPERT : LATENT;
    const unsigned char *block = direct + (size_t)e * EXPERT_RAW + (matrix < 2 ? (size_t)(matrix * 48 + rb) * EXP_BLOCK01 : (size_t)96 * EXP_BLOCK01 + (size_t)rb * EXP_BLOCK2);
    const unsigned char *sel_base = block + 64 * K / 2;
    const uint32_t rbase = ref_base[e * 3 + matrix];
    __shared__ float Ws[64][65], Xs[64][33];
    const int tid = threadIdx.x, tx = tid & 31, ty = tid >> 5;
    for (int t0 = 0; t0 < count; t0 += 32) {
        float acc[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) acc[i] = 0.0f;
        for (int k0 = 0; k0 < K; k0 += 64) {
            {   // 64 rows x 64 columns of weights: 4 threads per row, 16 columns each
                const int row = tid >> 2, part = tid & 3, col = k0 + part * 16;
                const uint2 bytes = *(const uint2 *)(block + (size_t)row * (K / 2) + col / 2);
                const unsigned sel = sel_base[(size_t)row * (K / 32) + col / 32];
                const float *vals = mapvals + (size_t)refs[rbase + sel] * 16;
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    const unsigned b = ((j < 4 ? bytes.x : bytes.y) >> ((j & 3) * 8)) & 0xff;
                    Ws[part * 16 + 2 * j][row] = vals[b & 15];
                    Ws[part * 16 + 2 * j + 1][row] = vals[b >> 4];
                }
            }
            {   // 32 tokens x 64 inputs
                const int tk = tid >> 3, kp = (tid & 7) * 8;
                float v[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
                if (t0 + tk < count) {
                    const int slot = start + t0 + tk;
                    const float *src = X + (size_t)(matrix == 2 ? slot : slot_token[slot]) * x_stride + k0 + kp;
                    const float4 a = *(const float4 *)src, b = *(const float4 *)(src + 4);
                    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
                }
                #pragma unroll
                for (int j = 0; j < 8; j++) Xs[kp + j][tk] = v[j];
            }
            __syncthreads();
            #pragma unroll 8
            for (int kk = 0; kk < 64; kk++) {
                const float x = Xs[kk][tx];
                #pragma unroll
                for (int i = 0; i < 8; i++) acc[i] = fmaf(Ws[kk][ty * 8 + i], x, acc[i]);
            }
            __syncthreads();
        }
        if (t0 + tx < count) {
            float *o = O + (size_t)(start + t0 + tx) * o_stride + rb * 64 + ty * 8;
            #pragma unroll
            for (int i = 0; i < 8; i++) o[i] = acc[i];
        }
    }
}

// mixture[n][c] = sum over ranks in order of weight * down[slot of (n, rank)][c]
__global__ void mix_ranks(float *mixture, const float *down, const int *slot_of, const float *weights, int N)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)N * LATENT) return;
    const int n = (int)(i / LATENT), c = (int)(i % LATENT);
    float m = 0.0f;
    for (int r = 0; r < TOP; r++) m = m + weights[n * TOP + r] * down[(size_t)slot_of[n * TOP + r] * LATENT + c];
    mixture[i] = m;
}

__global__ void final_sum(float *res, const float *routed, const float *shared, size_t count)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) res[i] = res[i] + (routed[i] + shared[i]);
}

// ---------------------------------------------------------------- host
static uint16_t u16(const unsigned char *p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static unsigned json_count(const std::vector<unsigned char> &text, const char *key, int *found)
{
    std::string s((const char *)text.data(), text.size());
    const size_t at = s.find(std::string("\"") + key + "\"");
    if (at == std::string::npos) { *found = 0; return 0; }
    const size_t colon = s.find(':', at);
    *found = 1;
    return (unsigned)strtoul(s.c_str() + colon + 1, NULL, 10);
}
static double seconds(const struct timespec &a, const struct timespec &b) { return (b.tv_sec - a.tv_sec) + 1e-9 * (b.tv_nsec - a.tv_nsec); }

int main(int argc, char **argv)
{
    if (argc < 5) { fprintf(stderr, "usage: layer1-gpu DATASET_DIR EMBED_F32 LAYER0_OUT_F32 REF_F32 [REPEATS]\n"); return 2; }
    const int repeats = argc > 5 ? atoi(argv[5]) : 3;
    char path[4096];
    CHECK(cudaSetDevice(0));
    cudaDeviceProp prop; CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("device 0: %s, %.1f GiB\n", prop.name, prop.totalGlobalMem / 1073741824.0);
    struct timespec c0, c1;

    // ---- trunk-1
    Trunk T; memset(&T, 0, sizeof T);
    snprintf(path, sizeof path, "%s/trunk-1/index.bin", argv[1]);
    std::vector<unsigned char> index = slurp(path);
    if (index.size() < 1168 || memcmp(index.data(), "K3TRK001", 8)) { fprintf(stderr, "trunk-1/index.bin is not a trunk index\n"); return 1; }
    const unsigned count = u32(index.data() + 16);
    if (u32(index.data() + 12) != 1) { fprintf(stderr, "trunk-1 is not layer 1\n"); return 1; }
    for (unsigned p = 0; p < count; p++) {
        const unsigned char *e = index.data() + 1168 + (size_t)p * 72;
        Record r = { u32(e), u32(e + 4), u32(e + 8), u32(e + 12), u32(e + 16), u64(e + 24), u64(e + 32) };
        if (r.id >= 64 || r.group > 2) { fprintf(stderr, "bad record\n"); return 1; }
        T.rec[r.id] = r;
    }
    const char *names[3] = { "common.bin", "attention.bin", "routed-shared.bin" };
    size_t uploaded = 0;
    clock_gettime(CLOCK_MONOTONIC, &c0);
    for (int g = 0; g < 3; g++) {
        const size_t want = (size_t)u64(index.data() + 24 + g * 8);
        if (!want) continue;
        snprintf(path, sizeof path, "%s/trunk-1/%s", argv[1], names[g]);
        std::vector<unsigned char> bytes = slurp(path);
        if (bytes.size() != want) { fprintf(stderr, "%s has the wrong size\n", path); return 1; }
        CHECK(cudaMalloc(&T.group[g], bytes.size()));
        CHECK(cudaMemcpy(T.group[g], bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
        uploaded += bytes.size();
    }
    snprintf(path, sizeof path, "%s/trunk-1/operator.bin", argv[1]);
    std::vector<unsigned char> opbytes = slurp(path);
    if (opbytes.size() != (size_t)QKV_PREFIX + (size_t)ROWS * 3 * QKV_COMP || memcmp(opbytes.data(), "K3QKV001", 8) || u32(opbytes.data() + 12) != 1) { fprintf(stderr, "trunk-1/operator.bin is not what the engine reads\n"); return 1; }
    CHECK(cudaMalloc(&T.op, opbytes.size()));
    CHECK(cudaMemcpy(T.op, opbytes.data(), opbytes.size(), cudaMemcpyHostToDevice));
    uploaded += opbytes.size();
    clock_gettime(CLOCK_MONOTONIC, &c1);
    printf("trunk on the card: %.2f GB in %.2f s\n", uploaded / 1e9, seconds(c0, c1));

    // ---- expert-1: the tables and all 896 experts
    snprintf(path, sizeof path, "%s/expert-1/constants.bin", argv[1]);
    std::vector<unsigned char> constants = slurp(path);
    snprintf(path, sizeof path, "%s/expert-1/maps.json", argv[1]);
    std::vector<unsigned char> json = slurp(path);
    int f1, f2, f3;
    const unsigned map_count = json_count(json, "unique_maps", &f1), template_count = json_count(json, "unique_templates", &f2), ref_count = json_count(json, "template_refs", &f3);
    if (!f1 || !f2 || !f3) { fprintf(stderr, "maps.json lacks a count\n"); return 1; }
    const size_t other = (size_t)map_count * 16 + ((size_t)template_count + 1) * 2 + (size_t)ref_count * 2;
    if (constants.size() <= other || (constants.size() - other) % 2) { fprintf(stderr, "constants.bin does not fit maps.json\n"); return 1; }
    const unsigned palette_count = (unsigned)((constants.size() - other) / 2);
    std::vector<float> palette(palette_count);
    for (unsigned i = 0; i < palette_count; i++) { uint32_t bits = (uint32_t)u16(constants.data() + i * 2) << 16; memcpy(&palette[i], &bits, 4); }
    const unsigned char *maps = constants.data() + palette_count * 2, *offsets = maps + map_count * 16, *refs_p = offsets + (template_count + 1) * 2;
    std::vector<float> mapvals((size_t)map_count * 16);
    for (unsigned m = 0; m < map_count; m++) for (unsigned n = 0; n < 16; n++) mapvals[m * 16 + n] = maps[m * 16 + n] < palette_count ? palette[maps[m * 16 + n]] : 0.0f;
    snprintf(path, sizeof path, "%s/expert-1/experts.bin", argv[1]);
    FILE *ef = fopen(path, "rb");
    if (!ef) { fprintf(stderr, "cannot open %s\n", path); return 1; }
    std::vector<unsigned char> eindex(32 + 2688 * 2);
    if (fread(eindex.data(), 1, eindex.size(), ef) != eindex.size() || memcmp(eindex.data(), "K3MAPS01", 8)) { fprintf(stderr, "experts.bin is not an expert index\n"); return 1; }
    fclose(ef);
    std::vector<uint32_t> ref_base(2688);
    for (unsigned i = 0; i < 2688; i++) { const unsigned t = u16(eindex.data() + 32 + i * 2); if (t >= template_count) { fprintf(stderr, "bad template\n"); return 1; } ref_base[i] = u16(offsets + t * 2); }
    float *d_mapvals; uint16_t *d_refs; uint32_t *d_ref_base;
    CHECK(cudaMalloc(&d_mapvals, mapvals.size() * 4)); CHECK(cudaMemcpy(d_mapvals, mapvals.data(), mapvals.size() * 4, cudaMemcpyHostToDevice));
    std::vector<uint16_t> refs(ref_count); for (unsigned i = 0; i < ref_count; i++) refs[i] = u16(refs_p + i * 2);
    CHECK(cudaMalloc(&d_refs, refs.size() * 2)); CHECK(cudaMemcpy(d_refs, refs.data(), refs.size() * 2, cudaMemcpyHostToDevice));
    CHECK(cudaMalloc(&d_ref_base, ref_base.size() * 4)); CHECK(cudaMemcpy(d_ref_base, ref_base.data(), ref_base.size() * 4, cudaMemcpyHostToDevice));
    printf("expert tables: %u palette values, %u maps, %u templates, %u references\n", palette_count, map_count, template_count, ref_count);

    snprintf(path, sizeof path, "%s/expert-1/experts.direct", argv[1]);
    const size_t expert_bytes = (size_t)EXPERTS * EXPERT_RAW;
    int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "cannot open %s\n", path); return 1; }
    off_t fsize = lseek(fd, 0, SEEK_END);
    if ((size_t)fsize != expert_bytes) { fprintf(stderr, "%s has %lld bytes, wanted %zu\n", path, (long long)fsize, expert_bytes); return 1; }
    unsigned char *d_direct; CHECK(cudaMalloc(&d_direct, expert_bytes));
    const size_t chunk = 256u << 20;
    unsigned char *pinned; CHECK(cudaMallocHost(&pinned, chunk));
    clock_gettime(CLOCK_MONOTONIC, &c0);
    for (size_t off = 0; off < expert_bytes; off += chunk) {
        const size_t n = expert_bytes - off < chunk ? expert_bytes - off : chunk;
        size_t got = 0;
        while (got < n) { ssize_t r = pread(fd, pinned + got, n - got, (off_t)(off + got)); if (r <= 0) { fprintf(stderr, "read failed\n"); return 1; } got += (size_t)r; }
        CHECK(cudaMemcpy(d_direct + off, pinned, n, cudaMemcpyHostToDevice));
    }
    clock_gettime(CLOCK_MONOTONIC, &c1);
    close(fd);
    printf("all 896 experts on the card: %.2f GB in %.2f s (%.2f GB/s, read from disk or page cache and copied)\n", expert_bytes / 1e9, seconds(c0, c1), expert_bytes / 1e9 / seconds(c0, c1));

    // ---- input rows
    std::vector<unsigned char> embed = slurp(argv[2]), in0 = slurp(argv[3]), ref = slurp(argv[4]);
    if (embed.size() % (W * 4) || in0.size() != embed.size() || ref.size() != embed.size()) { fprintf(stderr, "the row files do not fit\n"); return 1; }
    const int N = (int)(embed.size() / (W * 4));
    const int SLOTS = N * TOP;
    printf("%d tokens, %d expert slots\n", N, SLOTS);

    float *X0, *IN, *Agg, *Xn, *raw[3], *qkv[3], *beta, *dhid, *decay, *gate, *attn, *res, *post, *post_t, *scores, *choices, *weights, *latent, *G, *U, *D, *mixture, *lnorm, *routed, *sg, *su, *so, *state;
    int *selected, *bad, *d_slot_token, *d_slot_of, *d_start, *d_count;
    auto alloc = [&](float **p, size_t floats) { CHECK(cudaMalloc((void **)p, floats * 4)); };
    alloc(&X0, (size_t)N * W); alloc(&IN, (size_t)N * W); alloc(&Agg, (size_t)N * W); alloc(&Xn, (size_t)N * W);
    for (int c = 0; c < 3; c++) { alloc(&raw[c], (size_t)N * ROWS); alloc(&qkv[c], (size_t)N * ROWS); }
    alloc(&beta, (size_t)N * HEADS); alloc(&dhid, (size_t)N * HEAD); alloc(&decay, (size_t)N * ROWS); alloc(&gate, (size_t)N * ROWS); alloc(&attn, (size_t)N * ROWS);
    alloc(&res, (size_t)N * W); alloc(&post, (size_t)N * W); alloc(&post_t, (size_t)N * W);
    alloc(&scores, (size_t)N * EXPERTS); alloc(&choices, (size_t)N * EXPERTS); alloc(&weights, (size_t)N * TOP); alloc(&latent, (size_t)N * LATENT);
    alloc(&G, (size_t)SLOTS * EXPERT); alloc(&U, (size_t)SLOTS * EXPERT); alloc(&D, (size_t)SLOTS * LATENT); alloc(&mixture, (size_t)N * LATENT); alloc(&lnorm, (size_t)N * LATENT);
    alloc(&routed, (size_t)N * W); alloc(&sg, (size_t)N * SHARED); alloc(&su, (size_t)N * SHARED); alloc(&so, (size_t)N * W); alloc(&state, (size_t)HEADS * HEAD * HEAD);
    CHECK(cudaMalloc(&selected, (size_t)N * TOP * 4)); CHECK(cudaMalloc(&bad, 4));
    CHECK(cudaMalloc(&d_slot_token, (size_t)SLOTS * 4)); CHECK(cudaMalloc(&d_slot_of, (size_t)N * TOP * 4)); CHECK(cudaMalloc(&d_start, EXPERTS * 4)); CHECK(cudaMalloc(&d_count, EXPERTS * 4));
    CHECK(cudaMemcpy(X0, embed.data(), embed.size(), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(IN, in0.data(), in0.size(), cudaMemcpyHostToDevice));
    size_t free_b, total_b; CHECK(cudaMemGetInfo(&free_b, &total_b));
    printf("card memory in use: %.2f GiB of %.2f GiB\n", (total_b - free_b) / 1073741824.0, total_b / 1073741824.0);

    cudaEvent_t t0, t1; CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));
    const char *stage_name[] = { "aggregate + norm", "q/k/v matrices", "convolution + l2", "decay + beta", "recurrence", "gate + output matrix", "residual + aggregate + norm",
                                 "router + top 16 + lists", "latent + experts (gate, up, down)", "mix + latent norm + back projection", "shared expert + sum" };
    const int stages = 11;
    float stage_ms[16];
    std::vector<float> out((size_t)N * W);
    std::vector<int> h_selected((size_t)N * TOP), h_slot_token(SLOTS), h_slot_of((size_t)N * TOP), h_start(EXPERTS), h_count(EXPERTS);
    for (int rep = 0; rep < repeats; rep++) {
        CHECK(cudaMemset(state, 0, (size_t)HEADS * HEAD * HEAD * 4)); CHECK(cudaMemset(bad, 0, 4));
        CHECK(cudaDeviceSynchronize());
        struct timespec a, b; clock_gettime(CLOCK_MONOTONIC, &a);
        int s = 0;
        auto mark = [&](void) { CHECK(cudaEventRecord(t1)); CHECK(cudaEventSynchronize(t1)); float ms; CHECK(cudaEventElapsedTime(&ms, t0, t1)); stage_ms[s++] = ms; CHECK(cudaEventRecord(t0)); };
        CHECK(cudaEventRecord(t0));
        aggregate<<<N, 256>>>(Agg, X0, IN, rec_data(T, 37));
        normalize_rows<<<N, 256>>>(Xn, Agg, T.op + 88, W);                                                         mark();
        for (int c = 0; c < 3; c++) {
            dim3 grid((ROWS + BN - 1) / BN, (N + BM - 1) / BM);
            gemm_i8<<<grid, 256>>>(Xn, N, W, T.op + QKV_PREFIX + (size_t)c * QKV_COMP + 20, (size_t)3 * QKV_COMP, T.op + QKV_PREFIX + (size_t)c * QKV_COMP, (size_t)3 * QKV_COMP, ROWS, raw[c], ROWS);
        }                                                                                                          mark();
        for (int c = 0; c < 3; c++) conv_silu<<<blocks((size_t)N * ROWS), 256>>>(qkv[c], raw[c], T.op, c, N);
        l2_heads<<<N * HEADS, HEAD>>>(qkv[0]); l2_heads<<<N * HEADS, HEAD>>>(qkv[1]);                              mark();
        gemm(Xn, N, W, T, 11, beta); beta_sigmoid<<<blocks((size_t)N * HEADS), 256>>>(beta, (size_t)N * HEADS);
        gemm(Xn, N, W, T, 12, dhid); gemm(dhid, N, HEAD, T, 13, decay);
        decay_transform<<<blocks((size_t)N * ROWS), 256>>>(decay, rec_data(T, 39), rec_data(T, 18), N);            mark();
        recurrence<<<HEADS, HEAD>>>(state, qkv[0], qkv[1], qkv[2], decay, beta, attn, N);                           mark();
        gemm(Xn, N, W, T, 6, gate);
        normalize_rows<<<N * HEADS, HEAD>>>(attn, attn, rec_data(T, 19), HEAD);
        gate_mul<<<blocks((size_t)N * ROWS), 256>>>(gate, attn, (size_t)N * ROWS);
        gemm(gate, N, ROWS, T, 7, res);                                                                             mark();
        add_into<<<blocks((size_t)N * W), 256>>>(res, IN, (size_t)N * W);        // residual = input + attention output
        aggregate<<<N, 256>>>(Agg, X0, res, rec_data(T, 38));
        normalize_rows<<<N, 256>>>(post, Agg, rec_data(T, 5), W);                                                   mark();
        // router
        transpose_rows<<<blocks((size_t)N * W), 256>>>(post_t, post, N, W);
        router_scores<<<dim3((N + 127) / 128, EXPERTS), 128>>>(scores, choices, rec_data(T, 29), rec_data(T, 30), post_t, N);
        top16<<<blocks(N), 256>>>(scores, choices, selected, weights, bad, N);
        CHECK(cudaMemcpy(h_selected.data(), selected, h_selected.size() * 4, cudaMemcpyDeviceToHost));
        { int h_bad = 0; CHECK(cudaMemcpy(&h_bad, bad, 4, cudaMemcpyDeviceToHost)); if (h_bad) { fprintf(stderr, "non-finite router choice for %d tokens\n", h_bad); return 1; } }
        std::fill(h_count.begin(), h_count.end(), 0);
        for (int i = 0; i < N * TOP; i++) h_count[h_selected[i]]++;
        { int acc = 0; for (int e = 0; e < EXPERTS; e++) { h_start[e] = acc; acc += h_count[e]; } }
        { std::vector<int> fill(EXPERTS, 0);
          for (int n = 0; n < N; n++) for (int r = 0; r < TOP; r++) { const int e = h_selected[n * TOP + r]; const int slot = h_start[e] + fill[e]++; h_slot_token[slot] = n; h_slot_of[n * TOP + r] = slot; } }
        CHECK(cudaMemcpy(d_slot_token, h_slot_token.data(), SLOTS * 4, cudaMemcpyHostToDevice)); CHECK(cudaMemcpy(d_slot_of, h_slot_of.data(), (size_t)N * TOP * 4, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_start, h_start.data(), EXPERTS * 4, cudaMemcpyHostToDevice)); CHECK(cudaMemcpy(d_count, h_count.data(), EXPERTS * 4, cudaMemcpyHostToDevice));
                                                                                                                    mark();
        gemm(post, N, W, T, 31, latent);
        expert_proj<<<dim3(48, EXPERTS), 256>>>(d_direct, d_mapvals, d_refs, d_ref_base, 0, latent, LATENT, d_slot_token, d_start, d_count, G, EXPERT);
        expert_proj<<<dim3(48, EXPERTS), 256>>>(d_direct, d_mapvals, d_refs, d_ref_base, 1, latent, LATENT, d_slot_token, d_start, d_count, U, EXPERT);
        dense_activation<<<blocks((size_t)SLOTS * EXPERT), 256>>>(G, U, (size_t)SLOTS * EXPERT);
        expert_proj<<<dim3(56, EXPERTS), 256>>>(d_direct, d_mapvals, d_refs, d_ref_base, 2, G, EXPERT, d_slot_token, d_start, d_count, D, LATENT);   mark();
        mix_ranks<<<blocks((size_t)N * LATENT), 256>>>(mixture, D, d_slot_of, weights, N);
        normalize_rows<<<N, 256>>>(lnorm, mixture, rec_data(T, 33), LATENT);
        gemm(lnorm, N, LATENT, T, 32, routed);                                                                      mark();
        gemm(post, N, W, T, 34, sg); gemm(post, N, W, T, 35, su);
        dense_activation<<<blocks((size_t)N * SHARED), 256>>>(sg, su, (size_t)N * SHARED);
        gemm(sg, N, SHARED, T, 36, so);
        final_sum<<<blocks((size_t)N * W), 256>>>(res, routed, so, (size_t)N * W);                                  mark();
        CHECK(cudaGetLastError());
        clock_gettime(CLOCK_MONOTONIC, &b);
        printf("run %d: layer 1 for %d tokens in %.1f ms (", rep + 1, N, 1000.0 * seconds(a, b));
        for (int i = 0; i < stages; i++) printf("%s%s %.1f", i ? ", " : "", stage_name[i], stage_ms[i]);
        printf(" ms)\n");
    }
    { int used = 0, maxc = 0; for (int e = 0; e < EXPERTS; e++) { used += h_count[e] > 0; if (h_count[e] > maxc) maxc = h_count[e]; }
      printf("routing: %d of %d experts used, most tokens on one expert %d, mean per used expert %.1f\n", used, EXPERTS, maxc, used ? (double)SLOTS / used : 0.0); }
    CHECK(cudaMemcpy(out.data(), res, out.size() * 4, cudaMemcpyDeviceToHost));

    const float *r = (const float *)ref.data();
    size_t same = 0, nonfinite = 0; double maxabs = 0, sumd2 = 0, sumr2 = 0, worst_pos_err = 0, min_cos = 1.0; int worst_pos = 0;
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
    printf("against the CPU engine's layer 1 output (%d x %d values): bit-identical %zu (%.2f%%), non-finite %zu, max abs difference %.3g, relative RMS difference %.3g, "
           "lowest cosine of a position %.9f, worst position %d (relative RMS %.3g)\n", N, W, same, 100.0 * same / ((double)N * W), nonfinite, maxabs, sqrt(sumd2 / sumr2),
           min_cos, worst_pos, worst_pos_err);
    return 0;
}
