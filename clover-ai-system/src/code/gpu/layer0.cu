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
#include "kernels.cuh"

int main(int argc, char **argv)
{
    if (argc < 4) { fprintf(stderr, "usage: layer0-gpu TRUNK0_DIR EMBED_F32 REF_F32 [REPEATS]\n"); return 2; }
    const int repeats = argc > 4 ? atoi(argv[4]) : 3;
    char path[4096];
    Trunk L; memset(&L, 0, sizeof L);

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
            gemm_raw(Xn, N, W, L.op + QKV_PREFIX + (size_t)c * QKV_COMP + 20, (size_t)3 * QKV_COMP, L.op + QKV_PREFIX + (size_t)c * QKV_COMP, (size_t)3 * QKV_COMP, ROWS, raw[c], ROWS);
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
