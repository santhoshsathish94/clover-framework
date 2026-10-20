// The whole model on one GPU, one layer after another, for a window of tokens: the correctness step before the eight-card pipeline.
//
//   model-gpu DATASET_DIR REFS_DIR LAST_LAYER
//
// DATASET_DIR  the dataset (trunk-N, expert-N)
// REFS_DIR     the pod's recorded values: embed.f32 (layer 0's input), final.snaps.f32 (8 snapshots), final.resid.f32 (layer 92's output), N x 7168 floats each
//              (src/work/inputs/refs); the window is as many rows as embed.f32 holds
// LAST_LAYER   0..92: run layers 0..LAST_LAYER. After layer 12k-1 the output is compared with the pod's snapshot k (k = 1..7), after layer 92 with its output.
//
// Each layer's trunk and experts are loaded when the layer comes (single card, nothing prefetched here). Prints the load and compute time of every layer.
// build: nvcc -O3 --fmad=false -arch=sm_86 -o model-gpu model-gpu.cu [-DFAST_ROUTER -DEXPERT_V2 -DINT8_PLANES -lcublas]
//
// One layer on its own, from the real activations of a full run:
//   MODEL_GPU_DUMP_DIR=dir   (full run, LAST_LAYER 92) writes in-L.f32 (the input of layer L, L = 1..93; in-93 is layer 92's output) and snap-k.f32 (the input of layer 12k)
//   MODEL_GPU_ONLY_LAYER=L MODEL_GPU_INPUTS=dir   with LAST_LAYER = L: runs only layer L from those files MODEL_GPU_REPEAT times (default 5), prints each compute time,
//                            and compares the output with the next layer's recorded input (the full run's own output of layer L)
#include "model.cuh"
#include <cuda_profiler_api.h>

static void write_rows(const char *dir, const char *name, int k, const float *dev, int N)
{
    std::vector<float> h((size_t)N * W);
    CHECK(cudaMemcpy(h.data(), dev, h.size() * 4, cudaMemcpyDeviceToHost));
    char p[4096]; snprintf(p, sizeof p, "%s/%s-%d.f32", dir, name, k);
    FILE *f = fopen(p, "wb"); if (!f) { fprintf(stderr, "cannot write %s\n", p); exit(1); }
    fwrite(h.data(), 4, h.size(), f); fclose(f);
}

int main(int argc, char **argv)
{
    if (argc < 4) { fprintf(stderr, "usage: model-gpu DATASET_DIR REFS_DIR LAST_LAYER\n"); return 2; }
    const char *dataset = argv[1];
    const int last = atoi(argv[3]);
    if (last < 0 || last > LAST_LAYER_INDEX) { fprintf(stderr, "LAST_LAYER must be 0..92\n"); return 2; }
    char path[4096];
    snprintf(path, sizeof path, "%s/embed.f32", argv[2]); std::vector<unsigned char> embed = slurp(path);
    snprintf(path, sizeof path, "%s/final.snaps.f32", argv[2]); std::vector<unsigned char> snaps_ref = slurp(path);
    snprintf(path, sizeof path, "%s/final.resid.f32", argv[2]); std::vector<unsigned char> resid_ref = slurp(path);
    const int N = (int)(embed.size() / (W * 4));
    if (embed.size() % (W * 4) || resid_ref.size() < (size_t)N * W * 4 || snaps_ref.size() < (size_t)N * W * 4 * 8) { fprintf(stderr, "the reference files do not fit\n"); return 1; }
    const float *ref_snaps = (const float *)snaps_ref.data();
    printf("%d tokens, layers 0..%d\n", N, last);

    CHECK(cudaSetDevice(0));
    Work w; work_alloc(w, 0, N, true, last >= 1);
    size_t free_b, total_b; CHECK(cudaMemGetInfo(&free_b, &total_b));
    printf("card 0 memory in use after the buffers%s: %.2f GiB of %.2f GiB\n", last >= 1 ? " and the expert buffer" : "", (total_b - free_b) / 1073741824.0, total_b / 1073741824.0);
    const size_t chunk = 256u << 20;
    unsigned char *pinned; CHECK(cudaMallocHost(&pinned, chunk));
    cudaEvent_t t0, t1; CHECK(cudaEventCreate(&t0)); CHECK(cudaEventCreate(&t1));
    CHECK(cudaMemcpy(w.snap[0], embed.data(), embed.size(), cudaMemcpyHostToDevice));
    struct timespec c0, c1;
    double total_compute = 0, total_load = 0;
    const char *dump = getenv("MODEL_GPU_DUMP_DIR"), *only = getenv("MODEL_GPU_ONLY_LAYER"), *inputs = getenv("MODEL_GPU_INPUTS");
    if (dump && !*dump) dump = nullptr;

    if (only && *only) {
        const int L = atoi(only);
        if (!inputs || !*inputs || L < 0 || L > LAST_LAYER_INDEX || last != L) { fprintf(stderr, "MODEL_GPU_ONLY_LAYER=L needs LAST_LAYER = L and MODEL_GPU_INPUTS=dir (a MODEL_GPU_DUMP_DIR of a full run)\n"); return 2; }
        auto rows_file = [&](const char *name, int k) {
            char p[4096]; snprintf(p, sizeof p, "%s/%s-%d.f32", inputs, name, k);
            std::vector<unsigned char> b = slurp(p);
            if (b.size() != (size_t)N * W * 4) { fprintf(stderr, "%s does not hold %d x %d floats\n", p, N, W); exit(1); }
            return b;
        };
        const int reps = getenv("MODEL_GPU_REPEAT") ? atoi(getenv("MODEL_GPU_REPEAT")) : 5;
        const char *pl = getenv("MODEL_GPU_PROFILE_LAYER");
        const bool prof = pl && *pl && atoi(pl) == L;
        Layer y; clock_gettime(CLOCK_MONOTONIC, &c0);
        layer_load_trunk(dataset, L, y);
        double t_experts = 0;
        if (L >= 1) { layer_load_expert_tables(dataset, L, y); t_experts = upload_experts(dataset, L, w.d_experts, pinned, chunk); }
        const int nsnap_in = snapshots_in(L > 0 ? L : 1);
        if (L >= 1) {
            std::vector<unsigned char> in = rows_file("in", L);
            CHECK(cudaMemcpy(w.cur, in.data(), in.size(), cudaMemcpyHostToDevice));
            for (int k = 1; k < nsnap_in; k++) { std::vector<unsigned char> s = rows_file("snap", k); CHECK(cudaMemcpy(w.snap[k], s.data(), s.size(), cudaMemcpyHostToDevice)); }
        }
        const float *sn[9]; for (int i = 0; i < nsnap_in; i++) sn[i] = w.snap[i];
        clock_gettime(CLOCK_MONOTONIC, &c1);
        printf("layer %d (%s) loaded in %.2f s (experts %.2f s)\n", L, L == 0 ? "dense" : y.mla ? "MLA" : "KDA", seconds(c0, c1), t_experts);
        std::vector<float> times;
        for (int r = 0; r < reps; r++) {
            if (prof && r == reps - 1) CHECK(cudaProfilerStart());
            CHECK(cudaEventRecord(t0));
            if (L == 0) run_layer0(w, y, w.snap[0], w.nxt); else run_layer(w, y, w.cur, sn, nsnap_in, w.nxt);
            CHECK(cudaEventRecord(t1)); CHECK(cudaEventSynchronize(t1)); CHECK(cudaGetLastError());
            if (prof && r == reps - 1) CHECK(cudaProfilerStop());
            float ms; CHECK(cudaEventElapsedTime(&ms, t0, t1)); times.push_back(ms);
            printf("  run %d: compute %.1f ms\n", r + 1, ms);
        }
        std::vector<float> sorted = times; std::sort(sorted.begin(), sorted.end());
        printf("layer %d, %d tokens: best %.1f ms, median %.1f ms of %d runs (the first run includes warm-up)\n", L, N, sorted.front(), sorted[sorted.size() / 2], reps);
        std::vector<unsigned char> ref = rows_file("in", L + 1);
        char name[96]; snprintf(name, sizeof name, "output of layer %d against the full run's", L);
        compare_rows(name, w.nxt, (const float *)ref.data(), N);
        return 0;
    }

    {   // layer 0 (or, with MODEL_GPU_LAYER0_ROWS=file, the CPU engine's layer 0 output rows as layer 1's input, to separate the layers' own differences from what they inherit)
        Layer y; clock_gettime(CLOCK_MONOTONIC, &c0);
        const char *l0rows = getenv("MODEL_GPU_LAYER0_ROWS");
        if (l0rows && *l0rows) {
            std::vector<unsigned char> rows = slurp(l0rows);
            if (rows.size() != embed.size()) { fprintf(stderr, "%s does not hold %d x %d floats\n", l0rows, N, W); return 1; }
            CHECK(cudaMemcpy(w.cur, rows.data(), rows.size(), cudaMemcpyHostToDevice));
            printf("layer  0: not run, the rows of %s are layer 1's input\n", l0rows);
        } else {
        layer_load_trunk(dataset, 0, y);
        clock_gettime(CLOCK_MONOTONIC, &c1);
        CHECK(cudaEventRecord(t0));
        run_layer0(w, y, w.snap[0], w.cur);
        CHECK(cudaEventRecord(t1)); CHECK(cudaEventSynchronize(t1)); CHECK(cudaGetLastError());
        float ms; CHECK(cudaEventElapsedTime(&ms, t0, t1));
        printf("layer  0 (dense)  trunk %.2f s, compute %.1f ms\n", seconds(c0, c1), ms);
        total_compute += ms; total_load += seconds(c0, c1);
        layer_free(y);
        }
    }
    int nsnap = 1;
    for (int L = 1; L <= last; L++) {
        Layer y;
        clock_gettime(CLOCK_MONOTONIC, &c0);
        layer_load_trunk(dataset, L, y);
        layer_load_expert_tables(dataset, L, y);
        clock_gettime(CLOCK_MONOTONIC, &c1);
        const double t_trunk = seconds(c0, c1);
        const double t_experts = upload_experts(dataset, L, w.d_experts, pinned, chunk);
        const float *sn[9];
        for (int i = 0; i < nsnap; i++) sn[i] = w.snap[i];
        if (dump) write_rows(dump, "in", L, w.cur, N);
        const char *pl = getenv("MODEL_GPU_PROFILE_LAYER");   // nsys profile --capture-range=cudaProfilerApi records only this layer's compute
        const bool prof = pl && *pl && atoi(pl) == L;
        if (prof) CHECK(cudaProfilerStart());
        CHECK(cudaEventRecord(t0));
        run_layer(w, y, w.cur, sn, nsnap, w.nxt);
        CHECK(cudaEventRecord(t1)); CHECK(cudaEventSynchronize(t1)); CHECK(cudaGetLastError());
        if (prof) CHECK(cudaProfilerStop());
        float ms; CHECK(cudaEventElapsedTime(&ms, t0, t1));
        printf("layer %2d (%s)  trunk %.2f s, experts %.2f s, compute %.1f ms\n", L, y.mla ? "MLA" : "KDA", t_trunk, t_experts, ms);
        total_compute += ms; total_load += t_trunk + t_experts;
        if (L % 12 == 0) { CHECK(cudaMemcpy(w.snap[nsnap], w.cur, (size_t)N * W * 4, cudaMemcpyDeviceToDevice)); if (dump) write_rows(dump, "snap", L / 12, w.snap[nsnap], N); nsnap++; }
        if ((L + 1) % 12 == 0 && (L + 1) / 12 <= 7) {
            char name[96]; snprintf(name, sizeof name, "output of layer %d against the pod's snapshot %d", L, (L + 1) / 12);
            compare_rows(name, w.nxt, ref_snaps + (size_t)((L + 1) / 12) * N * W, N);
        }
        std::swap(w.cur, w.nxt);
        layer_free(y);
    }
    if (dump) write_rows(dump, "in", last + 1, w.cur, N);
    printf("compute %.2f s in all, loading %.2f s in all\n", total_compute / 1000, total_load);
    if (argc > 4) {   // the CPU engine's output of layer LAST_LAYER for the same window (PF_LAST_LAYER=L PF_SAVE_ROWS=base gives base.layerL)
        std::vector<unsigned char> cpu = slurp(argv[4]);
        if (cpu.size() != (size_t)N * W * 4) { fprintf(stderr, "%s does not hold %d x %d floats\n", argv[4], N, W); return 1; }
        char name[96]; snprintf(name, sizeof name, "output of layer %d against the CPU engine's", last);
        compare_rows(name, w.cur, (const float *)cpu.data(), N);
    }
    if (last == LAST_LAYER_INDEX) {
        compare_rows("output of layer 92 against the pod's", w.cur, (const float *)resid_ref.data(), N);
        for (int k = 1; k < 8 && k < nsnap; k++) { char name[64]; snprintf(name, sizeof name, "snapshot %d against the pod's", k); compare_rows(name, w.snap[k], ref_snaps + (size_t)k * N * W, N); }
    } else {
        std::vector<float> h((size_t)N * W);
        CHECK(cudaMemcpy(h.data(), w.cur, h.size() * 4, cudaMemcpyDeviceToHost));
        unsigned long long fnv = 1469598103934665603ULL; const unsigned char *bytes = (const unsigned char *)h.data();
        for (size_t i = 0; i < h.size() * 4; i++) { fnv ^= bytes[i]; fnv *= 1099511628211ULL; }
        printf("output of layer %d: FNV-1a 64 %016llx\n", last, fnv);
    }
    return 0;
}
