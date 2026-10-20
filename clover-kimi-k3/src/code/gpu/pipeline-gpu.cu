// The whole model through eight cards as a pipeline: each card holds the trunk of its own layers for good; only the experts rotate. The experts of the next layer
// of a card come from disk to pinned RAM (reader threads) to the card (one loader thread per card) while the other cards work; a window of tokens passes
// from the card of layer L to the card of layer L+1. The CPU does all the I/O.
//
//   pipeline-gpu DATASET_DIR REFS_DIR OPEN_WEIGHTS_DIR BUILD_DIR
// Settings (environment): PIPE_TOKENS (window, the first rows of the recorded prompt; default 500: 1000 does not fit beside the trunk), PIPE_SLOTS (pinned RAM
//   slots, one layer's experts each; default 8), PIPE_READERS (reader threads; default 4), PIPE_LAST (last layer, default 92), PIPE_MINGAP (the fewest layers
//   between two layers of one card, default 4), PIPE_CARDS (default 8).
// Checks at the end against the pod's recorded rows for the same positions: the output of layer 92 and snapshots 1 to 7; the next token from the CPU tail
// library (BUILD_DIR/pipeline-stage.so, OPEN_WEIGHTS_DIR) for the last position.
//
// build: nvcc -O3 --fmad=false -arch=sm_86 -o pipeline-gpu pipeline-gpu.cu -ldl -lpthread [-DFAST_ROUTER -DEXPERT_V2 -DINT8_PLANES -lcublas]
#include "model.cuh"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <dlfcn.h>

typedef struct { const unsigned char *bytes; size_t length; } Text;
static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static int env_int(const char *name, int dflt) { const char *v = getenv(name); return v && *v ? atoi(v) : dflt; }

struct LayerLog { double act_ready = -1, exp_ready = -1, start = -1, end = -1; int gpu = -1; };

static std::mutex mu;
static std::condition_variable cv;

int main(int argc, char **argv)
{
    if (argc < 5) { fprintf(stderr, "usage: pipeline-gpu DATASET_DIR REFS_DIR OPEN_WEIGHTS_DIR BUILD_DIR\n"); return 2; }
    const char *dataset = argv[1];
    const int LAST = env_int("PIPE_LAST", LAST_LAYER_INDEX), NTOK = env_int("PIPE_TOKENS", 500), SLOTS = env_int("PIPE_SLOTS", 8), READERS = env_int("PIPE_READERS", 4);
    const int MINGAP = env_int("PIPE_MINGAP", 4);
    int have = 0; CHECK(cudaGetDeviceCount(&have));
    const int G = env_int("PIPE_CARDS", 8);
    if (G < 1 || G > have || LAST < 1 || LAST > LAST_LAYER_INDEX) { fprintf(stderr, "bad PIPE_CARDS (%d cards present) or PIPE_LAST\n", have); return 2; }

    char path[4096];
    snprintf(path, sizeof path, "%s/embed.f32", argv[2]); std::vector<unsigned char> embed = slurp(path);
    snprintf(path, sizeof path, "%s/final.snaps.f32", argv[2]); std::vector<unsigned char> snaps_ref = slurp(path);
    snprintf(path, sizeof path, "%s/final.resid.f32", argv[2]); std::vector<unsigned char> resid_ref = slurp(path);
    const int NFULL = (int)(embed.size() / (W * 4));
    const int N = NTOK < NFULL ? NTOK : NFULL;
    const float *ref_snaps = (const float *)snaps_ref.data();
    printf("window %d tokens (of the recorded %d), layers 0..%d, %d cards, %d RAM slots, %d reader threads\n", N, NFULL, LAST, G, SLOTS, READERS);

    // ---- which card holds which layer: in order, to the card with the least trunk so far among those whose last layer is at least MINGAP back
    std::vector<size_t> trunk_bytes(LAST + 1, 0);
    for (int L = 0; L <= LAST; L++) {
        const char *files[6] = { "common.bin", "attention.bin", "routed-shared.bin", "operator.bin", "kda.bin", "dense.bin" };
        for (int f = 0; f < 6; f++) {
            snprintf(path, sizeof path, "%s/trunk-%d/%s", dataset, L, files[f]);
            FILE *fp = fopen(path, "rb"); if (!fp) continue;
            fseek(fp, 0, SEEK_END); trunk_bytes[L] += (size_t)ftell(fp); fclose(fp);
        }
    }
    std::vector<int> gpu_of(LAST + 1, 0), last_on(G, -1000);
    std::vector<size_t> load(G, 0);
    std::vector<std::vector<int>> owned(G);
    for (int L = 0; L <= LAST; L++) {
        int best = -1;
        for (int g = 0; g < G; g++) {
            if (L - last_on[g] < MINGAP) continue;
            if (best < 0 || load[g] < load[best]) best = g;
        }
        if (L == 0) best = 0;
        if (best < 0) { best = 0; for (int g = 1; g < G; g++) if (last_on[g] < last_on[best]) best = g; }
        gpu_of[L] = best; last_on[best] = L; load[best] += trunk_bytes[L]; owned[best].push_back(L);
    }
    for (int g = 0; g < G; g++) printf("card %d: %zu layers, trunk %.2f GiB\n", g, owned[g].size(), load[g] / 1073741824.0);

    // ---- buffers on every card
    std::vector<Work> works(G);
    {
        std::vector<std::thread> th;
        for (int g = 0; g < G; g++) th.emplace_back([&, g]() { work_alloc(works[g], g, N, g == gpu_of[0], true); });
        for (auto &t : th) t.join();
    }
    for (int g = 0; g < G; g++) { CHECK(cudaSetDevice(g)); size_t f, t; CHECK(cudaMemGetInfo(&f, &t)); printf("card %d: %.2f GiB in use after the buffers and the expert buffer, trunk still to come %.2f GiB, %.2f GiB would be left\n", g, (t - f) / 1073741824.0, load[g] / 1073741824.0, f / 1073741824.0 - load[g] / 1073741824.0); }

    // ---- trunk of every layer on its card, for good (parallel over cards)
    std::vector<Layer> layers(LAST + 1);
    const double t_trunk0 = now_s();
    {
        std::vector<std::thread> th;
        for (int g = 0; g < G; g++) th.emplace_back([&, g]() {
            CHECK(cudaSetDevice(g));
            for (int L : owned[g]) { layer_load_trunk(dataset, L, layers[L]); if (L >= 1) layer_load_expert_tables(dataset, L, layers[L]); }
        });
        for (auto &t : th) t.join();
    }
    printf("trunk of all %d layers on the cards: %.2f s\n", LAST + 1, now_s() - t_trunk0);
    for (int g = 0; g < G; g++) { CHECK(cudaSetDevice(g)); size_t f, t; CHECK(cudaMemGetInfo(&f, &t)); printf("card %d: %.2f GiB free\n", g, f / 1073741824.0); }
    // embeddings to every card (layer 0's input and snapshot 0)
    for (int g = 0; g < G; g++) { CHECK(cudaSetDevice(g)); CHECK(cudaMemcpy(works[g].snap[0], embed.data(), (size_t)N * W * 4, cudaMemcpyHostToDevice)); }

    // ---- RAM slots
    const size_t layer_bytes = (size_t)EXPERTS * EXPERT_RAW;
    std::vector<unsigned char *> slot_buf(SLOTS);
    const double t_pin0 = now_s();
    for (int s = 0; s < SLOTS; s++) CHECK(cudaMallocHost(&slot_buf[s], layer_bytes));
    printf("%d pinned RAM slots of %.2f GB: %.2f s\n", SLOTS, layer_bytes / 1e9, now_s() - t_pin0);

    // ---- shared state
    std::vector<int> moe_layers; for (int L = 1; L <= LAST; L++) moe_layers.push_back(L);
    std::vector<char> act_ready(LAST + 2, 0), exp_ready(LAST + 2, 0), buf_free(G, 1);
    std::vector<int> slot_layer(SLOTS, -1); std::vector<char> slot_ready(SLOTS, 0);
    std::vector<LayerLog> log(LAST + 2);
    std::atomic<int> next_read(0);
    std::atomic<double> bytes_read(0);
    const double t_start = now_s();
    double t_run0 = 0;

    // readers: disk to pinned RAM, layers in order
    std::vector<std::thread> threads;
    for (int r = 0; r < READERS; r++) threads.emplace_back([&]() {
        for (;;) {
            const int idx = next_read++;
            if (idx >= (int)moe_layers.size()) return;
            const int L = moe_layers[idx];
            int s = -1;
            { std::unique_lock<std::mutex> lk(mu);
              cv.wait(lk, [&]() { for (int i = 0; i < SLOTS; i++) if (slot_layer[i] == -1) return true; return false; });
              for (int i = 0; i < SLOTS; i++) if (slot_layer[i] == -1) { s = i; break; }
              slot_layer[s] = -2; }
            char p[4096]; snprintf(p, sizeof p, "%s/expert-%d/experts.direct", dataset, L);
            int fd = open(p, O_RDONLY);
            if (fd < 0) { fprintf(stderr, "cannot open %s\n", p); exit(1); }
            size_t got = 0;
            while (got < layer_bytes) { ssize_t k = pread(fd, slot_buf[s] + got, layer_bytes - got < (64u << 20) ? layer_bytes - got : (64u << 20), (off_t)got); if (k <= 0) { fprintf(stderr, "read of %s failed\n", p); exit(1); } got += (size_t)k; }
            close(fd);
            bytes_read = bytes_read + (double)layer_bytes;
            { std::lock_guard<std::mutex> lk(mu); slot_layer[s] = L; slot_ready[s] = 1; }
            cv.notify_all();
        }
    });
    // loaders: pinned RAM to the card, the card's next layer, once the card has finished its previous one
    for (int g = 0; g < G; g++) threads.emplace_back([&, g]() {
        CHECK(cudaSetDevice(g));
        cudaStream_t st; CHECK(cudaStreamCreate(&st));
        for (int L : owned[g]) {
            if (L == 0) continue;
            int s = -1;
            { std::unique_lock<std::mutex> lk(mu);
              cv.wait(lk, [&]() { if (!buf_free[g]) return false; for (int i = 0; i < SLOTS; i++) if (slot_layer[i] == L && slot_ready[i]) return true; return false; });
              for (int i = 0; i < SLOTS; i++) if (slot_layer[i] == L && slot_ready[i]) { s = i; break; }
              buf_free[g] = 0; }
            const size_t chunk = 256u << 20;
            for (size_t off = 0; off < layer_bytes; off += chunk) {
                const size_t n = layer_bytes - off < chunk ? layer_bytes - off : chunk;
                CHECK(cudaMemcpyAsync(works[g].d_experts + off, slot_buf[s] + off, n, cudaMemcpyHostToDevice, st));
            }
            CHECK(cudaStreamSynchronize(st));
            { std::lock_guard<std::mutex> lk(mu); slot_layer[s] = -1; slot_ready[s] = 0; exp_ready[L] = 1; log[L].exp_ready = now_s() - t_start; }
            cv.notify_all();
        }
    });
    // compute workers
    for (int g = 0; g < G; g++) threads.emplace_back([&, g]() {
        CHECK(cudaSetDevice(g));
        Work &w = works[g];
        for (int L : owned[g]) {
            { std::unique_lock<std::mutex> lk(mu); cv.wait(lk, [&]() { return act_ready[L] && (L == 0 || exp_ready[L]); }); }
            const double t0 = now_s();
            const float *sn[9]; const int nsnap = snapshots_in(L > 0 ? L : 1);
            for (int i = 0; i < nsnap && i < 8; i++) sn[i] = w.snap[i];
            if (L == 0) run_layer0(w, layers[0], w.snap[0], w.nxt);
            else run_layer(w, layers[L], w.cur, sn, nsnap, w.nxt);
            CHECK(cudaDeviceSynchronize()); CHECK(cudaGetLastError());
            const double t1 = now_s();
            { std::lock_guard<std::mutex> lk(mu); log[L].start = t0 - t_start; log[L].end = t1 - t_start; log[L].gpu = g; if (L >= 1) buf_free[g] = 1; }
            cv.notify_all();
            if (L < LAST) {   // pass the window on; the input of a layer divisible by 12 becomes a snapshot on every card
                const int g2 = gpu_of[L + 1];
                if (L >= 1 && L % 12 == 0) for (int h = 0; h < G; h++) CHECK(cudaMemcpyPeer(works[h].snap[L / 12], h, w.cur, g, (size_t)N * W * 4));
                CHECK(cudaMemcpyPeer(works[g2].cur, g2, w.nxt, g, (size_t)N * W * 4));
                CHECK(cudaDeviceSynchronize());
                CHECK(cudaSetDevice(g2)); CHECK(cudaDeviceSynchronize()); CHECK(cudaSetDevice(g));
            }
            { std::lock_guard<std::mutex> lk(mu); act_ready[L + 1] = 1; log[L + 1].act_ready = now_s() - t_start; }
            cv.notify_all();
        }
    });

    // the run starts when the experts of layers 1..8 (8 layers, as planned) are on their cards
    { std::unique_lock<std::mutex> lk(mu);
      cv.wait(lk, [&]() { for (int L = 1; L <= 8 && L <= LAST; L++) if (!exp_ready[L]) return false; return true; });
      t_run0 = now_s(); act_ready[0] = 1; log[0].act_ready = t_run0 - t_start; }
    const double t_pre = t_run0 - t_start;
    cv.notify_all();
    for (auto &t : threads) t.join();
    const double t_end = now_s();
    const double run = t_end - t_run0;

    printf("\npreload (the first layer's experts on every card, disk to RAM to card): %.2f s\n", t_pre);
    printf("run, layers 0..%d for %d tokens: %.2f s (%.2f GB read from disk in all)\n\n", LAST, N, run, bytes_read.load() / 1e9);
    double compute = 0, stall = 0;
    printf("layer  card  starts(s)  compute(ms)  waited for experts(ms)\n");
    for (int L = 0; L <= LAST; L++) {
        const double c = (log[L].end - log[L].start) * 1000, waited = L >= 1 ? fmax(0.0, (log[L].exp_ready - log[L].act_ready) * 1000) : 0.0;
        compute += c; stall += waited;
        if (L <= 24 || L % 12 == 0 || L >= LAST - 3) printf("%4d   %3d  %9.2f  %11.1f  %12.1f\n", L, log[L].gpu, log[L].start - (t_run0 - t_start), c, waited);
    }
    printf("sum of compute %.2f s, sum of the waits for experts %.2f s; the cards computed for %.1f%% of 8 x the run time\n", compute / 1000, stall / 1000, 100.0 * (compute / 1000) / (G * run));

    // ---- checks
    const int gl = gpu_of[LAST];
    CHECK(cudaSetDevice(gl));
    const float *final_ref = (const float *)resid_ref.data();
    if (LAST == LAST_LAYER_INDEX) {
        compare_rows("output of layer 92 against the pod's", works[gl].nxt, final_ref, N);
        for (int k = 1; k < 8; k++) { char name[64]; snprintf(name, sizeof name, "snapshot %d against the pod's", k); compare_rows(name, works[gl].snap[k], ref_snaps + (size_t)k * NFULL * W, N); }
        // the next token for the last position, from the CPU tail
        std::vector<float> row(W), snaps8((size_t)8 * W);
        CHECK(cudaMemcpy(row.data(), works[gl].nxt + (size_t)(N - 1) * W, W * 4, cudaMemcpyDeviceToHost));
        for (int s = 0; s < 8; s++) CHECK(cudaMemcpy(snaps8.data() + (size_t)s * W, works[gl].snap[s] + (size_t)(N - 1) * W, W * 4, cudaMemcpyDeviceToHost));
        char so[4096]; snprintf(so, sizeof so, "%s/pipeline-stage.so", argv[4]);
        void *tail = dlopen(so, RTLD_NOW | RTLD_LOCAL);
        if (!tail) { fprintf(stderr, "tail dlopen: %s\n", dlerror()); return 1; }
        int (*tail_open)(const char *, const char *, const char *, const char *, const char *, void **) = (int (*)(const char *, const char *, const char *, const char *, const char *, void **))dlsym(tail, "transformer_93_open");
        void *(*tail_make)(const void *) = (void *(*)(const void *))dlsym(tail, "transformer_93_sequence_create");
        int (*tail_run)(const void *, void *, const float *, const float *, unsigned, int, const char *, uint32_t *, Text *) = (int (*)(const void *, void *, const float *, const float *, unsigned, int, const char *, uint32_t *, Text *))dlsym(tail, "transformer_93_process");
        if (!tail_open || !tail_make || !tail_run) { fprintf(stderr, "tail symbols missing\n"); return 1; }
        char leaves[4096], head[4096], tik[4096], config[4096];
        snprintf(leaves, sizeof leaves, "%s/model-00094-of-000096.safetensors", argv[3]);
        snprintf(head, sizeof head, "%s/model-00094-of-000096.safetensors#language_model.lm_head.weight", argv[3]);
        snprintf(tik, sizeof tik, "%s/tiktoken.model", argv[3]); snprintf(config, sizeof config, "%s/tokenizer_config.json", argv[3]);
        void *model = NULL;
        if (!tail_open(leaves, head, tik, NULL, config, &model)) { fprintf(stderr, "tail open failed\n"); return 1; }
        void *seq = tail_make(model);
        uint32_t token = 0; Text word = { NULL, 0 };
        const double a = now_s();
        if (!seq || !tail_run(model, seq, row.data(), snaps8.data(), 8, 1, "", &token, &word)) { fprintf(stderr, "tail failed\n"); return 1; }
        printf("next token after position %d: id %u, text \"%.*s\" (tail %.1f ms on the CPU); for the full recorded prompt (1000 tokens) the CPU engine and the pod give id 3027 \" team\"\n", N - 1, token, (int)word.length, (const char *)word.bytes, 1000 * (now_s() - a));
    } else {
        std::vector<float> h((size_t)N * W);
        CHECK(cudaMemcpy(h.data(), works[gl].nxt, h.size() * 4, cudaMemcpyDeviceToHost));
        unsigned long long fnv = 1469598103934665603ULL; const unsigned char *bytes = (const unsigned char *)h.data();
        for (size_t i = 0; i < h.size() * 4; i++) { fnv ^= bytes[i]; fnv *= 1099511628211ULL; }
        printf("output of layer %d: FNV-1a 64 %016llx\n", LAST, fnv);
        if ((LAST + 1) % 12 == 0 && (LAST + 1) / 12 <= 7) compare_rows("against the pod's snapshot", works[gl].nxt, ref_snaps + (size_t)((LAST + 1) / 12) * NFULL * W, N);
    }
    return 0;
}
