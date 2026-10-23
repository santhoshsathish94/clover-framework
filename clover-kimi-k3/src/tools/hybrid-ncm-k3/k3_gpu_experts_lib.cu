// Shared CUDA library `libk3_gpu_experts.so` for executing Kimi K3's real MoE expert layers
// (`trunk-0` + `trunk-L` + all 896 experts per layer from `/opt/clover-k3/datasets/computed/expert-L/experts.direct`)
// directly across the RTX 3090 GPUs (`cuda:0`, `cuda:1`, `cuda:5`, `cuda:4`) in Agent Mode (`HybridAgent + K3 Experts + NCM`).
//
// Build on `gpu`:
//   /usr/local/cuda-12.6/bin/nvcc -O3 --fmad=false -arch=sm_86 -std=c++17 -Xcompiler -fPIC -shared \
//     -DFAST_ROUTER -DEXPERT_V2 -DINT8_PLANES -I/home/usergpu/k3gpu/src/code/gpu \
//     k3_gpu_experts_lib.cu -lcublas -lpthread -o libk3_gpu_experts.so

#include "decode.cuh"
#include <mutex>
#include <thread>

struct K3MultiCardChain {
    bool initialized = false;
    std::string dataset_dir;
    int max_n = 0;
    int n_moe_stages = 0;
    int moe_layers[4] = {1, 12, 48, 92};
    int moe_cards[4]  = {0, 1, 5, 4};

    // Card 0 also holds Layer 0 (dense KDA + MLP)
    Layer l0;
    Work w0;

    // 4 MoE stages (each holds trunk-L + all 896 experts of expert-L/experts.direct in VRAM)
    Layer y_moe[4];
    Work  w_moe[4];

    // Sparse Top-16 On-Demand Mode (`decode.cuh` `moe_block_sparse` + `HostCache` + `CardPool`) across layers 1..92
    bool sparse_initialized = false;
    int sparse_max_layer = 0;
    HostCache host_cache;
    CardPool sparse_pool;
    Layer    sparse_layers[93];
    KdaState sparse_kda[93];
    MlaCache sparse_mla[93];
    int      sparse_fd[93];

    // Host staging buffers for cross-card activation & snapshot handoff (`max_n x 7168` floats)
    std::vector<float> h_snap0;
    std::vector<float> h_cur;
    std::vector<float> h_snaps[8];

    std::mutex mu;
};

static K3MultiCardChain g_chain;

extern "C" {

// Initializes Layer 0 + 4 full 896-expert MoE layers (3,584 Kimi K3 experts total = 67.4 GB) across 4 GPUs.
// `layers4` and `cards4` are int[4] arrays (e.g. layers={1, 12, 48, 92}, cards={0, 1, 5, 4}).
int k3_experts_init(const char *dataset_dir, const int *layers4, const int *cards4, int n_stages, int max_n)
{
    std::lock_guard<std::mutex> lock(g_chain.mu);
    if (g_chain.initialized) return 0;
    if (n_stages < 1 || n_stages > 4) return -1;

    g_chain.dataset_dir = dataset_dir;
    g_chain.max_n = max_n;
    g_chain.n_moe_stages = n_stages;
    for (int i = 0; i < n_stages; i++) {
        g_chain.moe_layers[i] = layers4[i];
        g_chain.moe_cards[i]  = cards4[i];
    }
    for (int L = 0; L <= 92; L++) g_chain.sparse_fd[L] = -1;

    g_chain.h_snap0.resize((size_t)max_n * W);
    g_chain.h_cur.resize((size_t)max_n * W);
    for (int k = 0; k < 8; k++) g_chain.h_snaps[k].resize((size_t)max_n * W);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    // Load all 4 GPU stages in parallel threads (one thread + pinned buffer per card)
    std::vector<std::thread> loader_threads;
    for (int s = 0; s < n_stages; s++) {
        loader_threads.emplace_back([dataset_dir, max_n, s]() {
            const int card = g_chain.moe_cards[s];
            const int L = g_chain.moe_layers[s];
            CHECK(cudaSetDevice(card));
            const size_t chunk = 64u << 20;
            unsigned char *pinned = nullptr;
            CHECK(cudaMallocHost(&pinned, chunk));
            if (s == 0) {
                work_alloc(g_chain.w0, card, max_n, true, true, 4, 3);
                layer_load_trunk(dataset_dir, 0, g_chain.l0);
                layer_load_trunk(dataset_dir, L, g_chain.y_moe[0]);
                layer_load_expert_tables(dataset_dir, L, g_chain.y_moe[0]);
                double t_e = upload_experts(dataset_dir, L, g_chain.w0.d_experts, pinned, chunk);
                fprintf(stderr, "[K3-GPU-Experts] Card %d: loaded trunk-0 + Layer %d (all 896 experts, %.2fs)\n", card, L, t_e);
            } else {
                const int kinds = layer_is_mla(L) ? 2 : 1;
                work_alloc(g_chain.w_moe[s], card, max_n, false, true, 8, kinds);
                layer_load_trunk(dataset_dir, L, g_chain.y_moe[s]);
                layer_load_expert_tables(dataset_dir, L, g_chain.y_moe[s]);
                double t_e = upload_experts(dataset_dir, L, g_chain.w_moe[s].d_experts, pinned, chunk);
                fprintf(stderr, "[K3-GPU-Experts] Card %d: loaded Layer %d (%s, all 896 experts, %.2fs)\n",
                        card, L, g_chain.y_moe[s].mla ? "MLA+MoE" : "KDA+MoE", t_e);
            }
            CHECK(cudaFreeHost(pinned));
        });
    }
    for (auto &th : loader_threads) th.join();

    clock_gettime(CLOCK_MONOTONIC, &t1);
    g_chain.initialized = true;
    fprintf(stderr, "[K3-GPU-Experts] Ready: Layer 0 + %d MoE layers (%d experts = %.1f GB in VRAM) loaded in %.2fs\n",
            n_stages, n_stages * EXPERTS, n_stages * (EXPERTS * (double)EXPERT_RAW / 1e9), seconds(t0, t1));
    return 0;
}

// Runs `n` token embeddings (`in_embed [n, 7168]` float32) through:
//   Layer 0 (dense KDA + MLP) -> MoE Stage 0 (896 experts) -> MoE Stage 1 (896 experts) -> ...
// Outputs:
//   `out_hidden`   : `[n, 7168]` float32 final hidden states after Kimi K3 MoE experts
//   `out_selected` : `[n_stages, n, 16]` int32 top-16 routed expert IDs (0..895) per stage and token
//   `out_weights`  : `[n_stages, n, 16]` float32 routing weights per stage and token
// Returns elapsed compute milliseconds.
float k3_experts_forward(const float *in_embed, int n, float *out_hidden, int *out_selected, float *out_weights)
{
    std::lock_guard<std::mutex> lock(g_chain.mu);
    if (!g_chain.initialized || n <= 0 || n > g_chain.max_n) return -1.0f;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    const size_t bytes = (size_t)n * W * sizeof(float);

    // --- Stage 0 on `moe_cards[0]` (Layer 0 + MoE Layer `moe_layers[0]`) ---
    const int c0 = g_chain.moe_cards[0];
    CHECK(cudaSetDevice(c0));
    Work &w0 = g_chain.w0;
    w0.N = n;
    w0.moe_chunk = std::max(1, std::min(n, 2048));
    w0.SLOTS = w0.moe_chunk * TOP;

    CHECK(cudaMemcpy(w0.snap[0], in_embed, bytes, cudaMemcpyHostToDevice));
    run_layer0(w0, g_chain.l0, w0.snap[0], w0.cur);

    const float *sn0[2] = { w0.snap[0], nullptr };
    run_layer(w0, g_chain.y_moe[0], w0.cur, sn0, 1, w0.nxt);
    CHECK(cudaDeviceSynchronize());

    // Copy Stage 0 chosen top-16 experts (`[n, 16]`) and weights (`[n, 16]`)
    if (out_selected) {
        memcpy(out_selected, w0.h_selected.data(), (size_t)n * TOP * sizeof(int));
    }
    if (out_weights) {
        CHECK(cudaMemcpy(out_weights, w0.weights, (size_t)n * TOP * sizeof(float), cudaMemcpyDeviceToHost));
    }

    // Copy `w0.nxt` to host `h_cur` (and save `in_embed` as snapshot 0, `h_cur` as snapshot 1)
    CHECK(cudaMemcpy(g_chain.h_cur.data(), w0.nxt, bytes, cudaMemcpyDeviceToHost));
    memcpy(g_chain.h_snaps[0].data(), in_embed, bytes);
    memcpy(g_chain.h_snaps[1].data(), g_chain.h_cur.data(), bytes);
    int nsnap = 2;

    // --- Subsequent MoE Stages `1 .. n_moe_stages - 1` across `moe_cards[s]` ---
    for (int s = 1; s < g_chain.n_moe_stages; s++) {
        const int card = g_chain.moe_cards[s];
        CHECK(cudaSetDevice(card));
        Work &ws = g_chain.w_moe[s];
        ws.N = n;
        ws.moe_chunk = std::max(1, std::min(n, 2048));
        ws.SLOTS = ws.moe_chunk * TOP;

        CHECK(cudaMemcpy(ws.cur, g_chain.h_cur.data(), bytes, cudaMemcpyHostToDevice));
        const float *sn[9];
        for (int k = 0; k < nsnap; k++) {
            CHECK(cudaMemcpy(ws.snap[k], g_chain.h_snaps[k].data(), bytes, cudaMemcpyHostToDevice));
            sn[k] = ws.snap[k];
        }

        run_layer(ws, g_chain.y_moe[s], ws.cur, sn, nsnap, ws.nxt);
        CHECK(cudaDeviceSynchronize());

        if (out_selected) {
            memcpy(out_selected + (size_t)s * n * TOP, ws.h_selected.data(), (size_t)n * TOP * sizeof(int));
        }
        if (out_weights) {
            CHECK(cudaMemcpy(out_weights + (size_t)s * n * TOP, ws.weights, (size_t)n * TOP * sizeof(float), cudaMemcpyDeviceToHost));
        }

        CHECK(cudaMemcpy(g_chain.h_cur.data(), ws.nxt, bytes, cudaMemcpyDeviceToHost));
        if (nsnap < 8) {
            memcpy(g_chain.h_snaps[nsnap].data(), g_chain.h_cur.data(), bytes);
            nsnap++;
        }
    }

    if (out_hidden) {
        memcpy(out_hidden, g_chain.h_cur.data(), bytes);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    return (float)(seconds(t0, t1) * 1000.0);
}

// Initializes Sparse Top-16 On-Demand Mode (`decode.cuh` `moe_block_sparse` + `HostCache` + `CardPool`)
// for layers `1..max_layer` (up to all 92 MoE layers).
int k3_sparse_init(int max_layer, int host_cache_gb)
{
    std::lock_guard<std::mutex> lock(g_chain.mu);
    if (!g_chain.initialized) return -1;
    if (max_layer < 1) max_layer = 1;
    if (max_layer > 92) max_layer = 92;

    if (!g_chain.sparse_initialized) {
        g_chain.host_cache.init((size_t)host_cache_gb << 30);
        const int sp_card = g_chain.moe_cards[1];
        CHECK(cudaSetDevice(sp_card));
        size_t free_b = 0, total_b = 0;
        CHECK(cudaMemGetInfo(&free_b, &total_b));
        const size_t want_pool = (size_t)72 * EXPERT_RAW;
        const size_t reserve = free_b > (want_pool + (512u << 20)) ? (free_b - want_pool) : (free_b / 2);
        g_chain.sparse_pool.init(sp_card, reserve, 16);
        g_chain.sparse_initialized = true;
    }

    const int sp_card = g_chain.moe_cards[1];
    CHECK(cudaSetDevice(sp_card));
    for (int L = g_chain.sparse_max_layer + 1; L <= max_layer; L++) {
        bool is_res = false;
        for (int s = 0; s < g_chain.n_moe_stages; s++) {
            if (g_chain.moe_layers[s] == L) { is_res = true; break; }
        }
        if (is_res) continue;

        layer_load_trunk(g_chain.dataset_dir.c_str(), L, g_chain.sparse_layers[L]);
        layer_load_expert_tables(g_chain.dataset_dir.c_str(), L, g_chain.sparse_layers[L]);
        if (g_chain.sparse_layers[L].mla) {
            g_chain.sparse_mla[L].cap = g_chain.max_n;
            CHECK(cudaMalloc(&g_chain.sparse_mla[L].kv, (size_t)g_chain.max_n * MLA_EXP * 4));
            CHECK(cudaMalloc(&g_chain.sparse_mla[L].lat, (size_t)g_chain.max_n * MLA_KVLAT * 4));
        } else {
            kda_state_alloc(g_chain.sparse_kda[L]);
        }
        char path[4096];
        snprintf(path, sizeof path, "%s/expert-%d/experts.direct", g_chain.dataset_dir.c_str(), L);
        g_chain.sparse_fd[L] = open(path, O_RDONLY | O_DIRECT);
        if (g_chain.sparse_fd[L] < 0) g_chain.sparse_fd[L] = open(path, O_RDONLY);
    }
    if (max_layer > g_chain.sparse_max_layer) g_chain.sparse_max_layer = max_layer;
    return 0;
}

float k3_sparse_forward(const float *in_embed, int n, int max_layer, float *out_hidden, int *out_stats)
{
    std::lock_guard<std::mutex> lock(g_chain.mu);
    if (!g_chain.initialized || !g_chain.sparse_initialized || n <= 0 || n > g_chain.max_n) return -1.0f;
    if (max_layer > g_chain.sparse_max_layer) max_layer = g_chain.sparse_max_layer;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    const size_t bytes = (size_t)n * W * sizeof(float);

    const int c0 = g_chain.moe_cards[0];
    CHECK(cudaSetDevice(c0));
    Work &w0 = g_chain.w0;
    w0.N = n;
    CHECK(cudaMemcpy(w0.snap[0], in_embed, bytes, cudaMemcpyHostToDevice));
    run_layer0(w0, g_chain.l0, w0.snap[0], w0.cur);
    CHECK(cudaMemcpy(g_chain.h_cur.data(), w0.cur, bytes, cudaMemcpyDeviceToHost));
    memcpy(g_chain.h_snaps[0].data(), in_embed, bytes);
    int nsnap = 1;

    CardPool &pool = g_chain.sparse_pool;
    unsigned long needed0 = pool.st.needed, hc0 = pool.st.hit_card, hh0 = pool.st.hit_host, dr0 = pool.st.from_drive;

    for (int L = 1; L <= max_layer; L++) {
        int res_s = -1;
        for (int s = 0; s < g_chain.n_moe_stages; s++) {
            if (g_chain.moe_layers[s] == L) { res_s = s; break; }
        }
        if (res_s >= 0) {
            const int card = g_chain.moe_cards[res_s];
            CHECK(cudaSetDevice(card));
            Work &ws = (res_s == 0) ? g_chain.w0 : g_chain.w_moe[res_s];
            ws.N = n;
            CHECK(cudaMemcpy(ws.cur, g_chain.h_cur.data(), bytes, cudaMemcpyHostToDevice));
            const float *sn[9];
            for (int k = 0; k < nsnap; k++) {
                CHECK(cudaMemcpy(ws.snap[k], g_chain.h_snaps[k].data(), bytes, cudaMemcpyHostToDevice));
                sn[k] = ws.snap[k];
            }
            run_layer(ws, g_chain.y_moe[res_s], ws.cur, sn, nsnap, ws.nxt);
            CHECK(cudaDeviceSynchronize());
            if (L % 12 == 0 && nsnap < 8) {
                memcpy(g_chain.h_snaps[nsnap].data(), g_chain.h_cur.data(), bytes);
                nsnap++;
            }
            CHECK(cudaMemcpy(g_chain.h_cur.data(), ws.nxt, bytes, cudaMemcpyDeviceToHost));
        } else {
            const int sp_card = g_chain.moe_cards[1];
            CHECK(cudaSetDevice(sp_card));
            Work &ws = g_chain.w_moe[1];
            ws.N = n;
            CHECK(cudaMemcpy(ws.cur, g_chain.h_cur.data(), bytes, cudaMemcpyHostToDevice));
            const float *sn[9];
            for (int k = 0; k < nsnap; k++) {
                CHECK(cudaMemcpy(ws.snap[k], g_chain.h_snaps[k].data(), bytes, cudaMemcpyHostToDevice));
                sn[k] = ws.snap[k];
            }
            Layer &y = g_chain.sparse_layers[L];
            if (!y.mla) kda_state_reset(g_chain.sparse_kda[L]);
            run_layer_state(
                ws, y,
                y.mla ? nullptr : &g_chain.sparse_kda[L],
                y.mla ? &g_chain.sparse_mla[L] : nullptr,
                0, pool, g_chain.host_cache, g_chain.sparse_fd[L], 16,
                ws.cur, sn, nsnap, ws.nxt
            );
            CHECK(cudaDeviceSynchronize());
            if (L % 12 == 0 && nsnap < 8) {
                memcpy(g_chain.h_snaps[nsnap].data(), g_chain.h_cur.data(), bytes);
                nsnap++;
            }
            CHECK(cudaMemcpy(g_chain.h_cur.data(), ws.nxt, bytes, cudaMemcpyDeviceToHost));
        }
    }

    if (out_hidden) memcpy(out_hidden, g_chain.h_cur.data(), bytes);
    if (out_stats) {
        out_stats[0] = (int)(pool.st.needed - needed0);
        out_stats[1] = (int)(pool.st.hit_card - hc0);
        out_stats[2] = (int)(pool.st.hit_host - hh0);
        out_stats[3] = (int)(pool.st.from_drive - dr0);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    return (float)(seconds(t0, t1) * 1000.0);
}

} // extern "C"
