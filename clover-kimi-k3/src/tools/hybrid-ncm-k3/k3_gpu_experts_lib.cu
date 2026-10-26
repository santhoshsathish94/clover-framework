// Shared CUDA library `libk3_gpu_experts.so` for executing Kimi K3's real MoE expert layers
// (`trunk-0` + `trunk-L` + all 896 experts per layer from `/opt/clover-k3/datasets/computed/expert-L/experts.direct`)
// directly across the RTX 3090 GPUs (`cuda:0`, `cuda:1`, `cuda:5`, `cuda:4`) in Agent Mode (`HybridAgent + K3 Experts + NCM`).
//
// Build on `gpu`:
//   INC=""; for c in copyq slotcache routetable planner supply poolinit loader engcfg; do INC="$INC -I/home/usergpu/k3gpu/src/rust/$c/include"; done
//   /usr/local/cuda-12.6/bin/nvcc -O3 --fmad=false -arch=sm_86 -std=c++17 -Xcompiler -fPIC -shared \
//     -DFAST_ROUTER -DEXPERT_V2 -DINT8_PLANES -I/home/usergpu/k3gpu/src/code/gpu $INC \
//     k3_gpu_experts_lib.cu -lcublas -ldl -lpthread -o libk3_gpu_experts.so

#include "decode.cuh"
#include <dlfcn.h>
#include <mutex>
#include <thread>

// Optional Rust host-path selectors (`K3GPU_SLOTCACHE`, `K3GPU_POOLINIT`, `K3GPU_SUPPLY`) for `libk3_gpu_experts.so`
// (`K3Full93Chain` and `K3MultiCardChain` use `SlotCacheSel`, `HostCache::init`, `CardPool::init`, and `pool_ensure`).
static bool g_k3_rust_configured = false;
static void k3_configure_rust_hostpath()
{
    if (g_k3_rust_configured) return;
    g_k3_rust_configured = true;

    auto want_rust = [](const char *var, const char *name, bool built) -> bool {
        const char *m = getenv(var);
        if (!m || !*m || strcmp(m, "cpp") == 0) {
            fprintf(stderr, "K3GPU: %s cpp\n", name);
            return false;
        }
        if (strcmp(m, "rust") != 0) {
            fprintf(stderr, "%s must be cpp or rust (got %s)\n", var, m);
            exit(1);
        }
        if (!built) {
            fprintf(stderr, "%s=rust: this binary was built without the header\n", var);
            exit(1);
        }
        return true;
    };
    auto open_lib = [](const char *var, const char *dflt_lib, const char **lib_out) -> void * {
        char lv[64];
        snprintf(lv, sizeof lv, "%s_LIB", var);
        const char *lib = getenv(lv);
        if (!lib || !*lib) lib = dflt_lib;
        void *h = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
        if (!h) {
            fprintf(stderr, "%s=rust: cannot load %s: %s\n", var, lib, dlerror());
            exit(1);
        }
        *lib_out = lib;
        return h;
    };

#ifdef K3GPU_SLOTCACHE_BUILT
    if (want_rust("K3GPU_SLOTCACHE", "slot cache", true)) {
        const char *lib = nullptr;
        void *h = open_lib("K3GPU_SLOTCACHE", "libslotcache.so", &lib);
#define K3_SC_SYM(f, name) g_sc.f = (decltype(g_sc.f))dlsym(h, name); if (!g_sc.f) { fprintf(stderr, "K3GPU_SLOTCACHE=rust: %s has no %s\n", lib, name); exit(1); }
        K3_SC_SYM(set_policy, "sc_set_policy") K3_SC_SYM(set_next_use, "sc_set_next_use") K3_SC_SYM(nw, "sc_new") K3_SC_SYM(add_spare, "sc_add_spare")
        K3_SC_SYM(contains, "sc_contains") K3_SC_SYM(find, "sc_find") K3_SC_SYM(touch, "sc_touch") K3_SC_SYM(worth, "sc_worth")
        K3_SC_SYM(acquire, "sc_acquire") K3_SC_SYM(try_acquire, "sc_try_acquire") K3_SC_SYM(remove, "sc_remove") K3_SC_SYM(release, "sc_release")
#undef K3_SC_SYM
        g_sc.on = true;
        fprintf(stderr, "K3GPU: slot cache rust (%s)\n", lib);
    }
#else
    want_rust("K3GPU_SLOTCACHE", "slot cache", false);
#endif
    const char *pol = getenv("K3GPU_POLICY");
    slotcache_set_policy((pol && *pol) ? atoi(pol) : 3);

#ifdef K3GPU_POOLINIT_BUILT
    if (want_rust("K3GPU_POOLINIT", "pool init", true)) {
        const char *lib = nullptr;
        void *h = open_lib("K3GPU_POOLINIT", "libpoolinit.so", &lib);
        g_pi.host_init = (decltype(g_pi.host_init))dlsym(h, "pi_host_init");
        g_pi.pool_init = (decltype(g_pi.pool_init))dlsym(h, "pi_pool_init");
        if (!g_pi.host_init || !g_pi.pool_init) {
            fprintf(stderr, "K3GPU_POOLINIT=rust: %s missing pi_host_init/pi_pool_init\n", lib);
            exit(1);
        }
        g_pi.on = true;
        fprintf(stderr, "K3GPU: pool init rust (%s)\n", lib);
    }
#else
    want_rust("K3GPU_POOLINIT", "pool init", false);
#endif

#ifdef K3GPU_SUPPLY_BUILT
    if (want_rust("K3GPU_SUPPLY", "supply", true)) {
        const char *lib = nullptr;
        void *h = open_lib("K3GPU_SUPPLY", "libsupply.so", &lib);
        g_sy.init = (decltype(g_sy.init))dlsym(h, "sy_init");
        g_sy.ensure = (decltype(g_sy.ensure))dlsym(h, "sy_ensure");
        g_sy.wb_skipped = (decltype(g_sy.wb_skipped))dlsym(h, "sy_wb_skipped");
        if (!g_sy.init || !g_sy.ensure || !g_sy.wb_skipped) {
            fprintf(stderr, "K3GPU_SUPPLY=rust: %s missing sy_init/sy_ensure/sy_wb_skipped\n", lib);
            exit(1);
        }
        sy_config c = {};
        c.experts = EXPERTS; c.expert_raw = EXPERT_RAW; c.vq = g_vq ? 1 : 0; c.prefill = g_prefill ? 1 : 0; c.chunks = g_chunks; c.slab = g_slab;
        g_sy.init(&g_sy_ops, &c);
        g_sy.on = true;
        fprintf(stderr, "K3GPU: supply rust (%s)\n", lib);
    }
#else
    want_rust("K3GPU_SUPPLY", "supply", false);
#endif
}

// Selects the top-16 experts out of the `k_res` resident experts (`d_res_ids[0..k_res-1]`, which are the `k_res` highest-router-bias
// experts of the layer in `0..895`) using the exact double-precision router scores (`scores`, `choices`) of `router_scores` in `moe.cuh`,
// and normalizes the 16 weights to sum to 1.0.
__global__ void top16_bounded(const float *scores, const float *choices, const int *res_ids, int *selected, float *weights, int *bad, int N, int k_res)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const float *sc = scores + (size_t)n * EXPERTS, *ch = choices + (size_t)n * EXPERTS;
    float best[TOP]; int sel[TOP]; int count = 0;
    for (int j = 0; j < k_res; j++) {
        const int e = res_ids[j];
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

// Eyeriss v2 Adaptive Multi-Token Broadcast Coalescing (`coalesce_cold_broadcast`):
// When `N > 1` tokens are routed in a window, finds cold experts (`vram_mask[e] == 0`) demanded by >= 2 tokens
// in the window and marks the top `b_shared` multi-token experts with `vram_mask[e] = 3` so all tokens in the
// window share the same fetched experts in `expert_tile_v3<16/32>`.
__global__ void coalesce_cold_broadcast(const float *choices, unsigned char *vram_mask, int N, int k_tok_tail, int b_shared)
{
    __shared__ float s_util[EXPERTS];
    __shared__ int s_freq[EXPERTS];
    const int tid = threadIdx.x;
    for (int e = tid; e < EXPERTS; e += blockDim.x) {
        s_util[e] = 0.0f;
        s_freq[e] = 0;
    }
    __syncthreads();

    const int n_start = max(0, N - max(k_tok_tail, min(N, 16)));
    for (int n = n_start + tid; n < N; n += blockDim.x) {
        const float *ch = choices + (size_t)n * EXPERTS;
        // Find the 16th highest choice for token n
        float top16_th = -1e30f;
        float best[TOP]; int cnt = 0;
        for (int e = 0; e < EXPERTS; e++) {
            float v = ch[e];
            if (cnt < TOP || v > best[TOP - 1]) {
                int p = cnt < TOP ? cnt : TOP - 1;
                while (p && v > best[p - 1]) { best[p] = best[p - 1]; p--; }
                best[p] = v;
                if (cnt < TOP) cnt++;
            }
        }
        top16_th = best[TOP - 1];
        for (int e = 0; e < EXPERTS; e++) {
            if (vram_mask[e] == 0 && ch[e] >= top16_th) {
                atomicAdd(&s_util[e], ch[e]);
                atomicAdd(&s_freq[e], 1);
            }
        }
    }
    __syncthreads();

    if (tid == 0 && b_shared > 0) {
        for (int k = 0; k < b_shared; k++) {
            int best_e = -1;
            float best_u = 0.0f;
            for (int e = 0; e < EXPERTS; e++) {
                if (s_freq[e] >= 2 && s_util[e] > best_u) {
                    best_u = s_util[e];
                    best_e = e;
                }
            }
            if (best_e < 0) break;
            vram_mask[best_e] = 3;
            s_util[best_e] = 0.0f;
            s_freq[best_e] = 0;
        }
    }
}

// Smart 896-expert router:
//   - Scans all 896 experts (`0..895`) for every token `n`.
//   - Any expert `e` with `vram_mask[e] >= 1` (`1` = permanent `d_res_exp[L]`, `2` = already resident in `pool_card[c]` dynamic VRAM,
//     `3` = Eyeriss v2 multi-token broadcast-coalesced expert) is available at 0 extra cold-miss budget to ALL tokens `0..N-1`.
//   - For frontier tokens (`n >= N - k_tok_tail`), up to `k_dyn_max` highest-scoring cold experts (`vram_mask[e] == 0`) per token are admitted from `0..895`,
//     keeping the working set inside `pool_card[c]` (~220 dynamic VRAM slots/card) while preserving full 896-expert top-rank routing!
__global__ void top16_smart_896(const float *scores, const float *choices, const unsigned char *vram_mask,
                                int *selected, float *weights, int *bad, int N, int k_tok_tail, int k_dyn_max)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const float *sc = scores + (size_t)n * EXPERTS, *ch = choices + (size_t)n * EXPERTS;

    // 1. Find top-16 among VRAM-resident / broadcast-coalesced experts (`vram_mask[e] >= 1`)
    float best_v[TOP]; int sel_v[TOP]; int cnt_v = 0;
    // 2. Find true top-16 across ALL 896 experts (`0..895`)
    float best_all[TOP]; int sel_all[TOP]; int cnt_all = 0;

    const bool is_frontier = (n >= N - k_tok_tail);
    for (int e = 0; e < EXPERTS; e++) {
        const float choice = ch[e];
        if (!isfinite(choice)) { atomicAdd(bad, 1); return; }
        if (vram_mask[e]) {
            if (cnt_v < TOP || choice > best_v[TOP - 1]) {
                int pos = cnt_v < TOP ? cnt_v : TOP - 1;
                while (pos && choice > best_v[pos - 1]) { best_v[pos] = best_v[pos - 1]; sel_v[pos] = sel_v[pos - 1]; pos--; }
                best_v[pos] = choice; sel_v[pos] = e;
                if (cnt_v < TOP) cnt_v++;
            }
        }
        if (is_frontier) {
            if (cnt_all < TOP || choice > best_all[TOP - 1]) {
                int pos = cnt_all < TOP ? cnt_all : TOP - 1;
                while (pos && choice > best_all[pos - 1]) { best_all[pos] = best_all[pos - 1]; sel_all[pos] = sel_all[pos - 1]; pos--; }
                best_all[pos] = choice; sel_all[pos] = e;
                if (cnt_all < TOP) cnt_all++;
            }
        }
    }

    int final_sel[TOP];
    if (!is_frontier || k_dyn_max <= 0) {
        for (int r = 0; r < TOP; r++) final_sel[r] = sel_v[r];
    } else if (k_dyn_max >= TOP) {
        for (int r = 0; r < TOP; r++) final_sel[r] = sel_all[r];
    } else {
        int out_cnt = 0, dyn_used = 0;
        for (int r = 0; r < TOP && out_cnt < TOP; r++) {
            const int e = sel_all[r];
            if (vram_mask[e] >= 1) {
                // Permanent VRAM (`1`), dynamic CardPool VRAM (`2`), or multi-token broadcast-coalesced (`3`): free!
                final_sel[out_cnt++] = e;
            } else if (dyn_used < k_dyn_max) {
                // Top-`k_dyn_max` cold expert (`vram_mask[e] == 0`)
                final_sel[out_cnt++] = e;
                dyn_used++;
            }
        }
        for (int rv = 0; rv < cnt_v && out_cnt < TOP; rv++) {
            const int ev = sel_v[rv];
            bool dup = false;
            for (int j = 0; j < out_cnt; j++) if (final_sel[j] == ev) { dup = true; break; }
            if (!dup) final_sel[out_cnt++] = ev;
        }
    }

    double total = 0.0;
    for (int r = 0; r < TOP; r++) total += (double)sc[final_sel[r]];
    const float inverse = (float)(1.0 / (total + 1e-20));
    for (int r = 0; r < TOP; r++) {
        selected[n * TOP + r] = final_sel[r];
        weights[n * TOP + r] = sc[final_sel[r]] * inverse;
    }
}

struct K3Full93Chain {
    bool initialized = false;
    int routing_mode = 2;  // 0 = bounded k_res only; 1 = full 896 all tokens; 2 = smart 896 (k_tok_tail=4, k_dyn_max=6)
    int k_tok_tail = 4;
    int k_dyn_max = 6;
    int max_n = 0;
    int k_res = 28;  // permanent resident VRAM experts per layer across all 92 MoE layers + dynamic CardPool for the remaining 868 experts/layer
    Layer y[93];
    unsigned char *d_res_exp[93];
    const unsigned char **d_eptr[93];
    int *d_res_ids[93];
    unsigned char *d_vram_mask[6];
    std::vector<int> h_res_ids[93];
    std::vector<const unsigned char *> h_res_map[93];  // [896]: non-null if expert e is permanently pinned in d_res_exp[L]
    int fd[93];
    CardPool pool_card[6];
    HostCache host_cache;
    Work w_card[6];
    unsigned long perm_hits = 0;
    std::vector<float> h_cur;
    std::vector<float> h_snaps[8];
    std::mutex mu;
};

static K3Full93Chain g_full93;

// Executes a full MoE layer `y.L` routing across ALL 896 experts (`0..895`):
//   - `routing_mode == 1`: unconstrained `top16` across all 896 experts for all N tokens.
//   - `routing_mode >= 2`: `top16_smart_896` across all 896 experts (all VRAM experts + top `k_dyn_max` cold experts for the last `k_tok_tail` frontier tokens).
static void run_moe_layer_full896(Work &w, const Layer &y, CardPool &pool, HostCache &hc, int fd,
                                  const std::vector<const unsigned char *> &res_map, unsigned char *d_vram_mask,
                                  int routing_mode, int k_tok_tail, int k_dyn_max,
                                  const float *in, const float *const *snaps, int nsnap_in, float *out, int *out_uniq_count)
{
    const int N = w.N, L = y.L;
    const Trunk &T = y.T;
    aggregate_with(w, snaps, nsnap_in, in, rec_data(T, 37));
    normalize_rows<<<N, 256>>>(w.Xn, w.Agg, y.input_gains, W);
    if (y.mla) mla_block(w, y, out); else kda_attention(w, y, out);
    if (L % 12 != 0) add_into<<<blocks((size_t)N * W), 256>>>(out, in, (size_t)N * W);
    const float *sn[9]; int ns = nsnap_in;
    for (int i = 0; i < nsnap_in; i++) sn[i] = snaps[i];
    if (L % 12 == 0) sn[ns++] = in;
    aggregate_with(w, sn, ns, out, rec_data(T, 38));
    normalize_rows<<<N, 256>>>(w.post, w.Agg, rec_data(T, 5), W);

    // Exact router + top-16 selection across ALL 896 experts (`0..895`)
    CHECK(cudaMemset(w.bad, 0, 4));
    transpose_rows<<<blocks((size_t)N * W), 256>>>(w.post_t, w.post, N, W);
    router_scores<<<dim3((N + 127) / 128, EXPERTS), 128>>>(w.scores, w.choices, rec_data(T, 29), rec_data(T, 30), w.post_t, N);
    if (routing_mode == 1) {
        top16<<<blocks(N), 256>>>(w.scores, w.choices, w.selected, w.weights, w.bad, N);
    } else {
        unsigned char h_vram_mask[EXPERTS];
        for (int e = 0; e < EXPERTS; e++) {
            h_vram_mask[e] = (res_map[e] != nullptr) ? 1 : (pool.c.contains(ekey(L, e)) ? 2 : 0);
        }
        CHECK(cudaMemcpy(d_vram_mask, h_vram_mask, EXPERTS, cudaMemcpyHostToDevice));
        if (N > 1 && k_dyn_max > 0) {
            const int b_shared = std::min(pool.slots / 4, std::max(8, k_dyn_max * 2));
            coalesce_cold_broadcast<<<1, 256>>>(w.choices, d_vram_mask, N, k_tok_tail, b_shared);
        }
        top16_smart_896<<<blocks(N), 256>>>(w.scores, w.choices, d_vram_mask, w.selected, w.weights, w.bad, N, k_tok_tail, k_dyn_max);
    }
    CHECK(cudaMemcpy(w.h_selected.data(), w.selected, (size_t)N * TOP * 4, cudaMemcpyDeviceToHost));

    gemm(w.post, N, W, T, 31, w.latent);
    int total_uniq = 0;
    for (int t0 = 0; t0 < N; t0 += w.moe_chunk) {
        const int nc = std::min(w.moe_chunk, N - t0), slots = nc * TOP;
        std::fill(w.h_count.begin(), w.h_count.end(), 0);
        for (int i = 0; i < slots; i++) w.h_count[w.h_selected[(size_t)t0 * TOP + i]]++;
        { int acc = 0; for (int e = 0; e < EXPERTS; e++) { w.h_start[e] = acc; acc += w.h_count[e]; } }
        { std::vector<int> fill(EXPERTS, 0);
          for (int n = 0; n < nc; n++) for (int r = 0; r < TOP; r++) {
              const int e = w.h_selected[(size_t)(t0 + n) * TOP + r];
              const int slot = w.h_start[e] + fill[e]++;
              w.h_slot_token[slot] = t0 + n;
              w.h_slot_of[n * TOP + r] = slot;
          }
        }
        CHECK(cudaMemcpy(w.d_slot_token, w.h_slot_token.data(), (size_t)slots * 4, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(w.d_slot_of, w.h_slot_of.data(), (size_t)slots * 4, cudaMemcpyHostToDevice));

        std::vector<int> perm_uniq, dyn_hits, dyn_misses;
        for (int e = 0; e < EXPERTS; e++) {
            if (!w.h_count[e]) continue;
            if (res_map[e]) {
                pool.h_eptr[e] = res_map[e];
                perm_uniq.push_back(e);
            } else if (pool.c.contains(ekey(L, e))) {
                dyn_hits.push_back(e);
            } else {
                dyn_misses.push_back(e);
            }
        }
        g_full93.perm_hits += perm_uniq.size();
        // SpecMD `access_group` Hit-First Intra-Layer Ordering: pass all resident CardPool hits before any misses!
        std::vector<int> dyn_uniq = dyn_hits;
        dyn_uniq.insert(dyn_uniq.end(), dyn_misses.begin(), dyn_misses.end());
        total_uniq = std::max(total_uniq, (int)(perm_uniq.size() + dyn_uniq.size()));

        if (dyn_uniq.size() <= (size_t)pool.slots) {
            // Fast path: all dynamic misses fit in `pool` simultaneously alongside all permanent experts!
            if (!dyn_uniq.empty()) {
                pool_ensure(pool, hc, L, fd, dyn_uniq, w.h_count.data(), 16);
                for (int e : perm_uniq) pool.h_eptr[e] = res_map[e];
            }
            CHECK(cudaMemcpy(pool.d_eptr, pool.h_eptr.data(), EXPERTS * sizeof(void *), cudaMemcpyHostToDevice));
            std::vector<int4> wide, narrow;
            for (int e = 0; e < EXPERTS; e++) {
                if (!w.h_count[e]) continue;
                for (int i0 = 0; i0 < w.h_count[e]; i0 += 32) {
                    const int cnt = std::min(32, w.h_count[e] - i0);
                    (cnt <= 16 ? narrow : wide).push_back(make_int4(e, w.h_start[e] + i0, cnt, cnt <= 16 ? 16 : 32));
                }
            }
            wide.insert(wide.end(), narrow.begin(), narrow.end());
            const int n_items = (int)wide.size();
            if (n_items > 0) {
                CHECK(cudaMemcpy(w.d_items, wide.data(), (size_t)n_items * sizeof(int4), cudaMemcpyHostToDevice));
                expert_proj_v3p<<<dim3(48, n_items), 32>>>(pool.d_eptr, y.d_mapvals, y.d_refs, y.d_ref_base, 0, w.latent, LATENT, w.d_slot_token, w.d_items, w.G, EXPERT);
                expert_proj_v3p<<<dim3(48, n_items), 32>>>(pool.d_eptr, y.d_mapvals, y.d_refs, y.d_ref_base, 1, w.latent, LATENT, w.d_slot_token, w.d_items, w.U, EXPERT);
                dense_activation<<<blocks((size_t)slots * EXPERT), 256>>>(w.G, w.U, (size_t)slots * EXPERT);
                expert_proj_v3p<<<dim3(56, n_items), 32>>>(pool.d_eptr, y.d_mapvals, y.d_refs, y.d_ref_base, 2, w.G, EXPERT, w.d_slot_token, w.d_items, w.D, LATENT);
            }
        } else {
            // Chunked fallback if > `pool.slots` dynamic experts are needed in one large batch
            std::vector<int> all_uniq;
            for (int e = 0; e < EXPERTS; e++) if (w.h_count[e]) all_uniq.push_back(e);
            for (size_t gi = 0; gi < all_uniq.size();) {
                const size_t gj = std::min(all_uniq.size(), gi + (size_t)pool.slots);
                const std::vector<int> grp(all_uniq.begin() + gi, all_uniq.begin() + gj);
                std::vector<int> grp_dyn;
                for (int e : grp) {
                    if (res_map[e]) pool.h_eptr[e] = res_map[e];
                    else grp_dyn.push_back(e);
                }
                if (!grp_dyn.empty()) {
                    pool_ensure(pool, hc, L, fd, grp_dyn, w.h_count.data(), 16);
                    for (int e : grp) if (res_map[e]) pool.h_eptr[e] = res_map[e];
                }
                CHECK(cudaMemcpy(pool.d_eptr, pool.h_eptr.data(), EXPERTS * sizeof(void *), cudaMemcpyHostToDevice));
                std::vector<int4> wide, narrow;
                for (int e : grp)
                    for (int i0 = 0; i0 < w.h_count[e]; i0 += 32) {
                        const int cnt = std::min(32, w.h_count[e] - i0);
                        (cnt <= 16 ? narrow : wide).push_back(make_int4(e, w.h_start[e] + i0, cnt, cnt <= 16 ? 16 : 32));
                    }
                wide.insert(wide.end(), narrow.begin(), narrow.end());
                const int n_items = (int)wide.size();
                CHECK(cudaMemcpy(w.d_items, wide.data(), (size_t)n_items * sizeof(int4), cudaMemcpyHostToDevice));
                const int s0 = w.h_start[grp.front()], s1 = w.h_start[grp.back()] + w.h_count[grp.back()];
                expert_proj_v3p<<<dim3(48, n_items), 32>>>(pool.d_eptr, y.d_mapvals, y.d_refs, y.d_ref_base, 0, w.latent, LATENT, w.d_slot_token, w.d_items, w.G, EXPERT);
                expert_proj_v3p<<<dim3(48, n_items), 32>>>(pool.d_eptr, y.d_mapvals, y.d_refs, y.d_ref_base, 1, w.latent, LATENT, w.d_slot_token, w.d_items, w.U, EXPERT);
                dense_activation<<<blocks((size_t)(s1 - s0) * EXPERT), 256>>>(w.G + (size_t)s0 * EXPERT, w.U + (size_t)s0 * EXPERT, (size_t)(s1 - s0) * EXPERT);
                expert_proj_v3p<<<dim3(56, n_items), 32>>>(pool.d_eptr, y.d_mapvals, y.d_refs, y.d_ref_base, 2, w.G, EXPERT, w.d_slot_token, w.d_items, w.D, LATENT);
                gi = gj;
            }
        }
        mix_ranks<<<blocks((size_t)nc * LATENT), 256>>>(w.mixture + (size_t)t0 * LATENT, w.D, w.d_slot_of, w.weights + (size_t)t0 * TOP, nc);
    }
    if (out_uniq_count) *out_uniq_count = total_uniq;
    normalize_rows<<<N, 256>>>(w.lnorm, w.mixture, rec_data(T, 33), LATENT);
    gemm(w.lnorm, N, LATENT, T, 32, w.routed);
    gemm(w.post, N, W, T, 34, w.sg); gemm(w.post, N, W, T, 35, w.su);
    dense_activation<<<blocks((size_t)N * SHARED), 256>>>(w.sg, w.su, (size_t)N * SHARED);
    gemm(w.sg, N, SHARED, T, 36, w.so);
    final_sum<<<blocks((size_t)N * W), 256>>>(out, w.routed, w.so, (size_t)N * W);
}

static void run_moe_layer_resident(Work &w, const Layer &y, const unsigned char *const *d_eptr,
                                   const int *d_res_ids, const std::vector<int> &h_res_ids, int k_res,
                                   const float *in, const float *const *snaps, int nsnap_in, float *out, int *out_uniq_count)
{
    const int N = w.N, L = y.L;
    const Trunk &T = y.T;
    aggregate_with(w, snaps, nsnap_in, in, rec_data(T, 37));
    normalize_rows<<<N, 256>>>(w.Xn, w.Agg, y.input_gains, W);
    if (y.mla) mla_block(w, y, out); else kda_attention(w, y, out);
    if (L % 12 != 0) add_into<<<blocks((size_t)N * W), 256>>>(out, in, (size_t)N * W);
    const float *sn[9]; int ns = nsnap_in;
    for (int i = 0; i < nsnap_in; i++) sn[i] = snaps[i];
    if (L % 12 == 0) sn[ns++] = in;
    aggregate_with(w, sn, ns, out, rec_data(T, 38));
    normalize_rows<<<N, 256>>>(w.post, w.Agg, rec_data(T, 5), W);

    // Exact router + top-16 selection among the layer's `k_res` highest-prior experts + 4-bit `expert_proj_v3p` + shared expert
    CHECK(cudaMemset(w.bad, 0, 4));
    transpose_rows<<<blocks((size_t)N * W), 256>>>(w.post_t, w.post, N, W);
    router_scores<<<dim3((N + 127) / 128, EXPERTS), 128>>>(w.scores, w.choices, rec_data(T, 29), rec_data(T, 30), w.post_t, N);
    top16_bounded<<<blocks(N), 256>>>(w.scores, w.choices, d_res_ids, w.selected, w.weights, w.bad, N, k_res);
    CHECK(cudaMemcpy(w.h_selected.data(), w.selected, (size_t)N * TOP * 4, cudaMemcpyDeviceToHost));

    gemm(w.post, N, W, T, 31, w.latent);
    const int slots = N * TOP;
    std::fill(w.h_count.begin(), w.h_count.end(), 0);
    for (int i = 0; i < slots; i++) w.h_count[w.h_selected[i]]++;
    int uniq_cnt = 0;
    { int acc = 0;
      for (int e : h_res_ids) {
          w.h_start[e] = acc;
          acc += w.h_count[e];
          if (w.h_count[e]) uniq_cnt++;
      }
    }
    if (out_uniq_count) *out_uniq_count = uniq_cnt;

    { std::vector<int> fill(EXPERTS, 0);
      for (int n = 0; n < N; n++) for (int r = 0; r < TOP; r++) {
          const int e = w.h_selected[n * TOP + r];
          const int slot = w.h_start[e] + fill[e]++;
          w.h_slot_token[slot] = n;
          w.h_slot_of[n * TOP + r] = slot;
      }
    }
    CHECK(cudaMemcpy(w.d_slot_token, w.h_slot_token.data(), (size_t)slots * 4, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(w.d_slot_of, w.h_slot_of.data(), (size_t)slots * 4, cudaMemcpyHostToDevice));

    std::vector<int4> wide, narrow;
    for (int e : h_res_ids) {
        for (int i0 = 0; i0 < w.h_count[e]; i0 += 32) {
            const int cnt = std::min(32, w.h_count[e] - i0);
            (cnt <= 16 ? narrow : wide).push_back(make_int4(e, w.h_start[e] + i0, cnt, cnt <= 16 ? 16 : 32));
        }
    }
    wide.insert(wide.end(), narrow.begin(), narrow.end());
    const int n_items = (int)wide.size();
    if (n_items > 0) {
        CHECK(cudaMemcpy(w.d_items, wide.data(), (size_t)n_items * sizeof(int4), cudaMemcpyHostToDevice));
        expert_proj_v3p<<<dim3(48, n_items), 32>>>(d_eptr, y.d_mapvals, y.d_refs, y.d_ref_base, 0, w.latent, LATENT, w.d_slot_token, w.d_items, w.G, EXPERT);
        expert_proj_v3p<<<dim3(48, n_items), 32>>>(d_eptr, y.d_mapvals, y.d_refs, y.d_ref_base, 1, w.latent, LATENT, w.d_slot_token, w.d_items, w.U, EXPERT);
        dense_activation<<<blocks((size_t)slots * EXPERT), 256>>>(w.G, w.U, (size_t)slots * EXPERT);
        expert_proj_v3p<<<dim3(56, n_items), 32>>>(d_eptr, y.d_mapvals, y.d_refs, y.d_ref_base, 2, w.G, EXPERT, w.d_slot_token, w.d_items, w.D, LATENT);
    }
    mix_ranks<<<blocks((size_t)N * LATENT), 256>>>(w.mixture, w.D, w.d_slot_of, w.weights, N);
    normalize_rows<<<N, 256>>>(w.lnorm, w.mixture, rec_data(T, 33), LATENT);
    gemm(w.lnorm, N, LATENT, T, 32, w.routed);
    gemm(w.post, N, W, T, 34, w.sg); gemm(w.post, N, W, T, 35, w.su);
    dense_activation<<<blocks((size_t)N * SHARED), 256>>>(w.sg, w.su, (size_t)N * SHARED);
    gemm(w.sg, N, SHARED, T, 36, w.so);
    final_sum<<<blocks((size_t)N * W), 256>>>(out, w.routed, w.so, (size_t)N * W);
}

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
        k3_configure_rust_hostpath();
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

// Initializes ALL 93 Kimi K3 Trunks (`trunk-0` through `trunk-92`, 52.67 GiB) + all 92 Expert Mapping Tables
// + `k_res` resident 4-bit palette-mapped experts per layer across all 92 MoE layers (`92 * k_res` experts in VRAM)
// across the 6 GPUs `cuda:0..5` in parallel.
int k3_full93_init(const char *dataset_dir, int max_n, int k_res)
{
    std::lock_guard<std::mutex> lock(g_full93.mu);
    if (g_full93.initialized) return 0;
    if (k_res < 16) k_res = 16;
    if (k_res > 28) k_res = 28;  // Keep >= 6.0 GiB free per card for dynamic CardPool (~220 slots/card) + headroom

    g_full93.max_n = max_n;
    g_full93.k_res = k_res;
    const char *env_f896 = getenv("K3_FULL896_ROUTING");
    g_full93.routing_mode = (env_f896 && *env_f896) ? atoi(env_f896) : 2;
    g_full93.h_cur.resize((size_t)max_n * W);
    for (int k = 0; k < 8; k++) g_full93.h_snaps[k].resize((size_t)max_n * W);
    for (int L = 0; L <= 92; L++) g_full93.fd[L] = -1;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    k3_configure_rust_hostpath();

    // Initialize 48 GB pinned HostCache (~2,936 slots in pinned system RAM) for dynamic 896-expert routing
    const char *env_hgb = getenv("K3_FULL93_HOST_GB");
    const size_t host_gb = (env_hgb && atoi(env_hgb) > 0) ? (size_t)atoi(env_hgb) : 48;
    g_full93.host_cache.init(host_gb << 30);

    const int l_start[6] = {0, 16, 32, 47, 62, 77};
    const int l_end[6]   = {15, 31, 46, 61, 76, 92};

    std::vector<std::thread> threads;
    for (int c = 0; c < 6; c++) {
        threads.emplace_back([dataset_dir, max_n, k_res, c, &l_start, &l_end]() {
            CHECK(cudaSetDevice(c));
            work_alloc(g_full93.w_card[c], c, max_n, c == 0, false, 8, 3);
            CHECK(cudaMalloc(&g_full93.d_vram_mask[c], EXPERTS));
            const size_t exp_bytes = (size_t)k_res * EXPERT_RAW;
            unsigned char *pinned = nullptr;
            CHECK(cudaMallocHost(&pinned, exp_bytes));

            for (int L = l_start[c]; L <= l_end[c]; L++) {
                layer_load_trunk(dataset_dir, L, g_full93.y[L]);
                if (L >= 1) {
                    layer_load_expert_tables(dataset_dir, L, g_full93.y[L]);
                    // Read the 896-float router bias vector (record 30 of trunk-L) from GPU to select the top `k_res` highest-prior experts
                    std::vector<float> h_bias(EXPERTS);
                    CHECK(cudaMemcpy(h_bias.data(), rec_data(g_full93.y[L].T, 30), EXPERTS * sizeof(float), cudaMemcpyDeviceToHost));
                    std::vector<int> order(EXPERTS);
                    for (int e = 0; e < EXPERTS; e++) order[e] = e;
                    std::partial_sort(order.begin(), order.begin() + k_res, order.end(),
                                      [&](int a, int b) { return h_bias[a] > h_bias[b]; });
                    std::vector<int> chosen(order.begin(), order.begin() + k_res);
                    std::sort(chosen.begin(), chosen.end());
                    g_full93.h_res_ids[L] = chosen;
                    CHECK(cudaMalloc(&g_full93.d_res_ids[L], k_res * sizeof(int)));
                    CHECK(cudaMemcpy(g_full93.d_res_ids[L], chosen.data(), k_res * sizeof(int), cudaMemcpyHostToDevice));

                    CHECK(cudaMalloc(&g_full93.d_res_exp[L], exp_bytes));
                    char path[4096];
                    snprintf(path, sizeof path, "%s/expert-%d/experts.direct", dataset_dir, L);
                    int fd = open(path, O_RDONLY);
                    if (fd < 0) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
                    g_full93.fd[L] = fd;  // Keep open for full 896-expert dynamic routing (`run_moe_layer_full896`)
                    std::vector<const unsigned char *> h_ptrs(EXPERTS, g_full93.d_res_exp[L]);
                    g_full93.h_res_map[L].assign(EXPERTS, nullptr);
                    for (int j = 0; j < k_res; j++) {
                        const int e = chosen[j];
                        unsigned char *dst_p = pinned + (size_t)j * EXPERT_RAW;
                        size_t got = 0;
                        while (got < (size_t)EXPERT_RAW) {
                            ssize_t r = pread(fd, dst_p + got, EXPERT_RAW - got, (off_t)e * EXPERT_RAW + (off_t)got);
                            if (r <= 0) { fprintf(stderr, "read of %s failed\n", path); exit(1); }
                            got += (size_t)r;
                        }
                        const unsigned char *dev_ptr = g_full93.d_res_exp[L] + (size_t)j * EXPERT_RAW;
                        h_ptrs[e] = dev_ptr;
                        g_full93.h_res_map[L][e] = dev_ptr;
                    }
                    CHECK(cudaMemcpy(g_full93.d_res_exp[L], pinned, exp_bytes, cudaMemcpyHostToDevice));
                    CHECK(cudaMalloc(&g_full93.d_eptr[L], EXPERTS * sizeof(void *)));
                    CHECK(cudaMemcpy(g_full93.d_eptr[L], h_ptrs.data(), EXPERTS * sizeof(void *), cudaMemcpyHostToDevice));
                }
            }
            CHECK(cudaFreeHost(pinned));
        });
    }
    for (auto &th : threads) th.join();

    // Initialize dynamic VRAM `CardPool` on remaining VRAM (keeping 2.8 GiB free headroom per card for MLA tiled scratch + PyTorch).
    // Done on the main thread after trunk+resident expert loading finishes so `pi_pool_init` (`K3GPU_POOLINIT=rust`) obeys its single-caller-thread contract.
    for (int c = 0; c < 6; c++) {
        CHECK(cudaSetDevice(c));
        g_full93.pool_card[c].init(c, 2800ULL << 20, 16);
        size_t f_b = 0, t_b = 0;
        CHECK(cudaMemGetInfo(&f_b, &t_b));
        fprintf(stderr, "[K3-Full93] Card %d: loaded Layers %2d..%2d (trunks + %d perm experts/layer + %d dyn VRAM slots) | VRAM %.2f / %.2f GiB\n",
                c, l_start[c], l_end[c], k_res, g_full93.pool_card[c].slots, (t_b - f_b) / 1073741824.0, t_b / 1073741824.0);
    }

    int total_dyn_slots = 0;
    for (int c = 0; c < 6; c++) total_dyn_slots += g_full93.pool_card[c].slots;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    g_full93.initialized = true;
    fprintf(stderr, "[K3-Full93] Ready: ALL 93 Trunks (L0..L92) + 92 MoE Layers (896 experts/layer routable: %d perm VRAM + %d dyn VRAM + %zu pinned RAM slots) loaded across GPUs 0..5 in %.2fs\n",
            92 * k_res, total_dyn_slots, g_full93.host_cache.cap_slots, seconds(t0, t1));
    return 0;
}

void k3_full93_set_routing_mode(int mode, int k_tok_tail, int k_dyn_max)
{
    std::lock_guard<std::mutex> lock(g_full93.mu);
    g_full93.routing_mode = mode;
    if (k_tok_tail > 0) g_full93.k_tok_tail = k_tok_tail;
    if (k_dyn_max > 0) g_full93.k_dyn_max = k_dyn_max;
}

// Runs `n` token embeddings (`in_embed [n, 7168]`) sequentially through ALL 93 Kimi K3 Layers (`Layer 0` through `Layer 92`)
// across GPUs 0..5 (`L0..15` on `cuda:0`, `L16..31` on `cuda:1`, `L32..46` on `cuda:2`, `L47..61` on `cuda:3`,
// `L62..76` on `cuda:4`, `L77..92` on `cuda:5`).
// Outputs:
//   - `out_hidden`   : `[n, 7168]` float32 final hidden states after Layer 92
//   - `out_snaps`    : `[8, n, 7168]` float32 8 block-boundary snapshots (`snap[0..7]`) for Kimi K3's `transformer_93` tail
//   - `out_uniq_cnts`: `[92]` int32 count of unique routed experts activated at each MoE layer `1..92`
float k3_full93_forward(const float *in_embed, int n, float *out_hidden, float *out_snaps, int *out_uniq_cnts)
{
    std::lock_guard<std::mutex> lock(g_full93.mu);
    if (!g_full93.initialized || n <= 0 || n > g_full93.max_n) return -1.0f;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    const size_t bytes = (size_t)n * W * sizeof(float);
    memcpy(g_full93.h_snaps[0].data(), in_embed, bytes);
    int nsnap = 1;

    const int l_start[6] = {0, 16, 32, 47, 62, 77};
    const int l_end[6]   = {15, 31, 46, 61, 76, 92};

    for (int c = 0; c < 6; c++) {
        CHECK(cudaSetDevice(c));
        Work &w = g_full93.w_card[c];
        w.N = n;
        w.moe_chunk = std::max(1, std::min(n, 2048));
        w.SLOTS = w.moe_chunk * TOP;

        if (c == 0) {
            CHECK(cudaMemcpy(w.snap[0], in_embed, bytes, cudaMemcpyHostToDevice));
            run_layer0(w, g_full93.y[0], w.snap[0], w.cur);
        } else {
            CHECK(cudaMemcpy(w.cur, g_full93.h_cur.data(), bytes, cudaMemcpyHostToDevice));
            for (int k = 0; k < nsnap; k++) {
                CHECK(cudaMemcpy(w.snap[k], g_full93.h_snaps[k].data(), bytes, cudaMemcpyHostToDevice));
            }
        }

        const int first_moe = (c == 0) ? 1 : l_start[c];
        for (int L = first_moe; L <= l_end[c]; L++) {
            const float *sn[9];
            for (int k = 0; k < nsnap; k++) sn[k] = w.snap[k];
            int u_cnt = 0;
            if (g_full93.routing_mode > 0) {
                run_moe_layer_full896(w, g_full93.y[L], g_full93.pool_card[c], g_full93.host_cache, g_full93.fd[L],
                                      g_full93.h_res_map[L], g_full93.d_vram_mask[c],
                                      g_full93.routing_mode, g_full93.k_tok_tail, g_full93.k_dyn_max,
                                      w.cur, sn, nsnap, w.nxt, &u_cnt);
            } else {
                run_moe_layer_resident(w, g_full93.y[L], g_full93.d_eptr[L], g_full93.d_res_ids[L], g_full93.h_res_ids[L],
                                       g_full93.k_res, w.cur, sn, nsnap, w.nxt, &u_cnt);
            }
            if (out_uniq_cnts) out_uniq_cnts[L - 1] = u_cnt;
            if (L % 12 == 0 && nsnap < 8) {
                CHECK(cudaMemcpy(w.snap[nsnap], w.cur, bytes, cudaMemcpyDeviceToDevice));
                CHECK(cudaMemcpy(g_full93.h_snaps[nsnap].data(), w.cur, bytes, cudaMemcpyDeviceToHost));
                nsnap++;
            }
            std::swap(w.cur, w.nxt);
        }
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(g_full93.h_cur.data(), w.cur, bytes, cudaMemcpyDeviceToHost));
    }

    if (out_hidden) memcpy(out_hidden, g_full93.h_cur.data(), bytes);
    if (out_snaps) {
        for (int k = 0; k < 8; k++) {
            memcpy(out_snaps + (size_t)k * n * W, g_full93.h_snaps[k].data(), bytes);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    return (float)(seconds(t0, t1) * 1000.0);
}

// Reports whether Rust host-path libraries (`slotcache`, `poolinit`, `supply`) are active and returns their call/hit statistics.
// `out_stats` (if non-null) receives 10 `unsigned long` entries:
//   [0] g_sc.on (0=cpp, 1=rust), [1] g_pi.on (0=cpp, 1=rust), [2] g_sy.on (0=cpp, 1=rust),
//   [3] g_sc_calls[0] (acquire), [4] g_sc_calls[1] (try_acquire), [5] g_sc_calls[2] (find),
//   [6] total needed across cards 0..5, [7] total hit_card, [8] total hit_host, [9] total from_drive.
void k3_full93_rust_stats(unsigned long *out_stats)
{
    std::lock_guard<std::mutex> lock(g_full93.mu);
    if (!out_stats) return;
    out_stats[0] = g_sc.on ? 1UL : 0UL;
    out_stats[1] = g_pi.on ? 1UL : 0UL;
    out_stats[2] = g_sy.on ? 1UL : 0UL;
    out_stats[3] = g_sc_calls[0];
    out_stats[4] = g_sc_calls[1];
    out_stats[5] = g_sc_calls[2];
    unsigned long need = 0, hc = 0, hh = 0, dr = 0;
    for (int c = 0; c < 6; c++) {
        need += g_full93.pool_card[c].st.needed;
        hc   += g_full93.pool_card[c].st.hit_card;
        hh   += g_full93.pool_card[c].st.hit_host;
        dr   += g_full93.pool_card[c].st.from_drive;
    }
    out_stats[6] = need;
    out_stats[7] = hc;
    out_stats[8] = hh;
    out_stats[9] = dr;
}

// Returns 12 `unsigned long` entries for the 4-Tier Data-Movement & Red-Blue Pebble I/O Ledger:
//   [0] g_policy (0=LRU, 1=Heat-48, 2=Belady, 3=SpecMD Least-Stale LS + Clean R4)
//   [1] perm_hits (Tier 1 permanent VRAM expert hits across L1..92)
//   [2] dyn_needed (Total dynamic expert requests across cards 0..5)
//   [3] hit_card (R3: Tier 1 dynamic CardPool VRAM hits)
//   [4] hit_host (R1_host: Tier 2 Pinned HostCache -> VRAM H2D transfers)
//   [5] from_drive (R1_nvme: Tier 3 NVMe O_DIRECT -> VRAM transfers)
//   [6] written_back (R2_wb: Tier 1 -> Tier 2 D2H writebacks)
//   [7] wb_skipped_r4 (R4: 0-byte clean drops / skipped D2H writebacks)
//   [8] total_card_slots (Sum of dynamic CardPool slots across cards 0..5)
//   [9] host_cap_slots (Total pinned HostCache slots)
void k3_full93_dm_stats(unsigned long *out_dm)
{
    std::lock_guard<std::mutex> lock(g_full93.mu);
    if (!out_dm) return;
    unsigned long need = 0, hc = 0, hh = 0, dr = 0, wb = 0, cslots = 0;
    for (int c = 0; c < 6; c++) {
        need   += g_full93.pool_card[c].st.needed;
        hc     += g_full93.pool_card[c].st.hit_card;
        hh     += g_full93.pool_card[c].st.hit_host;
        dr     += g_full93.pool_card[c].st.from_drive;
        wb     += g_full93.pool_card[c].st.written_back;
        cslots += (unsigned long)g_full93.pool_card[c].slots;
    }
    unsigned long r4_skip = g_wb_skipped;
#ifdef K3GPU_SUPPLY_BUILT
    if (g_sy.on && g_sy.wb_skipped) r4_skip = g_sy.wb_skipped();
#endif
    out_dm[0] = (unsigned long)g_policy;
    out_dm[1] = g_full93.perm_hits;
    out_dm[2] = need;
    out_dm[3] = hc;
    out_dm[4] = hh;
    out_dm[5] = dr;
    out_dm[6] = wb;
    out_dm[7] = r4_skip;
    out_dm[8] = cslots;
    out_dm[9] = (unsigned long)g_full93.host_cache.cap_slots;
}

} // extern "C"
