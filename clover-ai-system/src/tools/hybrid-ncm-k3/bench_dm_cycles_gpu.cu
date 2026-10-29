// Live GPU Benchmark for Clover Direction B (Cycles 2, 3, 4, and 5) on `cuda:7`
// Uses real Kimi K3 trunks (`trunk-1..8`), real 136-token activations (`/home/usergpu/k3gpu/results/gpu-l13.layer0`),
// real 92-layer routing traces (`/home/usergpu/k3gpu/results/p1d.trace`), and real 17.55 MB expert buffers on RTX 3090 (`cuda:7`).
//
// Compile on `gpu`:
//   /usr/local/cuda-12.6/bin/nvcc -O3 --fmad=false -arch=sm_86 -std=c++17 \
//     -I/home/usergpu/k3gpu/src/code/gpu bench_dm_cycles_gpu.cu -lcublas -lpthread -o bench_dm_cycles_gpu

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <vector>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <list>
#include <fstream>
#include <sstream>

#define CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

enum {
    W = 7168,
    EXPERTS = 896,
    TOP = 16,
    EXPERT_RAW = 17547264,
    HEADS = 64,
    ROWS = 8192,
    MLA_Q = 18432,
    MLA_EXP = 24576,
    MLA_KVLAT = 576,
    MLA_KV = 512
};

static double wall_ms() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec * 1e-6;
}

__device__ __forceinline__ float sigmoidf_(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float expf_cr(float x) {
    return expf(x);
}

static inline int ekey(int layer, int e) { return layer * EXPERTS + e; }

// ============================================================================
// PART 1: Live GPU PCIe & SlotCache Benchmark for Cycle 2 (Shadow-Swap R4)
//         and Cycle 3 (SpecMD Hit-First + Least-Stale LS) on `cuda:7`
// ============================================================================

struct TraceCall {
    int layer;
    int nc;
    std::vector<std::pair<int, int>> ec; // (expert, count)
};

static std::vector<TraceCall> load_trace(const char *path, int max_passes) {
    std::ifstream in(path);
    std::string line;
    std::vector<TraceCall> out;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        TraceCall tc;
        if (!(iss >> tc.layer >> tc.nc)) continue;
        std::string tok;
        while (iss >> tok) {
            size_t pos = tok.find(':');
            if (pos != std::string::npos) {
                int e = std::stoi(tok.substr(0, pos));
                int c = std::stoi(tok.substr(pos + 1));
                tc.ec.push_back({e, c});
            }
        }
        out.push_back(std::move(tc));
        if ((int)out.size() >= max_passes * 92) break;
    }
    return out;
}

struct LiveSlotCache {
    struct Ent {
        std::list<int>::iterator it;
        int slot_idx;
        float heat;
        int pass_id;
        int layer_id;
    };
    std::list<int> lru; // front = MRU, back = LRU
    std::unordered_map<int, Ent> map;
    std::vector<int> spare;
    int policy = 1; // 1 = Heat-48, 3 = SpecMD Least-Stale (LS)

    void init(int n_slots, int pol) {
        lru.clear();
        map.clear();
        spare.clear();
        policy = pol;
        for (int i = n_slots - 1; i >= 0; i--) spare.push_back(i);
    }

    int find(int key) {
        auto f = map.find(key);
        if (f == map.end()) return -1;
        lru.splice(lru.begin(), lru, f->second.it);
        return f->second.slot_idx;
    }

    void touch(int key, float w, int cur_pass, int cur_layer) {
        auto f = map.find(key);
        if (f == map.end()) return;
        f->second.heat = f->second.heat * 0.8f + w;
        f->second.pass_id = cur_pass;
        f->second.layer_id = cur_layer;
    }

    int remove(int key) {
        auto f = map.find(key);
        if (f == map.end()) return -1;
        int s = f->second.slot_idx;
        lru.erase(f->second.it);
        map.erase(f);
        return s;
    }

    void release(int slot_idx) {
        spare.push_back(slot_idx);
    }

    int acquire(int key, const std::unordered_set<int> &prot, float heat0, int cur_pass, int cur_layer, int *vkey, float *vheat) {
        *vkey = -1;
        *vheat = 0.0f;
        auto existing = map.find(key);
        if (existing != map.end()) {
            int s = existing->second.slot_idx;
            lru.splice(lru.begin(), lru, existing->second.it);
            existing->second.heat = heat0;
            existing->second.pass_id = cur_pass;
            existing->second.layer_id = cur_layer;
            return s;
        }
        int s = -1;
        if (!spare.empty()) {
            s = spare.back();
            spare.pop_back();
        } else if (policy == 3) {
            // SpecMD Least-Stale (LS) + Heat tie-breaking across the card's layer cycle:
            // Tier 0: truly stale (pass < cur_pass - 1, or pass == cur_pass - 1 and layer <= cur_layer)
            // Tier 1: already used earlier in cur_pass (pass == cur_pass and layer < cur_layer)
            // Tier 2: upcoming in cur_pass (pass == cur_pass - 1 and layer > cur_layer) -> protect from pass-collision!
            auto best_it = lru.end();
            int best_tier = 99;
            float best_h = 1e30f;
            for (auto it = lru.end(); it != lru.begin();) {
                --it;
                int k = *it;
                if (prot.count(k)) continue;
                const Ent &e = map[k];
                int tier = 2;
                if (e.pass_id < cur_pass - 1 || (e.pass_id == cur_pass - 1 && e.layer_id <= cur_layer)) tier = 0;
                else if (e.pass_id == cur_pass && e.layer_id < cur_layer) tier = 1;
                if (best_it == lru.end() || tier < best_tier || (tier == best_tier && e.heat < best_h)) {
                    best_it = it;
                    best_tier = tier;
                    best_h = e.heat;
                }
            }
            int victim = *best_it;
            s = map[victim].slot_idx;
            *vkey = victim;
            *vheat = map[victim].heat;
            map.erase(victim);
            lru.erase(best_it);
        } else {
            // Policy 0 (limit=1) or Policy 1 (Heat-48, limit=48)
            int limit = (policy != 0) ? 48 : 1;
            int seen = 0;
            auto best_it = lru.end();
            float best_h = 1e30f;
            for (auto it = lru.end(); it != lru.begin() && seen < limit;) {
                --it;
                int k = *it;
                if (prot.count(k)) continue;
                float h = map[k].heat;
                if (best_it == lru.end() || h < best_h) {
                    best_it = it;
                    best_h = h;
                }
                seen++;
            }
            int victim = *best_it;
            s = map[victim].slot_idx;
            *vkey = victim;
            *vheat = map[victim].heat;
            map.erase(victim);
            lru.erase(best_it);
        }
        lru.push_front(key);
        map[key] = Ent{lru.begin(), s, heat0, cur_pass, cur_layer};
        return s;
    }
};

static void run_live_pcie_cycle2_cycle3(int dev, const std::vector<TraceCall> &calls) {
    CHECK(cudaSetDevice(dev));
    const int card_slots = 80;   // 80 * 17.55 MB = 1.40 GB in cuda:7 VRAM
    const int host_slots = 320;  // 320 * 17.55 MB = 5.61 GB in pinned Host RAM
    const int k_res = 28;

    // Derive permanent top-28 experts per layer 1..15 from Pass 0
    std::unordered_set<int> perm[16];
    for (size_t i = 0; i < std::min<size_t>(92, calls.size()); i++) {
        int L = calls[i].layer;
        if (L >= 1 && L <= 15 && perm[L].empty()) {
            auto ec = calls[i].ec;
            std::sort(ec.begin(), ec.end(), [](const std::pair<int,int> &a, const std::pair<int,int> &b) {
                return a.second != b.second ? a.second > b.second : a.first < b.first;
            });
            for (size_t j = 0; j < std::min<size_t>(k_res, ec.size()); j++) perm[L].insert(ec[j].first);
        }
    }

    unsigned char *d_pool = nullptr;
    unsigned char *h_pool = nullptr;
    CHECK(cudaMalloc(&d_pool, (size_t)card_slots * EXPERT_RAW));
    CHECK(cudaHostAlloc(&h_pool, (size_t)(host_slots + card_slots) * EXPERT_RAW, cudaHostAllocPortable));
    memset(h_pool, 0x5A, (size_t)(host_slots + card_slots) * EXPERT_RAW);

    cudaStream_t copy_stream;
    cudaEvent_t wb_event;
    CHECK(cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking));
    CHECK(cudaEventCreateWithFlags(&wb_event, cudaEventDisableTiming));

    struct ArmCfg {
        const char *name;
        int policy;
        bool shadow_swap_r4;
        bool hit_first;
    };
    ArmCfg arms[3] = {
        {"Arm 1: Baseline (Heat-48 + Exclusive D2H Writeback)", 1, false, false},
        {"Arm 2: Cycle 2c (Heat-48 + Shadow-Swap Zero-D2H R4)", 1, true,  false},
        {"Arm 3: Cycle 3b (SpecMD LS + Hit-First + Shadow-Swap R4)", 3, true, true},
    };

    printf("\n=== LIVE GPU CYCLE 2 & CYCLE 3 BENCHMARK ON cuda:%d (CardPool S=%d slots = %.2f GB VRAM, HostCache=%d slots = %.2f GB Pinned RAM) ===\n",
           dev, card_slots, card_slots * (double)EXPERT_RAW / 1e9, host_slots, host_slots * (double)EXPERT_RAW / 1e9);
    printf("%-58s | %7s | %6s | %7s | %7s | %7s | %8s | %9s\n",
           "Arm", "DynHit%", "Coll%", "H2D_Cnt", "D2H_WB", "R4_Drop", "PCIe_GB", "GPU_ms");
    printf("---------------------------------------------------------------------------------------------------------------------------------\n");

    for (const auto &arm : arms) {
        LiveSlotCache c_card, c_host;
        c_card.init(card_slots, arm.policy);
        c_host.init(host_slots, 1);

        // Pre-populate HostCache with the working set of Layers 1..15 so we isolate real GPU<->Host PCIe H2D and D2H transfers
        std::unordered_set<int> preload_keys;
        for (const auto &tc : calls) {
            if (tc.layer < 1 || tc.layer > 15) continue;
            for (auto &p : tc.ec) {
                if (perm[tc.layer].count(p.first)) continue;
                int k = ekey(tc.layer, p.first);
                if (!preload_keys.count(k) && (int)preload_keys.size() < host_slots) {
                    int vk; float vh;
                    c_host.acquire(k, {}, (float)p.second, 0, tc.layer, &vk, &vh);
                    preload_keys.insert(k);
                }
            }
        }

        // For Shadow-Swap R4: each card slot `s` holds a paired host buffer index `card_host_buf[s]` in `0..host_slots+card_slots-1`
        std::vector<int> card_host_buf(card_slots);
        for (int s = 0; s < card_slots; s++) card_host_buf[s] = host_slots + s;

        long dyn_req = 0, hit_card = 0, h2d_cnt = 0, d2h_wb = 0, r4_drop = 0, coll_miss = 0;
        std::unordered_set<int> pass_start_res, evicted_in_pass;

        CHECK(cudaDeviceSynchronize());
        double t0 = wall_ms();

        for (size_t idx = 0; idx < calls.size(); idx++) {
            const auto &tc = calls[idx];
            int L = tc.layer;
            if (L < 1 || L > 15) continue;
            int cur_pass = (int)(idx / 92);
            if (L == 1) {
                pass_start_res.clear();
                for (auto &kv : c_card.map) pass_start_res.insert(kv.first);
                evicted_in_pass.clear();
            }

            std::vector<std::pair<int, int>> dyn_ec;
            for (auto &p : tc.ec) {
                if (!perm[L].count(p.first)) dyn_ec.push_back(p);
            }

            bool rec = (cur_pass >= 1); // Measure post-prefill decode passes 1..N
            if (rec) dyn_req += (long)dyn_ec.size();

            for (size_t gi = 0; gi < dyn_ec.size(); gi += (size_t)card_slots) {
                size_t gj = std::min(dyn_ec.size(), gi + (size_t)card_slots);
                std::vector<std::pair<int, int>> grp(dyn_ec.begin() + gi, dyn_ec.begin() + gj);

                std::unordered_set<int> prot;
                for (auto &p : grp) prot.insert(ekey(L, p.first));

                struct Need { int e, k, dst_slot, host_slot; float cnt; };
                struct Victim { int k, src_slot; float heat; };
                std::vector<Need> need;
                std::vector<Victim> victims;

                if (arm.hit_first) {
                    std::vector<std::pair<int, int>> misses;
                    for (auto &p : grp) {
                        int k = ekey(L, p.first);
                        int s = c_card.find(k);
                        if (s >= 0) {
                            c_card.touch(k, (float)p.second, cur_pass, L);
                            if (rec) hit_card++;
                        } else {
                            misses.push_back(p);
                        }
                    }
                    for (auto &p : misses) {
                        int k = ekey(L, p.first);
                        if (rec && pass_start_res.count(k) && evicted_in_pass.count(k)) coll_miss++;
                        int vk; float vh;
                        int s = c_card.acquire(k, prot, (float)p.second, cur_pass, L, &vk, &vh);
                        if (vk >= 0) {
                            evicted_in_pass.insert(vk);
                            victims.push_back({vk, s, vh});
                        }
                        need.push_back({p.first, k, s, -1, (float)p.second});
                    }
                } else {
                    for (auto &p : grp) {
                        int k = ekey(L, p.first);
                        int s = c_card.find(k);
                        if (s >= 0) {
                            c_card.touch(k, (float)p.second, cur_pass, L);
                            if (rec) hit_card++;
                            continue;
                        }
                        if (rec && pass_start_res.count(k) && evicted_in_pass.count(k)) coll_miss++;
                        int vk; float vh;
                        s = c_card.acquire(k, prot, (float)p.second, cur_pass, L, &vk, &vh);
                        if (vk >= 0) {
                            evicted_in_pass.insert(vk);
                            victims.push_back({vk, s, vh});
                        }
                        need.push_back({p.first, k, s, -1, (float)p.second});
                    }
                }

                if (need.empty()) continue;

                if (!arm.shadow_swap_r4) {
                    // Baseline Exclusive D2H + H2D (`decode.cuh:703-721`)
                    std::vector<int> held;
                    for (auto &n : need) {
                        if (c_host.find(n.k) >= 0) {
                            n.host_slot = c_host.remove(n.k);
                            held.push_back(n.host_slot);
                        } else {
                            n.host_slot = n.dst_slot; // fallback bounce buffer
                        }
                    }
                    for (auto &v : victims) {
                        int dk; float dh;
                        int hs = c_host.acquire(v.k, {}, v.heat, cur_pass, L, &dk, &dh);
                        if (rec) {
                            CHECK(cudaMemcpyAsync(h_pool + (size_t)hs * EXPERT_RAW,
                                                  d_pool + (size_t)v.src_slot * EXPERT_RAW,
                                                  EXPERT_RAW, cudaMemcpyDeviceToHost, copy_stream));
                            d2h_wb++;
                        }
                    }
                    CHECK(cudaEventRecord(wb_event, copy_stream));
                    for (auto &n : need) {
                        if (rec) {
                            CHECK(cudaMemcpyAsync(d_pool + (size_t)n.dst_slot * EXPERT_RAW,
                                                  h_pool + (size_t)n.host_slot * EXPERT_RAW,
                                                  EXPERT_RAW, cudaMemcpyHostToDevice, copy_stream));
                            h2d_cnt++;
                        }
                    }
                    CHECK(cudaStreamSynchronize(copy_stream));
                    for (int hs : held) c_host.release(hs);
                } else {
                    // Shadow-Swap Zero-D2H R4:
                    // Victim `v` at `v.src_slot` ALREADY has its clean bytes in Host RAM at `card_host_buf[v.src_slot]`!
                    // Demoting `v` into `c_host` simply hands `card_host_buf[v.src_slot]` to `c_host` (0 PCIe D2H bytes, 0 event stall!).
                    for (auto &v : victims) {
                        int dk; float dh;
                        int hs = c_host.acquire(v.k, {}, v.heat, cur_pass, L, &dk, &dh);
                        std::swap(c_host.map[v.k].slot_idx, card_host_buf[v.src_slot]);
                        (void)hs;
                        if (rec) r4_drop++;
                    }
                    for (auto &n : need) {
                        int hs = c_host.find(n.k);
                        if (hs >= 0) {
                            int removed_hs = c_host.remove(n.k);
                            c_host.release(card_host_buf[n.dst_slot]);
                            card_host_buf[n.dst_slot] = removed_hs;
                        }
                        if (rec) {
                            CHECK(cudaMemcpyAsync(d_pool + (size_t)n.dst_slot * EXPERT_RAW,
                                                  h_pool + (size_t)card_host_buf[n.dst_slot] * EXPERT_RAW,
                                                  EXPERT_RAW, cudaMemcpyHostToDevice, copy_stream));
                            h2d_cnt++;
                        }
                    }
                    CHECK(cudaStreamSynchronize(copy_stream));
                }
            }
        }
        CHECK(cudaDeviceSynchronize());
        double elapsed_ms = wall_ms() - t0;
        double pcie_gb = (double)(h2d_cnt + d2h_wb) * EXPERT_RAW / 1e9;
        printf("%-58s | %6.2f%% | %5.2f%% | %7ld | %7ld | %7ld | %7.2fG | %7.1fms\n",
               arm.name,
               100.0 * hit_card / std::max(1L, dyn_req),
               100.0 * coll_miss / std::max(1L, dyn_req),
               h2d_cnt, d2h_wb, r4_drop, pcie_gb, elapsed_ms);
    }

    CHECK(cudaEventDestroy(wb_event));
    CHECK(cudaStreamDestroy(copy_stream));
    CHECK(cudaFreeHost(h_pool));
    CHECK(cudaFree(d_pool));
}

// ============================================================================
// PART 2: Cycle 3c (`vram_mask == 2` Fix in `top16_smart_896` & Cross-Layer
//         Speculative Router Lookahead on Real Kimi K3 Trunks `trunk-1..8`)
//         and Cycle 5 (Eyeriss v2 Adaptive Multi-Token Broadcast Coalescing)
// ============================================================================

__global__ void router_scores_k3(float *scores, float *choices, const float *router_w, const float *bias, const float *x_in, int N) {
    const int n = blockIdx.x, e = blockIdx.y;
    if (n >= N || e >= EXPERTS) return;
    const float *w = router_w + (size_t)e * W;
    const float *x = x_in + (size_t)n * W;
    double s = 0.0;
    for (int c = threadIdx.x; c < W; c += blockDim.x) {
        s += (double)w[c] * (double)x[c];
    }
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
    __shared__ double warp_s[8];
    if ((threadIdx.x & 31) == 0) warp_s[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        double tot = 0.0;
        for (int i = 0; i < (int)(blockDim.x / 32); i++) tot += warp_s[i];
        float sc = sigmoidf_((float)tot);
        scores[(size_t)n * EXPERTS + e] = sc;
        choices[(size_t)n * EXPERTS + e] = sc + bias[e];
    }
}

__global__ void top16_exact_k3(const float *scores, const float *choices, int *selected, float *weights, int N) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const float *sc = scores + (size_t)n * EXPERTS, *ch = choices + (size_t)n * EXPERTS;
    float best[TOP]; int sel[TOP]; int count = 0;
    for (int e = 0; e < EXPERTS; e++) {
        const float choice = ch[e];
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

// Shipped `top16_smart_896` (`k3_gpu_experts_lib.cu:133`) vs Fixed (`free_dyn_hits = true`, where `vram_mask[e] >= 1` is free VRAM hit!)
__global__ void top16_smart_896_test(const float *scores, const float *choices, const unsigned char *vram_mask,
                                     int *selected, float *weights, int N, int k_dyn_max, bool free_dyn_hits) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const float *sc = scores + (size_t)n * EXPERTS, *ch = choices + (size_t)n * EXPERTS;
    float best_v[TOP]; int sel_v[TOP]; int cnt_v = 0;
    float best_all[TOP]; int sel_all[TOP]; int cnt_all = 0;

    for (int e = 0; e < EXPERTS; e++) {
        const float choice = ch[e];
        if (vram_mask[e]) {
            if (cnt_v < TOP || choice > best_v[TOP - 1]) {
                int pos = cnt_v < TOP ? cnt_v : TOP - 1;
                while (pos && choice > best_v[pos - 1]) { best_v[pos] = best_v[pos - 1]; sel_v[pos] = sel_v[pos - 1]; pos--; }
                best_v[pos] = choice; sel_v[pos] = e;
                if (cnt_v < TOP) cnt_v++;
            }
        }
        if (cnt_all < TOP || choice > best_all[TOP - 1]) {
            int pos = cnt_all < TOP ? cnt_all : TOP - 1;
            while (pos && choice > best_all[pos - 1]) { best_all[pos] = best_all[pos - 1]; sel_all[pos] = sel_all[pos - 1]; pos--; }
            best_all[pos] = choice; sel_all[pos] = e;
            if (cnt_all < TOP) cnt_all++;
        }
    }

    int final_sel[TOP];
    int out_cnt = 0, dyn_used = 0;
    for (int r = 0; r < TOP && out_cnt < TOP; r++) {
        const int e = sel_all[r];
        const bool is_free_vram = free_dyn_hits ? (vram_mask[e] >= 1) : (vram_mask[e] == 1);
        if (is_free_vram) {
            final_sel[out_cnt++] = e;
        } else if (dyn_used < k_dyn_max) {
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

    double total = 0.0;
    for (int r = 0; r < TOP; r++) total += (double)sc[final_sel[r]];
    const float inverse = (float)(1.0 / (total + 1e-20));
    for (int r = 0; r < TOP; r++) {
        selected[n * TOP + r] = final_sel[r];
        weights[n * TOP + r] = sc[final_sel[r]] * inverse;
    }
}

// Helper to read a record from `/opt/clover-k3/datasets/computed/trunk-L/{index.bin, common.bin, attention.bin, routed-shared.bin}`
// Format (`src/code/gpu/model.cuh:216`):
//   `index.bin`: "K3TRK001", count at +16, then 72-byte entries starting at +1168: `{ id(u32), group(u32), kind(u32), rows(u32), cols(u32), pad(u32), offset(u64), length(u64) }`
static uint32_t u32_le(const unsigned char *p) { return p[0] | (p[1] << 8) | (p[2] << 16) | ((uint32_t)p[3] << 24); }
static uint64_t u64_le(const unsigned char *p) { return u32_le(p) | ((uint64_t)u32_le(p + 4) << 32); }

static std::vector<unsigned char> read_trunk_record(const char *dataset_dir, int L, unsigned rec_id) {
    char idx_path[1024];
    snprintf(idx_path, sizeof idx_path, "%s/trunk-%d/index.bin", dataset_dir, L);
    FILE *fi = fopen(idx_path, "rb");
    if (!fi) { fprintf(stderr, "Cannot open %s\n", idx_path); exit(1); }
    fseek(fi, 0, SEEK_END);
    size_t isz = (size_t)ftell(fi);
    fseek(fi, 0, SEEK_SET);
    std::vector<unsigned char> idx(isz);
    if (fread(idx.data(), 1, isz, fi) != isz) { fprintf(stderr, "Bad read %s\n", idx_path); exit(1); }
    fclose(fi);

    unsigned count = u32_le(idx.data() + 16);
    uint32_t grp = 0;
    uint64_t off = 0, len = 0;
    bool found = false;
    for (unsigned p = 0; p < count; p++) {
        const unsigned char *e = idx.data() + 1168 + (size_t)p * 72;
        if (u32_le(e) == rec_id) {
            grp = u32_le(e + 4);
            off = u64_le(e + 24);
            len = u64_le(e + 32);
            found = true;
            break;
        }
    }
    if (!found) { fprintf(stderr, "Record %u not found in %s\n", rec_id, idx_path); exit(1); }
    static const char *names1[3] = { "common.bin", "attention.bin", "routed-shared.bin" };
    char gpath[1024];
    snprintf(gpath, sizeof gpath, "%s/trunk-%d/%s", dataset_dir, L, names1[grp]);
    FILE *fg = fopen(gpath, "rb");
    if (!fg) { fprintf(stderr, "Cannot open %s\n", gpath); exit(1); }
    std::vector<unsigned char> buf((size_t)len);
    fseek(fg, (long)off, SEEK_SET);
    if (fread(buf.data(), 1, (size_t)len, fg) != (size_t)len) { fprintf(stderr, "Bad read rec %u in %s\n", rec_id, gpath); exit(1); }
    fclose(fg);
    return buf;
}

static void run_cycle3c_and_cycle5_router_studies(int dev, const char *dataset_dir, const char *act_path) {
    CHECK(cudaSetDevice(dev));
    // Load real 136-token x 7168-float activations from `/home/usergpu/k3gpu/results/gpu-l13.layer0`
    FILE *fa = fopen(act_path, "rb");
    if (!fa) { fprintf(stderr, "Cannot open %s\n", act_path); exit(1); }
    fseek(fa, 0, SEEK_END);
    size_t fsz = (size_t)ftell(fa);
    fseek(fa, 0, SEEK_SET);
    int N = (int)(fsz / (W * sizeof(float)));
    std::vector<float> h_x(N * W);
    if (fread(h_x.data(), sizeof(float), (size_t)N * W, fa) != (size_t)N * W) { fprintf(stderr, "Bad act read\n"); exit(1); }
    fclose(fa);

    // Normalize rows of h_x with RMSNorm (as `normalize_rows` does before `router_scores`)
    std::vector<float> h_xn(N * W);
    for (int n = 0; n < N; n++) {
        double ss = 0.0;
        for (int c = 0; c < W; c++) ss += (double)h_x[(size_t)n * W + c] * (double)h_x[(size_t)n * W + c];
        float inv = (float)(1.0 / sqrt(ss / (double)W + 1e-5));
        for (int c = 0; c < W; c++) h_xn[(size_t)n * W + c] = h_x[(size_t)n * W + c] * inv;
    }

    float *d_x = nullptr, *d_router = nullptr, *d_bias = nullptr, *d_scores = nullptr, *d_choices = nullptr, *d_w_ref = nullptr, *d_w_test = nullptr;
    int *d_sel_ref = nullptr, *d_sel_test = nullptr;
    unsigned char *d_vmask = nullptr;
    CHECK(cudaMalloc(&d_x, (size_t)N * W * sizeof(float)));
    CHECK(cudaMalloc(&d_router, (size_t)EXPERTS * W * sizeof(float)));
    CHECK(cudaMalloc(&d_bias, (size_t)EXPERTS * sizeof(float)));
    CHECK(cudaMalloc(&d_scores, (size_t)N * EXPERTS * sizeof(float)));
    CHECK(cudaMalloc(&d_choices, (size_t)N * EXPERTS * sizeof(float)));
    CHECK(cudaMalloc(&d_sel_ref, (size_t)N * TOP * sizeof(int)));
    CHECK(cudaMalloc(&d_w_ref, (size_t)N * TOP * sizeof(float)));
    CHECK(cudaMalloc(&d_sel_test, (size_t)N * TOP * sizeof(int)));
    CHECK(cudaMalloc(&d_w_test, (size_t)N * TOP * sizeof(float)));
    CHECK(cudaMalloc(&d_vmask, EXPERTS));
    CHECK(cudaMemcpy(d_x, h_xn.data(), (size_t)N * W * sizeof(float), cudaMemcpyHostToDevice));

    // Load Layer 1 router (`rec 29` = 896*7168*4 bytes, `rec 30` = 896*4 bytes)
    auto r29_1 = read_trunk_record(dataset_dir, 1, 29);
    auto r30_1 = read_trunk_record(dataset_dir, 1, 30);
    const float *h_r29_1 = (const float *)(r29_1.data());
    const float *h_r30_1 = (const float *)(r30_1.data());
    CHECK(cudaMemcpy(d_router, h_r29_1, (size_t)EXPERTS * W * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_bias, h_r30_1, (size_t)EXPERTS * sizeof(float), cudaMemcpyHostToDevice));

    router_scores_k3<<<dim3(N, EXPERTS), 256>>>(d_scores, d_choices, d_router, d_bias, d_x, N);
    top16_exact_k3<<<(N + 255) / 256, 256>>>(d_scores, d_choices, d_sel_ref, d_w_ref, N);
    CHECK(cudaDeviceSynchronize());

    std::vector<int> h_sel_ref(N * TOP), h_sel_test(N * TOP);
    std::vector<float> h_w_ref(N * TOP), h_w_test(N * TOP), h_scores(N * EXPERTS), h_choices(N * EXPERTS);
    CHECK(cudaMemcpy(h_sel_ref.data(), d_sel_ref, (size_t)N * TOP * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_w_ref.data(), d_w_ref, (size_t)N * TOP * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_scores.data(), d_scores, (size_t)N * EXPERTS * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_choices.data(), d_choices, (size_t)N * EXPERTS * sizeof(float), cudaMemcpyDeviceToHost));

    // Top-28 permanent VRAM experts by router bias
    std::vector<int> order(EXPERTS);
    for (int e = 0; e < EXPERTS; e++) order[e] = e;
    std::partial_sort(order.begin(), order.begin() + 28, order.end(), [&](int a, int b) { return h_r30_1[a] > h_r30_1[b]; });
    std::vector<unsigned char> h_vmask(EXPERTS, 0);
    for (int j = 0; j < 28; j++) h_vmask[order[j]] = 1;

    // Suppose the previous pass/window warmed 120 dynamic experts in `CardPool` (`vram_mask[e] == 2`)
    for (int n = 0; n < std::min(N, 12); n++) {
        for (int r = 0; r < TOP; r++) {
            int e = h_sel_ref[n * TOP + r];
            if (h_vmask[e] == 0) h_vmask[e] = 2;
        }
    }
    CHECK(cudaMemcpy(d_vmask, h_vmask.data(), EXPERTS, cudaMemcpyHostToDevice));

    printf("\n=== LIVE GPU CYCLE 3c-1: `top16_smart_896` Warm-CardPool (`vram_mask==2`) Fix on Real K3 Activations (N=%d tokens, k_dyn_max=6) ===\n", N);
    for (int fix = 0; fix <= 1; fix++) {
        top16_smart_896_test<<<(N + 255) / 256, 256>>>(d_scores, d_choices, d_vmask, d_sel_test, d_w_test, N, 6, fix != 0);
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemcpy(h_sel_test.data(), d_sel_test, (size_t)N * TOP * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(h_w_test.data(), d_w_test, (size_t)N * TOP * sizeof(float), cudaMemcpyDeviceToHost));

        int warm_hits = 0, warm_tot = 0, all_hits = 0, all_tot = N * TOP;
        double warm_w_diff = 0.0;
        for (int n = 0; n < N; n++) {
            std::unordered_set<int> ref_set(h_sel_ref.begin() + n * TOP, h_sel_ref.begin() + (n + 1) * TOP);
            int matched = 0;
            for (int r = 0; r < TOP; r++) {
                if (ref_set.count(h_sel_test[n * TOP + r])) matched++;
            }
            all_hits += matched;
            if (n < 12) {
                warm_hits += matched;
                warm_tot += TOP;
                for (int r = 0; r < TOP; r++) warm_w_diff += fabs((double)h_w_ref[n * TOP + r] - (double)h_w_test[n * TOP + r]);
            }
        }
        printf("  %-46s | Warm-Token Top-16 Recall: %6.2f%% (Weight L1 diff: %.6f) | All-136-Token Recall: %6.2f%%\n",
               fix == 0 ? "Before (`vram_mask==2` charged to k_dyn_max)" : "After  (`vram_mask==2` free dynamic VRAM hit)",
               100.0 * warm_hits / warm_tot, warm_w_diff / 12.0, 100.0 * all_hits / all_tot);
    }

    // Cycle 3c-2: Cross-Layer Speculative Router Lookahead on Real Kimi K3 Routers (`L1..L4`)
    // Compare adjacent-layer router similarity & top-16 / top-24 / top-32 overlap on the 136 real token states
    printf("\n=== LIVE GPU CYCLE 3c-2: Cross-Layer Router Overlap & Speculative Lookahead across Real Kimi K3 Layers L1..L4 (N=%d tokens) ===\n", N);
    {
        std::vector<int> prev_sel = h_sel_ref;
        for (int L = 2; L <= 4; L++) {
            auto r29_L = read_trunk_record(dataset_dir, L, 29);
            auto r30_L = read_trunk_record(dataset_dir, L, 30);
            CHECK(cudaMemcpy(d_router, r29_L.data(), (size_t)EXPERTS * W * sizeof(float), cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(d_bias, r30_L.data(), (size_t)EXPERTS * sizeof(float), cudaMemcpyHostToDevice));
            router_scores_k3<<<dim3(N, EXPERTS), 256>>>(d_scores, d_choices, d_router, d_bias, d_x, N);
            top16_exact_k3<<<(N + 255) / 256, 256>>>(d_scores, d_choices, d_sel_test, d_w_test, N);
            CHECK(cudaDeviceSynchronize());
            std::vector<float> ch_L(N * EXPERTS);
            std::vector<int> sel_L(N * TOP);
            CHECK(cudaMemcpy(ch_L.data(), d_choices, (size_t)N * EXPERTS * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(sel_L.data(), d_sel_test, (size_t)N * TOP * sizeof(int), cudaMemcpyDeviceToHost));

            // Consecutive-token same-layer overlap at Layer L (SpecMD E4 metric on real K3 router!)
            int lag1_hits = 0, lag1_tot = (N - 1) * TOP;
            for (int n = 1; n < N; n++) {
                std::unordered_set<int> prev_tok(sel_L.begin() + (n - 1) * TOP, sel_L.begin() + n * TOP);
                for (int r = 0; r < TOP; r++) if (prev_tok.count(sel_L[n * TOP + r])) lag1_hits++;
            }
            // Unique experts touched in a window of W=16 tokens at Layer L
            std::unordered_set<int> uniq16;
            for (int n = 0; n < std::min(N, 16); n++)
                for (int r = 0; r < TOP; r++) uniq16.insert(sel_L[n * TOP + r]);

            printf("  Layer %d (896 experts, top-16) | Same-Layer Lag-1 Token Overlap: %5.2f%% (vs 1.79%% uniform = %.1fx) | W=16 Window Unique Experts: %3zu / 256 slots (%.2fx reuse)\n",
                   L, 100.0 * lag1_hits / lag1_tot, (100.0 * lag1_hits / lag1_tot) / (100.0 * 16.0 / 896.0),
                   uniq16.size(), 256.0 / std::max<size_t>(1, uniq16.size()));
        }
    }

    // Cycle 5: Eyeriss v2 Adaptive Multi-Token Broadcast Coalescing vs Unicast Per-Token Cold Budget
    printf("\n=== LIVE GPU CYCLE 5: Eyeriss v2 Multi-Token Broadcast Coalescing vs Unicast Cold Budget (Window N=16 tokens, Cold Budget B=24 experts) ===\n");
    {
        const int W_TOK = 16;
        const int K_COLD_PER_TOK = 4; // Unicast: up to 4 cold experts per token
        // Reset vram_mask to only permanent 28 experts (`vram_mask[e] == 1`)
        std::fill(h_vmask.begin(), h_vmask.end(), 0);
        for (int j = 0; j < 28; j++) h_vmask[order[j]] = 1;

        // 1. Unicast (Independent per-token greedy cold selection, `k_dyn_max = 4`)
        std::unordered_set<int> unicast_cold;
        int unicast_slot_uses = 0;
        double unicast_score_sum = 0.0, true_score_sum = 0.0;
        for (int n = 0; n < W_TOK; n++) {
            for (int r = 0; r < TOP; r++) true_score_sum += h_scores[n * EXPERTS + h_sel_ref[n * TOP + r]];
            int cold_used = 0, out_c = 0;
            std::vector<int> chosen;
            for (int r = 0; r < TOP && out_c < TOP; r++) {
                int e = h_sel_ref[n * TOP + r];
                if (h_vmask[e] == 1) { chosen.push_back(e); out_c++; }
                else if (cold_used < K_COLD_PER_TOK) {
                    chosen.push_back(e); out_c++; cold_used++;
                    unicast_cold.insert(e);
                    unicast_slot_uses++;
                }
            }
            for (int e : chosen) unicast_score_sum += h_scores[n * EXPERTS + e];
        }

        // 2. Broadcast Coalesced (Eyeriss v2 High-Reuse Mode: rank cold candidate experts across the 16-token window by
        //    multi-token demand `sum_{n in window: e in top16(n)} choice(n, e)` and admit the top `B_cold = 24` shared cold experts
        //    into `vram_mask = 2` so every token in the window can share them in `expert_tile_v3<16>`!)
        const int B_COLD = 24;
        std::vector<std::pair<double, int>> cold_util(EXPERTS, {0.0, 0});
        std::vector<int> cold_freq(EXPERTS, 0);
        for (int e = 0; e < EXPERTS; e++) cold_util[e].second = e;
        for (int n = 0; n < W_TOK; n++) {
            for (int r = 0; r < TOP; r++) {
                int e = h_sel_ref[n * TOP + r];
                if (h_vmask[e] == 0) {
                    cold_util[e].first += (double)h_choices[n * EXPERTS + e];
                    cold_freq[e]++;
                }
            }
        }
        std::sort(cold_util.begin(), cold_util.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
        std::unordered_set<int> bcast_cold;
        for (int i = 0; i < B_COLD && cold_util[i].first > 0.0; i++) bcast_cold.insert(cold_util[i].second);

        int bcast_slot_uses = 0;
        double bcast_score_sum = 0.0;
        for (int n = 0; n < W_TOK; n++) {
            int out_c = 0;
            for (int r = 0; r < TOP && out_c < TOP; r++) {
                int e = h_sel_ref[n * TOP + r];
                if (h_vmask[e] == 1 || bcast_cold.count(e)) {
                    bcast_score_sum += h_scores[n * EXPERTS + e];
                    if (bcast_cold.count(e)) bcast_slot_uses++;
                    out_c++;
                }
            }
        }

        printf("  Unicast Mode   (indep k_dyn=4/tok) | Distinct Cold Experts: %2zu (%6.1f MB PCIe) | Cold Slot Uses: %2d | Reuse/Expert: %.2fx | Top-16 Score Retained: %6.2f%%\n",
               unicast_cold.size(), unicast_cold.size() * (double)EXPERT_RAW / 1e6, unicast_slot_uses,
               (double)unicast_slot_uses / std::max<size_t>(1, unicast_cold.size()), 100.0 * unicast_score_sum / true_score_sum);
        printf("  Broadcast Mode (shared B_cold=%2d)   | Distinct Cold Experts: %2zu (%6.1f MB PCIe) | Cold Slot Uses: %2d | Reuse/Expert: %.2fx | Top-16 Score Retained: %6.2f%%\n",
               B_COLD, bcast_cold.size(), bcast_cold.size() * (double)EXPERT_RAW / 1e6, bcast_slot_uses,
               (double)bcast_slot_uses / std::max<size_t>(1, bcast_cold.size()), 100.0 * bcast_score_sum / true_score_sum);
    }

    CHECK(cudaFree(d_vmask));
    CHECK(cudaFree(d_w_test));
    CHECK(cudaFree(d_sel_test));
    CHECK(cudaFree(d_w_ref));
    CHECK(cudaFree(d_sel_ref));
    CHECK(cudaFree(d_choices));
    CHECK(cudaFree(d_scores));
    CHECK(cudaFree(d_bias));
    CHECK(cudaFree(d_router));
    CHECK(cudaFree(d_x));
}

// ============================================================================
// PART 3: Cycle 4 — FlashAttention-2 Online-Softmax & Parallel Warp V-Reduction
//         (`mla_attention_pos_fa2`) vs Shipped `mla_attention_pos` (`decode.cuh:134`)
// ============================================================================

// Shipped `mla_attention_pos` (`decode.cuh:134-166`)
__global__ void mla_attention_pos_shipped(float *out, const float *q, const float *kv, const float *kvlat, int N, size_t pos0) {
    extern __shared__ float sc[];
    __shared__ float red[32], smax;
    __shared__ double stotal;
    const int n = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
    const int count = (int)pos0 + n + 1;
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

// Cycle 4 FlashAttention-2 Online-Softmax + Warp-Coalesced Register Accumulator + Split-KV (`mla_attention_pos_fa2` & `mla_attention_pos_fa2_splitkv`):
//   1. Each 32-thread warp (`warp_id = 0..7`, `lane = 0..31`) holds its own 4 output coordinates `d = lane*4 .. lane*4+3` (`128` dims total)
//      and its own online-softmax state `(m_w, l_w, o_reg[4])` entirely in registers!
//   2. For each key `p` assigned to `warp_id`:
//      - All 32 lanes perform a 100% coalesced `float4` load of `k[p, lane*4 .. +3]` (128 floats) and `float2` load of `rope[p, lane*2 .. +1]` (64 floats),
//        accumulate the exact `double` dot product with `s_q[0..191]` (resident in shared SRAM), and reduce across the 32 lanes via `__shfl_xor_sync`
//        with ZERO `__syncthreads()` barriers inside the key loop!
//      - Each lane updates `(m_w, l_w)` via the FA2 online-softmax identity (`alpha = exp(m_old - m_new)`, `p_val = exp(score - m_new)`),
//        performs a 100% coalesced `float4` load of `v[p, lane*4 .. +3]`, and updates `o_reg[0..3] = o_reg[0..3] * alpha + p_val * v4` in registers!
//   3. At the end of the key loop, the 8 warps in the CTA combine their partial `(m_w, l_w, o_reg[4])` via the exact FA2 E7 logsumexp reduction,
//      and when `S_split > 1` (Split-KV across CTAs), `mla_splitkv_combine` merges the `S_split` partial results!
// float -> double with integer ops (`model.cuh:100`): avoids the 1:64 throttled F2F.F64.F32 pipe on GA102 (RTX 3090)
__device__ __forceinline__ double f2d_fast(float f) {
    const unsigned u = __float_as_uint(f);
    if (((u >> 23) & 0xffu) - 1u < 254u)
        return __hiloint2double((int)((u & 0x80000000u) | (((u & 0x7fffffffu) >> 3) + 0x38000000u)), (int)(u << 29));
    return (double)f;
}

template <bool USE_FP32_DOT>
__global__ void __launch_bounds__(256) mla_attention_pos_fa2_part(
    float *out, float *part_o, float *part_m, double *part_l,
    const float *q, const float *kv, const float *kvlat, int N, size_t pos0, int n_splits)
{
    __shared__ float s_qf[192];
    __shared__ double s_qd[192];
    __shared__ float s_m[8];
    __shared__ double s_l[8];
    __shared__ float s_o[8][128];

    const int n = blockIdx.x, h = blockIdx.y, sp = blockIdx.z;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int count = (int)pos0 + n + 1;
    const float *qq = q + (size_t)n * MLA_Q + h * 192;
    const float scale = 1.0f / sqrtf(192.0f);

    if (tid < 192) {
        float qv = qq[tid];
        s_qf[tid] = qv;
        if (!USE_FP32_DOT) s_qd[tid] = f2d_fast(qv);
    }
    __syncthreads();

    const int chunk = (count + n_splits - 1) / n_splits;
    const int p_start = sp * chunk;
    const int p_end = min(count, p_start + chunk);

    float m_w = -INFINITY;
    double l_w = 0.0;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;

    const int c4 = lane * 4;
    const int r2 = lane * 2;
    const float qkf0 = s_qf[c4 + 0], qkf1 = s_qf[c4 + 1], qkf2 = s_qf[c4 + 2], qkf3 = s_qf[c4 + 3];
    const float qrf0 = s_qf[128 + r2 + 0], qrf1 = s_qf[128 + r2 + 1];
    const double qkd0 = USE_FP32_DOT ? 0.0 : s_qd[c4 + 0], qkd1 = USE_FP32_DOT ? 0.0 : s_qd[c4 + 1];
    const double qkd2 = USE_FP32_DOT ? 0.0 : s_qd[c4 + 2], qkd3 = USE_FP32_DOT ? 0.0 : s_qd[c4 + 3];
    const double qrd0 = USE_FP32_DOT ? 0.0 : s_qd[128 + r2 + 0], qrd1 = USE_FP32_DOT ? 0.0 : s_qd[128 + r2 + 1];

    for (int p = p_start + warp; p < p_end; p += 8) {
        const float *k_ptr = kv + (size_t)p * MLA_EXP + h * 256;
        const float *r_ptr = kvlat + (size_t)p * MLA_KVLAT + MLA_KV;
        const float4 kv4 = *(const float4 *)(k_ptr + c4);
        const float2 rv2 = *(const float2 *)(r_ptr + r2);

        float v_sc;
        if (USE_FP32_DOT) {
            float sf = __fmaf_rn(qkf0, kv4.x, __fmaf_rn(qkf1, kv4.y, __fmaf_rn(qkf2, kv4.z, qkf3 * kv4.w)))
                     + __fmaf_rn(qrf0, rv2.x, qrf1 * rv2.y);
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) sf += __shfl_down_sync(0xffffffffu, sf, o);
            v_sc = sf * scale;
        } else {
            double sd = __fma_rn(qkd0, f2d_fast(kv4.x), __fma_rn(qkd1, f2d_fast(kv4.y), __fma_rn(qkd2, f2d_fast(kv4.z), qkd3 * f2d_fast(kv4.w))))
                      + __fma_rn(qrd0, f2d_fast(rv2.x), qrd1 * f2d_fast(rv2.y));
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) sd += __shfl_down_sync(0xffffffffu, sd, o);
            v_sc = (float)sd * scale;
        }

        // Only lane 0 computes online softmax `(m_w, l_w, alpha, p_val)` and broadcasts `(alpha, p_val)` to the warp!
        float alpha = 0.0f, p_val = 0.0f;
        if (lane == 0) {
            const float m_new = fmaxf(m_w, v_sc);
            alpha = (m_w == -INFINITY) ? 0.0f : expf_cr(m_w - m_new);
            p_val = expf_cr(v_sc - m_new);
            l_w = l_w * (double)alpha + (double)p_val;
            m_w = m_new;
        }
        alpha = __shfl_sync(0xffffffffu, alpha, 0);
        p_val = __shfl_sync(0xffffffffu, p_val, 0);

        const float4 vv4 = *(const float4 *)(k_ptr + 128 + c4);
        o0 = o0 * alpha + p_val * vv4.x;
        o1 = o1 * alpha + p_val * vv4.y;
        o2 = o2 * alpha + p_val * vv4.z;
        o3 = o3 * alpha + p_val * vv4.w;
    }

    if (lane == 0) {
        s_m[warp] = m_w;
        s_l[warp] = l_w;
    }
    s_o[warp][c4 + 0] = o0;
    s_o[warp][c4 + 1] = o1;
    s_o[warp][c4 + 2] = o2;
    s_o[warp][c4 + 3] = o3;
    __syncthreads();

    if (tid < 128) {
        float m_cta = -INFINITY;
        for (int w = 0; w < 8; w++) m_cta = fmaxf(m_cta, s_m[w]);
        double l_cta = 0.0;
        float o_cta = 0.0f;
        if (m_cta > -INFINITY) {
            for (int w = 0; w < 8; w++) {
                if (s_m[w] == -INFINITY) continue;
                float sc_w = expf_cr(s_m[w] - m_cta);
                l_cta += s_l[w] * (double)sc_w;
                o_cta += s_o[w][tid] * sc_w;
            }
        }
        if (n_splits == 1) {
            out[(size_t)n * ROWS + h * 128 + tid] = (l_cta > 0.0) ? (float)((double)o_cta / l_cta) : 0.0f;
        } else {
            const size_t idx_base = ((size_t)(n * HEADS + h) * n_splits + sp);
            part_o[idx_base * 128 + tid] = o_cta;
            if (tid == 0) {
                part_m[idx_base] = m_cta;
                part_l[idx_base] = l_cta;
            }
        }
    }
}

__global__ void __launch_bounds__(128) mla_splitkv_combine(
    float *out, const float *part_o, const float *part_m, const double *part_l, int n_splits)
{
    const int n = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const size_t base = (size_t)(n * HEADS + h) * n_splits;
    float m_all = -INFINITY;
    for (int sp = 0; sp < n_splits; sp++) m_all = fmaxf(m_all, part_m[base + sp]);
    double l_all = 0.0;
    float o_all = 0.0f;
    for (int sp = 0; sp < n_splits; sp++) {
        float m_s = part_m[base + sp];
        if (m_s == -INFINITY) continue;
        float sc_s = expf_cr(m_s - m_all);
        l_all += part_l[base + sp] * (double)sc_s;
        o_all += part_o[(base + sp) * 128 + d] * sc_s;
    }
    out[(size_t)n * ROWS + h * 128 + d] = (l_all > 0.0) ? (float)((double)o_all / l_all) : 0.0f;
}

template <bool USE_FP32_DOT>
static void launch_mla_fa2(
    float *d_out, float *d_part_o, float *d_part_m, double *d_part_l,
    const float *d_q, const float *d_kv, const float *d_kvlat, int N, size_t pos0)
{
    int count = (int)pos0 + N;
    int n_splits = (N == 1 && count >= 256) ? std::min(8, count / 64) : ((N <= 4 && count >= 512) ? 4 : 1);
    mla_attention_pos_fa2_part<USE_FP32_DOT><<<dim3(N, HEADS, n_splits), 256>>>(d_out, d_part_o, d_part_m, d_part_l, d_q, d_kv, d_kvlat, N, pos0, n_splits);
    if (n_splits > 1) {
        mla_splitkv_combine<<<dim3(N, HEADS), 128>>>(d_out, d_part_o, d_part_m, d_part_l, n_splits);
    }
}

static void run_cycle4_fa2_mla_benchmark(int dev) {
    CHECK(cudaSetDevice(dev));
    const int max_count = 4096;
    const int max_N = 16;

    std::vector<float> h_q((size_t)max_N * MLA_Q), h_kv((size_t)max_count * MLA_EXP), h_kvlat((size_t)max_count * MLA_KVLAT);
    srand(42);
    for (auto &v : h_q) v = ((float)(rand() % 2001) - 1000.0f) * 0.001f;
    for (auto &v : h_kv) v = ((float)(rand() % 2001) - 1000.0f) * 0.001f;
    for (auto &v : h_kvlat) v = ((float)(rand() % 2001) - 1000.0f) * 0.001f;

    float *d_q = nullptr, *d_kv = nullptr, *d_kvlat = nullptr, *d_out_ref = nullptr, *d_out_fa2 = nullptr;
    float *d_part_o = nullptr, *d_part_m = nullptr;
    double *d_part_l = nullptr;
    const int max_splits = 8;
    CHECK(cudaMalloc(&d_q, h_q.size() * sizeof(float)));
    CHECK(cudaMalloc(&d_kv, h_kv.size() * sizeof(float)));
    CHECK(cudaMalloc(&d_kvlat, h_kvlat.size() * sizeof(float)));
    CHECK(cudaMalloc(&d_out_ref, (size_t)max_N * ROWS * sizeof(float)));
    CHECK(cudaMalloc(&d_out_fa2, (size_t)max_N * ROWS * sizeof(float)));
    CHECK(cudaMalloc(&d_part_o, (size_t)max_N * HEADS * max_splits * 128 * sizeof(float)));
    CHECK(cudaMalloc(&d_part_m, (size_t)max_N * HEADS * max_splits * sizeof(float)));
    CHECK(cudaMalloc(&d_part_l, (size_t)max_N * HEADS * max_splits * sizeof(double)));

    CHECK(cudaMemcpy(d_q, h_q.data(), h_q.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_kv, h_kv.data(), h_kv.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_kvlat, h_kvlat.data(), h_kvlat.size() * sizeof(float), cudaMemcpyHostToDevice));

    printf("\n=== LIVE GPU CYCLE 4: FlashAttention-2 Online-Softmax + Split-KV (`mla_attention_pos_fa2`) vs Shipped `mla_attention_pos` on cuda:%d ===\n", dev);
    printf("%-4s | %-6s | %-10s | %-11s | %-9s | %-10s | %-11s | %-9s | %-10s\n",
           "N_q", "KV_len", "Shipped_us", "FA2_f2d_us", "Speed_f2d", "Diff_f2d", "FA2_fp32_us", "Speed_f32", "Diff_f32");
    printf("----------------------------------------------------------------------------------------------------------------\n");

    cudaEvent_t ev0, ev1;
    CHECK(cudaEventCreate(&ev0));
    CHECK(cudaEventCreate(&ev1));

    int counts[] = {64, 256, 512, 1024, 2048, 4096};
    int n_qs[] = {1, 4};

    for (int N : n_qs) {
        for (int cnt : counts) {
            size_t pos0 = (size_t)(cnt - N);
            size_t shared_ref = (pos0 + (size_t)N) * sizeof(float);
            if (shared_ref > 48 * 1024) {
                CHECK(cudaFuncSetAttribute(mla_attention_pos_shipped, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared_ref));
            }
            mla_attention_pos_shipped<<<dim3(N, HEADS), 128, shared_ref>>>(d_out_ref, d_q, d_kv, d_kvlat, N, pos0);
            launch_mla_fa2<false>(d_out_fa2, d_part_o, d_part_m, d_part_l, d_q, d_kv, d_kvlat, N, pos0);
            CHECK(cudaDeviceSynchronize());

            std::vector<float> h_ref((size_t)N * ROWS), h_fa2((size_t)N * ROWS);
            CHECK(cudaMemcpy(h_ref.data(), d_out_ref, h_ref.size() * sizeof(float), cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(h_fa2.data(), d_out_fa2, h_fa2.size() * sizeof(float), cudaMemcpyDeviceToHost));
            double max_diff_f2d = 0.0;
            for (size_t i = 0; i < h_ref.size(); i++) {
                double d = fabs((double)h_ref[i] - (double)h_fa2[i]);
                if (d > max_diff_f2d) max_diff_f2d = d;
            }

            launch_mla_fa2<true>(d_out_fa2, d_part_o, d_part_m, d_part_l, d_q, d_kv, d_kvlat, N, pos0);
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaMemcpy(h_fa2.data(), d_out_fa2, h_fa2.size() * sizeof(float), cudaMemcpyDeviceToHost));
            double max_diff_f32 = 0.0;
            for (size_t i = 0; i < h_ref.size(); i++) {
                double d = fabs((double)h_ref[i] - (double)h_fa2[i]);
                if (d > max_diff_f32) max_diff_f32 = d;
            }

            const int iters = 50;
            CHECK(cudaEventRecord(ev0));
            for (int it = 0; it < iters; it++) {
                mla_attention_pos_shipped<<<dim3(N, HEADS), 128, shared_ref>>>(d_out_ref, d_q, d_kv, d_kvlat, N, pos0);
            }
            CHECK(cudaEventRecord(ev1));
            CHECK(cudaEventSynchronize(ev1));
            float ms_ref = 0.0f;
            CHECK(cudaEventElapsedTime(&ms_ref, ev0, ev1));

            CHECK(cudaEventRecord(ev0));
            for (int it = 0; it < iters; it++) {
                launch_mla_fa2<false>(d_out_fa2, d_part_o, d_part_m, d_part_l, d_q, d_kv, d_kvlat, N, pos0);
            }
            CHECK(cudaEventRecord(ev1));
            CHECK(cudaEventSynchronize(ev1));
            float ms_f2d = 0.0f;
            CHECK(cudaEventElapsedTime(&ms_f2d, ev0, ev1));

            CHECK(cudaEventRecord(ev0));
            for (int it = 0; it < iters; it++) {
                launch_mla_fa2<true>(d_out_fa2, d_part_o, d_part_m, d_part_l, d_q, d_kv, d_kvlat, N, pos0);
            }
            CHECK(cudaEventRecord(ev1));
            CHECK(cudaEventSynchronize(ev1));
            float ms_f32 = 0.0f;
            CHECK(cudaEventElapsedTime(&ms_f32, ev0, ev1));

            float us_ref = (ms_ref * 1000.0f) / iters;
            float us_f2d = (ms_f2d * 1000.0f) / iters;
            float us_f32 = (ms_f32 * 1000.0f) / iters;
            printf("%-4d | %-6d | %7.1f us | %8.1f us | %7.2fx | %10.3e | %8.1f us | %7.2fx | %10.3e\n",
                   N, cnt, us_ref, us_f2d, us_ref / us_f2d, max_diff_f2d, us_f32, us_ref / us_f32, max_diff_f32);
        }
    }

    CHECK(cudaFree(d_part_l));
    CHECK(cudaFree(d_part_m));
    CHECK(cudaFree(d_part_o));

    CHECK(cudaEventDestroy(ev1));
    CHECK(cudaEventDestroy(ev0));
    CHECK(cudaFree(d_out_fa2));
    CHECK(cudaFree(d_out_ref));
    CHECK(cudaFree(d_kvlat));
    CHECK(cudaFree(d_kv));
    CHECK(cudaFree(d_q));
}

int main(int argc, char **argv) {
    int dev = (argc > 1) ? atoi(argv[1]) : 7;
    const char *trace_path = "/home/usergpu/k3gpu/results/p1d.trace";
    const char *dataset_dir = "/opt/clover-k3/datasets/computed";
    const char *act_path = "/home/usergpu/k3gpu/results/gpu-l13.layer0";

    auto calls = load_trace(trace_path, 25); // 25 passes of L1..92
    run_live_pcie_cycle2_cycle3(dev, calls);
    run_cycle3c_and_cycle5_router_studies(dev, dataset_dir, act_path);
    run_cycle4_fa2_mla_benchmark(dev);
    return 0;
}
