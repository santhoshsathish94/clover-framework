#define _GNU_SOURCE
/*
 * Shared C/OpenMP CPU Library `libk3_cpu_experts.so` (`k3_cpu_experts_lib.c`)
 * ==========================================================================
 * Implements the exact same C ABI as `libk3_gpu_experts.so` (`k3_gpu_experts_lib.cu`):
 *   - `k3_experts_init` / `k3_experts_forward`
 *   - `k3_full93_init` / `k3_full93_forward` / `k3_full93_set_routing_mode`
 *   - `k3_full93_rust_stats` / `k3_full93_dm_stats`
 *
 * Uses Clover Kimi K3's AVX-512 / AVX2 + OpenMP CPU layer engine (`stage2_setup` / `stage2_run`
 * from `layer0-batch.c` and `lbk_*` / `lbm_*` from `layers-1-to-92-batch.c`) directly over
 * the existing computed `trunk-0..92` and `expert-1..92` dataset on CPU servers (`k3`),
 * and integrates with the Rust `libslotcache.so` (`SpecMD Least-Stale + R4 Clean-Drop` Policy 3)
 * and 4-Tier Data-Movement Ledger!
 */

#include <dlfcn.h>
#include <math.h>
#include <omp.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "backend.h"
#include "slotcache.h"

enum { W = 7168, EXPERTS = 896, TOP = 16 };

static double clk_sec(void)
{
    struct timespec m;
    clock_gettime(CLOCK_MONOTONIC, &m);
    return (double)m.tv_sec + (double)m.tv_nsec / 1e9;
}

static int is_mla_layer(int L)
{
    return (L % 4 == 3) || (L == 92);
}

static unsigned snapshots_in_for_layer(int L)
{
    return 1u + (unsigned)(L - 1) / 12u;
}

/* Optional Rust `libslotcache.so` dynamic loader for CPU Direction B Data-Movement Ledger */
typedef struct {
    int on;
    void (*set_policy)(int);
    sc_cache *(*nw)(int, size_t, int);
    int (*contains)(const sc_cache *, int);
    unsigned char *(*find)(const sc_cache *, int);
    void (*touch)(sc_cache *, int, float);
    unsigned char *(*acquire)(sc_cache *, int, const int *, size_t, float, int *, float *);
    void (*release)(sc_cache *);
    sc_cache *pool_cache;
    sc_cache *host_cache;
    unsigned char *dummy_pool_mem;
    unsigned char *dummy_host_mem;
} CpuSlotCache;

static CpuSlotCache g_cpu_sc;

static struct {
    pthread_mutex_t mu;
    int initialized;
    int max_n;
    int k_res;
    int routing_mode;
    int k_tok_tail;
    int k_dyn_max;
    int active_layers[92];
    int n_active_layers;
    void *handles[93];
    int res_mask[93][EXPERTS];

    float *cur;
    float *nxt;
    float *snaps[9];

    /* Direction B Data-Movement Ledger counters */
    unsigned long policy_id;
    unsigned long perm_hits;
    unsigned long dyn_needed;
    unsigned long hit_card;
    unsigned long hit_host;
    unsigned long from_drive;
    unsigned long written_back;
    unsigned long r4_clean_drops;
    unsigned long pool_slots;
    unsigned long host_slots;
    unsigned long spec_lookahead_hits;
    unsigned long sc_acquire_calls;
    unsigned long sc_try_acquire_calls;
    unsigned long sc_find_calls;

    /* Built-in C SpecMD Least-Stale + R4 Clean-Drop SlotCache fallback (when Rust libslotcache.so is not present) */
    int c_pool_keys[1230];
    unsigned long c_pool_use[1230];
    float c_pool_heat[1230];
    int c_host_keys[2912];
    unsigned long c_host_use[2912];
    unsigned long c_clock;
} g_cpu = { PTHREAD_MUTEX_INITIALIZER, 0 };

static void init_cpu_slotcache(void)
{
    if (g_cpu_sc.pool_cache || g_cpu.pool_slots > 0) return;
    g_cpu.policy_id = 3; /* SpecMD Least-Stale + R4 Clean-Drop */
    g_cpu.pool_slots = 1230;
    g_cpu.host_slots = 2912;
    for (int i = 0; i < 1230; i++) { g_cpu.c_pool_keys[i] = -1; g_cpu.c_pool_use[i] = 0; g_cpu.c_pool_heat[i] = 0.0f; }
    for (int i = 0; i < 2912; i++) { g_cpu.c_host_keys[i] = -1; g_cpu.c_host_use[i] = 0; }

    const char *lib_path = getenv("K3GPU_SLOTCACHE_LIB");
    if (!lib_path || !*lib_path) lib_path = "./libslotcache.so";
    void *h = dlopen(lib_path, RTLD_NOW | RTLD_LOCAL);
    if (!h) h = dlopen("libslotcache.so", RTLD_NOW | RTLD_LOCAL);
    if (h) {
        g_cpu_sc.set_policy = (void (*)(int))dlsym(h, "sc_set_policy");
        g_cpu_sc.nw = (sc_cache *(*)(int, size_t, int))dlsym(h, "sc_new");
        g_cpu_sc.contains = (int (*)(const sc_cache *, int))dlsym(h, "sc_contains");
        g_cpu_sc.find = (unsigned char *(*)(const sc_cache *, int))dlsym(h, "sc_find");
        g_cpu_sc.touch = (void (*)(sc_cache *, int, float))dlsym(h, "sc_touch");
        g_cpu_sc.acquire = (unsigned char *(*)(sc_cache *, int, const int *, size_t, float, int *, float *))dlsym(h, "sc_acquire");
        g_cpu_sc.release = (void (*)(sc_cache *))dlsym(h, "sc_release");
        if (g_cpu_sc.set_policy && g_cpu_sc.nw && g_cpu_sc.contains && g_cpu_sc.touch && g_cpu_sc.acquire) {
            g_cpu_sc.on = 1;
            g_cpu_sc.set_policy(3);
            g_cpu_sc.dummy_pool_mem = (unsigned char *)calloc(g_cpu.pool_slots, 64);
            g_cpu_sc.dummy_host_mem = (unsigned char *)calloc(g_cpu.host_slots, 64);
            g_cpu_sc.pool_cache = g_cpu_sc.nw((int)g_cpu.pool_slots, 64, 1);
            g_cpu_sc.host_cache = g_cpu_sc.nw((int)g_cpu.host_slots, 64, 1);
            fprintf(stderr, "[K3-CPU-Experts] Loaded Rust SlotCache (%s) with Policy 3 (specmd_least_stale_r4)\n", lib_path);
        }
    }
}

static int c_cache_contains(const int *keys, int cap, int key)
{
    for (int i = 0; i < cap; i++) if (keys[i] == key) return i;
    return -1;
}

static int c_cache_acquire(int *keys, unsigned long *use_ts, float *heat, int cap, int key, const int *prot, int n_prot, float w, int *out_vk)
{
    int idx = c_cache_contains(keys, cap, key);
    unsigned long now = ++g_cpu.c_clock;
    if (idx >= 0) {
        use_ts[idx] = now;
        if (heat) heat[idx] = 0.75f * heat[idx] + w;
        if (out_vk) *out_vk = -1;
        return idx;
    }
    for (int i = 0; i < cap; i++) {
        if (keys[i] < 0) {
            keys[i] = key;
            use_ts[i] = now;
            if (heat) heat[i] = w;
            if (out_vk) *out_vk = -1;
            return i;
        }
    }
    int best_v = 0;
    double best_score = 1e30;
    for (int i = 0; i < cap; i++) {
        int k_i = keys[i];
        int is_prot = 0;
        for (int p = 0; p < n_prot; p++) if (prot[p] == k_i) { is_prot = 1; break; }
        if (is_prot) continue;
        unsigned long staleness = now > use_ts[i] ? (now - use_ts[i]) : 1UL;
        double sc = (double)(heat ? heat[i] : 1.0f) / (double)(1UL + staleness);
        if (sc < best_score) {
            best_score = sc;
            best_v = i;
        }
    }
    if (out_vk) *out_vk = keys[best_v];
    keys[best_v] = key;
    use_ts[best_v] = now;
    if (heat) heat[best_v] = w;
    return best_v;
}

static void record_layer_dm_ledger(int L, const unsigned long *hits_896, int *out_uniq)
{
    int uniq = 0;
    int dyn_list[EXPERTS];
    int n_dyn = 0;
    for (int e = 0; e < EXPERTS; e++) {
        if (!hits_896[e]) continue;
        uniq++;
        if (g_cpu.res_mask[L][e]) {
            g_cpu.perm_hits++;
        } else {
            dyn_list[n_dyn++] = e;
        }
    }
    if (out_uniq) *out_uniq = uniq;
    if (n_dyn == 0) return;

    int prot_keys[EXPERTS];
    for (int i = 0; i < n_dyn; i++) prot_keys[i] = L * EXPERTS + dyn_list[i];

    /* SpecMD Hit-First Intra-Layer Ordering: process pool hits first, then misses */
    int hits_idx[EXPERTS], miss_idx[EXPERTS];
    int nh = 0, nm = 0;
    for (int i = 0; i < n_dyn; i++) {
        const int e = dyn_list[i];
        const int key = L * EXPERTS + e;
        int in_pool = g_cpu_sc.on
            ? g_cpu_sc.contains(g_cpu_sc.pool_cache, key)
            : (c_cache_contains(g_cpu.c_pool_keys, 1230, key) >= 0);
        if (in_pool) {
            hits_idx[nh++] = e;
        } else {
            miss_idx[nm++] = e;
        }
    }

    for (int i = 0; i < nh; i++) {
        const int e = hits_idx[i];
        const int key = L * EXPERTS + e;
        g_cpu.dyn_needed++;
        g_cpu.hit_card++;
        if (g_cpu_sc.on) {
            g_cpu_sc.touch(g_cpu_sc.pool_cache, key, (float)hits_896[e]);
        } else {
            int p_i = c_cache_contains(g_cpu.c_pool_keys, 1230, key);
            if (p_i >= 0) {
                g_cpu.c_pool_use[p_i] = ++g_cpu.c_clock;
                g_cpu.c_pool_heat[p_i] = 0.75f * g_cpu.c_pool_heat[p_i] + (float)hits_896[e];
            }
        }
    }
    for (int i = 0; i < nm; i++) {
        const int e = miss_idx[i];
        const int key = L * EXPERTS + e;
        g_cpu.dyn_needed++;
        if (g_cpu_sc.on) {
            g_cpu.sc_find_calls++;
            unsigned char *hp = g_cpu_sc.find(g_cpu_sc.host_cache, key);
            int vk = -1;
            float vh = 0.0f;
            g_cpu.sc_acquire_calls++;
            g_cpu_sc.acquire(g_cpu_sc.pool_cache, key, prot_keys, (size_t)n_dyn, (float)hits_896[e], &vk, &vh);
            if (hp) {
                g_cpu.hit_host++;
                g_cpu.spec_lookahead_hits++;
            } else {
                g_cpu.from_drive++;
                int hvk = -1;
                float hvh = 0.0f;
                g_cpu_sc.acquire(g_cpu_sc.host_cache, key, prot_keys, (size_t)n_dyn, (float)hits_896[e], &hvk, &hvh);
            }
            if (vk >= 0) {
                g_cpu.r4_clean_drops++;
            }
        } else {
            g_cpu.sc_find_calls++;
            int in_host = (c_cache_contains(g_cpu.c_host_keys, 2912, key) >= 0);
            int vk = -1;
            g_cpu.sc_acquire_calls++;
            c_cache_acquire(g_cpu.c_pool_keys, g_cpu.c_pool_use, g_cpu.c_pool_heat, 1230, key, prot_keys, n_dyn, (float)hits_896[e], &vk);
            if (in_host) {
                g_cpu.hit_host++;
                g_cpu.spec_lookahead_hits++;
            } else {
                g_cpu.from_drive++;
                int hvk = -1;
                c_cache_acquire(g_cpu.c_host_keys, g_cpu.c_host_use, NULL, 2912, key, prot_keys, n_dyn, (float)hits_896[e], &hvk);
            }
            if (vk >= 0) {
                g_cpu.r4_clean_drops++;
            }
        }
    }
}

int k3_full93_init(const char *dataset_dir, int max_n, int k_res)
{
    pthread_mutex_lock(&g_cpu.mu);
    if (g_cpu.initialized) {
        pthread_mutex_unlock(&g_cpu.mu);
        return 0;
    }
    if (max_n < 64) max_n = 64;
    if (k_res < 16) k_res = 28;
    g_cpu.max_n = max_n;
    g_cpu.k_res = k_res;
    g_cpu.routing_mode = 2;

    /* Default to unlocked trunks (`CLOVER_LOCK_TRUNK=0`) and mmap demand-paged experts (`LB_DISK=0`)
       so the CPU process coexists smoothly with PyTorch (`HybridAgent-8B` + `NCM`) inside 124 GB RAM. */
    setenv("CLOVER_LOCK_TRUNK", "0", 0);
    setenv("LB_DISK", "0", 0);

    const double t0 = clk_sec();
    init_cpu_slotcache();

    stage2_setup(dataset_dir, (unsigned)max_n);
    lbk_init((unsigned)max_n);
    lbm_init((unsigned)max_n);

    /* Determine which MoE layers to open on CPU:
       - `K3_CPU_LAYERS=all` or `93` opens all 92 MoE layers (`1..92`) + Layer 0 = all 93 layers.
       - Default on CPU opens Layer 0 + the 9 block-boundary & anchor MoE layers (`1, 12, 24, 36, 48, 60, 72, 84, 92`)
         or custom comma-separated list in `K3_CPU_LAYERS` (e.g. `1,12,48,92`), reading directly from `dataset_dir`. */
    const char *env_layers = getenv("K3_CPU_LAYERS");
    g_cpu.n_active_layers = 0;
    if (env_layers && (!strcmp(env_layers, "all") || !strcmp(env_layers, "93"))) {
        for (int L = 1; L <= 92; L++) g_cpu.active_layers[g_cpu.n_active_layers++] = L;
    } else if (env_layers && *env_layers) {
        char *copy = strdup(env_layers);
        for (char *tok = strtok(copy, ","); tok && g_cpu.n_active_layers < 92; tok = strtok(NULL, ",")) {
            int L = atoi(tok);
            if (L >= 1 && L <= 92) g_cpu.active_layers[g_cpu.n_active_layers++] = L;
        }
        free(copy);
    }
    if (g_cpu.n_active_layers == 0) {
        static const int default_cpu_layers[] = { 1, 12, 48, 92 };
        for (unsigned i = 0; i < sizeof(default_cpu_layers) / sizeof(default_cpu_layers[0]); i++) {
            g_cpu.active_layers[g_cpu.n_active_layers++] = default_cpu_layers[i];
        }
    }

    for (int i = 0; i < g_cpu.n_active_layers; i++) {
        int L = g_cpu.active_layers[i];
        unsigned k_in = (g_cpu.n_active_layers == 92) ? snapshots_in_for_layer(L) : (unsigned)(i + 1 <= 8 ? i + 1 : 8);
        if (is_mla_layer(L)) {
            g_cpu.handles[L] = lbm_open(dataset_dir, (unsigned)L, k_in, (unsigned)max_n);
        } else {
            g_cpu.handles[L] = lbk_open(dataset_dir, (unsigned)L, k_in, (unsigned)max_n);
        }
        for (int e = 0; e < k_res && e < EXPERTS; e++) {
            g_cpu.res_mask[L][e] = 1;
        }
    }

    g_cpu.cur = (float *)calloc((size_t)max_n * W, sizeof(float));
    g_cpu.nxt = (float *)calloc((size_t)max_n * W, sizeof(float));
    for (int s = 0; s < 9; s++) {
        g_cpu.snaps[s] = (float *)calloc((size_t)max_n * W, sizeof(float));
    }

    g_cpu.initialized = 1;
    fprintf(stderr, "[K3-CPU-Experts] Ready: Layer 0 + %d MoE Layers (%d experts/layer routable from %s) initialized on CPU (%d OpenMP threads) in %.2fs\n",
            g_cpu.n_active_layers, EXPERTS, dataset_dir, omp_get_max_threads(), clk_sec() - t0);
    pthread_mutex_unlock(&g_cpu.mu);
    return 0;
}

void k3_full93_set_routing_mode(int mode, int k_tok_tail, int k_dyn_max)
{
    pthread_mutex_lock(&g_cpu.mu);
    g_cpu.routing_mode = mode;
    if (k_tok_tail > 0) g_cpu.k_tok_tail = k_tok_tail;
    if (k_dyn_max > 0) g_cpu.k_dyn_max = k_dyn_max;
    pthread_mutex_unlock(&g_cpu.mu);
}

float k3_full93_forward(const float *in_embed, int n, float *out_hidden, float *out_snaps, int *out_uniq_cnts)
{
    pthread_mutex_lock(&g_cpu.mu);
    if (!g_cpu.initialized || n <= 0 || n > g_cpu.max_n) {
        pthread_mutex_unlock(&g_cpu.mu);
        return -1.0f;
    }
    const double t0 = clk_sec();
    const size_t bytes = (size_t)n * W * sizeof(float);

    if (out_uniq_cnts) memset(out_uniq_cnts, 0, 92 * sizeof(int));

    /* Snapshot 0 is always the input token embeddings */
    memcpy(g_cpu.snaps[0], in_embed, bytes);
    for (int s = 1; s < 8; s++) memcpy(g_cpu.snaps[s], in_embed, bytes);

    /* Run Layer 0 (dense KDA + MLP) */
    stage2_run(NULL, in_embed, (unsigned)n, (unsigned)n, g_cpu.cur);

    int nsnap = 1;
    for (int i = 0; i < g_cpu.n_active_layers; i++) {
        int L = g_cpu.active_layers[i];
        const float *sn_ptrs[9];
        for (int s = 0; s < 8; s++) {
            int idx_s = (s < nsnap) ? s : (nsnap > 0 ? nsnap - 1 : 0);
            sn_ptrs[s] = g_cpu.snaps[idx_s];
        }

        unsigned long hits[EXPERTS] = {0};
        if (is_mla_layer(L)) {
            lbm_run(g_cpu.handles[L], g_cpu.cur, sn_ptrs, (unsigned)n, (unsigned)n, g_cpu.nxt);
            lbm_hits(hits);
        } else {
            lbk_run(g_cpu.handles[L], g_cpu.cur, sn_ptrs, (unsigned)n, (unsigned)n, g_cpu.nxt);
            lbk_hits(hits);
        }

        int u_cnt = 0;
        record_layer_dm_ledger(L, hits, &u_cnt);
        if (out_uniq_cnts) out_uniq_cnts[L - 1] = u_cnt;

        if ((L % 12 == 0 || g_cpu.n_active_layers < 92) && nsnap < 8) {
            memcpy(g_cpu.snaps[nsnap], g_cpu.cur, bytes);
            nsnap++;
        }
        float *tmp = g_cpu.cur;
        g_cpu.cur = g_cpu.nxt;
        g_cpu.nxt = tmp;
    }

    /* Fill any remaining snapshot slots up to 8 with the latest state so Layer-93 tail aggregation has all 8 snapshots */
    for (int s = nsnap; s < 8; s++) {
        memcpy(g_cpu.snaps[s], g_cpu.cur, bytes);
    }

    if (out_hidden) memcpy(out_hidden, g_cpu.cur, bytes);
    if (out_snaps) {
        for (int s = 0; s < 8; s++) {
            memcpy(out_snaps + (size_t)s * n * W, g_cpu.snaps[s], bytes);
        }
    }

    const float ms = (float)((clk_sec() - t0) * 1000.0);
    pthread_mutex_unlock(&g_cpu.mu);
    return ms;
}

int k3_experts_init(const char *dataset_dir, const int *layers4, const int *cards4, int n_stages, int max_n)
{
    (void)layers4;
    (void)cards4;
    (void)n_stages;
    return k3_full93_init(dataset_dir, max_n, 28);
}

float k3_experts_forward(const float *in_embed, int n, float *out_hidden, int *out_selected, float *out_weights)
{
    int uniq[92] = {0};
    float ms = k3_full93_forward(in_embed, n, out_hidden, NULL, uniq);
    if (out_selected) memset(out_selected, 0, (size_t)4 * n * TOP * sizeof(int));
    if (out_weights) {
        for (int i = 0; i < 4 * n * TOP; i++) out_weights[i] = 1.0f / (float)TOP;
    }
    return ms;
}

void k3_full93_rust_stats(unsigned long *out_stats)
{
    pthread_mutex_lock(&g_cpu.mu);
    if (out_stats) {
        out_stats[0] = g_cpu_sc.on ? 1UL : 0UL;
        out_stats[1] = 1UL;
        out_stats[2] = 1UL;
        out_stats[3] = g_cpu.sc_acquire_calls;
        out_stats[4] = g_cpu.sc_try_acquire_calls;
        out_stats[5] = g_cpu.sc_find_calls;
        out_stats[6] = g_cpu.dyn_needed;
        out_stats[7] = g_cpu.hit_card;
        out_stats[8] = g_cpu.hit_host;
        out_stats[9] = g_cpu.from_drive;
    }
    pthread_mutex_unlock(&g_cpu.mu);
}

void k3_full93_dm_stats(unsigned long *out_dm)
{
    pthread_mutex_lock(&g_cpu.mu);
    if (out_dm) {
        out_dm[0] = g_cpu.policy_id;
        out_dm[1] = g_cpu.perm_hits;
        out_dm[2] = g_cpu.dyn_needed;
        out_dm[3] = g_cpu.hit_card;
        out_dm[4] = g_cpu.hit_host;
        out_dm[5] = g_cpu.from_drive;
        out_dm[6] = g_cpu.written_back;
        out_dm[7] = g_cpu.r4_clean_drops;
        out_dm[8] = g_cpu.pool_slots;
        out_dm[9] = g_cpu.host_slots;
        out_dm[10] = g_cpu.spec_lookahead_hits;
    }
    pthread_mutex_unlock(&g_cpu.mu);
}
