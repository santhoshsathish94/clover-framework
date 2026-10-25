// GPU decoding: the layer runner of model.cuh made to continue from the state earlier tokens left, with the experts supplied one token window at a time.
//
// What is new compared with model.cuh (which runs a whole window from position 0 with every expert of the layer on the card):
//  - state that lives between calls: the KDA recurrent state and the last three rows of the convolution (per layer), the MLA key/value rows (per layer,
//    by position), optional copies of the KDA state after each token of a pass (so a rejected draft can be undone);
//  - the experts of a layer are NOT all on the card. After the router has chosen them, only the chosen experts are made resident in a pool of expert slots
//    in the card's memory (least recently used out first); an expert that is not in the card comes from a host cache in pinned RAM (least recently used
//    out first), and if it is not there either, from the NVMe drive (O_DIRECT, several reads at once). expert_proj_v3p takes the expert's address from a
//    table instead of a position in one 896-expert block.
// The arithmetic of every kernel is that of model.cuh / moe.cuh / kernels.cuh (the copies below differ only in where the state and the experts come from).
#pragma once
#include "model.cuh"
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#if __has_include("slotcache.h")
#include "slotcache.h"   // contract of the Rust slot cache (src/rust/slotcache/include); a build without it has only the C++ cache
#define K3GPU_SLOTCACHE_BUILT 1
#endif
#if __has_include("poolinit.h")
#include "poolinit.h"   // contract of the Rust pool start-up (src/rust/poolinit/include): HostCache::init and CardPool::init
#define K3GPU_POOLINIT_BUILT 1
#endif
#if __has_include("supply.h")
#include "supply.h"   // contract of the Rust card supply (src/rust/supply/include): pool_ensure
#define K3GPU_SUPPLY_BUILT 1
#endif
#if __has_include("planner.h")
#include "planner.h"   // contract of the Rust supply planner (src/rust/planner/include): read-ahead and card-queue planning, pend_take / arr_wait
#define K3GPU_PLANNER_BUILT 1
#endif

// ---------------------------------------------------------------- state that lives between calls
struct KdaState {
    float *S = nullptr;            // HEADS x HEAD x HEAD, the recurrent state
    float *hist = nullptr;         // 3 components x 3 rows x ROWS: the last three raw rows of q, k, v (oldest first)
    float *S_keep = nullptr;       // keep x HEADS x HEAD x HEAD: the state after token j of the last pass
    float *hist_keep = nullptr;    // keep x (3 x 3 x ROWS)
    int keep = 0;
};
struct MlaCache { float *kv = nullptr, *lat = nullptr; size_t cap = 0; };   // per position: MLA_EXP floats, MLA_KVLAT floats

static const size_t KDA_STATE_FLOATS = (size_t)HEADS * HEAD * HEAD, KDA_HIST_FLOATS = (size_t)9 * ROWS;

static void kda_state_alloc(KdaState &s)
{
    CHECK(cudaMalloc(&s.S, KDA_STATE_FLOATS * 4)); CHECK(cudaMalloc(&s.hist, KDA_HIST_FLOATS * 4));
    CHECK(cudaMemset(s.S, 0, KDA_STATE_FLOATS * 4)); CHECK(cudaMemset(s.hist, 0, KDA_HIST_FLOATS * 4));
}
static void kda_state_reset(KdaState &s) { CHECK(cudaMemset(s.S, 0, KDA_STATE_FLOATS * 4)); CHECK(cudaMemset(s.hist, 0, KDA_HIST_FLOATS * 4)); }
static void kda_state_keep_alloc(KdaState &s, int count)
{
    if (s.keep >= count) return;
    if (s.S_keep) cudaFree(s.S_keep);
    if (s.hist_keep) cudaFree(s.hist_keep);
    CHECK(cudaMalloc(&s.S_keep, (size_t)count * KDA_STATE_FLOATS * 4)); CHECK(cudaMalloc(&s.hist_keep, (size_t)count * KDA_HIST_FLOATS * 4));
    s.keep = count;
}
static void kda_state_use(KdaState &s, int j)
{
    if (j < 0 || j >= s.keep) { fprintf(stderr, "use_position(%d): only %d positions were kept\n", j, s.keep); exit(1); }
    CHECK(cudaMemcpy(s.S, s.S_keep + (size_t)j * KDA_STATE_FLOATS, KDA_STATE_FLOATS * 4, cudaMemcpyDeviceToDevice));
    CHECK(cudaMemcpy(s.hist, s.hist_keep + (size_t)j * KDA_HIST_FLOATS, KDA_HIST_FLOATS * 4, cudaMemcpyDeviceToDevice));
}

// ---------------------------------------------------------------- kernels with state
// conv_silu with the three rows before this window taken from `hist` (3 x ROWS, oldest first) instead of zeros
__global__ void conv_silu_h(float *out, const float *raw, const float *hist, const unsigned char *op, int comp, int N)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)N * ROWS) return;
    const int n = (int)(idx / ROWS), row = (int)(idx % ROWS);
    const unsigned char *taps = op + QKV_PREFIX + ((size_t)row * 3 + comp) * QKV_COMP + 4;
    auto R = [&](int t) { return t >= 0 ? raw[(size_t)t * ROWS + row] : hist[(size_t)(3 + t) * ROWS + row]; };
    const float r0 = raw[idx];
    const float h2 = R(n - 1), h1 = R(n - 2), h0 = R(n - 3);
    float v = f32at(taps) * r0;
    v = v + f32at(taps + 4) * h0;
    v = v + f32at(taps + 8) * h1;
    v = v + f32at(taps + 12) * h2;
    out[idx] = v * sigmoidf_(v);
}
// afterwards: hist = the last three rows of (hist, raw[0..N)); hist_keep[j] (j < keepN) = the same after token j. One thread per row of ROWS.
__global__ void conv_hist(float *hist, float *hist_keep, size_t keep_stride, const float *raw, int N, int keepN)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= ROWS) return;
    auto R = [&](int t) { return t >= 0 ? raw[(size_t)t * ROWS + row] : hist[(size_t)(3 + t) * ROWS + row]; };
    for (int j = 0; j < keepN; j++)
        for (int i = 0; i < 3; i++) hist_keep[(size_t)j * keep_stride + (size_t)i * ROWS + row] = R(j - 2 + i);
    const float a = R(N - 3), b = R(N - 2), c = R(N - 1);
    hist[row] = a; hist[(size_t)ROWS + row] = b; hist[(size_t)2 * ROWS + row] = c;
}

// recurrence of kernels.cuh, plus: after token n < keepN the head's state is also written to Skeep[n]
__global__ void __launch_bounds__(HEAD) recurrence_keep(float *state, const float *q, const float *k, const float *v, const float *decay, const float *beta, float *out, int N,
    float *Skeep, int keepN)
{
    const int head = blockIdx.x, col = threadIdx.x;
    float *Sg = state + (size_t)head * HEAD * HEAD;
    __shared__ __align__(16) float key[2][HEAD], kbeta[2][HEAD], sq[2][HEAD], dec[2][HEAD];
    const float query_scale = 1.0f / sqrtf(128.0f);
    float S[HEAD];
    #pragma unroll
    for (int row = 0; row < HEAD; row++) S[row] = Sg[row * HEAD + col];
    size_t base = (size_t)head * HEAD;
    float kr = 0.f, dr = 0.f, qr = 0.f, vr = 0.f, br = 0.f;
    if (N > 0) { kr = k[base + col]; dr = decay[base + col]; qr = q[base + col]; vr = v[base + col]; br = beta[head]; }
    for (int n = 0; n < N; n++) {
        const int bf = n & 1;
        key[bf][col] = kr; dec[bf][col] = dr; kbeta[bf][col] = kr * br; sq[bf][col] = qr * query_scale;
        const float value = vr;
        const size_t cur = base;
        base += ROWS;
        if (n + 1 < N) { kr = k[base + col]; dr = decay[base + col]; qr = q[base + col]; vr = v[base + col]; br = beta[(size_t)(n + 1) * HEADS + head]; }
        __syncthreads();
        float prediction = 0.0f;
        #pragma unroll
        for (int r4 = 0; r4 < HEAD / 4; r4++) {
            const float4 dv = *(const float4 *)&dec[bf][r4 * 4], kv = *(const float4 *)&key[bf][r4 * 4];
            const float s0 = dv.x * S[r4 * 4 + 0]; prediction = prediction + s0 * kv.x;
            const float s1 = dv.y * S[r4 * 4 + 1]; prediction = prediction + s1 * kv.y;
            const float s2 = dv.z * S[r4 * 4 + 2]; prediction = prediction + s2 * kv.z;
            const float s3 = dv.w * S[r4 * 4 + 3]; prediction = prediction + s3 * kv.w;
            S[r4 * 4 + 0] = s0; S[r4 * 4 + 1] = s1; S[r4 * 4 + 2] = s2; S[r4 * 4 + 3] = s3;
        }
        const float delta = value - prediction;
        float o = 0.0f;
        #pragma unroll
        for (int r4 = 0; r4 < HEAD / 4; r4++) {
            const float4 bv = *(const float4 *)&kbeta[bf][r4 * 4], qv = *(const float4 *)&sq[bf][r4 * 4];
            const float s0 = S[r4 * 4 + 0] + bv.x * delta; o = o + s0 * qv.x; S[r4 * 4 + 0] = s0;
            const float s1 = S[r4 * 4 + 1] + bv.y * delta; o = o + s1 * qv.y; S[r4 * 4 + 1] = s1;
            const float s2 = S[r4 * 4 + 2] + bv.z * delta; o = o + s2 * qv.z; S[r4 * 4 + 2] = s2;
            const float s3 = S[r4 * 4 + 3] + bv.w * delta; o = o + s3 * qv.w; S[r4 * 4 + 3] = s3;
        }
        out[cur + col] = o;
        if (Skeep && n < keepN) {
            float *Sk = Skeep + (size_t)n * HEADS * HEAD * HEAD + (size_t)head * HEAD * HEAD;
            #pragma unroll
            for (int row = 0; row < HEAD; row++) Sk[row * HEAD + col] = S[row];
        }
    }
    #pragma unroll
    for (int row = 0; row < HEAD; row++) Sg[row * HEAD + col] = S[row];
}

// mla_attention of model.cuh for N new queries at positions pos0.. over the cached keys 0..pos0+n: one block per (query, head); same arithmetic
__global__ void mla_attention_pos(float *out, const float *q, const float *kv, const float *kvlat, int N, size_t pos0)
{
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

// expert_proj_v3 with the expert's address taken from a table: eptr[expert] is the first byte of that expert's EXPERT_RAW bytes (the tables of
// map values / references / reference bases are still indexed by the expert's own number)
__global__ void __launch_bounds__(32) expert_proj_v3p(const unsigned char *const *eptr, const float *mapvals, const uint16_t *refs, const uint32_t *ref_base, int matrix,
    const float *X, int x_stride, const int *slot_token, const int4 *items, float *O, int o_stride)
{
    __shared__ __align__(16) unsigned smem[2048 + 1024];
    const int4 it = items[blockIdx.y];
    const unsigned char *eblk = eptr[it.x];
    const uint32_t rbase = ref_base[it.x * 3 + matrix];
    if (it.w == 16) expert_tile_v3<16>(smem, (float *)(smem + 2048), eblk, mapvals, refs, rbase, matrix, X, x_stride, slot_token, it.y, it.z, O, o_stride);
    else expert_tile_v3<32>(smem, (float *)(smem + 2048), eblk, mapvals, refs, rbase, matrix, X, x_stride, slot_token, it.y, it.z, O, o_stride);
}

// ---------------------------------------------------------------- expert supply: pinned RAM cache, card pool, drive reads
static inline int ekey(int layer, int e) { return layer * EXPERTS + e; }
static double wall_now_() { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

// Slots with replacement of the least recently used entry (policy 0), or (policy 1) of the entry with the lowest heat among the 48 least recently used ones.
// An entry's heat is decayed by 0.8 at every use and raised by the number of tokens that used the expert in that call, so after a prompt pass the experts
// that many prompt tokens chose stay and the ones chosen by a single token go first. Slots in use by the current call are named in `prot` and are not replaced.
static int g_policy = 1, g_chunks = 0;
// policy 2 needs the route cache: g_nu_fn(key) = the number of the next call (after the one being processed) that needs the expert, or a huge number. The entry
// replaced is then the one needed farthest in the future (Belady / MIN); an expert that would be needed later than everything already held is not admitted by worth().
static int (*g_nu_fn)(int key) = nullptr;
static bool g_vq = false;   // K3GPU_VQUEUE: the cards' pools are filled ahead of the calls from the route cache (backend-gpu.cu)
struct SlotCache {
    std::list<int> lru;                                   // keys, most recent first
    struct Ent { std::list<int>::iterator it; unsigned char *p; float heat; int nu; };
    std::unordered_map<int, Ent> map;
    std::set<std::pair<int, int>> by_next;                // (next use, key), only kept in policy 2
    std::vector<unsigned char *> spare;                   // slots never used or released
    bool belady() const { return g_policy == 2 && g_nu_fn; }
    unsigned char *find(int key)
    {
        auto f = map.find(key);
        if (f == map.end()) return nullptr;
        lru.splice(lru.begin(), lru, f->second.it);
        return f->second.p;
    }
    void touch(int key, float w)
    {
        auto f = map.find(key); if (f == map.end()) return;
        if (belady()) { Ent &e = f->second; by_next.erase({ e.nu, key }); e.nu = g_nu_fn(key); by_next.insert({ e.nu, key }); }
        else f->second.heat = f->second.heat * 0.8f + w;
    }
    // true if the expert deserves a slot: a free slot exists, or (policy 2) it is needed sooner than the farthest one held
    bool worth(int key) const
    {
        if (!belady() || !spare.empty() || by_next.empty()) return true;
        return g_nu_fn(key) < by_next.rbegin()->first;
    }
    // a slot for `key`; if an entry had to be replaced, *vkey and *vheat name it (its bytes are still in the slot), else *vkey = -1
    unsigned char *acquire(int key, const std::unordered_set<int> &prot, float heat0, int *vkey, float *vheat)
    {
        unsigned char *p; *vkey = -1;
        if (!spare.empty()) { p = spare.back(); spare.pop_back(); }
        else if (belady()) {
            auto it = by_next.end(); int victim = -1;
            while (it != by_next.begin()) { --it; if (!prot.count(it->second)) { victim = it->second; break; } }
            if (victim < 0) { fprintf(stderr, "expert cache: every slot is in use by one call (too few slots)\n"); exit(1); }
            Ent &v = map[victim]; p = v.p; *vkey = victim; *vheat = v.heat;
            by_next.erase({ v.nu, victim }); lru.erase(v.it); map.erase(victim);
        } else {
            auto it = lru.end(), best = lru.end(); float best_heat = 0; int seen = 0;
            while (it != lru.begin() && seen < (g_policy ? 48 : 1)) {
                --it;
                if (prot.count(*it)) continue;
                const float h = map[*it].heat;
                if (best == lru.end() || h < best_heat) { best = it; best_heat = h; }
                seen++;
            }
            if (best == lru.end()) { fprintf(stderr, "expert cache: every slot is in use by one call (too few slots)\n"); exit(1); }
            const int victim = *best;
            p = map[victim].p; *vkey = victim; *vheat = map[victim].heat;
            if (!by_next.empty()) by_next.erase({ map[victim].nu, victim });
            map.erase(victim); lru.erase(best);
        }
        lru.push_front(key);
        int nu = 0; if (belady()) { nu = g_nu_fn(key); by_next.insert({ nu, key }); }
        map[key] = Ent{ lru.begin(), p, heat0, nu };
        return p;
    }
    // read-ahead (policy 2): a slot for `key`, needed at call cand_nu, only if one is free or the held entry needed farthest ahead (not in prot) is needed later; else nullptr
    unsigned char *try_acquire(int key, int cand_nu, const std::unordered_set<int> &prot, float heat0, int *vkey, float *vheat)
    {
        *vkey = -1;
        if (spare.empty()) {
            auto it = by_next.end(); int victim = -1;
            while (it != by_next.begin()) { --it; if (!prot.count(it->second)) { if (it->first > cand_nu) victim = it->second; break; } }
            if (victim < 0) return nullptr;
        }
        return acquire(key, prot, heat0, vkey, vheat);
    }
    // takes `key` out of the cache; its slot is returned to the caller, who releases it when its bytes are no longer needed
    unsigned char *remove(int key)
    {
        auto f = map.find(key);
        if (f == map.end()) return nullptr;
        unsigned char *p = f->second.p; by_next.erase({ f->second.nu, key }); lru.erase(f->second.it); map.erase(f);
        return p;
    }
    void release(unsigned char *p) { spare.push_back(p); }
};

// ---- which slot cache runs (K3GPU_SLOTCACHE). SlotCache above is the C++ cache, untouched; "rust" sends every call to libslotcache.so (src/rust/slotcache), loaded by
// backend-gpu.cu before the first slot is added. The choice is per process and never changes, so a cache never switches implementation. There is no fallback.
// Like the original, none of this is thread-safe: the main thread is the only caller.
struct ScApi {
    bool on = false;
#ifdef K3GPU_SLOTCACHE_BUILT
    void (*set_policy)(int) = nullptr; void (*set_next_use)(sc_next_use_fn) = nullptr;
    sc_cache *(*nw)() = nullptr;
    void (*add_spare)(sc_cache *, unsigned char *) = nullptr;
    int (*contains)(const sc_cache *, int) = nullptr;
    unsigned char *(*find)(sc_cache *, int) = nullptr;
    void (*touch)(sc_cache *, int, float) = nullptr;
    int (*worth)(const sc_cache *, int) = nullptr;
    unsigned char *(*acquire)(sc_cache *, int, const int *, size_t, float, int *, float *) = nullptr;
    unsigned char *(*try_acquire)(sc_cache *, int, int, const int *, size_t, float, int *, float *) = nullptr;
    unsigned char *(*remove)(sc_cache *, int) = nullptr;
    void (*release)(sc_cache *, unsigned char *) = nullptr;
#endif
};
static ScApi g_sc;
static unsigned long g_sc_calls[3];   // acquire, try_acquire and find calls, whichever cache took them
static void slotcache_set_policy(int p)
{
    g_policy = p;
#ifdef K3GPU_SLOTCACHE_BUILT
    if (g_sc.on) g_sc.set_policy(p);
#endif
}
static void slotcache_set_next_use(int (*f)(int))
{
    g_nu_fn = f;
#ifdef K3GPU_SLOTCACHE_BUILT
    if (g_sc.on) g_sc.set_next_use(f);
#endif
}
struct SlotCacheSel {
    SlotCache cpp;
#ifdef K3GPU_SLOTCACHE_BUILT
    sc_cache *rs = nullptr;
    sc_cache *R() { if (!rs) rs = g_sc.nw(); return rs; }
    static const std::vector<int> &prot_vec(const std::unordered_set<int> &prot) { static std::vector<int> v; v.assign(prot.begin(), prot.end()); return v; }   // membership only: order is irrelevant
#endif
    void add_spare(unsigned char *p)
    {
#ifdef K3GPU_SLOTCACHE_BUILT
        if (g_sc.on) { g_sc.add_spare(R(), p); return; }
#endif
        cpp.spare.push_back(p);
    }
    bool contains(int key) const
    {
#ifdef K3GPU_SLOTCACHE_BUILT
        if (g_sc.on) return rs && g_sc.contains(rs, key);
#endif
        return cpp.map.count(key) != 0;
    }
    unsigned char *find(int key)
    {
        g_sc_calls[2]++;
#ifdef K3GPU_SLOTCACHE_BUILT
        if (g_sc.on) return rs ? g_sc.find(rs, key) : nullptr;
#endif
        return cpp.find(key);
    }
    void touch(int key, float w)
    {
#ifdef K3GPU_SLOTCACHE_BUILT
        if (g_sc.on) { if (rs) g_sc.touch(rs, key, w); return; }
#endif
        cpp.touch(key, w);
    }
    bool worth(int key) const
    {
#ifdef K3GPU_SLOTCACHE_BUILT
        if (g_sc.on) return !rs || g_sc.worth(rs, key);
#endif
        return cpp.worth(key);
    }
    unsigned char *acquire(int key, const std::unordered_set<int> &prot, float heat0, int *vkey, float *vheat)
    {
        g_sc_calls[0]++;
#ifdef K3GPU_SLOTCACHE_BUILT
        if (g_sc.on) { const std::vector<int> &v = prot_vec(prot); return g_sc.acquire(R(), key, v.data(), v.size(), heat0, vkey, vheat); }
#endif
        return cpp.acquire(key, prot, heat0, vkey, vheat);
    }
    unsigned char *try_acquire(int key, int cand_nu, const std::unordered_set<int> &prot, float heat0, int *vkey, float *vheat)
    {
        g_sc_calls[1]++;
#ifdef K3GPU_SLOTCACHE_BUILT
        if (g_sc.on) { const std::vector<int> &v = prot_vec(prot); return g_sc.try_acquire(R(), key, cand_nu, v.data(), v.size(), heat0, vkey, vheat); }
#endif
        return cpp.try_acquire(key, cand_nu, prot, heat0, vkey, vheat);
    }
    unsigned char *remove(int key)
    {
#ifdef K3GPU_SLOTCACHE_BUILT
        if (g_sc.on) return rs ? g_sc.remove(rs, key) : nullptr;
#endif
        return cpp.remove(key);
    }
    void release(unsigned char *p)
    {
#ifdef K3GPU_SLOTCACHE_BUILT
        if (g_sc.on) { g_sc.release(R(), p); return; }
#endif
        cpp.release(p);
    }
};

// ---- which pool start-up runs (K3GPU_POOLINIT, see backend-gpu.cu): cpp, the default, is HostCache::init / CardPool::init below; rust = libpoolinit.so (src/rust/poolinit), which makes the
// same memory and the same stderr lines. The cache handed to the library is the SlotCacheSel itself, and sc_add_spare goes back to its add_spare, so this works with either slot cache.
struct PiApi {
    bool on = false;
#ifdef K3GPU_POOLINIT_BUILT
    void (*host_init)(const pi_ops *, const pi_host_cfg *, pi_host_out *) = nullptr;
    void (*pool_init)(const pi_ops *, const pi_pool_cfg *, pi_pool_out *) = nullptr;
#endif
};
static PiApi g_pi;
#ifdef K3GPU_POOLINIT_BUILT
static_assert(PI_HOST_ALLOC_DEFAULT == cudaHostAllocDefault && PI_HOST_REGISTER_PORTABLE == cudaHostRegisterPortable && PI_STREAM_NON_BLOCKING == cudaStreamNonBlocking && PI_EVENT_DISABLE_TIMING == cudaEventDisableTiming,
              "poolinit.h flag values must be the CUDA ones");
static int pi_set_device(int d) { return (int)cudaSetDevice(d); }
static int pi_mem_get_info(size_t *f, size_t *t) { return (int)cudaMemGetInfo(f, t); }
static int pi_device_malloc(void **p, size_t n) { return (int)cudaMalloc(p, n); }
static int pi_host_register(void *p, size_t n, unsigned f) { return (int)cudaHostRegister(p, n, f); }
static int pi_host_alloc(void **p, size_t n, unsigned f) { return (int)cudaHostAlloc(p, n, f); }
static int pi_stream_create(void **s, unsigned f) { return (int)cudaStreamCreateWithFlags((cudaStream_t *)s, f); }
static int pi_event_create(void **e, unsigned f) { return (int)cudaEventCreateWithFlags((cudaEvent_t *)e, f); }
static const char *pi_error_string(int c) { return cudaGetErrorString((cudaError_t)c); }
static void pi_add_spare(sc_cache *c, unsigned char *s) { ((SlotCacheSel *)c)->add_spare(s); }
static const pi_ops g_pi_ops = { pi_set_device, pi_mem_get_info, pi_device_malloc, pi_host_register, pi_host_alloc, pi_stream_create, pi_event_create, pi_error_string, pi_add_spare };
#endif

// The RAM cache is made all at once: cudaHostAlloc does not run in parallel (about 2 GiB/s, measured), but memory that 16 threads have already touched
// can be registered as pinned at about 45 GiB/s (pinbench.cu), so 150 GB take seconds, not minutes.
struct HostCache {
    SlotCacheSel c; size_t cap_slots = 0; bool ready = false;
    enum { CHUNK_SLOTS = 32 };
    void init(size_t bytes)
    {
#ifdef K3GPU_POOLINIT_BUILT
        if (g_pi.on) {
            pi_host_cfg cfg = { bytes, (size_t)EXPERT_RAW, (sc_cache *)&c, __FILE__ };
            pi_host_out out = { 0, 0 };
            g_pi.host_init(&g_pi_ops, &cfg, &out);
            cap_slots = out.cap_slots; ready = out.ready != 0;
            return;
        }
#endif
        cap_slots = bytes / EXPERT_RAW; ready = true;
        const size_t chunk_bytes = (size_t)CHUNK_SLOTS * EXPERT_RAW, n_chunks = cap_slots / CHUNK_SLOTS;
        std::vector<unsigned char *> base(n_chunks, nullptr);
        std::atomic<size_t> next(0);
        const double t0 = wall_now_();
        auto touch = [&]() {
            for (;;) {
                const size_t i = next.fetch_add(1);
                if (i >= n_chunks) return;
                void *p = mmap(nullptr, chunk_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                if (p == MAP_FAILED) { fprintf(stderr, "host expert cache: cannot map %zu bytes\n", chunk_bytes); exit(1); }
                madvise(p, chunk_bytes, MADV_HUGEPAGE);
                memset(p, 0, chunk_bytes);
                base[i] = (unsigned char *)p;
            }
        };
        std::vector<std::thread> th;
        for (int t = 1; t < 16; t++) th.emplace_back(touch);
        touch();
        for (auto &t : th) t.join();
        CHECK(cudaSetDevice(0));
        for (size_t i = 0; i < n_chunks; i++) CHECK(cudaHostRegister(base[i], chunk_bytes, cudaHostRegisterPortable));
        for (size_t i = n_chunks; i-- > 0;)
            for (int s = CHUNK_SLOTS; s-- > 0;) c.add_spare(base[i] + (size_t)s * EXPERT_RAW);
        cap_slots = n_chunks * CHUNK_SLOTS;
        fprintf(stderr, "K3GPU: host expert cache of %zu slots (%.1f GB) pinned in %.1f s\n", cap_slots, cap_slots * (double)EXPERT_RAW / 1e9, wall_now_() - t0);
    }
};

struct ExpertStats { unsigned long needed = 0, hit_card = 0, hit_host = 0, from_drive = 0, written_back = 0; double t_read = 0, t_copy = 0, t_ensure = 0; };

enum { MAX_READERS = 32 };
struct CardPool {
    int dev = 0, slots = 0, readers = 0; bool ready = false;
    SlotCacheSel c;
    unsigned char *base = nullptr;
    cudaStream_t copy = nullptr;                       // host RAM <-> card traffic of the main thread
    cudaStream_t rstream[MAX_READERS];                 // one per reading thread
    unsigned char *bounce[MAX_READERS];                // one pinned slot per reading thread: the drive's bytes land here and go on to the card
    cudaEvent_t written_back = nullptr;
    const unsigned char **d_eptr = nullptr;
    std::vector<const unsigned char *> h_eptr;
    ExpertStats st;
    void init(int device, size_t reserve_bytes, int nreaders)
    {
#ifdef K3GPU_POOLINIT_BUILT
        if (g_pi.on) {
            pi_pool_cfg cfg = { device, reserve_bytes, nreaders, (size_t)EXPERT_RAW, (int)EXPERTS, (sc_cache *)&c, __FILE__ };
            pi_pool_out out; memset(&out, 0, sizeof out);
            g_pi.pool_init(&g_pi_ops, &cfg, &out);
            dev = out.dev; slots = out.slots; readers = out.readers; base = out.base; copy = (cudaStream_t)out.copy; written_back = (cudaEvent_t)out.written_back; d_eptr = out.d_eptr;
            for (int i = 0; i < MAX_READERS; i++) { rstream[i] = (cudaStream_t)out.rstream[i]; bounce[i] = out.bounce[i]; }
            h_eptr.assign(EXPERTS, nullptr);
            ready = true;
            return;
        }
#endif
        dev = device; readers = std::min(nreaders, (int)MAX_READERS);
        CHECK(cudaSetDevice(dev));
        size_t f, t; CHECK(cudaMemGetInfo(&f, &t));
        const size_t usable = f > reserve_bytes ? f - reserve_bytes : 0;
        slots = (int)(usable / EXPERT_RAW);
        if (slots < 64) { fprintf(stderr, "card %d: only %d expert slots fit after the reserve\n", dev, slots); exit(1); }
        CHECK(cudaMalloc(&base, (size_t)slots * EXPERT_RAW));
        for (int i = slots; i-- > 0;) c.add_spare(base + (size_t)i * EXPERT_RAW);
        CHECK(cudaStreamCreateWithFlags(&copy, cudaStreamNonBlocking));
        for (int i = 0; i < readers; i++) {
            CHECK(cudaStreamCreateWithFlags(&rstream[i], cudaStreamNonBlocking));
            CHECK(cudaHostAlloc((void **)&bounce[i], EXPERT_RAW, cudaHostAllocDefault));
        }
        CHECK(cudaEventCreateWithFlags(&written_back, cudaEventDisableTiming));
        CHECK(cudaMalloc(&d_eptr, EXPERTS * sizeof(void *)));
        h_eptr.assign(EXPERTS, nullptr);
        ready = true;
    }
};

// runs the jobs on up to nthreads threads; a job is told the number of the thread that runs it
static void run_parallel(std::vector<std::function<void(int)>> &jobs, int nthreads)
{
    if (jobs.empty()) return;
    std::atomic<size_t> next(0);
    auto work = [&](int tid) { for (;;) { const size_t i = next.fetch_add(1); if (i >= jobs.size()) return; jobs[i](tid); } };
    const int n = std::min<int>(nthreads, (int)jobs.size());
    std::vector<std::thread> th;
    for (int i = 1; i < n; i++) th.emplace_back(work, i);
    work(0);
    for (auto &t : th) t.join();
}

static FILE *g_trace = nullptr;
static double wall_now() { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

// Read-ahead into the RAM cache (route cache, K3GPU_ROUTECACHE): experts that a later call will need are read from the drive into slots of the RAM cache by
// background threads, while the current call computes. A slot is named in g_pend until the call that needs it has taken it (main thread only; the reading
// thread only touches the Pending's state), and is not replaced while it is named.
struct Pending { std::atomic<int> state; unsigned char *slot; Pending() : state(0), slot(nullptr) {} };   // 0 queued or being read, 1 read
static std::unordered_map<int, Pending *> g_pend;
static std::unordered_set<int> g_pend_keys;
static unsigned long g_wb_skipped = 0;   // policy 2: experts a card gave up that were not worth a RAM slot (needed later than everything held there)
static double g_t_pwait = 0; static unsigned long g_pf_used = 0, g_pf_waited = 0;
static void (*g_route_hook)(int layer, const std::vector<int> &experts) = nullptr;   // called once per MoE call before its experts are made resident
// K3GPU_PREFILL (timing only): the store behind the cards behaves as if every expert were already in RAM. No drive is read; an expert's bytes come from 4 real records of its layer
// (g_slab[layer]), the router's choice is replaced by the route cache's (g_force_routes), so the outputs are NOT the model's.
static bool g_prefill = false;
// While the main thread uploads a layer's activations, the reading threads start no new copy to a card (an upload queues behind every expert copy already on the card's copy engine).
static std::atomic<int> g_act_hold(0);
static inline void act_gate() { while (g_act_hold.load(std::memory_order_acquire)) std::this_thread::yield(); }
static unsigned char *g_slab[93];
static void (*g_force_routes)(int layer, int n, int *selected) = nullptr;
// ---- which planner runs (K3GPU_PLANNER, see backend-gpu.cu): cpp, the default, is rc_plan / vq_sweep / vq_issue / rc_plan_vq / route_hook in backend-gpu.cu together with g_pend,
// g_pend_keys, g_arr and the functions below; rust = libplanner.so (src/rust/planner), which keeps the pending reads, the arrivals and the release / write-back lists itself.
// With it on, the three functions below hand over to the library, and so does every place that read g_pend_keys. It needs the Rust copy queue, slot cache and route table.
struct SpApi {
    bool on = false;
#ifdef K3GPU_PLANNER_BUILT
    void (*init)(const sp_ops *, const sp_config *) = nullptr;
    void (*layer_set)(int, int, int) = nullptr; void (*layer_set_fd)(int, int) = nullptr;
    void (*pool_set)(int, sc_cache *, int, int) = nullptr; void (*host_set)(sc_cache *, size_t) = nullptr;
    void (*route_hook)(int, const int *, size_t) = nullptr;
    void (*pend_take)(int) = nullptr; void (*arr_wait)(int) = nullptr;
    unsigned char *(*host_acquire)(int, float, int *, float *) = nullptr;
    void (*get_stats)(sp_stats *) = nullptr;
#endif
};
static SpApi g_sp;
static void pend_take(int key)   // waits until the expert's bytes are in its RAM slot, then forgets the entry
{
#ifdef K3GPU_PLANNER_BUILT
    if (g_sp.on) { g_sp.pend_take(key); return; }
#endif
    auto f = g_pend.find(key);
    if (f == g_pend.end()) return;
    Pending *p = f->second;
    if (p->state.load(std::memory_order_acquire) == 0) {
        const double t0 = wall_now();
        while (p->state.load(std::memory_order_acquire) == 0) std::this_thread::yield();
        g_t_pwait += wall_now() - t0; g_pf_waited++;
    }
    g_pf_used++; g_pend.erase(f); g_pend_keys.erase(key); delete p;
}

// Experts on their way into a card's pool (K3GPU_VQUEUE): the pool entry exists (so it is counted and protected by its next use) before its bytes have arrived;
// whoever is about to use or replace the slot waits here first (main thread only; a reading thread sets the state when its copy to the card is done).
struct Arrival { std::atomic<int> state; Arrival() : state(0) {} };
static std::unordered_map<int, std::shared_ptr<Arrival>> g_arr;
static double g_t_vwait = 0; static unsigned long g_vq_waited = 0;
static void arr_wait(int key)
{
#ifdef K3GPU_PLANNER_BUILT
    if (g_sp.on) { g_sp.arr_wait(key); return; }
#endif
    auto f = g_arr.find(key);
    if (f == g_arr.end()) return;
    std::shared_ptr<Arrival> a = f->second; g_arr.erase(f);
    if (a->state.load(std::memory_order_acquire) == 0) {
        const double t0 = wall_now();
        while (a->state.load(std::memory_order_acquire) == 0) std::this_thread::yield();
        g_t_vwait += wall_now() - t0; g_vq_waited++;
    }
}

// The RAM cache's acquire for a card's victim, with the pending reads protected: the planner library owns them when it runs, g_pend_keys otherwise.
static unsigned char *host_acquire_pend(HostCache &hc, int key, float heat, int *vkey, float *vheat)
{
#ifdef K3GPU_PLANNER_BUILT
    if (g_sp.on) return g_sp.host_acquire(key, heat, vkey, vheat);
#endif
    return hc.c.acquire(key, g_pend_keys, heat, vkey, vheat);
}

// ---- which supply runs (K3GPU_SUPPLY, see backend-gpu.cu): cpp, the default, is pool_ensure below; rust = libsupply.so (src/rust/supply), which does the same calls in the same order
// through the table below. The caches it is given are the SlotCacheSel objects, so it works with either slot cache.
struct SyApi {
    bool on = false;
#ifdef K3GPU_SUPPLY_BUILT
    void (*init)(const sy_ops *, const sy_config *) = nullptr;
    void (*ensure)(sy_pool *, sy_host *, int, int, const int *, size_t, const int *, int) = nullptr;
    unsigned long (*wb_skipped)(void) = nullptr;
#endif
};
static SyApi g_sy;
#ifdef K3GPU_SUPPLY_BUILT
static int sy_device_sync() { return (int)cudaDeviceSynchronize(); }
static int sy_stream_sync(void *s) { return (int)cudaStreamSynchronize((cudaStream_t)s); }
static int sy_set_device(int d) { return (int)cudaSetDevice(d); }
static int sy_d2h(void *dst, const void *src, size_t n, void *s) { return (int)cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToHost, (cudaStream_t)s); }
static int sy_h2d(void *dst, const void *src, size_t n, void *s) { return (int)cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, (cudaStream_t)s); }
static int sy_event_record(void *e, void *s) { return (int)cudaEventRecord((cudaEvent_t)e, (cudaStream_t)s); }
static int sy_wait_event(void *s, void *e, unsigned f) { return (int)cudaStreamWaitEvent((cudaStream_t)s, (cudaEvent_t)e, f); }
static void sy_check(int status, const char *expr) { if (status) { fprintf(stderr, "%s: %s: %s\n", __FILE__, expr, cudaGetErrorString((cudaError_t)status)); exit(1); } }
static unsigned char *sy_sc_find(sc_cache *c, int key) { return ((SlotCacheSel *)c)->find(key); }
static void sy_sc_touch(sc_cache *c, int key, float w) { ((SlotCacheSel *)c)->touch(key, w); }
static int sy_sc_worth(const sc_cache *c, int key) { return ((SlotCacheSel *)c)->worth(key) ? 1 : 0; }
static unsigned char *sy_sc_acquire(sc_cache *c, int key, const int *prot, size_t n, float heat0, int *vkey, float *vheat)
{
    SlotCacheSel *s = (SlotCacheSel *)c;
#ifdef K3GPU_SLOTCACHE_BUILT
    if (g_sc.on) { g_sc_calls[0]++; return g_sc.acquire(s->R(), key, prot, n, heat0, vkey, vheat); }
#endif
    return s->acquire(key, std::unordered_set<int>(prot, prot + n), heat0, vkey, vheat);
}
static unsigned char *sy_sc_remove(sc_cache *c, int key) { return ((SlotCacheSel *)c)->remove(key); }
static void sy_sc_release(sc_cache *c, unsigned char *p) { ((SlotCacheSel *)c)->release(p); }
static HostCache *g_sy_host = nullptr;   // the RAM cache the call in progress was given (sy_ops.host_acquire has no argument for it)
static unsigned char *sy_host_acquire(int key, float heat, int *vkey, float *vheat) { return host_acquire_pend(*g_sy_host, key, heat, vkey, vheat); }
static const sy_ops g_sy_ops = { sy_device_sync, sy_stream_sync, sy_set_device, sy_d2h, sy_h2d, sy_event_record, sy_wait_event, sy_check,
                                 sy_sc_find, sy_sc_touch, sy_sc_worth, sy_sc_acquire, sy_sc_remove, sy_sc_release, pend_take, arr_wait, sy_host_acquire };
static void sy_pool_ensure(CardPool &pool, HostCache &hc, int layer, int fd, const std::vector<int> &experts, const int *counts, int readers)
{
    sy_pool p; memset(&p, 0, sizeof p);
    p.dev = pool.dev; p.readers = pool.readers; p.cache = (sc_cache *)&pool.c; p.copy = pool.copy; p.written_back = pool.written_back; p.h_eptr = pool.h_eptr.data();
    for (int i = 0; i < MAX_READERS; i++) { p.rstream[i] = pool.rstream[i]; p.bounce[i] = pool.bounce[i]; }
    p.st.needed = pool.st.needed; p.st.hit_card = pool.st.hit_card; p.st.hit_host = pool.st.hit_host; p.st.from_drive = pool.st.from_drive; p.st.written_back = pool.st.written_back;
    p.st.t_read = pool.st.t_read; p.st.t_copy = pool.st.t_copy; p.st.t_ensure = pool.st.t_ensure;
    sy_host h = { (sc_cache *)&hc.c, hc.cap_slots };
    g_sy_host = &hc;
    g_sy.ensure(&p, &h, layer, fd, experts.data(), experts.size(), counts, readers);
    pool.st.needed = p.st.needed; pool.st.hit_card = p.st.hit_card; pool.st.hit_host = p.st.hit_host; pool.st.from_drive = p.st.from_drive; pool.st.written_back = p.st.written_back;
    pool.st.t_read = p.st.t_read; pool.st.t_copy = p.st.t_copy; pool.st.t_ensure = p.st.t_ensure;
}
#endif

// makes the given experts of `layer` resident in the card's pool; afterwards pool.h_eptr[e] is the address of expert e
// Where an expert comes from: the card's pool (hit); else the RAM cache, which is taken out of it (the expert moves up to the card); else the drive, read
// straight into a small pinned slot of the reading thread and on to the card (it is NOT kept in RAM). The RAM cache holds what the cards have let go: an expert
// that a card replaces is copied down into it first, so the card pool and the RAM cache never hold the same expert and their sizes add up.
// `counts[e]` is the number of tokens of this call that chose expert e (the heat of the expert).
static void pool_ensure(CardPool &pool, HostCache &hc, int layer, int fd, const std::vector<int> &experts, const int *counts, int readers)
{
#ifdef K3GPU_SUPPLY_BUILT
    if (g_sy.on) { sy_pool_ensure(pool, hc, layer, fd, experts, counts, readers); return; }
#endif
    struct Timed { double &acc; double t0; Timed(double &a) : acc(a), t0(wall_now()) {} ~Timed() { acc += wall_now() - t0; } } timed(pool.st.t_ensure);
    if (g_vq) CHECK(cudaStreamSynchronize(0)); else CHECK(cudaDeviceSynchronize());   // kernels of an earlier group or layer may still be reading slots that are about to be replaced (with the queue on, copies of other streams are not waited for)
    std::unordered_set<int> prot, noprot;
    for (int e : experts) prot.insert(ekey(layer, e));
    struct Need { int e; unsigned char *dst, *host; };
    struct Victim { int key; float heat; unsigned char *src; };
    std::vector<Need> need; std::vector<Victim> victims;
    pool.st.needed += experts.size();
    for (int e : experts) {
        const int key = ekey(layer, e);
        unsigned char *p = pool.c.find(key);
        if (p) { if (g_vq) arr_wait(key); pool.c.touch(key, (float)counts[e]); pool.h_eptr[e] = p; pool.st.hit_card++; continue; }
        int vk; float vh;
        p = pool.c.acquire(key, prot, (float)counts[e], &vk, &vh);
        if (vk >= 0) { if (g_vq) arr_wait(vk); victims.push_back(Victim{ vk, vh, p }); }
        pool.h_eptr[e] = p; need.push_back(Need{ e, p, nullptr });
    }
    if (need.empty()) return;
    // 1. experts found in RAM leave the RAM cache; their slots are held until the copies are done, so nothing is written over them
    std::vector<unsigned char *> held;
    for (auto &n : need) {
        const int key = ekey(layer, n.e);
        if (hc.c.find(key)) { pend_take(key); n.host = hc.c.remove(key); held.push_back(n.host); pool.st.hit_host++; }
    }
    CHECK(cudaSetDevice(pool.dev));
    // 2. the experts the card gives up are copied down into the RAM cache (if the cache is full its least useful entry is dropped)
    for (auto &v : victims) {
        if (!hc.cap_slots || g_prefill) break;
        if (!hc.c.worth(v.key)) { g_wb_skipped++; continue; }
        int dk; float dh;
        unsigned char *h = host_acquire_pend(hc, v.key, v.heat, &dk, &dh);
        CHECK(cudaMemcpyAsync(h, v.src, EXPERT_RAW, cudaMemcpyDeviceToHost, pool.copy));
        pool.st.written_back++;
    }
    CHECK(cudaEventRecord(pool.written_back, pool.copy));   // from here on a slot of the pool can be overwritten
    // 3. RAM -> card, queued behind the copies down on the same stream
    for (auto &n : need) if (n.host) CHECK(cudaMemcpyAsync(n.dst, g_prefill ? g_slab[layer] + (size_t)(n.e & 3) * EXPERT_RAW : n.host, EXPERT_RAW, cudaMemcpyHostToDevice, pool.copy));
    // 4. drive -> pinned slot -> card, one expert at a time per reading thread; the reads start at once, the copy to the card waits for the copies down
    std::vector<std::function<void(int)>> jobs;
    const int dev = pool.dev;
    int n_drive = 0;
    for (auto &n : need) if (!n.host) n_drive++;
    // a call with few experts cannot fill the drive with whole-expert reads (each takes ~10 ms alone), so each expert is read in chunks by several threads
    const int pool_readers = std::min(readers, pool.readers);
    int nchunk = g_chunks > 0 ? g_chunks : (n_drive >= pool_readers || n_drive == 0) ? 1 : std::min(8, (pool_readers + n_drive - 1) / n_drive);
    const size_t chunk = ((((size_t)EXPERT_RAW + nchunk - 1) / nchunk) + 4095) / 4096 * 4096;
    for (auto &n : need) {
        if (n.host) continue;
        pool.st.from_drive++;
        for (int k = 0; k < nchunk; k++) {
            const size_t co = (size_t)k * chunk; if (co >= (size_t)EXPERT_RAW) break;
            const size_t len = std::min(chunk, (size_t)EXPERT_RAW - co);
            unsigned char *dev_dst = n.dst + co; const off_t off = (off_t)n.e * EXPERT_RAW + (off_t)co; const int e = n.e;
            jobs.push_back([=, &pool](int tid) {
                unsigned char *b = pool.bounce[tid];
                if (g_prefill) b = g_slab[layer] + (size_t)(e & 3) * EXPERT_RAW + co;
                else {
                size_t got = 0;
                while (got < len) {
                    const ssize_t r = pread(fd, b + got, len - got, off + (off_t)got);
                    if (r <= 0) { fprintf(stderr, "reading expert %d of layer %d failed: %s\n", e, layer, r < 0 ? strerror(errno) : "end of file"); exit(1); }
                    got += (size_t)r;
                }
                }
                CHECK(cudaSetDevice(dev));
                CHECK(cudaStreamWaitEvent(pool.rstream[tid], pool.written_back, 0));
                CHECK(cudaMemcpyAsync(dev_dst, b, len, cudaMemcpyHostToDevice, pool.rstream[tid]));
                CHECK(cudaStreamSynchronize(pool.rstream[tid]));   // the pinned slot is free for this thread's next read
            });
        }
    }
    const double t0 = wall_now();
    run_parallel(jobs, pool_readers);
    const double t1 = wall_now();
    CHECK(cudaStreamSynchronize(pool.copy));
    for (unsigned char *h : held) hc.c.release(h);
    pool.st.t_read += t1 - t0; pool.st.t_copy += wall_now() - t1;
}

// ---------------------------------------------------------------- the pieces of a layer, with state and sparse experts
// KDA attention from w.Xn into `res`, continuing from st (the caller resets st for a fresh sequence)
static void kda_attention_state(Work &w, const Layer &y, KdaState &st, float *res)
{
    const int N = w.N; const Trunk &T = y.T;
    for (int c = 0; c < 3; c++)
        gemm_raw(w.Xn, N, W, T.op + QKV_PREFIX + (size_t)c * QKV_COMP + 20, (size_t)3 * QKV_COMP, T.op + QKV_PREFIX + (size_t)c * QKV_COMP, (size_t)3 * QKV_COMP, ROWS, w.raw[c], ROWS);
    for (int c = 0; c < 3; c++) conv_silu_h<<<blocks((size_t)N * ROWS), 256>>>(w.qkv[c], w.raw[c], st.hist + (size_t)c * 3 * ROWS, T.op, c, N);
    for (int c = 0; c < 3; c++)   // before w.raw[0..2] are reused below
        conv_hist<<<(ROWS + 127) / 128, 128>>>(st.hist + (size_t)c * 3 * ROWS, st.hist_keep ? st.hist_keep + (size_t)c * 3 * ROWS : nullptr, KDA_HIST_FLOATS, w.raw[c], N, std::min(N, st.keep));
    l2_heads<<<N * HEADS, HEAD>>>(w.qkv[0]); l2_heads<<<N * HEADS, HEAD>>>(w.qkv[1]);
    gemm(w.Xn, N, W, T, 11, w.beta); beta_sigmoid<<<blocks((size_t)N * HEADS), 256>>>(w.beta, (size_t)N * HEADS);
    gemm(w.Xn, N, W, T, 12, w.dhid); gemm(w.dhid, N, HEAD, T, 13, w.decay);
    decay_transform<<<blocks((size_t)N * ROWS), 256>>>(w.decay, rec_data(T, 39), rec_data(T, 18), N);
    recurrence_keep<<<HEADS, HEAD>>>(st.S, w.qkv[0], w.qkv[1], w.qkv[2], w.decay, w.beta, w.attn, N, st.S_keep, std::min(N, st.keep));
    gemm(w.Xn, N, W, T, 6, w.gate);
    normalize_rows<<<N * HEADS, HEAD>>>(w.attn, w.attn, rec_data(T, 19), HEAD);
    gate_mul<<<blocks((size_t)N * ROWS), 256>>>(w.gate, w.attn, (size_t)N * ROWS);
    gemm(w.gate, N, ROWS, T, 7, res);
}

// MLA attention from w.Xn into `res`; this window's keys and values are written into the cache at pos0.. and the queries attend to the cache
static void mla_block_cached(Work &w, const Layer &y, MlaCache &mc, size_t pos0, float *res)
{
    const int N = w.N; const Trunk &T = y.T;
    if (pos0 + (size_t)N > mc.cap) { fprintf(stderr, "layer %d: position %zu is beyond the %zu positions of the attention cache\n", y.L, pos0 + N, mc.cap); exit(1); }
    float *kv = mc.kv + pos0 * MLA_EXP, *lat = mc.lat + pos0 * MLA_KVLAT;
    mla_project(w.Xn, N, y, 20, w.mqlat);
    normalize_rows<<<N, 256>>>(w.mqlat, w.mqlat, mla_rec(y, 21), MLA_QLAT);
    mla_project(w.mqlat, N, y, 22, w.mq);
    mla_project(w.Xn, N, y, 23, lat);
    normalize_prefix<<<N, 256>>>(lat, mla_rec(y, 24), MLA_KV, MLA_KVLAT);
    copy_prefix<<<blocks((size_t)N * MLA_KV), 256>>>(w.mkvn, lat, MLA_KV, MLA_KVLAT, (size_t)N);
    mla_project(w.mkvn, N, y, 25, kv);
    if (pos0 == 0) {   // a fresh window: the tiled kernel of model.cuh (the same arithmetic)
        static float *scratch[16]; static size_t scratch_bytes[16];
        int dev = 0, sms = 0, per_sm = 0;
        const size_t SN = (((size_t)N + 3) / 4 | 1) * 4;
        CHECK(cudaGetDevice(&dev)); CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
        const size_t smem = (size_t)MLA_QT * 192 * 8 + 32 * 129 * 8;
        CHECK(cudaFuncSetAttribute(mla_attention_tiled, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, mla_attention_tiled, 256, smem));
        const int items = ((N + MLA_QT - 1) / MLA_QT) * HEADS, grid = std::min(items, sms * std::max(per_sm, 1));
        const size_t need = (size_t)grid * MLA_QT * SN * 4;
        if (dev >= 16) { fprintf(stderr, "device index %d not supported\n", dev); exit(1); }
        if (scratch_bytes[dev] < need) { if (scratch[dev]) CHECK(cudaFree(scratch[dev])); CHECK(cudaMalloc(&scratch[dev], need)); scratch_bytes[dev] = need; }
        mla_attention_tiled<<<grid, 256, smem>>>(w.attn, w.mq, mc.kv, mc.lat, N, (int)SN, scratch[dev], items);
    } else {
        const size_t shared = (pos0 + (size_t)N) * 4;
        if (shared > 48 * 1024) CHECK(cudaFuncSetAttribute(mla_attention_pos, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared));
        mla_attention_pos<<<dim3(N, HEADS), 128, shared>>>(w.attn, w.mq, mc.kv, mc.lat, N, pos0);
    }
    gemm(w.Xn, N, W, T, 6, w.gate);
    gate_mul<<<blocks((size_t)N * ROWS), 256>>>(w.gate, w.attn, (size_t)N * ROWS);
    gemm(w.gate, N, ROWS, T, 7, res);
}

static unsigned short g_last_routes[93][16];   // the first token's 16 chosen experts of each layer in the last call

// the mixture of experts for layer y.L from w.post, added into `res`; the experts it needs are made resident group by group
static void moe_block_sparse(Work &w, const Layer &y, CardPool &pool, HostCache &hc, int fd, int readers, float *res)
{
    const int N = w.N; const Trunk &T = y.T;
    CHECK(cudaMemset(w.bad, 0, 4));
    transpose_rows<<<blocks((size_t)N * W), 256>>>(w.post_t, w.post, N, W);
    router_scores<<<dim3((N + 127) / 128, EXPERTS), 128>>>(w.scores, w.choices, rec_data(T, 29), rec_data(T, 30), w.post_t, N);
    top16<<<blocks(N), 256>>>(w.scores, w.choices, w.selected, w.weights, w.bad, N);
    CHECK(cudaMemcpy(w.h_selected.data(), w.selected, (size_t)N * TOP * 4, cudaMemcpyDeviceToHost));
    if (g_force_routes) g_force_routes(y.L, N, w.h_selected.data());
    { int h_bad = 0; CHECK(cudaMemcpy(&h_bad, w.bad, 4, cudaMemcpyDeviceToHost)); if (h_bad && !g_prefill) { fprintf(stderr, "layer %d: non-finite router choice for %d tokens\n", y.L, h_bad); exit(1); } }
    for (int r = 0; r < TOP; r++) g_last_routes[y.L][r] = (unsigned short)w.h_selected[r];
    {   // MODEL_GPU_ROUTES=dir: write each layer's chosen experts (N x 16 ints) to dir/routes-L.bin (appended per call)
        const char *dir = getenv("MODEL_GPU_ROUTES");
        if (dir && *dir) {
            char name[4096]; snprintf(name, sizeof name, "%s/routes-%d.bin", dir, y.L);
            FILE *f = fopen(name, "ab");
            if (f) { fwrite(w.h_selected.data(), 4, (size_t)N * TOP, f); fclose(f); }
        }
    }
    gemm(w.post, N, W, T, 31, w.latent);
    for (int t0 = 0; t0 < N; t0 += w.moe_chunk) {
        const int nc = std::min(w.moe_chunk, N - t0), slots = nc * TOP;
        std::fill(w.h_count.begin(), w.h_count.end(), 0);
        for (int i = 0; i < slots; i++) w.h_count[w.h_selected[(size_t)t0 * TOP + i]]++;
        { int acc = 0; for (int e = 0; e < EXPERTS; e++) { w.h_start[e] = acc; acc += w.h_count[e]; } }
        { std::vector<int> fill(EXPERTS, 0);
          for (int n = 0; n < nc; n++) for (int r = 0; r < TOP; r++) { const int e = w.h_selected[(size_t)(t0 + n) * TOP + r]; const int slot = w.h_start[e] + fill[e]++; w.h_slot_token[slot] = t0 + n; w.h_slot_of[n * TOP + r] = slot; } }
        CHECK(cudaMemcpy(w.d_slot_token, w.h_slot_token.data(), (size_t)slots * 4, cudaMemcpyHostToDevice)); CHECK(cudaMemcpy(w.d_slot_of, w.h_slot_of.data(), (size_t)slots * 4, cudaMemcpyHostToDevice));
        std::vector<int> uniq;
        for (int e = 0; e < EXPERTS; e++) if (w.h_count[e]) uniq.push_back(e);
        if (g_trace) {   // K3GPU_TRACE=file: one line per call and chunk: layer, tokens, then expert:count for every expert chosen
            fprintf(g_trace, "%d %d", y.L, nc);
            for (int e : uniq) fprintf(g_trace, " %d:%d", e, w.h_count[e]);
            fputc('\n', g_trace);
        }
        if (g_route_hook) g_route_hook(y.L, uniq);
        for (size_t gi = 0; gi < uniq.size();) {
            const size_t gj = std::min(uniq.size(), gi + (size_t)pool.slots);
            const std::vector<int> grp(uniq.begin() + gi, uniq.begin() + gj);
            pool_ensure(pool, hc, y.L, fd, grp, w.h_count.data(), readers);
            CHECK(cudaMemcpy(pool.d_eptr, pool.h_eptr.data(), EXPERTS * sizeof(void *), cudaMemcpyHostToDevice));
            std::vector<int4> wide, narrow;
            for (int e : grp)
                for (int i0 = 0; i0 < w.h_count[e]; i0 += 32) {
                    const int n = std::min(32, w.h_count[e] - i0);
                    (n <= 16 ? narrow : wide).push_back(make_int4(e, w.h_start[e] + i0, n, n <= 16 ? 16 : 32));
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
        mix_ranks<<<blocks((size_t)nc * LATENT), 256>>>(w.mixture + (size_t)t0 * LATENT, w.D, w.d_slot_of, w.weights + (size_t)t0 * TOP, nc);
    }
    normalize_rows<<<N, 256>>>(w.lnorm, w.mixture, rec_data(T, 33), LATENT);
    gemm(w.lnorm, N, LATENT, T, 32, w.routed);
    gemm(w.post, N, W, T, 34, w.sg); gemm(w.post, N, W, T, 35, w.su);
    dense_activation<<<blocks((size_t)N * SHARED), 256>>>(w.sg, w.su, (size_t)N * SHARED);
    gemm(w.sg, N, SHARED, T, 36, w.so);
    final_sum<<<blocks((size_t)N * W), 256>>>(res, w.routed, w.so, (size_t)N * W);
}

// layer 0 with state
static void run_layer0_state(Work &w, const Layer &y, KdaState &st, const float *in, float *out)
{
    const int N = w.N; const Trunk &T = y.T;
    normalize_rows<<<N, 256>>>(w.Xn, in, y.input_gains, W);
    kda_attention_state(w, y, st, out);
    const float *sn[1] = { in };
    aggregate_with(w, sn, 1, out, rec_data(T, 38));
    normalize_rows<<<N, 256>>>(w.post, w.Agg, rec_data(T, 5), W);
    gemm(w.post, N, W, T, 26, w.dg); gemm(w.post, N, W, T, 27, w.du);
    dense_activation<<<blocks((size_t)N * DENSE), 256>>>(w.dg, w.du, (size_t)N * DENSE);
    gemm(w.dg, N, DENSE, T, 28, w.dout);
    add_into<<<blocks((size_t)N * W), 256>>>(out, w.dout, (size_t)N * W);
}

// layers 1..92 with state and sparse experts (same order of operations as run_layer in model.cuh)
static void run_layer_state(Work &w, const Layer &y, KdaState *kst, MlaCache *mc, size_t pos0, CardPool &pool, HostCache &hc, int fd, int readers,
    const float *in, const float *const *snaps, int nsnap_in, float *out)
{
    const int N = w.N, L = y.L; const Trunk &T = y.T;
    aggregate_with(w, snaps, nsnap_in, in, rec_data(T, 37));
    normalize_rows<<<N, 256>>>(w.Xn, w.Agg, y.input_gains, W);
    if (y.mla) mla_block_cached(w, y, *mc, pos0, out); else kda_attention_state(w, y, *kst, out);
    if (L % 12 != 0) add_into<<<blocks((size_t)N * W), 256>>>(out, in, (size_t)N * W);
    const float *sn[9]; int ns = nsnap_in;
    for (int i = 0; i < nsnap_in; i++) sn[i] = snaps[i];
    if (L % 12 == 0) sn[ns++] = in;
    aggregate_with(w, sn, ns, out, rec_data(T, 38));
    normalize_rows<<<N, 256>>>(w.post, w.Agg, rec_data(T, 5), W);
    moe_block_sparse(w, y, pool, hc, fd, readers, out);
}
