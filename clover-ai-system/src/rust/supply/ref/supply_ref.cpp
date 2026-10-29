// Reference implementation of ../include/supply.h: the engine's pool_ensure (extracted by extract.py from decode.cuh as committed at 4d8d891, unchanged) behind a thin harness.
// It is the original's text with the adaptations listed below, not an independent implementation. Every adaptation is marked "ADAPTED (n)"; ref/NOTES.md lists them.
#include <vector>
#include <unordered_set>
#include <functional>
#include <atomic>
#include <thread>
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <unistd.h>
#include <sys/types.h>
#include "supply.h"

static sy_ops g_ops;
static sy_config g_cfg;
static bool g_inited = false;

// ADAPTED (1): the CUDA names of the original text map onto sy_ops. The calls keep their original spelling, argument order and types; only the bodies differ.
typedef void *cudaStream_t;
typedef void *cudaEvent_t;
enum cudaMemcpyKind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost };
static int cudaDeviceSynchronize() { return g_ops.device_sync(); }
static int cudaStreamSynchronize(cudaStream_t s) { return g_ops.stream_sync(s); }
static int cudaSetDevice(int dev) { return g_ops.set_device(dev); }
static int cudaMemcpyAsync(void *dst, const void *src, size_t n, cudaMemcpyKind kind, cudaStream_t s)
{
    return kind == cudaMemcpyHostToDevice ? g_ops.memcpy_h2d_async(dst, src, n, s) : g_ops.memcpy_d2h_async(dst, src, n, s);
}
static int cudaEventRecord(cudaEvent_t e, cudaStream_t s) { return g_ops.event_record(e, s); }
static int cudaStreamWaitEvent(cudaStream_t s, cudaEvent_t e, unsigned flags) { return g_ops.stream_wait_event(s, e, flags); }

// ADAPTED (2): kernels.cuh:17 is `#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)`.
// Here the status and the text of the argument go to ops.check on every call (a recording fake sees the whole sequence); ops.check must not return on a non-zero status, and if it does we exit(1).
#define CHECK(x) do { int e_ = (x); g_ops.check(e_, #x); if (e_ != 0) exit(1); } while (0)

// ADAPTED (3): ekey uses cfg.experts instead of the compile-time EXPERTS (decode.cuh:201); EXPERT_RAW is a compile-time enum constant in the original (moe.cuh:5), here cfg.expert_raw
// as a long (the text uses it in size_t and off_t expressions; values are identical for expert_raw < 2^31). Build with -fwrapv so int overflow of ekey wraps like the Rust version.
static int g_experts_run = 0;
static inline int ekey(int layer, int e) { return layer * g_experts_run + e; }
static long g_expert_raw = 0;
#define EXPERT_RAW g_expert_raw

// ADAPTED (4): g_vq (decode.cuh:211), g_chunks (:207), g_prefill (:500) are plain globals set from cfg in sy_init; g_slab is the table the caller gave (the original's array g_slab[93] of decode.cuh:504;
// the const is cast away because the text assigns an element to an `unsigned char *`; supply never writes through it).
static bool g_vq = false;
static bool g_prefill = false;
static int g_chunks = 0;
#define g_slab (const_cast<unsigned char *const *>(g_cfg.slab))

// The first part of the extracted text: ExpertStats (decode.cuh:437) and MAX_READERS (:439).
#include "supply_types.inc"
static_assert(sizeof(ExpertStats) == sizeof(sy_stats) && offsetof(ExpertStats, needed) == offsetof(sy_stats, needed) && offsetof(ExpertStats, hit_card) == offsetof(sy_stats, hit_card)
              && offsetof(ExpertStats, hit_host) == offsetof(sy_stats, hit_host) && offsetof(ExpertStats, from_drive) == offsetof(sy_stats, from_drive)
              && offsetof(ExpertStats, written_back) == offsetof(sy_stats, written_back) && offsetof(ExpertStats, t_read) == offsetof(sy_stats, t_read)
              && offsetof(ExpertStats, t_copy) == offsetof(sy_stats, t_copy) && offsetof(ExpertStats, t_ensure) == offsetof(sy_stats, t_ensure), "sy_stats must have the layout of ExpertStats");
static_assert(MAX_READERS == SY_MAX_READERS, "MAX_READERS");

// ADAPTED (5): SlotCacheSel (decode.cuh:326-401) is two small classes here: PoolSel forwards find / touch / acquire to ops.sc_*; HostSel forwards find / worth / remove / release to ops.sc_*
// and its acquire (the original passes g_pend_keys as `prot`) to ops.host_acquire, which is the planner's sp_host_acquire (prot = the planner's pending keys). The null-handle rules of the
// original are kept for the methods it guards (find NULL, touch nothing, worth true, remove NULL); acquire / release pass NULL through (the original creates a cache there).
// As in decode.cuh:332 a prot set becomes an int array for sc_acquire.
struct PoolSel {
    sc_cache *rs = nullptr;
    static const std::vector<int> &prot_vec(const std::unordered_set<int> &prot) { static std::vector<int> v; v.assign(prot.begin(), prot.end()); return v; }   // membership only: order is irrelevant
    unsigned char *find(int key) { return rs ? g_ops.sc_find(rs, key) : nullptr; }
    void touch(int key, float w) { if (rs) g_ops.sc_touch(rs, key, w); }
    unsigned char *acquire(int key, const std::unordered_set<int> &prot, float heat0, int *vkey, float *vheat)
    {
        const std::vector<int> &v = prot_vec(prot); return g_ops.sc_acquire(rs, key, v.data(), v.size(), heat0, vkey, vheat);
    }
};
struct HostSel {
    sc_cache *rs = nullptr;
    unsigned char *find(int key) { return rs ? g_ops.sc_find(rs, key) : nullptr; }
    bool worth(int key) const { return !rs || g_ops.sc_worth(rs, key); }
    unsigned char *remove(int key) { return rs ? g_ops.sc_remove(rs, key) : nullptr; }
    unsigned char *acquire(int key, const std::unordered_set<int> &, float heat0, int *vkey, float *vheat) { return g_ops.host_acquire(key, heat0, vkey, vheat); }
    void release(unsigned char *p) { g_ops.sc_release(rs, p); }
};
// ADAPTED (6): the minimal CardPool (decode.cuh:445) and HostCache (:416) with only the members pool_ensure touches; sy_ensure fills them from the caller's sy_pool / sy_host.
// h_eptr is a plain pointer instead of the std::vector the engine holds (the text only indexes it).
struct CardPool {
    int dev = 0, readers = 0;
    PoolSel c;
    cudaStream_t copy = nullptr;
    cudaStream_t rstream[MAX_READERS];
    unsigned char *bounce[MAX_READERS];
    cudaEvent_t written_back = nullptr;
    const unsigned char **h_eptr = nullptr;
    ExpertStats st;
};
struct HostCache { HostSel c; size_t cap_slots = 0; };
// ADAPTED (7): g_pend_keys is the planner's set of pending keys; here it is an empty set that only gives the text something to pass: HostSel::acquire ignores it and the planner adds its own.
static std::unordered_set<int> g_pend_keys;
// ADAPTED (8): pend_take / arr_wait (decode.cuh:506-517, 521-534) belong to the planner; here they call the ops table.
static void pend_take(int key) { g_ops.pend_take(key); }
static void arr_wait(int key) { g_ops.arr_wait(key); }

// The rest of the extracted text, unchanged: run_parallel (decode.cuh:473-484), wall_now (:487), g_wb_skipped (:495), pool_ensure with its comment (:536-620).
#include "supply_extracted.inc"

extern "C" {

// ADAPTED (9): harness-only: sy_init copies ops and cfg, resets g_wb_skipped (contract), and may be called again (the smoke test runs several scenarios in one process).
void sy_init(const sy_ops *ops, const sy_config *cfg)
{
    g_ops = *ops; g_cfg = *cfg;
    g_experts_run = cfg->experts; g_expert_raw = (long)cfg->expert_raw;
    g_vq = cfg->vq != 0; g_prefill = cfg->prefill != 0; g_chunks = cfg->chunks;
    g_wb_skipped = 0;
    g_inited = true;
}

// ADAPTED (10): the ABI entry. The engine's CardPool / HostCache are rebuilt from the caller's structs for the duration of the call and the counters are copied back at the end.
void sy_ensure(sy_pool *p, sy_host *h, int layer, int fd, const int *experts, size_t n, const int *counts, int readers)
{
    if (!g_inited) { fprintf(stderr, "supply: sy_ensure before sy_init\n"); exit(1); }
    CardPool pool;
    pool.dev = p->dev; pool.readers = p->readers; pool.c.rs = p->cache; pool.copy = p->copy; pool.written_back = p->written_back; pool.h_eptr = p->h_eptr;
    for (int i = 0; i < MAX_READERS; i++) { pool.rstream[i] = p->rstream[i]; pool.bounce[i] = p->bounce[i]; }
    memcpy((void *)&pool.st, &p->st, sizeof pool.st);
    HostCache hc; hc.c.rs = h->cache; hc.cap_slots = h->cap_slots;
    std::vector<int> v; if (n) v.assign(experts, experts + n);
    pool_ensure(pool, hc, layer, fd, v, counts, readers);
    memcpy(&p->st, &pool.st, sizeof pool.st);
}

unsigned long sy_wb_skipped(void) { return g_wb_skipped; }

}  // extern "C"
