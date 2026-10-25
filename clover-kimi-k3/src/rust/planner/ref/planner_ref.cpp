// Reference implementation of ../include/planner.h: the engine's planner text (extracted by extract.py from backend-gpu.cu and decode.cuh, unchanged) behind a thin harness.
// It is the original's text with the adaptations listed below, not an independent implementation. Every adaptation is marked "ADAPTED".
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <sys/types.h>
#include "planner.h"

enum { MAX_CARDS = 8, N_LAYERS = 92 };   // verbatim from backend-gpu.cu:31

static sp_ops g_ops;
static sp_config g_cfg;

// ADAPTED (1): ekey uses cfg.experts instead of the compile-time EXPERTS (decode.cuh:201).
static int g_experts_run = 0;
static inline int ekey(int layer, int e) { return layer * g_experts_run + e; }

// ADAPTED (2): EXPERT_RAW is a compile-time enum constant in the original (moe.cuh:5); here it is cfg.expert_raw. The text only uses it in `(off_t)e * EXPERT_RAW`, where a long gives the same value.
static long g_expert_raw = 0;
#define EXPERT_RAW g_expert_raw

static double wall_now() { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }   // same text as decode.cuh:487

// ADAPTED (3): decode.cuh:211 `static bool g_vq` and :500 `static bool g_prefill`, set from cfg in sp_init.
static bool g_vq = false;
static bool g_prefill = false;

// ADAPTED (4): CallSet is the original's text (backend-gpu.cu:56); g_rc is filled in sp_init from the ops table's rt_* accessors, as rc_load_sel does in rust mode (backend-gpu.cu:425-436).
struct CallSet { int layer; std::vector<int> experts; std::vector<int> counts; };
static std::vector<CallSet> g_rc;

// ADAPTED (5): the minimal Handle, CardPool, HostCache of backend-gpu.cu:33-36 and decode.cuh:403/440; only the members the planner text touches.
struct Handle { int L = 0, card = 0, fd = -1; };
static Handle *g_handles[N_LAYERS + 1];

// ADAPTED (6): SlotCacheSel forwards each method the planner text calls to the ops table's sc_* functions on the sc_cache handle `rs` (decode.cuh:326-401 holds the original,
// which forwards to libslotcache.so the same way). The original's null-handle rules are kept for the methods it guards (contains false, worth true, remove null). The ops
// table has no find / touch / add_spare (the planner text does not call them).
struct SlotCacheSel {
    sc_cache *rs = nullptr;
    static const std::vector<int> &prot_vec(const std::unordered_set<int> &prot) { static std::vector<int> v; v.assign(prot.begin(), prot.end()); return v; }   // as decode.cuh:332
    bool contains(int key) const { return rs && g_ops.sc_contains(rs, key); }
    bool worth(int key) const { return !rs || g_ops.sc_worth(rs, key); }
    unsigned char *acquire(int key, const std::unordered_set<int> &prot, float heat0, int *vkey, float *vheat)
    {
        const std::vector<int> &v = prot_vec(prot); return g_ops.sc_acquire(rs, key, v.data(), v.size(), heat0, vkey, vheat);
    }
    unsigned char *try_acquire(int key, int cand_nu, const std::unordered_set<int> &prot, float heat0, int *vkey, float *vheat)
    {
        const std::vector<int> &v = prot_vec(prot); return g_ops.sc_try_acquire(rs, key, cand_nu, v.data(), v.size(), heat0, vkey, vheat);
    }
    unsigned char *remove(int key) { return rs ? g_ops.sc_remove(rs, key) : nullptr; }
    void release(unsigned char *p) { g_ops.sc_release(rs, p); }
};
struct HostCache { SlotCacheSel c; size_t cap_slots = 0; };
struct CardPool { int dev = 0; bool ready = false; SlotCacheSel c; };
static CardPool g_pool[MAX_CARDS];
static HostCache g_host;

// ADAPTED (7): rc_nu_sel, rc_set_cur, slotcache_set_next_use call the ops table (backend-gpu.cu:84-98 and decode.cuh:319 are the originals, which also keep the cpp-mode state).
static int rc_nu_sel(int key) { return g_ops.rt_next_use(g_ops.table, key); }
static void rc_set_cur(long cur) { g_ops.rt_set_current(g_ops.table, cur); }
static void slotcache_set_next_use(int (*f)(int)) { g_ops.sc_set_next_use(f); }

// ADAPTED (8): pf_submit is defined after the extracted text (it needs PfJob); declared here because rc_plan and vq_issue call it.
struct PfJob;
static void pf_submit(const PfJob &j);

// The extracted text, unchanged: decode.cuh 492-496, 506-517, 521-534 (Pending, g_pend, g_pend_keys, g_wb_skipped, g_t_pwait/g_pf_used/g_pf_waited, pend_take, Arrival, g_arr, g_t_vwait,
// g_vq_waited, arr_wait) and backend-gpu.cu 58, 99, 102, 107-112, 267-288, 293-370, 389-399 (g_rc_pos/g_rc_planned/g_pend_cap, g_lookahead/g_rc_ok, g_pf_planned/g_pf_skipped, Rel, PfJob,
// rc_plan, the vq globals, vq_sweep, vq_issue, rc_plan_vq, route_hook). Line numbers are those of the sources the include was made from (see NOTES.md).
#include "planner_extracted.inc"

// pf_submit: PfJob -> cq_job exactly as pf_submit_rust (backend-gpu.cu:202-228) does, then ops.enqueue instead of g_cq_enqueue. The shared_ptrs of arr / rel are held until their
// flags are set, because the contract says the entry's memory stays valid until the queue has set its state (the engine's g_cq_hold).
struct CqHold { std::shared_ptr<Arrival> arr; std::shared_ptr<Rel> rel; };
static std::mutex g_cq_mu;
static std::vector<CqHold> &g_cq_hold = *new std::vector<CqHold>;
static void pf_submit(const PfJob &j)
{
    cq_job c = {};
    c.kind = j.kind; c.dev = j.dev; c.fd = j.fd; c.off = j.off; c.layer = j.layer; c.slot = j.slot; c.dst = j.dst; c.host_src = j.host_src; c.victim_host = j.victim_host;
    c.p_state = j.p ? reinterpret_cast<int *>(&j.p->state) : nullptr;
    c.wb_state = j.wb ? reinterpret_cast<int *>(&j.wb->state) : nullptr;
    c.arr_state = j.arr ? reinterpret_cast<int *>(&j.arr->state) : nullptr;
    c.rel_state = j.rel ? reinterpret_cast<int *>(&j.rel->state) : nullptr;
    if (j.arr || j.rel) {
        std::lock_guard<std::mutex> g(g_cq_mu);
        g_cq_hold.push_back(CqHold{ j.arr, j.rel });
        if ((g_cq_hold.size() & 127) == 0)
            for (size_t i = 0; i < g_cq_hold.size();) {
                const CqHold &h = g_cq_hold[i];
                const bool done = h.arr ? h.arr->state.load(std::memory_order_acquire) == 1 : h.rel->state.load(std::memory_order_acquire) == 1;
                if (done) { g_cq_hold[i] = std::move(g_cq_hold.back()); g_cq_hold.pop_back(); } else i++;
            }
    }
    g_ops.enqueue(&c);
}
static_assert(sizeof(std::atomic<int>) == 4 && std::atomic<int>::is_always_lock_free, "planner.h passes the flags as plain ints");

extern "C" {

// Harness-only addition: a second sp_init resets all planner state (the contract allows one planner per process; the smoke test runs several scenarios in one process).
void sp_init(const sp_ops *ops, const sp_config *cfg)
{
    g_ops = *ops; g_cfg = *cfg;
    g_experts_run = cfg->experts; g_expert_raw = (long)cfg->expert_raw;
    g_rc_pos = 0; g_rc_planned = 0; g_pf_planned = 0; g_pf_skipped = 0;
    g_vq_ram = g_vq_drive = g_vq_wb = g_vq_nowb = 0; g_t_plan = 0; g_t_pwait = g_t_vwait = 0; g_pf_used = g_pf_waited = g_vq_waited = g_wb_skipped = 0;
    g_pend.clear(); g_pend_keys.clear(); g_arr.clear(); g_rel.clear(); g_wb_keys.clear();
    { std::lock_guard<std::mutex> g(g_cq_mu); g_cq_hold.clear(); }
    for (int i = 0; i <= N_LAYERS; i++) { delete g_handles[i]; g_handles[i] = nullptr; }
    for (int i = 0; i < MAX_CARDS; i++) g_pool[i] = CardPool();
    g_host = HostCache();
    g_lookahead = cfg->lookahead; g_pend_cap = cfg->pend_cap; g_vq = cfg->vq != 0; g_vq_cap = cfg->vq_cap; g_vq_from = cfg->vq_from; g_prefill = cfg->prefill != 0; g_rc_ok = cfg->rc_ok != 0;
    // g_rc as rc_load_sel fills it (backend-gpu.cu:425-436)
    g_rc.clear();
    const size_t n = ops->rt_calls(ops->table);
    g_rc.resize(n);
    for (size_t i = 0; i < n; i++) {
        size_t ne = 0;
        g_rc[i].layer = ops->rt_layer(ops->table, i);
        const int *e = ops->rt_experts(ops->table, i, &ne);
        if (ne) g_rc[i].experts.assign(e, e + ne);
    }
}

void sp_layer_set(int layer, int fd, int card)
{
    if (!g_handles[layer]) g_handles[layer] = new Handle;
    Handle *h = g_handles[layer];
    h->L = layer; h->card = card;
    __atomic_store_n(&h->fd, fd, __ATOMIC_RELEASE);
}

void sp_layer_set_fd(int layer, int fd)
{
    Handle *h = g_handles[layer];
    if (h) __atomic_store_n(&h->fd, fd, __ATOMIC_RELEASE);
}

void sp_pool_set(int card, sc_cache *cache, int dev, int ready)
{
    g_pool[card].c.rs = cache; g_pool[card].dev = dev; g_pool[card].ready = ready != 0;
}

void sp_host_set(sc_cache *cache, size_t cap_slots)
{
    g_host.c.rs = cache; g_host.cap_slots = cap_slots;
}

void sp_route_hook(int layer, const int *uniq, size_t n)
{
    std::vector<int> v;
    if (n) v.assign(uniq, uniq + n);
    route_hook(layer, v);
}

void sp_pend_take(int key) { pend_take(key); }
void sp_arr_wait(int key) { arr_wait(key); }

unsigned char *sp_host_acquire(int key, float heat, int *vkey, float *vheat)
{
    return g_host.c.acquire(key, g_pend_keys, heat, vkey, vheat);
}

void sp_get_stats(sp_stats *out)
{
    out->pos = g_rc_pos; out->planned_to = g_rc_planned;
    out->rc_ok = g_rc_ok ? 1 : 0;
    out->planned = g_pf_planned; out->skipped = g_pf_skipped;
    out->vq_ram = g_vq_ram; out->vq_drive = g_vq_drive; out->vq_wb = g_vq_wb; out->vq_nowb = g_vq_nowb;
    out->pf_used = g_pf_used; out->pf_waited = g_pf_waited; out->vq_waited = g_vq_waited;
    out->t_pwait = g_t_pwait; out->t_vwait = g_t_vwait; out->t_plan = g_t_plan;
    out->pending = g_pend.size(); out->arrivals = g_arr.size(); out->releases = g_rel.size(); out->wb_keys = g_wb_keys.size();
}

}  // extern "C"
