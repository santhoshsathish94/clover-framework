#ifndef PLANNER_H
#define PLANNER_H
/* Contract of the supply planner: the read-ahead and card-queue planning of the K3GPU route cache, with the pending-read and arrival state it shares with pool_ensure.
   Behaviour reference (src/code/gpu/backend-gpu.cu unless noted): rc_plan, vq_sweep, vq_issue, rc_plan_vq, route_hook, and their globals g_rc_pos, g_rc_planned, g_pend_cap,
   g_lookahead, g_rc_ok, g_rel, g_wb_keys, g_vq_cap, g_vq_from, g_t_plan, g_pf_planned, g_pf_skipped, g_vq_ram, g_vq_drive, g_vq_wb, g_vq_nowb; and in src/code/gpu/decode.cuh:
   struct Pending, g_pend, g_pend_keys, struct Arrival, g_arr, pend_take, arr_wait, and their counters g_pf_used, g_pf_waited, g_t_pwait, g_vq_waited, g_t_vwait.
   Both the C++ reference (ref/, the original text of those functions run against a thin harness) and the Rust crate implement exactly this header; neither may change it.

   The planner does not own any CUDA object. It reads the other components through the function table `sp_ops` (the C ABIs of copyq.h, slotcache.h, routetable.h, loaded by the
   caller), so the planner works with whichever library provides them. Slot caches are the caller's sc_cache handles: one per card pool and one for the host (RAM) cache.
   Not thread-safe except where stated: every function is called from the main thread, except that the copy queue's worker threads store 1 into the state ints of the jobs the
   planner gave them (release ordering) and sp_layer_set_fd may be called from a loader thread (the planner reads the fd with acquire ordering).

   Keys: ekey(layer, e) = layer * experts + e in C int arithmetic. A "call" is one entry of the route table (rt_layer, rt_experts), numbered from 0.

   WHAT THE PLANNER DOES (each item is the original's behaviour; names in brackets are the originals):
   sp_route_hook(layer, uniq, n) [route_hook]: rt_set_current(pos); then if rc_ok: if pos >= calls, or call pos has another layer, or its experts list is not element-for-element equal to uniq: write
     "K3GPU route cache: call <pos> (layer <layer>) does not match the cache; look-ahead stopped\n" to stderr, set rc_ok = 0 and call ops.set_next_use(NULL); else time the following with a monotonic clock
     into t_plan: if vq && pos >= vq_from run the card-queue plan [rc_plan_vq(pos, uniq)] else the read-ahead plan [rc_plan(pos)]. Finally pos = pos + 1. (rc_ok == 0 from the start, or after a mismatch, skips
     everything but the rt_set_current call and the increment.)
   Read-ahead plan [rc_plan(pos)]: last = min(calls, pos + lookahead + 1); for i from max(planned, pos) to last-1: take call i; h = handle of its layer; if there is none or its fd < 0: planned = i+1 and go on.
     Otherwise for each expert e of the call, in order: key; skip if key is pending; skip, counting `skipped`, if (the card's pool is ready and holds key) or the host cache holds key; if the pending count >= pend_cap:
     planned = i and return; if !host.worth(key): planned = i and return; otherwise slot = host.acquire(key, prot = all pending keys, heat 1.0f) (a victim, if any, is dropped), create a pending entry
     key -> {state 0, slot}, count `planned`, and enqueue the job {kind 0, fd, off = e * expert_raw, slot, p_state = the pending entry's state}. After the experts of call i: planned = i+1.
   Card-queue plan [rc_plan_vq(pos, uniq)]: run the sweep [vq_sweep] first; prot = { ekey(layer of call pos, e) for e in uniq }; out = number of arrival entries whose state is 0; full[card] = false for all cards;
     last as above; for i from pos to last-1: call i, handle (skip the call if none or fd < 0), pool of the handle's card; for each expert e: key; if the pool is ready and holds key: next expert; if the pool is ready and
     !full[card] and out < vq_cap: if vq_issue(pool, handle, key, e, cand = (i == pos ? pos as int : ops.rt_next_use(key)), prot): out++ and next expert; else full[card] = 1 (and fall through);
     if key is pending or the host cache holds it: next expert; if pending count >= pend_cap or !host.worth(key): next expert; otherwise the same host read-ahead as in the read-ahead plan (acquire with prot = pending keys,
     pending entry, `planned` counted, job kind 0). The card-queue plan never changes planned_to and never counts `skipped`. full[card] is set whenever vq_issue returns false, whatever the reason.
   Sweep [vq_sweep]: for i over the release list, in index order with swap-with-last removal (the element at i is replaced by the last one and the list shrinks, and index i is looked at again): if its state is 1, host.release(its slot)
     and remove it, else i++. Then the same walk over wb_keys: if the key has no pending entry, remove it from wb_keys; else if that pending entry's state is 1: erase the key from the pending-key set, drop the pending entry and
     remove the key from wb_keys; else i++.
   Issue [vq_issue(pool, handle, key, e, cand, prot)]: in_ram = host.contains(key); if in_ram and the key has a pending entry whose state is 0: return false. dst = pool.try_acquire(key, cand, prot, 1.0f, &vk, &vh); if NULL return false.
     Job j = {dev = the pool's dev, dst, fd, off = e * expert_raw, layer = the handle's layer, kind = 1}. If in_ram: pend_take(key) [below]; j.host_src = host.remove(key); j.kind = 2; j.rel = a new release entry
     {state 0, slot = j.host_src} appended to the release list; vq_ram++. Else vq_drive++. If vk >= 0: arr_wait(vk) [below]; then if (!prefill and the host cache has cap_slots > 0 and pending count < cap_slots * 8 / 10 and host.worth(vk)):
     j.victim_host = host.acquire(vk, prot = pending keys, heat vh), j.wb = a new pending entry {state 0, slot = victim_host} stored under key vk (replacing any entry already there), vk added to the pending keys and appended to wb_keys,
     vq_wb++; else vq_nowb++. Finally j.arr = a new arrival entry stored under key in the arrival map (replacing any), enqueue j, return true. The enqueue takes the job's p_state = NULL, wb_state = j.wb's state or NULL,
     arr_state = j.arr's state, rel_state = j.rel's state or NULL, p/ slot/ dst/ host_src/ victim_host as set, dev and layer as set (kind 0 jobs carry dev 0 and layer 0).
   sp_pend_take(key) [pend_take]: no-op if there is no pending entry; if its state is 0: spin (yield) until it is 1, adding the wait time to t_pwait and counting pf_waited; count pf_used, erase the entry and the pending key, free it.
   sp_arr_wait(key) [arr_wait]: no-op if there is no arrival entry; remove it from the map first; if its state is 0: spin until it is 1, adding the wait time to t_vwait and counting vq_waited. The entry's memory stays valid until its state has been
     set by the queue (the job holds it).
   sp_host_acquire(key, heat, *vkey, *vheat): host.acquire(key, prot = all pending keys, heat, vkey, vheat): what pool_ensure does to write a card's victim down into the RAM cache.

   Job fields and `jobs sent` accounting are the caller's: the planner only calls ops.enqueue(&job) once per job, in the order given above. The planner never installs the slot cache's next-use function (the caller
   does, before the first sp_route_hook); it only clears it on a mismatch. Its observable effects are exactly: the calls to ops.sc_*, ops.rt_set_current, ops.rt_next_use, ops.enqueue, ops.sc_set_next_use, the stderr text, the
   state of the sp_stats counters and the state-int stores of its own entries. How often it calls the read-only accessors ops.rt_calls / rt_layer / rt_experts is NOT part of the contract (the original read a vector). The
   handle's fd is read once per call of a plan for the check and for every job of that call (the original re-read it per job; the two differ only if another thread changes it in between). */
#include <stddef.h>
#include <sys/types.h>
#include "copyq.h"
#include "slotcache.h"
#include "routetable.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef struct sp_ops {
    /* slot caches (slotcache.h) */
    void (*sc_set_next_use)(sc_next_use_fn fn);
    int (*sc_contains)(const sc_cache *c, int key);
    int (*sc_worth)(const sc_cache *c, int key);
    unsigned char *(*sc_acquire)(sc_cache *c, int key, const int *prot, size_t nprot, float heat0, int *vkey, float *vheat);
    unsigned char *(*sc_try_acquire)(sc_cache *c, int key, int cand_nu, const int *prot, size_t nprot, float heat0, int *vkey, float *vheat);
    unsigned char *(*sc_remove)(sc_cache *c, int key);
    void (*sc_release)(sc_cache *c, unsigned char *slot);
    /* route table (routetable.h): the live table */
    rt_table *table;
    size_t (*rt_calls)(const rt_table *t);
    int (*rt_layer)(const rt_table *t, size_t i);
    const int *(*rt_experts)(const rt_table *t, size_t i, size_t *n);
    void (*rt_set_current)(rt_table *t, long cur);
    int (*rt_next_use)(rt_table *t, int key);
    /* copy queue (copyq.h): cq_enqueue, or any function with its contract (the tests record the jobs and set their states later) */
    void (*enqueue)(const cq_job *job);
} sp_ops;

typedef struct sp_config {
    int experts;               /* experts per layer (896) */
    size_t expert_raw;         /* bytes per expert record (17547264) */
    int cards;                 /* 1..8 */
    int lookahead;             /* K3GPU_LOOKAHEAD calls (default 92) */
    size_t pend_cap;           /* most pending read-ahead entries (host cap_slots * 6 / 10) */
    int vq;                    /* K3GPU_VQUEUE */
    size_t vq_cap;             /* K3GPU_VQ_INFLIGHT */
    long vq_from;              /* K3GPU_VQ_FROM */
    int prefill;               /* K3GPU_PREFILL */
    int rc_ok;                 /* 1 if the table has at least one call */
} sp_config;

typedef struct sp_stats {
    size_t pos, planned_to;                        /* g_rc_pos, g_rc_planned */
    int rc_ok;
    unsigned long planned, skipped;                /* g_pf_planned, g_pf_skipped */
    unsigned long vq_ram, vq_drive, vq_wb, vq_nowb;
    unsigned long pf_used, pf_waited, vq_waited;
    double t_pwait, t_vwait, t_plan;               /* seconds */
    size_t pending, arrivals, releases, wb_keys;   /* current sizes */
} sp_stats;

/* One planner per process. sp_init copies *ops and *cfg. */
void sp_init(const sp_ops *ops, const sp_config *cfg);
/* The layers' handles: sp_layer_set at open (fd -1), sp_layer_set_fd when the file has been opened (any thread), layer 0..92. */
void sp_layer_set(int layer, int fd, int card);
void sp_layer_set_fd(int layer, int fd);
/* The pools: card 0..7. `cache` and `dev` are fixed once ready = 1. Before that the pool is "not ready". */
void sp_pool_set(int card, sc_cache *cache, int dev, int ready);
/* The host (RAM) cache and its capacity in slots. */
void sp_host_set(sc_cache *cache, size_t cap_slots);

void sp_route_hook(int layer, const int *uniq, size_t n);
void sp_pend_take(int key);
void sp_arr_wait(int key);
unsigned char *sp_host_acquire(int key, float heat, int *vkey, float *vheat);
void sp_get_stats(sp_stats *out);

#ifdef __cplusplus
}
#endif
#endif
