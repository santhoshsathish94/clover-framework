#ifndef SUPPLY_H
#define SUPPLY_H
/* Contract of the card supply: pool_ensure of src/code/gpu/decode.cuh (committed at 4d8d891, lines 536-620, with run_parallel 473-484 and the counter g_wb_skipped 495), the function that
   makes the experts one MoE call needs resident in one card's pool. Both the C++ reference (ref/, the original text of those lines run against a thin harness) and the Rust crate
   implement exactly this header; neither may change it.

   WHAT IT DOES. For the experts of ONE call (or one group of at most `pool.slots` of them) on ONE card, in the order given:
     - hit on the card: the pool's slot cache holds the key; the slot is made most recently used and heated (sc_touch); with the card queue on (cfg.vq) the arrival of that slot is awaited first;
     - else a slot is acquired from the pool's cache (the cache may give up a victim: its key, heat, and its bytes are still in the slot); the expert is "needed";
     - needed experts the RAM (host) cache holds are TAKEN OUT of it (the expert moves up to the card; the RAM slot is held until the copies are done, then released);
     - the other needed experts are read from the drive (pread) in one or more chunks per expert by reader threads, through a pinned bounce slot of the reading thread, and copied to the card;
     - the victims are written back (copied down) into the RAM cache unless the RAM cache has no capacity, prefill mode is on, or the victim is not worth a slot.
   The caller is the engine's MoE block: it builds the group, calls sy_ensure, then reads pool->h_eptr[e] for the experts of the group. The pool cache and the RAM cache are sc_cache handles
   of slotcache.h; the pending-read / arrival state of the planner (planner.h) is reached through function pointers (sp_pend_take, sp_arr_wait, sp_host_acquire of planner.h or any function with
   the same contract).

   ORDER OF EFFECTS (every call below goes through `sy_ops`, except pread and the clock; "check" = ops.check(status, <site text>) right after the CUDA call that returned `status`):
     0. start the clock of st.t_ensure. cfg.vq: stream_sync(NULL) else device_sync(); check.
     1. prot = the DISTINCT keys ekey(layer, e) of the experts (a set: duplicates in `experts` collapse). st.needed += n.
     2. For e in experts, in order, key = ekey(layer, e):
          p = sc_find(pool.cache, key) (the sc_cache is not called when the pointer is NULL: p = NULL).
          p != NULL: [vq: arr_wait(key)]; sc_touch(pool.cache, key, (float)counts[e]); h_eptr[e] = p; st.hit_card++; next expert.
          else p = sc_acquire(pool.cache, key, prot, nprot, (float)counts[e], &vk, &vh); if vk >= 0: [vq: arr_wait(vk)]; remember victim {vk, vh, p}. h_eptr[e] = p; remember need {e, p}.
          A key that appears twice in `experts` is a card hit the second time (the first acquire inserted it): hit_card counts it, sc_touch is called, no second read.
        If nothing is needed: return (the clock still stops: t_ensure is added; nothing else is).
     3. For each need in order: if sc_find(host.cache, key) != NULL: pend_take(key); host slot = sc_remove(host.cache, key); keep the slot in `held`; st.hit_host++.
     4. set_device(pool.dev); check.
     5. For each victim in order: if host.cap_slots == 0 or cfg.prefill: leave the loop (nothing counted); if !worth(host, vk) [a NULL host cache is worth]: g_wb_skipped++, next;
        else h = host_acquire(vk, vheat, &dk, &dh) (the RAM cache's own victim dk/dh is dropped, as in the original), memcpy_d2h_async(h, victim slot, raw, pool.copy); check; st.written_back++.
     6. event_record(pool.written_back, pool.copy); check. (always, even when no victim was written back)
     7. For each need that has a host slot, in order: memcpy_h2d_async(dst, (cfg.prefill ? slab[layer] + (e & 3) * raw : host slot), raw, pool.copy); check.
     8. n_drive = needs without a host slot; pool_readers = min(readers, pool.readers);
        nchunk = cfg.chunks > 0 ? cfg.chunks : (n_drive >= pool_readers || n_drive == 0) ? 1 : min(8, (pool_readers + n_drive - 1) / n_drive);
        chunk = ((raw + nchunk - 1) / nchunk + 4095) / 4096 * 4096   (size_t arithmetic).
        For each need without a host slot, in order: st.from_drive++; for k = 0..nchunk-1: co = k * chunk; stop at the first co >= raw; len = min(chunk, raw - co); one reader job
        {dst + co, offset e * raw + co, e, co, len}. The job list is in that order (expert order, then chunk order).
     9. t0 = now. The jobs run on min(pool_readers, number of jobs) threads (see THREADS); t1 = now. A job, on the thread numbered tid:
          not prefill: b = pool.bounce[tid]; read len bytes with pread(fd, b + got, len - got, offset + got) until all are there; a result <= 0 is fatal:
                       "reading expert %d of layer %d failed: %s\n" (e, layer, strerror(errno) when the result is < 0, "end of file" when 0) on stderr, exit(1).
          prefill:     b = slab[layer] + (e & 3) * raw + co; nothing is read.
          then: set_device(dev); check; stream_wait_event(pool.rstream[tid], pool.written_back, 0); check; memcpy_h2d_async(dst + co, b, len, pool.rstream[tid]); check;
                stream_sync(pool.rstream[tid]); check.
    10. stream_sync(pool.copy); check. Then sc_release(host.cache, h) for each held slot in order. st.t_read += t1 - t0; st.t_copy += now - t1; the clock of t_ensure stops (it also covers every step above).

   The sites of `check` (the text passed as `expr`, as the original's CHECK macro would stringize its argument; the engine's real check prints "<file>:<line>: <error text>\n" and exit(1)):
     "cudaStreamSynchronize(0)"  "cudaDeviceSynchronize()"  "cudaSetDevice(pool.dev)"  "cudaMemcpyAsync(h, v.src, EXPERT_RAW, cudaMemcpyDeviceToHost, pool.copy)"
     "cudaEventRecord(pool.written_back, pool.copy)"  "cudaMemcpyAsync(n.dst, g_prefill ? g_slab[layer] + (size_t)(n.e & 3) * EXPERT_RAW : n.host, EXPERT_RAW, cudaMemcpyHostToDevice, pool.copy)"
     "cudaSetDevice(dev)"  "cudaStreamWaitEvent(pool.rstream[tid], pool.written_back, 0)"  "cudaMemcpyAsync(dev_dst, b, len, cudaMemcpyHostToDevice, pool.rstream[tid])"
     "cudaStreamSynchronize(pool.rstream[tid])"  "cudaStreamSynchronize(pool.copy)"
   ops.check is called after EVERY CUDA call, with status 0 too (so a recording fake sees the full sequence). With a non-zero status it must not return; if it does, sy_ensure exits with status 1.

   cfg.prefill (K3GPU_PREFILL, timing only): the store behind the cards behaves as if every expert were in RAM. No drive is read and no victim is written back; the source of every copy to the card
   is the layer's slab: 4 real expert records, slab[layer] + (e & 3) * raw (+ co for a chunk). The router's choice is replaced by the engine's, so the results are NOT the model's.
   The slab table `cfg.slab` is the ADDRESS of the engine's g_slab (93 pointers); entries are read at each call (the engine fills them after start-up); slab[layer] must be non-NULL when prefill is
   on and the layer is used. `fd` is not used in prefill mode (it may be -1).
   cfg.chunks (K3GPU_CHUNKS, g_chunks): when > 0 it replaces the computed nchunk (a call with few experts cannot fill the drive with whole-expert reads, so the computed value splits each expert
   among the idle readers: up to 8 chunks). Chunks past the end of the record are not issued (co >= raw).
   cfg.vq (g_vq, K3GPU_VQUEUE with policy 2): the cards' pools are also filled ahead of the calls by the planner. Then (a) the first call is stream_sync(NULL) instead of device_sync (copies of other
   streams are not waited for), and (b) entries of the pool that may still be arriving are awaited with arr_wait before they are touched or replaced (hit key, victim key).

   KEYS. ekey(layer, e) = layer * cfg.experts + e in C int arithmetic (wrapping). Keys are >= 0 in practice; see slotcache.h for negative keys. e must be in 0..experts-1 (it indexes h_eptr and counts).
   `counts[e]` (int, indexed by expert number, `experts` entries) is the number of tokens of the call that chose expert e: the heat of the expert.

   THREADS. sy_init and sy_ensure are main-thread-only: sy_ensure calls sc_*, pend_take, arr_wait and host_acquire on the calling thread only (the caches are not thread-safe). The reader jobs run
   on min(min(readers, pool->readers), number of jobs) threads: the CALLING thread is thread 0 and runs jobs too; threads 1.. are created per call and joined before sy_ensure returns. Only
   these ops are called from reader threads: set_device, stream_wait_event, memcpy_h2d_async, stream_sync and check; they MUST be thread-safe (all others are called from the calling thread). A job is told the number
   of its thread, tid < min(readers, pool->readers) <= SY_MAX_READERS, and uses pool->bounce[tid] and pool->rstream[tid] (so a pool needs `readers` bounce slots and streams). Jobs are taken in list
   order by whichever thread is free; with one thread (pool_readers <= 1 or a single job) they run in list order on the calling thread. If pool_readers <= 0 the calling thread still runs every
   job with tid 0 (as the original) and bounce[0] / rstream[0] must exist. A fatal error in a reader thread exits the whole process with status 1 (as the original); which of several failing jobs
   prints first is not defined.
   What the caller must keep valid: ops and cfg are copied by sy_init; the caches (pool->cache, host->cache), the pool and host structs, bounce slots, streams, the event, h_eptr (cfg.experts
   entries), the slab table and `fd` must stay valid and unchanged while sy_ensure runs. The slots handed out by the caches are the caller's memory of raw bytes each.
   A NULL pool->cache is outside the contract. A NULL host->cache is allowed (it behaves as a cache that holds nothing: find is NULL, nothing else is called) only when host->cap_slots == 0 or
   prefill is on (the original lazily creates a cache on acquire / release, which this ABI cannot do).

   WHERE THE RUST VERSION DIFFERS FROM THE ORIGINAL, ON PURPOSE (the C++ reference keeps the original text and gets the same observable behaviour through its harness):
     1. the CUDA calls, check, the cache calls and the planner calls go through sy_ops (the original calls CUDA and its own classes directly). The reference's harness maps the original's names
        (cudaMemcpyAsync, CHECK, ...) onto sy_ops, so the original text is unchanged.
     2. `prot` is passed to sc_acquire as a sorted array of distinct keys (the original: a std::unordered_set turned into an array in hash order). Membership only; the count is the same.
     3. the reader jobs are a list of plain records taken by an atomic counter on scoped threads instead of std::function objects (same order, same thread numbering, same joins).
     4. fatal messages are written with write(2)-level stderr writes; the text is the original's.
     5. sy_ops.check carries the CHECK; it receives the status, not the original's file:line text (decode.cuh's line numbers would be wrong after a port). The engine's wiring prints its own text.
     6. The counter g_wb_skipped lives in the library (sy_wb_skipped); the original kept it as a global of decode.cuh that print_summary reads.
     7. Everything else in the original is kept in the same order: see ORDER OF EFFECTS.

   NOT PART OF THE CONTRACT: how often the clock is read; the order in which concurrent reader jobs make their (thread-safe) ops calls; the value of the times (st.t_* are seconds from a monotonic clock:
   t_ensure >= t_read + t_copy holds, and t_read and t_copy are only added when something was needed). */
#include <stddef.h>
#include <stdint.h>
#include "slotcache.h"
#ifdef __cplusplus
extern "C" {
#endif

#define SY_MAX_READERS 32   /* MAX_READERS of decode.cuh */

typedef struct sy_ops {
    /* CUDA effects: each returns the cudaError_t value (0 = success). Streams and events are the engine's cudaStream_t / cudaEvent_t, opaque here; a stream may be NULL (the default stream). */
    int (*device_sync)(void);                                                /* cudaDeviceSynchronize() */
    int (*stream_sync)(void *stream);                                        /* cudaStreamSynchronize(stream); thread-safe */
    int (*set_device)(int dev);                                              /* cudaSetDevice(dev); thread-safe */
    int (*memcpy_d2h_async)(void *dst, const void *src, size_t n, void *stream);   /* cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToHost, stream) */
    int (*memcpy_h2d_async)(void *dst, const void *src, size_t n, void *stream);   /* cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, stream); thread-safe */
    int (*event_record)(void *event, void *stream);                          /* cudaEventRecord(event, stream) */
    int (*stream_wait_event)(void *stream, void *event, unsigned flags);     /* cudaStreamWaitEvent(stream, event, flags); thread-safe */
    void (*check)(int status, const char *expr);                             /* the engine's CHECK; called with every status, must not return when status != 0; thread-safe */
    /* slot caches (slotcache.h): the pool's cache and the RAM cache */
    unsigned char *(*sc_find)(sc_cache *c, int key);
    void (*sc_touch)(sc_cache *c, int key, float w);
    int (*sc_worth)(const sc_cache *c, int key);
    unsigned char *(*sc_acquire)(sc_cache *c, int key, const int *prot, size_t nprot, float heat0, int *vkey, float *vheat);
    unsigned char *(*sc_remove)(sc_cache *c, int key);
    void (*sc_release)(sc_cache *c, unsigned char *slot);
    /* planner (planner.h) */
    void (*pend_take)(int key);                                              /* sp_pend_take */
    void (*arr_wait)(int key);                                               /* sp_arr_wait */
    unsigned char *(*host_acquire)(int key, float heat, int *vkey, float *vheat);   /* sp_host_acquire: sc_acquire on the RAM cache with prot = the planner's pending keys */
} sy_ops;

typedef struct sy_config {
    int experts;                        /* EXPERTS (896) */
    size_t expert_raw;                  /* EXPERT_RAW: bytes per expert record (17547264) */
    int vq;                             /* g_vq */
    int prefill;                        /* g_prefill */
    int chunks;                         /* g_chunks: > 0 forces the number of chunks per expert */
    const unsigned char *const *slab;   /* ADDRESS of the engine's g_slab table (93 pointers), read at each call; only used when prefill != 0 */
} sy_config;

/* Same fields and order as ExpertStats of decode.cuh (decode.cuh:437). The caller owns one per pool; sy_ensure adds to it. Times are seconds. */
typedef struct sy_stats {
    unsigned long needed, hit_card, hit_host, from_drive, written_back;
    double t_read, t_copy, t_ensure;
} sy_stats;

/* What pool_ensure reads of the engine's CardPool, in the engine's types. All pointers are the caller's. */
typedef struct sy_pool {
    int dev;                                        /* CardPool::dev */
    int readers;                                    /* CardPool::readers: reader slots, <= SY_MAX_READERS */
    sc_cache *cache;                                /* CardPool::c (its slot cache) */
    void *copy;                                     /* CardPool::copy (cudaStream_t): host RAM <-> card traffic of the main thread */
    void *rstream[SY_MAX_READERS];                  /* CardPool::rstream: one stream per reading thread */
    unsigned char *bounce[SY_MAX_READERS];          /* CardPool::bounce: one pinned slot of expert_raw bytes per reading thread */
    void *written_back;                             /* CardPool::written_back (cudaEvent_t) */
    const unsigned char **h_eptr;                   /* CardPool::h_eptr.data(): cfg.experts entries; sy_ensure sets h_eptr[e] for the experts it is given */
    sy_stats st;                                    /* CardPool::st */
} sy_pool;

typedef struct sy_host {
    sc_cache *cache;                                /* HostCache::c (the RAM cache); may be NULL only as described above */
    size_t cap_slots;                               /* HostCache::cap_slots */
} sy_host;

/* One supply per process. Copies *ops and *cfg and sets the g_wb_skipped counter to 0; a second call replaces both and resets the counter. Main thread. */
void sy_init(const sy_ops *ops, const sy_config *cfg);
/* pool_ensure(pool, host, layer, fd, std::vector<int>(experts, experts + n), counts, readers). `experts` are n expert numbers of `layer`, `fd` is the open file of that layer's records
   (offset e * expert_raw), `counts` is indexed by expert number, `readers` the engine's reader count (threads used = min(readers, pool->readers)). Main thread. Not before sy_init (message
   "supply: sy_ensure before sy_init", exit(1)). */
void sy_ensure(sy_pool *pool, sy_host *host, int layer, int fd, const int *experts, size_t n, const int *counts, int readers);
/* g_wb_skipped: card victims that were not worth a RAM slot (policy 2). Any thread. */
unsigned long sy_wb_skipped(void);

#ifdef __cplusplus
}
#endif
#endif
