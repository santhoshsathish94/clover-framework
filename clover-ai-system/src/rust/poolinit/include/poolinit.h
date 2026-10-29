#ifndef POOLINIT_H
#define POOLINIT_H
/* Contract of the supply pools' start-up: the pinned RAM cache (HostCache::init) and one card's expert pool (CardPool::init).
   Behaviour reference: src/code/gpu/decode.cuh as committed at 4d8d891, lines 403-471: struct HostCache (init, CHUNK_SLOTS), struct ExpertStats, enum MAX_READERS, struct CardPool (init),
   and, for the messages, the CHECK macro (kernels.cuh:17). Constants of the real engine: EXPERT_RAW = 17547264, EXPERTS = 896 (moe.cuh:5), CHUNK_SLOTS = 32, MAX_READERS = 32.
   Both the C++ reference (ref/, the original text behind a harness) and the Rust crate implement exactly this header; neither may change it.

   Every CUDA effect goes through the table `pi_ops` (the caller fills it with the real cudaSetDevice, cudaMemGetInfo, ... or with a recording fake), and the slot caches are reached
   through `sc_add_spare` (slotcache.h, or a wrapper). Everything else (mmap, madvise(MADV_HUGEPAGE), the zeroing of the mapped memory, the 16 touching threads) is done by the
   library itself with the real system calls: it is not in the table.

   DELIBERATE PARAMETRISATION: the original uses the compile-time constants EXPERT_RAW and EXPERTS; here they are fields of the configuration (`expert_raw`, `experts`), so that a
   test can run the same code with small values. CHUNK_SLOTS (32) and MAX_READERS (32) stay constants (PI_CHUNK_SLOTS, PI_MAX_READERS). The engine passes 17547264 and 896.

   Return codes of the ops are CUDA error codes: 0 = cudaSuccess, anything else is a failure (the library never interprets the number; it only gives it to error_string).

   Flag values passed to the ops are the numbers of the CUDA headers (checked in /usr/local/cuda/include/driver_types.h): cudaHostAllocDefault 0x00, cudaHostRegisterPortable 0x01,
   cudaStreamNonBlocking 0x01, cudaEventDisableTiming 0x02. A stream is passed to / returned from the ops as a `void *` (cudaStream_t), an event likewise (cudaEvent_t).

   THREADING. pi_host_init and pi_pool_init are called from the main thread, like the originals, and keep no state of their own between calls (no globals). pi_host_init starts 15
   helper threads and also works on the chunks itself (16 workers in all, as the original: always 15 threads, even when there are 0 chunks); the helpers do only mmap, madvise and the zeroing
   of memory, and are all joined before the first op is called, so every op, every sc_add_spare and every write to the output struct happens on the caller's thread. pi_pool_init
   starts no thread.

   pi_host_init(ops, cfg, out) [HostCache::init(bytes)], in this order:
     1. out->cap_slots = cfg->bytes / expert_raw; out->ready = 1.
     2. chunk_bytes = 32 * expert_raw, n_chunks = cap_slots / 32 (size_t arithmetic, wrapping like the original). 16 workers take chunk numbers 0..n_chunks-1 from one shared counter; for
        its chunk a worker calls mmap(NULL, chunk_bytes, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0); if that fails: stderr "host expert cache: cannot map <chunk_bytes> bytes\n",
        exit(1) (from the worker's thread); then madvise(p, chunk_bytes, MADV_HUGEPAGE) (the result is ignored), then memset(p, 0, chunk_bytes); base[chunk number] = p. The helpers are joined.
     3. ops->set_device(0); then, for i = 0 .. n_chunks-1 in order, ops->host_register(base[i], chunk_bytes, 0x01).
     4. for i = n_chunks-1 down to 0, for s = 31 down to 0: ops->sc_add_spare(cfg->cache, base[i] + s * expert_raw).  (so the first slot the cache hands out is chunk 0, slot 0)
     5. out->cap_slots = n_chunks * 32; stderr: "K3GPU: host expert cache of <cap_slots> slots (<cap_slots * (double)expert_raw / 1e9, one decimal> GB) pinned in <seconds, one decimal> s\n"
        where the seconds are wall time (CLOCK_MONOTONIC) from just before the workers start to just before this line is written.
   pi_pool_init(ops, cfg, out) [CardPool::init(device, reserve_bytes, nreaders)], in this order:
     1. out->dev = cfg->device; out->readers = min(cfg->nreaders, 32) (a negative nreaders stays negative: no reader is made).
     2. ops->set_device(dev); ops->mem_get_info(&free, &total); usable = free > reserve_bytes ? free - reserve_bytes : 0; out->slots = (int)(usable / expert_raw) (the cast wraps like C);
        if slots < 64: stderr "card %d: only %d expert slots fit after the reserve\n" (dev, slots), exit(1).
     3. ops->device_malloc(&out->base, slots * expert_raw); for i = slots-1 down to 0: ops->sc_add_spare(cfg->cache, base + i * expert_raw).  (the first slot handed out is slot 0)
     4. ops->stream_create(&out->copy, 0x01); for i in 0 .. readers-1: ops->stream_create(&out->rstream[i], 0x01), then ops->host_alloc(&out->bounce[i], expert_raw, 0x00).
     5. ops->event_create(&out->written_back, 0x02); ops->device_malloc(&out->d_eptr, experts * sizeof(void *)); out->ready = 1.
   The out fields are stored at the moments the original assigns the matching members (the pointers to the ops are the address of the field itself, as the original passes &member), so a
   process that exits in the middle shows exactly what the original object would show to an atexit handler. Fields of rstream / bounce at index >= readers are never written by init.
   The caller zero-initialises the whole out struct once (a static global is). `out->st` (ExpertStats) is NOT touched: in the original it is a member that init does not assign (it is
   zero from construction).
   The original's `h_eptr.assign(EXPERTS, nullptr)` (a host std::vector<const unsigned char *> of `experts` null pointers) is NOT done here: it is the caller's, after pi_pool_init returns.

   FAILURE (as the original; every failure ends the process, nothing is returned): a nonzero op result makes the library write "<src_file>:<line>: <ops->error_string(code)>\n" to
   stderr and call exit(1), where <line> is the line of the failing CHECK in decode.cuh@4d8d891: set_device 428 (host) / 454 (pool), host_register 429, mem_get_info 455, the
   slab device_malloc 459, copy stream 461, reader stream 463, bounce host_alloc 464, event 466, d_eptr device_malloc 467; <src_file> is cfg->src_file (the value of __FILE__ in
   decode.cuh, which depends on how the engine is built, so the caller supplies it), or "decode.cuh" if it is NULL. The text of every message above is byte for byte the original's.
   exit is the C library's exit(1) (atexit handlers run). Additions that the original does not have (it would crash): a NULL ops / cfg / out pointer, a NULL function pointer in `ops`
   among the ones the call uses (pi_host_init: set_device, host_register, error_string, sc_add_spare; pi_pool_init: set_device, mem_get_info, device_malloc, host_alloc, stream_create,
   event_create, error_string, sc_add_spare), or expert_raw == 0: "poolinit: <what>\n" to stderr and exit(1), before anything else is done. Not handled (the original would throw / crash
   too): an n_chunks so large that the table of base pointers cannot be allocated, experts < 0, a failure to start a helper thread (Rust: "poolinit: cannot start a helper thread\n" and abort()).
   `cache` is passed to ops->sc_add_spare unchanged (it must be a valid cache for the op; the library never looks at it). */
#include <stddef.h>
#include "slotcache.h"
#ifdef __cplusplus
extern "C" {
#endif

#define PI_CHUNK_SLOTS 32
#define PI_MAX_READERS 32
#define PI_HOST_ALLOC_DEFAULT 0x00u
#define PI_HOST_REGISTER_PORTABLE 0x01u
#define PI_STREAM_NON_BLOCKING 0x01u
#define PI_EVENT_DISABLE_TIMING 0x02u

typedef struct pi_ops {
    int (*set_device)(int device);                                              /* cudaSetDevice */
    int (*mem_get_info)(size_t *free_bytes, size_t *total_bytes);               /* cudaMemGetInfo */
    int (*device_malloc)(void **ptr, size_t bytes);                             /* cudaMalloc */
    int (*host_register)(void *ptr, size_t bytes, unsigned int flags);          /* cudaHostRegister */
    int (*host_alloc)(void **ptr, size_t bytes, unsigned int flags);            /* cudaHostAlloc */
    int (*stream_create)(void **stream, unsigned int flags);                    /* cudaStreamCreateWithFlags */
    int (*event_create)(void **event, unsigned int flags);                      /* cudaEventCreateWithFlags */
    const char *(*error_string)(int code);                                      /* cudaGetErrorString */
    void (*sc_add_spare)(sc_cache *cache, unsigned char *slot);                 /* sc_add_spare (slotcache.h) */
} pi_ops;

/* struct ExpertStats of decode.cuh (field for field). Part of pi_pool_out; init does not write it. */
typedef struct pi_expert_stats {
    unsigned long needed, hit_card, hit_host, from_drive, written_back;
    double t_read, t_copy, t_ensure;
} pi_expert_stats;

typedef struct pi_host_cfg {
    size_t bytes;              /* the argument of HostCache::init: K3GPU_HOST_GB << 30 */
    size_t expert_raw;         /* EXPERT_RAW (17547264); must be > 0 */
    sc_cache *cache;           /* the host slot cache that receives the spares */
    const char *src_file;      /* __FILE__ of decode.cuh for the CHECK messages; NULL = "decode.cuh" */
} pi_host_cfg;

typedef struct pi_host_out {   /* the members of HostCache that init assigns */
    size_t cap_slots;
    int ready;
} pi_host_out;

typedef struct pi_pool_cfg {
    int device;                /* CardPool::init's `device` */
    size_t reserve_bytes;
    int nreaders;
    size_t expert_raw;         /* EXPERT_RAW (17547264); must be > 0 */
    int experts;               /* EXPERTS (896) */
    sc_cache *cache;           /* the card's slot cache that receives the spares */
    const char *src_file;      /* as in pi_host_cfg */
} pi_pool_cfg;

typedef struct pi_pool_out {   /* the members of CardPool that init assigns, plus `st` which it leaves alone */
    int dev, slots, readers, ready;
    unsigned char *base;
    void *copy;                                /* cudaStream_t */
    void *rstream[PI_MAX_READERS];             /* cudaStream_t, entries 0 .. readers-1 */
    unsigned char *bounce[PI_MAX_READERS];     /* pinned host slots, entries 0 .. readers-1 */
    void *written_back;                        /* cudaEvent_t */
    const unsigned char **d_eptr;              /* device array of `experts` pointers */
    pi_expert_stats st;                        /* not written by init */
} pi_pool_out;

void pi_host_init(const pi_ops *ops, const pi_host_cfg *cfg, pi_host_out *out);
void pi_pool_init(const pi_ops *ops, const pi_pool_cfg *cfg, pi_pool_out *out);

#ifdef __cplusplus
}
#endif
#endif
