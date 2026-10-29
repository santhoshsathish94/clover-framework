// Reference implementation of ../include/poolinit.h: the engine's HostCache::init and CardPool::init (decode.cuh at 4d8d891, lines 403-471, extracted by extract.py, unchanged) behind a thin
// harness. It is the original's text with the adaptations listed below, not an independent implementation. Every adaptation is marked "ADAPTED".
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <thread>
#include <vector>
#include <sys/mman.h>
#include <sys/types.h>
#include "poolinit.h"

static pi_ops g_ops;

// ADAPTED (1): EXPERT_RAW and EXPERTS are compile-time enum constants in the original (moe.cuh:5, 17547264 and 896); here they are the configuration's values. Every use in the text is
// `size_t / EXPERT_RAW`, `(size_t)n * EXPERT_RAW`, `(double)EXPERT_RAW`, an argument of type size_t, or `EXPERTS * sizeof(void *)` / `h_eptr.assign(EXPERTS, ...)` with a non-negative value,
// where a size_t / int variable gives the same value as the enum (for expert_raw < 2^31, which includes the real one).
static size_t g_expert_raw = 0;
static int g_experts = 0;
#define EXPERT_RAW g_expert_raw
#define EXPERTS g_experts

// ADAPTED (2): the CUDA runtime is replaced by the ops table. Each call passes the address of the caller's variable straight through, as the CUDA call would store into it, and returns the
// op's code. The numeric flags are the CUDA header's (see poolinit.h).
typedef int cudaError_t;
enum { cudaSuccess = 0 };
typedef void *cudaStream_t;
typedef void *cudaEvent_t;
#define cudaHostRegisterPortable PI_HOST_REGISTER_PORTABLE
#define cudaHostAllocDefault PI_HOST_ALLOC_DEFAULT
#define cudaStreamNonBlocking PI_STREAM_NON_BLOCKING
#define cudaEventDisableTiming PI_EVENT_DISABLE_TIMING
static const char *cudaGetErrorString(cudaError_t e) { return g_ops.error_string(e); }
static cudaError_t cudaSetDevice(int d) { return g_ops.set_device(d); }
static cudaError_t cudaMemGetInfo(size_t *f, size_t *t) { return g_ops.mem_get_info(f, t); }
template <class T> static cudaError_t cudaMalloc(T **p, size_t n) { return g_ops.device_malloc((void **)(void *)p, n); }
static cudaError_t cudaHostRegister(void *p, size_t n, unsigned flags) { return g_ops.host_register(p, n, flags); }
static cudaError_t cudaHostAlloc(void **p, size_t n, unsigned flags) { return g_ops.host_alloc(p, n, flags); }
static cudaError_t cudaStreamCreateWithFlags(cudaStream_t *s, unsigned flags) { return g_ops.stream_create(s, flags); }
static cudaError_t cudaEventCreateWithFlags(cudaEvent_t *e, unsigned flags) { return g_ops.event_create(e, flags); }

// ADAPTED (3): the CHECK macro of kernels.cuh:17, with __FILE__ replaced by the configured file name (extract.py checks the original macro text and prints it). The included file numbers its
// lines from 1 (a #line before an #include does not carry into the file), so __LINE__ + 402 is the line of the original decode.cuh, where the extracted block starts at line 403.
static const char *g_check_file = "decode.cuh";
#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d: %s\n", g_check_file, __LINE__ + 402, cudaGetErrorString(e_)); exit(1); } } while (0)

// ADAPTED (4): the minimal SlotCacheSel: only add_spare, forwarded to the ops table (decode.cuh:326-401 holds the original, which forwards to libslotcache.so the same way, creating the cache
// on first use; here the cache is the caller's).
struct SlotCacheSel {
    sc_cache *rs = nullptr;
    void add_spare(unsigned char *p) { g_ops.sc_add_spare(rs, p); }
};

// the clock of the original (decode.cuh:202), extracted
#include "poolinit_wall.inc"
// the extracted text: struct HostCache, struct ExpertStats, enum MAX_READERS, struct CardPool (decode.cuh:403-471)
#include "poolinit_extracted.inc"

// ADAPTED (5): the header's mirrors must be the original's constants and layout (a mismatch is a compile error).
static_assert((int)MAX_READERS == PI_MAX_READERS, "MAX_READERS");
static_assert((int)HostCache::CHUNK_SLOTS == PI_CHUNK_SLOTS, "CHUNK_SLOTS");
static_assert(sizeof(ExpertStats) == sizeof(pi_expert_stats), "ExpertStats size");
static_assert(offsetof(ExpertStats, needed) == offsetof(pi_expert_stats, needed) && offsetof(ExpertStats, hit_card) == offsetof(pi_expert_stats, hit_card) &&
              offsetof(ExpertStats, hit_host) == offsetof(pi_expert_stats, hit_host) && offsetof(ExpertStats, from_drive) == offsetof(pi_expert_stats, from_drive) &&
              offsetof(ExpertStats, written_back) == offsetof(pi_expert_stats, written_back) && offsetof(ExpertStats, t_read) == offsetof(pi_expert_stats, t_read) &&
              offsetof(ExpertStats, t_copy) == offsetof(pi_expert_stats, t_copy) && offsetof(ExpertStats, t_ensure) == offsetof(pi_expert_stats, t_ensure), "ExpertStats offsets");

// ADAPTED (6): checks that the original does not have (it would crash): NULL pointers and expert_raw == 0. Same texts as the Rust crate.
static void bad(const char *what)
{
    fprintf(stderr, "poolinit: %s\n", what);
    exit(1);
}
#define NEED(p) do { if (!(p)) bad(#p " is NULL"); } while (0)

// ADAPTED (7): the out struct is filled from the original's object at the end, and, if the process exits in the middle (a failed CHECK, the "cannot map" or "only N slots" exit), by an atexit
// handler, so that a handler that runs at that exit sees what the original object would hold. The object is value-initialised (arrays zero), and only non-null stream / bounce entries below
// `readers` are copied in the exit case. The handler is disarmed before the call returns.
static HostCache *g_live_host = nullptr;
static pi_host_out *g_live_host_out = nullptr;
static CardPool *g_live_pool = nullptr;
static pi_pool_out *g_live_pool_out = nullptr;
static void copy_host(const HostCache *h, pi_host_out *o) { o->cap_slots = h->cap_slots; o->ready = h->ready ? 1 : 0; }
static void copy_pool(const CardPool *p, pi_pool_out *o, bool final_copy)
{
    o->dev = p->dev; o->slots = p->slots; o->readers = p->readers; o->ready = p->ready ? 1 : 0;
    o->base = p->base; o->copy = p->copy; o->written_back = p->written_back; o->d_eptr = p->d_eptr;
    const int n = std::min(std::max(p->readers, 0), (int)MAX_READERS);
    for (int i = 0; i < n; i++) {
        if (final_copy || p->rstream[i]) o->rstream[i] = p->rstream[i];
        if (final_copy || p->bounce[i]) o->bounce[i] = p->bounce[i];
    }
}
static void at_exit_copy()
{
    if (g_live_host && g_live_host_out) copy_host(g_live_host, g_live_host_out);
    if (g_live_pool && g_live_pool_out) copy_pool(g_live_pool, g_live_pool_out, false);
}
static void arm() { atexit(at_exit_copy); }   // registered at every call, so that it runs before any handler the caller registered before this call (LIFO)

extern "C" {

void pi_host_init(const pi_ops *ops, const pi_host_cfg *cfg, pi_host_out *out)
{
    NEED(ops); NEED(cfg); NEED(out);
    NEED(ops->set_device); NEED(ops->host_register); NEED(ops->error_string); NEED(ops->sc_add_spare);
    if (cfg->expert_raw == 0) bad("expert_raw is 0");
    g_ops = *ops; g_expert_raw = cfg->expert_raw; g_check_file = cfg->src_file ? cfg->src_file : "decode.cuh";
    HostCache *h = new HostCache();
    h->c.rs = cfg->cache;
    g_live_host = h; g_live_host_out = out; arm();
    h->init(cfg->bytes);
    copy_host(h, out);
    g_live_host = nullptr; g_live_host_out = nullptr;
    delete h;
}

void pi_pool_init(const pi_ops *ops, const pi_pool_cfg *cfg, pi_pool_out *out)
{
    NEED(ops); NEED(cfg); NEED(out);
    NEED(ops->set_device); NEED(ops->mem_get_info); NEED(ops->device_malloc); NEED(ops->host_alloc); NEED(ops->stream_create); NEED(ops->event_create);
    NEED(ops->error_string); NEED(ops->sc_add_spare);
    if (cfg->expert_raw == 0) bad("expert_raw is 0");
    g_ops = *ops; g_expert_raw = cfg->expert_raw; g_experts = cfg->experts; g_check_file = cfg->src_file ? cfg->src_file : "decode.cuh";
    CardPool *p = new CardPool();
    p->c.rs = cfg->cache;
    g_live_pool = p; g_live_pool_out = out; arm();
    p->init(cfg->device, cfg->reserve_bytes, cfg->nreaders);
    copy_pool(p, out, true);
    g_live_pool = nullptr; g_live_pool_out = nullptr;
    delete p;
}

}  // extern "C"
