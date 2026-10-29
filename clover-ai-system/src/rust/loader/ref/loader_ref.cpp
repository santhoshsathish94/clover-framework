// Reference implementation of ../include/loader.h: the engine's loader text (extracted by extract.py from backend-gpu.cu and kernels.cuh as committed at 4d8d891, byte for byte) behind a thin
// harness. It is the original's text with the adaptations listed below, not an independent implementation. Every adaptation is marked "ADAPTED (n)"; the list is in NOTES.md.
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <sys/types.h>
#include <unistd.h>
#include "loader.h"

// ADAPTED (1): the CUDA runtime names the extracted text uses, as fakes that go to the contract's callbacks. CHECK (extracted from kernels.cuh:17) needs cudaError_t, cudaSuccess and
// cudaGetErrorString; the callbacks do their own failure handling (header), so the fakes always succeed.
typedef int cudaError_t;
enum { cudaSuccess = 0 };
#define cudaHostAllocDefault 0u
static const char *cudaGetErrorString(cudaError_t) { return "unknown error"; }
static ld_callbacks g_cb;
static cudaError_t cudaSetDevice(int card) { g_cb.set_device(g_cb.ctx, card); return cudaSuccess; }
static cudaError_t cudaHostAlloc(void **p, size_t bytes, unsigned) { g_cb.host_alloc(g_cb.ctx, reinterpret_cast<unsigned char **>(p), bytes); return cudaSuccess; }

// Extracted: CHECK (kernels.cuh:17), enum { MAX_CARDS = 8, N_LAYERS = 92 } (backend-gpu.cu:31), struct Loader ... loader_wait (backend-gpu.cu:538-572).
#include "loader_a.inc"

// ADAPTED (2): EXPERT_RAW is a compile-time enum constant in the original (moe.cuh:5, 17547264); here it is cfg.expert_raw. g_prefill is decode.cuh:500, g_slab is decode.cuh:504
// (`static unsigned char *g_slab[93]`): here a pointer to the caller's table (cfg.slab), indexed the same way.
static long g_expert_raw = 0;
#define EXPERT_RAW g_expert_raw
static bool g_prefill = false;
static unsigned char **g_slab = nullptr;

// ADAPTED (3): the Handle of backend-gpu.cu:33-36 reduced to what the extracted text touches. L, card, k_in and fd are written through to the int the caller passed (a NULL pointer keeps
// only a local copy); y.user and user carry the caller's opaque handle to the callbacks; g_handles stores that opaque handle into the caller's table.
struct IntRef {
    int *p = nullptr; int v = -1;
    IntRef &operator=(int x) { v = x; if (p) *p = x; return *this; }
    operator int() const { return v; }
};
struct Layer { void *user = nullptr; };
struct Handle { IntRef L, card, k_in, fd; Layer y; void *user = nullptr; };
struct HandleTable {
    void **t = nullptr;
    struct Slot { void **p; Slot &operator=(Handle *h) { *p = h->user; return *this; } };
    Slot operator[](unsigned i) const { return Slot{ t + i }; }
};
static HandleTable g_handles;

// ADAPTED (4): the engine calls configure() (environment) and card_of(L) (card layout); here the card comes from the request (a thread-local set by ld_open_layer just before the call).
static void configure() {}
static thread_local int t_card = 0;
static int card_of(int) { return t_card; }

// ADAPTED (5): the model/CUDA calls of the job become the contract's callbacks. y.user is the caller's handle.
static void ensure_work(int card) { g_cb.ensure_work(g_cb.ctx, card); }
static void layer_load_trunk(const char *ds, int L, Layer &y) { g_cb.load_trunk(g_cb.ctx, y.user, ds, L); }
static void layer_load_expert_tables(const char *ds, int L, Layer &y) { g_cb.load_expert_tables(g_cb.ctx, y.user, ds, L); }

// ADAPTED (6): the five lines `h->mla = h->y.mla; if (h->mla) { h->mc.cap = positions; cudaMalloc x2 } else kda_state_alloc(h->kda);` of the job are replaced (by extract.py) with one call
// to this function, which is the callback alloc_state.
static void loader_state_alloc(Handle *h, unsigned positions) { g_cb.alloc_state(g_cb.ctx, h->user, positions); }

// Extracted: open_layer (backend-gpu.cu:574-610) with the five-line state group replaced by the call above.
#include "loader_b.inc"

// ---- the C entry points of loader.h
static bool g_inited = false;
static ld_config g_cfg;

// ADAPTED (7): the guards of the header's "DELIBERATE DIFFERENCES" (the original has none: out-of-bounds access, no init).
static void ld_fatal(const char *fmt, ...)
{
    va_list ap; va_start(ap, fmt); vfprintf(stderr, fmt, ap); va_end(ap);
    exit(1);
}

extern "C" {

void ld_init(const ld_config *cfg)
{
    if (!cfg) ld_fatal("loader: cfg is NULL\n");
    if (g_inited) return;
    if (!cfg->cb.set_device) ld_fatal("loader: set_device is NULL\n");
    if (!cfg->cb.ensure_work) ld_fatal("loader: ensure_work is NULL\n");
    if (!cfg->cb.load_trunk) ld_fatal("loader: load_trunk is NULL\n");
    if (!cfg->cb.load_expert_tables) ld_fatal("loader: load_expert_tables is NULL\n");
    if (!cfg->cb.alloc_state) ld_fatal("loader: alloc_state is NULL\n");
    if (!cfg->handles) ld_fatal("loader: handles is NULL\n");
    if (cfg->prefill && !cfg->cb.host_alloc) ld_fatal("loader: host_alloc is NULL\n");
    if (cfg->prefill && !cfg->slab) ld_fatal("loader: slab is NULL\n");
    g_cfg = *cfg; g_cb = cfg->cb;
    g_expert_raw = (long)cfg->expert_raw; g_prefill = cfg->prefill != 0; g_slab = cfg->slab; g_handles.t = cfg->handles;
    g_inited = true;
}

void ld_submit(int card, void (*fn)(void *), void *arg)
{
    if (!g_inited) ld_fatal("loader: ld_submit before ld_init\n");
    if (!fn) ld_fatal("loader: job is NULL\n");
    if (card < 0 || card >= MAX_CARDS) ld_fatal("loader: card %d out of range (0..%d)\n", card, MAX_CARDS - 1);
    loader_submit(card, [fn, arg]() { fn(arg); });
}

void ld_wait(void) { loader_wait(); }

void ld_open_layer(const ld_layer *r)
{
    if (!g_inited) ld_fatal("loader: ld_open_layer before ld_init\n");
    if (!r) ld_fatal("loader: req is NULL\n");
    if (!r->dataset) ld_fatal("loader: dataset is NULL\n");
    if (r->card < 0 || r->card >= MAX_CARDS) ld_fatal("loader: card %d out of range (0..%d)\n", r->card, MAX_CARDS - 1);
    if (g_prefill && r->layer >= 1 && r->layer > (unsigned)N_LAYERS) ld_fatal("loader: layer %u has no slab slot (0..%d)\n", r->layer, N_LAYERS);
    Handle *h = new Handle;   // lives on: the job may run after this returns, and the original's handles live for the process
    h->L.p = r->h_L; h->card.p = r->h_card; h->k_in.p = r->h_k_in; h->fd.p = r->h_fd;
    h->user = r->handle; h->y.user = r->handle;
    t_card = r->card;
    open_layer(r->dataset, r->layer, r->k, r->positions, h);
}

}  // extern "C"
