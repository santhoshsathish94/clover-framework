#ifndef LOADER_H
#define LOADER_H
/* Contract of the per-card layer loader. Behaviour reference: `struct Loader`, `g_loader[]`, `pending_mu()`, `pending_cv()`, `g_pending`, `loader_submit()`, `loader_wait()` and `open_layer()` in
   src/code/gpu/backend-gpu.cu, read as committed at 4d8d891 (lines 538-572 and 574-610 of that blob); `g_slab` is decode.cuh:504, `g_prefill` decode.cuh:500, `CHECK` kernels.cuh:17.
   Both the C++ reference (ref/, the original text behind a harness) and the Rust crate implement exactly this header; neither may change it.

   WHAT THE LOADER OWNS
   - One worker thread per card (cards 0..7), created by the first job submitted to that card (not before), detached and never stopped or destroyed. As its first action the thread calls
     cb.set_device(card), before it waits for its first job. Jobs of one card run on that card's thread, one at a time, in the order they were queued (FIFO). Jobs of different cards run
     concurrently.
   - A counter of jobs that are submitted and not yet finished. It is raised before the job is queued, lowered after the job returned. ld_wait() returns when the counter is 0; a job
     submitted by another thread (or by a running job) while ld_wait() waits counts: ld_wait() returns only at a moment when the counter is 0.
   - ld_open_layer(): the host-side part of open_layer(): the bookkeeping of the handle (done at once, on the calling thread) and a job (run on the card's thread) that calls the caller's callbacks
     in the original's order and then opens the layer's experts file and, with prefill, reads the 4-record slab. Everything CUDA or model code stays with the caller as callbacks.

   THE JOB OF ld_open_layer, in this order (original: the lambda passed to loader_submit):
     1. cb.set_device(card)                            [CHECK(cudaSetDevice(h->card))]
     2. cb.ensure_work(card)                           [ensure_work(h->card)]
     3. cb.load_trunk(handle, dataset, layer)          [layer_load_trunk(ds.c_str(), L, h->y)]
     4. layer >= 1: cb.load_expert_tables(handle, dataset, layer)    [layer_load_expert_tables(...)]
     5. cb.alloc_state(handle, positions)              [h->mla = h->y.mla; if (h->mla) { h->mc.cap = positions; two cudaMalloc } else kda_state_alloc(h->kda)]
     6. layer >= 1: path = "<dataset>/expert-<layer>/experts.direct" (layer as unsigned decimal; the text is cut to 4095 bytes, as snprintf into char[4096] does).
        *h_fd = open(path, O_RDONLY | O_DIRECT); if (*h_fd < 0) *h_fd = open(path, O_RDONLY) (any failure of the first open, whatever errno); if (*h_fd < 0): write
        "cannot open <path>\n" to stderr (path as cut above), exit(1).
     7. layer >= 1 and prefill: cb.host_alloc(&slab[layer], 4 * expert_raw)    [CHECK(cudaHostAlloc((void **)&g_slab[L], 4 * EXPERT_RAW, cudaHostAllocDefault))]
        then for r = 0..3: got = 0; while got < expert_raw: rd = pread(fd, slab[layer] + r * expert_raw + got, expert_raw - got, r * expert_raw + got); if rd <= 0 (end of file OR error, errno
        is not looked at, EINTR is not retried): write "prefill: cannot read a record of layer <layer>\n" to stderr (layer as unsigned decimal), exit(1); else got += rd.
   Layer 0 has no experts file: steps 6 and 7 are skipped and *h_fd is not written. The file descriptor is never closed.

   BOOKKEEPING of ld_open_layer, on the calling thread, before the job is queued (original, in this order): *h_L = (int)layer; *h_card = card; *h_k_in = (int)k; if layer <= 92:
   handles[layer] = handle. A NULL h_* pointer means "do not write that field" (the job then keeps its own copy of the value it needs).

   THREADING
   - ld_submit, ld_open_layer and ld_wait may be called from any thread, also from inside a running job (a job may submit more jobs, to its own card or another; the new job runs after
     the current one returned if it is for the same card). ld_wait called from inside a job deadlocks (the job itself is counted), as in the original; it is not checked.
   - Callbacks run on loader threads, with no loader lock held; callbacks of different cards may run at the same time and must be thread-safe against each other.
   - Memory written by a job (the h_fd int, slab entries and the bytes read into them, whatever the callbacks wrote) is visible to a thread that returned from ld_wait() after the job finished
     (the counter's mutex orders them).
   - exit(1) is the C library's exit(), called on the loader thread: atexit handlers run and stdio is flushed, while other threads keep running (as in the original).
   - The caller keeps valid for the life of the process: the ld_callbacks ctx, the slab and handles tables, the *h_* ints and the handle of every layer opened. `dataset` is copied during the call.
   - Not guaranteed: any ordering between jobs of different cards; ld_wait observing a job as submitted before the submitting call has entered ld_submit/ld_open_layer.

   DELIBERATE DIFFERENCES FROM THE ORIGINAL (all outside what the original defines or fixes a data race):
   - The original creates g_loader[card] without a lock: two threads making the FIRST submit to the same card at the same moment could create two queues and two threads for that card (FIFO and
     the one-thread-per-card rule broken). Here the first submit to a card is thread-safe: exactly one thread per card.
   - card outside 0..7 (original: out-of-bounds array access): "loader: card <n> out of range (0..7)\n" to stderr, exit(1). layer > 92 with prefill and layer >= 1 (original: out-of-bounds
     g_slab write): "loader: layer <n> has no slab slot (0..92)\n", exit(1). Both are checked at the call, in this order, before any bookkeeping is written and before anything is queued. A call before ld_init (original: none, it has no init): "loader: <function> before ld_init\n", exit(1); ld_wait before
     ld_init returns at once. A NULL cfg / req / job, or a required NULL in ld_init (set_device, ensure_work, load_trunk, load_expert_tables, alloc_state, handles; with prefill also host_alloc
     and slab): "loader: <what> is NULL\n", exit(1). A second ld_init is ignored (first configuration kept), also an invalid one. Failure to create a thread: "loader: cannot start the thread of card <n>: <reason>\n", exit(1)
     (the original's std::thread constructor would throw and terminate).
   - expert_raw is configuration (the original's EXPERT_RAW is the compile-time constant 17547264, which is what the engine passes); any value is accepted, 0 means "no bytes" in both.

   ABI: C, Linux x86-64 / aarch64 (O_DIRECT differs per architecture and is taken from <fcntl.h> in the reference; the Rust crate has its own constant for each, checked by a unit test). off_t is 64 bits. */
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct ld_callbacks {
    void *ctx;                      /* first argument of every callback */
    /* A failing CUDA call is the callback's business: print the message and exit(1) inside it (the original's CHECK). */
    void (*set_device)(void *ctx, int card);                                              /* CHECK(cudaSetDevice(card)); called at thread start and at the start of every ld_open_layer job */
    void (*ensure_work)(void *ctx, int card);                                             /* ensure_work(card) */
    void (*load_trunk)(void *ctx, void *handle, const char *dataset, int layer);          /* layer_load_trunk(dataset, layer, handle->y); the string is valid for the call only */
    void (*load_expert_tables)(void *ctx, void *handle, const char *dataset, int layer);  /* layer_load_expert_tables(...); layer >= 1 only */
    void (*alloc_state)(void *ctx, void *handle, unsigned positions);                     /* the state group of step 5 */
    void (*host_alloc)(void *ctx, unsigned char **dst, size_t bytes);                     /* CHECK(cudaHostAlloc((void **)dst, bytes, cudaHostAllocDefault)); must store the pointer into *dst; only with prefill, layer >= 1 */
} ld_callbacks;

typedef struct ld_config {
    ld_callbacks cb;
    size_t expert_raw;              /* bytes per expert record: 17547264 */
    int prefill;                    /* K3GPU_PREFILL */
    unsigned char **slab;           /* ADDRESS OF THE CALLER'S TABLE of 93 pointers (the engine's g_slab); the loader stores into slab[layer] through cb.host_alloc and reads it back. Used only with prefill */
    void **handles;                 /* ADDRESS OF THE CALLER'S TABLE of 93 pointers (the engine's g_handles, N_LAYERS + 1 entries) */
} ld_config;

typedef struct ld_layer {
    void *handle;                   /* the caller's Handle*, passed to the callbacks and stored into handles[layer] */
    const char *dataset;            /* NUL-terminated; copied during the call */
    unsigned layer;                 /* L */
    unsigned k;                     /* k_in */
    unsigned positions;             /* passed to cb.alloc_state */
    int card;                       /* card_of(layer), computed by the caller (0..7) */
    int *h_L, *h_card, *h_k_in;     /* addresses of the Handle's L, card, k_in (written at once), see BOOKKEEPING */
    int *h_fd;                      /* address of the Handle's fd (written by the job, layer >= 1 only) */
} ld_layer;

/* Once, before any other call but ld_wait. The configuration is copied. */
void ld_init(const ld_config *cfg);
/* loader_submit: queue fn(arg) on the card's thread. Returns after queueing; never waits for the job. */
void ld_submit(int card, void (*fn)(void *arg), void *arg);
/* loader_wait: block until no submitted job is unfinished. */
void ld_wait(void);
/* open_layer's host-side part: bookkeeping now, the job as listed above queued on card `req->card`. Returns at once. */
void ld_open_layer(const ld_layer *req);

#ifdef __cplusplus
}
#endif
#endif
