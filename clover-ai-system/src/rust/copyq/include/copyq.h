#ifndef COPYQ_H
#define COPYQ_H
/* Contract of the per-card copy queue. Behaviour reference: PfJob (68-72), PfQueue (73) and pf_enqueue() (76-130) in src/code/gpu/backend-gpu.cu, act_gate() and
   Pending/Arrival in src/code/gpu/decode.cuh. Both the C++ reference (ref/) and the Rust crate implement exactly this header; neither may change it.
   Every `*_state` pointer is the address of a lock-free 4-byte atomic int owned by the caller, initially 0 and alive until it reads 1 (the original kept Arrival and Rel
   alive with shared_ptr; here the caller does, and the engine's g_arr / g_rel maps already do); the queue only ever stores 1 into it, with release ordering.
   cq_config is read once by cq_init, so call it after the engine's configuration is final (after K3GPU_CARDS / READERS / PREFILL are read). */
#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct cq_job {
    int kind;                       /* 0: drive -> RAM slot (no GPU). 1: drive -> pinned bounce -> card. 2: RAM slot -> card. */
    int dev;                        /* card for kinds 1 and 2 (0..cards-1); ignored for kind 0 */
    int fd; off_t off;              /* drive source for kinds 0 and 1: read EXPERT_RAW bytes at byte offset `off` with pread */
    int layer;                      /* index into the slab table when cfg.prefill is set (0..92) */
    unsigned char *slot;            /* kind 0: destination RAM slot */
    int *p_state;                   /* kind 0: set to 1 when the slot holds the bytes (also set when cfg.prefill skips the read) */
    unsigned char *dst;             /* kinds 1, 2: destination on card `dev` (device pointer) */
    const unsigned char *host_src;  /* kind 2: source RAM slot */
    unsigned char *victim_host;     /* kinds 1, 2: if not NULL, FIRST copy EXPERT_RAW bytes card dst -> victim_host and synchronise */
    int *wb_state;                  /* set to 1 right after that victim copy completed (only when victim_host != NULL) */
    int *arr_state;                 /* kinds 1, 2: set to 1 when the new bytes are on the card (after the copy was synchronised) */
    int *rel_state;                 /* kinds 1, 2: may be NULL; set to 1 after the copy was synchronised, BEFORE arr_state */
} cq_job;

typedef struct cq_config {
    size_t expert_raw;              /* bytes per expert record: 17547264 */
    int readers;                    /* K3GPU_READERS, default 16: threads of the drive->RAM queue; each card queue gets max(2, readers / max(1, cards)) */
    int cards;                      /* number of cards in use */
    int prefill;                    /* K3GPU_PREFILL: no drive read; source for kinds 1 and 2 is slab[layer] + ((off / expert_raw) & 3) * expert_raw */
    const unsigned char *const *slab; /* ADDRESS OF THE CALLER'S TABLE of 93 pointers (the engine's g_slab), not a copy: entries are read when a job runs, because the engine fills them after start-up. Used only when prefill != 0 */
    const int *act_hold;            /* address of an atomic int; while it is non-zero every kind 1/2 job waits (yield loop) just before its H2D copy */
} cq_config;

/* Must be called once before the first cq_enqueue. Worker threads are started lazily by the first job of a queue and never stopped (as the original). */
void cq_init(const cq_config *cfg);
/* Thread-safe. Queue 0..7 = card cq_job.dev, queue 8 = kind 0 jobs. FIFO per queue, any number of worker threads per queue. A failed pread or CUDA call
   prints a message to stderr and exits the process with status 1 (as the original). */
void cq_enqueue(const cq_job *job);

#ifdef __cplusplus
}
#endif
#endif
