/* lw_probe: extra stress for the pending counter / ld_wait paths, written by the independent verifier (NOT part of the crate's driver, which stays untouched).
   usage: lw_probe <lib.so> <rounds> <waiters> <seed> [stack]
   The main thread submits one job (or a burst of jobs to several cards), then calls ld_wait and checks that every job had finished (premature return); W waiter threads call ld_wait in a
   loop all the time. Every 25 rounds the job stream stops for ~30 ms and every waiter must still be making progress (nothing is pending, so ld_wait returns at once): a waiter that is not
   has slept through its wake-up (STUCK; it heals at the next job's notify, so a hang at the very end is only the rare case). A watchdog reports HANG after 40 s without progress. With the
   5th argument `stack` it only runs one job that touches 4 MB of stack on a loader thread.
   exit 0: PROBE OK; 1: premature return / stack failure; 3: hang; 4: stuck waiter seen. */
#define _GNU_SOURCE
#include <alloca.h>
#include <dlfcn.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "loader.h"

static void (*p_init)(const ld_config *);
static void (*p_submit)(int, void (*)(void *), void *);
static void (*p_wait)(void);

static void cb_card(void *c, int card) { (void)c; (void)card; }
static void cb_layer(void *c, void *h, const char *d, int l) { (void)c; (void)h; (void)d; (void)l; }
static void cb_state(void *c, void *h, unsigned p) { (void)c; (void)h; (void)p; }
static void cb_alloc(void *c, unsigned char **d, size_t b) { (void)c; (void)d; (void)b; }

#define MAXJ 4000000
static atomic_char *done;
static atomic_int stop_flag, finished_waiters;
static atomic_long progress, wcnt[64];

static void job(void *a)
{
    long i = (long)a;
    for (volatile int k = 0; k < (int)(i % 7) * 20; k++) {}
    atomic_store(&done[i], 1);
}
static void *waiter(void *a)
{
    const int idx = (int)(long)a - 1;
    unsigned s = (unsigned)(long)a * 2654435761u;
    while (!atomic_load(&stop_flag)) {
        p_wait();
        atomic_fetch_add(&wcnt[idx], 1);
        s = s * 1103515245u + 12345u;
        if ((s >> 16) % 4 == 0) usleep((s >> 8) % 30);
    }
    atomic_fetch_add(&finished_waiters, 1);
    return NULL;
}
static void *watchdog(void *a)
{
    (void)a;
    long last = -1;
    for (int t = 0; t < 40; t++) {
        sleep(1);
        long p = atomic_load(&progress);
        if (p == -2) return NULL;
        if (p != last) { last = p; t = 0; }
    }
    printf("PROBE HANG progress=%ld waiters_finished=%d\n", atomic_load(&progress), atomic_load(&finished_waiters));
    fflush(stdout);
    _exit(3);
}

static atomic_int stack_ok;
static void job_stack(void *a)
{
    (void)a;
    volatile char *p = alloca(4u << 20);   /* a loader thread with an 8 MB stack survives this, one with 2 MB does not */
    for (size_t i = 0; i < (4u << 20); i += 4096) p[i] = 1;
    atomic_store(&stack_ok, 1);
}

int main(int argc, char **argv)
{
    if (argc < 5) return 2;
    void *h = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 2; }
    p_init = dlsym(h, "ld_init"); p_submit = dlsym(h, "ld_submit"); p_wait = dlsym(h, "ld_wait");
    if (argc > 5 && !strcmp(argv[5], "stack")) {
        static unsigned char *slab0[93]; static void *handles0[93];
        ld_config c0; memset(&c0, 0, sizeof c0);
        c0.cb.set_device = cb_card; c0.cb.ensure_work = cb_card; c0.cb.load_trunk = cb_layer; c0.cb.load_expert_tables = cb_layer; c0.cb.alloc_state = cb_state; c0.cb.host_alloc = cb_alloc;
        c0.expert_raw = 4096; c0.slab = slab0; c0.handles = handles0;
        p_init(&c0);
        p_submit(0, job_stack, NULL);
        p_wait();
        printf("PROBE %s stack job (4 MB of stack on a loader thread)\n", atomic_load(&stack_ok) ? "OK" : "FAILED");
        return atomic_load(&stack_ok) ? 0 : 1;
    }
    const long rounds = atol(argv[2]); const int W = atoi(argv[3]); const unsigned seed = (unsigned)atoi(argv[4]);
    if (rounds <= 0 || rounds * 8 > MAXJ) return 2;
    done = calloc((size_t)MAXJ, 1);
    static unsigned char *slab[93]; static void *handles[93];
    ld_config c; memset(&c, 0, sizeof c);
    c.cb.set_device = cb_card; c.cb.ensure_work = cb_card; c.cb.load_trunk = cb_layer; c.cb.load_expert_tables = cb_layer; c.cb.alloc_state = cb_state; c.cb.host_alloc = cb_alloc;
    c.expert_raw = 4096; c.slab = slab; c.handles = handles;
    p_init(&c);
    pthread_t wd, ws[64]; pthread_create(&wd, NULL, watchdog, NULL);
    if (W > 64) return 2;
    for (int i = 0; i < W; i++) pthread_create(&ws[i], NULL, waiter, (void *)(long)(i + 1));
    long next = 0, premature = 0, stuck = 0, checks = 0; unsigned s = seed * 7919u + 1;
    for (long r = 0; r < rounds; r++) {
        s = s * 1103515245u + 12345u;
        /* 70 %: one job on a random card; 30 %: a hot burst of 2..8 tiny jobs on ONE card (the worker is awake and pops the next job at once) */
        const int hot = ((s >> 16) % 10) < 3;
        const int burst = hot ? 2 + (int)((s >> 8) % 7) : 1;
        const int card = (int)((s >> 4) % 8);
        const long first = next;
        for (int b = 0; b < burst; b++) { p_submit(hot ? card : (int)((s >> 4) % 8), job, (void *)next); next++; }
        p_wait();
        for (long i = first; i < next; i++) if (!atomic_load(&done[i])) premature++;
        if (r % 25 == 24) {   /* quiesce: nothing is pending, so every waiter must be spinning through ld_wait; one that is not has slept through the last wake-up */
            long snap[64];
            usleep(2000);
            for (int i = 0; i < W; i++) snap[i] = atomic_load(&wcnt[i]);
            usleep(25000);
            checks++;
            int bad = 0;
            for (int i = 0; i < W; i++) if (atomic_load(&wcnt[i]) == snap[i]) bad = 1;
            if (bad) stuck++;
        }
        atomic_store(&progress, r);
    }
    atomic_store(&stop_flag, 1);
    p_wait();
    /* every waiter must come back: one that went to sleep with the count already 0 never will */
    while (atomic_load(&finished_waiters) < W) usleep(1000);
    for (int i = 0; i < W; i++) pthread_join(ws[i], NULL);
    atomic_store(&progress, -2);
    long wakes = 0; for (int i = 0; i < W; i++) wakes += atomic_load(&wcnt[i]);
    printf("PROBE %s rounds=%ld jobs=%ld premature=%ld quiesce_checks=%ld stuck_waiter_checks=%ld waiter_wakes=%ld\n", premature ? "PREMATURE" : (stuck ? "STUCK" : "OK"), rounds, next, premature, checks, stuck, wakes);
    return premature ? 1 : (stuck ? 4 : 0);
}
