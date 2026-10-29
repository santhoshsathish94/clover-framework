/* Smoke test of the planner contract against libplanner_ref.so, WITHOUT any real slot cache, route table or copy queue library.
   The test brings hand-written fakes: sc_* functions on a small array cache with scripted answers (spares popped last-in first-out, scripted victims, scripted
   try_acquire outcomes, a "not worth" key), a route table of a few calls, a next-use answer table and an enqueue that records every job. Every expected value was
   derived by hand from the contract text in ../include/planner.h and the original code (derivations in the comments); none was taken from running the library.
   Build: see build.sh. Run: ./smoke_ref
   Conventions: experts per layer 100, so key = layer * 100 + expert; expert_raw 1000, so off = expert * 1000. Slots are made-up addresses 0x1000 etc. (never dereferenced). */
#include "planner.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>

static int checks = 0, fails = 0;
#define CHECK(c) do { checks++; if (!(c)) { fails++; printf("FAIL line %d: %s\n", __LINE__, #c); } } while (0)
#define NONE 1000000000
#define P(x) ((unsigned char *)(uintptr_t)(x))

/* ---------------------------------------------------------------- fake slot cache */
typedef struct { char kind; int key, cand; float heat; int nprot; int prot[12]; } ScCall;   /* kind 'A' acquire, 'T' try_acquire; prot sorted ascending */
struct sc_cache {
    int nheld; int hk[16]; unsigned char *hs[16];
    int nspare; unsigned char *spare[16];                       /* popped from the end */
    int noworth_key;                                            /* sc_worth is 0 for this key (0 = none: no key 0 is used) */
    int tryok[8]; int ntry_script, ntry_used;                   /* outcome of the i-th try_acquire call; beyond the script: fail */
    struct { int key; float heat; unsigned char *slot; } victim[8]; int nvictim, ivictim;   /* used in order when no spare is left */
    ScCall log[24]; int nlog;
    int removed[8]; int nrem;
    unsigned char *released[8]; int nrelease;
};
static sc_cache P0, P1, P2, H;

static int held_idx(const sc_cache *c, int key) { for (int i = 0; i < c->nheld; i++) if (c->hk[i] == key) return i; return -1; }
static void held_put(sc_cache *c, int key, unsigned char *s) { int i = held_idx(c, key); if (i < 0) i = c->nheld++; c->hk[i] = key; c->hs[i] = s; }
static void held_del(sc_cache *c, int i) { for (; i + 1 < c->nheld; i++) { c->hk[i] = c->hk[i + 1]; c->hs[i] = c->hs[i + 1]; } c->nheld--; }
static void log_call(sc_cache *c, char kind, int key, int cand, float heat, const int *prot, size_t nprot)
{
    ScCall *l = &c->log[c->nlog++];
    l->kind = kind; l->key = key; l->cand = cand; l->heat = heat; l->nprot = (int)nprot;
    for (size_t i = 0; i < nprot; i++) { int v = prot[i], j = (int)i; while (j > 0 && l->prot[j - 1] > v) { l->prot[j] = l->prot[j - 1]; j--; } l->prot[j] = v; }
}
static unsigned char *take(sc_cache *c, int key, int *vkey, float *vheat)
{
    unsigned char *s;
    if (c->nspare > 0) { s = c->spare[--c->nspare]; *vkey = -1; }
    else {
        if (c->ivictim >= c->nvictim) { fprintf(stderr, "fake cache: no scripted victim left\n"); exit(2); }
        s = c->victim[c->ivictim].slot; *vkey = c->victim[c->ivictim].key; *vheat = c->victim[c->ivictim].heat; c->ivictim++;
        int i = held_idx(c, *vkey); if (i >= 0) held_del(c, i);
    }
    held_put(c, key, s);
    return s;
}
static int f_contains(const sc_cache *c, int key) { return held_idx(c, key) >= 0; }
static int f_worth(const sc_cache *c, int key) { return key != c->noworth_key; }
static unsigned char *f_acquire(sc_cache *c, int key, const int *prot, size_t nprot, float heat0, int *vkey, float *vheat)
{ log_call(c, 'A', key, 0, heat0, prot, nprot); return take(c, key, vkey, vheat); }
static unsigned char *f_try(sc_cache *c, int key, int cand, const int *prot, size_t nprot, float heat0, int *vkey, float *vheat)
{
    log_call(c, 'T', key, cand, heat0, prot, nprot); *vkey = -1;
    const int ok = c->ntry_used < c->ntry_script ? c->tryok[c->ntry_used] : 0; c->ntry_used++;
    return ok ? take(c, key, vkey, vheat) : NULL;
}
static unsigned char *f_remove(sc_cache *c, int key)
{
    int i = held_idx(c, key); if (i < 0) return NULL;
    unsigned char *s = c->hs[i]; held_del(c, i); c->removed[c->nrem++] = key; return s;
}
static void f_release(sc_cache *c, unsigned char *slot) { c->released[c->nrelease++] = slot; c->spare[c->nspare++] = slot; }
static int nsetnu, setnu_null;
static void f_setnu(sc_next_use_fn fn) { nsetnu++; setnu_null = fn == NULL; }

/* ---------------------------------------------------------------- fake route table */
struct rt_table {
    int n; int layer[8]; int ne[8]; int ex[8][4];
    int nu_key[4], nu_val[4], nnu;       /* next-use answers; any other key: NONE */
    int nuq[16], nnuq;                   /* keys asked */
    long cur[16]; int ncur;              /* rt_set_current values */
};
static rt_table RT;
static void add_call(int layer, int ne, int e0, int e1)
{
    int i = RT.n++; RT.layer[i] = layer; RT.ne[i] = ne; RT.ex[i][0] = e0; RT.ex[i][1] = e1;
}
static void add_call4(int layer, int ne, int e0, int e1, int e2, int e3)
{
    int i = RT.n++; RT.layer[i] = layer; RT.ne[i] = ne; RT.ex[i][0] = e0; RT.ex[i][1] = e1; RT.ex[i][2] = e2; RT.ex[i][3] = e3;
}
static size_t f_calls(const rt_table *t) { return (size_t)t->n; }
static int f_layer(const rt_table *t, size_t i) { return t->layer[i]; }
static const int *f_experts(const rt_table *t, size_t i, size_t *n) { *n = (size_t)t->ne[i]; return t->ne[i] ? t->ex[i] : NULL; }
static void f_setcur(rt_table *t, long cur) { t->cur[t->ncur++] = cur; }
static int f_nu(rt_table *t, int key)
{
    t->nuq[t->nnuq++] = key;
    for (int i = 0; i < t->nnu; i++) if (t->nu_key[i] == key) return t->nu_val[i];
    return NONE;
}

/* ---------------------------------------------------------------- recording enqueue */
static cq_job jobs[64];
static int njobs;
static void f_enqueue(const cq_job *j) { jobs[njobs] = *j; __atomic_store_n(&njobs, njobs + 1, __ATOMIC_RELEASE); }

static void reset_fakes(void)
{
    memset(&P0, 0, sizeof P0); memset(&P1, 0, sizeof P1); memset(&P2, 0, sizeof P2); memset(&H, 0, sizeof H); memset(&RT, 0, sizeof RT);
    memset(jobs, 0, sizeof jobs); njobs = 0; nsetnu = 0; setnu_null = 0;
}
static void start(int experts, size_t pend_cap, int lookahead, int vq, size_t vq_cap, long vq_from, int prefill, int rc_ok)
{
    sp_ops o; memset(&o, 0, sizeof o);
    o.sc_set_next_use = f_setnu; o.sc_contains = f_contains; o.sc_worth = f_worth; o.sc_acquire = f_acquire; o.sc_try_acquire = f_try; o.sc_remove = f_remove; o.sc_release = f_release;
    o.table = &RT; o.rt_calls = f_calls; o.rt_layer = f_layer; o.rt_experts = f_experts; o.rt_set_current = f_setcur; o.rt_next_use = f_nu; o.enqueue = f_enqueue;
    sp_config c; memset(&c, 0, sizeof c);
    c.experts = experts; c.expert_raw = 1000; c.cards = 2; c.lookahead = lookahead; c.pend_cap = pend_cap; c.vq = vq; c.vq_cap = vq_cap; c.vq_from = vq_from; c.prefill = prefill; c.rc_ok = rc_ok;
    sp_init(&o, &c);
}

/* ---------------------------------------------------------------- helpers */
static sp_stats stats(void) { sp_stats s; memset(&s, 0, sizeof s); sp_get_stats(&s); return s; }
static void hook(int layer, int ne, int e0, int e1) { int u[2] = { e0, e1 }; sp_route_hook(layer, u, (size_t)ne); }

static void job_is(int i, int kind, int dev, int fd, long off, int layer, void *slot, void *dst, void *host_src, void *victim, int p, int wb, int arr, int rel, int line)
{
    const cq_job *j = &jobs[i];
    checks++;
    int ok = j->kind == kind && j->dev == dev && j->fd == fd && j->off == off && j->layer == layer && j->slot == slot && j->dst == dst && j->host_src == host_src &&
             j->victim_host == victim && (j->p_state != NULL) == p && (j->wb_state != NULL) == wb && (j->arr_state != NULL) == arr && (j->rel_state != NULL) == rel;
    if (!ok) {
        fails++;
        printf("FAIL line %d: job %d is kind %d dev %d fd %d off %ld layer %d slot %p dst %p host_src %p victim %p p/wb/arr/rel %d%d%d%d\n", line, i, j->kind, j->dev, j->fd, (long)j->off, j->layer,
               (void *)j->slot, (void *)j->dst, (const void *)j->host_src, (void *)j->victim_host, j->p_state != NULL, j->wb_state != NULL, j->arr_state != NULL, j->rel_state != NULL);
    }
}
#define JOB(i, kind, dev, fd, off, layer, slot, dst, host_src, victim, p, wb, arr, rel) job_is(i, kind, dev, fd, off, layer, (void *)(slot), (void *)(dst), (void *)(host_src), (void *)(victim), p, wb, arr, rel, __LINE__)

static void log_is(const sc_cache *c, int idx, char kind, int key, int cand, float heat, int nprot, const int *prot, int line)
{
    checks++;
    int ok = idx < c->nlog && c->log[idx].kind == kind && c->log[idx].key == key && c->log[idx].cand == cand && c->log[idx].heat == heat && c->log[idx].nprot == nprot;
    for (int i = 0; ok && i < nprot; i++) ok = c->log[idx].prot[i] == prot[i];
    if (!ok) { fails++; printf("FAIL line %d: cache log entry %d (kind %c key %d cand %d heat %g nprot %d) differs from the expected %c %d %d %g %d\n", line, idx, idx < c->nlog ? c->log[idx].kind : '?',
                               idx < c->nlog ? c->log[idx].key : -1, idx < c->nlog ? c->log[idx].cand : -1, idx < c->nlog ? c->log[idx].heat : -1.0, idx < c->nlog ? c->log[idx].nprot : -1, kind, key, cand, heat, nprot); }
}
#define ACQ(c, idx, key, heat, ...) do { static const int pr[] = { 0, ##__VA_ARGS__ }; log_is(c, idx, 'A', key, 0, heat, (int)(sizeof pr / sizeof pr[0]) - 1, pr + 1, __LINE__); } while (0)
#define TRY(c, idx, key, cand, ...) do { static const int pr[] = { 0, ##__VA_ARGS__ }; log_is(c, idx, 'T', key, cand, 1.0f, (int)(sizeof pr / sizeof pr[0]) - 1, pr + 1, __LINE__); } while (0)

#define STATS(pos_, planned_to_, rc_ok_, planned_, skipped_, pending_, arrivals_, releases_, wb_keys_) do { sp_stats s_ = stats(); \
    CHECK(s_.pos == (pos_)); CHECK(s_.planned_to == (planned_to_)); CHECK(s_.rc_ok == (rc_ok_)); CHECK(s_.planned == (planned_)); CHECK(s_.skipped == (skipped_)); \
    CHECK(s_.pending == (pending_)); CHECK(s_.arrivals == (arrivals_)); CHECK(s_.releases == (releases_)); CHECK(s_.wb_keys == (wb_keys_)); } while (0)
#define VQSTATS(ram, drive, wb, nowb, used, waited, vwaited) do { sp_stats s_ = stats(); \
    CHECK(s_.vq_ram == (ram)); CHECK(s_.vq_drive == (drive)); CHECK(s_.vq_wb == (wb)); CHECK(s_.vq_nowb == (nowb)); CHECK(s_.pf_used == (used)); CHECK(s_.pf_waited == (waited)); CHECK(s_.vq_waited == (vwaited)); } while (0)

static void set1(int *state) { __atomic_store_n(state, 1, __ATOMIC_RELEASE); }
static int get1(const int *state) { return __atomic_load_n(state, __ATOMIC_ACQUIRE); }

typedef struct { int *p; int ms; } setter_t;
static void *setter(void *a) { setter_t *s = (setter_t *)a; usleep((useconds_t)s->ms * 1000); set1(s->p); return NULL; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

/* stderr capture */
static FILE *capf; static int saved_fd = -1;
static void cap_begin(void) { fflush(stderr); capf = tmpfile(); saved_fd = dup(2); dup2(fileno(capf), 2); }
static void cap_end(char *buf, size_t n)
{
    fflush(stderr); dup2(saved_fd, 2); close(saved_fd);
    lseek(fileno(capf), 0, SEEK_SET);
    ssize_t got = read(fileno(capf), buf, n - 1); buf[got < 0 ? 0 : got] = 0; fclose(capf);
}

/* ---------------------------------------------------------------- scenarios */

/* Common setup of A, B, C. Calls (layer: experts): 0 L1 {3,5}; 1 L2 {3,7}; 2 L1 {5,9}; 3 L3 {1}; 4 L2 {3}; 5 L4 {2}. Handles: L1 fd 11 card 0; L2 fd 12 card 1; L3 none; L4 fd -1 card 0.
   Pool 0 ready and holds key 105; pool 1 NOT ready (its cache holds 203, which must be ignored). Host cache: capacity 10, holds 109 (slot 0x5109), spares popped 0x1000, 0x2000, 0x3000. */
static void setup_readahead(size_t pend_cap)
{
    reset_fakes();
    add_call(1, 2, 3, 5); add_call(2, 2, 3, 7); add_call(1, 2, 5, 9); add_call(3, 1, 1, 0); add_call(2, 1, 3, 0); add_call(4, 1, 2, 0);
    start(100, pend_cap, 10, 0, 256, 0, 0, 1);
    sp_layer_set(1, 11, 0); sp_layer_set(2, 12, 1); sp_layer_set(4, -1, 0);
    held_put(&P0, 105, P(0x7105)); held_put(&P1, 203, P(0x7203));
    sp_pool_set(0, &P0, 4, 1); sp_pool_set(1, &P1, 5, 0);
    held_put(&H, 109, P(0x5109)); H.spare[0] = P(0x3000); H.spare[1] = P(0x2000); H.spare[2] = P(0x1000); H.nspare = 3;
    sp_host_set(&H, 10);
}

static void scenario_A(void)
{
    /* Call 0 (layer 1, {3,5}) matches. rc_plan(0): last = min(6, 0+10+1) = 6; i = 0..5.
       i=0: L1 pool 0 ready. e=3 key 103: not pending, not on card, not in host; pending 0 < cap 3; worth; acquire(103, prot {}, 1.0) pops 0x1000; job kind 0 fd 11 off 3000 slot 0x1000 with p_state. e=5 key 105: on card -> skipped 1.
       i=1: L2 pool 1 NOT ready, so its 203 is ignored: e=3 key 203: planned with prot {103}, slot 0x2000, fd 12 off 3000; e=7 key 207: pending 2 < 3, planned with prot {103,203}, slot 0x3000, fd 12 off 7000. planned_to 2.
       i=2: L1: 105 on card -> skipped 2; 109 in host -> skipped 3. planned_to 3.  i=3: layer 3 has no handle -> planned_to 4.  i=4: L2 {3}: 203 pending -> neither planned nor counted. planned_to 5.
       i=5: layer 4 fd -1 -> planned_to 6.   Result: 3 jobs, planned 3, skipped 3, pending 3, pos 1. */
    setup_readahead(3);
    hook(1, 2, 3, 5);
    CHECK(njobs == 3);
    JOB(0, 0, 0, 11, 3000, 0, P(0x1000), NULL, NULL, NULL, 1, 0, 0, 0);
    JOB(1, 0, 0, 12, 3000, 0, P(0x2000), NULL, NULL, NULL, 1, 0, 0, 0);
    JOB(2, 0, 0, 12, 7000, 0, P(0x3000), NULL, NULL, NULL, 1, 0, 0, 0);
    CHECK(jobs[0].p_state != jobs[1].p_state && jobs[1].p_state != jobs[2].p_state && get1(jobs[0].p_state) == 0);
    CHECK(H.nlog == 3);
    ACQ(&H, 0, 103, 1.0f); ACQ(&H, 1, 203, 1.0f, 103); ACQ(&H, 2, 207, 1.0f, 103, 203);
    CHECK(P0.nlog == 0 && P1.nlog == 0);
    STATS(1, 6, 1, 3, 3, 3, 0, 0, 0);
    CHECK(RT.ncur == 1 && RT.cur[0] == 0 && nsetnu == 0 && RT.nnuq == 0);
    /* sp_host_acquire: host has no spare now; scripted victim key 109 slot 0x5109 heat 0.75. prot = all pending keys {103,203,207}. */
    H.victim[0].key = 109; H.victim[0].heat = 0.75f; H.victim[0].slot = P(0x5109); H.nvictim = 1;
    int vk = 0; float vh = 0;
    CHECK(sp_host_acquire(555, 2.0f, &vk, &vh) == P(0x5109) && vk == 109 && vh == 0.75f);
    CHECK(H.nlog == 4); ACQ(&H, 3, 555, 2.0f, 103, 203, 207);
    /* sp_pend_take of 103 whose state is already 1: no wait; pf_used 1, pending 2. */
    set1(jobs[0].p_state);
    sp_pend_take(103);
    STATS(1, 6, 1, 3, 3, 2, 0, 0, 0);
    VQSTATS(0, 0, 0, 0, 1, 0, 0);
    /* Call 1 (layer 2 {3,7}): planned_to is 6, so the loop starts at max(6, 1) = 6 and plans nothing. */
    hook(2, 2, 3, 7);
    CHECK(njobs == 3);
    STATS(2, 6, 1, 3, 3, 2, 0, 0, 0);
    CHECK(RT.ncur == 2 && RT.cur[1] == 1);
}

static void scenario_B(void)
{
    /* pend_cap 2. Call 0: 103 planned (pending 1), 105 skipped (1); i=1: 203 planned (pending 2, slot 0x2000), 207: pending 2 >= 2 -> planned_to = 1 and return. 2 jobs. */
    setup_readahead(2);
    hook(1, 2, 3, 5);
    CHECK(njobs == 2);
    JOB(0, 0, 0, 11, 3000, 0, P(0x1000), NULL, NULL, NULL, 1, 0, 0, 0);
    JOB(1, 0, 0, 12, 3000, 0, P(0x2000), NULL, NULL, NULL, 1, 0, 0, 0);
    STATS(1, 1, 1, 2, 1, 2, 0, 0, 0);
    /* Call 1 (L2 {3,7}): i starts at max(1, 1) = 1: 203 pending -> next; 207: pending 2 >= 2 -> planned_to stays 1, return. No new job. */
    hook(2, 2, 3, 7);
    CHECK(njobs == 2);
    STATS(2, 1, 1, 2, 1, 2, 0, 0, 0);
    /* pend_take(103) after its read finished: pending 1. Then layer 4 gets its fd (sp_layer_set_fd 14). */
    set1(jobs[0].p_state); sp_pend_take(103);
    sp_layer_set_fd(4, 14);
    /* Call 2 (L1 {5,9}), pos 2: i starts at max(1, 2) = 2 (call 1's expert 207 is never planned: planned < pos). i=2: 105 on card -> skipped 2; 109 in host -> skipped 3. i=3 no handle.
       i=4: 203 pending -> next. i=5: layer 4, fd 14, card 0: e=2 key 402: not pending, not on card 0, not in host; pending 1 < 2; worth; acquire(402, prot {203}) pops 0x3000; job fd 14 off 2000. planned_to 6. */
    hook(1, 2, 5, 9);
    CHECK(njobs == 3);
    JOB(2, 0, 0, 14, 2000, 0, P(0x3000), NULL, NULL, NULL, 1, 0, 0, 0);
    CHECK(H.nlog == 3); ACQ(&H, 2, 402, 1.0f, 203);
    STATS(3, 6, 1, 3, 3, 2, 0, 0, 0);
}

static void scenario_C(void)
{
    /* pend_cap 3 but key 207 is not worth a slot. Call 0: 103 planned, 105 skipped; i=1: 203 planned (pending 2); 207: pending 2 < 3, worth false -> planned_to 1, return. */
    setup_readahead(3);
    H.noworth_key = 207;
    hook(1, 2, 3, 5);
    CHECK(njobs == 2);
    JOB(0, 0, 0, 11, 3000, 0, P(0x1000), NULL, NULL, NULL, 1, 0, 0, 0);
    JOB(1, 0, 0, 12, 3000, 0, P(0x2000), NULL, NULL, NULL, 1, 0, 0, 0);
    STATS(1, 1, 1, 2, 1, 2, 0, 0, 0);
    CHECK(H.nlog == 2);
}

static void mismatch_case(const char *what, int layer, int ne, int e0, int e1, const char *expect)
{
    reset_fakes();
    add_call(1, 2, 3, 5); add_call(2, 2, 3, 7); add_call(1, 2, 5, 9);
    start(100, 3, 10, 0, 256, 0, 0, 1);
    sp_layer_set(1, 11, 0); sp_layer_set(2, 12, 1);
    sp_pool_set(0, &P0, 4, 1); sp_pool_set(1, &P1, 5, 1);
    H.spare[0] = P(0x1000); H.nspare = 1; sp_host_set(&H, 10);
    char buf[256]; cap_begin(); hook(layer, ne, e0, e1); cap_end(buf, sizeof buf);
    checks++; if (strcmp(buf, expect) != 0) { fails++; printf("FAIL %s: stderr was \"%s\"\n", what, buf); }
    CHECK(njobs == 0);
    STATS(1, 0, 0, 0, 0, 0, 0, 0, 0);
    CHECK(nsetnu == 1 && setnu_null == 1);
    CHECK(RT.ncur == 1 && RT.cur[0] == 0);
    /* the next call, which would match: rc_ok is 0, so no plan, no message, no second set_next_use; only rt_set_current(1) and pos 2. */
    cap_begin(); hook(2, 2, 3, 7); cap_end(buf, sizeof buf);
    CHECK(buf[0] == 0 && njobs == 0 && nsetnu == 1 && RT.ncur == 2 && RT.cur[1] == 1);
    STATS(2, 0, 0, 0, 0, 0, 0, 0, 0);
}

static void scenario_D(void)
{
    /* call 0 is layer 1 {3,5}. Each case below is a mismatch at pos 0: the line "K3GPU route cache: call 0 (layer <L>) does not match the cache; look-ahead stopped" is written with the
       layer the hook was called with, rc_ok becomes 0 and set_next_use(NULL) is called once. */
    mismatch_case("different experts", 1, 2, 3, 6, "K3GPU route cache: call 0 (layer 1) does not match the cache; look-ahead stopped\n");
    mismatch_case("other layer", 2, 2, 3, 5, "K3GPU route cache: call 0 (layer 2) does not match the cache; look-ahead stopped\n");
    mismatch_case("shorter list", 1, 1, 3, 0, "K3GPU route cache: call 0 (layer 1) does not match the cache; look-ahead stopped\n");
}

static void scenario_D2(void)
{
    /* reordered experts {5,3} against {3,5}: element for element, so a mismatch. */
    mismatch_case("reordered", 1, 2, 5, 3, "K3GPU route cache: call 0 (layer 1) does not match the cache; look-ahead stopped\n");
    /* rc_ok given as 0 at init: a matching call plans nothing, writes nothing, never calls set_next_use; rt_set_current and pos still move. */
    reset_fakes();
    add_call(1, 2, 3, 5); add_call(2, 2, 3, 7);
    start(100, 3, 10, 0, 256, 0, 0, 0);
    sp_layer_set(1, 11, 0); sp_pool_set(0, &P0, 4, 1); H.spare[0] = P(0x1000); H.nspare = 1; sp_host_set(&H, 10);
    char buf[64]; cap_begin(); hook(1, 2, 3, 5); cap_end(buf, sizeof buf);
    CHECK(buf[0] == 0 && njobs == 0 && nsetnu == 0 && RT.ncur == 1 && RT.cur[0] == 0 && H.nlog == 0);
    STATS(1, 0, 0, 0, 0, 0, 0, 0, 0);
}

static void scenario_E(void)
{
    /* Card-queue plan from call 0 (vq 1, vq_from 0, vq_cap 3, lookahead 5, pend_cap 4, host capacity 10).
       Calls: 0 L1 {3,5}; 1 L2 {3,7}; 2 L1 {5,9}. L1 fd 11 card 0; L2 fd 12 card 1.
       Pool 0 (dev 4, ready): holds 150 (slot 0x7000); spare 0x6000; try_acquire outcomes ok, ok, fail; when no spare the scripted victim is 150, heat 2.5, slot 0x7000.
       Pool 1 (dev 5, ready): spare 0x8000; try_acquire outcomes fail, ok.
       Host: holds 103 (slot 0x5103); spares popped 0x4100, 0x4200, 0x4300, 0x4400. Next-use answers: 203 -> 1, 109 -> 2. */
    reset_fakes();
    add_call(1, 2, 3, 5); add_call(2, 2, 3, 7); add_call(1, 2, 5, 9);
    start(100, 4, 5, 1, 3, 0, 0, 1);
    sp_layer_set(1, 11, 0); sp_layer_set(2, 12, 1);
    held_put(&P0, 150, P(0x7000)); P0.spare[0] = P(0x6000); P0.nspare = 1; P0.tryok[0] = 1; P0.tryok[1] = 1; P0.tryok[2] = 0; P0.ntry_script = 3;
    P0.victim[0].key = 150; P0.victim[0].heat = 2.5f; P0.victim[0].slot = P(0x7000); P0.nvictim = 1;
    P1.spare[0] = P(0x8000); P1.nspare = 1; P1.tryok[0] = 0; P1.tryok[1] = 1; P1.ntry_script = 2;
    sp_pool_set(0, &P0, 4, 1); sp_pool_set(1, &P1, 5, 1);
    held_put(&H, 103, P(0x5103)); H.spare[0] = P(0x4400); H.spare[1] = P(0x4300); H.spare[2] = P(0x4200); H.spare[3] = P(0x4100); H.nspare = 4;
    sp_host_set(&H, 10);
    RT.nu_key[0] = 203; RT.nu_val[0] = 1; RT.nu_key[1] = 109; RT.nu_val[1] = 2; RT.nnu = 2;

    /* Hook 0 (L1 {3,5}), vq plan at pos 0: sweep finds nothing. prot {103,105}. out 0. last = min(3, 6) = 3.
       i=0 (pos): pool 0 ready, dev 4.
         e=3 key 103: not on card; ready, not full, out 0 < 3: vq_issue(cand = pos = 0): in_ram (host holds 103), no pending entry -> go on. try_acquire(103, 0, {103,105}) ok -> spare 0x6000, no victim.
           Job: dev 4 dst 0x6000 fd 11 off 3000 layer 1; in_ram: pend_take(103) no-op; host_src = host.remove(103) = 0x5103; kind 2; rel appended (slot 0x5103); vq_ram 1. No victim: no wb, no nowb.
           arr entry for 103; enqueue; true; out 1.
         e=5 key 105: ready, not full, out 1 < 3: vq_issue(cand 0): not in host. try_acquire(105, 0, ...) ok but no spare -> victim 150 heat 2.5, dst 0x7000. kind 1 (drive), vq_drive 1.
           vk 150 >= 0: arr_wait(150) no-op; not prefill, host cap 10, pending 0 < 8, worth: victim_host = host.acquire(150, prot {} , 2.5) pops 0x4100; wb pending (state 0) under key 150; wb_keys [150]; vq_wb 1.
           Job: dev 4 dst 0x7000 fd 11 off 5000 layer 1 victim_host 0x4100 wb_state, arr_state; no rel. out 2.
       i=1: L2 pool 1 (dev 5). e=3 key 203: not on card; ready, not full, out 2 < 3: vq_issue(cand = next_use(203) = 1): host does not hold 203; try_acquire(203, 1, {103,105}) -> FAIL (script). returns false: full[1] = 1.
           Then: not pending, not in host; pending 1 (the wb of 150) < cap 4; worth: host.acquire(203, prot {150}, 1.0) pops 0x4200; pending 203; planned 1; job kind 0 fd 12 off 3000 slot 0x4200 (dev 0 layer 0).
         e=7 key 207: pool ready but full[1] -> no issue; read-ahead: acquire(207, prot {150,203}) pops 0x4300; job fd 12 off 7000; planned 2.
       i=2: L1 pool 0: e=5 key 105 on card (just acquired) -> next. e=9 key 109: ready, not full[0], out 2 < 3: vq_issue(cand next_use(109) = 2): not in host; try_acquire(109, 2, ...) FAIL -> full[0].
           pending 3 < 4, worth: acquire(109, prot {150,203,207}) pops 0x4400; job fd 11 off 9000 slot 0x4400; planned 3.
       Totals: 5 jobs; planned 3; pending {150,203,207,109} 4; arrivals {103,105} 2; releases 1; wb_keys 1; vq_ram 1 vq_drive 1 vq_wb 1 vq_nowb 0. planned_to untouched (0). pos 1. next_use asked for 203 and 109. */
    hook(1, 2, 3, 5);
    CHECK(njobs == 5);
    JOB(0, 2, 4, 11, 3000, 1, NULL, P(0x6000), P(0x5103), NULL, 0, 0, 1, 1);
    JOB(1, 1, 4, 11, 5000, 1, NULL, P(0x7000), NULL, P(0x4100), 0, 1, 1, 0);
    JOB(2, 0, 0, 12, 3000, 0, P(0x4200), NULL, NULL, NULL, 1, 0, 0, 0);
    JOB(3, 0, 0, 12, 7000, 0, P(0x4300), NULL, NULL, NULL, 1, 0, 0, 0);
    JOB(4, 0, 0, 11, 9000, 0, P(0x4400), NULL, NULL, NULL, 1, 0, 0, 0);
    CHECK(jobs[0].arr_state != jobs[1].arr_state && get1(jobs[0].arr_state) == 0 && get1(jobs[1].wb_state) == 0 && get1(jobs[0].rel_state) == 0);
    CHECK(P0.nlog == 3); TRY(&P0, 0, 103, 0, 103, 105); TRY(&P0, 1, 105, 0, 103, 105); TRY(&P0, 2, 109, 2, 103, 105);
    CHECK(P1.nlog == 1); TRY(&P1, 0, 203, 1, 103, 105);
    CHECK(H.nlog == 4); ACQ(&H, 0, 150, 2.5f); ACQ(&H, 1, 203, 1.0f, 150); ACQ(&H, 2, 207, 1.0f, 150, 203); ACQ(&H, 3, 109, 1.0f, 150, 203, 207);
    CHECK(H.nrem == 1 && H.removed[0] == 103);
    CHECK(RT.nnuq == 2 && RT.nuq[0] == 203 && RT.nuq[1] == 109);
    STATS(1, 0, 1, 3, 0, 4, 2, 1, 1);
    VQSTATS(1, 1, 1, 0, 0, 0, 0);

    /* The queue finishes things: job 0 copy done (rel and arr set), job 1 write-down of 150 done (wb set; its arr 105 stays 0), job 2's read of 203 done. 207 and 109 reads still running.
       Hook 1 (L2 {3,7}), vq plan at pos 1:
       sweep: rel list [rel(0x5103)] state 1 -> host.release(0x5103), list empty. wb_keys [150]: pending 150 exists and state 1 -> erase key and entry; wb_keys empty. pending {203,207,109}.
       prot {203,207}. out = arrivals with state 0 = {105} = 1. last = min(3, 1+6) = 3.
       i=1 (pos): pool 1 (dev 5). e=3 key 203: not on card; ready, not full, out 1 < 3: vq_issue(cand = pos = 1): host holds 203, its pending state is 1 -> go on. try_acquire(203, 1, {203,207}) ok (script) -> spare 0x8000.
           Job kind 2: dev 5 dst 0x8000 fd 12 off 3000 layer 2; pend_take(203): state 1 -> pf_used 1, entry and key gone; host_src = host.remove(203) = 0x4200; rel(0x4200); vq_ram 2; arr entry 203; out 2.
         e=7 key 207: not on card; out 2 < 3: vq_issue(cand 1): host holds 207 and its pending state is 0 -> return false (no try_acquire call): full[1]. Then: pending -> next.
       i=2: L1 pool 0: e=5 on card -> next. e=9 key 109: not on card; ready, not full[0], out 2 < 3: vq_issue(cand next_use(109) = 2): host holds 109, pending state 0 -> false; full[0]. pending -> next.
       Totals: one new job (index 5); pending {207,109} 2; arrivals {103,105,203} 3; releases [0x4200] 1; wb_keys 0; vq_ram 2; pf_used 1; planned unchanged 3. next_use asked again for 109. */
    set1(jobs[0].rel_state); set1(jobs[0].arr_state); set1(jobs[1].wb_state); set1(jobs[2].p_state);
    hook(2, 2, 3, 7);
    CHECK(njobs == 6);
    JOB(5, 2, 5, 12, 3000, 2, NULL, P(0x8000), P(0x4200), NULL, 0, 0, 1, 1);
    CHECK(H.nrelease == 1 && H.released[0] == P(0x5103));
    CHECK(P1.nlog == 2); TRY(&P1, 1, 203, 1, 203, 207);
    CHECK(P0.nlog == 3 && H.nlog == 4 && H.nrem == 2 && H.removed[1] == 203);
    CHECK(RT.nnuq == 3 && RT.nuq[2] == 109);
    STATS(2, 0, 1, 3, 0, 2, 3, 1, 0);
    VQSTATS(2, 1, 1, 0, 1, 0, 0);
    CHECK(RT.ncur == 2 && RT.cur[0] == 0 && RT.cur[1] == 1);

    /* sp_pend_take(207) while its read is still running: spins until the test thread sets the state (about 60 ms), then counts: pf_used 2, pf_waited 1, t_pwait > 0, pending {109}. */
    { setter_t s = { jobs[3].p_state, 60 }; pthread_t th; pthread_create(&th, NULL, setter, &s);
      double t0 = now(); sp_pend_take(207); double dt = now() - t0; pthread_join(th, NULL);
      sp_stats st = stats();
      CHECK(dt >= 0.04 && dt < 5.0); CHECK(st.pf_used == 2); CHECK(st.pf_waited == 1); CHECK(st.t_pwait >= 0.04); CHECK(st.pending == 1); }
    /* sp_arr_wait(103): state already 1: entry removed, no wait. */
    sp_arr_wait(103);
    { sp_stats st = stats(); CHECK(st.arrivals == 2 && st.vq_waited == 0); }
    /* sp_arr_wait(105): arrival state 0 until the thread sets it: entry removed first, then the spin; vq_waited 1, t_vwait > 0. */
    { setter_t s = { jobs[1].arr_state, 60 }; pthread_t th; pthread_create(&th, NULL, setter, &s);
      double t0 = now(); sp_arr_wait(105); double dt = now() - t0; pthread_join(th, NULL);
      sp_stats st = stats();
      CHECK(dt >= 0.04 && dt < 5.0); CHECK(st.vq_waited == 1); CHECK(st.t_vwait >= 0.04); CHECK(st.arrivals == 1); }
    /* unknown keys: no-ops */
    sp_arr_wait(999); sp_pend_take(999);
    { sp_stats st = stats(); CHECK(st.arrivals == 1 && st.pending == 1 && st.pf_used == 2 && st.pf_waited == 1 && st.vq_waited == 1); }
}

static void scenario_F(void)
{
    /* Card-queue plan from call 1 (vq_from 1): hook 0 is the read-ahead plan. vq_cap 3, pend_cap 6, lookahead 5, host capacity 10.
       Calls: 0 L1 {3,5}; 1 L2 {3,7}; 2 L2 {7,9}; 3 L1 {5}; 4 L1 {5}. L1 fd 11 card 0; L2 fd 12 card 1.
       Pool 0 (dev 4, ready): holds 105. Pool 1 (dev 5, ready): holds 250 (slot 0x7250); no spare; try_acquire outcomes ok, ok, ok; scripted victims in order:
         {250, 3.0, 0x7250}, {203, 1.0, 0x7250}, {203, 0.5, 0x7777}.
       Host: spares popped 0x4100, 0x4200, 0x4300, 0x4400, 0x4500, 0x4600; key 250 is "not worth" a slot. */
    reset_fakes();
    add_call(1, 2, 3, 5); add_call(2, 2, 3, 7); add_call(2, 2, 7, 9); add_call(1, 1, 5, 0); add_call(1, 1, 5, 0);
    start(100, 6, 5, 1, 3, 1, 0, 1);
    sp_layer_set(1, 11, 0); sp_layer_set(2, 12, 1);
    held_put(&P0, 105, P(0x7105)); sp_pool_set(0, &P0, 4, 1);
    held_put(&P1, 250, P(0x7250)); P1.tryok[0] = P1.tryok[1] = P1.tryok[2] = 1; P1.ntry_script = 3;
    P1.victim[0].key = 250; P1.victim[0].heat = 3.0f; P1.victim[0].slot = P(0x7250);
    P1.victim[1].key = 203; P1.victim[1].heat = 1.0f; P1.victim[1].slot = P(0x7250);
    P1.victim[2].key = 203; P1.victim[2].heat = 0.5f; P1.victim[2].slot = P(0x7777); P1.nvictim = 3;
    sp_pool_set(1, &P1, 5, 1);
    for (int i = 0; i < 6; i++) H.spare[i] = P(0x4600 - 0x100 * i);
    H.nspare = 6; H.noworth_key = 250; sp_host_set(&H, 10);

    /* Hook 0 at pos 0 < vq_from 1: read-ahead plan. last = min(5, 6) = 5.
       i=0: e=3 key 103 planned (0x4100, fd 11 off 3000); e=5 key 105 on card -> skipped 1. i=1: 203 planned (0x4200, fd 12 off 3000); 207 planned (0x4300, off 7000).
       i=2: 207 pending -> next; 209 planned (0x4400, fd 12 off 9000). i=3: 105 on card -> skipped 2. i=4: 105 -> skipped 3. planned 4, planned_to 5, pending {103,203,207,209}. */
    hook(1, 2, 3, 5);
    CHECK(njobs == 4);
    JOB(0, 0, 0, 11, 3000, 0, P(0x4100), NULL, NULL, NULL, 1, 0, 0, 0);
    JOB(1, 0, 0, 12, 3000, 0, P(0x4200), NULL, NULL, NULL, 1, 0, 0, 0);
    JOB(2, 0, 0, 12, 7000, 0, P(0x4300), NULL, NULL, NULL, 1, 0, 0, 0);
    JOB(3, 0, 0, 12, 9000, 0, P(0x4400), NULL, NULL, NULL, 1, 0, 0, 0);
    STATS(1, 5, 1, 4, 3, 4, 0, 0, 0);
    CHECK(P1.nlog == 0 && RT.nnuq == 0);

    /* Reads of 203 and 209 done; 207 still running. Hook 1 (L2 {3,7}) at pos 1 >= vq_from: vq plan. sweep: nothing. prot {203,207}. out 0. last = min(5, 7) = 5.
       i=1 (pos): pool 1. e=3 key 203: not on card; vq_issue(cand = pos = 1): host holds 203, pending state 1 -> ok. try_acquire(203, 1, {203,207}) ok, no spare -> victim 250, heat 3.0, dst 0x7250.
           Job kind 2: dev 5 dst 0x7250 fd 12 off 3000 layer 2; pend_take(203) (pf_used 1); host_src = host.remove(203) = 0x4200; rel(0x4200); vq_ram 1.
           vk 250: arr_wait no-op; pending now {103,207,209} = 3 < 8 but host.worth(250) is false -> vq_nowb 1; no victim_host, no wb. arr entry 203. out 1.
         e=7 key 207: not on card; out 1 < 3: vq_issue: host holds 207, pending state 0 -> false; full[1]. pending -> next.
       i=2: L2 {7,9}, pool 1 full: 207 and 209 are pending -> next (no issue is tried: full[1]). i=3, i=4: 105 on card -> next.
       One new job (index 4). */
    set1(jobs[1].p_state); set1(jobs[3].p_state);
    hook(2, 2, 3, 7);
    CHECK(njobs == 5);
    JOB(4, 2, 5, 12, 3000, 2, NULL, P(0x7250), P(0x4200), NULL, 0, 0, 1, 1);
    CHECK(P1.nlog == 1); TRY(&P1, 0, 203, 1, 203, 207);
    CHECK(RT.nnuq == 0);
    STATS(2, 5, 1, 4, 3, 3, 1, 1, 0);
    VQSTATS(1, 0, 0, 1, 1, 0, 0);

    /* Job 4's rel is set (rel list entry released at the next sweep), its arrival 203 stays 0. 207's read is now done.
       Hook 2 (L2 {7,9}) at pos 2: sweep: rel [rel(0x4200)] state 1 -> release(0x4200) (host spare stack top). prot {207,209}. out = 1 (arrival 203). last = 5.
       i=2 (pos): pool 1. e=7 key 207: not on card; vq_issue(cand 2): host holds 207, pending state 1 -> ok. try_acquire(207, 2, {207,209}) ok -> victim 203 (heat 1.0), dst 0x7250.
           Job kind 2: dev 5 dst 0x7250 fd 12 off 7000 layer 2; pend_take(207) (pf_used 2); host_src = host.remove(207) = 0x4300; rel(0x4300); vq_ram 2.
           vk 203: arr_wait(203): the entry exists with state 0 -> spins until the test thread sets job 4's arr state (60 ms); entry removed; vq_waited 1.
           pending {103,209} = 2 < 8, worth(203): victim_host = host.acquire(203, prot {103,209}, 1.0) pops 0x4200 (just released); wb pending under 203; wb_keys [203]; vq_wb 1. arr 207. out 2.
         e=9 key 209: not on card; out 2 < 3: vq_issue(cand 2): host holds 209, pending state 1 -> ok. try_acquire(209, 2, {207,209}) ok -> victim 203 again (heat 0.5), dst 0x7777.
           Job kind 2: dev 5 dst 0x7777 off 9000; pend_take(209) (pf_used 3); host_src = 0x4400; rel(0x4400); vq_ram 3. vk 203: arr_wait(203): no entry now -> no-op.
           pending {103,203} = 2 < 8, worth: victim_host = host.acquire(203, prot {103,203}, 0.5) pops 0x4500; the new wb pending REPLACES the entry under 203 (the first one is leaked);
           wb_keys [203,203]; vq_wb 2. arr 209. out 3.
       i=3, i=4: 105 on card. Jobs 5 and 6. */
    set1(jobs[4].rel_state); set1(jobs[2].p_state);
    { setter_t s = { jobs[4].arr_state, 60 }; pthread_t th; pthread_create(&th, NULL, setter, &s);
      double t0 = now(); hook(2, 2, 7, 9); double dt = now() - t0; pthread_join(th, NULL);
      CHECK(dt >= 0.04 && dt < 5.0); }
    CHECK(njobs == 7);
    JOB(5, 2, 5, 12, 7000, 2, NULL, P(0x7250), P(0x4300), P(0x4200), 0, 1, 1, 1);
    JOB(6, 2, 5, 12, 9000, 2, NULL, P(0x7777), P(0x4400), P(0x4500), 0, 1, 1, 1);
    CHECK(jobs[5].wb_state != jobs[6].wb_state);
    CHECK(H.nrelease == 1 && H.released[0] == P(0x4200));
    CHECK(P1.nlog == 3); TRY(&P1, 1, 207, 2, 207, 209); TRY(&P1, 2, 209, 2, 207, 209);
    CHECK(H.nlog == 6 && H.nrem == 3 && H.removed[1] == 207 && H.removed[2] == 209);
    ACQ(&H, 4, 203, 1.0f, 103, 209); ACQ(&H, 5, 203, 0.5f, 103, 203);
    STATS(3, 5, 1, 4, 3, 2, 2, 2, 2);
    VQSTATS(3, 0, 2, 1, 3, 0, 1);
    { sp_stats st = stats(); CHECK(st.t_vwait >= 0.04); }

    /* Hook 3 (L1 {5}) at pos 3. States set: job 5 rel done, job 6 rel not; both wb writes done.
       sweep: rel list [rel(0x4300), rel(0x4400)]: i=0 state 1 -> release(0x4300), element 0 replaced by the last (0x4400), list shrinks; index 0 again: state 0 -> i=1, end. List [0x4400].
         wb_keys [203,203]: i=0: pending 203 (the second wb) state 1 -> erase key and entry, element 0 replaced by the last, list [203]; index 0 again: no pending under 203 -> removed. List empty. pending {103}.
       prot {105}; the plan finds 105 on card for calls 3 and 4. No job. */
    set1(jobs[5].rel_state); set1(jobs[5].wb_state); set1(jobs[6].wb_state);
    hook(1, 1, 5, 0);
    CHECK(njobs == 7);
    CHECK(H.nrelease == 2 && H.released[1] == P(0x4300));
    STATS(4, 5, 1, 4, 3, 1, 2, 1, 0);
    /* Hook 4 (L1 {5}): job 6's rel done -> release(0x4400), list empty. */
    set1(jobs[6].rel_state);
    hook(1, 1, 5, 0);
    CHECK(H.nrelease == 3 && H.released[2] == P(0x4400));
    STATS(5, 5, 1, 4, 3, 1, 2, 0, 0);
    CHECK(RT.ncur == 5 && RT.cur[4] == 4);
    /* Hook 5: pos 5 >= calls 5: mismatch message for call 5, rc_ok 0, set_next_use(NULL). */
    char buf[256]; cap_begin(); hook(1, 1, 5, 0); cap_end(buf, sizeof buf);
    checks++; if (strcmp(buf, "K3GPU route cache: call 5 (layer 1) does not match the cache; look-ahead stopped\n") != 0) { fails++; printf("FAIL: stderr was \"%s\"\n", buf); }
    CHECK(nsetnu == 1 && setnu_null == 1 && RT.ncur == 6 && RT.cur[5] == 5);
    STATS(6, 5, 0, 4, 3, 1, 2, 0, 0);
}

/* Scenario G: the host write-down decision of vq_issue, calls that are skipped, a pool that is not ready. Four variants of the same plan.
   Calls: 0 L1 {3,5,7,9}; 1 L2 {3}; 2 L3 {1} (no handle); 3 L4 {2} (fd -1); 4 L5 {4}. vq 1, vq_cap 5, lookahead 4, pend_cap 10, vq_from 0.
   L1 fd 11 card 0; L2 fd 12 card 1; L4 fd -1 card 0; L5 fd 15 card 2. Pool 0 (dev 4, ready): try_acquire always fails. Pool 1 (dev 5, ready): try_acquire ok (no spare) with victim key 250 heat 1.0 slot 0x7250.
   Pool 2 (dev 6) is NOT ready but holds 504. Host: spares popped 0x4100 .. 0x4600, capacity `cap`.
   Hook 0 (pos 0, uniq {3,5,7,9}): prot {103,105,107,109}; out 0; last = min(5, 0+4+1) = 5.
   i=0: e=3 key 103: vq_issue(cand 0): not in host; try_acquire fails -> full[0]. Read-ahead: acquire(103, prot {}) 0x4100 (job fd 11 off 3000). e=5, 7, 9: pool 0 is full -> read-ahead: 0x4200 / 0x4300 / 0x4400 (off 5000 / 7000 / 9000). pending 4.
   i=1: L2 e=3 key 203: pool 1 ready, not full, out 0: vq_issue(cand = next_use(203) = 1000000000): not in host; try_acquire ok -> victim 250 (heat 1.0), dst 0x7250. Job kind 1 dev 5 dst 0x7250 fd 12 off 3000 layer 2; vq_drive 1.
     vk 250: arr_wait no-op. Write-down iff (not prefill) and cap > 0 and pending (4) < cap * 8 / 10 and worth(250): cap 6 -> 4 < 4 false; prefill -> no; cap 0 -> no; cap 10 -> 4 < 8 yes: victim_host = acquire(250, prot {103,105,107,109}, 1.0) pops 0x4500, wb pending (wb_keys [250]), vq_wb 1; else vq_nowb 1.
   i=2: layer 3 has no handle -> skipped. i=3: layer 4 fd -1 -> skipped.
   i=4: L5 card 2: pool NOT ready, so its 504 is ignored by the `ready &&` tests: not pending, not in host; pending < 10; worth: acquire(504, ...) pops the next spare (0x4500, or 0x4600 after a write-down); job kind 0 fd 15 off 4000. planned 5.
   6 jobs; arrivals {203}; planned_to untouched (0); next_use asked only for 203. */
static void g_case(int prefill, size_t cap, int wb)
{
    reset_fakes();
    add_call4(1, 4, 3, 5, 7, 9); add_call(2, 1, 3, 0); add_call(3, 1, 1, 0); add_call(4, 1, 2, 0); add_call(5, 1, 4, 0);
    start(100, 10, 4, 1, 5, 0, prefill, 1);
    sp_layer_set(1, 11, 0); sp_layer_set(2, 12, 1); sp_layer_set(4, -1, 0); sp_layer_set(5, 15, 2);
    P0.ntry_script = 0;
    P1.tryok[0] = 1; P1.ntry_script = 1; P1.victim[0].key = 250; P1.victim[0].heat = 1.0f; P1.victim[0].slot = P(0x7250); P1.nvictim = 1;
    held_put(&P2, 504, P(0x7504));
    sp_pool_set(0, &P0, 4, 1); sp_pool_set(1, &P1, 5, 1); sp_pool_set(2, &P2, 6, 0);
    for (int i = 0; i < 6; i++) H.spare[i] = P(0x4600 - 0x100 * i);
    H.nspare = 6; sp_host_set(&H, cap);
    { int u[4] = { 3, 5, 7, 9 }; sp_route_hook(1, u, 4); }
    CHECK(njobs == 6);
    JOB(0, 0, 0, 11, 3000, 0, P(0x4100), NULL, NULL, NULL, 1, 0, 0, 0);
    JOB(1, 0, 0, 11, 5000, 0, P(0x4200), NULL, NULL, NULL, 1, 0, 0, 0);
    JOB(2, 0, 0, 11, 7000, 0, P(0x4300), NULL, NULL, NULL, 1, 0, 0, 0);
    JOB(3, 0, 0, 11, 9000, 0, P(0x4400), NULL, NULL, NULL, 1, 0, 0, 0);
    if (wb) {
        JOB(4, 1, 5, 12, 3000, 2, NULL, P(0x7250), NULL, P(0x4500), 0, 1, 1, 0);
        JOB(5, 0, 0, 15, 4000, 0, P(0x4600), NULL, NULL, NULL, 1, 0, 0, 0);
        CHECK(H.nlog == 6); ACQ(&H, 4, 250, 1.0f, 103, 105, 107, 109); ACQ(&H, 5, 504, 1.0f, 103, 105, 107, 109, 250);
    } else {
        JOB(4, 1, 5, 12, 3000, 2, NULL, P(0x7250), NULL, NULL, 0, 0, 1, 0);
        JOB(5, 0, 0, 15, 4000, 0, P(0x4500), NULL, NULL, NULL, 1, 0, 0, 0);
        CHECK(H.nlog == 5); ACQ(&H, 4, 504, 1.0f, 103, 105, 107, 109);
    }
    CHECK(P0.nlog == 1); TRY(&P0, 0, 103, 0, 103, 105, 107, 109);
    CHECK(P1.nlog == 1); TRY(&P1, 0, 203, NONE, 103, 105, 107, 109);
    CHECK(RT.nnuq == 1 && RT.nuq[0] == 203);
    STATS(1, 0, 1, 5, 0, wb ? 6 : 5, 1, 0, wb ? 1 : 0);
    VQSTATS(0, 1, wb ? 1 : 0, wb ? 0 : 1, 0, 0, 0);
}

/* Scenario H: the in-flight cap. Calls: 0 L1 {3,5}; 1 L1 {7}; 2 L1 {9}. vq 1, vq_cap 1, lookahead 0 (last = pos + 1: only the current call), pend_cap 10. L1 fd 11 card 0.
   Pool 0 (dev 4, ready): spares popped 0x6100, 0x6200, 0x6300; try_acquire always ok. Host spares popped 0x4100, 0x4200, 0x4300.
   Hook 0: out 0. e=3 key 103: vq_issue(cand 0): not in host; try ok -> spare 0x6100; kind 1 dev 4 dst 0x6100 fd 11 off 3000 layer 1; arrival 103; out 1. e=5: out 1 < 1 false -> no issue; read-ahead: acquire(105) 0x4100, job kind 0 fd 11 off 5000.
   Hook 1 (call 1, L1 {7}): out = arrivals with state 0 = {103} = 1 -> no issue for 107; read-ahead acquire(107, prot {105}) 0x4200, job fd 11 off 7000.
   Then arrival 103 is set: Hook 2 (call 2, L1 {9}): out 0 -> vq_issue(109, cand pos = 2, prot {109}): try ok -> spare 0x6200; kind 1 dev 4 dst 0x6200 fd 11 off 9000. */
static void scenario_H(void)
{
    reset_fakes();
    add_call(1, 2, 3, 5); add_call(1, 1, 7, 0); add_call(1, 1, 9, 0);
    start(100, 10, 0, 1, 1, 0, 0, 1);
    sp_layer_set(1, 11, 0);
    P0.spare[0] = P(0x6300); P0.spare[1] = P(0x6200); P0.spare[2] = P(0x6100); P0.nspare = 3; P0.tryok[0] = P0.tryok[1] = P0.tryok[2] = 1; P0.ntry_script = 3;
    sp_pool_set(0, &P0, 4, 1);
    H.spare[0] = P(0x4300); H.spare[1] = P(0x4200); H.spare[2] = P(0x4100); H.nspare = 3; sp_host_set(&H, 10);
    hook(1, 2, 3, 5);
    CHECK(njobs == 2);
    JOB(0, 1, 4, 11, 3000, 1, NULL, P(0x6100), NULL, NULL, 0, 0, 1, 0);
    JOB(1, 0, 0, 11, 5000, 0, P(0x4100), NULL, NULL, NULL, 1, 0, 0, 0);
    CHECK(P0.nlog == 1); TRY(&P0, 0, 103, 0, 103, 105);
    STATS(1, 0, 1, 1, 0, 1, 1, 0, 0);
    hook(1, 1, 7, 0);
    CHECK(njobs == 3);
    JOB(2, 0, 0, 11, 7000, 0, P(0x4200), NULL, NULL, NULL, 1, 0, 0, 0);
    CHECK(P0.nlog == 1);
    STATS(2, 0, 1, 2, 0, 2, 1, 0, 0);
    set1(jobs[0].arr_state);
    hook(1, 1, 9, 0);
    CHECK(njobs == 4);
    JOB(3, 1, 4, 11, 9000, 1, NULL, P(0x6200), NULL, NULL, 0, 0, 1, 0);
    CHECK(P0.nlog == 2); TRY(&P0, 1, 109, 2, 109);
    STATS(3, 0, 1, 2, 0, 2, 2, 0, 0);
    VQSTATS(0, 2, 0, 0, 0, 0, 0);
    CHECK(RT.nnuq == 0);
}

int main(void)
{
    scenario_A();
    scenario_B();
    scenario_C();
    scenario_D();
    scenario_D2();
    scenario_E();
    scenario_F();
    g_case(0, 6, 0);    /* cap 6: pending 4 is not < 6 * 8 / 10 = 4: no write-down */
    g_case(1, 10, 0);   /* prefill: no write-down */
    g_case(0, 0, 0);    /* host capacity 0: no write-down */
    g_case(0, 10, 1);   /* write-down */
    scenario_H();
    printf("%s (%d checks, %d failed)\n", fails ? "FAIL" : "PASS", checks, fails);
    return fails ? 1 : 0;
}
