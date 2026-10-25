/* Smoke test for libplanner.so (Rust): the planner against fake caches, a fake route table and a recording queue, written in C.
   build: gcc -O1 -I include -I ../copyq/include -I ../slotcache/include -I ../routetable/include -o smoke_rust smoke_rust.c -L target/release -lplanner -lpthread -Wl,-rpath,$PWD/target/release
   (on the box the three sibling include directories are copyq-include, slotcache-include, routetable-include next to include)
   run:   ./smoke_rust */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "planner.h"

static int failures = 0;
static void check(int ok, const char *what) { printf("%s: %s\n", ok ? "PASS" : "FAIL", what); if (!ok) failures++; }

/* fake slot caches: a list of held keys, slots handed out from one array */
struct sc_cache { int held[64]; int n; };
static struct sc_cache host, pool0;
static unsigned char slots[64][16];
static int nslot, nacq, ntry, last_cand = -7, nu_unset;
static size_t last_nprot;

static int f_contains(const sc_cache *c, int key) { for (int i = 0; i < c->n; i++) if (c->held[i] == key) return 1; return 0; }
static int f_worth(const sc_cache *c, int key) { (void)c; (void)key; return 1; }
static unsigned char *grant(sc_cache *c, int key, int *vkey) { *vkey = -1; c->held[c->n++] = key; return slots[nslot++]; }
static unsigned char *f_acquire(sc_cache *c, int key, const int *prot, size_t nprot, float heat, int *vkey, float *vheat)
{ (void)prot; (void)heat; (void)vheat; nacq++; last_nprot = nprot; return grant(c, key, vkey); }
static unsigned char *f_try(sc_cache *c, int key, int cand, const int *prot, size_t nprot, float heat, int *vkey, float *vheat)
{ (void)prot; (void)heat; (void)vheat; ntry++; last_cand = cand; last_nprot = nprot; return grant(c, key, vkey); }
static unsigned char *f_remove(sc_cache *c, int key)
{ for (int i = 0; i < c->n; i++) if (c->held[i] == key) { c->held[i] = c->held[--c->n]; return slots[63]; } return NULL; }
static void f_release(sc_cache *c, unsigned char *s) { (void)c; (void)s; }
static void f_set_nu(sc_next_use_fn f) { if (!f) nu_unset = 1; }

/* fake route table */
struct rt_table { size_t ncalls; int layer[4]; size_t nexp[4]; int exps[4][4]; };
static struct rt_table tab;
static long cur_set = -9;
static size_t t_calls(const rt_table *t) { return t->ncalls; }
static int t_layer(const rt_table *t, size_t i) { return t->layer[i]; }
static const int *t_experts(const rt_table *t, size_t i, size_t *n) { *n = t->nexp[i]; return t->nexp[i] ? t->exps[i] : NULL; }
static void t_set_cur(rt_table *t, long cur) { (void)t; cur_set = cur; }
static int t_next_use(rt_table *t, int key) { (void)t; (void)key; return 1000000000; }

/* recording queue */
static cq_job jobs[32];
static int nj;
static void q_enqueue(const cq_job *j) { jobs[nj++] = *j; }

static void set_state(int *p, int ms)
{
    usleep(ms * 1000);
    __atomic_store_n(p, 1, __ATOMIC_RELEASE);
}
static void *setter(void *arg) { set_state((int *)arg, 60); return NULL; }

int main(void)
{
    sp_ops ops = { f_set_nu, f_contains, f_worth, f_acquire, f_try, f_remove, f_release,
                   &tab, t_calls, t_layer, t_experts, t_set_cur, t_next_use, q_enqueue };
    sp_config cfg = { .experts = 10, .expert_raw = 1000, .cards = 1, .lookahead = 5, .pend_cap = 100, .vq = 0, .vq_cap = 256, .vq_from = 0, .prefill = 0, .rc_ok = 1 };
    sp_stats st;

    /* pools and host set before sp_init are kept; layer 1 opens without a file */
    sp_pool_set(0, &pool0, 5, 1);
    sp_host_set(&host, 10);
    sp_init(&ops, &cfg);
    sp_layer_set(1, -1, 0);

    tab.ncalls = 2;
    tab.layer[0] = 1; tab.nexp[0] = 2; tab.exps[0][0] = 3; tab.exps[0][1] = 4;
    tab.layer[1] = 1; tab.nexp[1] = 1; tab.exps[1][0] = 5;
    pool0.held[pool0.n++] = 13; /* key 13 (layer 1, expert 3) is on the card */

    /* read-ahead plan: layer 1 has no file yet -> nothing planned; then it opens and a new planner plans */
    int u0[2] = { 3, 4 };
    sp_route_hook(1, u0, 2);
    sp_get_stats(&st);
    check(nj == 0 && st.planned_to == 2 && st.pos == 1 && cur_set == 0, "layer without a file is passed over");
    sp_layer_set_fd(1, 11);
    sp_init(&ops, &cfg);
    sp_route_hook(1, u0, 2);
    sp_get_stats(&st);
    check(nj == 2 && jobs[0].kind == 0 && jobs[0].fd == 11 && jobs[0].off == 4000 && jobs[0].p_state && jobs[0].slot == slots[0], "job 0: expert 4, kind 0, fd 11, off 4000");
    check(jobs[1].off == 5000 && jobs[1].slot == slots[1], "job 1: expert 5 of call 1");
    check(st.planned == 2 && st.skipped == 1 && st.planned_to == 2 && st.pending == 2 && st.pos == 1, "stats: planned 2, skipped 1, pending 2");
    check(nacq == 2 && last_nprot == 1, "second acquire had the first pending key as prot");

    /* pend_take waits for the queue's store */
    pthread_t th;
    pthread_create(&th, NULL, setter, jobs[0].p_state);
    sp_pend_take(14);
    pthread_join(th, NULL);
    sp_get_stats(&st);
    check(st.pf_used == 1 && st.pf_waited == 1 && st.t_pwait > 0.04 && st.pending == 1, "pend_take spun until the state was set");

    /* card queue plan */
    cfg.vq = 1;
    sp_init(&ops, &cfg);
    nj = 0; tab.ncalls = 1; tab.nexp[0] = 1; tab.exps[0][0] = 7;
    int u1[1] = { 7 };
    sp_route_hook(1, u1, 1);
    sp_get_stats(&st);
    check(nj == 1 && jobs[0].kind == 1 && jobs[0].dev == 5 && jobs[0].dst == slots[nslot - 1] && jobs[0].arr_state && jobs[0].layer == 1 && jobs[0].off == 7000, "vq job: kind 1, dev 5, layer 1, off 7000");
    check(ntry == 1 && last_cand == 0 && st.vq_drive == 1 && st.arrivals == 1, "try_acquire with cand = pos 0; one arrival");
    pthread_create(&th, NULL, setter, jobs[0].arr_state);
    sp_arr_wait(17);
    pthread_join(th, NULL);
    sp_get_stats(&st);
    check(st.vq_waited == 1 && st.t_vwait > 0.04 && st.arrivals == 0, "arr_wait spun until the arrival was set");

    /* a call beyond the table stops the look-ahead */
    int u2[1] = { 9 };
    fprintf(stderr, "(expected next line: route cache mismatch message)\n");
    sp_route_hook(1, u2, 1);
    sp_get_stats(&st);
    check(st.rc_ok == 0 && nu_unset == 1 && st.pos == 2, "mismatch: rc_ok 0, next-use function unset");

    puts(failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
