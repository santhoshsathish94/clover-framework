/* Smoke test of loader.h against ANY implementation (linked: libloader_ref.so or libloader.so). Hand-derived expectations (derivations in the comments), recording fake callbacks, no CUDA.
   Usage: smoke <mode>   mode = basic | noprefill | prefill | missing | eof | badcard
   basic: ld_submit order, thread start, wait, cross-card concurrency (3 cards on a timed barrier), a job submitting a job.
   noprefill / prefill: ld_open_layer of layers 0 and 7 (bookkeeping, callback order, fd, slab). missing / eof / badcard: exit status and stderr text (run in a child process). */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include "loader.h"

#define RAW 8192u   /* a multiple of 4096 so that O_DIRECT reads are allowed where the file system accepts O_DIRECT */
static int g_fail = 0, g_checks = 0;
#define CHECKX(c, ...) do { g_checks++; if (!(c)) { g_fail++; fprintf(stdout, "FAIL line %d: ", __LINE__); fprintf(stdout, __VA_ARGS__); fputc('\n', stdout); } } while (0)

static pthread_mutex_t g_lm = PTHREAD_MUTEX_INITIALIZER;
static char g_log[16384]; static size_t g_loglen;
static void LOG(const char *fmt, ...)
{
    char b[512]; va_list ap; va_start(ap, fmt); int n = vsnprintf(b, sizeof b, fmt, ap); va_end(ap);
    pthread_mutex_lock(&g_lm);
    if (g_loglen + (size_t)n + 1 < sizeof g_log) { memcpy(g_log + g_loglen, b, (size_t)n); g_loglen += (size_t)n; g_log[g_loglen++] = '\n'; g_log[g_loglen] = 0; }
    pthread_mutex_unlock(&g_lm);
}
static void log_reset(void) { pthread_mutex_lock(&g_lm); g_loglen = 0; g_log[0] = 0; pthread_mutex_unlock(&g_lm); }
static int log_eq(const char *want) { pthread_mutex_lock(&g_lm); int r = strcmp(g_log, want) == 0; if (!r) fprintf(stdout, "  log is:\n%s  want:\n%s", g_log, want); pthread_mutex_unlock(&g_lm); return r; }

static void cb_set_device(void *ctx, int card) { (void)ctx; LOG("set_device %d", card); }
static void cb_ensure_work(void *ctx, int card) { (void)ctx; LOG("ensure_work %d", card); }
static void cb_trunk(void *ctx, void *h, const char *ds, int L) { (void)ctx; LOG("trunk L%d h=%d ds=%s", L, *(int *)h, ds); }
static void cb_tables(void *ctx, void *h, const char *ds, int L) { (void)ctx; (void)ds; LOG("tables L%d h=%d", L, *(int *)h); }
static void cb_state(void *ctx, void *h, unsigned positions) { (void)ctx; LOG("state h=%d positions=%u", *(int *)h, positions); }
static void cb_host_alloc(void *ctx, unsigned char **dst, size_t bytes)
{
    (void)ctx; void *p = NULL;
    if (posix_memalign(&p, 4096, bytes ? bytes : 1)) { fprintf(stderr, "posix_memalign\n"); exit(2); }
    memset(p, 0xEE, bytes); *dst = (unsigned char *)p; LOG("host_alloc %zu", bytes);
}

static unsigned char *g_slab[93];
static void *g_handles[93];
static void cfg_init(int prefill)
{
    ld_config c; memset(&c, 0, sizeof c);
    c.cb.ctx = NULL; c.cb.set_device = cb_set_device; c.cb.ensure_work = cb_ensure_work; c.cb.load_trunk = cb_trunk; c.cb.load_expert_tables = cb_tables;
    c.cb.alloc_state = cb_state; c.cb.host_alloc = cb_host_alloc; c.expert_raw = RAW; c.prefill = prefill; c.slab = g_slab; c.handles = g_handles;
    ld_init(&c);
}

/* ---- basic */
static pthread_mutex_t g_bm = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t g_bc = PTHREAD_COND_INITIALIZER; static int g_bn, g_bparties, g_bbroken;
static void job_log(void *arg) { LOG("j%d", (int)(long)arg); }
static void job_barrier(void *arg)
{
    (void)arg; struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts); ts.tv_sec += 10;
    pthread_mutex_lock(&g_bm); g_bn++;
    if (g_bn == g_bparties) pthread_cond_broadcast(&g_bc);
    while (g_bn < g_bparties && !g_bbroken) if (pthread_cond_timedwait(&g_bc, &g_bm, &ts) == ETIMEDOUT) { g_bbroken = 1; pthread_cond_broadcast(&g_bc); }
    pthread_mutex_unlock(&g_bm);
}
static void job_spawn(void *arg) { LOG("parent"); ld_submit((int)(long)arg, job_log, (void *)7L); LOG("parent done"); }

static void mode_basic(void)
{
    cfg_init(0);
    ld_wait();   /* nothing submitted: returns at once (the alarm would fire otherwise) */
    g_checks++;
    /* three jobs on card 2: its thread calls set_device(2) first, then the jobs in submission order */
    log_reset();
    for (long i = 0; i < 3; i++) ld_submit(2, job_log, (void *)i);
    ld_wait();
    CHECKX(log_eq("set_device 2\nj0\nj1\nj2\n"), "fifo on one card");
    /* a second wait with nothing pending, and a later job on the same card: no second set_device (the thread exists) */
    log_reset(); ld_wait(); ld_submit(2, job_log, (void *)9L); ld_wait();
    CHECKX(log_eq("j9\n"), "no thread restart");
    /* three cards on one barrier: only passes if the three first jobs run at the same time */
    g_bparties = 3; g_bn = 0; g_bbroken = 0; log_reset();
    ld_submit(3, job_barrier, NULL); ld_submit(4, job_barrier, NULL); ld_submit(5, job_barrier, NULL);
    ld_wait();
    CHECKX(!g_bbroken && g_bn == 3, "cross-card concurrency (barrier broken=%d n=%d)", g_bbroken, g_bn);
    /* a job that submits a job: wait covers the child (parent, then parent done, then the child later on card 6's thread) */
    log_reset(); ld_submit(6, job_spawn, (void *)6L); ld_wait();
    CHECKX(log_eq("set_device 6\nparent\nparent done\nj7\n"), "child job");
}

/* ---- files */
static char g_dir[256];
static void make_file(const char *dataset, unsigned layer, unsigned records)
{
    char p[512]; snprintf(p, sizeof p, "%s/expert-%u", dataset, layer); mkdir(p, 0700);
    snprintf(p, sizeof p, "%s/expert-%u/experts.direct", dataset, layer);
    FILE *f = fopen(p, "wb"); if (!f) { perror(p); exit(2); }
    for (unsigned r = 0; r < records; r++) for (unsigned i = 0; i < RAW; i++) fputc((int)((r * 37u + i * 13u + 5u) & 0xff), f);
    fclose(f);
}
static void rm_tree(const char *dataset, unsigned layer)
{
    char p[512]; snprintf(p, sizeof p, "%s/expert-%u/experts.direct", dataset, layer); unlink(p);
    snprintf(p, sizeof p, "%s/expert-%u", dataset, layer); rmdir(p);
}

static void mode_open(int prefill)
{
    cfg_init(prefill);
    snprintf(g_dir, sizeof g_dir, "/tmp/ldsmoke.%d", (int)getpid()); mkdir(g_dir, 0700);
    make_file(g_dir, 7, 5);   /* 5 records: the slab takes the first 4 */
    int hv0 = 1000; int L0 = -1, C0 = -1, K0 = -1, F0 = 424242;
    ld_layer q0; memset(&q0, 0, sizeof q0);
    q0.handle = &hv0; q0.dataset = g_dir; q0.layer = 0; q0.k = 4; q0.positions = 77; q0.card = 1; q0.h_L = &L0; q0.h_card = &C0; q0.h_k_in = &K0; q0.h_fd = &F0;
    log_reset(); ld_open_layer(&q0);
    /* bookkeeping happens on return, before the job has to have run */
    CHECKX(L0 == 0 && C0 == 1 && K0 == 4, "layer 0 bookkeeping L=%d card=%d k=%d", L0, C0, K0);
    CHECKX(g_handles[0] == &hv0, "g_handles[0]");
    ld_wait();
    /* layer 0: thread start set_device, job set_device, ensure_work, trunk, no tables, state; no file, fd untouched */
    char want0[512]; snprintf(want0, sizeof want0, "set_device 1\nset_device 1\nensure_work 1\ntrunk L0 h=1000 ds=%s\nstate h=1000 positions=77\n", g_dir);
    CHECKX(log_eq(want0), "layer 0 job order");
    CHECKX(F0 == 424242, "layer 0 must not write fd (%d)", F0);

    int hv7 = 7000; int L7 = -1, C7 = -1, K7 = -1, F7 = -1;
    ld_layer q7; memset(&q7, 0, sizeof q7);
    q7.handle = &hv7; q7.dataset = g_dir; q7.layer = 7; q7.k = 2; q7.positions = 5; q7.card = 3; q7.h_L = &L7; q7.h_card = &C7; q7.h_k_in = &K7; q7.h_fd = &F7;
    log_reset(); ld_open_layer(&q7); ld_wait();
    CHECKX(L7 == 7 && C7 == 3 && K7 == 2 && g_handles[7] == &hv7, "layer 7 bookkeeping");
    char want7[1024];
    snprintf(want7, sizeof want7, "set_device 3\nset_device 3\nensure_work 3\ntrunk L7 h=7000 ds=%s\ntables L7 h=7000\nstate h=7000 positions=5\n%s", g_dir, prefill ? "host_alloc 32768\n" : "");
    CHECKX(log_eq(want7), "layer 7 job order (prefill=%d)", prefill);
    CHECKX(F7 >= 0, "fd opened (%d)", F7);
    if (F7 >= 0) {   /* O_DIRECT fd: read an aligned block into aligned memory */
        void *buf = NULL; if (posix_memalign(&buf, 4096, 4096)) exit(2);
        ssize_t got = pread(F7, buf, 4096, 4096);   /* record 0, bytes 4096..8191 */
        int ok = got == 4096; for (unsigned i = 0; ok && i < 4096; i++) if (((unsigned char *)buf)[i] != (unsigned char)((((4096u + i) * 13u) + 5u) & 0xff)) ok = 0;
        CHECKX(ok, "fd reads record 0 (got %zd)", got); free(buf);
    }
    if (prefill) {
        CHECKX(g_slab[7] != NULL, "slab[7] set");
        int ok = g_slab[7] != NULL;
        for (unsigned r = 0; ok && r < 4; r++) for (unsigned i = 0; i < RAW; i++) if (g_slab[7][(size_t)r * RAW + i] != (unsigned char)((r * 37u + i * 13u + 5u) & 0xff)) { ok = 0; break; }
        CHECKX(ok, "slab = first 4 records of the file");
    } else CHECKX(g_slab[7] == NULL, "slab untouched without prefill");
    rm_tree(g_dir, 7); rmdir(g_dir);
}

/* ---- fatal paths, in a child so the exit status and the stderr text can be read */
static int run_child(void (*body)(void), char *err, size_t errsz)
{
    int pfd[2]; if (pipe(pfd)) exit(2);
    fflush(stdout);
    pid_t pid = fork();
    if (pid == 0) { dup2(pfd[1], 2); close(pfd[0]); close(pfd[1]); body(); _exit(77); }
    close(pfd[1]);
    size_t n = 0; ssize_t r;
    while ((r = read(pfd[0], err + n, errsz - 1 - n)) > 0) n += (size_t)r;
    err[n] = 0; close(pfd[0]);
    int st = 0; waitpid(pid, &st, 0);
    return WIFEXITED(st) ? WEXITSTATUS(st) : 1000 + WTERMSIG(st);
}
static void child_missing(void)
{
    cfg_init(0);
    int hv = 1, L = 0, C = 0, K = 0, F = -5;
    ld_layer q; memset(&q, 0, sizeof q); q.handle = &hv; q.dataset = "/tmp/ldsmoke-does-not-exist"; q.layer = 3; q.positions = 1; q.card = 0; q.h_L = &L; q.h_card = &C; q.h_k_in = &K; q.h_fd = &F;
    ld_open_layer(&q); ld_wait(); _exit(55);
}
static char g_eofdir[256];
static void child_eof(void)
{
    cfg_init(1);
    make_file(g_eofdir, 9, 3);   /* only 3 records: record 3 hits end of file at its first byte */
    int hv = 1, L = 0, C = 0, K = 0, F = -5;
    ld_layer q; memset(&q, 0, sizeof q); q.handle = &hv; q.dataset = g_eofdir; q.layer = 9; q.positions = 1; q.card = 2; q.h_L = &L; q.h_card = &C; q.h_k_in = &K; q.h_fd = &F;
    ld_open_layer(&q); ld_wait(); _exit(55);
}
static void child_badcard(void)
{
    cfg_init(0);
    ld_submit(8, job_log, NULL); _exit(55);
}

int main(int argc, char **argv)
{
    alarm(60);
    const char *m = argc > 1 ? argv[1] : "basic";
    if (!strcmp(m, "basic")) mode_basic();
    else if (!strcmp(m, "noprefill")) mode_open(0);
    else if (!strcmp(m, "prefill")) mode_open(1);
    else if (!strcmp(m, "missing")) {
        char e[1024]; int st = run_child(child_missing, e, sizeof e);
        CHECKX(st == 1, "missing file: exit status %d", st);
        CHECKX(!strcmp(e, "cannot open /tmp/ldsmoke-does-not-exist/expert-3/experts.direct\n"), "missing file: stderr [%s]", e);
    } else if (!strcmp(m, "eof")) {
        snprintf(g_eofdir, sizeof g_eofdir, "/tmp/ldsmoke.eof.%d", (int)getpid()); mkdir(g_eofdir, 0700);
        char e[1024]; int st = run_child(child_eof, e, sizeof e);
        CHECKX(st == 1, "eof: exit status %d", st);
        CHECKX(!strcmp(e, "prefill: cannot read a record of layer 9\n"), "eof: stderr [%s]", e);
        rm_tree(g_eofdir, 9); rmdir(g_eofdir);
    } else if (!strcmp(m, "badcard")) {
        char e[1024]; int st = run_child(child_badcard, e, sizeof e);
        CHECKX(st == 1, "badcard: exit status %d", st);
        CHECKX(!strcmp(e, "loader: card 8 out of range (0..7)\n"), "badcard: stderr [%s]", e);
    } else { fprintf(stderr, "unknown mode %s\n", m); return 2; }
    printf("%s %s (%d checks, %d failed)\n", g_fail ? "FAIL" : "PASS", m, g_checks, g_fail);
    return g_fail ? 1 : 0;
}
