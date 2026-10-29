/* Smoke test of the supply library (ref/libsupply_ref.so or target/release/libsupply.so): hand-derived scenarios with scripted fake caches and a recording ops table. Every expected log
   line below was derived from supply.h (ORDER OF EFFECTS) and the original text before the first run. The same file is ref/smoke_ref.c and ../smoke_rust.c.
   Fake naming in the log:  DS device_sync, SS(s) stream_sync (0 = default, c = pool.copy, rN = rstream[N]), SD(dev) set_device, D2H/H2D(dst,src,n,stream), ER(ev,s) event_record, WE(s,ev) wait_event,
   kX check of site X (k0 cudaStreamSynchronize(0), kD device_sync, kS set_device(pool.dev), kW d2h, kE event record, kH h2d on pool.copy, kd set_device(dev), kw wait event, kc h2d reader, ks reader stream sync,
   kC pool.copy sync; "!n" appended when the status n was not 0), PF/PT/PA pool find/touch/acquire, HF/HW/HR/HL host find/worth/remove/release, HA host_acquire, PN pend_take, AW arr_wait.
   Slots: Pn pool slot n, Hn host slot n, Bn bounce n, S+off slab. */
#define _GNU_SOURCE
#include "supply.h"
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#define RAW 10000
#define NEXP 16
#define LAYER 3
static unsigned char PM[8][RAW], HM[8][RAW], BM[4][RAW], SM[4 * RAW];
static const unsigned char *slabtab[93];
static char copy_s, rs_s[4], ev_s;
static char LOG[65536];
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static int g_pass = 0, g_fail = 0;

static void lg(const char *f, ...)
{
    char b[512]; va_list ap; va_start(ap, f); vsnprintf(b, sizeof b, f, ap); va_end(ap);
    pthread_mutex_lock(&mu); strncat(LOG, b, sizeof LOG - strlen(LOG) - 1); pthread_mutex_unlock(&mu);
}
static const char *nm(const void *p)
{
    static __thread char ring[8][32]; static __thread int r;
    char *o = ring[r++ & 7];
    const unsigned char *q = (const unsigned char *)p;
    if (!q) return "-";
    for (int i = 0; i < 8; i++) {
        if (q >= PM[i] && q < PM[i] + RAW) { long d = q - PM[i]; if (d) snprintf(o, 32, "P%d+%ld", i, d); else snprintf(o, 32, "P%d", i); return o; }
        if (q >= HM[i] && q < HM[i] + RAW) { long d = q - HM[i]; if (d) snprintf(o, 32, "H%d+%ld", i, d); else snprintf(o, 32, "H%d", i); return o; }
    }
    for (int i = 0; i < 4; i++) if (q >= BM[i] && q < BM[i] + RAW) { long d = q - BM[i]; if (d) snprintf(o, 32, "B%d+%ld", i, d); else snprintf(o, 32, "B%d", i); return o; }
    if (q >= SM && q < SM + sizeof SM) { snprintf(o, 32, "S+%ld", (long)(q - SM)); return o; }
    return "?";
}
static const char *sn(const void *s)
{
    static __thread char ring[8][8]; static __thread int r; char *o = ring[r++ & 7];
    if (!s) return "0";
    if (s == &copy_s) return "c";
    for (int i = 0; i < 4; i++) if (s == &rs_s[i]) { snprintf(o, 8, "r%d", i); return o; }
    return "?";
}

/* ---- scripted fake caches */
typedef struct { int n; int key[32]; unsigned char *slot[32]; } Tab;
typedef struct { Tab held; int na, ia; unsigned char *aslot[8]; int avk[8]; float avh[8]; int nnw; int notworth[8]; } Fake;
static Fake POOL, HOST;
static Fake *fk(const sc_cache *c) { return (Fake *)c; }
static const char *which(const sc_cache *c) { return fk(c) == &POOL ? "P" : "H"; }
static void tab_put(Tab *t, int key, unsigned char *slot) { t->key[t->n] = key; t->slot[t->n++] = slot; }
static int tab_idx(Tab *t, int key) { for (int i = 0; i < t->n; i++) if (t->key[i] == key) return i; return -1; }
static unsigned char *f_find(sc_cache *c, int key)
{
    Tab *t = &fk(c)->held; int i = tab_idx(t, key);
    lg("%sF%d>%s;", which(c), key, i < 0 ? "-" : nm(t->slot[i])); return i < 0 ? NULL : t->slot[i];
}
static void f_touch(sc_cache *c, int key, float w) { lg("%sT%d,%g;", which(c), key, (double)w); }
static int f_worth(const sc_cache *c, int key)
{
    Fake *f = fk(c); int w = 1; for (int i = 0; i < f->nnw; i++) if (f->notworth[i] == key) w = 0;
    lg("%sW%d=%d;", which(c), key, w); return w;
}
static int cmpi(const void *a, const void *b) { return *(const int *)a - *(const int *)b; }
static unsigned char *f_acquire(sc_cache *c, int key, const int *prot, size_t np, float h0, int *vk, float *vh)
{
    Fake *f = fk(c); int s[64]; char list[256] = ""; for (size_t i = 0; i < np && i < 64; i++) s[i] = prot[i];
    qsort(s, np, sizeof(int), cmpi);
    for (size_t i = 0; i < np; i++) { char t[16]; snprintf(t, sizeof t, "%s%d", i ? "," : "", s[i]); strcat(list, t); }
    unsigned char *slot = f->aslot[f->ia]; *vk = f->avk[f->ia]; if (*vk >= 0) *vh = f->avh[f->ia]; f->ia++;
    lg("%sA%d,%g,[%s]>%s,%d;", which(c), key, (double)h0, list, nm(slot), *vk);
    tab_put(&f->held, key, slot); return slot;
}
static unsigned char *f_remove(sc_cache *c, int key)
{
    Tab *t = &fk(c)->held; int i = tab_idx(t, key); unsigned char *s = NULL;
    if (i >= 0) { s = t->slot[i]; t->key[i] = t->key[t->n - 1]; t->slot[i] = t->slot[t->n - 1]; t->n--; }
    lg("%sR%d>%s;", which(c), key, nm(s)); return s;
}
static void f_release(sc_cache *c, unsigned char *s) { (void)c; lg("HL(%s);", nm(s)); }
static unsigned char *f_host_acquire(int key, float heat, int *vk, float *vh)
{
    Fake *f = &HOST; unsigned char *slot = f->aslot[f->ia]; *vk = f->avk[f->ia]; if (*vk >= 0) *vh = f->avh[f->ia]; f->ia++;
    lg("HA%d,%g>%s,%d;", key, (double)heat, nm(slot), *vk); return slot;
}
static void f_pend_take(int key) { lg("PN%d;", key); }
static void f_arr_wait(int key) { lg("AW%d;", key); }

/* ---- fake CUDA */
static int inject_dev_status = 0, check_returns = 0;
static int f_device_sync(void) { lg("DS;"); return 0; }
static int f_stream_sync(void *s) { lg("SS(%s);", sn(s)); return 0; }
static int f_set_device(int d) { lg("SD(%d);", d); return (inject_dev_status && d == 5) ? inject_dev_status : 0; }
static pthread_mutex_t rec_mu = PTHREAD_MUTEX_INITIALIZER;
static struct { long dst; size_t n; } recs[64]; static int nrecs;
static int f_d2h(void *dst, const void *src, size_t n, void *s) { memcpy(dst, src, n); lg("D2H(%s,%s,%zu,%s);", nm(dst), nm(src), n, sn(s)); return 0; }
static int f_h2d(void *dst, const void *src, size_t n, void *s)
{
    memcpy(dst, src, n); lg("H2D(%s,%s,%zu,%s);", nm(dst), nm(src), n, sn(s));
    if (s != &copy_s) { pthread_mutex_lock(&rec_mu); recs[nrecs].dst = (long)((unsigned char *)dst - PM[0]); recs[nrecs++].n = n; pthread_mutex_unlock(&rec_mu); }
    return 0;
}
static int f_event_record(void *e, void *s) { (void)e; lg("ER(ev,%s);", sn(s)); return 0; }
static int f_wait_event(void *s, void *e, unsigned fl) { (void)e; (void)fl; lg("WE(%s,ev);", sn(s)); return 0; }
static const char *SITES[11][2] = {
    { "cudaStreamSynchronize(0)", "k0" }, { "cudaDeviceSynchronize()", "kD" }, { "cudaSetDevice(pool.dev)", "kS" },
    { "cudaMemcpyAsync(h, v.src, EXPERT_RAW, cudaMemcpyDeviceToHost, pool.copy)", "kW" }, { "cudaEventRecord(pool.written_back, pool.copy)", "kE" },
    { "cudaMemcpyAsync(n.dst, g_prefill ? g_slab[layer] + (size_t)(n.e & 3) * EXPERT_RAW : n.host, EXPERT_RAW, cudaMemcpyHostToDevice, pool.copy)", "kH" },
    { "cudaSetDevice(dev)", "kd" }, { "cudaStreamWaitEvent(pool.rstream[tid], pool.written_back, 0)", "kw" },
    { "cudaMemcpyAsync(dev_dst, b, len, cudaMemcpyHostToDevice, pool.rstream[tid])", "kc" }, { "cudaStreamSynchronize(pool.rstream[tid])", "ks" }, { "cudaStreamSynchronize(pool.copy)", "kC" } };
static void f_check(int status, const char *expr)
{
    const char *code = "k?";
    for (int i = 0; i < 11; i++) if (!strcmp(expr, SITES[i][0])) code = SITES[i][1];
    if (status) { lg("%s!%d;", code, status); if (!check_returns) { fprintf(stderr, "cuda error %d at %s\n", status, expr); exit(1); } }
    else lg("%s;", code);
}

/* ---- the world */
static unsigned char FILEB[NEXP * RAW];
static sy_ops ops = { f_device_sync, f_stream_sync, f_set_device, f_d2h, f_h2d, f_event_record, f_wait_event, f_check, f_find, f_touch, f_worth, f_acquire, f_remove, f_release, f_pend_take, f_arr_wait, f_host_acquire };
static sy_pool pool; static sy_host host; static const unsigned char *eptr[NEXP]; static int counts[NEXP];
static int g_fd = -1, g_fd_short = -1;
static void world(int vq, int prefill, int chunks, size_t cap, int host_null)
{
    memset(&POOL, 0, sizeof POOL); memset(&HOST, 0, sizeof HOST); LOG[0] = 0; nrecs = 0; inject_dev_status = 0; check_returns = 0;
    memset(&pool, 0, sizeof pool); memset(eptr, 0, sizeof eptr);
    for (int e = 0; e < NEXP; e++) counts[e] = 1;
    for (int i = 0; i < 8; i++) { memset(PM[i], 0xA0 + i, RAW); memset(HM[i], 0xB0 + i, RAW); }
    pool.dev = 5; pool.readers = 4; pool.cache = (sc_cache *)&POOL; pool.copy = &copy_s; pool.written_back = &ev_s; pool.h_eptr = eptr;
    for (int i = 0; i < 4; i++) { pool.rstream[i] = &rs_s[i]; pool.bounce[i] = BM[i]; }
    host.cache = host_null ? NULL : (sc_cache *)&HOST; host.cap_slots = cap;
    sy_config cfg; memset(&cfg, 0, sizeof cfg); cfg.experts = NEXP; cfg.expert_raw = RAW; cfg.vq = vq; cfg.prefill = prefill; cfg.chunks = chunks; cfg.slab = slabtab;
    sy_init(&ops, &cfg);
}
static void acq(Fake *f, unsigned char *slot, int vk, float vh) { f->aslot[f->na] = slot; f->avk[f->na] = vk; f->avh[f->na] = vh; f->na++; }
static void ensure(int fd, const int *ex, size_t n, int readers) { sy_ensure(&pool, &host, LAYER, fd, ex, n, counts, readers); }
static void check_s(const char *name, const char *got, const char *want)
{
    if (!strcmp(got, want)) { g_pass++; printf("PASS %s\n", name); }
    else { g_fail++; printf("FAIL %s\n  got : %s\n  want: %s\n", name, got, want); }
}
static void check_i(const char *name, long got, long want)
{
    if (got == want) { g_pass++; printf("PASS %s\n", name); }
    else { g_fail++; printf("FAIL %s: got %ld want %ld\n", name, got, want); }
}
static int same(const unsigned char *p, const unsigned char *q, size_t n) { return memcmp(p, q, n) == 0; }
static void check_stats(const char *name, unsigned long needed, unsigned long hc, unsigned long hh, unsigned long fd, unsigned long wb, unsigned long skipped)
{
    char b[160], w[160];
    snprintf(b, sizeof b, "%lu %lu %lu %lu %lu %lu", pool.st.needed, pool.st.hit_card, pool.st.hit_host, pool.st.from_drive, pool.st.written_back, sy_wb_skipped());
    snprintf(w, sizeof w, "%lu %lu %lu %lu %lu %lu", needed, hc, hh, fd, wb, skipped);
    check_s(name, b, w);
    check_i("  t_ensure > 0", pool.st.t_ensure > 0, 1);
    check_i("  t_ensure >= t_read + t_copy", pool.st.t_ensure >= pool.st.t_read + pool.st.t_copy, 1);
}
#define JOB(slot, src, n, s) "SD(5);kd;WE(" s ",ev);kw;H2D(" slot "," src "," #n "," s ");kc;SS(" s ");ks;"

/* run `fn` in a child with stderr captured; returns the exit status, the captured text + "LOG:" + the log in `out` */
static int child_text(void (*fn)(void), char *out, size_t cap)
{
    int p[2]; if (pipe(p)) exit(2);
    pid_t pid = fork();
    if (pid == 0) { dup2(p[1], 2); close(p[0]); close(p[1]); fn(); fprintf(stderr, "LOG:%s", LOG); _exit(0); }
    close(p[1]); size_t n = 0; ssize_t r; while ((r = read(p[0], out + n, cap - n - 1)) > 0) n += (size_t)r; out[n] = 0; close(p[0]);
    int st = 0; waitpid(pid, &st, 0); return WIFEXITED(st) ? WEXITSTATUS(st) : 1000 + WTERMSIG(st);
}
static void child_exit_log(void) { fprintf(stderr, "LOG:%s", LOG); }
static void f1_run(void) { world(0, 0, 0, 4, 0); acq(&POOL, PM[0], -1, 0); acq(&POOL, PM[1], -1, 0); atexit(child_exit_log); int ex[2] = { 1, 4 }; ensure(g_fd_short, ex, 2, 1); }
static void f2_run(void) { world(0, 0, 0, 4, 0); acq(&POOL, PM[0], -1, 0); atexit(child_exit_log); int ex[1] = { 1 }; ensure(-1, ex, 1, 1); }
static void f3_run(void) { world(0, 0, 0, 4, 0); acq(&POOL, PM[0], -1, 0); inject_dev_status = 700; atexit(child_exit_log); int ex[1] = { 1 }; ensure(g_fd, ex, 1, 1); }
static void f4_run(void) { world(0, 0, 0, 4, 0); acq(&POOL, PM[0], -1, 0); inject_dev_status = 700; check_returns = 1; atexit(child_exit_log); int ex[1] = { 1 }; ensure(g_fd, ex, 1, 1); }
static void f0_run(void) { int ex[1] = { 1 }; sy_ensure(&pool, &host, LAYER, -1, ex, 1, counts, 1); }

int main(void)
{
    for (size_t i = 0; i < sizeof FILEB; i++) FILEB[i] = (unsigned char)((i * 131 + (i >> 7)) & 0xff);
    for (size_t i = 0; i < sizeof SM; i++) SM[i] = (unsigned char)((i * 17 + (i >> 5) + 3) & 0xff);
    char path[] = "/tmp/supply-smoke-XXXXXX"; g_fd = mkstemp(path); if (g_fd < 0 || write(g_fd, FILEB, sizeof FILEB) != (ssize_t)sizeof FILEB) { perror("tmp file"); return 2; }
    char path2[] = "/tmp/supply-smoke2-XXXXXX"; g_fd_short = mkstemp(path2); if (g_fd_short < 0 || write(g_fd_short, FILEB, 45000) != 45000) { perror("tmp file 2"); return 2; }
    slabtab[LAYER] = SM;
    char buf[8192];

    { int st = child_text(f0_run, buf, sizeof buf); check_s("F0 before sy_init: message", buf, "supply: sy_ensure before sy_init\n"); check_i("F0 exit status", st, 1); }

    /* S1: all hits (vq off). prot has the 2 keys; heat = counts; nothing else is touched. */
    world(0, 0, 0, 4, 0); tab_put(&POOL.held, 50, PM[0]); tab_put(&POOL.held, 53, PM[1]); counts[2] = 3; counts[5] = 1;
    { int ex[2] = { 2, 5 }; ensure(g_fd, ex, 2, 4); }
    check_s("S1 all hits: log", LOG, "DS;kD;PF50>P0;PT50,3;PF53>P1;PT53,1;");
    check_i("S1 h_eptr[2]", eptr[2] == PM[0], 1); check_i("S1 h_eptr[5]", eptr[5] == PM[1], 1);
    check_stats("S1 stats", 2, 2, 0, 0, 0, 0);
    check_i("S1 t_read == 0 and t_copy == 0", pool.st.t_read == 0 && pool.st.t_copy == 0, 1);

    /* S2: all from RAM, vq on: stream_sync(0) first, pend_take before remove, copy RAM -> card on pool.copy, then the held slot is released after the sync. */
    world(1, 0, 0, 4, 0); acq(&POOL, PM[2], -1, 0); tab_put(&HOST.held, 49, HM[3]); counts[1] = 2; memcpy(HM[3], FILEB + 1 * RAW, RAW);
    { int ex[1] = { 1 }; ensure(g_fd, ex, 1, 1); }
    check_s("S2 all from RAM: log", LOG, "SS(0);k0;PF49>-;PA49,2,[49]>P2,-1;HF49>H3;PN49;HR49>H3;SD(5);kS;ER(ev,c);kE;H2D(P2,H3,10000,c);kH;SS(c);kC;HL(H3);");
    check_i("S2 bytes on the card", same(PM[2], FILEB + 1 * RAW, RAW), 1);
    check_stats("S2 stats", 1, 0, 1, 0, 0, 0);

    /* S3: all from the drive, one reader: jobs in expert order on the calling thread, tid 0. */
    world(0, 0, 0, 4, 0); acq(&POOL, PM[0], -1, 0); acq(&POOL, PM[1], -1, 0); acq(&POOL, PM[2], -1, 0);
    { int ex[3] = { 0, 7, 9 }; ensure(g_fd, ex, 3, 1); }
    check_s("S3 all from the drive: log", LOG,
            "DS;kD;PF48>-;PA48,1,[48,55,57]>P0,-1;PF55>-;PA55,1,[48,55,57]>P1,-1;PF57>-;PA57,1,[48,55,57]>P2,-1;HF48>-;HF55>-;HF57>-;SD(5);kS;ER(ev,c);kE;"
            JOB("P0", "B0", 10000, "r0") JOB("P1", "B0", 10000, "r0") JOB("P2", "B0", 10000, "r0") "SS(c);kC;");
    check_i("S3 bytes e0", same(PM[0], FILEB + 0 * RAW, RAW), 1); check_i("S3 bytes e7", same(PM[1], FILEB + 7 * RAW, RAW), 1); check_i("S3 bytes e9", same(PM[2], FILEB + 9 * RAW, RAW), 1);
    check_stats("S3 stats", 3, 0, 0, 3, 0, 0);
    check_i("S3 t_read and t_copy added", pool.st.t_read > 0 && pool.st.t_copy > 0, 1);

    /* S4: mix: card hit, RAM hit, 2 drive reads, two victims (one written back, one not worth a RAM slot). */
    world(0, 0, 0, 2, 0);
    tab_put(&POOL.held, 49, PM[0]); acq(&POOL, PM[1], 200, 1.5f); acq(&POOL, PM[2], 201, 0.5f); acq(&POOL, PM[3], -1, 0);
    tab_put(&HOST.held, 50, HM[0]); memcpy(HM[0], FILEB + 2 * RAW, RAW); HOST.notworth[HOST.nnw++] = 201; acq(&HOST, HM[1], 300, 2.0f);
    counts[1] = 1; counts[2] = 2; counts[3] = 3; counts[4] = 4; memset(PM[1], 0x5A, RAW);
    { int ex[4] = { 1, 2, 3, 4 }; ensure(g_fd, ex, 4, 1); }
    check_s("S4 mix: log", LOG,
            "DS;kD;PF49>P0;PT49,1;PF50>-;PA50,2,[49,50,51,52]>P1,200;PF51>-;PA51,3,[49,50,51,52]>P2,201;PF52>-;PA52,4,[49,50,51,52]>P3,-1;"
            "HF50>H0;PN50;HR50>H0;HF51>-;HF52>-;SD(5);kS;HW200=1;HA200,1.5>H1,300;D2H(H1,P1,10000,c);kW;HW201=0;ER(ev,c);kE;H2D(P1,H0,10000,c);kH;"
            JOB("P2", "B0", 10000, "r0") JOB("P3", "B0", 10000, "r0") "SS(c);kC;HL(H0);");
    check_i("S4 victim 200 bytes written down to H1", HM[1][0] == 0x5A && HM[1][RAW - 1] == 0x5A, 1);
    check_i("S4 e2 from RAM", same(PM[1], FILEB + 2 * RAW, RAW), 1); check_i("S4 e3", same(PM[2], FILEB + 3 * RAW, RAW), 1); check_i("S4 e4", same(PM[3], FILEB + 4 * RAW, RAW), 1);
    check_stats("S4 stats", 4, 1, 1, 2, 1, 1);

    /* S5a: host cap_slots 0 (cache present): no worth, no host_acquire, no write-back, victim dropped. */
    world(0, 0, 0, 0, 0); acq(&POOL, PM[0], 200, 1.0f);
    { int ex[1] = { 1 }; ensure(g_fd, ex, 1, 1); }
    check_s("S5a host cap 0: log", LOG, "DS;kD;PF49>-;PA49,1,[49]>P0,200;HF49>-;SD(5);kS;ER(ev,c);kE;" JOB("P0", "B0", 10000, "r0") "SS(c);kC;");
    check_stats("S5a stats", 1, 0, 0, 1, 0, 0);
    /* S5b: host cache NULL with cap 0: nothing is asked of it. */
    world(0, 0, 0, 0, 1); acq(&POOL, PM[0], 200, 1.0f);
    { int ex[1] = { 1 }; ensure(g_fd, ex, 1, 1); }
    check_s("S5b NULL host cache: log", LOG, "DS;kD;PF49>-;PA49,1,[49]>P0,200;SD(5);kS;ER(ev,c);kE;" JOB("P0", "B0", 10000, "r0") "SS(c);kC;");

    /* S6: prefill with a slab: no victim write-back, no pread (fd -1), sources are slab[layer] + (e & 3) * raw. */
    world(0, 1, 0, 2, 0); acq(&POOL, PM[0], 300, 1.0f); acq(&POOL, PM[1], -1, 0); tab_put(&HOST.held, 54, HM[1]);
    { int ex[2] = { 6, 13 }; ensure(-1, ex, 2, 1); }
    check_s("S6 prefill: log", LOG,
            "DS;kD;PF54>-;PA54,1,[54,61]>P0,300;PF61>-;PA61,1,[54,61]>P1,-1;HF54>H1;PN54;HR54>H1;HF61>-;SD(5);kS;ER(ev,c);kE;H2D(P0,S+20000,10000,c);kH;"
            JOB("P1", "S+10000", 10000, "r0") "SS(c);kC;HL(H1);");
    check_i("S6 e6 bytes from slab record 2", same(PM[0], SM + 2 * RAW, RAW), 1); check_i("S6 e13 bytes from slab record 1", same(PM[1], SM + 1 * RAW, RAW), 1);
    check_stats("S6 stats", 2, 0, 1, 1, 0, 0);

    /* S7: duplicate expert, vq on: the second occurrence is a card hit (arr_wait, touch), one read. */
    world(1, 0, 0, 4, 0); acq(&POOL, PM[0], -1, 0); counts[4] = 4;
    { int ex[2] = { 4, 4 }; ensure(g_fd, ex, 2, 1); }
    check_s("S7 duplicate: log", LOG, "SS(0);k0;PF52>-;PA52,4,[52]>P0,-1;PF52>P0;AW52;PT52,4;HF52>-;SD(5);kS;ER(ev,c);kE;" JOB("P0", "B0", 10000, "r0") "SS(c);kC;");
    check_stats("S7 stats", 2, 1, 0, 1, 0, 0);

    /* S8: vq on, victim: arr_wait(victim) right after the acquire, then the write-down. */
    world(1, 0, 0, 2, 0); acq(&POOL, PM[0], 77, 2.5f); acq(&HOST, HM[2], -1, 0); memset(PM[0], 0x77, RAW);
    { int ex[1] = { 1 }; ensure(g_fd, ex, 1, 1); }
    check_s("S8 vq victim: log", LOG, "SS(0);k0;PF49>-;PA49,1,[49]>P0,77;AW77;HF49>-;SD(5);kS;HW77=1;HA77,2.5>H2,-1;D2H(H2,P0,10000,c);kW;ER(ev,c);kE;" JOB("P0", "B0", 10000, "r0") "SS(c);kC;");
    check_i("S8 victim bytes in H2", HM[2][0] == 0x77 && HM[2][RAW - 1] == 0x77, 1); check_stats("S8 stats", 1, 0, 0, 1, 1, 0);

    /* S9: g_chunks = 3 on one expert: chunk = 4096, chunks 4096 + 4096 + 1808 (the 4th would start past the record). */
    world(0, 0, 3, 4, 0); acq(&POOL, PM[0], -1, 0);
    { int ex[1] = { 2 }; ensure(g_fd, ex, 1, 1); }
    check_s("S9 chunks=3: log", LOG, "DS;kD;PF50>-;PA50,1,[50]>P0,-1;HF50>-;SD(5);kS;ER(ev,c);kE;" JOB("P0", "B0", 4096, "r0") JOB("P0+4096", "B0", 4096, "r0") JOB("P0+8192", "B0", 1808, "r0") "SS(c);kC;");
    check_i("S9 bytes", same(PM[0], FILEB + 2 * RAW, RAW), 1);
    world(0, 0, 4, 4, 0); acq(&POOL, PM[0], -1, 0);
    { int ex[1] = { 2 }; ensure(g_fd, ex, 1, 1); }
    check_s("S9b chunks=4: the fourth chunk starts at 12288 >= 10000 and is not issued", LOG,
            "DS;kD;PF50>-;PA50,1,[50]>P0,-1;HF50>-;SD(5);kS;ER(ev,c);kE;" JOB("P0", "B0", 4096, "r0") JOB("P0+4096", "B0", 4096, "r0") JOB("P0+8192", "B0", 1808, "r0") "SS(c);kC;");

    /* S10: computed chunks with real threads: readers 16 but pool.readers 4 -> pool_readers 4; 2 experts -> nchunk = min(8, (4 + 2 - 1) / 2) = 2, chunk 8192 -> 4 jobs on 4 threads. */
    world(0, 0, 0, 4, 0); acq(&POOL, PM[0], -1, 0); acq(&POOL, PM[1], -1, 0);
    { int ex[2] = { 0, 1 }; ensure(g_fd, ex, 2, 16); }
    check_i("S10 reader records", nrecs, 4);
    { int c0 = 0, c1 = 0, c2 = 0, c3 = 0; for (int i = 0; i < nrecs; i++) { long d = recs[i].dst - 0; size_t n = recs[i].n;
        if (d == 0 && n == 8192) { c0++; }
        if (d == 8192 && n == 1808) { c1++; }
        if (d == (long)(PM[1] - PM[0]) && n == 8192) { c2++; }
        if (d == (long)(PM[1] - PM[0]) + 8192 && n == 1808) { c3++; } }
      check_i("S10 each of the 4 chunks exactly once", c0 == 1 && c1 == 1 && c2 == 1 && c3 == 1, 1); }
    check_i("S10 bytes e0", same(PM[0], FILEB, RAW), 1); check_i("S10 bytes e1", same(PM[1], FILEB + RAW, RAW), 1);
    { const char *head = "DS;kD;PF48>-;PA48,1,[48,49]>P0,-1;PF49>-;PA49,1,[48,49]>P1,-1;HF48>-;HF49>-;SD(5);kS;ER(ev,c);kE;"; const char *tail = "SS(c);kC;"; size_t L = strlen(LOG);
      check_i("S10 log starts with the main-thread part", strncmp(LOG, head, strlen(head)) == 0, 1); check_i("S10 log ends with the final sync", L >= strlen(tail) && strcmp(LOG + L - strlen(tail), tail) == 0, 1);
      int n = 0; for (char *q = LOG; (q = strstr(q, "SD(5);")); q++) n++; check_i("S10 the main thread's set_device and four jobs' set_device", n, 5); }
    check_stats("S10 stats", 2, 0, 0, 2, 0, 0);

    /* S11: 8 experts, 4 reader threads, nchunk 1: every expert's bytes are on the card. */
    world(0, 0, 0, 4, 0); for (int i = 0; i < 8; i++) acq(&POOL, PM[i], -1, 0);
    { int ex[8] = { 0, 1, 2, 3, 4, 5, 6, 7 }; ensure(g_fd, ex, 8, 4); }
    { int ok = 1; for (int i = 0; i < 8; i++) ok &= same(PM[i], FILEB + (size_t)i * RAW, RAW); check_i("S11 bytes of all 8 experts", ok, 1); }
    check_i("S11 reader records", nrecs, 8);

    /* failure paths in a child process */
    { int st = child_text(f1_run, buf, sizeof buf); check_s("F1 EOF in expert 4 (partial read): message and log", buf,
        "reading expert 4 of layer 3 failed: end of file\nLOG:DS;kD;PF49>-;PA49,1,[49,52]>P0,-1;PF52>-;PA52,1,[49,52]>P1,-1;HF49>-;HF52>-;SD(5);kS;ER(ev,c);kE;" JOB("P0", "B0", 10000, "r0"));
      check_i("F1 exit status", st, 1); }
    { int st = child_text(f2_run, buf, sizeof buf); check_s("F2 bad fd: message and log", buf,
        "reading expert 1 of layer 3 failed: Bad file descriptor\nLOG:DS;kD;PF49>-;PA49,1,[49]>P0,-1;HF49>-;SD(5);kS;ER(ev,c);kE;"); check_i("F2 exit status", st, 1); }
    { int st = child_text(f3_run, buf, sizeof buf); check_s("F3 CUDA error through check: nothing after it", buf,
        "cuda error 700 at cudaSetDevice(pool.dev)\nLOG:DS;kD;PF49>-;PA49,1,[49]>P0,-1;HF49>-;SD(5);kS!700;"); check_i("F3 exit status", st, 1); }
    { int st = child_text(f4_run, buf, sizeof buf); check_s("F4 check returns with a status: supply exits 1 at once", buf,
        "LOG:DS;kD;PF49>-;PA49,1,[49]>P0,-1;HF49>-;SD(5);kS!700;"); check_i("F4 exit status", st, 1); }

    close(g_fd); close(g_fd_short); unlink(path); unlink(path2);
    printf("%s (%d checks, %d failed)\n", g_fail ? "FAIL" : "PASS", g_pass + g_fail, g_fail);
    return g_fail ? 1 : 0;
}
