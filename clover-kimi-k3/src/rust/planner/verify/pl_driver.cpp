// Differential driver of the supply planner (include/planner.h): the C++ reference (ref/libplanner_ref.so) against the Rust crate (libplanner.so), same C ABI.
// One binary, three modes:
//   child : runs ONE scenario unit against ONE planner library inside this process and writes a canonical log (the planner's stderr goes into the same file).
//   unit  : generates the unit's trace, starts a child per planner library, compares the logs byte for byte, prints/writes one result line.
//   run   : schedules many units over many processes and prints one summary line per scenario.
// Inside a child everything but the planner is REAL: slot caches (libslotcache.so), route table (libroutetable.so, loaded from a trace file); only the copy queue is a fake that records
// jobs and completes them from a seeded RNG (or, in P6, from a second thread). The sp_ops table is a logging wrapper around the real functions.
// Scenarios: P1 read-ahead (vq 0) | P2 card queue (vq 1) | P3 the real trace | P4 mismatching sp_route_hook calls | P5 corner configs | P6 concurrency (invariants only).
#include "planner.h"
#include <dlfcn.h>
#include <fcntl.h>
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <climits>
#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// ------------------------------------------------------------------------------------------------ utilities
static uint64_t mix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ULL; x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL; x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL; return x ^ (x >> 31);
}
static uint64_t fnv(const std::string &s) { uint64_t h = 1469598103934665603ULL; for (unsigned char c : s) { h ^= c; h *= 1099511628211ULL; } return h; }
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed = 0) : s(seed * 0x9E3779B97F4A7C15ULL + 0x1234567ULL) {}
    uint64_t u() { s += 0x9E3779B97F4A7C15ULL; uint64_t z = s; z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL; return z ^ (z >> 31); }
    uint32_t r(uint32_t n) { return n ? (uint32_t)(u() % n) : 0; }
    bool p(double q) { return (double)(u() >> 11) * (1.0 / 9007199254740992.0) < q; }
    int range(int a, int b) { return a + (int)r((uint32_t)(b - a + 1)); }
};
static uint32_t fbits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static std::string sfmt(const char *f, ...)
{
    char b[1024]; va_list ap; va_start(ap, f); vsnprintf(b, sizeof b, f, ap); va_end(ap); return b;
}
static const uintptr_t SLOT_BASE = 0x100000;
static unsigned char *slotp(int id) { return (unsigned char *)(SLOT_BASE + (uintptr_t)id * 64); }
static int slotid(const void *p) { return p ? (int)(((uintptr_t)p - SLOT_BASE) / 64) : -1; }

// ------------------------------------------------------------------------------------------------ unit configuration (a pure function of (scenario, seed))
struct UCfg {
    std::string scn; int seed = 0;
    int experts = 896; long expert_raw = 17547264; int cards = 8; int lookahead = 92;
    size_t pend_cap = 0; int vq = 0; size_t vq_cap = 256; long vq_from = 0; int prefill = 0; int rc_force0 = 0;
    int policy = 1; int install_nu = 0;
    int pool_slots = 60, host_slots = 300; long host_cap = 300; int layout = 0;
    unsigned char no_handle[93]; int late_fd[93]; int ready_at[8];   // late_fd: -1 fd set at start, -2 never, >= 0 step; ready_at: -1 ready at start, -2 never, >= 0 step
    double p_complete = 0.3, p_spin = 0.0;
    bool real_trace = false; std::string trace_path;
    int ncalls = 0, layer_mode = 0, maxlen = 20, pop = 20, dup_pct = 0, empty_tail = 0;
    std::vector<std::pair<int, int>> mm; int extra_hooks = 0;   // P4: (position, kind) and hooks past the end of the table
    bool threads = false; std::string size_name;
    UCfg() { memset(no_handle, 0, sizeof no_handle); for (int &x : late_fd) x = -1; for (int &x : ready_at) x = -1; }
};

static UCfg make_cfg(const std::string &scn, int seed, const std::string &p3trace)
{
    UCfg c; c.scn = scn; c.seed = seed;
    Rng R(mix64((uint64_t)seed) ^ fnv(scn));
    auto pk = [&](std::initializer_list<long> l) { return *(l.begin() + R.r((uint32_t)l.size())); };
    auto pkd = [&](std::initializer_list<double> l) { return *(l.begin() + R.r((uint32_t)l.size())); };
    if (scn == "P3") {   // seed = config index: policy x vq x size x repetition; the whole real trace
        const int pol = seed % 3, vq = (seed / 3) % 2, sz = (seed / 6) % 2, rs = seed / 12;
        c.policy = pol; c.vq = vq; c.real_trace = true; c.trace_path = p3trace; c.cards = 8; c.layout = 0; c.install_nu = pol == 2 ? 1 : 0;
        if (sz == 0) {
            c.pool_slots = 1000; c.host_slots = 9000; c.host_cap = 9000; c.pend_cap = 5400; c.lookahead = 92; c.size_name = "real";
            c.vq_cap = (rs % 2) ? 2048 : 256; c.vq_from = ((rs / 2) % 2) ? 92 : 0;
        } else {
            c.pool_slots = 60; c.host_slots = 300; c.host_cap = 300; c.size_name = "tiny";
            c.pend_cap = (rs % 3 == 0) ? 180 : (rs % 3 == 1 ? 50 : 3); c.lookahead = (rs % 3 == 0) ? 92 : (rs % 3 == 1 ? 5 : 500);
            c.vq_cap = 4; c.vq_from = ((rs / 3) % 3 == 0) ? 0 : (((rs / 3) % 3 == 1) ? 92 : 1000000);
        }
        c.prefill = (rs % 4 == 3) ? 1 : 0;
        c.p_complete = (rs % 3 == 0) ? 0.3 : (rs % 3 == 1 ? 0.05 : 0.9); c.p_spin = (rs % 2) ? 0.0 : 0.01;
        return c;
    }
    const int sz = (int)R.r(100);
    if (sz < 10) { c.pool_slots = 1000; c.host_slots = 9000; c.size_name = "real"; }
    else if (sz < 30) { c.pool_slots = 200; c.host_slots = 1500; c.size_name = "mid"; }
    else { c.pool_slots = (int)pk({ 60, 60, 40, 30 }); c.host_slots = (int)pk({ 300, 300, 120, 600 }); c.size_name = "tiny"; }
    c.host_cap = c.host_slots;
    const long full = (long)c.host_slots * 6 / 10;
    c.pend_cap = (size_t)std::min<long>(pk({ 3, 50, full, full }), full);
    c.cards = (int)pk({ 8, 8, 8, 8, 4, 3, 2, 1 }); c.layout = (int)R.r(2);
    c.lookahead = (int)pk({ 0, 1, 5, 5, 92, 92, 92, 500 });
    c.vq_cap = (size_t)pk({ 256, 256, 2048, 4 }); c.vq_from = pk({ 0, 0, 30, 92, 1000000 });
    c.prefill = R.p(0.1) ? 1 : 0;
    if (scn == "P1") { c.vq = 0; c.policy = (int)R.r(3); }
    else if (scn == "P2") { c.vq = 1; c.policy = R.p(0.7) ? 2 : (int)R.r(2); }
    else if (scn == "P4") { c.vq = (int)R.r(2); c.policy = (int)R.r(3); }
    else if (scn == "P5") { c.vq = (int)R.r(2); c.policy = (int)R.r(3); }
    else { c.vq = R.p(0.7) ? 1 : 0; c.policy = R.p(0.6) ? 2 : (int)R.r(2); if (c.vq) c.vq_from = pk({ 0, 0, 3 }); }   // P6
    c.install_nu = (c.policy == 2) ? (R.p(0.9) ? 1 : 0) : (R.p(0.3) ? 1 : 0);
    c.ncalls = (c.size_name == "real") ? R.range(20, 60) : R.range(30, 220);
    c.layer_mode = (int)R.r(2); c.maxlen = (int)pk({ 4, 12, 40, 120 }); c.pop = R.range(6, 60); c.dup_pct = (int)pk({ 0, 0, 0, 10, 40 });
    c.p_complete = pkd({ 0.0, 0.05, 0.3, 0.9 }); c.p_spin = pkd({ 0.0, 0.0, 0.02, 0.1 });
    if (R.p(0.15)) for (int L = 0; L < 93; L++) c.no_handle[L] = R.p(0.08) ? 1 : 0;
    if (R.p(0.15)) for (int L = 0; L < 93; L++) if (!c.no_handle[L] && R.p(0.15)) c.late_fd[L] = R.p(0.3) ? -2 : R.range(0, c.ncalls);
    if (R.p(0.12)) for (int k = 0; k < 8; k++) { const int w = (int)R.r(3); c.ready_at[k] = w == 0 ? -1 : (w == 1 ? R.range(0, c.ncalls) : -2); }
    if (scn == "P4") {
        if (R.p(0.25)) { c.extra_hooks = R.range(1, 4); }
        else { const int n = R.range(1, 3); for (int i = 0; i < n; i++) c.mm.push_back({ (int)R.r((uint32_t)c.ncalls), R.range(1, 6) }); c.extra_hooks = R.range(0, 3); }
    }
    if (scn == "P5") {
        c.lookahead = (int)pk({ 0, 0, 1, 1 << 30, INT_MAX, 5 });
        c.pend_cap = (size_t)pk({ 0, 0, 1, 3, full, 1 << 20 });   // 1<<20 can fill the host cache: the fatal exit of the slot cache is then part of the comparison
        c.vq_cap = (size_t)pk({ 0, 1, 4, 256 }); c.vq_from = pk({ 0, 5, 1000000 });
        c.prefill = R.p(0.3) ? 1 : 0;
        if (R.p(0.2)) c.host_cap = 0;
        if (R.p(0.1)) { c.host_slots = (int)pk({ 0, 1, 2 }); c.host_cap = c.host_slots; }
        c.experts = (int)pk({ 896, 896, 896, 17, 1 });
        if (R.p(0.1)) c.rc_force0 = 1;
        if (R.p(0.08)) { c.ncalls = 0; c.extra_hooks = 6; for (int &x : c.late_fd) x = -1; for (int &x : c.ready_at) x = -1; }
        else c.extra_hooks = R.range(0, 2);
        if (R.p(0.3)) for (int k = 0; k < 8; k++) c.ready_at[k] = -2;
        if (R.p(0.3)) for (int L = 0; L < 93; L++) c.no_handle[L] = R.p(0.6) ? 1 : 0;
        c.cards = (int)pk({ 1, 3, 8 });
    }
    if (scn == "P6") { c.threads = true; c.empty_tail = 2; c.extra_hooks = 0; for (int &x : c.ready_at) x = -1; }
    return c;
}

static std::string cfg_str(const UCfg &c)
{
    int nh = 0, nl = 0, nn = 0, nr = 0;
    for (int L = 0; L < 93; L++) { nh += c.no_handle[L]; nl += c.late_fd[L] != -1; nn += c.late_fd[L] == -2; }
    for (int k = 0; k < 8; k++) nr += c.ready_at[k] != -1;
    return sfmt("CFG scn=%s seed=%d size=%s experts=%d raw=%ld cards=%d layout=%d la=%d pend_cap=%zu vq=%d vq_cap=%zu vq_from=%ld prefill=%d rc0=%d policy=%d nu=%d pool=%d host=%d hostcap=%ld "
                "nohandle=%d latefd=%d never=%d notready=%d pc=%.2f ps=%.2f real=%d ncalls=%d mm=%zu extra=%d thr=%d",
                c.scn.c_str(), c.seed, c.size_name.c_str(), c.experts, c.expert_raw, c.cards, c.layout, c.lookahead, c.pend_cap, c.vq, c.vq_cap, c.vq_from, c.prefill, c.rc_force0, c.policy, c.install_nu,
                c.pool_slots, c.host_slots, c.host_cap, nh, nl, nn, nr, c.p_complete, c.p_spin, (int)c.real_trace, c.ncalls, c.mm.size(), c.extra_hooks, (int)c.threads);
}

static std::string gen_trace(const UCfg &c)
{
    Rng R((uint64_t)c.seed * 7919ULL + 17);
    std::string out;
    const int E = c.experts;
    const int n = c.ncalls + c.empty_tail;
    for (int i = 0; i < n; i++) {
        if (i >= c.ncalls) { out += sfmt("%d 0\n", 5); continue; }   // P6 tail: an empty call
        int L = c.layer_mode == 0 ? (int)R.r(93) : 1 + (i % 92);
        if (c.layer_mode == 1 && R.p(0.2)) L = 1 + (int)R.r(92);
        int len = R.p(0.04) ? 0 : 1 + (int)R.r((uint32_t)c.maxlen);
        std::vector<int> ex;
        for (int k = 0; k < len; k++) {
            const int e = R.p(0.6) ? (int)(mix64((uint64_t)L * 131 + R.r((uint32_t)c.pop)) % (uint64_t)E) : (int)R.r((uint32_t)E);
            if (std::find(ex.begin(), ex.end(), e) == ex.end()) ex.push_back(e);
        }
        if (!ex.empty() && R.p(c.dup_pct / 100.0)) { const int nd = 1 + (int)R.r(3); for (int k = 0; k < nd; k++) ex.push_back(ex[R.r((uint32_t)ex.size())]); }
        const bool counts = R.p(0.8), partial = R.p(0.1);
        out += sfmt("%d %d", L, 1 + (int)R.r(64));
        for (int e : ex) {
            out += sfmt(" %d", e);
            if (counts && (!partial || R.p(0.5))) out += sfmt(":%d", 1 + (int)R.r(9));
        }
        out += "\n";
    }
    return out;
}

// ------------------------------------------------------------------------------------------------ canonical log
static int g_logfd = -1;
static int g_exit_code = 2;   // what the child leaves with when exit() is reached by someone else: 2 while loading, 1 while running (the slot cache's and the planner's fatal exits), 3 a violation, 0 at the normal end
static std::string g_buf;
static std::mutex g_logmu;
static unsigned long g_nlines = 0, g_nz = 0;
static void flush_locked()
{
    size_t off = 0;
    while (off < g_buf.size()) { const ssize_t w = write(g_logfd, g_buf.data() + off, g_buf.size() - off); if (w <= 0) _exit(5); off += (size_t)w; }
    g_buf.clear();
}
static void flush_log() { std::lock_guard<std::mutex> g(g_logmu); flush_locked(); }
static void logs(const char *s, size_t n, bool z)
{
    std::lock_guard<std::mutex> g(g_logmu);
    g_buf.append(s, n); g_buf.push_back('\n'); g_nlines++; if (z) g_nz++;
    if (g_buf.size() >= 65536) flush_locked();   // flush points depend only on the content, so they are the same in both children
}
static void LOG(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void LOG(const char *fmt, ...)
{
    char b[640]; va_list ap; va_start(ap, fmt); int n = vsnprintf(b, sizeof b, fmt, ap); va_end(ap);
    if (n < 0) n = 0;
    if (n >= (int)sizeof b) n = (int)sizeof b - 1;
    logs(b, (size_t)n, false);
}
static void LOGZ(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void LOGZ(const char *fmt, ...)
{
    char b[640]; va_list ap; va_start(ap, fmt); int n = vsnprintf(b, sizeof b, fmt, ap); va_end(ap);
    if (n < 0) n = 0;
    if (n >= (int)sizeof b) n = (int)sizeof b - 1;
    logs(b, (size_t)n, true);
}
static void viol(const std::string &m)
{
    LOG("VIOLATION %s", m.c_str()); flush_log(); g_exit_code = 3; exit(3);
}

// ------------------------------------------------------------------------------------------------ the libraries
static struct {
    void (*set_policy)(int); void (*set_next_use)(sc_next_use_fn); sc_cache *(*nw)(); void (*add_spare)(sc_cache *, unsigned char *); void (*release)(sc_cache *, unsigned char *);
    int (*contains)(const sc_cache *, int); unsigned char *(*find)(sc_cache *, int); void (*touch)(sc_cache *, int, float); int (*worth)(const sc_cache *, int);
    unsigned char *(*acquire)(sc_cache *, int, const int *, size_t, float, int *, float *);
    unsigned char *(*try_acquire)(sc_cache *, int, int, const int *, size_t, float, int *, float *);
    unsigned char *(*remove)(sc_cache *, int);
} SC;
static struct {
    rt_table *(*load)(const char *, int); size_t (*calls)(const rt_table *); int (*layer)(const rt_table *, size_t); const int *(*experts)(const rt_table *, size_t, size_t *);
    const int *(*counts)(const rt_table *, size_t, size_t *); void (*set_current)(rt_table *, long); int (*next_use)(rt_table *, int);
} RT;
static struct {
    void (*init)(const sp_ops *, const sp_config *); void (*layer_set)(int, int, int); void (*layer_set_fd)(int, int); void (*pool_set)(int, sc_cache *, int, int);
    void (*host_set)(sc_cache *, size_t); void (*route_hook)(int, const int *, size_t); void (*pend_take)(int); void (*arr_wait)(int);
    unsigned char *(*host_acquire)(int, float, int *, float *); void (*get_stats)(sp_stats *);
} SP;
static bool g_symfail = false;
static void *must(void *h, const char *lib, const char *name)
{
    void *s = dlsym(h, name);
    if (!s) { fprintf(stderr, "missing symbol %s in %s\n", name, lib); g_symfail = true; }
    return s;
}
#define BIND(st, field, h, lib, name) do { void *s_ = must(h, lib, name); memcpy((void *)&st.field, &s_, sizeof s_); } while (0)
static void load_libs(const char *sc, const char *rt, const char *sp)
{
    void *h = dlopen(sc, RTLD_NOW | RTLD_LOCAL); if (!h) { fprintf(stderr, "dlopen %s: %s\n", sc, dlerror()); exit(2); }
    BIND(SC, set_policy, h, sc, "sc_set_policy"); BIND(SC, set_next_use, h, sc, "sc_set_next_use"); BIND(SC, nw, h, sc, "sc_new"); BIND(SC, add_spare, h, sc, "sc_add_spare");
    BIND(SC, release, h, sc, "sc_release"); BIND(SC, contains, h, sc, "sc_contains"); BIND(SC, find, h, sc, "sc_find"); BIND(SC, touch, h, sc, "sc_touch"); BIND(SC, worth, h, sc, "sc_worth");
    BIND(SC, acquire, h, sc, "sc_acquire"); BIND(SC, try_acquire, h, sc, "sc_try_acquire"); BIND(SC, remove, h, sc, "sc_remove");
    h = dlopen(rt, RTLD_NOW | RTLD_LOCAL); if (!h) { fprintf(stderr, "dlopen %s: %s\n", rt, dlerror()); exit(2); }
    BIND(RT, load, h, rt, "rt_load"); BIND(RT, calls, h, rt, "rt_calls"); BIND(RT, layer, h, rt, "rt_layer"); BIND(RT, experts, h, rt, "rt_experts"); BIND(RT, counts, h, rt, "rt_counts");
    BIND(RT, set_current, h, rt, "rt_set_current"); BIND(RT, next_use, h, rt, "rt_next_use");
    h = dlopen(sp, RTLD_NOW | RTLD_LOCAL); if (!h) { fprintf(stderr, "dlopen %s: %s\n", sp, dlerror()); exit(2); }
    BIND(SP, init, h, sp, "sp_init"); BIND(SP, layer_set, h, sp, "sp_layer_set"); BIND(SP, layer_set_fd, h, sp, "sp_layer_set_fd"); BIND(SP, pool_set, h, sp, "sp_pool_set");
    BIND(SP, host_set, h, sp, "sp_host_set"); BIND(SP, route_hook, h, sp, "sp_route_hook"); BIND(SP, pend_take, h, sp, "sp_pend_take"); BIND(SP, arr_wait, h, sp, "sp_arr_wait");
    BIND(SP, host_acquire, h, sp, "sp_host_acquire"); BIND(SP, get_stats, h, sp, "sp_get_stats");
    if (g_symfail) exit(2);
}

// ------------------------------------------------------------------------------------------------ child state: caches, jobs, model of the planner's bookkeeping
static UCfg C;
static Rng RD;
static rt_table *g_table = nullptr;
static sc_cache *g_cache[9];   // 0 = host, 1 + card = pool
static int host_n = 0;
struct Call { int layer = 0; std::vector<int> ex, cnt; };
static std::vector<Call> tab;
static unsigned long long g_nuc = 0;   // calls of the next-use function by the slot caches
static int cn(const sc_cache *c) { if (!c) return -1; for (int i = 0; i < 9; i++) if (g_cache[i] == c) return i; return -2; }
static int ekey(int layer, int e) { return layer * C.experts + e; }
static int card_of(int L) { return C.layout == 0 ? (L == 0 ? 0 : std::min(C.cards - 1, L * C.cards / 93)) : (L == 0 ? 0 : (L - 1) % C.cards); }
static int dev_of(int card) { return (card * 5 + 2) % 11; }
static int fd_of(int L) { return 1000 + L; }
static int pool_lo(int card) { return host_n + card * C.pool_slots; }

struct Job {
    uint32_t id = 0; cq_job j; bool k0 = false; int key = -1; int wbkey = -1;
    std::atomic<bool> completed{ false };
};
static uint32_t job_seq = 0;
static std::vector<Job *> outstanding;                    // sequential mode: jobs not yet completed (id order)
static std::unordered_map<int, Job *> pend_m, arr_m;      // model of g_pend (key -> latest entry) and g_arr
static std::vector<Job *> rel_m; static std::vector<int> wb_m;   // model of g_rel and g_wb_keys
static uint64_t n_k0, n_k1, n_k2, n_wb, n_try_ok, n_try_ok_vk, n_erase, n_pfw, n_vqw;
static std::mutex live_mu; static std::unordered_map<int *, uint32_t> live;   // state pointers of jobs that are not completed
static int last_try_key = -1, last_try_vk = -1;
static bool fd_set_m[93], ready_m[8]; static bool model_rc_ok = true; static unsigned long hooks_done = 0;
static uint64_t n_wb_skipped = 0, n_pool_hit = 0, n_from_host = 0, n_written = 0;

static void do_stores(Job *J)
{
    auto st = [&](int *p) { if (!p) return; { std::lock_guard<std::mutex> g(live_mu); live.erase(p); } __atomic_store_n(p, 1, __ATOMIC_RELEASE); };
    if (J->j.kind == 0) st(J->j.p_state);
    else { if (J->j.victim_host) st(J->j.wb_state); st(J->j.rel_state); st(J->j.arr_state); }
}
static void complete_sync(Job *J)
{
    if (J->completed.load()) return;
    do_stores(J); J->completed.store(true); LOG("dQ %u", J->id);
}
struct Helper { std::thread th; Job *j; };
static std::vector<Helper *> helpers;
static void start_async(Job *J)
{
    J->completed.store(true);   // the model's view; the stores follow from the helper thread
    const unsigned d = 3000 + 10 * RD.r(200);   // 3-5 ms: the planner must have read state 0 before the store, or the model's 'it waited' is wrong (40-240 us lost the race under load)
    Helper *h = new Helper; h->j = J; h->th = std::thread([J, d]() { usleep(d); do_stores(J); });
    helpers.push_back(h);
}
static void join_helpers()
{
    for (Helper *h : helpers) { h->th.join(); LOG("dQ %u h", h->j->id); delete h; }
    helpers.clear();
}

// P6: completions from a second thread
static std::mutex wmu; static std::condition_variable wcv; static std::deque<Job *> wq; static bool wstop = false; static std::atomic<int> winflight{ 0 };
static std::thread *wth = nullptr;   // heap: a static std::thread still joinable at exit() would terminate the process
static void worker_loop(uint64_t seed)
{
    Rng r(seed);
    for (;;) {
        Job *J;
        { std::unique_lock<std::mutex> lk(wmu); wcv.wait(lk, []() { return wstop || !wq.empty(); }); if (wq.empty()) return; J = wq.front(); wq.pop_front(); }
        usleep(r.r(100) < 90 ? r.r(150) : 300 + r.r(1500));
        do_stores(J); J->completed.store(true); LOG("dQ %u w", J->id);
        winflight.fetch_sub(1);
    }
}
static void drain_workers() { while (winflight.load() > 0) { usleep(50); alarm(120); } }

static void protsig(const int *p, size_t n, char *out)
{
    static std::vector<uint32_t> stamp; static uint32_t epoch = 0;
    const size_t NS = 1 << 20;
    if (stamp.empty()) stamp.assign(NS, 0);
    uint64_t h = 0; size_t u = 0; bool ok = true;
    for (size_t i = 0; i < n; i++) if (p[i] < 0 || (size_t)p[i] >= NS) { ok = false; break; }
    if (ok) { epoch++; for (size_t i = 0; i < n; i++) if (stamp[(size_t)p[i]] != epoch) { stamp[(size_t)p[i]] = epoch; h += mix64((uint64_t)(int64_t)p[i]); u++; } }
    else { std::vector<int> v(p, p + n); std::sort(v.begin(), v.end()); v.erase(std::unique(v.begin(), v.end()), v.end()); for (int k : v) h += mix64((uint64_t)(int64_t)k); u = v.size(); }
    snprintf(out, 40, "%zu:%016llx", u, (unsigned long long)h);
}
static std::string listsig(const int *p, size_t n)
{
    uint64_t h = 1469598103934665603ULL; for (size_t i = 0; i < n; i++) { h ^= (uint64_t)(uint32_t)p[i]; h *= 1099511628211ULL; }
    return sfmt("%zu:%016llx", n, (unsigned long long)h);
}

static int nu_fn(int key) { g_nuc++; return RT.next_use(g_table, key); }

// ---- the sp_ops wrappers (planner side): log every call, forward to the real function
static void w_set_next_use(sc_next_use_fn fn) { LOG("pX %s", fn ? "fn" : "NULL"); SC.set_next_use(fn); }
static int w_contains(const sc_cache *c, int key) { const int r = SC.contains(c, key); LOG("pC %d %d %d", cn(c), key, r); return r; }
static int w_worth(const sc_cache *c, int key) { const int r = SC.worth(c, key); LOG("pW %d %d %d", cn(c), key, r); return r; }
static unsigned char *w_acquire(sc_cache *c, int key, const int *prot, size_t np, float h0, int *vk, float *vh)
{
    char ps[40]; protsig(prot, np, ps);
    unsigned char *s = SC.acquire(c, key, prot, np, h0, vk, vh);
    const int v = *vk;
    LOG("pA %d %d %08x %s %d %d %08x", cn(c), key, fbits(h0), ps, slotid(s), v, v >= 0 ? fbits(*vh) : 0u);
    return s;
}
static unsigned char *w_try_acquire(sc_cache *c, int key, int cand, const int *prot, size_t np, float h0, int *vk, float *vh)
{
    char ps[40]; protsig(prot, np, ps);
    unsigned char *s = SC.try_acquire(c, key, cand, prot, np, h0, vk, vh);
    const int v = *vk;
    LOG("pT %d %d %d %08x %s %d %d %08x", cn(c), key, cand, fbits(h0), ps, slotid(s), v, v >= 0 ? fbits(*vh) : 0u);
    last_try_key = key; last_try_vk = s ? v : -1;
    if (s) {
        n_try_ok++;
        if (v >= 0) {   // the planner's arr_wait(vk) follows at once
            n_try_ok_vk++;
            auto it = arr_m.find(v);
            if (it != arr_m.end()) {
                Job *J = it->second;
                if (!C.threads && !J->completed.load()) { if (RD.p(C.p_spin)) { start_async(J); n_vqw++; } else complete_sync(J); }
                arr_m.erase(it);
            }
        }
    }
    return s;
}
static unsigned char *w_remove(sc_cache *c, int key)
{
    unsigned char *s = SC.remove(c, key);
    LOG("pR %d %d %d", cn(c), key, slotid(s));
    if (c == g_cache[0]) { auto it = pend_m.find(key); if (it != pend_m.end()) { pend_m.erase(it); n_erase++; } }   // pend_take(key) precedes the planner's host remove
    return s;
}
static void w_release(sc_cache *c, unsigned char *s) { LOG("pL %d %d", cn(c), slotid(s)); SC.release(c, s); }
static size_t w_rt_calls(const rt_table *t) { return RT.calls(t); }
static int w_rt_layer(const rt_table *t, size_t i) { return RT.layer(t, i); }
static const int *w_rt_experts(const rt_table *t, size_t i, size_t *n) { return RT.experts(t, i, n); }
static void w_rt_set_current(rt_table *t, long cur) { RT.set_current(t, cur); LOG("pU %ld", cur); flush_log(); }   // the mismatch text, if any, follows at once
static int w_rt_next_use(rt_table *t, int key) { const int r = RT.next_use(t, key); LOG("pN %d %d", key, r); return r; }

static std::string job_fields_error(Job *J, bool *k0out)
{
    const cq_job &j = J->j;
    auto in_host = [&](const void *p) { const int s = slotid(p); return s >= 0 && s < host_n; };
    auto in_pool = [&](const void *p, int card) { const int s = slotid(p); return s >= pool_lo(card) && s < pool_lo(card) + C.pool_slots; };
    const int L = j.fd - 1000;
    if (j.kind < 0 || j.kind > 2) return "kind out of range";
    if (L < 0 || L > 92) return "fd does not belong to a layer";
    if (C.no_handle[L]) return "job for a layer without a handle";
    if (!C.threads && !fd_set_m[L]) return "job for a layer whose fd was not set yet";
    int *st[4] = { j.p_state, j.wb_state, j.arr_state, j.rel_state };
    for (int a = 0; a < 4; a++) for (int b = a + 1; b < 4; b++) if (st[a] && st[a] == st[b]) return "two state pointers of one job are equal";
    if (j.kind == 0) {
        if (!j.p_state) return "kind 0 without p_state";
        if (!in_host(j.slot)) return "kind 0 slot not a host slot";
        if (j.wb_state || j.arr_state || j.rel_state || j.dst || j.host_src || j.victim_host) return "kind 0 with a field of kind 1/2";
        if (j.dev != 0 || j.layer != 0) return "kind 0 dev/layer not 0";
        if (j.off < 0 || j.off % C.expert_raw != 0 || j.off / C.expert_raw >= C.experts) return "kind 0 off is not e * expert_raw";
        J->key = ekey(L, (int)(j.off / C.expert_raw)); J->k0 = true; *k0out = true;
        return "";
    }
    const int card = card_of(j.layer);
    if (j.layer != L) return "kind 1/2 layer != layer of fd";
    if (j.p_state || j.slot) return "kind 1/2 with p_state or slot";
    if (!C.threads && !ready_m[card]) return "kind 1/2 job for a pool that is not ready";
    if (j.dev != dev_of(card)) return "kind 1/2 dev is not the pool's dev";
    if (!in_pool(j.dst, card)) return "kind 1/2 dst not a slot of the layer's pool";
    if (!j.arr_state) return "kind 1/2 without arr_state";
    if ((j.wb_state != nullptr) != (j.victim_host != nullptr)) return "wb_state / victim_host not both set or both NULL";
    if (j.victim_host && !in_host(j.victim_host)) return "victim_host not a host slot";
    if (j.kind == 1 && (j.host_src || j.rel_state)) return "kind 1 with host_src or rel_state";
    if (j.kind == 2 && (!in_host(j.host_src) || !j.rel_state)) return "kind 2 without host_src / rel_state";
    if (last_try_key < 0 || j.off < 0 || j.off % C.expert_raw != 0 || ekey(L, (int)(j.off / C.expert_raw)) != last_try_key) return "kind 1/2 off does not match the key of the last try_acquire";
    J->key = last_try_key; *k0out = false;
    return "";
}
static void w_enqueue(const cq_job *cj)
{
    Job *J = new Job; J->id = ++job_seq; J->j = *cj;
    bool k0 = false;
    const std::string err = job_fields_error(J, &k0);
    if (!err.empty()) viol(sfmt("job %u: %s", J->id, err.c_str()));
    std::string lerr;
    {
        std::lock_guard<std::mutex> g(live_mu);
        int *st[4] = { J->j.p_state, J->j.wb_state, J->j.arr_state, J->j.rel_state };
        for (int a = 0; a < 4; a++) if (st[a]) { if (live.count(st[a])) { lerr = sfmt("job %u: a state pointer is used by a live job (%u)", J->id, live[st[a]]); break; } live[st[a]] = J->id; }
    }
    if (!lerr.empty()) viol(lerr);
    char np[16], nw_[16], na[16], nr[16];
    snprintf(np, sizeof np, "J%up", J->id); snprintf(nw_, sizeof nw_, "J%uw", J->id); snprintf(na, sizeof na, "J%ua", J->id); snprintf(nr, sizeof nr, "J%ur", J->id);
    const cq_job &j = J->j;
    LOG("pE J%u k%d dev%d fd%d off%ld L%d s%d d%d hs%d vh%d p=%s wb=%s ar=%s rl=%s", J->id, j.kind, j.dev, j.fd, (long)j.off, j.layer, slotid(j.slot), slotid(j.dst), slotid(j.host_src),
        slotid(j.victim_host), j.p_state ? np : "-", j.wb_state ? nw_ : "-", j.arr_state ? na : "-", j.rel_state ? nr : "-");
    if (k0) { pend_m[J->key] = J; n_k0++; }
    else {
        if (j.kind == 1) n_k1++; else n_k2++;
        arr_m[J->key] = J;
        if (j.kind == 2) rel_m.push_back(J);
        if (j.victim_host) {   // the entry is stored under the victim of the try_acquire just before
            if (last_try_vk < 0) viol(sfmt("job %u: victim_host set but the last try_acquire had no victim", J->id));
            pend_m[last_try_vk] = J; wb_m.push_back(last_try_vk); n_wb++; J->wbkey = last_try_vk;
        }
    }
    if (C.threads) { winflight.fetch_add(1); { std::lock_guard<std::mutex> g(wmu); wq.push_back(J); } wcv.notify_one(); }
    else outstanding.push_back(J);
}

// ---- the driver side: its own calls into the slot caches and the planner (logged with a 'd' prefix)
static unsigned char *d_find(int ci, int key) { unsigned char *s = SC.find(g_cache[ci], key); LOG("dF %d %d %d", ci, key, slotid(s)); return s; }
static void d_touch(int ci, int key, float w) { SC.touch(g_cache[ci], key, w); LOG("dH %d %d %08x", ci, key, fbits(w)); }
static int d_worth(int ci, int key) { const int r = SC.worth(g_cache[ci], key); LOG("dW %d %d %d", ci, key, r); return r; }
static unsigned char *d_remove(int ci, int key) { unsigned char *s = SC.remove(g_cache[ci], key); LOG("dR %d %d %d", ci, key, slotid(s)); return s; }
static void d_release(int ci, unsigned char *s) { LOG("dL %d %d", ci, slotid(s)); SC.release(g_cache[ci], s); }
static unsigned char *d_acquire(int ci, int key, const std::vector<int> &prot, float h0, int *vk, float *vh)
{
    char ps[40]; protsig(prot.data(), prot.size(), ps);
    unsigned char *s = SC.acquire(g_cache[ci], key, prot.data(), prot.size(), h0, vk, vh);
    const int v = *vk;
    LOG("dA %d %d %08x %s %d %d %08x", ci, key, fbits(h0), ps, slotid(s), v, v >= 0 ? fbits(*vh) : 0u);
    return s;
}

// the planner's sweep, as the contract describes it, on the model
static void model_sweep()
{
    for (size_t i = 0; i < rel_m.size();) { if (rel_m[i]->completed.load()) { rel_m[i] = rel_m.back(); rel_m.pop_back(); } else i++; }
    for (size_t i = 0; i < wb_m.size();) {
        auto f = pend_m.find(wb_m[i]);
        if (f == pend_m.end()) { wb_m[i] = wb_m.back(); wb_m.pop_back(); }
        else if (f->second->completed.load()) { pend_m.erase(f); wb_m[i] = wb_m.back(); wb_m.pop_back(); }
        else i++;
    }
}

static void drv_pend_take(int key)
{
    LOG("dP %d", key);
    auto it = pend_m.find(key); bool spin = false;
    if (it != pend_m.end() && !C.threads) {
        Job *J = it->second;
        if (!J->completed.load()) { if (RD.p(C.p_spin)) { spin = true; start_async(J); } else complete_sync(J); }
    }
    SP.pend_take(key);
    if (spin) { join_helpers(); n_pfw++; }
    it = pend_m.find(key);
    if (it != pend_m.end()) { pend_m.erase(it); n_erase++; }
}
static void drv_arr_wait(int key)
{
    LOG("dV %d", key);
    auto it = arr_m.find(key); bool spin = false;
    if (it != arr_m.end() && !C.threads) {
        Job *J = it->second;
        if (!J->completed.load()) { if (RD.p(C.p_spin)) { spin = true; start_async(J); } else complete_sync(J); }
    }
    SP.arr_wait(key);
    if (spin) { join_helpers(); n_vqw++; }
    it = arr_m.find(key);
    if (it != arr_m.end()) arr_m.erase(it);
}
static void drv_host_acquire(int key, float heat)
{
    LOG("dHA %d %08x", key, fbits(heat));
    int vk = -1; float vh = 0;
    unsigned char *s = SP.host_acquire(key, heat, &vk, &vh);
    LOG("dHR %d %d %08x", slotid(s), vk, vk >= 0 ? fbits(vh) : 0u);
}

static void check_stats(const sp_stats &s)
{
    auto eq = [&](const char *what, unsigned long long got, unsigned long long want) {
        if (got != want) viol(sfmt("stats %s = %llu, model %llu (after %lu hooks)", what, got, want, hooks_done));
    };
    eq("pos", s.pos, hooks_done); eq("rc_ok", s.rc_ok != 0, model_rc_ok); eq("planned", s.planned, n_k0); eq("vq_ram", s.vq_ram, n_k2); eq("vq_drive", s.vq_drive, n_k1);
    eq("vq_wb", s.vq_wb, n_wb); eq("vq_wb+vq_nowb", s.vq_wb + s.vq_nowb, n_try_ok_vk); eq("vq_ram+vq_drive", s.vq_ram + s.vq_drive, n_try_ok); eq("arrivals", s.arrivals, arr_m.size());
    if (s.t_plan < 0 || s.t_pwait < 0 || s.t_vwait < 0) viol("a negative time in the stats");
    if (!C.threads) {
        eq("pending", s.pending, pend_m.size()); eq("releases", s.releases, rel_m.size()); eq("wb_keys", s.wb_keys, wb_m.size());
        eq("pf_used", s.pf_used, n_erase); eq("pf_waited", s.pf_waited, n_pfw); eq("vq_waited", s.vq_waited, n_vqw);
        if ((s.t_pwait > 0) != (n_pfw > 0) || (s.t_vwait > 0) != (n_vqw > 0)) viol("t_pwait / t_vwait are not > 0 exactly when a wait happened");
    } else if (s.pending > pend_m.size()) viol(sfmt("stats pending = %zu exceeds the model's superset %zu", s.pending, pend_m.size()));
}
static sp_stats step_stats()
{
    sp_stats s; memset(&s, 0, sizeof s); SP.get_stats(&s);
    LOGZ("Z pos=%zu to=%zu ok=%d pl=%lu sk=%lu vr=%lu vd=%lu vw=%lu vn=%lu pu=%lu pw=%lu vq=%lu tpw=%d tvw=%d tp0=%d pn=%zu ar=%zu rl=%zu wk=%zu nuc=%llu",
         s.pos, s.planned_to, s.rc_ok, s.planned, s.skipped, s.vq_ram, s.vq_drive, s.vq_wb, s.vq_nowb, s.pf_used, s.pf_waited, s.vq_waited, s.t_pwait > 0, s.t_vwait > 0, s.t_plan == 0,
         s.pending, s.arrivals, s.releases, s.wb_keys, g_nuc);
    check_stats(s);
    return s;
}

static void advance()
{
    std::vector<Job *> keep;
    for (Job *J : outstanding) {
        if (J->completed.load()) continue;
        if (RD.p(C.p_complete)) complete_sync(J); else keep.push_back(J);
    }
    outstanding.swap(keep);
}

// pool_ensure of decode.cuh (the calls it makes on the pool cache and the host cache, and into the planner), for one group of experts of one call
static void pool_ensure(int card, int layer, const std::vector<int> &experts, const std::vector<int> &cnt)
{
    const int pi = 1 + card;
    std::vector<int> prot; for (int e : experts) prot.push_back(ekey(layer, e));
    struct Need { int e; unsigned char *dst, *host; };
    struct Victim { int key; float heat; unsigned char *src; };
    std::vector<Need> need; std::vector<Victim> victims;
    for (int e : experts) {
        const int key = ekey(layer, e);
        unsigned char *p = d_find(pi, key);
        if (p) { if (C.vq) drv_arr_wait(key); d_touch(pi, key, (float)cnt[(size_t)e]); n_pool_hit++; continue; }
        int vk = -1; float vh = 0;
        p = d_acquire(pi, key, prot, (float)cnt[(size_t)e], &vk, &vh);
        if (vk >= 0) { if (C.vq) drv_arr_wait(vk); victims.push_back(Victim{ vk, vh, p }); }
        need.push_back(Need{ e, p, nullptr });
    }
    if (need.empty()) return;
    std::vector<unsigned char *> held;
    for (auto &n : need) {
        const int key = ekey(layer, n.e);
        if (d_find(0, key)) { drv_pend_take(key); n.host = d_remove(0, key); held.push_back(n.host); n_from_host++; }
    }
    for (auto &v : victims) {
        if (!C.host_cap || C.prefill) break;
        if (!d_worth(0, v.key)) { n_wb_skipped++; continue; }
        drv_host_acquire(v.key, v.heat); n_written++;
    }
    for (unsigned char *h : held) d_release(0, h);
}

static void fill_counts(const Call &a, std::vector<int> &by_e)
{
    by_e.assign((size_t)C.experts, 1);
    if (a.cnt.size() == a.ex.size()) for (size_t i = 0; i < a.ex.size(); i++) if (a.ex[i] >= 0 && a.ex[i] < C.experts) by_e[(size_t)a.ex[i]] = a.cnt[i];
}

static Call actual_call(long pos)
{
    Call a;
    if ((size_t)pos < tab.size()) a = tab[(size_t)pos];
    else {
        Rng r(mix64((uint64_t)C.seed * 31 + (uint64_t)pos)); a.layer = (int)r.r(93);
        const int n = (int)r.r(6); for (int i = 0; i < n; i++) a.ex.push_back((int)r.r((uint32_t)C.experts));
        a.cnt.assign((size_t)n, 1);
    }
    for (auto &m : C.mm) {
        if (m.first != pos) continue;
        Rng r(mix64((uint64_t)C.seed * 77 + (uint64_t)pos + (uint64_t)m.second));
        switch (m.second) {
        case 1: a.layer = (a.layer + 1 + (int)r.r(92)) % 93; break;
        case 2: if (!a.ex.empty()) { const size_t i = r.r((uint32_t)a.ex.size()); a.ex[i] = (a.ex[i] + 1 + (int)r.r((uint32_t)std::max(1, C.experts - 1))) % C.experts; } else a.ex.push_back((int)r.r((uint32_t)C.experts)); break;
        case 3: if (!a.ex.empty()) a.ex.pop_back(); else a.ex.push_back((int)r.r((uint32_t)C.experts)); break;
        case 4: a.ex.push_back((int)r.r((uint32_t)C.experts)); break;
        case 5: if (a.ex.size() >= 2) std::swap(a.ex[0], a.ex[1]); else a.ex.push_back((int)r.r((uint32_t)C.experts)); break;
        default: a.ex.clear(); break;
        }
        a.cnt.assign(a.ex.size(), 1 + (int)r.r(4));
    }
    return a;
}

static void drv_hook(const Call &a, long pos)
{
    const bool matches = model_rc_ok && (size_t)pos < tab.size() && tab[(size_t)pos].layer == a.layer && tab[(size_t)pos].ex == a.ex;
    LOG("dK %ld %d %s", pos, a.layer, listsig(a.ex.data(), a.ex.size()).c_str());
    if (matches && C.vq && pos >= C.vq_from) model_sweep();
    static const int dummy = 0;
    SP.route_hook(a.layer, a.ex.empty() ? &dummy : a.ex.data(), a.ex.size());
    join_helpers();
    if (model_rc_ok && !matches) model_rc_ok = false;
    hooks_done++;
}

static void on_alarm(int)
{
    if (g_logfd >= 0 && !g_buf.empty()) { const ssize_t w = write(g_logfd, g_buf.data(), g_buf.size()); (void)w; }
    static const char m[] = "TIMEOUT\n"; const ssize_t w = write(g_logfd >= 0 ? g_logfd : 2, m, sizeof m - 1); (void)w;
    _exit(4);
}
// exit() is called by the slot cache and the planner on their fatal paths while the P6 worker thread may still be running; running the static destructors then races the worker against
// the log buffer and the job queue (glibc "corrupted double-linked list" in the reference child of P6 seed 41). So: flush, then leave with the status the process was going to leave with.
static void at_exit_flush() { flush_log(); _exit(g_exit_code); }

static int run_child(const UCfg &cfg, const std::string &planner, const std::string &scp, const std::string &rtp, const std::string &logp, const std::string &trace)
{
    C = cfg;
    g_logfd = open(logp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (g_logfd < 0) { perror("open log"); return 2; }
    dup2(g_logfd, 2);   // the planner's stderr text lands in the same file
    atexit(at_exit_flush);
    signal(SIGALRM, on_alarm); alarm(120);
    load_libs(scp.c_str(), rtp.c_str(), planner.c_str());
    g_exit_code = 1;
    RD = Rng(mix64((uint64_t)C.seed * 2654435761ULL + 99));
    LOG("%s", cfg_str(C).c_str());

    g_table = RT.load(trace.c_str(), C.experts);
    const size_t nc = RT.calls(g_table);
    for (size_t i = 0; i < nc; i++) {
        Call a; a.layer = RT.layer(g_table, i);
        size_t n = 0; const int *e = RT.experts(g_table, i, &n); if (n) a.ex.assign(e, e + n);
        size_t m = 0; const int *k = RT.counts(g_table, i, &m); if (m) a.cnt.assign(k, k + m);
        tab.push_back(std::move(a));
    }
    host_n = C.host_slots;
    for (int ci = 0; ci <= C.cards; ci++) g_cache[ci] = SC.nw();
    SC.set_policy(C.policy); LOG("dY policy %d", C.policy);
    if (C.install_nu) { SC.set_next_use(nu_fn); LOG("dX fn"); }
    for (int ci = 0; ci <= C.cards; ci++) {
        const int n = ci == 0 ? host_n : C.pool_slots, lo = ci == 0 ? 0 : pool_lo(ci - 1);
        for (int i = n; i-- > 0;) { SC.add_spare(g_cache[ci], slotp(lo + i)); LOG("dS %d %d", ci, lo + i); }
    }
    model_rc_ok = !C.rc_force0 && !tab.empty();
    sp_config pc; memset(&pc, 0, sizeof pc);
    pc.experts = C.experts; pc.expert_raw = (size_t)C.expert_raw; pc.cards = C.cards; pc.lookahead = C.lookahead; pc.pend_cap = C.pend_cap; pc.vq = C.vq; pc.vq_cap = C.vq_cap;
    pc.vq_from = C.vq_from; pc.prefill = C.prefill; pc.rc_ok = model_rc_ok ? 1 : 0;
    sp_ops ops;
    memset(&ops, 0, sizeof ops);
    ops.sc_set_next_use = w_set_next_use; ops.sc_contains = w_contains; ops.sc_worth = w_worth; ops.sc_acquire = w_acquire; ops.sc_try_acquire = w_try_acquire; ops.sc_remove = w_remove;
    ops.sc_release = w_release; ops.table = g_table; ops.rt_calls = w_rt_calls; ops.rt_layer = w_rt_layer; ops.rt_experts = w_rt_experts; ops.rt_set_current = w_rt_set_current;
    ops.rt_next_use = w_rt_next_use; ops.enqueue = w_enqueue;
    SP.init(&ops, &pc); LOG("dY init");   // the wiring comes after sp_init: the reference's sp_init resets it
    for (int L = 0; L < 93; L++) {
        if (C.no_handle[L]) continue;
        const int fd = C.late_fd[L] == -1 ? fd_of(L) : -1;
        SP.layer_set(L, fd, card_of(L)); fd_set_m[L] = fd >= 0; LOG("dY ls %d %d %d", L, fd, card_of(L));
    }
    for (int k = 0; k < C.cards; k++) {
        if (C.ready_at[k] == -1) { SP.pool_set(k, g_cache[1 + k], dev_of(k), 1); ready_m[k] = true; LOG("dY pool %d 1", k); }
        else { SP.pool_set(k, nullptr, 0, 0); LOG("dY pool %d 0", k); }
    }
    SP.host_set(g_cache[0], (size_t)C.host_cap); LOG("dY host %ld", C.host_cap);

    std::thread *fdth = nullptr;
    if (C.threads) {
        wth = new std::thread(worker_loop, mix64((uint64_t)C.seed * 3 + 1));
        std::vector<int> late; for (int L = 0; L < 93; L++) if (!C.no_handle[L] && C.late_fd[L] >= 0) late.push_back(L);
        fdth = new std::thread([late]() { Rng r(mix64((uint64_t)C.seed * 5 + 1)); for (int L : late) { usleep(r.r(300)); SP.layer_set_fd(L, fd_of(L)); LOG("dY fd %d", L); } });
    }
    const long total = (long)tab.size() + C.extra_hooks;
    for (long pos = 0; pos < total; pos++) {
        alarm(120);
        if (C.threads && C.empty_tail && pos == total - 1) drain_workers();
        if (!C.threads) {
            for (int L = 0; L < 93; L++) if (!C.no_handle[L] && C.late_fd[L] == pos) { SP.layer_set_fd(L, fd_of(L)); fd_set_m[L] = true; LOG("dY fd %d", L); }
            for (int k = 0; k < C.cards; k++) if (C.ready_at[k] == pos) { SP.pool_set(k, g_cache[1 + k], dev_of(k), 1); ready_m[k] = true; LOG("dY pool %d 1", k); }
            advance();
        }
        const Call a = actual_call(pos);
        drv_hook(a, pos);
        step_stats();
        const int card = card_of(a.layer);
        if (C.threads ? true : ready_m[card]) {
            std::vector<int> cnt; fill_counts(a, cnt);
            for (size_t gi = 0; gi < a.ex.size(); gi += (size_t)C.pool_slots) {
                const std::vector<int> grp(a.ex.begin() + (long)gi, a.ex.begin() + (long)std::min(a.ex.size(), gi + (size_t)C.pool_slots));
                pool_ensure(card, a.layer, grp, cnt);
                step_stats();
            }
        }
    }
    if (fdth) { fdth->join(); }
    if (C.threads) { drain_workers(); { std::lock_guard<std::mutex> g(wmu); wstop = true; } wcv.notify_all(); wth->join(); }
    else { for (Job *J : outstanding) if (!J->completed.load()) complete_sync(J); outstanding.clear(); }
    join_helpers();
    const sp_stats fs = step_stats();
    if (!live.empty()) viol(sfmt("%zu state pointers still live after every job completed", live.size()));
    if (C.threads && C.vq && C.vq_from <= total - 1 && model_rc_ok) {
        if (fs.releases != 0 || fs.wb_keys != 0) viol(sfmt("after draining and one sweep: releases=%zu wb_keys=%zu (expected 0, 0)", fs.releases, fs.wb_keys));
    }
    LOGZ("END jobs=%u k0=%llu k1=%llu k2=%llu wb=%llu hit=%llu fromhost=%llu written=%llu wbskip=%llu", job_seq, (unsigned long long)n_k0, (unsigned long long)n_k1, (unsigned long long)n_k2,
         (unsigned long long)n_wb, (unsigned long long)n_pool_hit, (unsigned long long)n_from_host, (unsigned long long)n_written, (unsigned long long)n_wb_skipped);
    flush_log();
    g_exit_code = 0;
    return 0;
}

// ------------------------------------------------------------------------------------------------ parent: units and runs
struct Opts {
    std::string mode, scn, ref, rust, sc, rt, out, preload, planner, log, trace;
    std::string p3trace = "/home/usergpu/k3gpu/results/p1d.trace";
    int seed = 0, jobs = 16, keep_raw = 2; bool keep_going = false;
    std::map<std::string, int> n; std::vector<std::string> scns;
};
static std::string g_self;
static void mkdir_p(const std::string &p)
{
    std::string cur;
    for (size_t i = 0; i <= p.size(); i++) { if (i == p.size() || p[i] == '/') { if (!cur.empty()) mkdir(cur.c_str(), 0755); } if (i < p.size()) cur.push_back(p[i]); }
}
static pid_t spawn_child(const Opts &o, const std::string &planner, const std::string &log, const std::string &trace, bool is_ref)
{
    std::vector<std::string> a = { g_self, "child", "--scn", o.scn, "--seed", std::to_string(o.seed), "--planner", planner, "--sc", o.sc, "--rt", o.rt, "--log", log, "--trace", trace, "--p3trace", o.p3trace };
    const pid_t p = fork();
    if (p == 0) {
        if (is_ref && !o.preload.empty()) { setenv("LD_PRELOAD", o.preload.c_str(), 1); setenv("ASAN_OPTIONS", "detect_leaks=0", 1); setenv("UBSAN_OPTIONS", "print_stacktrace=0", 1); }
        const int dn = open("/dev/null", O_RDWR); dup2(dn, 0); dup2(dn, 1); dup2(dn, 2);
        std::vector<char *> av; for (auto &s : a) av.push_back((char *)s.c_str()); av.push_back(nullptr);
        execv(g_self.c_str(), av.data()); _exit(127);
    }
    return p;
}
static std::string status_str(int st)
{
    if (WIFEXITED(st)) return sfmt("exit%d", WEXITSTATUS(st));
    if (WIFSIGNALED(st)) return sfmt("sig%d", WTERMSIG(st));
    return "unknown";
}
static void count_file(const std::string &p, uint64_t &lines, uint64_t &z)
{
    lines = z = 0; FILE *f = fopen(p.c_str(), "rb"); if (!f) return;
    std::vector<char> b(1 << 20); char prev = '\n'; size_t n;
    while ((n = fread(b.data(), 1, b.size(), f)) > 0) for (size_t i = 0; i < n; i++) { const char c = b[i]; if (c == '\n') lines++; if (prev == '\n' && c == 'Z') z++; prev = c; }
    fclose(f);
}
static bool same_files(const std::string &a, const std::string &b, uint64_t &off)
{
    FILE *fa = fopen(a.c_str(), "rb"), *fb = fopen(b.c_str(), "rb"); off = 0;
    if (!fa || !fb) { if (fa) fclose(fa); if (fb) fclose(fb); return false; }
    std::vector<char> x(1 << 20), y(1 << 20); bool same = true;
    for (;;) {
        const size_t nx = fread(x.data(), 1, x.size(), fa), ny = fread(y.data(), 1, y.size(), fb);
        const size_t m = std::min(nx, ny);
        size_t i = 0; while (i < m && x[i] == y[i]) i++;
        if (i < m) { off += i; same = false; break; }
        off += m;
        if (nx != ny) { same = false; break; }
        if (nx == 0) break;
    }
    fclose(fa); fclose(fb); return same;
}
static std::string diff_report(const std::string &a, const std::string &b, uint64_t off)
{
    std::string out;
    for (int w = 0; w < 2; w++) {
        const std::string &p = w ? b : a;
        FILE *f = fopen(p.c_str(), "rb"); if (!f) continue;
        uint64_t line = 0, pos = 0; int c;
        while (pos < off && (c = fgetc(f)) != EOF) { if (c == '\n') line++; pos++; }
        fclose(f);
        std::vector<std::string> ctx; FILE *g = fopen(p.c_str(), "rb"); char *buf = nullptr; size_t cap = 0; uint64_t ln = 0; ssize_t n;
        const uint64_t from = line >= 5 ? line - 5 : 0;
        out += sfmt("--- %s: first difference at byte %llu, line %llu (1-based %llu)\n", p.c_str(), (unsigned long long)off, (unsigned long long)line, (unsigned long long)line + 1);
        while ((n = getline(&buf, &cap, g)) > 0) {
            if (ln >= from && ln <= line + 3) { std::string s(buf, (size_t)n); if (s.size() > 400) s = s.substr(0, 400) + "...\n"; out += sfmt("%s%llu: %s", ln == line ? ">>" : "  ", (unsigned long long)ln + 1, s.c_str()); }
            if (ln > line + 3) break;
            ln++;
        }
        free(buf); fclose(g);
    }
    return out;
}

// the wait counters (pf_waited, vq_waited, t_pwait, t_vwait) are checked against a model that assumes the helper thread is still asleep when the planner looks at its state; a helper starved
// for longer than its 3-5 ms delay (loaded box) makes a correct planner count one wait fewer. Only that violation is retried, once; both attempts are kept and the retry is reported.
static bool wait_counter_violation(const std::string &log)
{
    FILE *lf = fopen(log.c_str(), "rb"); if (!lf) return false;
    std::string all; char buf[65536]; size_t k; while ((k = fread(buf, 1, sizeof buf, lf)) > 0) all.append(buf, k); fclose(lf);
    return all.find("VIOLATION stats pf_waited") != std::string::npos || all.find("VIOLATION stats vq_waited") != std::string::npos || all.find("VIOLATION t_pwait / t_vwait") != std::string::npos;
}

static int do_unit(const Opts &o, const std::string &retried_after = std::string())
{
    UCfg c = make_cfg(o.scn, o.seed, o.p3trace);
    const std::string dir = o.out + "/" + o.scn; mkdir_p(dir);
    const std::string pre = dir + "/u" + std::to_string(o.seed);
    std::string trace = c.real_trace ? c.trace_path : pre + ".trace";
    if (!c.real_trace) { FILE *f = fopen(trace.c_str(), "wb"); if (!f) { perror("trace"); return 2; } const std::string t = gen_trace(c); fwrite(t.data(), 1, t.size(), f); fclose(f); }
    struct R { std::string lib, log; bool isref; pid_t pid; int st; };
    std::vector<R> rs;
    if (o.scn == "P6") rs = { { o.ref, pre + ".ref1.log", true, 0, 0 }, { o.ref, pre + ".ref2.log", true, 0, 0 }, { o.rust, pre + ".rust1.log", false, 0, 0 }, { o.rust, pre + ".rust2.log", false, 0, 0 } };
    else rs = { { o.ref, pre + ".ref.log", true, 0, 0 }, { o.rust, pre + ".rust.log", false, 0, 0 } };
    for (auto &r : rs) r.pid = spawn_child(o, r.lib, r.log, trace, r.isref);
    for (auto &r : rs) { int st = 0; waitpid(r.pid, &st, 0); r.st = st; }
    bool pass = true; std::string reason = "-", extra;
    uint64_t la = 0, za = 0, lb = 0, zb = 0;
    count_file(rs[0].log, la, za);
    if (o.scn == "P6") {
        count_file(rs[2].log, lb, zb);
        for (auto &r : rs) if (!(WIFEXITED(r.st) && WEXITSTATUS(r.st) == 0)) { pass = false; reason = sfmt("%s: %s", r.log.c_str(), status_str(r.st).c_str()); break; }
        if (!pass) {   // the slot cache ends the process with its own message when every slot is in use; if all four children did, that is the scenario's outcome, not a difference
            int n_fatal = 0;
            for (auto &r : rs) { FILE *lf = fopen(r.log.c_str(), "rb"); if (!lf) continue; std::string all; char buf[65536]; size_t k; while ((k = fread(buf, 1, sizeof buf, lf)) > 0) all.append(buf, k); fclose(lf); if (all.find("every slot is in use by one call") != std::string::npos && WIFEXITED(r.st) && WEXITSTATUS(r.st) == 1) n_fatal++; }
            if (n_fatal == 4) { pass = true; reason = "-"; extra = "all_four_children_ended_in_the_slot_cache_fatal "; }
        }
        uint64_t off1 = 0, off2 = 0;
        const bool dr = same_files(rs[0].log, rs[1].log, off1), du = same_files(rs[2].log, rs[3].log, off2);
        extra += sfmt("nondet_ref=%d nondet_rust=%d", dr ? 0 : 1, du ? 0 : 1);
    } else {
        count_file(rs[1].log, lb, zb);
        uint64_t off = 0;
        const bool same = same_files(rs[0].log, rs[1].log, off);
        const std::string s0 = status_str(rs[0].st), s1 = status_str(rs[1].st);
        if (!same) { pass = false; reason = "logs differ"; const std::string d = diff_report(rs[0].log, rs[1].log, off); FILE *f = fopen((pre + ".diff").c_str(), "wb"); if (f) { fwrite(d.data(), 1, d.size(), f); fclose(f); } }
        else if (s0 != s1) { pass = false; reason = "status differs " + s0 + " vs " + s1; }
        else if (s0 != "exit0" && s0 != "exit1") { pass = false; reason = "both children ended with " + s0; }
        extra = "ref=" + s0 + " rust=" + s1;
        if (!pass && retried_after.empty() && (wait_counter_violation(rs[0].log) || wait_counter_violation(rs[1].log))) {
            for (auto &r : rs) { const std::string cmd = "mv -f '" + r.log + "' '" + r.log + ".try1'"; if (system(cmd.c_str()) != 0) fprintf(stderr, "mv failed\n"); }
            printf("UNIT %s %d RETRY first attempt: %s (logs kept as *.log.try1)\n", o.scn.c_str(), o.seed, reason.c_str()); fflush(stdout);
            return do_unit(o, reason);
        }
        if (!retried_after.empty()) extra += " retried_after_wait_counter_violation=1";
    }
    {
        FILE *f = fopen((pre + ".res").c_str(), "wb");
        if (f) { fprintf(f, "%s %s %d %llu %llu %llu %llu %s %s\n", pass ? "PASS" : "FAIL", o.scn.c_str(), o.seed, (unsigned long long)la, (unsigned long long)za, (unsigned long long)lb, (unsigned long long)zb, extra.c_str(), reason.c_str()); fclose(f); }
    }
    if (pass && o.seed >= o.keep_raw) for (auto &r : rs) { const std::string cmd = "gzip -1 -f '" + r.log + "'"; if (system(cmd.c_str()) != 0) fprintf(stderr, "gzip failed\n"); }
    printf("UNIT %s %d %s lines=%llu/%llu %s %s\n", o.scn.c_str(), o.seed, pass ? "PASS" : "FAIL", (unsigned long long)la, (unsigned long long)lb, extra.c_str(), reason.c_str());
    return pass ? 0 : 1;
}

static int do_run(Opts o)
{
    if (o.scns.empty()) o.scns = { "P3", "P2", "P4", "P1", "P5", "P6" };
    std::map<std::string, int> dflt = { { "P1", 500 }, { "P2", 500 }, { "P3", 36 }, { "P4", 500 }, { "P5", 600 }, { "P6", 100 } };
    for (auto &kv : dflt) if (!o.n.count(kv.first)) o.n[kv.first] = kv.second;
    struct Spec { std::string scn; int seed; };
    std::vector<Spec> specs;
    for (auto &s : o.scns) for (int i = 0; i < o.n[s]; i++) specs.push_back({ s, s == "P3" ? i : i + 1 });
    mkdir_p(o.out);
    const time_t t0 = time(nullptr);
    std::map<pid_t, Spec> act; size_t next = 0, done = 0; int fails = 0; bool stop = false;
    while (next < specs.size() || !act.empty()) {
        while (!stop && next < specs.size() && (int)act.size() < o.jobs) {
            const Spec sp = specs[next++];
            const pid_t p = fork();
            if (p == 0) {
                std::vector<std::string> a = { g_self, "unit", "--scn", sp.scn, "--seed", std::to_string(sp.seed), "--ref", o.ref, "--rust", o.rust, "--sc", o.sc, "--rt", o.rt, "--out", o.out,
                                               "--p3trace", o.p3trace, "--keep-raw", std::to_string(o.keep_raw) };
                if (!o.preload.empty()) { a.push_back("--preload"); a.push_back(o.preload); }
                const int fd = open((o.out + "/units.log").c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644); dup2(fd, 1); dup2(fd, 2);
                std::vector<char *> av; for (auto &s : a) av.push_back((char *)s.c_str()); av.push_back(nullptr);
                execv(g_self.c_str(), av.data()); _exit(127);
            }
            act[p] = sp;
        }
        if (act.empty()) break;
        int st = 0; const pid_t p = waitpid(-1, &st, 0);
        if (p < 0) break;
        auto it = act.find(p); if (it == act.end()) continue;
        done++;
        if (!(WIFEXITED(st) && WEXITSTATUS(st) == 0)) { fails++; printf("FAIL unit %s %d (%s)\n", it->second.scn.c_str(), it->second.seed, status_str(st).c_str()); fflush(stdout); if (!o.keep_going) stop = true; }
        act.erase(it);
        if (done % 100 == 0) { printf("[%zu/%zu done, %d fail, %lds]\n", done, specs.size(), fails, (long)(time(nullptr) - t0)); fflush(stdout); }
    }
    // summary from the per-unit result files
    std::string summary; int rc = fails ? 1 : 0;
    for (auto &s : o.scns) {
        int units = 0, pass = 0, fail = 0, exit1 = 0, nd_ref = 0, nd_rust = 0, retried = 0; unsigned long long la = 0, za = 0, lb = 0, zb = 0;
        for (int i = 0; i < o.n[s]; i++) {
            const int seed = s == "P3" ? i : i + 1;
            FILE *f = fopen((o.out + "/" + s + "/u" + std::to_string(seed) + ".res").c_str(), "rb"); if (!f) continue;
            char line[1024]; if (!fgets(line, sizeof line, f)) { fclose(f); continue; } fclose(f);
            char res[16], scn[16]; int sd; unsigned long long a1, a2, b1, b2;
            if (sscanf(line, "%15s %15s %d %llu %llu %llu %llu", res, scn, &sd, &a1, &a2, &b1, &b2) != 7) continue;
            units++; if (!strcmp(res, "PASS")) pass++; else fail++;
            la += a1; za += a2; lb += b1; zb += b2;
            if (strstr(line, "ref=exit1")) exit1++;
            if (strstr(line, "retried_after_wait_counter_violation=1")) retried++;
            if (strstr(line, "nondet_ref=1")) nd_ref++;
            if (strstr(line, "nondet_rust=1")) nd_rust++;
        }
        const unsigned long long opa = la - za - 2ULL * units, opb = lb - zb - 2ULL * units;   // minus the Z lines and the CFG / END line of each log
        std::string l = sfmt("%s units=%d pass=%d fail=%d logged_ops_ref=%llu logged_ops_rust=%llu (lines %llu/%llu)", s.c_str(), units, pass, fail, opa, opb, la, lb);
        if (s == "P6") l += sfmt(" nondeterministic_pairs ref=%d rust=%d (of %d)", nd_ref, nd_rust, units); else l += sfmt(" units_ending_in_fatal_exit1=%d", exit1);
        l += sfmt(" units_retried_after_wait_counter_violation=%d", retried);
        l += (fail == 0 && units == o.n[s]) ? " PASS" : " FAIL";
        if (!(fail == 0 && units == o.n[s])) rc = 1;
        summary += l + "\n";
    }
    summary += sfmt("wall %lds\n", (long)(time(nullptr) - t0));
    printf("%s", summary.c_str());
    FILE *f = fopen((o.out + "/summary.txt").c_str(), "wb"); if (f) { fputs(summary.c_str(), f); fclose(f); }
    return rc;
}

int main(int argc, char **argv)
{
    char buf[4096]; const ssize_t n = readlink("/proc/self/exe", buf, sizeof buf - 1); g_self = n > 0 ? std::string(buf, (size_t)n) : argv[0];
    if (argc < 2) { fprintf(stderr, "usage: pl_driver run|unit|child --ref LIB --rust LIB [--sc LIB --rt LIB --out DIR --n P1=500,... --scn P1,P2 --jobs N --preload LIBASAN --keep-going]\n"); return 2; }
    Opts o; o.mode = argv[1];
    for (int i = 2; i < argc; i++) {
        const std::string k = argv[i]; auto val = [&]() -> std::string { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", k.c_str()); exit(2); } return argv[++i]; };
        if (k == "--scn") { const std::string v = val(); if (o.mode == "run") { size_t p = 0; while (p <= v.size()) { size_t q = v.find(',', p); if (q == std::string::npos) q = v.size(); if (q > p) o.scns.push_back(v.substr(p, q - p)); p = q + 1; } } else o.scn = v; }
        else if (k == "--seed") o.seed = atoi(val().c_str());
        else if (k == "--ref") o.ref = val(); else if (k == "--rust") o.rust = val(); else if (k == "--sc") o.sc = val(); else if (k == "--rt") o.rt = val();
        else if (k == "--out") o.out = val(); else if (k == "--preload") o.preload = val(); else if (k == "--planner") o.planner = val(); else if (k == "--log") o.log = val();
        else if (k == "--trace") o.trace = val(); else if (k == "--p3trace") o.p3trace = val(); else if (k == "--jobs") o.jobs = atoi(val().c_str());
        else if (k == "--keep-raw") o.keep_raw = atoi(val().c_str()); else if (k == "--keep-going") o.keep_going = true;
        else if (k == "--n") { const std::string v = val(); size_t p = 0; while (p < v.size()) { size_t q = v.find(',', p); if (q == std::string::npos) q = v.size(); const std::string t = v.substr(p, q - p); const size_t e = t.find('='); if (e != std::string::npos) o.n[t.substr(0, e)] = atoi(t.c_str() + e + 1); p = q + 1; } }
        else { fprintf(stderr, "unknown option %s\n", k.c_str()); return 2; }
    }
    if (o.mode == "child") { const UCfg c = make_cfg(o.scn, o.seed, o.p3trace); return run_child(c, o.planner, o.sc, o.rt, o.log, o.trace); }
    if (o.mode == "unit") return do_unit(o);
    if (o.mode == "run") return do_run(o);
    fprintf(stderr, "unknown mode %s\n", o.mode.c_str()); return 2;
}
