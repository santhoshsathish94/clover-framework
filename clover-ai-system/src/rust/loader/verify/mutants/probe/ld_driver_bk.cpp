// Differential driver for loader.h: the same scenario, seeded, runs against libloader_ref.so (the engine's text) and libloader.so (Rust); the lines that start with "C " are
// canonical (they must be byte-identical between the two), "F " lines are failed checks, "N " lines are notes (timings, informational counts), RESULT is the verdict.
// Every CUDA effect is a recording callback. open() and pread() are interposed (this executable exports them, -rdynamic) so the file system's answers can be forced: refuse O_DIRECT,
// short reads, an error at the n-th read; the interposers log every call on an experts.direct file.
// Usage: ld_driver <label ref|rust> <lib.so> <scenario> <seed>
// Scenarios: fifo barrier wait stress race open shm natural real fatal init2   (what each checks and cannot check: RESULTS.md)
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <poll.h>
#include <random>
#include <set>
#include <string>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>
#include "loader.h"

typedef long long ll;
static std::string g_label, g_scn;
static std::atomic<int> g_checks(0), g_failed(0);
static std::mutex g_pm;
static std::string g_ds;   // dataset directory of this process: replaced by <DS> in canonical lines

static std::string norm(std::string s)
{
    if (g_ds.empty()) return s;
    for (size_t p = 0; (p = s.find(g_ds, p)) != std::string::npos;) { s.replace(p, g_ds.size(), "<DS>"); p += 4; }
    return s;
}
static std::string esc(const std::string &s)
{
    std::string o;
    for (unsigned char c : s) {
        if (c == '\n') o += "\\n"; else if (c == '\\') o += "\\\\"; else if (c < 32 || c > 126) { char b[8]; snprintf(b, sizeof b, "\\x%02x", c); o += b; } else o += (char)c;
    }
    return o;
}
static void emit(const char *fmt, ...)
{
    char *b = nullptr; va_list ap; va_start(ap, fmt); if (vasprintf(&b, fmt, ap) < 0) b = nullptr; va_end(ap);
    std::lock_guard<std::mutex> g(g_pm); printf("C %s\n", b ? norm(b).c_str() : "?"); fflush(stdout); free(b);
}
static void note(const char *fmt, ...)
{
    char *b = nullptr; va_list ap; va_start(ap, fmt); if (vasprintf(&b, fmt, ap) < 0) b = nullptr; va_end(ap);
    std::lock_guard<std::mutex> g(g_pm); printf("N %s\n", b ? b : "?"); fflush(stdout); free(b);
}
static void fail_msg(int line, const char *fmt, ...)
{
    char *b = nullptr; va_list ap; va_start(ap, fmt); if (vasprintf(&b, fmt, ap) < 0) b = nullptr; va_end(ap);
    std::lock_guard<std::mutex> g(g_pm); printf("F line %d: %s\n", line, b ? norm(b).c_str() : "?"); fflush(stdout); free(b);
}
#define CK(c, ...) do { g_checks++; if (!(c)) { g_failed++; fail_msg(__LINE__, __VA_ARGS__); } } while (0)
static void finish(int rc, bool hard = true)
{
    {
        std::lock_guard<std::mutex> g(g_pm);
        printf("RESULT %s %s %s checks=%d failed=%d\n", g_label.c_str(), g_scn.c_str(), g_failed ? "FAIL" : "PASS", g_checks.load(), g_failed.load());
        fflush(stdout);
    }
    if (hard) _exit(rc);
    exit(rc);   // the normal way out, with the loader's idle detached threads alive: must not hang or crash
}
static void hard_fail(int line, const char *what)
{
    g_checks++; g_failed++; fail_msg(line, "%s", what); finish(1);
}

struct Rng {
    std::mt19937_64 g;
    explicit Rng(uint64_t s) : g(s * 0x9E3779B97F4A7C15ull + 0x1234567ull) {}
    uint64_t below(uint64_t n) { return n ? g() % n : 0; }
    bool chance(int a, int b) { return (int)below((uint64_t)b) < a; }
    template <class T> T pick(std::initializer_list<T> l) { return *(l.begin() + below(l.size())); }
};
static ll now_ms() { return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static bool spin_until(const std::function<bool()> &p, int ms)
{
    const ll t0 = now_ms();
    while (!p()) { if (now_ms() - t0 > ms) return false; usleep(200); }
    return true;
}

// ------------------------------------------------------------------ the library under test
static struct Api { void (*init)(const ld_config *); void (*submit)(int, void (*)(void *), void *); void (*wait)(void); void (*open_layer)(const ld_layer *); } api;
static void load_lib(const char *path)
{
    void *h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "dlopen %s: %s\n", path, dlerror()); exit(2); }
    api.init = (void (*)(const ld_config *))dlsym(h, "ld_init");
    api.submit = (void (*)(int, void (*)(void *), void *))dlsym(h, "ld_submit");
    api.wait = (void (*)(void))dlsym(h, "ld_wait");
    api.open_layer = (void (*)(const ld_layer *))dlsym(h, "ld_open_layer");
    if (!api.init || !api.submit || !api.wait || !api.open_layer) { fprintf(stderr, "missing symbol in %s\n", path); exit(2); }
}

// ------------------------------------------------------------------ interposed open / pread
static std::atomic<int> g_open_mode(0);        // 0 pass through, 1 refuse O_DIRECT (EINVAL), 2 refuse every experts.direct open (ENOENT)
static std::atomic<long> g_pread_cap(0);       // > 0: no tracked pread returns more than this
static std::atomic<long> g_pread_fail_at(0);   // n > 0: the n-th tracked pread fails
static std::atomic<int> g_pread_fail_errno(EINTR);
static std::atomic<long> g_open_hook_calls(0), g_pread_hook_calls(0), g_pread_count(0);
struct OpenRec { std::string path; int flags; int ret; int err; };
struct PreadRec { std::string path; size_t req; long off; long ret; bool operator==(const PreadRec &o) const { return path == o.path && req == o.req && off == o.off && ret == o.ret; } };
static std::mutex g_io_m;
static std::vector<OpenRec> g_opens;
static std::vector<PreadRec> g_preads;
static std::map<int, std::string> g_fdpath;
static bool ends_with(const char *s, const char *suf) { size_t a = strlen(s), b = strlen(suf); return a >= b && !strcmp(s + a - b, suf); }

static int open_hook(const char *path, int flags, mode_t mode)
{
    g_open_hook_calls++;
    const bool track = ends_with(path, "experts.direct");
    int ret, err = 0;
    if (track && g_open_mode == 2) { ret = -1; err = ENOENT; }
    else if (track && g_open_mode == 1 && (flags & O_DIRECT)) { ret = -1; err = EINVAL; }
    else { ret = (int)syscall(SYS_openat, AT_FDCWD, path, flags, mode); err = errno; }
    if (track) {
        std::lock_guard<std::mutex> g(g_io_m);
        g_opens.push_back(OpenRec{ path, flags, ret, err });
        if (ret >= 0) g_fdpath[ret] = path;
    }
    errno = err;
    return ret;
}
static ssize_t pread_hook(int fd, void *buf, size_t n, off_t off)
{
    g_pread_hook_calls++;
    std::string path;
    { std::lock_guard<std::mutex> g(g_io_m); auto it = g_fdpath.find(fd); if (it != g_fdpath.end()) path = it->second; }
    if (path.empty()) return (ssize_t)syscall(SYS_pread64, fd, buf, n, off);
    const size_t req = n;
    const long cap = g_pread_cap;
    if (cap > 0 && n > (size_t)cap) n = (size_t)cap;
    const long idx = ++g_pread_count;
    ssize_t r; int err = 0;
    if (g_pread_fail_at && idx == g_pread_fail_at) { r = -1; err = g_pread_fail_errno; }
    else { r = (ssize_t)syscall(SYS_pread64, fd, buf, n, off); err = errno; }
    { std::lock_guard<std::mutex> g(g_io_m); g_preads.push_back(PreadRec{ path, req, (long)off, (long)r }); }
    errno = err;
    return r;
}
extern "C" {
int open(const char *path, int flags, ...)
{
    mode_t mode = 0;
    if (flags & (O_CREAT | O_TMPFILE)) { va_list ap; va_start(ap, flags); mode = va_arg(ap, mode_t); va_end(ap); }
    return open_hook(path, flags, mode);
}
int open64(const char *path, int flags, ...)
{
    mode_t mode = 0;
    if (flags & (O_CREAT | O_TMPFILE)) { va_list ap; va_start(ap, flags); mode = va_arg(ap, mode_t); va_end(ap); }
    return open_hook(path, flags, mode);
}
ssize_t pread(int fd, void *buf, size_t n, off_t off) { return pread_hook(fd, buf, n, off); }
ssize_t pread64(int fd, void *buf, size_t n, off64_t off) { return pread_hook(fd, buf, n, (off_t)off); }
}
static ssize_t real_pread(int fd, void *buf, size_t n, long off) { return (ssize_t)syscall(SYS_pread64, fd, buf, n, off); }

// ------------------------------------------------------------------ recording callbacks
struct Ev { int card; int tid; std::string s; };
static std::mutex g_em;
static std::vector<Ev> g_ev;
static std::atomic<int> g_tid_next(1);
static int cur_tid() { static thread_local int t = g_tid_next++; return t; }
static void ev(int card, const std::string &s) { std::lock_guard<std::mutex> g(g_em); g_ev.push_back(Ev{ card, cur_tid(), s }); }

struct LayerRec { unsigned layer = 0; int card = 0; unsigned k = 0, positions = 0; int L = -1, C = -1, K = -1, fd = 424242; };
static unsigned char *g_slab[93 + 16];     // the contract's 93 entries, then 16 canaries that a store past the tables would overwrite
static void *g_handles[93 + 16];
static void set_canaries() { for (int i = 93; i < 93 + 16; i++) { g_slab[i] = (unsigned char *)(uintptr_t)(0xBEEF0000u + (unsigned)i); g_handles[i] = (void *)(uintptr_t)(0xD00D0000u + (unsigned)i); } }
static void check_canaries() { for (int i = 93; i < 93 + 16; i++) { CK(g_slab[i] == (unsigned char *)(uintptr_t)(0xBEEF0000u + (unsigned)i), "slab[%d] was written", i); CK(g_handles[i] == (void *)(uintptr_t)(0xD00D0000u + (unsigned)i), "handles[%d] was written", i); } }
static int g_magic;                  // ctx identity
static std::atomic<int> g_ctx_bad(0);
static std::atomic<bool> g_hold(false);   // while set, every job waits in ensure_work: lets a scenario change the caller's buffers while jobs are queued but not started
static std::mutex g_bm; static std::condition_variable g_bc; static int g_bn = 0, g_bparties = 0; static bool g_bbroken = false;   // barrier used by the trunk callback
static std::set<int> g_barrier_done;
static bool g_trunk_barrier = false;

static void barrier_arrive()
{
    std::unique_lock<std::mutex> u(g_bm);
    g_bn++;
    if (g_bn >= g_bparties) g_bc.notify_all();
    const auto dl = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (g_bn < g_bparties && !g_bbroken) if (g_bc.wait_until(u, dl) == std::cv_status::timeout) { g_bbroken = true; g_bc.notify_all(); }
}
static void chk_ctx(void *ctx) { if (ctx != &g_magic) g_ctx_bad++; }
static void cb_set_device(void *ctx, int card) { chk_ctx(ctx); ev(card, "set_device"); }
static void cb_ensure_work(void *ctx, int card) { chk_ctx(ctx); while (g_hold) usleep(100); ev(card, "ensure_work"); }
static void cb_trunk(void *ctx, void *h, const char *ds, int L)
{
    chk_ctx(ctx); LayerRec *r = (LayerRec *)h;
    CK(r->L == (int)r->layer && (r->C == r->card || r->C == (r->card + 1) % 8) && r->K == (int)r->k, "layer %u: bookkeeping not written when the job started (L=%d card=%d k_in=%d)", r->layer, r->L, r->C, r->K);
    if (r->layer <= 92) CK(g_handles[r->layer] == r, "layer %u: handles[] entry not written when the job started", r->layer);
    ev(r->card, "trunk L" + std::to_string(L) + " ds=" + norm(ds));
    if (g_trunk_barrier) { bool first; { std::lock_guard<std::mutex> g(g_bm); first = g_barrier_done.insert(r->card).second; } if (first) barrier_arrive(); }
}
static void cb_tables(void *ctx, void *h, const char *ds, int L) { chk_ctx(ctx); (void)ds; ev(((LayerRec *)h)->card, "tables L" + std::to_string(L)); }
static void cb_state(void *ctx, void *h, unsigned positions) { chk_ctx(ctx); LayerRec *r = (LayerRec *)h; ev(r->card, "state L" + std::to_string((int)r->layer) + " pos=" + std::to_string(positions)); }
static LayerRec *g_recs[93];
static void cb_host_alloc(void *ctx, unsigned char **dst, size_t bytes)
{
    chk_ctx(ctx);
    const long slot = (long)(dst - g_slab);
    int card = (slot >= 0 && slot < 93 && g_recs[slot]) ? g_recs[slot]->card : -1;
    void *p = nullptr;
    if (posix_memalign(&p, 4096, bytes ? bytes : 1)) { fprintf(stderr, "posix_memalign\n"); _exit(2); }
    memset(p, 0xEE, bytes);
    *dst = (unsigned char *)p;
    ev(card, "host_alloc L" + std::to_string(slot) + " bytes=" + std::to_string(bytes));
}
static void do_init(int prefill, size_t raw)
{
    ld_config c; memset(&c, 0, sizeof c);
    c.cb.ctx = &g_magic; c.cb.set_device = cb_set_device; c.cb.ensure_work = cb_ensure_work; c.cb.load_trunk = cb_trunk; c.cb.load_expert_tables = cb_tables;
    c.cb.alloc_state = cb_state; c.cb.host_alloc = cb_host_alloc; c.expert_raw = raw; c.prefill = prefill; c.slab = g_slab; c.handles = g_handles;
    api.init(&c);
}

// ------------------------------------------------------------------ jobs
struct JobBox { std::function<void()> f; };
static void tramp(void *a) { std::unique_ptr<JobBox> b((JobBox *)a); b->f(); }
static void submit_fn(int card, std::function<void()> f) { api.submit(card, tramp, new JobBox{ std::move(f) }); }

// per card execution log
struct CardLog { std::vector<int> order; std::set<int> tids; int running = 0, maxrun = 0; };
static std::mutex g_cl_m;
static std::map<int, CardLog> g_cl;
static void job_enter(int card, int ticket)
{
    std::lock_guard<std::mutex> g(g_cl_m);
    CardLog &c = g_cl[card]; c.order.push_back(ticket); c.tids.insert(cur_tid()); c.running++; c.maxrun = std::max(c.maxrun, c.running);
}
static void job_leave(int card) { std::lock_guard<std::mutex> g(g_cl_m); g_cl[card].running--; }

static std::map<int, std::vector<std::string>> events_by_card()
{
    std::map<int, std::vector<std::string>> m;
    std::lock_guard<std::mutex> g(g_em);
    for (auto &e : g_ev) m[e.card].push_back(e.s);
    return m;
}
static std::map<int, std::set<int>> ev_tids_by_card()
{
    std::map<int, std::set<int>> m;
    std::lock_guard<std::mutex> g(g_em);
    for (auto &e : g_ev) m[e.card].insert(e.tid);
    return m;
}
static std::string join(const std::vector<std::string> &v, const char *sep)
{
    std::string o; for (size_t i = 0; i < v.size(); i++) { if (i) o += sep; o += v[i]; } return o;
}

static void timed_wait(const char *what, int line)
{
    std::atomic<bool> ret(false);
    std::thread t([&] { api.wait(); ret = true; });
    if (!spin_until([&] { return ret.load(); }, 20000)) { hard_fail(line, what); }
    t.join();
}

// ------------------------------------------------------------------ scenario fifo
static int thread_count()
{
    int n = 0;
    if (DIR *d = opendir("/proc/self/task")) { while (struct dirent *e = readdir(d)) if (e->d_name[0] != '.') n++; closedir(d); }
    return n;
}
// A joined helper thread can stay listed in /proc/self/task for a moment after pthread_join returned (seen once: 6 threads, then 5 on the next read): wait until the count equals the
// expected value on 5 reads in a row; a thread the library left running (or one it did not start) keeps the count different for the whole 3 s.
static int settled_thread_count(int expected)
{
    int last = thread_count(), stable = 0;
    const ll t0 = now_ms();
    while (now_ms() - t0 < 3000) {
        last = thread_count();
        if (last == expected) { if (++stable >= 5) return last; } else stable = 0;
        usleep(10000);
    }
    return last;
}
static void scn_fifo(Rng &r)
{
    const int base_threads = thread_count();
    do_init(0, 4096);
    const int ncards = 1 + (int)r.below(8);
    std::vector<int> cards = { 0, 1, 2, 3, 4, 5, 6, 7 };
    std::shuffle(cards.begin(), cards.end(), r.g); cards.resize((size_t)ncards);
    std::map<int, int> njobs; std::vector<int> subs;
    for (int c : cards) { njobs[c] = r.chance(1, 4) ? 0 : (int)r.below(40); for (int i = 0; i < njobs[c]; i++) subs.push_back(c); }
    std::shuffle(subs.begin(), subs.end(), r.g);
    std::map<int, int> ticket; int total = 0;
    std::atomic<int> executed(0);
    const bool sleepy = r.chance(1, 2);
    for (int c : subs) {
        const int t = ticket[c]++; const int us = sleepy ? (int)r.below(150) : 0; total++;
        submit_fn(c, [c, t, us, &executed] { job_enter(c, t); if (us) usleep((useconds_t)us); executed++; job_leave(c); });
    }
    const ll t0 = now_ms(); timed_wait("fifo: ld_wait did not return", __LINE__); const ll dt = now_ms() - t0; (void)dt;
    CK(executed == total, "executed %d of %d", executed.load(), total);
    auto evs = events_by_card(); auto evt = ev_tids_by_card();
    std::set<int> alltids; int cards_used = 0;
    for (int c : cards) {
        CardLog &cl = g_cl[c];
        std::vector<int> want; for (int i = 0; i < njobs[c]; i++) want.push_back(i);
        CK(cl.order == want, "card %d: execution order is not the submission order", c);
        CK(cl.maxrun <= 1, "card %d: jobs overlapped (max running %d)", c, cl.maxrun);
        if (njobs[c] > 0) {
            cards_used++;
            CK(cl.tids.size() == 1, "card %d: jobs ran on %zu threads", c, cl.tids.size());
            CK(evs[c] == std::vector<std::string>{ "set_device" }, "card %d: thread start events: %s", c, join(evs[c], ",").c_str());
            CK(evt[c].size() == 1 && cl.tids == evt[c], "card %d: the start hook ran on another thread than the jobs", c);
            for (int t : cl.tids) CK(alltids.insert(t).second, "card %d shares a thread with another card", c);
        } else CK(evs.count(c) == 0, "card %d got no job but a thread started", c);
        emit("fifo card %d jobs %d order_ok start_events %zu", c, njobs[c], evs.count(c) ? evs[c].size() : (size_t)0);
    }
    CK(g_ctx_bad == 0, "ctx pointer wrong in %d calls", g_ctx_bad.load());
    // exactly one new thread per card that got a job (lazy start, nothing else started, detached threads still alive and idle)
    const int tc = settled_thread_count(base_threads + cards_used);
    CK(tc == base_threads + cards_used, "threads: %d now, %d before, %d cards used", tc, base_threads, cards_used);
    emit("fifo cards_used %d total %d new_threads %d", cards_used, total, tc - base_threads);
    // a second wait with nothing pending returns at once
    const ll t1 = now_ms(); api.wait(); CK(now_ms() - t1 < 500, "second wait took %lld ms", now_ms() - t1);
}

// ------------------------------------------------------------------ scenario barrier
static void scn_barrier(Rng &r)
{
    do_init(0, 4096);
    const int k = 2 + (int)r.below(7);
    std::vector<int> cards = { 0, 1, 2, 3, 4, 5, 6, 7 };
    std::shuffle(cards.begin(), cards.end(), r.g); cards.resize((size_t)k);
    g_bparties = k; g_bn = 0; g_bbroken = false;
    std::map<int, int> extra; std::atomic<int> passed(0);
    for (int c : cards) {
        extra[c] = (int)r.below(4);
        submit_fn(c, [c, &passed] { job_enter(c, 0); barrier_arrive(); passed++; job_leave(c); });
        for (int i = 1; i <= extra[c]; i++) submit_fn(c, [c, i] { job_enter(c, i); job_leave(c); });
    }
    timed_wait("barrier: ld_wait did not return", __LINE__);
    CK(!g_bbroken, "the %d first jobs did not run at the same time (barrier timed out, n=%d)", k, g_bn);
    CK(passed == k, "passed %d of %d", passed.load(), k);
    std::sort(cards.begin(), cards.end());
    for (int c : cards) {
        std::vector<int> want; for (int i = 0; i <= extra[c]; i++) want.push_back(i);
        CK(g_cl[c].order == want, "card %d order", c);
        CK(g_cl[c].maxrun == 1 && g_cl[c].tids.size() == 1, "card %d overlap or threads", c);
        emit("barrier card %d jobs %d", c, 1 + extra[c]);
    }
    emit("barrier parties %d passed %d", k, passed.load());
}

// ------------------------------------------------------------------ scenario wait
struct Node { int card; std::vector<Node *> kids; };
static Node *mk_tree(Rng &r, int depth, int &count)
{
    Node *n = new Node{ (int)r.below(8), {} }; count++;
    if (depth < 3) { int f = (int)r.below(4); for (int i = 0; i < f; i++) n->kids.push_back(mk_tree(r, depth + 1, count)); }
    return n;
}
static void run_node(Node *n, std::atomic<int> *cnt, int delay_us)
{
    (*cnt)++;
    for (Node *k : n->kids) submit_fn(k->card, [k, cnt, delay_us] { run_node(k, cnt, delay_us); });
    if (delay_us) usleep((useconds_t)delay_us);
}
static void scn_wait(Rng &r)
{
    // (a) wait before ld_init returns at once, and after init with nothing pending
    { const ll t0 = now_ms(); api.wait(); CK(now_ms() - t0 < 500, "wait before init took %lld ms", now_ms() - t0); }
    do_init(0, 4096);
    { const ll t0 = now_ms(); api.wait(); CK(now_ms() - t0 < 500, "wait with nothing pending took %lld ms", now_ms() - t0); }
    emit("wait zero_pending ok");
    // (b) quick jobs, wait, wait again
    { std::atomic<int> n(0); const int m = 1 + (int)r.below(20); for (int i = 0; i < m; i++) submit_fn((int)r.below(8), [&n] { n++; });
      timed_wait("wait (b)", __LINE__); CK(n == m, "wait returned with %d of %d jobs done", n.load(), m); api.wait(); emit("wait quick jobs %d done", m); }
    // (c) gated jobs: wait must block until they finish
    {
        const int n = 1 + (int)r.below(6); std::atomic<bool> gate(false), returned(false); std::atomic<int> done(0);
        for (int i = 0; i < n; i++) submit_fn((int)r.below(8), [&] { while (!gate) usleep(100); done++; });
        std::thread w([&] { api.wait(); returned = true; });
        usleep(100 * 1000);
        CK(!returned, "wait returned while %d gated jobs were blocked", n);
        gate = true;
        if (!spin_until([&] { return returned.load(); }, 20000)) hard_fail(__LINE__, "wait (c) did not return after the gate opened");
        CK(done == n, "wait (c) returned with %d of %d jobs done", done.load(), n);
        w.join(); emit("wait gated %d ok", n);
    }
    // (d) a job submitted by another thread during the wait is waited for
    {
        const int ca = (int)r.below(8), cb = r.chance(1, 3) ? ca : (int)r.below(8);
        std::atomic<bool> ga(false), gb(false), a_done(false), b_done(false), returned(false);
        submit_fn(ca, [&] { while (!ga) usleep(100); a_done = true; });
        std::thread w([&] { api.wait(); returned = true; });
        usleep(50 * 1000);
        std::thread s([&] { submit_fn(cb, [&] { while (!gb) usleep(100); b_done = true; }); });
        s.join();
        ga = true;
        CK(spin_until([&] { return a_done.load(); }, 10000), "(d) A never ran");
        usleep(100 * 1000);
        CK(!returned, "wait returned although the job submitted during the wait was still blocked (same card %d)", ca == cb);
        CK(!b_done || ca == cb, "(d) B finished before its gate");
        gb = true;
        if (!spin_until([&] { return returned.load(); }, 20000)) hard_fail(__LINE__, "wait (d) did not return after B finished");
        CK(b_done, "wait (d) returned before B finished");
        w.join(); emit("wait submit_during_wait ok");
    }
    // (e) jobs that submit jobs: the wait covers every descendant
    {
        const int roots = 1 + (int)r.below(3); const int delay = r.chance(1, 2) ? 20000 : 0; int total = 0;
        std::vector<Node *> rs; for (int i = 0; i < roots; i++) rs.push_back(mk_tree(r, 0, total));
        std::atomic<int> cnt(0); std::atomic<bool> gate(false), returned(false);
        for (Node *n : rs) submit_fn(n->card, [n, &cnt, &gate, delay] { while (!gate) usleep(100); run_node(n, &cnt, delay); });
        std::thread w([&] { api.wait(); returned = true; });
        usleep(50 * 1000); CK(!returned, "(e) wait returned before the roots ran"); gate = true;
        if (!spin_until([&] { return returned.load(); }, 30000)) hard_fail(__LINE__, "wait (e) did not return");
        CK(cnt == total, "(e) wait returned with %d of %d nodes executed", cnt.load(), total);
        w.join(); emit("wait nested nodes %d ok", total);
    }
    // (f) several waiters
    {
        const int nw = 1 + (int)r.below(4); std::atomic<bool> gate(false); std::atomic<int> back(0);
        submit_fn((int)r.below(8), [&] { while (!gate) usleep(100); });
        std::vector<std::thread> ws; for (int i = 0; i < nw; i++) ws.emplace_back([&] { api.wait(); back++; });
        usleep(60 * 1000); CK(back == 0, "(f) a waiter returned early"); gate = true;
        if (!spin_until([&] { return back.load() == nw; }, 20000)) hard_fail(__LINE__, "wait (f) waiters did not all return");
        for (auto &t : ws) t.join();
        emit("wait waiters %d ok", nw);
    }
    CK(g_ctx_bad == 0, "ctx pointer wrong");
}

// ------------------------------------------------------------------ scenario stress: several submitters, nested jobs, concurrent waiters
static std::mutex g_cardmu[8];
static std::map<int, std::vector<int>> g_expected_order;   // by card, tickets in submission order, guarded by g_cardmu
static std::map<int, int> g_next_ticket;
static std::atomic<int> g_submitted(0), g_executed(0);
static void stress_submit(int card, int depth, uint64_t seed);
static void stress_submit(int card, int depth, uint64_t seed)
{
    std::lock_guard<std::mutex> g(g_cardmu[card]);
    const int t = g_next_ticket[card]++;
    { std::lock_guard<std::mutex> g2(g_cl_m); g_expected_order[card].push_back(t); }
    g_submitted++;
    submit_fn(card, [card, t, depth, seed] {
        job_enter(card, t);
        Rng r(seed);
        if (depth < 2 && r.chance(1, 3)) { const int f = 1 + (int)r.below(2); for (int i = 0; i < f; i++) stress_submit((int)r.below(8), depth + 1, r.g()); }
        if (r.chance(1, 2)) usleep((useconds_t)r.below(100));
        g_executed++;
        job_leave(card);
    });
}
static void scn_stress(Rng &r)
{
    do_init(0, 4096);
    // warm every card from this thread first: the original has an unlocked first-submit race (see the `race` scenario), which this scenario must not depend on
    for (int c = 0; c < 8; c++) stress_submit(c, 2, r.g());
    timed_wait("stress: warm-up wait", __LINE__);
    const int T = 2 + (int)r.below(5);
    std::vector<uint64_t> seeds; std::vector<int> counts; for (int i = 0; i < T; i++) { counts.push_back(5 + (int)r.below(60)); seeds.push_back(r.g()); }
    std::atomic<bool> stop(false); std::atomic<long> waits(0);
    std::thread waiter([&] { while (!stop) { api.wait(); waits++; usleep(500); } });
    std::vector<std::thread> ts;
    for (int i = 0; i < T; i++) ts.emplace_back([i, &counts, &seeds] { Rng q(seeds[(size_t)i]); for (int j = 0; j < counts[(size_t)i]; j++) stress_submit((int)q.below(8), 0, q.g()); });
    for (auto &t : ts) t.join();
    timed_wait("stress: final ld_wait", __LINE__);
    stop = true; waiter.join();
    CK(g_submitted == g_executed, "submitted %d executed %d", g_submitted.load(), g_executed.load());
    long total_expected = 0;
    for (int c = 0; c < 8; c++) {
        CardLog &cl = g_cl[c];
        CK(cl.order == g_expected_order[c], "card %d: execution order differs from the ticket order (jobs %zu)", c, cl.order.size());
        CK(cl.maxrun == 1 && cl.tids.size() == 1, "card %d: overlap %d, threads %zu", c, cl.maxrun, cl.tids.size());
        total_expected += (long)g_expected_order[c].size();
    }
    CK(total_expected == g_submitted, "ticket count");
    CK(g_ctx_bad == 0, "ctx pointer wrong");
    note("stress submitters %d jobs %d concurrent waits returned %ld", T, g_submitted.load(), waits.load());
    emit("stress submitted==executed ok");
}

// ------------------------------------------------------------------ scenario race: concurrent FIRST submits to one card (the original has an unlocked check there)
static void scn_race(Rng &r)
{
    do_init(0, 4096);
    int violations = 0, trials = 0;
    for (int c = 0; c < 8; c++) {
        const int T = 2 + (int)r.below(4);
        std::atomic<int> go(0), ran(0); std::mutex m; std::set<int> tids;
        std::vector<std::thread> ts;
        for (int i = 0; i < T; i++) ts.emplace_back([&, c] { go++; while (go < T) std::this_thread::yield(); submit_fn(c, [&] { { std::lock_guard<std::mutex> g(m); tids.insert(cur_tid()); } usleep(200); ran++; }); });
        for (auto &t : ts) t.join();
        timed_wait("race: ld_wait", __LINE__);
        trials++;
        int starts = 0; { std::lock_guard<std::mutex> g(g_em); for (auto &e : g_ev) if (e.card == c) starts++; }
        if (tids.size() != 1 || starts != 1 || ran != T) violations++;
    }
    note("race trials %d violations %d (one thread and one start hook per card expected)", trials, violations);
    if (g_label == "ref") note("ref: the original's first submit is not locked; violations are informational");
    else CK(violations == 0, "%d of %d trials had more than one thread / start hook for a card", violations, trials);
    emit("race trials %d done", trials);
}

// ------------------------------------------------------------------ files
static std::vector<std::string> g_created;   // files and dirs we made, removed at the end (files first)
static void mkdir_p(const std::string &p) { if (mkdir(p.c_str(), 0700) == 0) g_created.push_back(p); }
static std::string layer_dir(unsigned L) { return g_ds + "/expert-" + std::to_string(L); }
static std::string layer_file(unsigned L) { return layer_dir(L) + "/experts.direct"; }
static void write_file(unsigned L, const std::vector<unsigned char> &data)
{
    mkdir_p(layer_dir(L));
    FILE *f = fopen(layer_file(L).c_str(), "wb");
    if (!f) { perror(layer_file(L).c_str()); _exit(2); }
    if (!data.empty() && fwrite(data.data(), 1, data.size(), f) != data.size()) { perror("fwrite"); _exit(2); }
    fclose(f); g_created.push_back(layer_file(L));
}
static void cleanup_files()
{
    for (auto it = g_created.rbegin(); it != g_created.rend(); ++it) { if (unlink(it->c_str()) != 0) rmdir(it->c_str()); }
    g_created.clear();
    if (!g_ds.empty()) rmdir(g_ds.c_str());
}
static std::string g_base = "/tmp";   // where the dataset directory is made
static int g_force_mode = -1;         // scenario shm: -1 = random, else the open mode
static void init_ds() { char b[96]; snprintf(b, sizeof b, "%s/ldloader-verify.%d", g_base.c_str(), (int)getpid()); g_ds = b; mkdir_p(g_ds); }

// the pread calls the original's loop makes on a file of `fsize` bytes, with a cap on each read and an optional failing call (n-th, 1-based)
static std::vector<PreadRec> model_reads(const std::string &path, size_t raw, long cap, size_t fsize, long fail_at, bool *ok)
{
    std::vector<PreadRec> v; long idx = 0; *ok = true;
    for (size_t r = 0; r < 4; r++) {
        size_t got = 0;
        while (got < raw) {
            const size_t req = raw - got; const long off = (long)(r * raw + got); idx++;
            if (fail_at && idx == fail_at) { v.push_back(PreadRec{ path, req, off, -1 }); *ok = false; return v; }
            size_t n = req; if (cap > 0 && n > (size_t)cap) n = (size_t)cap;
            const size_t avail = (size_t)off >= fsize ? 0 : fsize - (size_t)off;
            const long ret = (long)std::min(n, avail);
            v.push_back(PreadRec{ path, req, off, ret });
            if (ret <= 0) { *ok = false; return v; }
            got += (size_t)ret;
        }
    }
    return v;
}
static std::vector<unsigned char> rand_bytes(Rng &r, size_t n) { std::vector<unsigned char> v(n); for (size_t i = 0; i < n; i += 8) { uint64_t x = r.g(); for (size_t j = 0; j < 8 && i + j < n; j++) v[i + j] = (unsigned char)(x >> (8 * j)); } return v; }

// ------------------------------------------------------------------ scenario open: ld_open_layer on several layers, with the file system's answers forced
static void scn_open(Rng &r)
{
    init_ds();
    const int prefill = r.chance(1, 2);
    const int mode_r = r.chance(1, 2) ? 1 : 0;
    const int mode = g_force_mode >= 0 ? g_force_mode : mode_r;
    const size_t raw = r.pick<size_t>({ 4096, 8192, 65536 });
    const long cap = mode == 0 ? r.pick<long>({ 0, 512, 4096, 8192 }) : r.pick<long>({ 0, 1, 7, 512, 5000 });
    g_open_mode = mode; g_pread_cap = cap;
    g_trunk_barrier = r.chance(1, 3);
    // layers: in prefill mode 0..92 (the slab table has 93 entries); without it also numbers beyond the tables
    std::vector<unsigned> pool; for (unsigned i = 0; i <= 92; i++) pool.push_back(i);
    if (!prefill) { pool.push_back(93); pool.push_back(100000); pool.push_back(4294967295u); }
    std::shuffle(pool.begin(), pool.end(), r.g);
    const int n = 1 + (int)r.below(10);
    std::vector<unsigned> layers(pool.begin(), pool.begin() + n);
    if (r.chance(1, 3) && std::find(layers.begin(), layers.end(), 0u) == layers.end()) layers.push_back(0);
    if (r.chance(1, 4) && std::find(layers.begin(), layers.end(), 92u) == layers.end()) layers.push_back(92);
    std::map<unsigned, std::vector<unsigned char>> content;
    std::vector<LayerRec *> recs; std::set<int> used_cards;
    for (unsigned L : layers) {
        LayerRec *lr = new LayerRec; lr->layer = L; lr->card = (int)r.below(8); lr->k = (unsigned)r.below(12); lr->positions = (unsigned)r.below(5000) + 1;
        if (r.chance(1, 8)) lr->k = 4294967295u;           // (int)k == -1
        if (r.chance(1, 8)) lr->positions = 4294967295u;
        recs.push_back(lr); used_cards.insert(lr->card);
        if (L >= 1) { const size_t records = 4 + r.below(3); content[L] = rand_bytes(r, records * raw); write_file(L, content[L]); }
        if (L < 93) g_recs[L] = lr;
    }
    if (g_trunk_barrier) { g_bparties = (int)used_cards.size(); g_bn = 0; g_bbroken = false; g_barrier_done.clear(); }
    do_init(prefill, raw);
    const bool hold = r.chance(1, 2);
    std::string dsbuf = g_ds;   // the caller's buffer: clobbered after the calls while the jobs are held, to prove the dataset string was copied
    emit("open cfg prefill=%d mode=%d raw=%zu cap=%ld layers=%zu barrier=%d hold=%d", prefill, mode, raw, cap, layers.size(), g_trunk_barrier ? 1 : 0, hold ? 1 : 0);
    g_hold = hold;
    for (LayerRec *lr : recs) {
        ld_layer q; memset(&q, 0, sizeof q);
        q.handle = lr; q.dataset = &dsbuf[0]; q.layer = lr->layer; q.k = lr->k; q.positions = lr->positions; q.card = lr->card;
        q.h_L = &lr->L; q.h_card = &lr->C; q.h_k_in = &lr->K; q.h_fd = &lr->fd;
        api.open_layer(&q);
        CK(lr->L == (int)lr->layer && lr->C == lr->card && lr->K == (int)lr->k, "layer %u: bookkeeping L=%d card=%d k_in=%d right after the call", lr->layer, lr->L, lr->C, lr->K);
        if (lr->layer <= 92) CK(g_handles[lr->layer] == lr, "layer %u: handles[] entry", lr->layer);
    }
    if (hold) { std::fill(dsbuf.begin(), dsbuf.end(), 'X'); for (LayerRec *lr : recs) lr->C = (lr->card + 1) % 8; g_hold = false; }
    timed_wait("open: ld_wait did not return", __LINE__);
    if (hold) for (LayerRec *lr : recs) lr->C = lr->card;
    CK(!g_bbroken, "the first trunk calls of the cards did not run at the same time");
    // events per card: the thread start, then per layer in submission order
    std::map<int, std::vector<std::string>> want;
    for (LayerRec *lr : recs) {
        auto &w = want[lr->card]; if (w.empty()) w.push_back("set_device");
        const int Li = (int)lr->layer;
        w.push_back("set_device"); w.push_back("ensure_work"); w.push_back("trunk L" + std::to_string(Li) + " ds=<DS>");
        if (lr->layer >= 1) w.push_back("tables L" + std::to_string(Li));
        w.push_back("state L" + std::to_string((int)lr->layer) + " pos=" + std::to_string(lr->positions));
        if (lr->layer >= 1 && prefill) w.push_back("host_alloc L" + std::to_string(lr->layer) + " bytes=" + std::to_string(4 * raw));
    }
    auto got = events_by_card();
    auto tids = ev_tids_by_card();
    CK(g_ctx_bad == 0, "ctx pointer wrong");
    for (auto &kv : want) {
        CK(got[kv.first] == kv.second, "card %d events differ:\n   got  %s\n   want %s", kv.first, join(got[kv.first], ";").c_str(), join(kv.second, ";").c_str());
        CK(tids[kv.first].size() == 1, "card %d: events from %zu threads", kv.first, tids[kv.first].size());
        emit("open card %d events %s", kv.first, join(got[kv.first], ";").c_str());
    }
    CK(got.size() == want.size(), "events for cards that had no layer");
    // layers: fd, files, slab
    std::vector<LayerRec *> sorted = recs; std::sort(sorted.begin(), sorted.end(), [](LayerRec *a, LayerRec *b) { return a->layer < b->layer; });
    std::lock_guard<std::mutex> gio(g_io_m);
    for (LayerRec *lr : sorted) {
        std::string line = "open layer " + std::to_string(lr->layer) + " L=" + std::to_string(lr->L) + " card=" + std::to_string(lr->C) + " k_in=" + std::to_string(lr->K);
        if (lr->layer == 0) {
            CK(lr->fd == 424242, "layer 0 wrote fd (%d)", lr->fd);
            CK(g_slab[0] == nullptr, "layer 0 touched slab[0]");
            line += " fd=untouched";
        } else {
            const std::string path = layer_file(lr->layer);
            CK(lr->fd >= 0, "layer %u: fd %d", lr->layer, lr->fd);
            std::vector<OpenRec> mine; for (auto &o : g_opens) if (o.path == path) mine.push_back(o);
            std::string os;
            if (mode == 1) {
                CK(mine.size() == 2 && (mine[0].flags & O_DIRECT) && mine[0].ret < 0 && !(mine[1].flags & O_DIRECT) && mine[1].ret >= 0, "layer %u: open sequence with O_DIRECT refused (%zu opens)", lr->layer, mine.size());
                os = "m1:direct-refused,plain-ok";
            } else {
                const bool accepted = !mine.empty() && mine[0].ret >= 0;
                CK(!mine.empty() && (mine[0].flags & O_DIRECT) && (mine[0].flags & O_ACCMODE) == O_RDONLY, "layer %u: first open has no O_DIRECT / is not read-only", lr->layer);
                CK(accepted ? mine.size() == 1 : (mine.size() == 2 && !(mine[1].flags & O_DIRECT) && mine[1].ret >= 0), "layer %u: open sequence (%zu opens)", lr->layer, mine.size());
                os = accepted ? "m0:direct-ok" : "m0:direct-refused,plain-ok";
            }
            CK(!mine.empty() && lr->fd == mine.back().ret, "layer %u: fd %d is not the last successful open", lr->layer, lr->fd);
            {   // the fd is still open and is the layer's file: an aligned 4096-byte read at 0 works for O_DIRECT and plain fds
                void *ab = nullptr; if (posix_memalign(&ab, 4096, 4096)) _exit(2);
                const ssize_t got = real_pread(lr->fd, ab, 4096, 0);
                CK(got == 4096 && !memcmp(ab, content[lr->layer].data(), 4096), "layer %u: reading the fd gave %zd bytes / other content", lr->layer, got);
                free(ab);
            }
            std::vector<PreadRec> pr; for (auto &p : g_preads) if (p.path == path) pr.push_back(p);
            char b[160];
            if (prefill) {
                bool ok; auto m = model_reads(path, raw, cap, content[lr->layer].size(), 0, &ok);
                CK(ok && pr == m, "layer %u: pread sequence differs (got %zu calls, model %zu)", lr->layer, pr.size(), m.size());
                CK(g_slab[lr->layer] && !memcmp(g_slab[lr->layer], content[lr->layer].data(), 4 * raw), "layer %u: slab is not the first 4 records", lr->layer);
                long sum = 0; for (auto &p : pr) sum += p.ret;
                snprintf(b, sizeof b, " fd=ok open=%s preads=%zu bytes=%ld slab=eq", os.c_str(), pr.size(), sum);
            } else {
                CK(pr.empty(), "layer %u: pread without prefill", lr->layer);
                CK(lr->layer >= 93 || g_slab[lr->layer] == nullptr, "layer %u: slab touched without prefill", lr->layer);
                snprintf(b, sizeof b, " fd=ok open=%s preads=0 slab=null", os.c_str());
            }
            line += b;
        }
        emit("%s", line.c_str());
    }
    cleanup_files();
}

// ------------------------------------------------------------------ scenario natural: the kernel itself refuses O_DIRECT (experts.direct is a symlink to a procfs file; no forced answer)
static void scn_natural(Rng &r)
{
    init_ds();
    g_open_mode = 0; g_pread_cap = 0;
    do_init(0, 4096);
    std::set<unsigned> ls; const int n = 1 + (int)r.below(3);
    while ((int)ls.size() < n) ls.insert(1 + (unsigned)r.below(92));
    std::vector<LayerRec *> recs;
    for (unsigned L : ls) {
        mkdir_p(layer_dir(L));
        if (symlink("/proc/version", layer_file(L).c_str()) != 0) { perror("symlink"); _exit(2); }
        g_created.push_back(layer_file(L));
        LayerRec *lr = new LayerRec; lr->layer = L; lr->card = (int)r.below(8); lr->k = 1; lr->positions = 2; g_recs[L] = lr; recs.push_back(lr);
        ld_layer q; memset(&q, 0, sizeof q); q.handle = lr; q.dataset = g_ds.c_str(); q.layer = L; q.k = 1; q.positions = 2; q.card = lr->card;
        q.h_L = &lr->L; q.h_card = &lr->C; q.h_k_in = &lr->K; q.h_fd = &lr->fd;
        api.open_layer(&q);
    }
    timed_wait("natural: ld_wait", __LINE__);
    std::lock_guard<std::mutex> gio(g_io_m);
    for (LayerRec *lr : recs) {
        const std::string path = layer_file(lr->layer);
        std::vector<OpenRec> mine; for (auto &o : g_opens) if (o.path == path) mine.push_back(o);
        CK(mine.size() == 2 && (mine[0].flags & O_DIRECT) && mine[0].ret < 0 && mine[0].err == EINVAL && !(mine[1].flags & O_DIRECT) && mine[1].ret >= 0, "layer %u: natural open sequence (%zu opens, first err %d)", lr->layer, mine.size(), mine.empty() ? 0 : mine[0].err);
        char b[13] = { 0 };
        const ssize_t got = lr->fd >= 0 ? real_pread(lr->fd, b, 12, 0) : -1;
        CK(got == 12 && !strcmp(b, "Linux versio"), "layer %u: reading the fallback fd gave %zd bytes [%s]", lr->layer, got, b);
        emit("natural layer %u open=direct-refused(errno %d),plain-ok read=[%s]", lr->layer, mine.empty() ? 0 : mine[0].err, b);
    }
    cleanup_files();
}

// ------------------------------------------------------------------ scenario real: the engine's EXPERT_RAW (17547264) on a sparse file
static void scn_real(Rng &r)
{
    init_ds();
    const size_t raw = 17547264;
    g_open_mode = r.chance(1, 2) ? 1 : 0; g_pread_cap = 0;
    // 5 records, sparse; marker blocks at the start and end of every record
    mkdir_p(layer_dir(9));
    int fd = (int)syscall(SYS_openat, AT_FDCWD, layer_file(9).c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) { perror("create"); _exit(2); }
    g_created.push_back(layer_file(9));
    if (ftruncate(fd, (off_t)(5 * raw)) != 0) { perror("ftruncate"); _exit(2); }
    for (size_t rec = 0; rec < 5; rec++) for (int w = 0; w < 2; w++) {
        std::vector<unsigned char> blk = rand_bytes(r, 4096);
        const long off = (long)(rec * raw + (w ? raw - 4096 : 0));
        if (pwrite(fd, blk.data(), blk.size(), (off_t)off) != 4096) { perror("pwrite"); _exit(2); }
    }
    close(fd);
    LayerRec lr; lr.layer = 9; lr.card = (int)r.below(8); lr.k = 3; lr.positions = 100; g_recs[9] = &lr;
    do_init(1, raw);
    ld_layer q; memset(&q, 0, sizeof q); q.handle = &lr; q.dataset = g_ds.c_str(); q.layer = 9; q.k = 3; q.positions = 100; q.card = lr.card;
    q.h_L = &lr.L; q.h_card = &lr.C; q.h_k_in = &lr.K; q.h_fd = &lr.fd;
    api.open_layer(&q);
    timed_wait("real: ld_wait", __LINE__);
    CK(lr.fd >= 0 && g_slab[9], "fd %d slab %p", lr.fd, (void *)g_slab[9]);
    // the slab equals the first 4 records, compared against the file with the interposer's real read
    bool eq = g_slab[9] != nullptr; const size_t chunk = 1 << 20; std::vector<unsigned char> buf(chunk);
    int cfd = (int)syscall(SYS_openat, AT_FDCWD, layer_file(9).c_str(), O_RDONLY, 0);
    for (size_t off = 0; eq && off < 4 * raw; off += chunk) {
        const size_t n = std::min(chunk, 4 * raw - off);
        if (real_pread(cfd, buf.data(), n, (long)off) != (ssize_t)n || memcmp(buf.data(), g_slab[9] + off, n)) eq = false;
    }
    close(cfd);
    CK(eq, "slab differs from the first 4 records of the file");
    // the fifth record was not read
    std::string path = layer_file(9); std::vector<PreadRec> pr; { std::lock_guard<std::mutex> g(g_io_m); for (auto &p : g_preads) if (p.path == path) pr.push_back(p); }
    long maxend = 0; for (auto &p : pr) maxend = std::max(maxend, p.off + p.ret);
    CK(maxend == (long)(4 * raw), "the loader read up to byte %ld, expected %zu", maxend, 4 * raw);
    std::string s; for (auto &p : pr) s += "(" + std::to_string(p.req) + "@" + std::to_string(p.off) + "->" + std::to_string(p.ret) + ")";
    emit("real open_mode=%d fd=%s preads=%zu %s slab=%s", (int)g_open_mode, lr.fd >= 0 ? "ok" : "bad", pr.size(), s.c_str(), eq ? "eq" : "DIFF");
    cleanup_files();
}

// ------------------------------------------------------------------ scenario init2: a second ld_init is ignored, even an invalid one
static std::atomic<int> g_b_calls(0);
static void bcb_card(void *, int) { g_b_calls++; }
static void bcb_layer(void *, void *, const char *, int) { g_b_calls++; }
static void bcb_state(void *, void *, unsigned) { g_b_calls++; }
static void bcb_alloc(void *, unsigned char **, size_t) { g_b_calls++; }
static void scn_init2(Rng &r)
{
    init_ds();
    do_init(0, 4096);
    ld_config bad; memset(&bad, 0, sizeof bad);            // everything NULL: would be rejected as a first init
    api.init(&bad);
    unsigned char *otherslab[93]; void *otherh[93]; memset(otherslab, 0, sizeof otherslab); memset(otherh, 0, sizeof otherh);
    ld_config c2; memset(&c2, 0, sizeof c2);               // valid, but another configuration (prefill on, other tables, other callbacks)
    c2.cb.ctx = nullptr; c2.cb.set_device = bcb_card; c2.cb.ensure_work = bcb_card; c2.cb.load_trunk = bcb_layer; c2.cb.load_expert_tables = bcb_layer; c2.cb.alloc_state = bcb_state;
    c2.cb.host_alloc = bcb_alloc; c2.expert_raw = 8192; c2.prefill = 1; c2.slab = otherslab; c2.handles = otherh;
    api.init(&c2);
    const unsigned L = 1 + (unsigned)r.below(92);
    write_file(L, rand_bytes(r, 6 * 4096));
    LayerRec lr; lr.layer = L; lr.card = (int)r.below(8); lr.k = 1; lr.positions = 7; g_recs[L] = &lr;
    ld_layer q; memset(&q, 0, sizeof q); q.handle = &lr; q.dataset = g_ds.c_str(); q.layer = L; q.k = 1; q.positions = 7; q.card = lr.card;
    q.h_L = &lr.L; q.h_card = &lr.C; q.h_k_in = &lr.K; q.h_fd = &lr.fd;
    api.open_layer(&q); timed_wait("init2: ld_wait", __LINE__);
    CK(g_b_calls == 0, "the second configuration's callbacks were called %d times", g_b_calls.load());
    CK(g_handles[L] == &lr && otherh[L] == nullptr, "the first handles table must be the one written");
    CK(g_slab[L] == nullptr && otherslab[L] == nullptr, "the first configuration has no prefill: no slab");
    auto evs = events_by_card();
    CK(evs[lr.card].size() == 6, "events: %s", join(evs[lr.card], ";").c_str());
    emit("init2 second init ignored layer %u events %zu", L, evs[lr.card].size());
    cleanup_files();
}

// ------------------------------------------------------------------ scenario fatal: the exit(1) paths, each in a child process
struct ChildResult { int code; bool timeout; std::string out, err; };
static ChildResult run_child(const std::function<void()> &body)
{
    int po[2], pe[2];
    if (pipe(po) || pipe(pe)) { perror("pipe"); _exit(2); }
    fflush(stdout); fflush(stderr);
    const pid_t pid = fork();
    if (pid == 0) {
        dup2(po[1], 1); dup2(pe[1], 2); close(po[0]); close(po[1]); close(pe[0]); close(pe[1]);
        alarm(30);
        body(); _exit(77);
    }
    close(po[1]); close(pe[1]);
    ChildResult cr; cr.timeout = false;
    struct pollfd p[2] = { { po[0], POLLIN, 0 }, { pe[0], POLLIN, 0 } };
    bool open0 = true, open1 = true; const ll t0 = now_ms();
    while (open0 || open1) {
        if (now_ms() - t0 > 25000) { cr.timeout = true; kill(pid, SIGKILL); break; }
        p[0].fd = open0 ? po[0] : -1; p[1].fd = open1 ? pe[0] : -1;
        if (poll(p, 2, 200) < 0) { if (errno == EINTR) continue; break; }
        char b[8192];
        for (int i = 0; i < 2; i++) if (p[i].revents & (POLLIN | POLLHUP)) { ssize_t n = read(i ? pe[0] : po[0], b, sizeof b); if (n <= 0) { (i ? open1 : open0) = false; } else (i ? cr.err : cr.out).append(b, (size_t)n); }
    }
    close(po[0]); close(pe[0]);
    int st = 0; waitpid(pid, &st, 0);
    cr.code = WIFEXITED(st) ? WEXITSTATUS(st) : 1000 + WTERMSIG(st);
    return cr;
}
static bool g_atexit_events = true;
static void child_atexit()
{
    std::string s = "ATEXIT";
    if (g_atexit_events) {
        std::string evs; { std::lock_guard<std::mutex> g(g_em); for (auto &e : g_ev) { evs += " " + std::to_string(e.card) + ":" + e.s; } }
        long no, np; { std::lock_guard<std::mutex> g(g_io_m); no = (long)g_opens.size(); np = (long)g_preads.size(); }
        s += evs + " opens=" + std::to_string(no) + " preads=" + std::to_string(np);
    }
    s += "\n";
    ssize_t w = write(2, s.data(), s.size()); (void)w;
}
static void child_prelude() { atexit(child_atexit); printf("BUFFERED"); }   // stdout is a pipe: this text appears only if exit() flushes stdio

static std::string cut_path(const std::string &ds, unsigned L) { std::string p = ds + "/expert-" + std::to_string(L) + "/experts.direct"; if (p.size() > 4095) p.resize(4095); return p; }
static std::string layer_events(int card, unsigned L, bool tables, const std::string &dsnorm, unsigned pos)
{
    const std::string c = std::to_string(card);
    std::string s;
    s += " " + c + ":set_device";   // thread start
    s += " " + c + ":set_device " + c + ":ensure_work " + c + ":trunk L" + std::to_string((int)L) + " ds=" + dsnorm;
    if (tables) s += " " + c + ":tables L" + std::to_string((int)L);
    s += " " + c + ":state L" + std::to_string((int)L) + " pos=" + std::to_string(pos);
    return s;
}
static int tracked_opens(const std::string &cut) { return cut.size() >= 14 && cut.compare(cut.size() - 14, 14, "experts.direct") == 0 ? 2 : 0; }   // the interposer logs only paths ending in experts.direct
static void expect_child(const char *name, const ChildResult &cr, int want_code, const std::string &want_out, const std::string &want_err)
{
    CK(!cr.timeout, "%s: the child did not finish (timeout)", name);
    CK(cr.code == want_code, "%s: exit status %d, expected %d", name, cr.code, want_code);
    CK(cr.out == want_out, "%s: stdout [%s], expected [%s]", name, esc(cr.out).c_str(), esc(want_out).c_str());
    CK(cr.err == want_err, "%s: stderr [%s], expected [%s]", name, esc(cr.err).c_str(), esc(want_err).c_str());
    emit("fatal %s: exit=%d out=[%s] err=[%s]%s", name, cr.code, esc(cr.out).c_str(), esc(cr.err).c_str(), cr.timeout ? " TIMEOUT" : "");
}
static void open_one(LayerRec &lr, const char *ds, unsigned L, int card, unsigned positions = 3)
{
    lr.layer = L; lr.card = card; lr.positions = positions; if (L < 93) g_recs[L] = &lr;
    ld_layer q; memset(&q, 0, sizeof q); q.handle = &lr; q.dataset = ds; q.layer = L; q.k = 1; q.positions = positions; q.card = card;
    q.h_L = &lr.L; q.h_card = &lr.C; q.h_k_in = &lr.K; q.h_fd = &lr.fd; api.open_layer(&q);
}
static void scn_fatal(Rng &r)
{
    init_ds();
    const char *BUF = "BUFFERED";
    // missing file: both opens fail (any open mode), message with the path, thread exits while main waits
    for (int variant = 0; variant < 3; variant++) {
        const unsigned L = 1 + (unsigned)r.below(92); const int card = (int)r.below(8); const int prefill = r.chance(1, 2);
        const std::string ds = g_ds + "/nonexistent";
        ChildResult cr = run_child([&] { child_prelude(); g_open_mode = variant; do_init(prefill, 4096); LayerRec lr; open_one(lr, ds.c_str(), L, card); api.wait(); _exit(55); });
        const std::string want_err = "cannot open " + cut_path(ds, L) + "\n" + "ATEXIT" + layer_events(card, L, true, norm(ds), 3) + " opens=2 preads=0\n";
        expect_child(("missing_file_openmode" + std::to_string(variant)).c_str(), cr, 1, BUF, want_err);
    }
    // a path longer than 4095 bytes is cut in the message (snprintf into char[4096])
    // a path whose total length is 4094..4097 bytes: cut to 4095 in the message and in the open call (snprintf into char[4096])
    for (size_t target : { (size_t)4094, (size_t)4095, (size_t)4096, (size_t)4097 }) {
        const unsigned L = 1 + (unsigned)r.below(92); const int card = (int)r.below(8);
        const std::string suffix = "/expert-" + std::to_string(L) + "/experts.direct";
        std::string ds = g_ds + "/n"; while (ds.size() + suffix.size() < target) ds += std::string(1, (char)('a' + r.below(26)));
        ChildResult cr = run_child([&] { child_prelude(); do_init(0, 4096); LayerRec lr; open_one(lr, ds.c_str(), L, card); api.wait(); _exit(55); });
        const std::string want_err = "cannot open " + cut_path(ds, L) + "\n" + "ATEXIT" + layer_events(card, L, true, norm(ds), 3) + " opens=" + std::to_string(tracked_opens(cut_path(ds, L))) + " preads=0\n";
        CK((ds + suffix).size() == target, "long path construction");
        expect_child(("long_path_total" + std::to_string(target)).c_str(), cr, 1, BUF, want_err);
    }
    // end of file inside a record, with caps on the read size
    for (int variant = 0; variant < 4; variant++) {
        const unsigned L = 1 + (unsigned)r.below(92); const int card = (int)r.below(8);
        const int mode = variant & 1; const size_t raw = r.pick<size_t>({ 4096, 8192 });
        const long cap = mode == 0 ? r.pick<long>({ 0, 512, 4096 }) : r.pick<long>({ 0, 1, 7, 512 });
        size_t fsize = (size_t)r.below(4 * raw); if (mode == 0) fsize -= fsize % 512;
        write_file(L, rand_bytes(r, fsize));
        bool ok; std::vector<PreadRec> m = model_reads(layer_file(L), raw, cap, fsize, 0, &ok);
        ChildResult cr = run_child([&] { child_prelude(); g_open_mode = mode; g_pread_cap = cap; do_init(1, raw); LayerRec lr; open_one(lr, g_ds.c_str(), L, card); api.wait(); _exit(55); });
        const std::string want_err = "prefill: cannot read a record of layer " + std::to_string(L) + "\n" + "ATEXIT" + layer_events(card, L, true, norm(g_ds), 3) + " " + std::to_string(card) + ":host_alloc L" + std::to_string(L) + " bytes=" + std::to_string(4 * raw) + " opens=" + std::to_string(mode == 1 ? 2 : 1) + " preads=" + std::to_string(m.size()) + "\n";
        CK(!ok, "model: the file of %zu bytes should end inside the 4 records", fsize);
        expect_child(("eof_fsize" + std::to_string(fsize) + "_cap" + std::to_string(cap) + "_mode" + std::to_string(mode)).c_str(), cr, 1, BUF, want_err);
        unlink(layer_file(L).c_str());
    }
    // a read error at the n-th read: same message, no retry (the preads count in the ATEXIT line is n)
    for (int variant = 0; variant < 3; variant++) {
        const unsigned L = 1 + (unsigned)r.below(92); const int card = (int)r.below(8);
        const int mode = 1; const size_t raw = 4096; const long cap = r.pick<long>({ 0, 1000, 4000 });
        write_file(L, rand_bytes(r, 4 * raw));
        bool ok; std::vector<PreadRec> full = model_reads(layer_file(L), raw, cap, 4 * raw, 0, &ok);
        CK(ok, "model: full read");
        const long fail_at = 1 + (long)r.below(full.size()); const int en = r.pick<int>({ EINTR, EIO, EAGAIN });
        ChildResult cr = run_child([&] { child_prelude(); g_open_mode = mode; g_pread_cap = cap; g_pread_fail_at = fail_at; g_pread_fail_errno = en; do_init(1, raw); LayerRec lr; open_one(lr, g_ds.c_str(), L, card); api.wait(); _exit(55); });
        const std::string want_err = "prefill: cannot read a record of layer " + std::to_string(L) + "\n" + "ATEXIT" + layer_events(card, L, true, norm(g_ds), 3) + " " + std::to_string(card) + ":host_alloc L" + std::to_string(L) + " bytes=" + std::to_string(4 * raw) + " opens=2 preads=" + std::to_string(fail_at) + "\n";
        expect_child(("readerr_at" + std::to_string(fail_at) + "_errno" + std::to_string(en) + "_cap" + std::to_string(cap)).c_str(), cr, 1, BUF, want_err);
        unlink(layer_file(L).c_str());
    }
    // guards of the header (differences from the original: bounds / no init / NULLs)
    struct G { const char *name; std::function<void()> body; std::string err; };
    std::vector<G> gs;
    const std::string noev = "ATEXIT opens=0 preads=0\n";
    gs.push_back({ "submit_before_init", [] { api.submit(0, [](void *) {}, nullptr); }, "loader: ld_submit before ld_init\n" });
    gs.push_back({ "open_before_init", [] { LayerRec lr; open_one(lr, "/x", 1, 0); }, "loader: ld_open_layer before ld_init\n" });
    gs.push_back({ "submit_card8", [] { do_init(0, 4096); api.submit(8, [](void *) {}, nullptr); }, "loader: card 8 out of range (0..7)\n" });
    gs.push_back({ "submit_card_minus1", [] { do_init(0, 4096); api.submit(-1, [](void *) {}, nullptr); }, "loader: card -1 out of range (0..7)\n" });
    gs.push_back({ "submit_null_job", [] { do_init(0, 4096); api.submit(0, nullptr, nullptr); }, "loader: job is NULL\n" });
    gs.push_back({ "open_card9", [] { do_init(0, 4096); LayerRec lr; open_one(lr, "/x", 1, 9); }, "loader: card 9 out of range (0..7)\n" });
    gs.push_back({ "open_null_req", [] { do_init(0, 4096); api.open_layer(nullptr); }, "loader: req is NULL\n" });
    gs.push_back({ "open_null_dataset", [] { do_init(0, 4096); LayerRec lr; ld_layer q; memset(&q, 0, sizeof q); q.handle = &lr; q.layer = 1; q.card = 0; api.open_layer(&q); }, "loader: dataset is NULL\n" });
    gs.push_back({ "open_layer93_prefill", [] { do_init(1, 4096); LayerRec lr; open_one(lr, "/x", 93, 0); }, "loader: layer 93 has no slab slot (0..92)\n" });
    gs.push_back({ "open_layer_huge_prefill", [] { do_init(1, 4096); LayerRec lr; open_one(lr, "/x", 4000000000u, 0); }, "loader: layer 4000000000 has no slab slot (0..92)\n" });
    gs.push_back({ "init_null_cfg", [] { api.init(nullptr); }, "loader: cfg is NULL\n" });
    auto bad_cfg = [](const char *what, std::function<void(ld_config &)> f) {
        return std::function<void()>([what, f] { (void)what; ld_config c; memset(&c, 0, sizeof c); c.cb.set_device = cb_set_device; c.cb.ensure_work = cb_ensure_work; c.cb.load_trunk = cb_trunk; c.cb.load_expert_tables = cb_tables;
            c.cb.alloc_state = cb_state; c.cb.host_alloc = cb_host_alloc; c.expert_raw = 4096; c.slab = g_slab; c.handles = g_handles; f(c); api.init(&c); });
    };
    gs.push_back({ "init_no_set_device", bad_cfg("a", [](ld_config &c) { c.cb.set_device = nullptr; }), "loader: set_device is NULL\n" });
    gs.push_back({ "init_no_ensure_work", bad_cfg("a", [](ld_config &c) { c.cb.ensure_work = nullptr; }), "loader: ensure_work is NULL\n" });
    gs.push_back({ "init_no_load_trunk", bad_cfg("a", [](ld_config &c) { c.cb.load_trunk = nullptr; }), "loader: load_trunk is NULL\n" });
    gs.push_back({ "init_no_load_expert_tables", bad_cfg("a", [](ld_config &c) { c.cb.load_expert_tables = nullptr; }), "loader: load_expert_tables is NULL\n" });
    gs.push_back({ "init_no_alloc_state", bad_cfg("a", [](ld_config &c) { c.cb.alloc_state = nullptr; }), "loader: alloc_state is NULL\n" });
    gs.push_back({ "init_no_handles", bad_cfg("a", [](ld_config &c) { c.handles = nullptr; }), "loader: handles is NULL\n" });
    gs.push_back({ "init_prefill_no_host_alloc", bad_cfg("a", [](ld_config &c) { c.prefill = 1; c.cb.host_alloc = nullptr; }), "loader: host_alloc is NULL\n" });
    gs.push_back({ "init_prefill_no_slab", bad_cfg("a", [](ld_config &c) { c.prefill = 1; c.slab = nullptr; }), "loader: slab is NULL\n" });
    for (auto &g : gs) {
        ChildResult cr = run_child([&] { child_prelude(); g.body(); _exit(55); });
        expect_child(g.name, cr, 1, BUF, g.err + noev);
    }
    // the exit happens while another card's job is running and the main thread waits: still exit status 1 with the message
    {
        const unsigned L = 1 + (unsigned)r.below(92); const std::string ds = g_ds + "/nonexistent";
        ChildResult cr = run_child([&] { child_prelude(); g_atexit_events = false; do_init(0, 4096);
            api.submit(0, [](void *) { for (int i = 0; i < 100000; i++) usleep(1000); }, nullptr);
            LayerRec lr; open_one(lr, ds.c_str(), L, 1); api.wait(); _exit(55); });
        expect_child("exit_while_other_card_runs", cr, 1, BUF, "cannot open " + cut_path(ds, L) + "\nATEXIT\n");
    }
    cleanup_files();
}

int main(int argc, char **argv)
{
    if (argc < 5) { fprintf(stderr, "usage: %s <label> <lib.so> <scenario> <seed>\n", argv[0]); return 2; }
    g_label = argv[1]; g_scn = argv[3];
    const uint64_t seed = strtoull(argv[4], nullptr, 10);
    alarm(240);
    load_lib(argv[2]);
    set_canaries();
    Rng r(seed);
    printf("N driver %s lib=%s scenario=%s seed=%llu\n", g_label.c_str(), argv[2], argv[3], (unsigned long long)seed);
    emit("scenario %s seed %llu", argv[3], (unsigned long long)seed);
    const std::string s = argv[3];
    if (s == "fifo") scn_fifo(r);
    else if (s == "barrier") scn_barrier(r);
    else if (s == "wait") scn_wait(r);
    else if (s == "stress") scn_stress(r);
    else if (s == "race") scn_race(r);
    else if (s == "open") scn_open(r);
    else if (s == "shm") { g_base = "/dev/shm"; g_force_mode = 0; scn_open(r); }   // tmpfs, no forced open answer: reports what the kernel does with O_DIRECT there
    else if (s == "natural") scn_natural(r);
    else if (s == "real") scn_real(r);
    else if (s == "init2") scn_init2(r);
    else if (s == "fatal") scn_fatal(r);
    else { fprintf(stderr, "unknown scenario %s\n", argv[3]); return 2; }
    check_canaries();
    finish(g_failed ? 1 : 0, false);
    return 0;
}
