// A GPU backend for whole-model.c (see common/backend.h): the same functions the CPU backend defines (stage2_* for layer 0, lbk_* for KDA layers, lbm_* for MLA
// layers), run on the cards of one machine. Layers are spread over the cards in consecutive groups (K3GPU_CARDS, default 8); a layer's trunk and its state stay
// on its card, activations travel through host memory (7168 floats per token and snapshot). The experts are not held: after each layer's router, the chosen ones are
// brought to the card's pool of expert slots (decode.cuh), from the host cache or the drive.
//
// Settings: K3GPU_CARDS (8), K3GPU_HOST_GB (pinned RAM for the expert cache, 150), K3GPU_RESERVE_MB (card memory kept free of expert slots, 2500),
// K3GPU_READERS (drive reads at once, 16), K3GPU_VERBOSE (1: one line per layer call of a single token),
// K3GPU_COPYQ (cpp, the default, or rust: which implementation runs the copy queue) and K3GPU_COPYQ_LIB (path of libcopyq.so, default libcopyq.so),
// K3GPU_SLOTCACHE (cpp, the default, or rust: which implementation runs the expert slot caches) and K3GPU_SLOTCACHE_LIB (path of libslotcache.so, default libslotcache.so),
// K3GPU_ROUTETABLE (cpp or rust: the route-cache parser and next-use index) and K3GPU_ROUTETABLE_LIB (libroutetable.so),
// and, for the rest of the host path, K3GPU_POOLINIT (HostCache::init, CardPool::init; libpoolinit.so), K3GPU_SUPPLY (pool_ensure; libsupply.so), K3GPU_PLANNER (read-ahead and card-queue
// planning; libplanner.so, needs COPYQ, SLOTCACHE and ROUTETABLE on rust), K3GPU_LOADER (layer loader threads and open_layer's host side; libloader.so) and K3GPU_ENGCFG (start-up settings,
// card_of, force_routes and the exit summary; libengcfg.so), each cpp by default, each with a _LIB variable. No fallback: asking for rust and not getting it stops the run.
// build: see scripts/run-model-gpu.sh
#include "decode.cuh"
extern "C" {
#include "backend.h"
}
#if __has_include("copyq.h")
#include "copyq.h"   // contract of the Rust copy queue (src/rust/copyq/include); a build without it has only the C++ queue
#define K3GPU_COPYQ_BUILT 1
#endif
#if __has_include("routetable.h")
#include "routetable.h"   // contract of the Rust route table (src/rust/routetable/include); a build without it has only the C++ parser
#define K3GPU_ROUTETABLE_BUILT 1
#endif
#if __has_include("loader.h")
#include "loader.h"   // contract of the Rust layer loader (src/rust/loader/include); a build without it has only the C++ loader
#define K3GPU_LOADER_BUILT 1
#endif
#if __has_include("engcfg.h")
#include "engcfg.h"   // contract of the Rust start-up / summary code (src/rust/engcfg/include); a build without it has only the C++ code
#define K3GPU_ENGCFG_BUILT 1
#endif
// poolinit.h, supply.h and planner.h are included by decode.cuh (K3GPU_POOLINIT_BUILT, K3GPU_SUPPLY_BUILT, K3GPU_PLANNER_BUILT).
#include <dlfcn.h>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <string>

enum { MAX_CARDS = 8, N_LAYERS = 92 };

struct Handle {
    int L = 0, card = 0, k_in = 0; bool mla = false;
    Layer y; KdaState kda; MlaCache mc; int fd = -1;
};
struct Card { Work w; bool ready = false; unsigned alloc_n = 0; };

static Card g_card[MAX_CARDS];
static CardPool g_pool[MAX_CARDS];
static HostCache g_host;
static Handle g_l0;
static int g_cards = 0, g_readers = 16, g_verbose = 0, g_last_layer = 0;
static unsigned g_max_n = 0;
static size_t g_reserve = 0;
static double g_t_calls[3], g_n_calls[3];   // prompt-pass calls, single-token calls, multi-token calls: seconds and count
static double g_t_up = 0, g_t_run = 0, g_t_down = 0;
static bool g_configured = false;

static int env_int(const char *name, int dflt) { const char *v = getenv(name); return v && *v ? atoi(v) : dflt; }

// ---- which start-up, selector and summary code runs (K3GPU_ENGCFG): cpp, the default, is the code in this file; rust = libengcfg.so (src/rust/engcfg), which reads the settings,
// parses the three selector variables, prints their lines, computes card_of and force_routes, and prints the exit summary. Loaded before anything else in configure().
struct EcApi {
    bool on = false;
#ifdef K3GPU_ENGCFG_BUILT
    int (*env_int)(const char *, int) = nullptr; void (*settings_from_env)(ec_settings *, int) = nullptr; size_t (*pend_cap)(size_t) = nullptr; FILE *(*open_trace)(void) = nullptr;
    void (*print_route_cache)(const ec_settings *, size_t, size_t) = nullptr; void (*prefill_start)(size_t) = nullptr; void (*print_start_line)(const ec_settings *) = nullptr;
    int (*selector_mode)(int, int) = nullptr; const char *(*selector_lib)(int) = nullptr; void (*selector_announce_rust)(int, const char *) = nullptr;
    int (*card_of)(int, int, int) = nullptr; void (*force_routes)(size_t, size_t, const ec_call *, int, int, int *, int) = nullptr; void (*print_summary)(const ec_summary *) = nullptr;
#endif
};
static EcApi g_ec;
enum { SEL_COPYQ = 0, SEL_SLOTCACHE = 1, SEL_ROUTETABLE = 2 };
#ifdef K3GPU_ENGCFG_BUILT
static_assert((int)EC_COPYQ == (int)SEL_COPYQ && (int)EC_SLOTCACHE == (int)SEL_SLOTCACHE && (int)EC_ROUTETABLE == (int)SEL_ROUTETABLE, "engcfg.h component numbers");
#endif
static const struct { const char *name, *label, *hdr, *what, *dir, *dlib; } SEL[3] = {
    { "COPYQ", "copy queue", "copyq.h", "the C++ queue", "copyq", "libcopyq.so" },
    { "SLOTCACHE", "slot cache", "slotcache.h", "the C++ cache", "slotcache", "libslotcache.so" },
    { "ROUTETABLE", "route table", "routetable.h", "the C++ parser", "routetable", "libroutetable.so" } };
// the variable K3GPU_<NAME>: 0 = cpp (its line is printed), 1 = rust (the caller loads the library, then calls sel_announce); a wrong value or rust without the header stops the run
static int sel_mode(int comp, bool built)
{
#ifdef K3GPU_ENGCFG_BUILT
    if (g_ec.on) return g_ec.selector_mode(comp, built ? 1 : 0);
#endif
    const auto &s = SEL[comp]; char var[40]; snprintf(var, sizeof var, "K3GPU_%s", s.name);
    const char *m = getenv(var);
    if (!m || !*m || !strcmp(m, "cpp")) { fprintf(stderr, "K3GPU: %s cpp\n", s.label); return 0; }
    if (!built) { fprintf(stderr, "K3GPU_%s=%s: this build has no %s, only %s (build with -I src/rust/%s/include)\n", s.name, m, s.hdr, s.what, s.dir); exit(1); }
    if (strcmp(m, "rust")) { fprintf(stderr, "K3GPU_%s must be cpp or rust, not %s\n", s.name, m); exit(1); }
    return 1;
}
[[maybe_unused]] static const char *sel_lib(int comp)
{
#ifdef K3GPU_ENGCFG_BUILT
    if (g_ec.on) return g_ec.selector_lib(comp);
#endif
    char var[48]; snprintf(var, sizeof var, "K3GPU_%s_LIB", SEL[comp].name);
    const char *lib = getenv(var); return lib && *lib ? lib : SEL[comp].dlib;
}
[[maybe_unused]] static void sel_announce(int comp, const char *lib)
{
#ifdef K3GPU_ENGCFG_BUILT
    if (g_ec.on) { g_ec.selector_announce_rust(comp, lib); return; }
#endif
    fprintf(stderr, "K3GPU: %s rust (%s)\n", SEL[comp].label, lib);
}
// the components after the first four (K3GPU_POOLINIT, _SUPPLY, _PLANNER, _LOADER, _ENGCFG): cpp (line printed, returns NULL) or the dlopen handle of the library, whose line is printed by the caller after its symbols are found
static void *comp_select(const char *var, const char *label, const char *dflt_lib, bool built, const char **lib_out)
{
    const char *m = getenv(var);
    if (!m || !*m || !strcmp(m, "cpp")) { fprintf(stderr, "K3GPU: %s cpp\n", label); return nullptr; }
    if (strcmp(m, "rust")) { fprintf(stderr, "%s must be cpp or rust, not %s\n", var, m); exit(1); }
    if (!built) { fprintf(stderr, "%s=rust: this build has no header for it (build with -I of the crate's include directory, see scripts/run-model-gpu.sh)\n", var); exit(1); }
    char lv[64]; snprintf(lv, sizeof lv, "%s_LIB", var);
    const char *lib = getenv(lv); if (!lib || !*lib) lib = dflt_lib;
    void *h = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "%s=rust: cannot load %s: %s\n", var, lib, dlerror()); exit(1); }
    *lib_out = lib;
    return h;
}
#define COMP_SYM(var, h, dst, name) dst = (decltype(dst))dlsym(h, name); if (!dst) { fprintf(stderr, "%s=rust: the library has no %s\n", var, name); exit(1); }

// ---- route cache (K3GPU_ROUTECACHE = the K3GPU_TRACE file of an earlier run of the same request; K3GPU_LOOKAHEAD = how many MoE calls ahead, default 92).
// Routing is a function of the token prefix, so for a request seen before the experts of every later call are known before layer 0. They are read from the drive into
// the RAM cache by background threads while earlier calls compute; the call itself then takes them from RAM. Outputs do not depend on it (only where an expert
// comes from does); a call whose experts differ from the cache stops the look-ahead.
struct CallSet { int layer; std::vector<int> experts; std::vector<int> counts; };
static std::vector<CallSet> g_rc;
static size_t g_rc_pos = 0, g_rc_planned = 0, g_pend_cap = 0;
static long g_rc_cur = -1;   // number of the call being processed
static std::unordered_map<int, std::vector<int>> g_uses;   // expert key -> numbers of the calls that need it
static std::unordered_map<int, size_t> g_use_ptr;
// next call after the current one that needs the expert (policy 2), or a huge number
static int rc_nu(int key)
{
    auto it = g_uses.find(key);
    if (it == g_uses.end()) return 1000000000;
    const std::vector<int> &l = it->second; size_t &p = g_use_ptr[key];
    while (p < l.size() && l[p] <= g_rc_cur) p++;
    return p < l.size() ? l[p] : 1000000000;
}
// ---- which route table runs (K3GPU_ROUTETABLE): cpp, the default, parses the route-cache file with rc_load and answers rc_nu above; rust = libroutetable.so (src/rust/routetable),
// which owns the parser and the next-use index. In rust mode the parsed calls are copied into g_rc, so the planners are unchanged. No fallback.
struct RtApi {
    bool on = false;
#ifdef K3GPU_ROUTETABLE_BUILT
    rt_table *(*load)(const char *, int) = nullptr; size_t (*calls)(const rt_table *) = nullptr; int (*layer)(const rt_table *, size_t) = nullptr;
    const int *(*experts)(const rt_table *, size_t, size_t *) = nullptr; const int *(*counts)(const rt_table *, size_t, size_t *) = nullptr;
    void (*set_current)(rt_table *, long) = nullptr; int (*next_use)(rt_table *, int) = nullptr;
    rt_table *t = nullptr;   // never freed: print_summary reads the calls from an atexit handler
#endif
};
static RtApi g_rt;
static unsigned long g_rt_nu_calls = 0;
static int rc_nu_sel(int key)
{
    g_rt_nu_calls++;
#ifdef K3GPU_ROUTETABLE_BUILT
    if (g_rt.on) return g_rt.next_use(g_rt.t, key);
#endif
    return rc_nu(key);
}
static void rc_set_cur(long cur)
{
    g_rc_cur = cur;
#ifdef K3GPU_ROUTETABLE_BUILT
    if (g_rt.on && g_rt.t) g_rt.set_current(g_rt.t, cur);
#endif
}
static int g_lookahead = 0; static bool g_rc_ok = false;
static Handle *g_handles[N_LAYERS + 1];
static int g_layout = 0;   // K3GPU_LAYOUT: 0 = consecutive groups of layers per card, 1 = interleaved (layer L on card (L-1) % cards, layer 0 on card 0)
static unsigned long g_pf_planned = 0, g_pf_skipped = 0;

// Jobs of the reading threads: kind 0 = drive -> RAM slot (read-ahead into the RAM cache); kind 1 = drive -> pinned bounce -> card slot; kind 2 = RAM slot -> card slot.
// Kinds 1 and 2 may first copy the card slot's old expert down into a RAM slot (victim_host). `arr` is set when the new bytes are on the card, `wb` (a Pending of the
// RAM cache) when the old expert is in RAM, `rel` when the RAM slot a kind 2 job read from may be reused.
struct Rel { std::atomic<int> state; unsigned char *slot; Rel() : state(0), slot(nullptr) {} };
struct PfJob {
    int fd = -1; off_t off = 0; unsigned char *slot = nullptr; Pending *p = nullptr; int layer = 0;
    int kind = 0, dev = 0; unsigned char *dst = nullptr, *host_src = nullptr, *victim_host = nullptr; Pending *wb = nullptr;
    std::shared_ptr<Arrival> arr; std::shared_ptr<Rel> rel;
};
struct PfQueue { std::mutex mu; std::condition_variable cv; std::deque<PfJob> q; bool started = false; };
static PfQueue &pfq(int i) { static PfQueue *q = new PfQueue[MAX_CARDS + 1]; return q[i]; }   // never destroyed: the reading threads wait on it until the process ends. 0..7 = jobs for that card, 8 = drive to RAM read-ahead

static void pf_enqueue(const PfJob &j)
{
    const int qi = j.kind == 0 ? MAX_CARDS : j.dev;
    PfQueue &Q = pfq(qi);
    std::lock_guard<std::mutex> g(Q.mu);
    if (!Q.started) {
        Q.started = true;
        const int nthreads = qi == MAX_CARDS ? g_readers : std::max(2, g_readers / std::max(1, g_cards));
        for (int i = 0; i < nthreads; i++) std::thread([&Q]() {
            unsigned char *bounce = nullptr; cudaStream_t stream[MAX_CARDS] = {};
            for (;;) {
                PfJob jb;
                { std::unique_lock<std::mutex> u(Q.mu); Q.cv.wait(u, [&]() { return !Q.q.empty(); }); jb = Q.q.front(); Q.q.pop_front(); }
                if (jb.kind == 0) {
                    size_t got = 0;
                    while (!g_prefill && got < (size_t)EXPERT_RAW) {
                        const ssize_t r = pread(jb.fd, jb.slot + got, EXPERT_RAW - got, jb.off + (off_t)got);
                        if (r <= 0) { fprintf(stderr, "read-ahead: pread failed: %s\n", r < 0 ? strerror(errno) : "end of file"); exit(1); }
                        got += (size_t)r;
                    }
                    jb.p->state.store(1, std::memory_order_release);
                    continue;
                }
                CHECK(cudaSetDevice(jb.dev));
                if (!stream[jb.dev]) CHECK(cudaStreamCreateWithFlags(&stream[jb.dev], cudaStreamNonBlocking));
                cudaStream_t s = stream[jb.dev];
                if (jb.victim_host) {
                    CHECK(cudaMemcpyAsync(jb.victim_host, jb.dst, EXPERT_RAW, cudaMemcpyDeviceToHost, s)); CHECK(cudaStreamSynchronize(s));
                    jb.wb->state.store(1, std::memory_order_release);
                }
                if (jb.kind == 1) {
                    const unsigned char *src = bounce;
                    if (g_prefill) src = g_slab[jb.layer] + (size_t)((jb.off / EXPERT_RAW) & 3) * EXPERT_RAW;
                    else {
                    if (!bounce) CHECK(cudaHostAlloc((void **)&bounce, EXPERT_RAW, cudaHostAllocDefault));
                    src = bounce;
                    size_t got = 0;
                    while (got < (size_t)EXPERT_RAW) {
                        const ssize_t r = pread(jb.fd, bounce + got, EXPERT_RAW - got, jb.off + (off_t)got);
                        if (r <= 0) { fprintf(stderr, "card read-ahead: pread failed: %s\n", r < 0 ? strerror(errno) : "end of file"); exit(1); }
                        got += (size_t)r;
                    }
                    }
                    act_gate();
                    CHECK(cudaMemcpyAsync(jb.dst, src, EXPERT_RAW, cudaMemcpyHostToDevice, s));
                } else { act_gate(); CHECK(cudaMemcpyAsync(jb.dst, g_prefill ? g_slab[jb.layer] + (size_t)((jb.off / EXPERT_RAW) & 3) * EXPERT_RAW : jb.host_src, EXPERT_RAW, cudaMemcpyHostToDevice, s)); }
                CHECK(cudaStreamSynchronize(s));
                if (jb.rel) jb.rel->state.store(1, std::memory_order_release);
                jb.arr->state.store(1, std::memory_order_release);
            }
        }).detach();
    }
    Q.q.push_back(j);
    Q.cv.notify_one();
}

// ---- which copy queue runs (K3GPU_COPYQ). pf_enqueue above is the C++ queue; "rust" sends the same jobs to libcopyq.so (src/rust/copyq), loaded at start-up.
// There is no fallback: asking for rust and not getting it stops the run, so a log that says rust ran rust. The Rust queue does not own the flag objects (copyq.h),
// so the shared_ptrs a C++ job would carry are held here until the job has set its arrival flag.
static bool g_copyq_rust = false;
static std::atomic<unsigned long> g_cq_jobs[3];   // jobs sent, by kind, whichever queue took them
#ifdef K3GPU_COPYQ_BUILT
static void (*g_cq_enqueue)(const cq_job *) = nullptr;
struct CqHold { std::shared_ptr<Arrival> arr; std::shared_ptr<Rel> rel; };
static std::mutex g_cq_mu;
static std::vector<CqHold> &g_cq_hold = *new std::vector<CqHold>;   // never destroyed: Rust workers run until the process ends and may still store into a flag at exit
static_assert(sizeof(std::atomic<int>) == 4 && std::atomic<int>::is_always_lock_free, "copyq.h passes the flags as plain ints");
static void copyq_select()
{
    if (sel_mode(SEL_COPYQ, true) == 0) return;
    const char *lib = sel_lib(SEL_COPYQ);
    void *h = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "K3GPU_COPYQ=rust: cannot load %s: %s\n", lib, dlerror()); exit(1); }
    void (*init)(const cq_config *) = (void (*)(const cq_config *))dlsym(h, "cq_init");
    g_cq_enqueue = (void (*)(const cq_job *))dlsym(h, "cq_enqueue");
    if (!init || !g_cq_enqueue) { fprintf(stderr, "K3GPU_COPYQ=rust: %s has no cq_init / cq_enqueue\n", lib); exit(1); }
    cq_config c = {};
    c.expert_raw = EXPERT_RAW; c.readers = g_readers; c.cards = g_cards; c.prefill = g_prefill ? 1 : 0;
    c.slab = g_slab;   // the table itself: its entries are filled after start-up and read by the queue when a job runs
    c.act_hold = reinterpret_cast<const int *>(&g_act_hold);
    init(&c);
    g_copyq_rust = true;
    sel_announce(SEL_COPYQ, lib);
}
static void pf_submit_rust(const PfJob &j)
{
    cq_job c = {};
    c.kind = j.kind; c.dev = j.dev; c.fd = j.fd; c.off = j.off; c.layer = j.layer; c.slot = j.slot; c.dst = j.dst; c.host_src = j.host_src; c.victim_host = j.victim_host;
    c.p_state = j.p ? reinterpret_cast<int *>(&j.p->state) : nullptr;
    c.wb_state = j.wb ? reinterpret_cast<int *>(&j.wb->state) : nullptr;
    c.arr_state = j.arr ? reinterpret_cast<int *>(&j.arr->state) : nullptr;
    c.rel_state = j.rel ? reinterpret_cast<int *>(&j.rel->state) : nullptr;
    if (j.arr || j.rel) {
        std::lock_guard<std::mutex> g(g_cq_mu);
        g_cq_hold.push_back(CqHold{ j.arr, j.rel });
        if ((g_cq_hold.size() & 127) == 0)
            for (size_t i = 0; i < g_cq_hold.size();) {
                const CqHold &h = g_cq_hold[i];
                const bool done = h.arr ? h.arr->state.load(std::memory_order_acquire) == 1 : h.rel->state.load(std::memory_order_acquire) == 1;
                if (done) { g_cq_hold[i] = std::move(g_cq_hold.back()); g_cq_hold.pop_back(); } else i++;
            }
    }
    g_cq_enqueue(&c);
}
#else
static void copyq_select()
{
    sel_mode(SEL_COPYQ, false);   // cpp: its line; anything else: the build has no copyq.h, stop
}
#endif
static void pf_submit(const PfJob &j)
{
    g_cq_jobs[j.kind].fetch_add(1, std::memory_order_relaxed);
#ifdef K3GPU_COPYQ_BUILT
    if (g_copyq_rust) { pf_submit_rust(j); return; }
#endif
    pf_enqueue(j);
}

// ---- which slot cache runs (K3GPU_SLOTCACHE, see ScApi in decode.cuh): cpp, the default, or rust = libslotcache.so (src/rust/slotcache). Same rules as K3GPU_COPYQ: no fallback,
// one line at start-up (K3GPU: slot cache cpp|rust (...)) and one at exit with the call counts. It must run before the first slot is added (g_host.init).
#ifdef K3GPU_SLOTCACHE_BUILT
static void slotcache_select()
{
    if (sel_mode(SEL_SLOTCACHE, true) == 0) return;
    const char *lib = sel_lib(SEL_SLOTCACHE);
    void *h = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "K3GPU_SLOTCACHE=rust: cannot load %s: %s\n", lib, dlerror()); exit(1); }
#define SC_SYM(f, name) g_sc.f = (decltype(g_sc.f))dlsym(h, name); if (!g_sc.f) { fprintf(stderr, "K3GPU_SLOTCACHE=rust: %s has no %s\n", lib, name); exit(1); }
    SC_SYM(set_policy, "sc_set_policy") SC_SYM(set_next_use, "sc_set_next_use") SC_SYM(nw, "sc_new") SC_SYM(add_spare, "sc_add_spare") SC_SYM(contains, "sc_contains")
    SC_SYM(find, "sc_find") SC_SYM(touch, "sc_touch") SC_SYM(worth, "sc_worth") SC_SYM(acquire, "sc_acquire") SC_SYM(try_acquire, "sc_try_acquire")
    SC_SYM(remove, "sc_remove") SC_SYM(release, "sc_release")
#undef SC_SYM
    g_sc.on = true;
    sel_announce(SEL_SLOTCACHE, lib);
}
#else
static void slotcache_select()
{
    sel_mode(SEL_SLOTCACHE, false);   // cpp: its line; anything else: the build has no slotcache.h, stop
}
#endif

static void rc_plan(size_t pos)
{
    const size_t last = std::min(g_rc.size(), pos + (size_t)g_lookahead + 1);
    for (size_t i = std::max(g_rc_planned, pos); i < last; i++) {
        const CallSet &cs = g_rc[i]; Handle *hd = g_handles[cs.layer];
        if (!hd || hd->fd < 0) { g_rc_planned = i + 1; continue; }
        CardPool &pool = g_pool[hd->card];
        for (int e : cs.experts) {
            const int key = ekey(cs.layer, e);
            if (g_pend.count(key)) continue;
            if ((pool.ready && pool.c.contains(key)) || g_host.c.contains(key)) { g_pf_skipped++; continue; }
            if (g_pend.size() >= g_pend_cap) { g_rc_planned = i; return; }   // the RAM window is full; planning goes on at the next call
            if (!g_host.c.worth(key)) { g_rc_planned = i; return; }          // policy 2: RAM holds only experts needed sooner than this one
            int vk; float vh;
            unsigned char *slot = g_host.c.acquire(key, g_pend_keys, 1.0f, &vk, &vh);
            Pending *p = new Pending; p->slot = slot;
            g_pend[key] = p; g_pend_keys.insert(key); g_pf_planned++;
            { PfJob j; j.fd = hd->fd; j.off = (off_t)e * EXPERT_RAW; j.slot = slot; j.p = p; pf_submit(j); }
        }
        g_rc_planned = i + 1;
    }
}

// ---- the cards' queues (K3GPU_VQUEUE=1, needs the route cache and policy 2): at every call, the experts of the next calls that are not on their card are brought to
// the card's pool by the reading threads (from RAM if they are there, else from the drive), replacing the held expert that is needed farthest ahead, as long as
// the new one is needed sooner; nothing waits for a layer to be reached. What the pool does not take is read ahead into the RAM cache as before.
static std::vector<std::shared_ptr<Rel>> g_rel;
static std::vector<int> g_wb_keys;
static size_t g_vq_cap = 256;
static long g_vq_from = 0;   // K3GPU_VQ_FROM: the card queues work from this call on; before it the experts are only read ahead into RAM (92 = decode starts after a 92-call prompt pass)
static double g_t_plan = 0;   // seconds spent in the planning of the hook (main thread, on the critical path)
static unsigned long g_vq_ram = 0, g_vq_drive = 0, g_vq_wb = 0, g_vq_nowb = 0;

static void vq_sweep()
{
    for (size_t i = 0; i < g_rel.size();) {
        if (g_rel[i]->state.load(std::memory_order_acquire) == 1) { g_host.c.release(g_rel[i]->slot); g_rel[i] = g_rel.back(); g_rel.pop_back(); } else i++;
    }
    for (size_t i = 0; i < g_wb_keys.size();) {
        auto f = g_pend.find(g_wb_keys[i]);
        if (f == g_pend.end()) { g_wb_keys[i] = g_wb_keys.back(); g_wb_keys.pop_back(); }
        else if (f->second->state.load(std::memory_order_acquire) == 1) { g_pend_keys.erase(f->first); delete f->second; g_pend.erase(f); g_wb_keys[i] = g_wb_keys.back(); g_wb_keys.pop_back(); }
        else i++;
    }
}

// false if the pool takes nothing more for this expert now
static bool vq_issue(CardPool &pool, Handle *hd, int key, int e, int cand_nu, const std::unordered_set<int> &prot)
{
    const bool in_ram = g_host.c.contains(key);
    if (in_ram) { auto pf = g_pend.find(key); if (pf != g_pend.end() && pf->second->state.load(std::memory_order_acquire) == 0) return false; }   // its read has not finished
    int vk; float vh;
    unsigned char *dst = pool.c.try_acquire(key, cand_nu, prot, 1.0f, &vk, &vh);
    if (!dst) return false;
    PfJob j; j.dev = pool.dev; j.dst = dst; j.fd = hd->fd; j.off = (off_t)e * EXPERT_RAW; j.layer = hd->L;
    if (in_ram) {
        pend_take(key);
        j.host_src = g_host.c.remove(key); j.kind = 2;
        j.rel = std::make_shared<Rel>(); j.rel->slot = j.host_src; g_rel.push_back(j.rel); g_vq_ram++;
    } else { j.kind = 1; g_vq_drive++; }
    if (vk >= 0) {
        arr_wait(vk);   // the held expert was itself still on its way
        if (!g_prefill && g_host.cap_slots && g_pend.size() < g_host.cap_slots * 8 / 10 && g_host.c.worth(vk)) {
            int dk; float dh;
            j.victim_host = g_host.c.acquire(vk, g_pend_keys, vh, &dk, &dh);
            j.wb = new Pending; j.wb->slot = j.victim_host;
            g_pend[vk] = j.wb; g_pend_keys.insert(vk); g_wb_keys.push_back(vk); g_vq_wb++;
        } else g_vq_nowb++;
    }
    j.arr = std::make_shared<Arrival>(); g_arr[key] = j.arr;
    pf_submit(j);
    return true;
}

static void rc_plan_vq(size_t pos, const std::vector<int> &cur)
{
    vq_sweep();
    std::unordered_set<int> prot;
    for (int e : cur) prot.insert(ekey(g_rc[pos].layer, e));
    size_t out = 0;
    for (auto &kv : g_arr) if (kv.second->state.load(std::memory_order_acquire) == 0) out++;
    bool full[MAX_CARDS] = {};
    const size_t last = std::min(g_rc.size(), pos + (size_t)g_lookahead + 1);
    for (size_t i = pos; i < last; i++) {
        const CallSet &cs = g_rc[i]; Handle *hd = g_handles[cs.layer];
        if (!hd || hd->fd < 0) continue;
        CardPool &pool = g_pool[hd->card];
        for (int e : cs.experts) {
            const int key = ekey(cs.layer, e);
            if (pool.ready && pool.c.contains(key)) continue;   // on the card or on its way
            if (pool.ready && !full[hd->card] && out < g_vq_cap) {
                if (vq_issue(pool, hd, key, e, i == pos ? (int)pos : rc_nu_sel(key), prot)) { out++; continue; }
                full[hd->card] = true;
            }
            if (g_pend.count(key) || g_host.c.contains(key)) continue;
            if (g_pend.size() >= g_pend_cap || !g_host.c.worth(key)) continue;
            int vk; float vh;
            unsigned char *slot = g_host.c.acquire(key, g_pend_keys, 1.0f, &vk, &vh);
            Pending *p = new Pending; p->slot = slot;
            g_pend[key] = p; g_pend_keys.insert(key); g_pf_planned++;
            PfJob j; j.fd = hd->fd; j.off = (off_t)e * EXPERT_RAW; j.slot = slot; j.p = p; pf_submit(j);
        }
    }
}

// K3GPU_PREFILL: replaces the router's choice of this call by the route cache's (same experts, same number of tokens per expert; tokens are dealt out round-robin so that
// no token gets an expert twice). Called before route_hook, so g_rc_pos is the number of the current call.
static void force_routes(int L, int n, int *sel)
{
    static bool warned = false;
    auto warn = [&](const char *why) { if (!warned) { warned = true; fprintf(stderr, "K3GPU prefill: call %zu (layer %d): %s; the computed routes are used from here\n", g_rc_pos, L, why); } };
    if (g_rc_pos >= g_rc.size() || g_rc[g_rc_pos].layer != L) { warn("does not match the route cache"); return; }
    const CallSet &cs = g_rc[g_rc_pos];
    if (cs.counts.size() != cs.experts.size()) { warn("the route cache has no counts"); return; }
    long total = 0; bool fits = true;
    for (int c : cs.counts) { total += c; if (c > n) fits = false; }
    if (!fits || total != (long)n * TOP) { warn("counts do not fit this call"); return; }
    std::vector<int> fill(n, 0); long s = 0;
    for (size_t i = 0; i < cs.experts.size(); i++)
        for (int k = 0; k < cs.counts[i]; k++, s++) { const int t = (int)(s % n); sel[(size_t)t * TOP + fill[t]++] = cs.experts[i]; }
}

static void route_hook(int L, const std::vector<int> &uniq)
{
#ifdef K3GPU_PLANNER_BUILT
    if (g_sp.on) {   // the planner reads a layer's fd from its own table: copy the fds the loader threads have opened (the original read the handle's fd at each plan)
        for (int l = 1; l <= N_LAYERS; l++) if (g_handles[l] && g_handles[l]->fd >= 0) g_sp.layer_set_fd(l, g_handles[l]->fd);
        g_sp.route_hook(L, uniq.data(), uniq.size()); g_rc_pos++;
        return;
    }
#endif
    rc_set_cur((long)g_rc_pos);
    if (g_rc_ok) {
        if (g_rc_pos >= g_rc.size() || g_rc[g_rc_pos].layer != L || g_rc[g_rc_pos].experts != uniq) {
            fprintf(stderr, "K3GPU route cache: call %zu (layer %d) does not match the cache; look-ahead stopped\n", g_rc_pos, L);
            g_rc_ok = false; slotcache_set_next_use(nullptr);
        } else { const double tp = wall_now(); if (g_vq && (long)g_rc_pos >= g_vq_from) rc_plan_vq(g_rc_pos, uniq); else rc_plan(g_rc_pos); g_t_plan += wall_now() - tp; }
    }
    g_rc_pos++;
}

static void rc_load(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot read route cache %s\n", path); exit(1); }
    char *line = nullptr; size_t cap = 0;
    while (getline(&line, &cap, f) > 0) {
        char *p = line; char *end;
        CallSet cs; cs.layer = (int)strtol(p, &end, 10); p = end; strtol(p, &end, 10); p = end;
        for (;;) {
            while (*p == ' ') p++;
            if (*p < '0' || *p > '9') break;
            cs.experts.push_back((int)strtol(p, &end, 10)); p = end;
            if (*p == ':') { cs.counts.push_back((int)strtol(p + 1, &end, 10)); p = end; }
        }
        g_rc.push_back(std::move(cs));
    }
    free(line); fclose(f);
    for (size_t i = 0; i < g_rc.size(); i++) for (int e : g_rc[i].experts) g_uses[ekey(g_rc[i].layer, e)].push_back((int)i);
}

static void rc_load_sel(const char *path)
{
#ifdef K3GPU_ROUTETABLE_BUILT
    if (g_rt.on) {
        g_rt.t = g_rt.load(path, EXPERTS);
        const size_t n = g_rt.calls(g_rt.t);
        g_rc.resize(n);
        for (size_t i = 0; i < n; i++) {
            size_t ne = 0, nc = 0;
            g_rc[i].layer = g_rt.layer(g_rt.t, i);
            const int *e = g_rt.experts(g_rt.t, i, &ne), *c = g_rt.counts(g_rt.t, i, &nc);
            if (ne) g_rc[i].experts.assign(e, e + ne);
            if (nc) g_rc[i].counts.assign(c, c + nc);
        }
        return;
    }
#endif
    rc_load(path);
}

#ifdef K3GPU_ROUTETABLE_BUILT
static void routetable_select()
{
    if (sel_mode(SEL_ROUTETABLE, true) == 0) return;
    const char *lib = sel_lib(SEL_ROUTETABLE);
    void *h = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "K3GPU_ROUTETABLE=rust: cannot load %s: %s\n", lib, dlerror()); exit(1); }
#define RT_SYM(f, name) g_rt.f = (decltype(g_rt.f))dlsym(h, name); if (!g_rt.f) { fprintf(stderr, "K3GPU_ROUTETABLE=rust: %s has no %s\n", lib, name); exit(1); }
    RT_SYM(load, "rt_load") RT_SYM(calls, "rt_calls") RT_SYM(layer, "rt_layer") RT_SYM(experts, "rt_experts") RT_SYM(counts, "rt_counts") RT_SYM(set_current, "rt_set_current") RT_SYM(next_use, "rt_next_use")
#undef RT_SYM
    g_rt.on = true;
    sel_announce(SEL_ROUTETABLE, lib);
}
#else
static void routetable_select()
{
    sel_mode(SEL_ROUTETABLE, false);   // cpp: its line; anything else: the build has no routetable.h, stop
}
#endif

// ---- the rest of the host path: K3GPU_POOLINIT (HostCache::init / CardPool::init), K3GPU_SUPPLY (pool_ensure), K3GPU_PLANNER (read-ahead and card-queue planning), K3GPU_LOADER, K3GPU_ENGCFG.
// Each is cpp unless set to rust; "rust" loads the library or stops the run, and the start-up line says which one runs. The libraries and their contracts are in src/rust/<name>.
#ifdef K3GPU_POOLINIT_BUILT
static const bool k_built_poolinit = true;
#else
static const bool k_built_poolinit = false;
#endif
#ifdef K3GPU_SUPPLY_BUILT
static const bool k_built_supply = true;
#else
static const bool k_built_supply = false;
#endif
#ifdef K3GPU_LOADER_BUILT
static const bool k_built_loader = true;
#else
static const bool k_built_loader = false;
#endif
#ifdef K3GPU_ENGCFG_BUILT
static const bool k_built_engcfg = true;
#else
static const bool k_built_engcfg = false;
#endif
#if defined(K3GPU_PLANNER_BUILT) && defined(K3GPU_COPYQ_BUILT) && defined(K3GPU_SLOTCACHE_BUILT) && defined(K3GPU_ROUTETABLE_BUILT)
static const bool k_built_planner = true;
#else
static const bool k_built_planner = false;
#endif

static void engcfg_select()   // first thing in configure(): the other selectors and the settings go through it when it runs
{
    const char *lib = nullptr;
    void *h = comp_select("K3GPU_ENGCFG", "engine config", "libengcfg.so", k_built_engcfg, &lib);
    if (!h) return;
#ifdef K3GPU_ENGCFG_BUILT
    COMP_SYM("K3GPU_ENGCFG", h, g_ec.env_int, "ec_env_int") COMP_SYM("K3GPU_ENGCFG", h, g_ec.settings_from_env, "ec_settings_from_env") COMP_SYM("K3GPU_ENGCFG", h, g_ec.pend_cap, "ec_pend_cap")
    COMP_SYM("K3GPU_ENGCFG", h, g_ec.open_trace, "ec_open_trace") COMP_SYM("K3GPU_ENGCFG", h, g_ec.print_route_cache, "ec_print_route_cache")
    COMP_SYM("K3GPU_ENGCFG", h, g_ec.prefill_start, "ec_prefill_start") COMP_SYM("K3GPU_ENGCFG", h, g_ec.print_start_line, "ec_print_start_line")
    COMP_SYM("K3GPU_ENGCFG", h, g_ec.selector_mode, "ec_selector_mode") COMP_SYM("K3GPU_ENGCFG", h, g_ec.selector_lib, "ec_selector_lib")
    COMP_SYM("K3GPU_ENGCFG", h, g_ec.selector_announce_rust, "ec_selector_announce_rust") COMP_SYM("K3GPU_ENGCFG", h, g_ec.card_of, "ec_card_of")
    COMP_SYM("K3GPU_ENGCFG", h, g_ec.force_routes, "ec_force_routes") COMP_SYM("K3GPU_ENGCFG", h, g_ec.print_summary, "ec_print_summary")
    g_ec.on = true;
    fprintf(stderr, "K3GPU: engine config rust (%s)\n", lib);
#endif
}

static void poolinit_select()   // before g_host.init
{
    const char *lib = nullptr;
    void *h = comp_select("K3GPU_POOLINIT", "pool init", "libpoolinit.so", k_built_poolinit, &lib);
    if (!h) return;
#ifdef K3GPU_POOLINIT_BUILT
    COMP_SYM("K3GPU_POOLINIT", h, g_pi.host_init, "pi_host_init") COMP_SYM("K3GPU_POOLINIT", h, g_pi.pool_init, "pi_pool_init")
    g_pi.on = true;
    fprintf(stderr, "K3GPU: pool init rust (%s)\n", lib);
#endif
}

static void supply_start()   // after g_vq, g_prefill and g_chunks are known
{
    const char *lib = nullptr;
    void *h = comp_select("K3GPU_SUPPLY", "supply", "libsupply.so", k_built_supply, &lib);
    if (!h) return;
#ifdef K3GPU_SUPPLY_BUILT
    COMP_SYM("K3GPU_SUPPLY", h, g_sy.init, "sy_init") COMP_SYM("K3GPU_SUPPLY", h, g_sy.ensure, "sy_ensure") COMP_SYM("K3GPU_SUPPLY", h, g_sy.wb_skipped, "sy_wb_skipped")
    sy_config c = {};
    c.experts = EXPERTS; c.expert_raw = EXPERT_RAW; c.vq = g_vq ? 1 : 0; c.prefill = g_prefill ? 1 : 0; c.chunks = g_chunks; c.slab = g_slab;
    g_sy.init(&g_sy_ops, &c);
    g_sy.on = true;
    fprintf(stderr, "K3GPU: supply rust (%s)\n", lib);
#endif
}

// ---- which loader runs (K3GPU_LOADER): cpp, the default, is loader_submit / loader_wait / open_layer below; rust = libloader.so (src/rust/loader): the per-card threads, the pending-job counter
// and the host side of open_layer (opening the experts file, the prefill slab). What touches CUDA or model code comes back here as callbacks.
struct LdApi {
    bool on = false;
#ifdef K3GPU_LOADER_BUILT
    void (*init)(const ld_config *) = nullptr; void (*submit)(int, void (*)(void *), void *) = nullptr; void (*wait)(void) = nullptr; void (*open_layer)(const ld_layer *) = nullptr;
#endif
};
static LdApi g_ld;
static void ensure_work(int c);   // defined below
#ifdef K3GPU_LOADER_BUILT
static void ld_cb_set_device(void *, int card) { CHECK(cudaSetDevice(card)); }
static void ld_cb_ensure_work(void *, int card) { ensure_work(card); }
static void ld_cb_load_trunk(void *, void *h, const char *ds, int L) { layer_load_trunk(ds, L, ((Handle *)h)->y); }
static void ld_cb_load_tables(void *, void *h, const char *ds, int L) { layer_load_expert_tables(ds, L, ((Handle *)h)->y); }
static void ld_cb_alloc_state(void *, void *hv, unsigned positions)
{
    Handle *h = (Handle *)hv;
    h->mla = h->y.mla;
    if (h->mla) {
        h->mc.cap = positions;
        CHECK(cudaMalloc(&h->mc.kv, (size_t)positions * MLA_EXP * 4)); CHECK(cudaMalloc(&h->mc.lat, (size_t)positions * MLA_KVLAT * 4));
    } else kda_state_alloc(h->kda);
}
static void ld_cb_host_alloc(void *, unsigned char **dst, size_t bytes) { CHECK(cudaHostAlloc((void **)dst, bytes, cudaHostAllocDefault)); }
#endif

static void loader_start()   // after g_prefill is known
{
    const char *lib = nullptr;
    void *h = comp_select("K3GPU_LOADER", "loader", "libloader.so", k_built_loader, &lib);
    if (!h) return;
#ifdef K3GPU_LOADER_BUILT
    COMP_SYM("K3GPU_LOADER", h, g_ld.init, "ld_init") COMP_SYM("K3GPU_LOADER", h, g_ld.submit, "ld_submit") COMP_SYM("K3GPU_LOADER", h, g_ld.wait, "ld_wait")
    COMP_SYM("K3GPU_LOADER", h, g_ld.open_layer, "ld_open_layer")
    ld_config c = {};
    c.cb = { nullptr, ld_cb_set_device, ld_cb_ensure_work, ld_cb_load_trunk, ld_cb_load_tables, ld_cb_alloc_state, ld_cb_host_alloc };
    c.expert_raw = EXPERT_RAW; c.prefill = g_prefill ? 1 : 0; c.slab = g_slab; c.handles = (void **)g_handles;
    g_ld.init(&c);
    g_ld.on = true;
    fprintf(stderr, "K3GPU: loader rust (%s)\n", lib);
#endif
}

#if defined(K3GPU_PLANNER_BUILT) && defined(K3GPU_COPYQ_BUILT) && defined(K3GPU_SLOTCACHE_BUILT) && defined(K3GPU_ROUTETABLE_BUILT)
// what the planner library is given in place of the engine's own calls; the three counters the engine keeps for the other libraries are kept here, since the planner calls them directly
static int sp_rt_next_use(rt_table *t, int key) { g_rt_nu_calls++; return g_rt.next_use(t, key); }
static void sp_enqueue(const cq_job *j) { g_cq_jobs[j->kind].fetch_add(1, std::memory_order_relaxed); g_cq_enqueue(j); }
static unsigned char *sp_sc_acquire(sc_cache *c, int key, const int *prot, size_t n, float h0, int *vk, float *vh) { g_sc_calls[0]++; return g_sc.acquire(c, key, prot, n, h0, vk, vh); }
static unsigned char *sp_sc_try_acquire(sc_cache *c, int key, int nu, const int *prot, size_t n, float h0, int *vk, float *vh) { g_sc_calls[1]++; return g_sc.try_acquire(c, key, nu, prot, n, h0, vk, vh); }
#endif
static void planner_start()   // after the route cache, the host cache, the prefill flag and the copy queue are set up
{
    const char *lib = nullptr;
    void *h = comp_select("K3GPU_PLANNER", "planner", "libplanner.so", k_built_planner, &lib);
    if (!h) return;
#if defined(K3GPU_PLANNER_BUILT) && defined(K3GPU_COPYQ_BUILT) && defined(K3GPU_SLOTCACHE_BUILT) && defined(K3GPU_ROUTETABLE_BUILT)
    if (!g_copyq_rust || !g_sc.on || !g_rt.on) { fprintf(stderr, "K3GPU_PLANNER=rust needs K3GPU_COPYQ, K3GPU_SLOTCACHE and K3GPU_ROUTETABLE on rust too (it calls their libraries directly)\n"); exit(1); }
    COMP_SYM("K3GPU_PLANNER", h, g_sp.init, "sp_init") COMP_SYM("K3GPU_PLANNER", h, g_sp.layer_set, "sp_layer_set") COMP_SYM("K3GPU_PLANNER", h, g_sp.layer_set_fd, "sp_layer_set_fd")
    COMP_SYM("K3GPU_PLANNER", h, g_sp.pool_set, "sp_pool_set") COMP_SYM("K3GPU_PLANNER", h, g_sp.host_set, "sp_host_set") COMP_SYM("K3GPU_PLANNER", h, g_sp.route_hook, "sp_route_hook")
    COMP_SYM("K3GPU_PLANNER", h, g_sp.pend_take, "sp_pend_take") COMP_SYM("K3GPU_PLANNER", h, g_sp.arr_wait, "sp_arr_wait") COMP_SYM("K3GPU_PLANNER", h, g_sp.host_acquire, "sp_host_acquire")
    COMP_SYM("K3GPU_PLANNER", h, g_sp.get_stats, "sp_get_stats")
    sp_ops ops = {};
    ops.sc_set_next_use = slotcache_set_next_use; ops.sc_contains = g_sc.contains; ops.sc_worth = g_sc.worth; ops.sc_acquire = sp_sc_acquire; ops.sc_try_acquire = sp_sc_try_acquire;
    ops.sc_remove = g_sc.remove; ops.sc_release = g_sc.release;
    ops.table = g_rt.t; ops.rt_calls = g_rt.calls; ops.rt_layer = g_rt.layer; ops.rt_experts = g_rt.experts; ops.rt_set_current = g_rt.set_current; ops.rt_next_use = sp_rt_next_use;
    ops.enqueue = sp_enqueue;
    sp_config cfg = {};
    cfg.experts = EXPERTS; cfg.expert_raw = EXPERT_RAW; cfg.cards = g_cards; cfg.lookahead = g_lookahead; cfg.pend_cap = g_pend_cap; cfg.vq = g_vq ? 1 : 0; cfg.vq_cap = g_vq_cap;
    cfg.vq_from = g_vq_from; cfg.prefill = g_prefill ? 1 : 0; cfg.rc_ok = g_rc_ok ? 1 : 0;
    g_sp.init(&ops, &cfg);
    g_sp.host_set(g_host.c.R(), g_host.cap_slots);
    g_sp.on = true;
    fprintf(stderr, "K3GPU: planner rust (%s)\n", lib);
#endif
}
// the planner library keeps the counters the engine's own planner kept in globals; the summary and the verbose lines read the globals
static void planner_sync()
{
#ifdef K3GPU_PLANNER_BUILT
    if (!g_sp.on) return;
    sp_stats s; memset(&s, 0, sizeof s); g_sp.get_stats(&s);
    g_pf_planned = s.planned; g_pf_skipped = s.skipped; g_pf_used = s.pf_used; g_pf_waited = s.pf_waited; g_t_pwait = s.t_pwait;
    g_vq_ram = s.vq_ram; g_vq_drive = s.vq_drive; g_vq_wb = s.vq_wb; g_vq_nowb = s.vq_nowb; g_vq_waited = s.vq_waited; g_t_vwait = s.t_vwait; g_t_plan = s.t_plan; g_rc_ok = s.rc_ok != 0;
#endif
}
static unsigned long wb_skipped_total()
{
#ifdef K3GPU_SUPPLY_BUILT
    if (g_sy.on) return g_sy.wb_skipped();
#endif
    return g_wb_skipped;
}

static void print_summary_cpp()
{
    unsigned long need = 0, hc = 0, hh = 0, dr = 0; double tr = 0, tc = 0;
    for (int c = 0; c < g_cards; c++) {
        if (!g_pool[c].ready) continue;
        const ExpertStats &s = g_pool[c].st;
        need += s.needed; hc += s.hit_card; hh += s.hit_host; dr += s.from_drive; tr += s.t_read; tc += s.t_copy;
    }
    fprintf(stderr, "K3GPU copy queue %s: jobs sent %lu drive->RAM, %lu drive->card, %lu RAM->card\n", g_copyq_rust ? "rust" : "cpp", g_cq_jobs[0].load(), g_cq_jobs[1].load(), g_cq_jobs[2].load());
    fprintf(stderr, "K3GPU slot cache %s: %lu acquire, %lu try_acquire, %lu find\n", g_sc.on ? "rust" : "cpp", g_sc_calls[0], g_sc_calls[1], g_sc_calls[2]);
    fprintf(stderr, "K3GPU route table %s: %zu calls, %lu next_use calls\n", g_rt.on ? "rust" : "cpp", g_rc.size(), g_rt_nu_calls);
    if (!need) return;
    fprintf(stderr, "K3GPU experts: %lu needed, %lu already on the card (%.1f%%), %lu from the host cache (%.1f%%), %lu from the drive (%.1f%%, %.1f GB); drive reads %.1f s, host to card %.1f s\n",
        need, hc, 100.0 * hc / need, hh, 100.0 * hh / need, dr, 100.0 * dr / need, dr * (double)EXPERT_RAW / 1e9, tr, tc);
    for (int c = 0; c < g_cards; c++)
        if (g_pool[c].ready) fprintf(stderr, "K3GPU card %d: %d expert slots, %lu needed, %lu on the card, %lu host, %lu drive, %lu written back to RAM\n", c, g_pool[c].slots, g_pool[c].st.needed, g_pool[c].st.hit_card, g_pool[c].st.hit_host, g_pool[c].st.from_drive, g_pool[c].st.written_back);
    const char *names[3] = { "prompt-pass layer calls", "single-token layer calls", "multi-token layer calls" };
    if (!g_rc.empty()) fprintf(stderr, "K3GPU route cache: %zu of %zu calls seen, look-ahead %s; %lu experts read ahead, %lu already resident, %lu taken from read-ahead, %lu of them waited for (%.1f s); %lu card victims not kept in RAM\n", g_rc_pos, g_rc.size(), g_rc_ok ? "on to the end" : "stopped", g_pf_planned, g_pf_skipped, g_pf_used, g_pf_waited, g_t_pwait, wb_skipped_total());
    if (g_vq) fprintf(stderr, "K3GPU card queues: %lu experts brought ahead (%lu from RAM, %lu from the drive), %lu replaced experts copied to RAM, %lu dropped; %lu waited for (%.1f s); planning in the hook %.1f s\n", g_vq_ram + g_vq_drive, g_vq_ram, g_vq_drive, g_vq_wb, g_vq_nowb, g_vq_waited, g_t_vwait, g_t_plan);
    for (int i = 0; i < 3; i++) if (g_n_calls[i]) fprintf(stderr, "K3GPU %s: %.0f, %.3f s in all, %.1f ms each\n", names[i], g_n_calls[i], g_t_calls[i], 1000 * g_t_calls[i] / g_n_calls[i]);
}

static void print_summary()
{
    planner_sync();
    fprintf(stderr, "K3GPU host path: engine config %s, pool init %s, supply %s, planner %s, loader %s\n", g_ec.on ? "rust" : "cpp", g_pi.on ? "rust" : "cpp", g_sy.on ? "rust" : "cpp", g_sp.on ? "rust" : "cpp", g_ld.on ? "rust" : "cpp");
#ifdef K3GPU_ENGCFG_BUILT
    if (g_ec.on) {
        ec_summary s; memset(&s, 0, sizeof s);
        s.cards = g_cards; s.expert_raw = EXPERT_RAW;
        for (int c = 0; c < g_cards && c < EC_MAX_CARDS; c++) {
            const ExpertStats &e = g_pool[c].st;
            s.card[c].ready = g_pool[c].ready; s.card[c].slots = g_pool[c].slots; s.card[c].needed = e.needed; s.card[c].hit_card = e.hit_card; s.card[c].hit_host = e.hit_host;
            s.card[c].from_drive = e.from_drive; s.card[c].written_back = e.written_back; s.card[c].t_read = e.t_read; s.card[c].t_copy = e.t_copy;
        }
        s.copyq_rust = g_copyq_rust; for (int k = 0; k < 3; k++) { s.cq_jobs[k] = g_cq_jobs[k].load(); s.sc_calls[k] = g_sc_calls[k]; s.n_calls[k] = g_n_calls[k]; s.t_calls[k] = g_t_calls[k]; }
        s.slotcache_rust = g_sc.on; s.routetable_rust = g_rt.on; s.rc_calls = g_rc.size(); s.rt_nu_calls = g_rt_nu_calls;
        s.rc_pos = g_rc_pos; s.rc_ok = g_rc_ok;
        s.pf_planned = g_pf_planned; s.pf_skipped = g_pf_skipped; s.pf_used = g_pf_used; s.pf_waited = g_pf_waited; s.t_pwait = g_t_pwait; s.wb_skipped = wb_skipped_total();
        s.vq = g_vq; s.vq_ram = g_vq_ram; s.vq_drive = g_vq_drive; s.vq_wb = g_vq_wb; s.vq_nowb = g_vq_nowb; s.vq_waited = g_vq_waited; s.t_vwait = g_t_vwait; s.t_plan = g_t_plan;
        g_ec.print_summary(&s);
        return;
    }
#endif
    print_summary_cpp();
}

#ifdef K3GPU_ENGCFG_BUILT
// force_routes of the engcfg library: it is given the one entry of the route cache it reads
static void force_routes_ec(int L, int n, int *sel)
{
    ec_call cur = {};
    const bool have = g_rc_pos < g_rc.size();
    if (have) { const CallSet &cs = g_rc[g_rc_pos]; cur.layer = cs.layer; cur.experts = cs.experts.data(); cur.n_experts = cs.experts.size(); cur.counts = cs.counts.data(); cur.n_counts = cs.counts.size(); }
    g_ec.force_routes(g_rc_pos, g_rc.size(), have ? &cur : nullptr, L, n, sel, TOP);
}
// configure() with K3GPU_ENGCFG=rust: the same steps in the same order (the order is part of engcfg.h), the settings and messages from the library
static void configure_ec(int have)
{
    ec_settings s; g_ec.settings_from_env(&s, have);
    g_cards = s.cards; g_readers = s.readers; g_verbose = s.verbose; g_reserve = s.reserve;
    slotcache_select();
    slotcache_set_policy(s.policy);
    routetable_select();
    poolinit_select();
    g_chunks = s.chunks;
    g_trace = g_ec.open_trace();
    g_host.init(s.host_bytes);
    if (s.routecache) {
        rc_load_sel(s.routecache);
        g_lookahead = s.lookahead;
        g_pend_cap = g_ec.pend_cap(g_host.cap_slots);
        g_route_hook = route_hook; g_rc_ok = !g_rc.empty();
        if (g_policy == 2) slotcache_set_next_use(rc_nu_sel);
        g_vq = s.vqueue != 0; g_vq_cap = s.vq_cap; g_vq_from = s.vq_from;
        g_ec.print_route_cache(&s, g_rc.size(), g_pend_cap);
    }
    g_layout = s.layout; g_prefill = s.prefill != 0;
    if (s.prefill) { g_ec.prefill_start(g_rc.size()); g_force_routes = force_routes_ec; }
    copyq_select();
    supply_start(); loader_start(); planner_start();
    atexit(print_summary);
    g_ec.print_start_line(&s);
}
#endif

static void configure()
{
    if (g_configured) return;
    g_configured = true;
    engcfg_select();
    int have = 0; CHECK(cudaGetDeviceCount(&have));
#ifdef K3GPU_ENGCFG_BUILT
    if (g_ec.on) { configure_ec(have); return; }
#endif
    g_cards = std::min(std::min(env_int("K3GPU_CARDS", 8), have), (int)MAX_CARDS);
    g_readers = env_int("K3GPU_READERS", 16);
    g_verbose = env_int("K3GPU_VERBOSE", 0);
    g_reserve = (size_t)env_int("K3GPU_RESERVE_MB", 2500) << 20;
    slotcache_select();
    slotcache_set_policy(env_int("K3GPU_POLICY", 1));
    routetable_select();
    poolinit_select();
    g_chunks = env_int("K3GPU_CHUNKS", 0);
    { const char *tr = getenv("K3GPU_TRACE"); if (tr && *tr) { g_trace = fopen(tr, "w"); if (!g_trace) { fprintf(stderr, "cannot write %s\n", tr); exit(1); } } }
    g_host.init((size_t)env_int("K3GPU_HOST_GB", 150) << 30);
    { const char *rc = getenv("K3GPU_ROUTECACHE");
      if (rc && *rc) {
          rc_load_sel(rc);
          g_lookahead = env_int("K3GPU_LOOKAHEAD", 92);
          g_pend_cap = g_host.cap_slots * 6 / 10;
          g_route_hook = route_hook; g_rc_ok = !g_rc.empty();
          if (g_policy == 2) slotcache_set_next_use(rc_nu_sel);
          g_vq = env_int("K3GPU_VQUEUE", 0) != 0 && g_policy == 2;
          g_vq_cap = (size_t)env_int("K3GPU_VQ_INFLIGHT", 256);
          g_vq_from = env_int("K3GPU_VQ_FROM", 0);
          fprintf(stderr, "K3GPU route cache: %zu calls from %s, look-ahead %d calls, at most %zu experts read ahead, card queues %s (at most %zu in flight)\n", g_rc.size(), rc, g_lookahead, g_pend_cap, g_vq ? "on" : "off", g_vq_cap);
      } }
    g_layout = env_int("K3GPU_LAYOUT", 0);
    g_prefill = env_int("K3GPU_PREFILL", 0) != 0;
    if (g_prefill) {
        if (g_rc.empty()) { fprintf(stderr, "K3GPU_PREFILL needs K3GPU_ROUTECACHE\n"); exit(1); }
        g_force_routes = force_routes;
        fprintf(stderr, "K3GPU PREFILL ON: timing only. No drive is read, expert bytes are 4 records per layer, routes are taken from the route cache; outputs are not the model's.\n");
    }
    copyq_select();
    supply_start(); loader_start(); planner_start();
    atexit(print_summary);
    fprintf(stderr, "K3GPU: %d cards, host expert cache %d GB, %d drive readers\n", g_cards, env_int("K3GPU_HOST_GB", 150), g_readers);
}

static int card_of(int L)
{
#ifdef K3GPU_ENGCFG_BUILT
    if (g_ec.on) return g_ec.card_of(g_layout, g_cards, L);
#endif
    if (g_layout == 1) return L == 0 ? 0 : (L - 1) % g_cards; return L == 0 ? 0 : std::min(g_cards - 1, (L * g_cards) / 93);
}

static void ensure_work(int c)
{
    if (g_card[c].ready) return;
    CHECK(cudaSetDevice(c));
    work_alloc(g_card[c].w, c, (int)g_max_n, c == 0, false, 8, 3);
    g_card[c].alloc_n = g_max_n; g_card[c].ready = true;
}

// the card's pool of expert slots is made by the first layer with experts that runs on the card; the planner library is told when it is ready
static void pool_open(int card)
{
    if (g_pool[card].ready) return;
    g_pool[card].init(card, g_reserve, g_readers);
#ifdef K3GPU_PLANNER_BUILT
    if (g_sp.on) g_sp.pool_set(card, g_pool[card].c.R(), g_pool[card].dev, 1);
#endif
}
// the planner library has its own table of the layers' handles: the layer exists from open_layer on, its fd (-1 here) is copied in route_hook once the loader has opened it
static void plan_note_layer(const Handle *h)
{
#ifdef K3GPU_PLANNER_BUILT
    if (g_sp.on && h->L <= N_LAYERS) g_sp.layer_set(h->L, -1, h->card);
#endif
}

// Layers are opened by one loader thread per card, so the trunks of the cards load side by side; open_layer returns at once and the work that needs the
// layer (a forward pass, the state calls) first waits for every queued load.
struct Loader {
    std::mutex mu; std::condition_variable cv; std::deque<std::function<void()>> q; std::thread th; bool started = false;
};
static Loader *g_loader[MAX_CARDS];   // never destroyed: a loader thread waits for work until the process ends
static std::mutex &pending_mu() { static std::mutex *m = new std::mutex; return *m; }
static std::condition_variable &pending_cv() { static std::condition_variable *c = new std::condition_variable; return *c; }
static int g_pending = 0;
static void loader_submit(int card, std::function<void()> job)
{
#ifdef K3GPU_LOADER_BUILT
    if (g_ld.on) {
        g_ld.submit(card, [](void *arg) { std::function<void()> *f = (std::function<void()> *)arg; (*f)(); delete f; }, new std::function<void()>(std::move(job)));
        return;
    }
#endif
    if (!g_loader[card]) g_loader[card] = new Loader;
    Loader &l = *g_loader[card];
    { std::lock_guard<std::mutex> g(pending_mu()); g_pending++; }
    std::lock_guard<std::mutex> g(l.mu);
    if (!l.started) {
        l.started = true;
        l.th = std::thread([card, &l]() {
            CHECK(cudaSetDevice(card));
            for (;;) {
                std::function<void()> j;
                { std::unique_lock<std::mutex> u(l.mu); l.cv.wait(u, [&]() { return !l.q.empty(); }); j = std::move(l.q.front()); l.q.pop_front(); }
                j();
                { std::lock_guard<std::mutex> g2(pending_mu()); g_pending--; }
                pending_cv().notify_all();
            }
        });
        l.th.detach();
    }
    l.q.push_back(std::move(job));
    l.cv.notify_one();
}
static void loader_wait()
{
#ifdef K3GPU_LOADER_BUILT
    if (g_ld.on) { g_ld.wait(); return; }
#endif
    std::unique_lock<std::mutex> u(pending_mu());
    pending_cv().wait(u, []() { return g_pending == 0; });
}

static Handle *open_layer(const char *dataset, unsigned L, unsigned k, unsigned positions, Handle *into)
{
    configure();
    Handle *h = into ? into : new Handle;
#ifdef K3GPU_LOADER_BUILT
    if (g_ld.on) {
        ld_layer req = {};
        req.handle = h; req.dataset = dataset; req.layer = L; req.k = k; req.positions = positions; req.card = card_of((int)L);
        req.h_L = &h->L; req.h_card = &h->card; req.h_k_in = &h->k_in; req.h_fd = &h->fd;
        g_ld.open_layer(&req);   // the bookkeeping and the job below, in the library; g_handles is the table it fills
        plan_note_layer(h);
        return h;
    }
#endif
    h->L = (int)L; h->card = card_of((int)L); h->k_in = (int)k;
    if (L <= (unsigned)N_LAYERS) g_handles[L] = h;
    plan_note_layer(h);
    const std::string ds(dataset);
    loader_submit(h->card, [h, ds, L, positions]() {
        CHECK(cudaSetDevice(h->card));
        ensure_work(h->card);
        layer_load_trunk(ds.c_str(), (int)L, h->y);
        if (L >= 1) layer_load_expert_tables(ds.c_str(), (int)L, h->y);
        h->mla = h->y.mla;
        if (h->mla) {
            h->mc.cap = positions;
            CHECK(cudaMalloc(&h->mc.kv, (size_t)positions * MLA_EXP * 4)); CHECK(cudaMalloc(&h->mc.lat, (size_t)positions * MLA_KVLAT * 4));
        } else kda_state_alloc(h->kda);
        if (L >= 1) {
            char path[4096]; snprintf(path, sizeof path, "%s/expert-%u/experts.direct", ds.c_str(), L);
            h->fd = open(path, O_RDONLY | O_DIRECT);
            if (h->fd < 0) h->fd = open(path, O_RDONLY);
            if (h->fd < 0) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
            if (g_prefill) {   // 4 real records of this layer stand in for every expert of it
                CHECK(cudaHostAlloc((void **)&g_slab[L], (size_t)4 * EXPERT_RAW, cudaHostAllocDefault));
                for (int r = 0; r < 4; r++) {
                    size_t got = 0;
                    while (got < (size_t)EXPERT_RAW) {
                        const ssize_t rd = pread(h->fd, g_slab[L] + (size_t)r * EXPERT_RAW + got, EXPERT_RAW - got, (off_t)r * EXPERT_RAW + (off_t)got);
                        if (rd <= 0) { fprintf(stderr, "prefill: cannot read a record of layer %u\n", L); exit(1); }
                        got += (size_t)rd;
                    }
                }
            }
        }
    });
    return h;
}

// n rows through one layer: fresh = a new sequence (state reset), pos0 = position of the first row
static double forward(Handle &h, const float *resid, const float *const *snaps, unsigned n, size_t pos0, bool fresh, float *out)
{
    loader_wait();
    const double t0 = wall_now();
    Card &c = g_card[h.card];
    CHECK(cudaSetDevice(h.card));
    if (n > c.alloc_n) { fprintf(stderr, "a window of %u rows exceeds the %u the buffers were made for\n", n, c.alloc_n); exit(1); }
    Work &w = c.w; w.N = (int)n;
    if (h.L >= 1) pool_open(h.card);
    if (fresh && !h.mla) kda_state_reset(h.kda);
    const float *sn[9];
    if (h.L == 0) {
        CHECK(cudaMemcpy(w.snap[0], resid, (size_t)n * W * 4, cudaMemcpyHostToDevice));
        run_layer0_state(w, h.y, h.kda, w.snap[0], w.nxt);
    } else {
        const double ta = wall_now();
        g_act_hold.fetch_add(1);
        CHECK(cudaMemcpy(w.cur, resid, (size_t)n * W * 4, cudaMemcpyHostToDevice));
        for (int i = 0; i < h.k_in; i++) { CHECK(cudaMemcpy(w.snap[i], snaps[i], (size_t)n * W * 4, cudaMemcpyHostToDevice)); sn[i] = w.snap[i]; }
        g_act_hold.fetch_sub(1);
        const double tb = wall_now(); g_t_up += tb - ta;
        run_layer_state(w, h.y, h.mla ? nullptr : &h.kda, h.mla ? &h.mc : nullptr, pos0, g_pool[h.card], g_host, h.fd, g_readers, w.cur, sn, h.k_in, w.nxt);
        g_t_run += wall_now() - tb;
    }
    CHECK(cudaGetLastError());
    const double td = wall_now();
    CHECK(cudaMemcpy(out, w.nxt, (size_t)n * W * 4, cudaMemcpyDeviceToHost));
    g_t_down += wall_now() - td;
    g_last_layer = h.L;
    if (g_verbose >= 2 && h.L == N_LAYERS) {   // one line per pass: the experts it needed and where they came from (since the previous line)
        static unsigned long pn = 0, pc = 0, ph = 0, pd = 0; static double pe = 0, pr = 0, pcp = 0, pt = 0, ppl = 0;
        unsigned long cn = 0, cc = 0, ch = 0, cd = 0; double ce = 0, cr = 0, ccp = 0;
        for (int i = 0; i < g_cards; i++) if (g_pool[i].ready) { cn += g_pool[i].st.needed; cc += g_pool[i].st.hit_card; ch += g_pool[i].st.hit_host; cd += g_pool[i].st.from_drive; ce += g_pool[i].st.t_ensure; cr += g_pool[i].st.t_read; ccp += g_pool[i].st.t_copy; }
        const double ct = g_t_calls[0] + g_t_calls[1] + g_t_calls[2] + (wall_now() - t0);
        planner_sync();
        fprintf(stderr, "K3GPU pass of %u row(s): %lu experts needed, %lu on the card, %lu in host RAM, %lu from the drive; layer calls %.3f s of which expert supply %.3f s (drive reads %.3f s, host to card %.3f s)\n", n, cn - pn, cc - pc, ch - ph, cd - pd, ct - pt, ce - pe, cr - pr, ccp - pcp);
        pn = cn; pc = cc; ph = ch; pd = cd; pe = ce; pr = cr; pcp = ccp; pt = ct;
        if (g_vq) { fprintf(stderr, "K3GPU planning in the hook: %.3f s in this pass\n", g_t_plan - ppl); ppl = g_t_plan; }
        { static double pu = 0, pru = 0, pdn = 0; fprintf(stderr, "K3GPU pass phases: activations up %.3f s, layer run (includes expert supply) %.3f s, activations down incl. last kernels %.3f s\n", g_t_up - pu, g_t_run - pru, g_t_down - pdn); pu = g_t_up; pru = g_t_run; pdn = g_t_down; }
    }
    const double dt = wall_now() - t0;
    const int kind = (n == 1 && !fresh) ? 1 : (!fresh && pos0 > 0) ? 2 : 0;
    g_t_calls[kind] += dt; g_n_calls[kind] += 1;
    if (g_verbose && n == 1) fprintf(stderr, "K3GPU layer %d card %d: %.1f ms\n", h.L, h.card, 1000 * dt);
    return dt;
}

extern "C" {

unsigned char *backend_expert_memory(size_t bytes) { return (unsigned char *)calloc(1, bytes); }

// ---- stage 2: layer 0 (it keeps its own state; the driver calls it before the other layers)
double stage2_setup(const char *dataset, unsigned max_n)
{
    const double t0 = wall_now();
    configure();
    g_max_n = std::max(g_max_n, max_n);
    open_layer(dataset, 0, 1, 0, &g_l0);
    return wall_now() - t0;
}
void stage2_run(const uint32_t *, const float *emb, unsigned n, unsigned, float *out) { forward(g_l0, emb, nullptr, n, 0, true, out); }
void stage2_run_from(const uint32_t *, const float *emb, unsigned n, unsigned, float *out) { forward(g_l0, emb, nullptr, n, 1, false, out); }
void stage2_step(const uint32_t *, const float *emb, float *out) { forward(g_l0, emb, nullptr, 1, 1, false, out); }
void stage2_stepk(const uint32_t *, const float *emb, unsigned n, float *out) { forward(g_l0, emb, nullptr, n, 1, false, out); }
void stage2_keep_positions(unsigned count) { loader_wait(); CHECK(cudaSetDevice(g_l0.card)); kda_state_keep_alloc(g_l0.kda, (int)count); }
void stage2_use_position(unsigned position) { loader_wait(); CHECK(cudaSetDevice(g_l0.card)); kda_state_use(g_l0.kda, (int)position); }

// ---- stage 3: layers 1..92. One implementation, two sets of names (the driver names the kind of every layer it opens).
#define IMPL(P) \
    void P##_laps(double *out) { for (int i = 0; i < 24; i++) out[i] = 0; } \
    void P##_guess_stats(unsigned long *issued, unsigned long *used) { *issued = 0; *used = 0; } \
    void P##_set_slots(void *, unsigned char *const *, unsigned) {} \
    void P##_route_log(int) {} \
    void P##_routes(unsigned short *out) { for (int i = 0; i < 48; i++) out[i] = i < 16 ? g_last_routes[g_last_layer][i] : 0; } \
    void P##_hits(unsigned long *out) { for (int i = 0; i < 896; i++) out[i] = 0; } \
    void P##_set_buffer(unsigned char *) {} \
    double P##_init(unsigned max_n) { g_max_n = std::max(g_max_n, max_n); return 0.0; } \
    void *P##_open(const char *dataset, unsigned layer, unsigned k, unsigned max_n) { return open_layer(dataset, layer, k, max_n, nullptr); } \
    double P##_run(void *h, const float *resid, const float *const *snaps, unsigned n, unsigned, float *out) { return forward(*(Handle *)h, resid, snaps, n, 0, true, out); } \
    double P##_run_from(void *h, const float *resid, const float *const *snaps, unsigned n, unsigned, size_t pos0, float *out) { return forward(*(Handle *)h, resid, snaps, n, pos0, false, out); } \
    double P##_step(void *h, const float *resid, const float *const *snaps, size_t pos0, float *out) { return forward(*(Handle *)h, resid, snaps, 1, pos0, false, out); } \
    double P##_stepk(void *h, const float *resid, const float *const *snaps, unsigned n, size_t pos0, float *out) { return forward(*(Handle *)h, resid, snaps, n, pos0, false, out); } \
    void P##_set_scratch(unsigned char *const *, unsigned) {} \
    unsigned long P##_distinct(void) { return 0; } \
    void P##_keep_positions(void *h, unsigned count) { loader_wait(); Handle &x = *(Handle *)h; if (!x.mla) { CHECK(cudaSetDevice(x.card)); kda_state_keep_alloc(x.kda, (int)count); } } \
    void P##_use_position(void *h, unsigned position) { loader_wait(); Handle &x = *(Handle *)h; if (!x.mla) { CHECK(cudaSetDevice(x.card)); kda_state_use(x.kda, (int)position); } }
IMPL(lbk)
IMPL(lbm)

void lbk_cache_stats(unsigned long *hit, unsigned long *miss)
{
    unsigned long h = 0, m = 0;
    for (int c = 0; c < g_cards; c++) if (g_pool[c].ready) { h += g_pool[c].st.hit_card + g_pool[c].st.hit_host; m += g_pool[c].st.from_drive; }
    *hit = h; *miss = m;
}
void lbm_cache_stats(unsigned long *hit, unsigned long *miss) { *hit = 0; *miss = 0; }

}   // extern "C"
