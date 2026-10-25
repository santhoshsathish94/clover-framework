// A GPU backend for whole-model.c (see common/backend.h): the same functions the CPU backend defines (stage2_* for layer 0, lbk_* for KDA layers, lbm_* for MLA
// layers), run on the cards of one machine. Layers are spread over the cards in consecutive groups (K3GPU_CARDS, default 8); a layer's trunk and its state stay
// on its card, activations travel through host memory (7168 floats per token and snapshot). The experts are not held: after each layer's router, the chosen ones are
// brought to the card's pool of expert slots (decode.cuh), from the host cache or the drive.
//
// Settings: K3GPU_CARDS (8), K3GPU_HOST_GB (pinned RAM for the expert cache, 150), K3GPU_RESERVE_MB (card memory kept free of expert slots, 2500),
// K3GPU_READERS (drive reads at once, 16), K3GPU_VERBOSE (1: one line per layer call of a single token),
// K3GPU_COPYQ (cpp, the default, or rust: which implementation runs the copy queue) and K3GPU_COPYQ_LIB (path of libcopyq.so, default libcopyq.so),
// K3GPU_SLOTCACHE (cpp, the default, or rust: which implementation runs the expert slot caches) and K3GPU_SLOTCACHE_LIB (path of libslotcache.so, default libslotcache.so).
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
    const char *m = getenv("K3GPU_COPYQ");
    if (!m || !*m || !strcmp(m, "cpp")) { fprintf(stderr, "K3GPU: copy queue cpp\n"); return; }
    if (strcmp(m, "rust")) { fprintf(stderr, "K3GPU_COPYQ must be cpp or rust, not %s\n", m); exit(1); }
    const char *lib = getenv("K3GPU_COPYQ_LIB"); if (!lib || !*lib) lib = "libcopyq.so";
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
    fprintf(stderr, "K3GPU: copy queue rust (%s)\n", lib);
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
    const char *m = getenv("K3GPU_COPYQ");
    if (m && *m && strcmp(m, "cpp")) { fprintf(stderr, "K3GPU_COPYQ=%s: this build has no copyq.h, only the C++ queue (build with -I src/rust/copyq/include)\n", m); exit(1); }
    fprintf(stderr, "K3GPU: copy queue cpp\n");
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
    const char *m = getenv("K3GPU_SLOTCACHE");
    if (!m || !*m || !strcmp(m, "cpp")) { fprintf(stderr, "K3GPU: slot cache cpp\n"); return; }
    if (strcmp(m, "rust")) { fprintf(stderr, "K3GPU_SLOTCACHE must be cpp or rust, not %s\n", m); exit(1); }
    const char *lib = getenv("K3GPU_SLOTCACHE_LIB"); if (!lib || !*lib) lib = "libslotcache.so";
    void *h = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "K3GPU_SLOTCACHE=rust: cannot load %s: %s\n", lib, dlerror()); exit(1); }
#define SC_SYM(f, name) g_sc.f = (decltype(g_sc.f))dlsym(h, name); if (!g_sc.f) { fprintf(stderr, "K3GPU_SLOTCACHE=rust: %s has no %s\n", lib, name); exit(1); }
    SC_SYM(set_policy, "sc_set_policy") SC_SYM(set_next_use, "sc_set_next_use") SC_SYM(nw, "sc_new") SC_SYM(add_spare, "sc_add_spare") SC_SYM(contains, "sc_contains")
    SC_SYM(find, "sc_find") SC_SYM(touch, "sc_touch") SC_SYM(worth, "sc_worth") SC_SYM(acquire, "sc_acquire") SC_SYM(try_acquire, "sc_try_acquire")
    SC_SYM(remove, "sc_remove") SC_SYM(release, "sc_release")
#undef SC_SYM
    g_sc.on = true;
    fprintf(stderr, "K3GPU: slot cache rust (%s)\n", lib);
}
#else
static void slotcache_select()
{
    const char *m = getenv("K3GPU_SLOTCACHE");
    if (m && *m && strcmp(m, "cpp")) { fprintf(stderr, "K3GPU_SLOTCACHE=%s: this build has no slotcache.h, only the C++ cache (build with -I src/rust/slotcache/include)\n", m); exit(1); }
    fprintf(stderr, "K3GPU: slot cache cpp\n");
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
    const char *m = getenv("K3GPU_ROUTETABLE");
    if (!m || !*m || !strcmp(m, "cpp")) { fprintf(stderr, "K3GPU: route table cpp\n"); return; }
    if (strcmp(m, "rust")) { fprintf(stderr, "K3GPU_ROUTETABLE must be cpp or rust, not %s\n", m); exit(1); }
    const char *lib = getenv("K3GPU_ROUTETABLE_LIB"); if (!lib || !*lib) lib = "libroutetable.so";
    void *h = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "K3GPU_ROUTETABLE=rust: cannot load %s: %s\n", lib, dlerror()); exit(1); }
#define RT_SYM(f, name) g_rt.f = (decltype(g_rt.f))dlsym(h, name); if (!g_rt.f) { fprintf(stderr, "K3GPU_ROUTETABLE=rust: %s has no %s\n", lib, name); exit(1); }
    RT_SYM(load, "rt_load") RT_SYM(calls, "rt_calls") RT_SYM(layer, "rt_layer") RT_SYM(experts, "rt_experts") RT_SYM(counts, "rt_counts") RT_SYM(set_current, "rt_set_current") RT_SYM(next_use, "rt_next_use")
#undef RT_SYM
    g_rt.on = true;
    fprintf(stderr, "K3GPU: route table rust (%s)\n", lib);
}
#else
static void routetable_select()
{
    const char *m = getenv("K3GPU_ROUTETABLE");
    if (m && *m && strcmp(m, "cpp")) { fprintf(stderr, "K3GPU_ROUTETABLE=%s: this build has no routetable.h, only the C++ parser (build with -I src/rust/routetable/include)\n", m); exit(1); }
    fprintf(stderr, "K3GPU: route table cpp\n");
}
#endif

static void print_summary()
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
    if (!g_rc.empty()) fprintf(stderr, "K3GPU route cache: %zu of %zu calls seen, look-ahead %s; %lu experts read ahead, %lu already resident, %lu taken from read-ahead, %lu of them waited for (%.1f s); %lu card victims not kept in RAM\n", g_rc_pos, g_rc.size(), g_rc_ok ? "on to the end" : "stopped", g_pf_planned, g_pf_skipped, g_pf_used, g_pf_waited, g_t_pwait, g_wb_skipped);
    if (g_vq) fprintf(stderr, "K3GPU card queues: %lu experts brought ahead (%lu from RAM, %lu from the drive), %lu replaced experts copied to RAM, %lu dropped; %lu waited for (%.1f s); planning in the hook %.1f s\n", g_vq_ram + g_vq_drive, g_vq_ram, g_vq_drive, g_vq_wb, g_vq_nowb, g_vq_waited, g_t_vwait, g_t_plan);
    for (int i = 0; i < 3; i++) if (g_n_calls[i]) fprintf(stderr, "K3GPU %s: %.0f, %.3f s in all, %.1f ms each\n", names[i], g_n_calls[i], g_t_calls[i], 1000 * g_t_calls[i] / g_n_calls[i]);
}

static void configure()
{
    if (g_configured) return;
    g_configured = true;
    int have = 0; CHECK(cudaGetDeviceCount(&have));
    g_cards = std::min(std::min(env_int("K3GPU_CARDS", 8), have), (int)MAX_CARDS);
    g_readers = env_int("K3GPU_READERS", 16);
    g_verbose = env_int("K3GPU_VERBOSE", 0);
    g_reserve = (size_t)env_int("K3GPU_RESERVE_MB", 2500) << 20;
    slotcache_select();
    slotcache_set_policy(env_int("K3GPU_POLICY", 1));
    routetable_select();
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
    atexit(print_summary);
    fprintf(stderr, "K3GPU: %d cards, host expert cache %d GB, %d drive readers\n", g_cards, env_int("K3GPU_HOST_GB", 150), g_readers);
}

static int card_of(int L) { if (g_layout == 1) return L == 0 ? 0 : (L - 1) % g_cards; return L == 0 ? 0 : std::min(g_cards - 1, (L * g_cards) / 93); }

static void ensure_work(int c)
{
    if (g_card[c].ready) return;
    CHECK(cudaSetDevice(c));
    work_alloc(g_card[c].w, c, (int)g_max_n, c == 0, false, 8, 3);
    g_card[c].alloc_n = g_max_n; g_card[c].ready = true;
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
    std::unique_lock<std::mutex> u(pending_mu());
    pending_cv().wait(u, []() { return g_pending == 0; });
}

static Handle *open_layer(const char *dataset, unsigned L, unsigned k, unsigned positions, Handle *into)
{
    configure();
    Handle *h = into ? into : new Handle;
    h->L = (int)L; h->card = card_of((int)L); h->k_in = (int)k;
    if (L <= (unsigned)N_LAYERS) g_handles[L] = h;
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
    if (h.L >= 1 && !g_pool[h.card].ready) g_pool[h.card].init(h.card, g_reserve, g_readers);
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
