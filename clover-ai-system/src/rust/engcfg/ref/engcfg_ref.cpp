// Reference implementation of ../include/engcfg.h: the engine's configuration / summary text (committed 4d8d891, extracted by extract.py into x/*.inc, unchanged) behind extern "C" wrappers.
// It is the original's text behind a wrapper, not an independently verified implementation. Everything marked ADAPTED(n) is listed in NOTES.md of this directory.
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "engcfg.h"
#include "ecref.h"

// ADAPTED(1): TOP (16) and EXPERT_RAW (17547264) are compile-time constants in the engine (moe.cuh); the contract passes them in, so the extracted text sees variables of the same name.
static int ecr_top = 16, ecr_expert_raw = 17547264;
#define TOP ecr_top
#define EXPERT_RAW ecr_expert_raw

// the engine's own declarations of the globals the extracted text uses (backend-gpu.cu / decode.cuh, one line each, types and initial values as the engine has them)
#include "x/decls.inc"
#include "x/decls_dc.inc"

// ADAPTED(2): the engine's `static std::vector<CallSet> g_rc`. The extracted text reads only size(), empty() and g_rc[g_rc_pos]; the contract hands over just that entry.
struct RcView {
    size_t n = 0; CallSet cur;
    size_t size() const { return n; }
    bool empty() const { return n == 0; }
    const CallSet &operator[](size_t) const { return cur; }
};
static RcView g_rc;
// ADAPTED(3): g_pool[c] is a CardPool (CUDA); print_summary reads only .ready, .slots and .st
struct PoolView { bool ready = false; int slots = 0; ExpertStats st; };
static PoolView g_pool[MAX_CARDS];
// ADAPTED(4): g_sc / g_rt are structs of function pointers; print_summary reads only .on
struct OnFlag { bool on = false; };
static OnFlag g_sc, g_rt;
// ADAPTED(5): g_host is a HostCache (pinned memory). configure() reads cap_slots and calls init(bytes); here init records the byte count and (composite test only) derives cap_slots like decode.cuh:408/432
struct HostView {
    size_t cap_slots = 0, last_bytes = 0; bool record_only = true, print = false;
    void init(size_t bytes)
    {
        last_bytes = bytes;
        if (record_only) return;
        cap_slots = bytes / EXPERT_RAW / 32 * 32;
        if (print) fprintf(stderr, "K3GPU: host expert cache of %zu slots (stand-in)\n", cap_slots);
    }
};
static HostView g_host;

// the original functions, whole
#include "x/env_int.inc"
#include "x/card_of.inc"
#include "x/force_routes.inc"
#include "x/print_summary.inc"

// ---- the selectors' variable parsing, original lines. The built variant ends by falling through when the value is exactly "rust"; its `return;` means "cpp". The lambda makes `return;` legal.
static int sel_parse_copyq() { int rust = 0; [&]() {
#include "x/sel_copyq_parse.inc"
    rust = 1; }(); return rust; }
static int sel_parse_slotcache() { int rust = 0; [&]() {
#include "x/sel_slotcache_parse.inc"
    rust = 1; }(); return rust; }
static int sel_parse_routetable() { int rust = 0; [&]() {
#include "x/sel_routetable_parse.inc"
    rust = 1; }(); return rust; }
static void sel_nb_copyq() {
#include "x/sel_copyq_nb.inc"
}
static void sel_nb_slotcache() {
#include "x/sel_slotcache_nb.inc"
}
static void sel_nb_routetable() {
#include "x/sel_routetable_nb.inc"
}
static const char *sel_lib_copyq() {
#include "x/sel_copyq_lib.inc"
    return lib; }
static const char *sel_lib_slotcache() {
#include "x/sel_slotcache_lib.inc"
    return lib; }
static const char *sel_lib_routetable() {
#include "x/sel_routetable_lib.inc"
    return lib; }
static void sel_ann_copyq(const char *lib) {
#include "x/sel_copyq_announce.inc"
}
static void sel_ann_slotcache(const char *lib) {
#include "x/sel_slotcache_announce.inc"
}
static void sel_ann_routetable(const char *lib) {
#include "x/sel_routetable_announce.inc"
}

extern "C" {

int ec_env_int(const char *name, int dflt) { return env_int(name, dflt); }

void ec_settings_from_env(ec_settings *o, int devices)
{
    // ADAPTED(6): a fresh process has the declared initial values; this function runs many times in one process
    g_lookahead = 0; g_vq = false; g_vq_cap = 256; g_vq_from = 0; g_policy = 1; g_chunks = 0; g_layout = 0; g_prefill = false; g_pend_cap = 0;
    g_host.record_only = true; g_host.last_bytes = 0;
    // ADAPTED(7): line 491 `int have = 0; CHECK(cudaGetDeviceCount(&have));` -> the argument
    int have = devices;
#include "x/cfg_cards.inc"
    // ADAPTED(8): slotcache_set_policy(p) stores g_policy (and tells the Rust slot cache, which is not ours)
#define slotcache_set_policy(p) (g_policy = (p))
#include "x/cfg_policy.inc"
#undef slotcache_set_policy
#include "x/cfg_chunks.inc"
#include "x/cfg_host.inc"
    const char *routecache = nullptr;
    // lines 502-503 open a block and the `if (rc && *rc) {` condition; the closing braces and the copy of rc are ours
#include "x/cfg_rc_head.inc"
#include "x/cfg_lookahead.inc"
#include "x/cfg_vq.inc"
    routecache = rc;
    } }
#include "x/cfg_layout.inc"
#include "x/cfg_prefill_flag.inc"
    o->cards = g_cards; o->readers = g_readers; o->verbose = g_verbose; o->reserve = g_reserve;
    o->policy = g_policy; o->chunks = g_chunks;
    o->host_gb = env_int("K3GPU_HOST_GB", 150);   // ADAPTED(9): the engine evaluates this expression twice (lines 501 and 523); the struct stores the value
    o->host_bytes = g_host.last_bytes;
    o->routecache = routecache;
    o->lookahead = g_lookahead; o->vqueue = g_vq ? 1 : 0; o->vq_cap = g_vq_cap; o->vq_from = g_vq_from;
    o->layout = g_layout; o->prefill = g_prefill ? 1 : 0;
}

size_t ec_pend_cap(size_t cap_slots)
{
    g_host.cap_slots = cap_slots;
#include "x/cfg_pendcap.inc"
    return g_pend_cap;
}

FILE *ec_open_trace(void)
{
    g_trace = nullptr;
#include "x/cfg_trace.inc"
    return g_trace;
}

void ec_print_route_cache(const ec_settings *s, size_t calls, size_t pend_cap)
{
    const char *rc = s->routecache;
    g_rc.n = calls; g_lookahead = s->lookahead; g_pend_cap = pend_cap; g_vq = s->vqueue != 0; g_vq_cap = s->vq_cap;
#include "x/cfg_rc_msg.inc"
}

void ec_prefill_start(size_t calls)
{
    g_rc.n = calls;
#include "x/cfg_prefill_check.inc"
#include "x/cfg_prefill_msg.inc"
}

void ec_print_start_line(const ec_settings *s)
{
    g_cards = s->cards; g_readers = s->readers;
    // ADAPTED(10): the engine re-reads K3GPU_HOST_GB here (env_int call inside the fprintf); the contract prints s->host_gb
#define env_int(name, dflt) (s->host_gb)
#include "x/cfg_start_msg.inc"
#undef env_int
}

int ec_selector_mode(int component, int built)
{
    switch (component) {
    case EC_COPYQ: if (built) return sel_parse_copyq() ? EC_MODE_RUST : EC_MODE_CPP; sel_nb_copyq(); return EC_MODE_CPP;
    case EC_SLOTCACHE: if (built) return sel_parse_slotcache() ? EC_MODE_RUST : EC_MODE_CPP; sel_nb_slotcache(); return EC_MODE_CPP;
    case EC_ROUTETABLE: if (built) return sel_parse_routetable() ? EC_MODE_RUST : EC_MODE_CPP; sel_nb_routetable(); return EC_MODE_CPP;
    }
    abort();
}

const char *ec_selector_lib(int component)
{
    switch (component) {
    case EC_COPYQ: return sel_lib_copyq();
    case EC_SLOTCACHE: return sel_lib_slotcache();
    case EC_ROUTETABLE: return sel_lib_routetable();
    }
    abort();
}

void ec_selector_announce_rust(int component, const char *lib)
{
    switch (component) {
    case EC_COPYQ: sel_ann_copyq(lib); return;
    case EC_SLOTCACHE: sel_ann_slotcache(lib); return;
    case EC_ROUTETABLE: sel_ann_routetable(lib); return;
    }
    abort();
}

int ec_card_of(int layout, int cards, int layer)
{
    g_layout = layout; g_cards = cards;
    return card_of(layer);
}

void ec_force_routes(size_t pos, size_t size, const ec_call *cur, int L, int n, int *sel, int top)
{
    ecr_top = top; g_rc_pos = pos; g_rc.n = size;
    if (pos < size && cur) {
        g_rc.cur.layer = cur->layer;
        g_rc.cur.experts.assign(cur->experts, cur->experts + cur->n_experts);
        g_rc.cur.counts.assign(cur->counts, cur->counts + cur->n_counts);
    }
    force_routes(L, n, sel);
}

void ec_print_summary(const ec_summary *s)
{
    g_cards = s->cards; ecr_expert_raw = s->expert_raw;
    for (int c = 0; c < EC_MAX_CARDS; c++) {
        const ec_card_stats &k = s->card[c]; PoolView &p = g_pool[c];
        p.ready = k.ready != 0; p.slots = k.slots;
        p.st.needed = k.needed; p.st.hit_card = k.hit_card; p.st.hit_host = k.hit_host; p.st.from_drive = k.from_drive; p.st.written_back = k.written_back;
        p.st.t_read = k.t_read; p.st.t_copy = k.t_copy;
    }
    g_copyq_rust = s->copyq_rust != 0; for (int i = 0; i < 3; i++) g_cq_jobs[i].store(s->cq_jobs[i]);
    g_sc.on = s->slotcache_rust != 0; for (int i = 0; i < 3; i++) g_sc_calls[i] = s->sc_calls[i];
    g_rt.on = s->routetable_rust != 0; g_rc.n = s->rc_calls; g_rt_nu_calls = s->rt_nu_calls;
    g_rc_pos = s->rc_pos; g_rc_ok = s->rc_ok != 0;
    g_pf_planned = s->pf_planned; g_pf_skipped = s->pf_skipped; g_pf_used = s->pf_used; g_pf_waited = s->pf_waited; g_t_pwait = s->t_pwait; g_wb_skipped = s->wb_skipped;
    g_vq = s->vq != 0; g_vq_ram = s->vq_ram; g_vq_drive = s->vq_drive; g_vq_wb = s->vq_wb; g_vq_nowb = s->vq_nowb; g_vq_waited = s->vq_waited; g_t_vwait = s->t_vwait; g_t_plan = s->t_plan;
    for (int i = 0; i < 3; i++) { g_n_calls[i] = s->n_calls[i]; g_t_calls[i] = s->t_calls[i]; }
    print_summary();
}

}  // extern "C"

// ---------------------------------------------------------------------------------------------------------------------------------------------------------------
// Composite reference (not part of the contract; exported with the ecref_ prefix): the engine's configure() TEXT, whole, with its CUDA / dlopen / loader steps replaced by stand-ins.
// The differential driver compares its stderr, exit status and resulting globals with a C++ sequence of contract calls in the documented order (engcfg.h).
static int g_built[3];                 // ADAPTED(11): K3GPU_COPYQ_BUILT / SLOTCACHE_BUILT / ROUTETABLE_BUILT of the engine build, set per run
static int g_devices = 0;
static bool g_nu_set = false;
static void (*g_route_hook)(int, const std::vector<int> &) = nullptr;
static void (*g_force_routes)(int, int, int *) = nullptr;
#define CHECK(x) (x)
static int cudaGetDeviceCount(int *p) { *p = g_devices; return 0; }
static void slotcache_set_policy(int p) { g_policy = p; }
static void slotcache_set_next_use(int (*)(int)) { g_nu_set = true; }
static int rc_nu_sel(int) { return 0; }
static void route_hook(int, const std::vector<int> &) {}
// ADAPTED(12): rc_load_sel reads a file and parses it; here the table has strlen(path) % 3 calls (shared with the driver) so that empty and non-empty tables both occur
static void rc_load_sel(const char *path) { g_rc.n = strlen(path) % 3; }
// ADAPTED(13): the selector functions without dlopen: the original parsing, then (rust) the announce line as if the library had loaded
static void stand_select(int comp)
{
    if (ec_selector_mode(comp, g_built[comp]) == EC_MODE_RUST) ec_selector_announce_rust(comp, ec_selector_lib(comp));
}
static void copyq_select() { stand_select(EC_COPYQ); }
static void slotcache_select() { stand_select(EC_SLOTCACHE); }
static void routetable_select() { stand_select(EC_ROUTETABLE); }

#include "x/configure.inc"

extern "C" {

void ecref_configure(int devices, const int built[3])
{
    // fresh-process state (the declared initial values)
    g_cards = 0; g_readers = 16; g_verbose = 0; g_reserve = 0; g_configured = false; g_rc_pos = 0; g_pend_cap = 0; g_lookahead = 0; g_rc_ok = false; g_layout = 0;
    g_vq_cap = 256; g_vq_from = 0; g_policy = 1; g_chunks = 0; g_vq = false; g_trace = nullptr; g_prefill = false;
    g_rc.n = 0; g_nu_set = false; g_route_hook = nullptr; g_force_routes = nullptr;
    g_host.cap_slots = 0; g_host.last_bytes = 0; g_host.record_only = false; g_host.print = true;
    ecr_expert_raw = 17547264; ecr_top = 16;
    for (int c = 0; c < MAX_CARDS; c++) g_pool[c] = PoolView();
    g_copyq_rust = false; g_sc.on = false; g_rt.on = false; g_rt_nu_calls = 0;
    g_devices = devices; for (int i = 0; i < 3; i++) g_built[i] = built[i];
    configure();
}

void ecref_snapshot(ecref_snap *o)
{
    o->configured = g_configured; o->cards = g_cards; o->readers = g_readers; o->verbose = g_verbose; o->reserve = g_reserve; o->policy = g_policy; o->chunks = g_chunks;
    o->trace_open = g_trace != nullptr; o->host_bytes = g_host.last_bytes; o->host_slots = g_host.cap_slots; o->rc_calls = g_rc.n; o->lookahead = g_lookahead;
    o->pend_cap = g_pend_cap; o->rc_ok = g_rc_ok; o->nu_set = g_nu_set; o->vq = g_vq; o->vq_cap = g_vq_cap; o->vq_from = g_vq_from; o->hook_set = g_route_hook != nullptr;
    o->layout = g_layout; o->prefill = g_prefill; o->force_set = g_force_routes != nullptr;
}

}  // extern "C"
