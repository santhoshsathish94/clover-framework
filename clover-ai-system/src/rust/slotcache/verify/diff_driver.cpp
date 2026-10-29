// Differential driver for include/slotcache.h: loads two implementations (dlopen, separate symbol tables) and compares them operation by operation.
//   diff_driver LIB_REFERENCE LIB_CANDIDATE [--only S1,S2,..] [--jobs N] [--scale F] [--opsmul F] [--out DIR] [--trace FILE]
// Scenarios S1..S9 and Q5 as specified by the task; every random scenario is a list of (scenario, seed) units run by worker processes.
// Build: g++ -O2 -std=c++17 -I../include -o diff_driver diff_driver.cpp -ldl
#include "slotcache.h"
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
#include <unistd.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <ctime>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

typedef sc_cache C;

// ------------------------------------------------------------------------------------------------ library tables
struct Api {
    void *h = nullptr; std::string path;
    void (*set_policy)(int) = nullptr; void (*set_next_use)(sc_next_use_fn) = nullptr;
    C *(*nw)() = nullptr; void (*fr)(C *) = nullptr;
    void (*add_spare)(C *, unsigned char *) = nullptr; void (*release)(C *, unsigned char *) = nullptr;
    int (*contains)(const C *, int) = nullptr; size_t (*size)(const C *) = nullptr;
    unsigned char *(*find)(C *, int) = nullptr; void (*touch)(C *, int, float) = nullptr; int (*worth)(const C *, int) = nullptr;
    unsigned char *(*acquire)(C *, int, const int *, size_t, float, int *, float *) = nullptr;
    unsigned char *(*try_acquire)(C *, int, int, const int *, size_t, float, int *, float *) = nullptr;
    unsigned char *(*remove)(C *, int) = nullptr;
    size_t (*snap_lru)(const C *, sc_entry *, size_t) = nullptr;
    size_t (*snap_spare)(const C *, unsigned char **, size_t) = nullptr;
    size_t (*snap_next)(const C *, int *, int *, size_t) = nullptr;
};

static bool load_api(Api &a, const char *path)
{
    a.path = path;
    a.h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!a.h) { fprintf(stderr, "dlopen %s: %s\n", path, dlerror()); return false; }
    bool ok = true;
#define SYM(field, name) do { void *s_ = dlsym(a.h, name); if (!s_) { fprintf(stderr, "missing symbol %s in %s\n", name, path); ok = false; } else memcpy((void *)&a.field, &s_, sizeof s_); } while (0)
    SYM(set_policy, "sc_set_policy"); SYM(set_next_use, "sc_set_next_use"); SYM(nw, "sc_new"); SYM(fr, "sc_free");
    SYM(add_spare, "sc_add_spare"); SYM(release, "sc_release"); SYM(contains, "sc_contains"); SYM(size, "sc_size");
    SYM(find, "sc_find"); SYM(touch, "sc_touch"); SYM(worth, "sc_worth"); SYM(acquire, "sc_acquire"); SYM(try_acquire, "sc_try_acquire");
    SYM(remove, "sc_remove"); SYM(snap_lru, "sc_snapshot_lru"); SYM(snap_spare, "sc_snapshot_spare"); SYM(snap_next, "sc_snapshot_next");
#undef SYM
    return ok;
}

// ------------------------------------------------------------------------------------------------ small utilities
static std::string sfmt(const char *f, ...)
{
    va_list ap; va_start(ap, f); char buf[2048]; vsnprintf(buf, sizeof buf, f, ap); va_end(ap); return buf;
}
static uint32_t fbits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float bitsf(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static const uintptr_t SLOT_BASE = 0x100000;
static int slotno(const unsigned char *p) { return p ? (int)(((uintptr_t)p - SLOT_BASE) / 64) : -1; }
static uint64_t mix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ull; x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull; x = (x ^ (x >> 27)) * 0x94D049BB133111EBull; return x ^ (x >> 31);
}
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed = 1) : s(seed) {}
    uint64_t next() { return mix64(s += 0x9E3779B97F4A7C15ull); }
    uint32_t u(uint32_t n) { return n ? (uint32_t)(next() % n) : 0; }
    double f() { return (double)(next() >> 11) * (1.0 / 9007199254740992.0); }
    bool p(double pr) { return f() < pr; }
};

// ------------------------------------------------------------------------------------------------ next-use callbacks (one per library, own state, own call log)
struct NuState { std::vector<int> log; std::vector<uint32_t> ptr, cnt; uint64_t ncalls = 0; };
static NuState g_nus[2];
struct Model {
    int mode = 0;                          // 0 = the engine's rc_nu (per-key list of call numbers, pointer advancing with the current call), 1 = hash with ties and extremes
    std::vector<std::vector<int>> uses;
    long cur = 0;                          // the global current-call counter
    uint64_t hseed = 0; int hrange = 8;
};
static Model g_model;
static int nu_eval(int id, int key)
{
    NuState &s = g_nus[id]; s.log.push_back(key); s.ncalls++;
    if (g_model.mode == 0) {
        if (key < 0 || (size_t)key >= g_model.uses.size() || g_model.uses[key].empty()) return 1000000000;
        const std::vector<int> &l = g_model.uses[key]; uint32_t &p = s.ptr[key];
        while (p < l.size() && l[p] <= g_model.cur) p++;
        return p < l.size() ? l[p] : 1000000000;
    }
    if (key < 0 || (size_t)key >= s.cnt.size()) return 7;
    const uint32_t n = s.cnt[key]++;   // the per-key call counter is part of the value: a different call order gives different numbers
    const uint64_t x = mix64(g_model.hseed ^ ((uint64_t)(uint32_t)key << 17) ^ ((uint64_t)g_model.cur << 41) ^ ((uint64_t)n * 0x9E3779B1ull));
    const uint32_t r = (uint32_t)(x % (uint64_t)(g_model.hrange + 3));
    if ((int)r < g_model.hrange) return (int)r;
    if ((int)r == g_model.hrange) return 1000000000;
    if ((int)r == g_model.hrange + 1) return INT_MAX;
    return -(int)((x >> 40) % 5);
}
template <int ID> static int nu_cb(int key) { return nu_eval(ID, key); }
static sc_next_use_fn g_cbs[2] = { nu_cb<0>, nu_cb<1> };
static void model_reset(size_t nkeys)
{
    for (int i = 0; i < 2; i++) { g_nus[i].log.clear(); g_nus[i].ncalls = 0; g_nus[i].ptr.assign(nkeys, 0); g_nus[i].cnt.assign(nkeys, 0); }
    g_model.cur = 0;
}

// ------------------------------------------------------------------------------------------------ operations and the dual world
enum K { K_FIND, K_TOUCH, K_WORTH, K_ACQ, K_TRY, K_REMOVE, K_RELEASE, K_ADD, K_CONTAINS, K_POLICY, K_SETNU };
struct Op {
    K k = K_FIND; int cache = -1; int key = 0, cand = 0; float f = 0; unsigned char *slot = nullptr; std::vector<int> prot; int ival = 0; bool nullprot = false;
};
struct Res { uint64_t slot = 0, ival = 0, sz = 0, vkey = 0, vheat = 0; };

static std::string describe(const Op &o)
{
    std::string p;
    if (o.k == K_ACQ || o.k == K_TRY) { p = sfmt(" prot[%zu]={", o.prot.size()); for (size_t i = 0; i < o.prot.size() && i < 40; i++) p += sfmt("%s%d", i ? "," : "", o.prot[i]); p += o.prot.size() > 40 ? ",..}" : "}"; if (o.nullprot) p += " (NULL when empty)"; }
    switch (o.k) {
        case K_FIND: return sfmt("sc_find(cache %d, key %d)", o.cache, o.key);
        case K_TOUCH: return sfmt("sc_touch(cache %d, key %d, w=%.9g [%08x])", o.cache, o.key, o.f, fbits(o.f));
        case K_WORTH: return sfmt("sc_worth(cache %d, key %d)", o.cache, o.key);
        case K_ACQ: return sfmt("sc_acquire(cache %d, key %d, heat0=%.9g [%08x])%s", o.cache, o.key, o.f, fbits(o.f), p.c_str());
        case K_TRY: return sfmt("sc_try_acquire(cache %d, key %d, cand_nu=%d, heat0=%.9g [%08x])%s", o.cache, o.key, o.cand, o.f, fbits(o.f), p.c_str());
        case K_REMOVE: return sfmt("sc_remove(cache %d, key %d)", o.cache, o.key);
        case K_RELEASE: return sfmt("sc_release(cache %d, slot #%d)", o.cache, slotno(o.slot));
        case K_ADD: return sfmt("sc_add_spare(cache %d, slot #%d)", o.cache, slotno(o.slot));
        case K_CONTAINS: return sfmt("sc_contains(cache %d, key %d)", o.cache, o.key);
        case K_POLICY: return sfmt("sc_set_policy(%d)", o.ival);
        case K_SETNU: return sfmt("sc_set_next_use(%s)", o.ival ? "callback" : "NULL");
    }
    return "?";
}

struct Mismatch { std::string text; };

struct Dual {
    int idx = 0; C *c[2] = { nullptr, nullptr };
    size_t total = 0;                                   // slots ever handed to this cache
    std::vector<sc_entry> lru; std::vector<unsigned char *> spare; std::vector<int> nnu, nkey;   // library 0's state after the last comparison
    std::vector<unsigned char *> out;                   // slots returned by sc_remove and not yet given back
    size_t fresh_left = 0;
};

enum { NCNT = 16 };
struct World {
    Api *a;                                             // two tables
    std::vector<Dual> d;
    Rng rng;
    std::string scen; uint64_t seed = 0; uint64_t opidx = 0;
    int policy = -1; bool cb = false;                   // the driver's own knowledge of the two process-wide settings
    bool full_ro = true;                                // state snapshot also after contains / worth
    bool all_each_op = false;                           // compare every cache after every operation
    uintptr_t next_slot = SLOT_BASE;
    uint64_t cnt[NCNT] = {};
    Op lastop;
    std::vector<sc_entry> eb[2]; std::vector<unsigned char *> sb[2]; std::vector<int> nb[2], kb[2];

    unsigned char *fresh() { next_slot += 64; return (unsigned char *)next_slot; }

    Dual &add_cache()
    {
        d.emplace_back(); Dual &x = d.back(); x.idx = (int)d.size() - 1;
        x.c[0] = a[0].nw(); x.c[1] = a[1].nw();
        return x;
    }
    void free_all() { for (Dual &x : d) { a[0].fr(x.c[0]); a[1].fr(x.c[1]); } d.clear(); }

    std::string head() const { return sfmt("scenario %s seed %llu op #%llu", scen.c_str(), (unsigned long long)seed, (unsigned long long)opidx); }

    [[noreturn]] void fail(const std::string &why, const std::string &extra = "")
    {
        std::string t = "MISMATCH: " + head() + "\n  operation: " + describe(lastop) + "\n  " + why + "\n" + extra;
        throw Mismatch{ t };
    }

    Res run(int i, Dual *dd, const Op &op)
    {
        static int dummy[1] = { 0 };
        Api &A = a[i]; Res r; C *c = dd ? dd->c[i] : nullptr;
        int vk = 0x12345678; float vh = bitsf(0xDEADBEEFu);
        const int *pp = op.prot.empty() ? (op.nullprot ? nullptr : dummy) : op.prot.data();
        switch (op.k) {
            case K_FIND: r.slot = (uint64_t)(uintptr_t)A.find(c, op.key); break;
            case K_TOUCH: A.touch(c, op.key, op.f); break;
            case K_WORTH: r.ival = (uint64_t)(uint32_t)A.worth(c, op.key); break;
            case K_ACQ: r.slot = (uint64_t)(uintptr_t)A.acquire(c, op.key, pp, op.prot.size(), op.f, &vk, &vh); r.vkey = (uint32_t)vk; r.vheat = fbits(vh); break;
            case K_TRY: r.slot = (uint64_t)(uintptr_t)A.try_acquire(c, op.key, op.cand, pp, op.prot.size(), op.f, &vk, &vh); r.vkey = (uint32_t)vk; r.vheat = fbits(vh); break;
            case K_REMOVE: r.slot = (uint64_t)(uintptr_t)A.remove(c, op.key); break;
            case K_RELEASE: A.release(c, op.slot); break;
            case K_ADD: A.add_spare(c, op.slot); break;
            case K_CONTAINS: r.ival = (uint64_t)(uint32_t)A.contains(c, op.key); break;
            case K_POLICY: A.set_policy(op.ival); break;
            case K_SETNU: A.set_next_use(op.ival ? g_cbs[i] : nullptr); break;
        }
        if (dd) r.sz = A.size(c);
        return r;
    }

    static std::string fmt_state_entry(const sc_entry &e) { return sfmt("[key %d slot #%d heat %.9g/%08x nu %d]", e.key, slotno(e.slot), e.heat, fbits(e.heat), e.nu); }

    void cmp_state(Dual &dd)
    {
        const size_t cap = dd.total + 4;
        size_t nl[2], ns[2], nn[2];
        for (int i = 0; i < 2; i++) {
            eb[i].resize(cap); sb[i].resize(cap); nb[i].resize(cap); kb[i].resize(cap);
            nl[i] = a[i].snap_lru(dd.c[i], eb[i].data(), cap);
            ns[i] = a[i].snap_spare(dd.c[i], sb[i].data(), cap);
            nn[i] = a[i].snap_next(dd.c[i], nb[i].data(), kb[i].data(), cap);
        }
        std::string why; long first = -1; const char *what = nullptr;
        if (nl[0] != nl[1]) { what = "lru"; why = sfmt("lru count %zu vs %zu", nl[0], nl[1]); }
        else if (ns[0] != ns[1]) { what = "spare"; why = sfmt("spare count %zu vs %zu", ns[0], ns[1]); }
        else if (nn[0] != nn[1]) { what = "next"; why = sfmt("ordered-set count %zu vs %zu", nn[0], nn[1]); }
        else if (nl[0] > cap || ns[0] > cap || nn[0] > cap) { what = "count"; why = sfmt("a snapshot count exceeds the %zu slots given to this cache (lru %zu spare %zu next %zu)", dd.total, nl[0], ns[0], nn[0]); }
        else if (nl[0] != a[0].size(dd.c[0])) { what = "size"; why = sfmt("lru snapshot count %zu != sc_size %zu (library 0)", nl[0], a[0].size(dd.c[0])); }
        else {
            for (size_t i = 0; i < nl[0] && !what; i++) {
                const sc_entry &x = eb[0][i], &y = eb[1][i];
                if (x.key != y.key || x.slot != y.slot || fbits(x.heat) != fbits(y.heat) || x.nu != y.nu) { what = "lru"; first = (long)i; why = sfmt("lru entry %zu differs", i); }
            }
            for (size_t i = 0; i < ns[0] && !what; i++) if (sb[0][i] != sb[1][i]) { what = "spare"; first = (long)i; why = sfmt("spare entry %zu differs", i); }
            for (size_t i = 0; i < nn[0] && !what; i++) if (nb[0][i] != nb[1][i] || kb[0][i] != kb[1][i]) { what = "next"; first = (long)i; why = sfmt("ordered-set entry %zu differs", i); }
        }
        if (what) {
            std::string ex = sfmt("  state of cache %d after the operation (library 0 = %s, library 1 = %s): %s\n", dd.idx, a[0].path.c_str(), a[1].path.c_str(), why.c_str());
            for (int i = 0; i < 2; i++) {
                const size_t L = std::min(nl[i], cap), S = std::min(ns[i], cap), N = std::min(nn[i], cap);
                ex += sfmt("  lib %d: lru n=%zu spare n=%zu next n=%zu\n", i, nl[i], ns[i], nn[i]);
                size_t lo = 0, hi = std::min<size_t>(L, 6);
                if (first >= 0 && !strcmp(what, "lru")) { lo = (size_t)std::max<long>(0, first - 3); hi = std::min<size_t>(L, (size_t)first + 4); }
                ex += sfmt("    lru[%zu..%zu) (mru first):", lo, hi);
                for (size_t j = lo; j < hi; j++) ex += " " + fmt_state_entry(eb[i][j]);
                ex += "\n    spare (last = handed out next), last 6:";
                for (size_t j = S > 6 ? S - 6 : 0; j < S; j++) ex += sfmt(" #%d", slotno(sb[i][j]));
                ex += "\n    next (smallest first), first 6:";
                for (size_t j = 0; j < std::min<size_t>(N, 6); j++) ex += sfmt(" (%d,%d)", nb[i][j], kb[i][j]);
                if (first >= 0 && !strcmp(what, "next")) { ex += sfmt("\n    next around entry %ld:", first); for (size_t j = (size_t)std::max<long>(0, first - 2); j < std::min<size_t>(N, (size_t)first + 3); j++) ex += sfmt(" (%d,%d)", nb[i][j], kb[i][j]); }
                ex += "\n";
            }
            fail(std::string("state differs: ") + what, ex);
        }
        dd.lru.assign(eb[0].begin(), eb[0].begin() + nl[0]);
        dd.spare.assign(sb[0].begin(), sb[0].begin() + ns[0]);
        dd.nnu.assign(nb[0].begin(), nb[0].begin() + nn[0]);
        dd.nkey.assign(kb[0].begin(), kb[0].begin() + nn[0]);
    }
    void cmp_all() { for (Dual &x : d) cmp_state(x); }

    Res step(const Op &op)
    {
        opidx++; lastop = op;
        Dual *dd = op.cache >= 0 ? &d[op.cache] : nullptr;
        const int pol0 = policy; const bool cb0 = cb;
        Res r[2];
        for (int i = 0; i < 2; i++) r[i] = run(i, dd, op);
        if (op.k == K_POLICY) policy = op.ival; else if (op.k == K_SETNU) cb = op.ival != 0;
        if (r[0].slot != r[1].slot || r[0].ival != r[1].ival || r[0].sz != r[1].sz || r[0].vkey != r[1].vkey || r[0].vheat != r[1].vheat)
            fail("return values / out-parameters differ",
                 sfmt("  lib 0: slot #%d ival %llu size %llu vkey %d vheat %08x\n  lib 1: slot #%d ival %llu size %llu vkey %d vheat %08x\n",
                      slotno((unsigned char *)(uintptr_t)r[0].slot), (unsigned long long)r[0].ival, (unsigned long long)r[0].sz, (int)(uint32_t)r[0].vkey, (unsigned)r[0].vheat,
                      slotno((unsigned char *)(uintptr_t)r[1].slot), (unsigned long long)r[1].ival, (unsigned long long)r[1].sz, (int)(uint32_t)r[1].vkey, (unsigned)r[1].vheat));
        if (g_nus[0].log != g_nus[1].log) {
            std::string ex = "  callback call logs differ (requested keys, in order)\n";
            for (int i = 0; i < 2; i++) { ex += sfmt("  lib %d (%zu calls):", i, g_nus[i].log.size()); for (size_t j = 0; j < std::min<size_t>(g_nus[i].log.size(), 24); j++) ex += sfmt(" %d", g_nus[i].log[j]); ex += "\n"; }
            fail("next-use call logs differ", ex);
        }
        cnt[13] += g_nus[0].log.size(); g_nus[0].log.clear(); g_nus[1].log.clear();
        // coverage counters (library 0's result; the two were just checked equal)
        const int vk = (int)(uint32_t)r[0].vkey; const bool bel = pol0 == 2 && cb0;
        switch (op.k) {
            case K_ACQ: cnt[0]++; if (vk < 0) cnt[1]++; else { cnt[2]++; if (bel) cnt[3]++; else if (pol0 == 0) cnt[5]++; else cnt[4]++; } break;
            case K_TRY: cnt[6]++; if (!r[0].slot) cnt[7]++; else if (vk >= 0) cnt[2]++; break;
            case K_FIND: if (r[0].slot) cnt[8]++; break;
            case K_REMOVE: if (r[0].slot) cnt[10]++; break;
            case K_WORTH: if (r[0].ival) cnt[11]++; else cnt[12]++; break;
            default: break;
        }
        if (op.k == K_TOUCH && dd && a[0].contains(dd->c[0], op.key)) cnt[9]++;
        if (dd) {
            const bool ro = op.k == K_CONTAINS || op.k == K_WORTH;
            if (!ro || full_ro) cmp_state(*dd);
            if (all_each_op) for (Dual &x : d) if (&x != dd) cmp_state(x);
        } else cmp_all();
        return r[0];
    }

    // ---- preconditions of the contract, taken from library 0's state after the previous operation
    static bool in_prot(const std::vector<int> &p, int k) { for (int x : p) if (x == k) return true; return false; }
    bool acq_ok(const Dual &dd, const std::vector<int> &p) const   // true if sc_acquire cannot reach the fatal path
    {
        if (!dd.spare.empty()) return true;
        if (policy == 2 && cb) { for (size_t i = dd.nkey.size(); i-- > 0;) if (!in_prot(p, dd.nkey[i])) return true; return false; }
        for (const sc_entry &e : dd.lru) if (!in_prot(p, e.key)) return true;
        return false;
    }
    bool try_ok(const Dual &dd, const std::vector<int> &p, int cand) const   // true if sc_try_acquire cannot reach the fatal path
    {
        if (!dd.spare.empty()) return true;
        int victim = -1;
        for (size_t i = dd.nkey.size(); i-- > 0;) if (!in_prot(p, dd.nkey[i])) { if (dd.nnu[i] > cand) victim = dd.nkey[i]; break; }
        if (victim < 0) return true;                    // returns NULL
        if (policy == 2 && cb) return true;
        for (const sc_entry &e : dd.lru) if (!in_prot(p, e.key)) return true;
        return false;
    }
};

static Op mkop(K k, int cache, int key = 0) { Op o; o.k = k; o.cache = cache; o.key = key; return o; }

// ------------------------------------------------------------------------------------------------ random scenarios S1..S7
struct Cfg {
    const char *name; const char *what;
    int pol;        // 0,1,2 fixed; 3 switched among 0/1/2 during the run; 4 fixed, random per seed
    int cbm;        // 0 none; 1 set from the start; 2 set/unset at random times; 3 set for a random half of the seeds (policy decides if it is used); 4 per seed: set if policy 2, else half
    int style;      // callback kind: 0 engine lists, 1 hash with ties and extremes, 2 random per seed
    int nc;         // caches
    int sizeset;    // 0 normal, 1 small (3-8 slots), 2 large (2000-20000 slots)
    int frac;       // 0 integral heat mostly, 1 fractional heat, 2 both
    bool full_ro; bool all_each;
    int nseeds; long nops;
};

static float gen_heat(Rng &R, int frac)
{
    if (R.u(50) == 0) { static const float sp[] = { 0.0f, -0.0f, 1e-40f, 1e30f, -3.5f, INFINITY, -INFINITY }; return sp[R.u(7)]; }
    int mode = frac == 2 ? (int)R.u(2) : frac;
    if (mode == 0) { const uint32_t r = R.u(10); if (r < 7) return (float)R.u(41); if (r < 9) return (float)R.u(41) * 0.5f; return (float)R.u(1000) / 8.0f; }
    const uint32_t r = R.u(10);
    if (r < 3) return (float)R.u(41) * 0.1f;
    if (r < 5) return (float)R.u(1000) / 7.0f;
    if (r < 7) return (float)R.u(41) / 3.0f;
    if (r < 8) return (float)R.u(41);
    return (float)R.f() * 100.0f;
}

static int g_prot_mode = 0;   // 1 in S11: prot sets of 33..3000 keys, a tenth of them with keys outside [0, 2^20) or negative

static void gen_prot(World &w, const Dual &dd, int Kc, std::vector<int> &p, Rng &R)
{
    p.clear();
    if (g_prot_mode) {
        const uint32_t r = R.u(100);
        const int n = r < 10 ? 1 + (int)R.u(32) : 33 + (int)R.u((uint32_t)std::min(Kc + 40, 3000));
        for (int i = 0; i < n; i++) {
            const uint32_t t = R.u(10);
            if (t < 5 && !dd.lru.empty()) p.push_back(dd.lru[R.u((uint32_t)dd.lru.size())].key);
            else if (t < 9) p.push_back((int)R.u(Kc));
            else if (!p.empty()) p.push_back(p[R.u((uint32_t)p.size())]);
            else p.push_back((int)R.u(Kc));
        }
        if (R.u(10) == 0) { static const int ODD[] = { 1 << 20, (1 << 20) - 1, -3, 2147483647, (int)0x80000000, 1 << 25 }; p[R.u((uint32_t)p.size())] = ODD[R.u(6)]; }
        return;
    }
    const uint32_t r = R.u(100);
    const int n = r < 15 ? 0 : (r < 35 ? 1 + (int)R.u(4) : 1 + (int)R.u(32));
    (void)w;
    for (int i = 0; i < n; i++) {
        const uint32_t t = R.u(10);
        if (t < 5 && !dd.lru.empty()) {
            size_t idx = (R.u(10) < 4) ? dd.lru.size() - 1 - R.u((uint32_t)std::min<size_t>(dd.lru.size(), 60)) : R.u((uint32_t)dd.lru.size());
            p.push_back(dd.lru[idx].key);
        } else if (t < 8) p.push_back((int)R.u(Kc));
        else if (!p.empty()) p.push_back(p[R.u((uint32_t)p.size())]);
        else p.push_back((int)R.u(Kc));
    }
}

static void rand_run(World &w, const Cfg &cf, uint64_t seed, long nops)
{
    Rng &R = w.rng; R = Rng(mix64(seed * 1000003ull + (uint64_t)(uintptr_t)cf.name[1] * 7919ull + (uint64_t)cf.name[0]));
    g_prot_mode = !strcmp(cf.name, "S11") ? 1 : 0;
    w.full_ro = cf.full_ro; w.all_each_op = cf.all_each;
    const int nc = cf.nc;
    std::vector<int> nslots(nc), Kc(nc); int Kmax = 4;
    static const int NS[] = { 3, 5, 8, 16, 48, 64, 200, 1000 };
    static const int NL[] = { 2000, 5000, 10000, 20000 };
    static const double KR[] = { 0.5, 0.8, 1.0, 1.5, 3.0, 10.0 };
    for (int i = 0; i < nc; i++) {
        nslots[i] = cf.sizeset == 1 ? 3 + (int)R.u(6) : cf.sizeset == 2 ? NL[R.u(4)] : nc > 1 ? 3 + (int)R.u(60) : NS[R.u(8)];
        const double kr = cf.sizeset == 2 ? KR[R.u(5)] : KR[R.u(6)];
        Kc[i] = std::max(4, (int)(nslots[i] * kr)); Kmax = std::max(Kmax, Kc[i]);
    }
    const long H = nops / 6 + 30;
    g_model.mode = cf.style == 2 ? (int)R.u(2) : cf.style;
    g_model.uses.assign(Kmax, std::vector<int>());
    for (int k = 0; k < Kmax; k++) { const int n = (int)R.u(6); for (int j = 0; j < n; j++) g_model.uses[k].push_back((int)R.u((uint32_t)H + 1)); std::sort(g_model.uses[k].begin(), g_model.uses[k].end()); }
    g_model.hseed = R.next(); g_model.hrange = 2 + (int)R.u(10);
    model_reset(Kmax);

    static const int EXOTIC_POLICY[] = { 0, 1, 2, 3, 7, -1, 2147483647, (int)0x80000000 };   // S10: K3GPU_POLICY is read with atoi, any int can arrive
    const int pol_fixed = cf.pol <= 2 ? cf.pol : cf.pol == 5 ? EXOTIC_POLICY[R.u(8)] : (int)R.u(3);
    { Op o = mkop(K_POLICY, -1); o.ival = pol_fixed; w.step(o); }
    bool cb_on = cf.cbm == 1 || (cf.cbm == 2 && R.u(2)) || (cf.cbm == 3 && R.u(2)) || (cf.cbm == 4 && (pol_fixed == 2 || R.u(2)));
    { Op o = mkop(K_SETNU, -1); o.ival = cb_on ? 1 : 0; w.step(o); }

    for (int i = 0; i < nc; i++) {
        Dual &dd = w.add_cache();
        const double frac0 = R.u(3) == 0 ? 0.3 + R.f() * 0.5 : 1.0;
        size_t n0 = (size_t)std::ceil(nslots[i] * frac0);
        dd.fresh_left = nslots[i] - n0;
        for (size_t j = 0; j < n0; j++) { Op o = mkop(R.u(2) ? K_ADD : K_RELEASE, i); o.slot = w.fresh(); dd.total++; w.step(o); }
    }
    long cur = 0; long fill = cf.sizeset == 2 ? std::min<long>(nslots[0], nops / 2) : 0;
    std::vector<int> prot;
    for (long n = 0; n < nops; n++) {
        if (R.u(6) == 0) { cur += 1 + (R.u(4) == 0 ? (long)R.u(5) : 0); g_model.cur = cur; }
        if ((cf.pol == 3 || cf.pol == 5) && R.u(120) == 0) { Op o = mkop(K_POLICY, -1); o.ival = cf.pol == 5 ? EXOTIC_POLICY[R.u(8)] : (int)R.u(3); w.step(o); }
        if ((cf.cbm == 2) && R.u(80) == 0) { cb_on = !w.cb; Op o = mkop(K_SETNU, -1); o.ival = cb_on ? 1 : 0; w.step(o); }
        const int ci = (int)R.u(nc); Dual &dd = w.d[ci]; const int Kk = Kc[ci];
        auto held_key = [&]() -> int { return dd.lru[(R.u(5) == 0 && dd.lru.size() > 60) ? dd.lru.size() - 1 - R.u(60) : R.u((uint32_t)dd.lru.size())].key; };
        auto pick_key = [&]() -> int { return (R.u(10) < 6 && !dd.lru.empty()) ? held_key() : (int)R.u(Kk); };
        auto absent_key = [&]() -> int { for (int t = 0; t < 12; t++) { const int k = (int)R.u(Kk); if (!w.a[0].contains(dd.c[0], k)) return k; } return -1; };
        uint32_t r = R.u(100);
        if (n < fill) r = 40;                            // fill phase of the large caches
        if (r < 17) { w.step(mkop(K_FIND, ci, pick_key())); }
        else if (r < 34) { Op o = mkop(K_TOUCH, ci, pick_key()); o.f = gen_heat(R, cf.frac); w.step(o); }
        else if (r < 40) { w.step(mkop(K_WORTH, ci, pick_key())); }
        else if (r < 62) {
            const int key = absent_key(); bool done = false;
            if (key >= 0) {
                gen_prot(w, dd, Kk, prot, R);
                if (n < fill) prot.clear();
                while (!w.acq_ok(dd, prot) && !prot.empty()) prot.resize(prot.size() / 2);
                if (w.acq_ok(dd, prot)) { Op o = mkop(K_ACQ, ci, key); o.f = gen_heat(R, cf.frac); o.prot = prot; o.nullprot = R.u(2); w.step(o); done = true; }
            }
            if (!done) w.step(mkop(K_FIND, ci, pick_key()));
        } else if (r < 70) {
            const int key = absent_key(); bool done = false;
            if (key >= 0) {
                int cand; const uint32_t t = R.u(10);
                if (t < 4) cand = (int)R.u((uint32_t)H + 4);
                else if (t < 6 && !dd.nnu.empty()) cand = dd.nnu[R.u((uint32_t)dd.nnu.size())] + (int)R.u(3) - 1;
                else if (t < 7) cand = 1000000000; else if (t < 8) cand = INT_MAX; else if (t < 9) cand = -1; else cand = (int)cur + (int)R.u(5) - 2;
                gen_prot(w, dd, Kk, prot, R);
                while (!w.try_ok(dd, prot, cand) && !prot.empty()) prot.resize(prot.size() / 2);
                if (w.try_ok(dd, prot, cand)) { Op o = mkop(K_TRY, ci, key); o.cand = cand; o.f = gen_heat(R, cf.frac); o.prot = prot; o.nullprot = R.u(2); w.step(o); done = true; }
            }
            if (!done) w.step(mkop(K_FIND, ci, pick_key()));
        } else if (r < 76) { const Res rr = w.step(mkop(K_REMOVE, ci, pick_key())); if (rr.slot) dd.out.push_back((unsigned char *)(uintptr_t)rr.slot); }
        else if (r < 81) {
            if (!dd.out.empty()) { const size_t j = R.u((uint32_t)dd.out.size()); Op o = mkop(R.u(2) ? K_ADD : K_RELEASE, ci); o.slot = dd.out[j]; dd.out[j] = dd.out.back(); dd.out.pop_back(); w.step(o); }
            else w.step(mkop(K_CONTAINS, ci, pick_key()));
        } else if (r < 84) {
            if (dd.fresh_left > 0) { dd.fresh_left--; dd.total++; Op o = mkop(R.u(2) ? K_ADD : K_RELEASE, ci); o.slot = w.fresh(); w.step(o); }
            else w.step(mkop(K_CONTAINS, ci, pick_key()));
        } else if (r < 92) { w.step(mkop(K_CONTAINS, ci, pick_key())); }
        else { w.step(mkop(K_FIND, ci, pick_key())); }
    }
}

#include "dd_engine.inc"
#include "dd_cases.inc"
