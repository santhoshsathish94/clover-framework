// Differential driver for include/routetable.h: loads two implementations (dlopen, separate symbol tables) and compares them value by value.
//   diff_driver LIB_REFERENCE LIB_CANDIDATE [--only R1,R2,R3,R4] [--jobs N] [--scale F] [--out DIR] [--trace FILE] [--keep-going] [--flip N]
// R1 well-formed traces, R2 byte fuzzer, R3 the real trace (table + engine access pattern), R4 open failures in child processes.
// Each (scenario, seed) is a unit run by a worker process; at most ONE table per library is live at any time (sessions load, compare, free).
// Build: g++ -O2 -std=c++17 -Wall -Wextra -I../include -o diff_driver diff_driver.cpp -ldl
//
// Raw log lines (out/units.log): "BEGIN scen seed" and "UNIT scen seed cmp chk nu files bytes PASS|FAIL" (fields 4..8 are numbers). cmp = equality checks reference-vs-candidate
// (one per compared scalar or string); chk = single-library property checks (non-NULL pointer when length > 0, contract text of R4); nu = rt_next_use calls per library.
#include "routetable.h"
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
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
#include <ctime>
#include <map>
#include <set>
#include <string>
#include <vector>

using std::string;
typedef std::vector<int> VI;

// ------------------------------------------------------------------------------------------------ libraries
struct Api {
    void *h = nullptr; string path;
    rt_table *(*load)(const char *, int) = nullptr; void (*fr)(rt_table *) = nullptr;
    size_t (*calls)(const rt_table *) = nullptr; int (*layer)(const rt_table *, size_t) = nullptr;
    const int *(*experts)(const rt_table *, size_t, size_t *) = nullptr; const int *(*counts)(const rt_table *, size_t, size_t *) = nullptr;
    void (*setcur)(rt_table *, long) = nullptr; int (*nu)(rt_table *, int) = nullptr;
};

static bool load_api(Api &a, const char *path)
{
    a.path = path;
    a.h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!a.h) { fprintf(stderr, "dlopen %s: %s\n", path, dlerror()); return false; }
    bool ok = true;
#define SYM(field, name) do { void *s_ = dlsym(a.h, name); if (!s_) { fprintf(stderr, "missing symbol %s in %s\n", name, path); ok = false; } else memcpy((void *)&a.field, &s_, sizeof s_); } while (0)
    SYM(load, "rt_load"); SYM(fr, "rt_free"); SYM(calls, "rt_calls"); SYM(layer, "rt_layer"); SYM(experts, "rt_experts"); SYM(counts, "rt_counts");
    SYM(setcur, "rt_set_current"); SYM(nu, "rt_next_use");
#undef SYM
    return ok;
}

// ------------------------------------------------------------------------------------------------ utilities
static string sfmt(const char *f, ...)
{
    va_list ap; va_start(ap, f);
    va_list ap2; va_copy(ap2, ap);
    const int n = vsnprintf(nullptr, 0, f, ap); va_end(ap);
    string s((size_t)(n > 0 ? n : 0), '\0');
    if (n > 0) vsnprintf(&s[0], (size_t)n + 1, f, ap2);
    va_end(ap2);
    return s;
}
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
static string esc(const string &s, size_t max = 200)
{
    string o;
    for (size_t i = 0; i < s.size() && o.size() < max; i++) {
        const unsigned char c = (unsigned char)s[i];
        if (c == '\n') o += "\\n"; else if (c == '\r') o += "\\r"; else if (c == '\t') o += "\\t"; else if (c == '\\') o += "\\\\";
        else if (c >= 32 && c < 127) o += (char)c; else o += sfmt("\\x%02x", c);
    }
    if (s.size() > 0 && o.size() >= max) o += sfmt("...(%zu bytes)", s.size());
    return o;
}
static bool read_all(const string &path, string &out)
{
    FILE *f = fopen(path.c_str(), "rb"); if (!f) return false;
    char buf[65536]; size_t n; out.clear();
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) out.append(buf, n);
    fclose(f); return true;
}
static bool write_file(const string &path, const string &data)
{
    FILE *f = fopen(path.c_str(), "wb"); if (!f) return false;
    const bool ok = data.empty() || fwrite(data.data(), 1, data.size(), f) == data.size();
    return fclose(f) == 0 && ok;
}
static string hexdump(const string &path, size_t head = 384, size_t tail = 96)
{
    string d; if (!read_all(path, d)) return "  (cannot re-read " + path + ")\n";
    string o = sfmt("  file %s: %zu bytes; hex dump%s\n", path.c_str(), d.size(), d.size() > head + tail ? sfmt(" (first %zu and last %zu bytes)", head, tail).c_str() : "");
    auto dump = [&](size_t a, size_t b) {
        for (size_t i = a; i < b; i += 16) {
            o += sfmt("  %08zx ", i);
            for (size_t j = 0; j < 16; j++) { if (i + j < b) o += sfmt(" %02x", (unsigned char)d[i + j]); else o += "   "; }
            o += "  |";
            for (size_t j = 0; j < 16 && i + j < b; j++) { const unsigned char c = (unsigned char)d[i + j]; o += (c >= 32 && c < 127) ? (char)c : '.'; }
            o += "|\n";
        }
    };
    if (d.size() <= head + tail) dump(0, d.size());
    else { dump(0, head); o += sfmt("  ... %zu bytes omitted ...\n", d.size() - head - tail); dump(d.size() - tail, d.size()); }
    return o;
}

// ------------------------------------------------------------------------------------------------ run context, counters, mismatch reporting
struct Stats { uint64_t cmp = 0, chk = 0, nu = 0, files = 0, bytes = 0, ncalls = 0, nexp = 0, ncnt = 0, nneg = 0, nuhit = 0, feat[12] = {}; };   // from ncalls on = coverage, logged after the verdict
static Stats g_st;
static const char *const FEAT_NAMES[12] = { "empty", "only-newlines", "no-final-newline", "NUL", "tab", "CR", "byte>=0x80", "digit-run>=19", "digit-run>=10", "colon-without-digit", "sign-char", "line>4096" };
// number of generated files that contain each feature (a file counts once per feature)
static void note_features(const string &s)
{
    bool f[12] = {};
    f[0] = s.empty(); f[1] = !s.empty() && s.find_first_not_of('\n') == string::npos; f[2] = !s.empty() && s.back() != '\n';
    size_t run = 0, line = 0;
    for (size_t i = 0; i < s.size(); i++) {
        const unsigned char c = (unsigned char)s[i];
        if (c == 0) f[3] = true;
        if (c == '\t') f[4] = true;
        if (c == '\r') f[5] = true;
        if (c >= 0x80) f[6] = true;
        if (c >= '0' && c <= '9') { run++; if (run >= 19) f[7] = true; if (run >= 10) f[8] = true; } else run = 0;
        if (c == ':' && (i + 1 >= s.size() || s[i + 1] < '0' || s[i + 1] > '9')) f[9] = true;
        if (c == '-' || c == '+') f[10] = true;
        if (c == '\n') line = 0; else if (++line > 4096) f[11] = true;
    }
    for (int i = 0; i < 12; i++) g_st.feat[i] += f[i];
}
static bool put_file(const string &path, const string &data) { note_features(data); return write_file(path, data); }
static uint64_t g_flip = 0;                                   // --flip N: the candidate's N-th compared scalar of each unit is perturbed (plumbing self-test of the comparison path)
static string g_scen; static uint64_t g_seed = 0; static int g_session = 0;
static string g_file; static int g_E = 0; static const char *g_phase = "init";
static volatile int g_lib = 0; static uint64_t g_opidx = 0; static long g_cur = -1;
struct OpRec { int kind; long a; int key; int r0, r1; };       // kind 0 = rt_set_current(a), 1 = rt_next_use(key) with both results
static OpRec g_ring[12]; static uint64_t g_ringn = 0;
struct Mismatch { string text; };

[[noreturn]] static void throw_mismatch(const string &why)
{
    string t = sfmt("MISMATCH scenario %s seed %llu session %d phase '%s' op#%llu (driver's current value cur=%ld)\n  file %s experts=%d\n  %s\n",
                    g_scen.c_str(), (unsigned long long)g_seed, g_session, g_phase, (unsigned long long)g_opidx, g_cur, esc(g_file, 300).c_str(), g_E, why.c_str());
    const uint64_t n = std::min<uint64_t>(g_ringn, 12);
    if (n) {
        t += "  last operations (oldest first; library 0 = reference, library 1 = candidate):\n";
        for (uint64_t i = g_ringn - n; i < g_ringn; i++) {
            const OpRec &r = g_ring[i % 12];
            if (r.kind == 0) t += sfmt("    rt_set_current(%ld)\n", r.a);
            else t += sfmt("    rt_next_use(%d) -> %d | %d\n", r.key, r.r0, r.r1);
        }
    }
    throw Mismatch{ t };
}
#define EQ(A, B, ...) do { long long a_ = (long long)(A), b_ = (long long)(B); g_st.cmp++; if (g_flip && g_st.cmp == g_flip) b_ ^= 1; \
    if (a_ != b_) throw_mismatch(sfmt(__VA_ARGS__) + sfmt("\n  library 0 (reference) = %lld, library 1 (candidate) = %lld", a_, b_)); } while (0)
#define CHK(COND, ...) do { g_st.chk++; if (!(COND)) throw_mismatch(sfmt(__VA_ARGS__)); } while (0)

// ------------------------------------------------------------------------------------------------ one session = one live table per library
struct Tab { std::vector<int> layer; std::vector<size_t> off; VI ex; size_t n() const { return layer.size(); } };
struct W { Api *A; rt_table *t[2] = { nullptr, nullptr }; Tab T; };

static int ekey(int layer, int e, int E) { return (int)((uint32_t)layer * (uint32_t)E + (uint32_t)e); }

static void compare_table(W &w, bool fill)
{
    g_phase = fill ? "table compare" : "table re-compare after the operations";
    g_lib = 0; const size_t n0 = w.A[0].calls(w.t[0]); g_lib = 1; const size_t n1 = w.A[1].calls(w.t[1]);
    EQ(n0, n1, "rt_calls");
    if (fill) { w.T.layer.clear(); w.T.off.assign(1, 0); w.T.ex.clear(); }
    for (size_t i = 0; i < n0; i++) {
        g_lib = 0; const int l0 = w.A[0].layer(w.t[0], i); g_lib = 1; const int l1 = w.A[1].layer(w.t[1], i);
        EQ(l0, l1, "rt_layer(call %zu)", i);
        for (int pass = 0; pass < 2; pass++) {
            const char *nm = pass ? "counts" : "experts";
            size_t m0 = 0, m1 = 0; const int *p0, *p1;
            g_lib = 0; p0 = pass ? w.A[0].counts(w.t[0], i, &m0) : w.A[0].experts(w.t[0], i, &m0);
            g_lib = 1; p1 = pass ? w.A[1].counts(w.t[1], i, &m1) : w.A[1].experts(w.t[1], i, &m1);
            EQ(m0, m1, "%s list length of call %zu", nm, i);
            if (m0) {
                CHK(p0 != nullptr, "library 0 (reference) returned a NULL %s pointer with length %zu for call %zu", nm, m0, i);
                CHK(p1 != nullptr, "library 1 (candidate) returned a NULL %s pointer with length %zu for call %zu", nm, m1, i);
                for (size_t k = 0; k < m0; k++) EQ(p0[k], p1[k], "%s[%zu] of call %zu (layer %d)", nm, k, i, l0);
            }
            if (fill && !pass) w.T.ex.insert(w.T.ex.end(), p0, p0 + m0);
            if (fill) { (pass ? g_st.ncnt : g_st.nexp) += m0; for (size_t k = 0; k < m0; k++) g_st.nneg += p0[k] < 0; }
        }
        if (fill) g_st.ncalls++;
        if (fill) { w.T.layer.push_back(l0); w.T.off.push_back(w.T.ex.size()); }
    }
}

static void do_op(W &w, int kind, long a, int key)
{
    g_opidx++;
    OpRec &r = g_ring[g_ringn++ % 12]; r.kind = kind; r.a = a; r.key = key; r.r0 = r.r1 = 0;
    if (kind == 0) { g_cur = a; g_lib = 0; w.A[0].setcur(w.t[0], a); g_lib = 1; w.A[1].setcur(w.t[1], a); return; }
    g_lib = 0; const int r0 = w.A[0].nu(w.t[0], key); g_lib = 1; const int r1 = w.A[1].nu(w.t[1], key);
    r.r0 = r0; r.r1 = r1; g_st.nu++;
    EQ(r0, r1, "rt_next_use(%d)", key);
    g_st.nuhit += r0 != 1000000000;
}

// ------------------------------------------------------------------------------------------------ operation generator
struct OpGen {
    Rng &R; const Tab &T; int E; long cur = -1; int lastkey = 0;
    OpGen(Rng &r, const Tab &t, int e) : R(r), T(t), E(e) {}
    int existing_key()
    {
        for (int tries = 0; tries < 10 && T.n(); tries++) {
            const size_t c = R.u((uint32_t)T.n());
            const size_t b = T.off[c], e = T.off[c + 1];
            if (e > b) return ekey(T.layer[c], T.ex[b + R.u((uint32_t)(e - b))], E);
        }
        return (int)R.u(100000);
    }
    int gen_key()
    {
        const uint32_t r = R.u(100);
        if (r < 55) return lastkey = existing_key();
        if (r < 63) return lastkey = existing_key() + (R.u(2) ? 1 : -1);
        if (r < 68) return lastkey;
        if (r < 74) return 0;
        if (r < 77) return INT_MAX;
        if (r < 80) return INT_MIN;
        if (r < 83) return -1;
        if (r < 86) return -(int)(1 + R.u(1000000));
        if (r < 94) return (int)R.u(100000);
        if (r < 97) return (int)(uint32_t)R.next();
        return (int)R.u(93) * E;
    }
    long gen_cur()
    {
        const uint32_t r = R.u(100);
        long v;
        if (r < 30) v = cur + 1; else if (r < 42) v = cur; else if (r < 54) v = cur - (long)R.u(50); else if (r < 59) v = -1;
        else if (r < 62) v = -(long)R.u(1000000); else if (r < 64) v = LONG_MIN; else if (r < 66) v = LONG_MAX; else if (r < 68) v = INT_MAX;
        else if (r < 70) v = (long)INT_MAX + 1; else if (r < 72) v = 1L << 32; else if (r < 74) v = 1L << 40; else if (r < 76) v = -(1L << 40);
        else if (r < 78) v = INT_MIN; else if (r < 80) v = (long)INT_MIN - 1; else v = (long)R.u((uint32_t)T.n() + 4);
        return v;
    }
    // appends one action: a single operation or an engine-like burst (set_current(c), then the experts of calls c..c+K)
    void action(std::vector<std::pair<int, std::pair<long, int>>> &out)
    {
        const uint32_t r = R.u(100);
        if (r < 22 && T.n()) {
            long c = R.p(0.5) ? (long)R.u((uint32_t)T.n()) : std::max(0L, cur + (long)R.u(5) - 1);
            if (c > (long)T.n()) c = (long)T.n();
            cur = c; out.push_back({ 0, { c, 0 } });
            const size_t K = 1 + R.u(R.p(0.3) ? 100 : 8);
            for (size_t j = (size_t)c; j < T.n() && j <= (size_t)c + K; j++)
                for (size_t k = T.off[j]; k < T.off[j + 1]; k++) out.push_back({ 1, { 0, ekey(T.layer[j], T.ex[k], E) } });
        } else if (r < 45) { cur = gen_cur(); out.push_back({ 0, { cur, 0 } }); }
        else out.push_back({ 1, { 0, gen_key() } });
    }
};

static void run_ops(W &w, Rng &R, long nops, bool end_big)
{
    g_phase = "operations";
    OpGen G(R, w.T, g_E);
    long done = 0;
    std::vector<std::pair<int, std::pair<long, int>>> acts;
    while (done < nops) {
        acts.clear(); G.action(acts);
        for (auto &a : acts) { do_op(w, a.first, a.second.first, a.second.second); done++; }
    }
    if (end_big) { const long v = R.u(2) ? LONG_MAX : (long)R.u(100000); do_op(w, 0, v, 0); }
}

static void fresh_queries(W &w, Rng &R)
{
    g_phase = "fresh-table queries (no set_current yet, current = -1)";
    OpGen G(R, w.T, g_E);
    const int n = 4 + (int)R.u(12);
    for (int i = 0; i < n; i++) do_op(w, 1, 0, R.p(0.7) ? G.existing_key() : G.gen_key());
}

static void session(Api *A, Rng &R, const string &path, int E, long nops, int idx)
{
    g_session = idx; g_file = path; g_E = E; g_cur = -1; g_opidx = 0; g_ringn = 0;
    W w; w.A = A;
    { string d; if (read_all(path, d)) g_st.bytes += d.size(); g_st.files++; }
    g_phase = "rt_load";
    g_lib = 0; w.t[0] = A[0].load(path.c_str(), E);
    g_lib = 1; w.t[1] = A[1].load(path.c_str(), E);
    compare_table(w, true);
    fresh_queries(w, R);
    if (nops > 0) run_ops(w, R, nops, true);
    compare_table(w, false);
    g_phase = "rt_free";
    g_lib = 0; A[0].fr(w.t[0]); g_lib = 1; A[1].fr(w.t[1]);
}

static int pick_E(Rng &R)
{
    const uint32_t r = R.u(100);
    return r < 50 ? 896 : r < 60 ? 1 : r < 75 ? 7 : 100000;
}

// ------------------------------------------------------------------------------------------------ R1: well-formed traces
static string gen_r1(Rng &R)
{
    const int ncalls = R.p(0.04) ? 15000 + (int)R.u(10000) : 200 + (int)R.u(5800);
    const int lmode = (int)R.u(3), emode = (int)R.u(4), cmode = (int)R.u(4);
    static const int MAXE[] = { 4, 16, 24, 48 };
    const int maxe = MAXE[R.u(4)];
    const double dup = R.u(3) == 0 ? 0.0 : (R.u(2) ? 0.1 : 0.3);
    const double pempty = R.p(0.5) ? 0.0 : 0.1;
    string s; s.reserve((size_t)ncalls * 100);
    for (int i = 0; i < ncalls; i++) {
        const int layer = lmode == 0 ? (int)R.u(93) : lmode == 1 ? 1 + (i % 92) : (int)R.u(4);
        s += std::to_string(layer); s += ' '; s += std::to_string(R.p(0.5) ? (int)R.u(2000) : 1 + (int)R.u(4096));
        const int n = R.p(pempty) ? 0 : (int)R.u((uint32_t)maxe + 1);
        const int lc = cmode == 3 ? (int)R.u(3) : cmode;   // 0 all counts, 1 some, 2 none
        VI es;
        for (int k = 0; k < n; k++) {
            int e;
            if (!es.empty() && R.p(dup)) e = es[R.u((uint32_t)es.size())];
            else if (emode == 0) e = (int)R.u(896);
            else if (emode == 1) e = (int)std::min(R.u(896), std::min(R.u(896), R.u(896)));
            else if (emode == 2) e = (int)R.u(16);
            else e = (int)R.u(100000);
            es.push_back(e);
            s += ' '; s += std::to_string(e);
            if (lc == 0 || (lc == 1 && R.p(0.5))) { s += ':'; s += std::to_string(R.p(0.8) ? 1 + (int)R.u(64) : (int)R.u(5)); }
        }
        s += '\n';
    }
    return s;
}

static int unit_R1(Api *A, uint64_t seed, const string &dir)
{
    Rng R(mix64(seed * 0x1234567ull + 11));
    const string fa = dir + sfmt("/R1-%llu-a.trace", (unsigned long long)seed), fb = dir + sfmt("/R1-%llu-b.trace", (unsigned long long)seed);
    put_file(fa, gen_r1(R));
    const int E = pick_E(R);
    session(A, R, fa, E, 4000 + (long)R.u(36000), 1);
    session(A, R, fa, E, 1000 + (long)R.u(8000), 2);                 // free + reload of the same file
    if (R.u(2)) session(A, R, fa, pick_E(R), 2000 + (long)R.u(6000), 3);   // same file, other experts value
    put_file(fb, gen_r1(R));
    session(A, R, fb, pick_E(R), 2000 + (long)R.u(10000), 4);        // a different file after the free
    unlink(fa.c_str()); unlink(fb.c_str());
    return 0;
}

// ------------------------------------------------------------------------------------------------ R2: fuzzer
static char fuzz_byte(Rng &R)
{
    const uint32_t r = R.u(100);
    if (r < 38) return (char)('0' + R.u(10));
    if (r < 52) return ' ';
    if (r < 60) return '\n';
    if (r < 68) return ':';
    if (r < 71) return '-';
    if (r < 73) return '+';
    if (r < 76) return '\t';
    if (r < 78) return '\r';
    if (r < 79) return R.u(2) ? '\v' : '\f';
    if (r < 83) return (char)(R.u(2) ? 'a' + R.u(26) : 'A' + R.u(26));
    if (r < 85) return '\0';
    if (r < 87) return (char)(0x80 + R.u(128));
    if (r < 88) return ".,;_=x/\\%"[R.u(9)];
    return (char)('0' + R.u(10));
}
static const char *const NUMS[] = {
    "0", "1", "7", "895", "896", "92", "93", "100", "00012", "2147483647", "2147483648", "2147483649", "4294967295", "4294967296", "4294967297",
    "9223372036854775807", "9223372036854775808", "9223372036854775809", "18446744073709551615", "18446744073709551616", "99999999999999999999",
    "000000000000000000000001", "-1", "-2147483648", "-2147483649", "-9223372036854775808", "-9223372036854775809", "+5", "+0", "-0", "x", "e", "\xff", "\x80" };
static string rand_digits(Rng &R, size_t n)
{
    string s; for (size_t i = 0; i < n; i++) s += (char)('0' + R.u(10)); return s;
}
static string fuzz_token(Rng &R)
{
    const uint32_t r = R.u(100);
    if (r < 35) return std::to_string(R.u(900));
    if (r < 55) return NUMS[R.u(sizeof NUMS / sizeof *NUMS)];
    if (r < 62) return rand_digits(R, 1 + R.u(60));
    if (r < 70) return ":";
    if (r < 76) return " ";
    if (r < 79) return "\t";
    if (r < 82) return "-";
    if (r < 84) return "+";
    if (r < 87) return string(1, (char)('a' + R.u(26)));
    if (r < 89) return string("\0", 1);
    if (r < 91) return string(1, (char)(0x80 + R.u(128)));
    if (r < 94) return "\r\n";
    if (r < 95) return R.u(2) ? "\v" : "\f";
    if (r < 96) return std::to_string(R.u(896)) + ":" + std::to_string(R.u(70));
    if (r < 98) return std::to_string(R.u(896)) + ":" + NUMS[R.u(sizeof NUMS / sizeof *NUMS)];
    return std::to_string(R.u(896)) + ": " + std::to_string(R.u(70));
}
static string fuzz_wellformed(Rng &R, int lines)
{
    string s;
    for (int i = 0; i < lines; i++) {
        s += std::to_string(R.u(93)) + " " + std::to_string(R.u(2000));
        const int n = (int)R.u(20);
        for (int k = 0; k < n; k++) { s += " " + std::to_string(R.u(896)); if (R.u(2)) s += ":" + std::to_string(1 + R.u(64)); }
        s += "\n";
    }
    return s;
}
static const char *const SPECIAL[] = {
    "5 3 7:9999999999999999999999\n", "1 1 4294967297:4294967297\n", ":::::\n", "3 5 7:\n", "3 5 7:", "3 5 7:+\n", "3 5 7:-\n", "- 5\n", "+3 +4 5\n",
    "3 4 5:-4 6: 7 8:\t9\n", "1\t2\t3\n", "9223372036854775807 9223372036854775808 9223372036854775808\n", "92 1 3:4:5:6\n", "0 0 0:0 0:0\n", "\n", " \n", "\t\n", "x\n",
    "5 5 1\r\n", "5 5 1 \r\n", "7 7 7:7\r", "\0\n", "1 2 3\0 4 5\n", "1 2 \n", "1 2 3 4 x 5 6\n", "-1 -2 3\n", "2147483648 2147483649 4294967296:4294967297\n",
    "\f5 6 7:1\n", "1\v2 3\n", "\v\f 4 5 6\n", "4 5 6:\v7\n", "4 5 6:\f-7\n" };

static string gen_r2(Rng &R)
{
    string s;
    const uint32_t mode = R.u(100);
    if (mode < 4) { /* empty file */ }
    else if (mode < 8) s = string(1 + R.u(50), '\n');
    else if (mode < 10) { const int n = 1 + (int)R.u(60); for (int i = 0; i < n; i++) s += " \t\n"[R.u(3)]; }
    else if (mode < 35) { const size_t len = 1 + (R.u(4) == 0 ? R.u(4000) : R.u(200)); for (size_t i = 0; i < len; i++) s += fuzz_byte(R); }
    else if (mode < 65) {
        const int lines = 1 + (int)R.u(40);
        for (int i = 0; i < lines; i++) {
            const int nt = (int)R.u(13);
            for (int k = 0; k < nt; k++) { s += fuzz_token(R); const uint32_t sp = R.u(10); if (sp < 7) s += ' '; else if (sp == 7) s += "  "; else if (sp == 8) s += '\t'; }
            if (R.u(10) < 9) s += '\n';
        }
    } else if (mode < 90) {
        s = fuzz_wellformed(R, 1 + (int)R.u(40));
        const int nm = (int)R.u(1 + (uint32_t)s.size() / 8 + 1);
        for (int i = 0; i < nm && !s.empty(); i++) {
            const size_t pos = R.u((uint32_t)s.size());
            const uint32_t kind = R.u(5);
            if (kind == 0) s[pos] = fuzz_byte(R);
            else if (kind == 1) s.insert(s.begin() + (long)pos, fuzz_byte(R));
            else if (kind == 2) s.erase(pos, 1);
            else if (kind == 3) s.insert(pos, rand_digits(R, 1 + R.u(30)));
            else s.insert(pos, fuzz_token(R));
        }
    } else {
        const int k = 1 + (int)R.u(6);
        for (int i = 0; i < k; i++) {
            const uint32_t r = R.u(10);
            if (r < 6) { const char *p = SPECIAL[R.u(sizeof SPECIAL / sizeof *SPECIAL)]; if (!strcmp(p, "\0\n")) s += string("\0\n", 2); else if (!strncmp(p, "1 2 3", 5) && p[5] == '\0') s += string("1 2 3\0 4 5\n", 11); else s += p; }
            else if (r < 8) s += std::to_string(R.u(93)) + " " + std::to_string(R.u(99)) + " " + rand_digits(R, 100 + R.u(R.p(0.1) ? 60000 : 400)) + "\n";
            else s += std::to_string(R.u(93)) + " 5 " + std::to_string(R.u(896)) + ":" + rand_digits(R, 10 + R.u(40)) + " " + std::to_string(R.u(896)) + "\n";
        }
    }
    const uint32_t t = R.u(10);
    if (t < 3 && !s.empty() && s.back() == '\n') s.pop_back();
    else if (t == 3) s += '\n';
    else if (t == 4) s += "\r\n";
    return s;
}

static int unit_R2(Api *A, uint64_t seed, const string &dir)
{
    Rng R(mix64(seed * 0x7654321ull + 22));
    const string fa = dir + sfmt("/R2-%llu-a.trace", (unsigned long long)seed), fb = dir + sfmt("/R2-%llu-b.trace", (unsigned long long)seed);
    put_file(fa, gen_r2(R));
    const int E = pick_E(R);
    session(A, R, fa, E, 300 + (long)R.u(2700), 1);
    session(A, R, fa, E, 100 + (long)R.u(500), 2);
    put_file(fb, gen_r2(R));
    session(A, R, fb, pick_E(R), 300 + (long)R.u(2700), 3);
    unlink(fa.c_str()); unlink(fb.c_str());
    return 0;
}

// ------------------------------------------------------------------------------------------------ R3: the real trace
static void r3_sweep(W &w, size_t j0, size_t j1)
{
    for (size_t j = j0; j < j1 && j < w.T.n(); j++)
        for (size_t k = w.T.off[j]; k < w.T.off[j + 1]; k++) do_op(w, 1, 0, ekey(w.T.layer[j], w.T.ex[k], g_E));
}

static void r3_session(Api *A, Rng &R, const string &path, int E, int idx, double rp, bool vq, long istop, bool backward)
{
    g_session = idx; g_file = path; g_E = E; g_cur = -1; g_opidx = 0; g_ringn = 0; g_st.files++;
    W w; w.A = A;
    g_phase = "rt_load";
    g_lib = 0; w.t[0] = A[0].load(path.c_str(), E); g_lib = 1; w.t[1] = A[1].load(path.c_str(), E);
    compare_table(w, true);
    g_phase = "engine access pattern (current = i, look-ahead i..i+92)";
    const long N = (long)w.T.n();
    OpGen G(R, w.T, E);
    const long last = istop < 0 ? N + 93 : std::min(istop, N + 93);
    for (long i = 0; i < last; i++) {
        const int reps = R.p(rp) ? 2 + (int)R.u(2) : 1;
        for (int rr = 0; rr < reps; rr++) {
            do_op(w, 0, i, 0);
            if (rr == 0) {
                r3_sweep(w, (size_t)i, (size_t)i + 93);            // route_hook / rc_plan window: calls i..i+92
                if (vq) r3_sweep(w, (size_t)i + 1, (size_t)i + 93);   // rc_plan_vq: experts of the calls after i
                for (int q = 0; q < 6; q++) do_op(w, 1, 0, G.gen_key());
            } else r3_sweep(w, (size_t)i, (size_t)i + 1);
        }
    }
    if (backward) {
        g_phase = "backward pass (current decreasing, pointers already advanced)";
        for (long i = N + 2; i >= 0; i -= 1 + (long)R.u(3)) {
            do_op(w, 0, i, 0);
            r3_sweep(w, (size_t)i, (size_t)i + 5);
            for (int q = 0; q < 20; q++) do_op(w, 1, 0, G.existing_key());
        }
    }
    compare_table(w, false);
    g_phase = "rt_free";
    g_lib = 0; A[0].fr(w.t[0]); g_lib = 1; A[1].fr(w.t[1]);
}

static int unit_R3(Api *A, uint64_t seed, const string &trace)
{
    Rng R(mix64(seed * 0x31415ull + 33));
    static const double RP[] = { 0.0, 0.05, 0.2, 0.5, 0.1, 0.3 };
    const int E = seed < 6 ? 896 : seed == 6 ? 7 : seed == 7 ? 1 : 100000;
    const double rp = RP[seed % 6]; const bool vq = (seed & 1) != 0;
    r3_session(A, R, trace, E, 1, rp, vq, -1, true);
    r3_session(A, R, trace, E, 2, 0.1, true, 400, false);            // free + reload of the same file: state must be fresh
    r3_session(A, R, trace, E == 896 ? 7 : 896, 3, 0.0, false, 200, true);   // same file, other experts value
    return 0;
}

// ------------------------------------------------------------------------------------------------ R4: open failures, each library in a child process
struct ChildRes { int status = 0; string out, err; bool timeout = false; };
static string g_self;

static ChildRes run_child(const string &lib, const string &path, int E)
{
    ChildRes cr; int po[2], pe[2];
    if (pipe(po) || pipe(pe)) { cr.status = -1; return cr; }
    const pid_t pid = fork();
    if (pid == 0) {
        dup2(po[1], 1); dup2(pe[1], 2); close(po[0]); close(po[1]); close(pe[0]); close(pe[1]);
        const string es = std::to_string(E);
        execl(g_self.c_str(), g_self.c_str(), "--child", lib.c_str(), path.c_str(), es.c_str(), (char *)nullptr);
        _exit(127);
    }
    close(po[1]); close(pe[1]);
    struct pollfd fds[2] = { { po[0], POLLIN, 0 }, { pe[0], POLLIN, 0 } };
    int open_fds = 2; char buf[65536]; const time_t t0 = time(nullptr);
    while (open_fds > 0) {
        if (time(nullptr) - t0 > 30) { kill(pid, SIGKILL); cr.timeout = true; break; }
        if (poll(fds, 2, 1000) < 0 && errno != EINTR) break;
        for (int i = 0; i < 2; i++) {
            if (fds[i].fd < 0) continue;
            if (fds[i].revents & (POLLIN | POLLHUP)) {
                const ssize_t n = read(fds[i].fd, buf, sizeof buf);
                if (n > 0) (i ? cr.err : cr.out).append(buf, (size_t)n); else { close(fds[i].fd); fds[i].fd = -1; open_fds--; }
            }
        }
    }
    for (int i = 0; i < 2; i++) if (fds[i].fd >= 0) close(fds[i].fd);
    int st = 0; waitpid(pid, &st, 0);
    cr.status = WIFEXITED(st) ? WEXITSTATUS(st) : 1000 + WTERMSIG(st);   // 1000+sig = killed by a signal
    return cr;
}

static int child_main(const char *lib, const char *path, int E)
{
    Api a; if (!load_api(a, lib)) return 2;
    rt_table *t = a.load(path, E);
    const size_t n = a.calls(t); uint64_t h = 1469598103934665603ull;
    auto mix = [&](uint64_t v) { h ^= v; h *= 1099511628211ull; };
    for (size_t i = 0; i < n; i++) {
        mix((uint32_t)a.layer(t, i));
        for (int pass = 0; pass < 2; pass++) { size_t m = 0; const int *p = pass ? a.counts(t, i, &m) : a.experts(t, i, &m); mix(m); for (size_t k = 0; k < m; k++) mix((uint32_t)p[k]); }
    }
    printf("loaded calls=%zu hash=%016llx\n", n, (unsigned long long)h);
    fflush(stdout);
    a.fr(t);
    return 0;
}

struct R4Case { string name, path; int expect; };   // expect: -1 = open must fail with the contract text; >= 0 = loads, that many calls; -2 = no contract statement (report only)
static string g_r4log, g_r4phase;

static void append_log(const string &file, const string &line)
{
    const int fd = open(file.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd >= 0) { if (write(fd, line.data(), line.size()) < 0) {} close(fd); }
}
static string stat_str(int s) { return s >= 1000 ? sfmt("signal %d", s - 1000) : sfmt("exit %d", s); }

static void r4_run(const Api *A, const R4Case &c)
{
    g_r4phase = c.name; g_phase = g_r4phase.c_str(); g_file = c.path.size() > 120 ? c.path.substr(0, 120) + sfmt("...(%zu bytes)", c.path.size()) : c.path; g_E = 896;
    const ChildRes r0 = run_child(A[0].path, c.path, 896), r1 = run_child(A[1].path, c.path, 896);
    const string exp_fail = "cannot read route cache " + c.path + "\n";
    string verdict;
    if (c.expect == -1) {
        for (int i = 0; i < 2; i++) {
            const ChildRes &r = i ? r1 : r0;
            CHK(!r.timeout && r.status == 1 && r.out.empty() && r.err == exp_fail,
                "library %d (%s) does not follow the contract for an unopenable file: %s, stdout '%s', stderr '%s' (expected exit 1, empty stdout, stderr '%s')",
                i, i ? "candidate" : "reference", stat_str(r.status).c_str(), esc(r.out).c_str(), esc(r.err).c_str(), esc(exp_fail).c_str());
        }
        verdict = "contract-ok";
    } else if (c.expect >= 0) {
        const string pre = sfmt("loaded calls=%d hash=", c.expect);
        for (int i = 0; i < 2; i++) {
            const ChildRes &r = i ? r1 : r0;
            CHK(!r.timeout && r.status == 0 && r.err.empty() && r.out.compare(0, pre.size(), pre) == 0,
                "library %d (%s) did not load %s as expected: %s, stdout '%s', stderr '%s' (expected exit 0, '%s...')", i, i ? "candidate" : "reference", c.name.c_str(),
                stat_str(r.status).c_str(), esc(r.out).c_str(), esc(r.err).c_str(), pre.c_str());
        }
        verdict = "loads-ok";
    } else verdict = "report-only";
    append_log(g_r4log, sfmt("CASE %-34s path='%s' | ref: %s out='%s' err='%s' | rust: %s out='%s' err='%s' | %s\n", c.name.c_str(), esc(c.path, 60).c_str(), stat_str(r0.status).c_str(),
                            esc(r0.out, 60).c_str(), esc(r0.err, 90).c_str(), stat_str(r1.status).c_str(), esc(r1.out, 60).c_str(), esc(r1.err, 90).c_str(), verdict.c_str()));
    EQ(r0.status, r1.status, "exit status of case '%s' (path %s)", c.name.c_str(), esc(g_file, 300).c_str());
    g_st.cmp++; if (r0.out != r1.out || (g_flip && g_st.cmp == g_flip)) throw_mismatch(sfmt("stdout of case '%s' differs: reference '%s' | candidate '%s'", c.name.c_str(), esc(r0.out).c_str(), esc(r1.out).c_str()));
    g_st.cmp++; if (r0.err != r1.err || (g_flip && g_st.cmp == g_flip)) throw_mismatch(sfmt("stderr of case '%s' differs: reference '%s' | candidate '%s'", c.name.c_str(), esc(r0.err).c_str(), esc(r1.err).c_str()));
}

static void rm_deep(const string &base, const std::vector<string> &comps)
{
    if (chdir(base.c_str())) return;
    size_t d = 0;
    for (; d < comps.size(); d++) if (chdir(comps[d].c_str())) break;
    unlink("t.trace");
    for (size_t k = d; k > 0; k--) { if (chdir("..")) break; rmdir(comps[k - 1].c_str()); }
    if (chdir("/")) {}
    rmdir(base.c_str());
}

static int unit_R4(Api *A, uint64_t seed, const string &dir)
{
    if (seed == 0) {
        const string base = dir + "/r4fixed";
        mkdir(base.c_str(), 0755);
        const string good = base + "/good.trace", empty = base + "/empty.trace", noperm = base + "/noperm.trace", nodir = base + "/nodir", loop = base + "/loop", isdir = base + "/adir";
        write_file(good, "1 5 3:2 4\n0 1\n\n"); write_file(empty, ""); write_file(noperm, "1 1 1:1\n");
        chmod(noperm.c_str(), 0); mkdir(nodir.c_str(), 0755); write_file(nodir + "/inner.trace", "1 1 1:1\n"); chmod(nodir.c_str(), 0); mkdir(isdir.c_str(), 0755);
        if (symlink("loop", loop.c_str())) {}
        // a real file behind a ~3800-byte absolute path
        const string deepbase = dir + "/r4deep"; mkdir(deepbase.c_str(), 0755);
        std::vector<string> comps; string deep = deepbase;
        while (deep.size() + 252 + 10 < 3900) { comps.push_back(string(250, (char)('a' + comps.size() % 26))); deep += "/" + comps.back(); }
        {
            if (chdir(deepbase.c_str())) {}
            for (const string &c : comps) { mkdir(c.c_str(), 0755); if (chdir(c.c_str())) break; }
            write_file("t.trace", "3 3 7:7 8\n4 4 9\n5\n");
            if (chdir("/")) {}
        }
        std::vector<R4Case> cs = {
            { "missing-absolute", "/nonexistent-rt-verify/route.trace", -1 }, { "missing-in-existing-dir", base + "/does-not-exist", -1 }, { "empty-path", "", -1 },
            { "directory-tmp", "/tmp", -2 }, { "directory-adir", isdir, -2 }, { "directory-root", "/", -2 }, { "directory-dot", ".", -2 },
            { "unreadable-file-mode000", noperm, -1 }, { "file-in-dir-mode000", nodir + "/inner.trace", -1 }, { "symlink-loop", loop, -1 },
            { "component-300-bytes", base + "/" + string(300, 'z'), -1 }, { "path-5000-single-component", base + "/" + string(5000, 'a'), -1 },
            { "path-4800-short-components", [&] { string p = base; while (p.size() < 4800) p += "/a"; return p; }(), -1 },
            { "deep-real-file-~3800", deep + "/t.trace", 3 },
            { "special-bytes-missing", base + "/q%s%n \t\n\x01\xff\xfe'\"$(x)`;|&", -1 }, { "dev-null", "/dev/null", 0 }, { "proc-self-mem-EIO", "/proc/self/mem", -2 },
            { "regular-file-trailing-slash", good + "/", -1 }, { "good-small-trace", good, 3 }, { "empty-file", empty, 0 },
        };
        const uid_t uid = getuid();
        append_log(g_r4log, sfmt("# R4 fixed cases: uid %d; deep path length %zu bytes\n", (int)uid, deep.size() + 8));
        auto cleanup = [&]() {
            chmod(nodir.c_str(), 0755); chmod(noperm.c_str(), 0644);
            unlink(good.c_str()); unlink(empty.c_str()); unlink(noperm.c_str()); unlink((nodir + "/inner.trace").c_str()); rmdir(nodir.c_str()); rmdir(isdir.c_str()); unlink(loop.c_str()); rmdir(base.c_str());
            rm_deep(deepbase, comps);
        };
        try { for (const R4Case &c : cs) r4_run(A, c); } catch (...) { cleanup(); throw; }
        cleanup();
        return 0;
    }
    Rng R(mix64(seed * 0x2718ull + 44));
    for (int i = 0; i < 20; i++) {
        string p;
        if (R.p(0.5)) p = "/nonexistent-rt-verify/";
        const size_t len = 1 + R.u(R.p(0.2) ? 300 : 40);
        for (size_t k = 0; k < len; k++) { char c; do { c = (char)R.u(256); } while (c == '\0'); p += c; }
        const bool exists = access(p.c_str(), F_OK) == 0;
        r4_run(A, R4Case{ sfmt("random-path-%llu-%d", (unsigned long long)seed, i), p, exists ? -2 : -1 });
    }
    return 0;
}

// ------------------------------------------------------------------------------------------------ worker and parent
struct Opts {
    string ref, cand, out, trace; std::set<string> only; int jobs = 48; double scale = 1.0; bool keep_going = false; uint64_t flip = 0;
};
struct Unit { string scen; uint64_t seed; };

static void crash_handler(int sig)
{
    char b[600];
    const int n = snprintf(b, sizeof b, "CRASH signal %d scenario %s seed %llu session %d phase '%s' library %d (0 = reference, 1 = candidate) op#%llu file %s\n", sig, g_scen.c_str(),
                           (unsigned long long)g_seed, g_session, g_phase, (int)g_lib, (unsigned long long)g_opidx, g_file.c_str());
    if (n > 0 && write(2, b, (size_t)n) < 0) {}
    signal(sig, SIG_DFL); raise(sig);
}

static int worker(const Opts &o, const Unit &u)
{
    const string work = o.out + "/work", failf = o.out + "/fail";
    g_scen = u.scen; g_seed = u.seed; g_r4log = o.out + "/r4-cases.log";
    const string errp = work + sfmt("/%s-%llu.err", u.scen.c_str(), (unsigned long long)u.seed);
    const int efd = open(errp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (efd >= 0) { dup2(efd, 2); close(efd); }
    for (int s : { SIGSEGV, SIGABRT, SIGBUS, SIGILL, SIGFPE }) signal(s, crash_handler);
    append_log(o.out + "/units.log", sfmt("BEGIN %s %llu\n", u.scen.c_str(), (unsigned long long)u.seed));
    Api A[2];
    if (u.scen != "R4" && (!load_api(A[0], o.ref.c_str()) || !load_api(A[1], o.cand.c_str()))) return 2;
    if (u.scen == "R4") { A[0].path = o.ref; A[1].path = o.cand; }
    g_flip = o.flip;
    string verdict = "PASS", detail;
    try {
        if (u.scen == "R1") unit_R1(A, u.seed, work);
        else if (u.scen == "R2") unit_R2(A, u.seed, work);
        else if (u.scen == "R3") unit_R3(A, u.seed, o.trace);
        else if (u.scen == "R4") unit_R4(A, u.seed, work);
    } catch (Mismatch &m) {
        verdict = "FAIL";
        detail = m.text;
        if (u.scen != "R3" && u.scen != "R4") detail += hexdump(g_file);
        else if (u.scen == "R3") detail += sfmt("  file %s (read only; %s)\n", g_file.c_str(), "not dumped");
        const string fp = failf + sfmt("/%s-%llu.txt", u.scen.c_str(), (unsigned long long)u.seed);
        write_file(fp, detail);
        if (u.scen != "R3" && u.scen != "R4") { string d; if (read_all(g_file, d)) write_file(failf + sfmt("/%s-%llu.input", u.scen.c_str(), (unsigned long long)u.seed), d); }
    }
    {
        string fl; for (int i = 0; i < 12; i++) fl += sfmt("%s%llu", i ? "," : "", (unsigned long long)g_st.feat[i]);
        append_log(o.out + "/units.log", sfmt("UNIT %s %llu %llu %llu %llu %llu %llu %s calls=%llu exp=%llu cnt=%llu neg=%llu nuhit=%llu feat=%s\n", u.scen.c_str(), (unsigned long long)u.seed,
                                              (unsigned long long)g_st.cmp, (unsigned long long)g_st.chk, (unsigned long long)g_st.nu, (unsigned long long)g_st.files, (unsigned long long)g_st.bytes,
                                              verdict.c_str(), (unsigned long long)g_st.ncalls, (unsigned long long)g_st.nexp, (unsigned long long)g_st.ncnt, (unsigned long long)g_st.nneg,
                                              (unsigned long long)g_st.nuhit, fl.c_str()));
    }
    if (verdict == "PASS") {
        struct stat sb; if (stat(errp.c_str(), &sb) == 0 && sb.st_size == 0) unlink(errp.c_str());
        return 0;
    }
    return 1;
}

static void usage() { fprintf(stderr, "usage: diff_driver LIB_REFERENCE LIB_CANDIDATE [--only R1,R2,R3,R4] [--jobs N] [--scale F] [--out DIR] [--trace FILE] [--keep-going] [--flip N]\n"); }

int main(int argc, char **argv)
{
    if (argc >= 5 && !strcmp(argv[1], "--child")) return child_main(argv[2], argv[3], atoi(argv[4]));
    Opts o; std::vector<string> pos;
    {
        char buf[4096]; const ssize_t n = readlink("/proc/self/exe", buf, sizeof buf - 1);
        if (n > 0) { buf[n] = 0; g_self = buf; } else g_self = argv[0];
    }
    const char *home = getenv("HOME");
    o.trace = string(home ? home : "") + "/k3gpu/results/p1d.trace";
    for (int i = 1; i < argc; i++) {
        const string a = argv[i];
        if (a == "--only" && i + 1 < argc) { string s = argv[++i]; size_t p = 0; while (p <= s.size()) { size_t q = s.find(',', p); if (q == string::npos) q = s.size(); if (q > p) o.only.insert(s.substr(p, q - p)); p = q + 1; } }
        else if (a == "--jobs" && i + 1 < argc) o.jobs = atoi(argv[++i]);
        else if (a == "--scale" && i + 1 < argc) o.scale = atof(argv[++i]);
        else if (a == "--out" && i + 1 < argc) o.out = argv[++i];
        else if (a == "--trace" && i + 1 < argc) o.trace = argv[++i];
        else if (a == "--flip" && i + 1 < argc) o.flip = strtoull(argv[++i], nullptr, 10);
        else if (a == "--keep-going") o.keep_going = true;
        else if (a.size() && a[0] == '-') { usage(); return 2; }
        else pos.push_back(a);
    }
    if (pos.size() != 2) { usage(); return 2; }
    {
        char rp[PATH_MAX];
        if (!realpath(pos[0].c_str(), rp)) { fprintf(stderr, "cannot resolve %s\n", pos[0].c_str()); return 2; } o.ref = rp;
        if (!realpath(pos[1].c_str(), rp)) { fprintf(stderr, "cannot resolve %s\n", pos[1].c_str()); return 2; } o.cand = rp;
        if (o.out.empty()) o.out = "out/run";
        mkdir(o.out.c_str(), 0755);
        if (!realpath(o.out.c_str(), rp)) { fprintf(stderr, "cannot create %s\n", o.out.c_str()); return 2; } o.out = rp;
    }
    mkdir((o.out + "/work").c_str(), 0755); mkdir((o.out + "/fail").c_str(), 0755);
    auto want = [&](const char *s) { return o.only.empty() || o.only.count(s); };
    auto cnt = [&](int n) { return std::max(1, (int)(n * o.scale + 0.5)); };
    std::vector<Unit> units;
    if (want("R3")) for (int s = 0; s < std::min(9, cnt(9)); s++) units.push_back({ "R3", (uint64_t)s });
    if (want("R1")) for (int s = 0; s < cnt(400); s++) units.push_back({ "R1", (uint64_t)s });
    if (want("R2")) for (int s = 0; s < cnt(20000); s++) units.push_back({ "R2", (uint64_t)s });
    if (want("R4")) for (int s = 0; s <= cnt(20); s++) units.push_back({ "R4", (uint64_t)s });
    write_file(o.out + "/units.log", "# BEGIN scen seed | UNIT scen seed cmp chk nu files bytes PASS|FAIL\n");
    write_file(o.out + "/r4-cases.log", "");
    string cmdline; for (int i = 0; i < argc; i++) cmdline += string(i ? " " : "") + argv[i];
    printf("diff_driver: ref=%s\n             cand=%s\n             out=%s jobs=%d scale=%g flip=%llu units=%zu trace=%s\n", o.ref.c_str(), o.cand.c_str(), o.out.c_str(), o.jobs, o.scale,
           (unsigned long long)o.flip, units.size(), o.trace.c_str());
    fflush(stdout);
    std::map<pid_t, Unit> running; size_t next = 0; bool stop = false; int nfail = 0, ncrash = 0; string first_fail;
    std::vector<string> crashes;
    while (next < units.size() || !running.empty()) {
        while ((int)running.size() < o.jobs && next < units.size() && (!stop || o.keep_going)) {
            const Unit u = units[next++]; fflush(stdout); fflush(stderr);
            const pid_t pid = fork();
            if (pid == 0) _exit(worker(o, u));
            if (pid < 0) { perror("fork"); return 2; }
            running[pid] = u;
        }
        if (running.empty()) break;
        int st = 0; const pid_t pid = waitpid(-1, &st, 0);
        if (pid < 0) break;
        auto it = running.find(pid); if (it == running.end()) continue;
        const Unit u = it->second; running.erase(it);
        const bool ok = WIFEXITED(st) && WEXITSTATUS(st) == 0;
        if (ok) continue;
        if (WIFEXITED(st) && WEXITSTATUS(st) == 1) { nfail++; stop = true; if (first_fail.empty()) first_fail = o.out + sfmt("/fail/%s-%llu.txt", u.scen.c_str(), (unsigned long long)u.seed); }
        else {
            ncrash++; stop = true;
            string e; read_all(o.out + sfmt("/work/%s-%llu.err", u.scen.c_str(), (unsigned long long)u.seed), e);
            const string line = sfmt("CRASH %s %llu %s | worker stderr: %s\n", u.scen.c_str(), (unsigned long long)u.seed,
                                     WIFSIGNALED(st) ? sfmt("signal %d", WTERMSIG(st)).c_str() : sfmt("exit %d", WEXITSTATUS(st)).c_str(), esc(e, 600).c_str());
            append_log(o.out + "/units.log", line); crashes.push_back(line);
            if (first_fail.empty()) { first_fail = o.out + "/units.log"; }
        }
    }
    // summary recomputed from the raw log
    string log; read_all(o.out + "/units.log", log);
    struct Sum { uint64_t begin = 0, done = 0, pass = 0, fail = 0, crash = 0, cmp = 0, chk = 0, nu = 0, files = 0, bytes = 0, calls = 0, exp = 0, cnt = 0, neg = 0, nuhit = 0, feat[12] = {}; };
    std::map<string, Sum> S;
    size_t p = 0;
    while (p < log.size()) {
        size_t q = log.find('\n', p); if (q == string::npos) q = log.size();
        const string ln = log.substr(p, q - p); p = q + 1;
        char sc[16] = ""; unsigned long long seed, cmp, chk, nu, files, bytes; char v[16] = "";
        if (!strncmp(ln.c_str(), "BEGIN ", 6) && sscanf(ln.c_str(), "BEGIN %15s", sc) == 1) S[sc].begin++;
        else if (!strncmp(ln.c_str(), "UNIT ", 5) && sscanf(ln.c_str(), "UNIT %15s %llu %llu %llu %llu %llu %llu %15s", sc, &seed, &cmp, &chk, &nu, &files, &bytes, v) == 8) {
            Sum &m = S[sc]; m.done++; m.cmp += cmp; m.chk += chk; m.nu += nu; m.files += files; m.bytes += bytes; if (!strcmp(v, "PASS")) m.pass++; else m.fail++;
            unsigned long long a1, a2, a3, a4, a5;
            if (const char *x = strstr(ln.c_str(), " calls=")) if (sscanf(x, " calls=%llu exp=%llu cnt=%llu neg=%llu nuhit=%llu", &a1, &a2, &a3, &a4, &a5) == 5) { m.calls += a1; m.exp += a2; m.cnt += a3; m.neg += a4; m.nuhit += a5; }
            if (const char *x = strstr(ln.c_str(), " feat=")) { unsigned long long f[12] = {}; if (sscanf(x, " feat=%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu", &f[0], &f[1], &f[2], &f[3], &f[4], &f[5], &f[6], &f[7], &f[8], &f[9], &f[10], &f[11]) == 12) for (int i = 0; i < 12; i++) m.feat[i] += f[i]; }
        } else if (!strncmp(ln.c_str(), "CRASH ", 6) && sscanf(ln.c_str(), "CRASH %15s", sc) == 1) S[sc].crash++;
    }
    bool all_ok = true;
    printf("\n");
    for (const char *sc : { "R1", "R2", "R3", "R4" }) {
        if (!S.count(sc)) continue;
        const Sum &m = S[sc];
        const bool ok = m.begin == m.done && m.fail == 0 && m.crash == 0 && m.pass == m.done && m.done > 0;
        all_ok = all_ok && ok;
        printf("%s  seeds=%llu (begun %llu, completed %llu, fail %llu, crash %llu)  comparisons=%llu  contract-checks=%llu  next_use-calls=%llu (non-exhausted results %llu)  tables-loaded=%llu  input-bytes=%llu  calls=%llu experts=%llu counts=%llu negative-values=%llu  %s\n", sc,
               (unsigned long long)m.pass, (unsigned long long)m.begin, (unsigned long long)m.done, (unsigned long long)m.fail, (unsigned long long)m.crash, (unsigned long long)m.cmp,
               (unsigned long long)m.chk, (unsigned long long)m.nu, (unsigned long long)m.nuhit, (unsigned long long)m.files, (unsigned long long)m.bytes, (unsigned long long)m.calls,
               (unsigned long long)m.exp, (unsigned long long)m.cnt, (unsigned long long)m.neg, ok ? "PASS" : "FAIL");
        if (!strcmp(sc, "R1") || !strcmp(sc, "R2")) { printf("    generated files containing:"); for (int i = 0; i < 12; i++) printf(" %s=%llu", FEAT_NAMES[i], (unsigned long long)m.feat[i]); printf("\n"); }
    }
    if (!first_fail.empty()) {
        string t; read_all(first_fail, t);
        printf("\nFIRST FAILURE (%s):\n%s\n", first_fail.c_str(), t.size() > 6000 ? (t.substr(0, 6000) + "\n...(trimmed)").c_str() : t.c_str());
    }
    printf("\noverall: %s\n", all_ok && nfail == 0 && ncrash == 0 ? "PASS" : "FAIL");
    return all_ok && nfail == 0 && ncrash == 0 ? 0 : 1;
}
