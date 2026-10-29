// Differential driver for include/engcfg.h: loads a reference and a candidate library (dlopen, separate symbol tables) and compares them on random inputs.
//   diff_driver LIB_REFERENCE LIB_CANDIDATE [--only E1,E2,...] [--jobs N] [--scale F] [--out DIR] [--keep-going] [--flip N] [--seed0 N] [--e9-vm-mb N]
// E1 env_int  E2 settings + pend_cap  E3 card_of  E4 force_routes call scripts  E5 print_summary  E6 selectors / trace / prefill / messages  E7 composite configure()  E8 out-of-contract deaths
// E9 force_routes at sizes E4 cannot reach (driver v2). Since v2 the candidate's stderr is also captured write by write (check_writes), E8 compares the text of the deaths, E4 scripts end with six appended calls.
// Each (scenario, seed) is a unit run by one worker process; units are independent. Raw log lines (out/units.log):
//   "BEGIN scen seed"  and  "UNIT scen seed cmp chk cases PASS|FAIL|CRASH"   (fields 4..6 are numbers)
// cmp = equality checks reference-vs-candidate (one per compared scalar, string or byte block); chk = single-library checks; cases = random cases generated.
// Anything that can exit, die or register an exit handler runs in a forked child with stderr redirected to a file; the rest captures fd 2 in-process.
// Build: g++ -O2 -std=c++17 -Wall -Wextra -I../include -I../ref -o diff_driver diff_driver.cpp -ldl
#include "engcfg.h"
#include "ecref.h"
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <ctime>
#include <functional>
#include <map>
#include <string>
#include <vector>

using std::string;
using std::vector;

// the layouts the contract promises (LP64, worked out by hand from the field order)
static_assert(sizeof(ec_settings) == 88 && offsetof(ec_settings, host_bytes) == 40 && offsetof(ec_settings, vq_from) == 72, "ec_settings");
static_assert(sizeof(ec_call) == 40 && sizeof(ec_card_stats) == 64 && sizeof(ec_summary) == 784 && offsetof(ec_summary, t_calls) == 760, "ec_call / ec_summary");

// ------------------------------------------------------------------------------------------------ libraries
struct Api {
    void *h = nullptr; string path; bool is_ref = false;
    int (*env_int)(const char *, int) = nullptr;
    void (*settings)(ec_settings *, int) = nullptr;
    size_t (*pend_cap)(size_t) = nullptr;
    FILE *(*open_trace)(void) = nullptr;
    void (*print_rc)(const ec_settings *, size_t, size_t) = nullptr;
    void (*prefill)(size_t) = nullptr;
    void (*start_line)(const ec_settings *) = nullptr;
    int (*sel_mode)(int, int) = nullptr;
    const char *(*sel_lib)(int) = nullptr;
    void (*sel_ann)(int, const char *) = nullptr;
    int (*card_of)(int, int, int) = nullptr;
    void (*force)(size_t, size_t, const ec_call *, int, int, int *, int) = nullptr;
    void (*summary)(const ec_summary *) = nullptr;
    void (*ref_configure)(int, const int *) = nullptr;
    void (*ref_snapshot)(ecref_snap *) = nullptr;
};

static bool load_api(Api &a, const char *path, bool ref)
{
    a.path = path; a.is_ref = ref;
    a.h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!a.h) { fprintf(stderr, "dlopen %s: %s\n", path, dlerror()); return false; }
    bool ok = true;
#define SYM(field, name) do { void *s_ = dlsym(a.h, name); if (!s_) { fprintf(stderr, "missing symbol %s in %s\n", name, path); ok = false; } else memcpy((void *)&a.field, &s_, sizeof s_); } while (0)
    SYM(env_int, "ec_env_int"); SYM(settings, "ec_settings_from_env"); SYM(pend_cap, "ec_pend_cap"); SYM(open_trace, "ec_open_trace"); SYM(print_rc, "ec_print_route_cache");
    SYM(prefill, "ec_prefill_start"); SYM(start_line, "ec_print_start_line"); SYM(sel_mode, "ec_selector_mode"); SYM(sel_lib, "ec_selector_lib");
    SYM(sel_ann, "ec_selector_announce_rust"); SYM(card_of, "ec_card_of"); SYM(force, "ec_force_routes"); SYM(summary, "ec_print_summary");
    if (ref) { SYM(ref_configure, "ecref_configure"); SYM(ref_snapshot, "ecref_snapshot"); }
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
    int64_t range(int64_t lo, int64_t hi) { return lo + (int64_t)(next() % (uint64_t)(hi - lo + 1)); }
    double f() { return (double)(next() >> 11) * (1.0 / 9007199254740992.0); }
    bool p(double pr) { return f() < pr; }
};
static string esc(const string &s, size_t max = 300)
{
    string o;
    for (size_t i = 0; i < s.size() && o.size() < max; i++) {
        const unsigned char c = (unsigned char)s[i];
        if (c == '\n') o += "\\n"; else if (c == '\r') o += "\\r"; else if (c == '\t') o += "\\t"; else if (c == '\\') o += "\\\\";
        else if (c >= 32 && c < 127) o += (char)c; else o += sfmt("\\x%02x", c);
    }
    if (o.size() >= max) o += sfmt("...(%zu bytes)", s.size());
    return o;
}
static bool read_file(const string &path, string &out)
{
    out.clear();
    FILE *f = fopen(path.c_str(), "rb"); if (!f) return false;
    char buf[65536]; size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) out.append(buf, n);
    fclose(f); return true;
}
static bool write_file(const string &path, const string &data)
{
    FILE *f = fopen(path.c_str(), "wb"); if (!f) return false;
    const bool ok = data.empty() || fwrite(data.data(), 1, data.size(), f) == data.size();
    return fclose(f) == 0 && ok;
}

// ------------------------------------------------------------------------------------------------ comparison bookkeeping (one worker = one unit)
static uint64_t g_cmp = 0, g_chk = 0, g_cases = 0, g_flip = 0;
static bool g_failed = false, g_keep_going = false;
static int g_nfail = 0;
static string g_report, g_scen;
static uint64_t g_seed = 0;
static std::map<string, uint64_t> g_stats;   // branch counters of this unit, written as STAT lines: they show what the random cases actually reached
static void stat(const string &k, uint64_t n = 1)
{
    string key = k;
    for (char &c : key) if (c == ' ' || (unsigned char)c < 33 || (unsigned char)c >= 127) c = '_';
    g_stats[key] += n;
}
static const char MARKER[] = "[driver atexit handler ran]\n";
static string err_kind(const string &err_in)
{
    // the fixed part of the last stderr line (the driver's own atexit marker is dropped), so that the raw bytes of a variable value do not make a key of their own
    string err = err_in;
    if (err.size() >= sizeof MARKER - 1 && err.compare(err.size() - (sizeof MARKER - 1), sizeof MARKER - 1, MARKER) == 0) err.resize(err.size() - (sizeof MARKER - 1));
    const size_t e = err.size() > 1 ? err.rfind('\n', err.size() - 2) : string::npos;
    const string l = e == string::npos ? err : err.substr(e + 1);
    static const char *templates[] = { " must be cpp or rust, not ", ": this build has no copyq.h", ": this build has no slotcache.h", ": this build has no routetable.h", "cannot write ",
        "K3GPU_PREFILL needs K3GPU_ROUTECACHE", "K3GPU PREFILL ON" };
    for (const char *t : templates) {
        const size_t p = l.find(t);
        if (p == string::npos) continue;
        string head = l.compare(0, 6, "K3GPU_") == 0 ? l.substr(6, std::min<size_t>(p, 6 + 12) - 6) : "";
        head = head.substr(0, head.find_first_of("= "));
        return string(t) + (head.empty() ? "" : " [" + head + "]");
    }
    return err.empty() ? "(empty)" : "(other)";
}
static uint64_t count_of(const string &s, const char *needle) { uint64_t n = 0; for (size_t p = s.find(needle); p != string::npos; p = s.find(needle, p + 1)) n++; return n; }

static void fail(const string &what, const string &a, const string &b, const string &desc)
{
    g_failed = true;
    if (g_nfail++ >= 5) return;
    g_report += sfmt("MISMATCH %s unit %s seed %llu cmp #%llu (case %llu)\n  reference: %s\n  candidate: %s\n  case: %s\n", what.c_str(), g_scen.c_str(), (unsigned long long)g_seed,
                     (unsigned long long)g_cmp, (unsigned long long)g_cases, a.c_str(), b.c_str(), desc.c_str());
}
#define EQI(a, b, WHAT, DESC) do { g_cmp++; long long va_ = (long long)(a), vb_ = (long long)(b); if (g_flip && g_cmp == g_flip) vb_++; \
    if (va_ != vb_) fail(WHAT, std::to_string(va_), std::to_string(vb_), (DESC)); } while (0)
#define EQU(a, b, WHAT, DESC) do { g_cmp++; unsigned long long va_ = (unsigned long long)(a), vb_ = (unsigned long long)(b); if (g_flip && g_cmp == g_flip) vb_++; \
    if (va_ != vb_) fail(WHAT, std::to_string(va_), std::to_string(vb_), (DESC)); } while (0)
#define EQS(a, b, WHAT, DESC) do { g_cmp++; string va_ = (a), vb_ = (b); if (g_flip && g_cmp == g_flip) vb_ += "X"; \
    if (va_ != vb_) { size_t k_ = 0; while (k_ < va_.size() && k_ < vb_.size() && va_[k_] == vb_[k_]) k_++; \
        fail(WHAT, "[" + esc(va_) + "]", "[" + esc(vb_) + "] (first difference at byte " + std::to_string(k_) + ")", (DESC)); } } while (0)
#define STOP() (g_failed && !g_keep_going)

// ------------------------------------------------------------------------------------------------ environment
static const char *ALL_VARS[] = { "K3GPU_CARDS", "K3GPU_READERS", "K3GPU_VERBOSE", "K3GPU_RESERVE_MB", "K3GPU_POLICY", "K3GPU_CHUNKS", "K3GPU_HOST_GB", "K3GPU_ROUTECACHE", "K3GPU_LOOKAHEAD",
    "K3GPU_VQUEUE", "K3GPU_VQ_INFLIGHT", "K3GPU_VQ_FROM", "K3GPU_LAYOUT", "K3GPU_PREFILL", "K3GPU_TRACE", "K3GPU_COPYQ", "K3GPU_COPYQ_LIB", "K3GPU_SLOTCACHE", "K3GPU_SLOTCACHE_LIB",
    "K3GPU_ROUTETABLE", "K3GPU_ROUTETABLE_LIB", "ENGCFG_T" };
struct Val { bool set = false; string s; };
static void clear_env() { for (const char *n : ALL_VARS) unsetenv(n); }
static void put_env(const char *name, const Val &v)
{
    if (!v.set) { unsetenv(name); return; }
    if (setenv(name, v.s.c_str(), 1) != 0) { perror("setenv"); exit(2); }
}
static string env_dump()
{
    string o;
    for (const char *n : ALL_VARS) { const char *v = getenv(n); if (v) o += sfmt(" %s=[%s]", n, esc(v).c_str()); }
    return o.empty() ? " (none of the K3GPU variables set)" : o;
}

static const char *SPECIAL_NUMS[] = { "0", "1", "-1", "2", "3", "7", "8", "9", "16", "92", "93", "150", "256", "2500", "+5", "-0", "+0", "0x10", "0X1F", "0b11", "010", "1e3", "5.9", "-5.9", ".5", "5.",
    " 5", "\t5", "\n5", "\v5", "\f5", "\r5", " \t \n\v\f\r 5", "\xa0" "5", "\x85" "5", "\x1c" "5", "5 ", "5x", "x5", "--5", "+-5", "-+5", "- 5", "+ 5", "5-", "5+", "12 34", "1,000", "1_000",
    "2147483647", "2147483648", "-2147483648", "-2147483649", "4294967295", "4294967296", "4294967297", "9223372036854775807", "9223372036854775808", "-9223372036854775807",
    "-9223372036854775808", "-9223372036854775809", "18446744073709551615", "18446744073709551616", "18446744073709551617", "99999999999999999999999999",
    "-99999999999999999999999999", "00000000000000000000000000000000000000000000000000000000000000000000000012",
    "0000000000000000000000000000000000000000000000000000000000000000000000009223372036854775808", "x", "abc", "nan", "inf", "-", "+", " ", "0 ", "00", "000x" };
static string gen_numstr(Rng &r)
{
    string s;
    const char *ws = " \t\n\v\f\r\xa0\x85\x1f";
    for (uint32_t k = r.u(4) == 0 ? r.u(4) : 0; k > 0; k--) s += ws[r.u(9)];
    if (r.u(4) == 0) s += r.p(0.5) ? '+' : '-';
    const int nd = r.p(0.5) ? (int)r.range(1, 6) : (int)r.range(1, 26);
    for (int i = 0; i < nd; i++) s += (char)('0' + r.u(10));
    if (r.u(6) == 0) { const char *junk = "xX.e,_ \n0-+"; for (uint32_t k = r.range(1, 3); k > 0; k--) s += junk[r.u(11)]; }
    return s;
}
static string gen_bytes(Rng &r, int maxlen = 12)
{
    string s;
    for (int i = (int)r.range(1, maxlen); i > 0; i--) s += (char)r.range(1, 255);
    return s;
}
static Val gen_val(Rng &r)
{
    Val v;
    const uint32_t k = r.u(100);
    if (k < 12) return v;                                   // unset
    v.set = true;
    if (k < 20) return v;                                   // empty
    if (k < 50) v.s = SPECIAL_NUMS[r.u(sizeof SPECIAL_NUMS / sizeof *SPECIAL_NUMS)];
    else if (k < 95) v.s = gen_numstr(r);
    else v.s = gen_bytes(r);
    return v;
}
static int gen_int(Rng &r)
{
    switch (r.u(8)) {
    case 0: return 0; case 1: return 1; case 2: return -1; case 3: return INT_MAX; case 4: return INT_MIN;
    case 5: { static const int c[] = { 8, 16, 92, 150, 256, 2500, 93, 7 }; return c[r.u(8)]; }
    default: return (int)r.next();
    }
}
static uint64_t gen_u64(Rng &r)
{
    switch (r.u(10)) {
    case 0: return 0;
    case 1: return r.u(10);
    case 2: return r.u(1001);
    case 3: return r.next() >> 32;
    case 4: return (1ull << 32) + (uint64_t)r.range(-3, 3);
    case 5: return (1ull << 53) + (uint64_t)r.range(-5, 5);
    case 6: return (1ull << 63) + (uint64_t)r.range(-5, 5);
    case 7: return UINT64_MAX - r.u(5);
    case 8: return r.u(100000);
    default: return r.next();
    }
}
static double gen_double(Rng &r)
{
    double v;
    switch (r.u(16)) {
    case 0: v = 0.0; break;
    case 1: v = -0.0; break;
    case 2: v = (double)r.range(0, 640) / 16.0; break;                                  // exact binary fractions: ties for %.0f %.1f %.3f
    case 3: v = (double)r.range(0, 99999) / 1000.0 + 0.0005; break;                      // decimal near-ties at the 4th place
    case 4: v = (double)r.range(0, 2000) / 100.0 + 0.05; break;                         // decimal near-ties at the 2nd place
    case 5: v = (double)r.range(0, 2000) / 10.0 + 0.5; break;                           // k + 0.5
    case 6: v = r.f() * 1000.0; break;
    case 7: v = r.f() * 1e7; break;
    case 8: v = std::pow(10.0, (double)r.range(15, 308)); break;
    case 9: v = r.p(0.5) ? DBL_MAX : std::pow(10.0, (double)r.range(15, 30)) * (double)r.range(1, 9); break;
    case 10: v = std::pow(10.0, -(double)r.range(1, 320)); break;
    case 11: { static const double c[] = { DBL_MIN, 5e-324, DBL_MIN / 2, DBL_EPSILON, 0.1, 0.2, 0.3, 1.0, 1.5, 2.5, 3.5, 0.05, 0.15, 0.25, 0.35, 0.45 }; v = c[r.u(16)]; break; }
    case 12: { const uint64_t m = r.next() & 0x000FFFFFFFFFFFFFull; const uint64_t b = 0x7FF0000000000000ull | (r.p(0.3) ? 0 : m); memcpy(&v, &b, 8); break; }  // inf / nan, any payload
    case 13: { const uint64_t b = r.next(); memcpy(&v, &b, 8); break; }                // any bit pattern
    default: v = (double)(int64_t)r.range(0, 100000) / (double)r.range(1, 1000); break;
    }
    if (r.u(12) == 0) v = -v;
    return v;
}
static string hexd(double v) { return sfmt("%a (%.17g)", v, v); }

// ------------------------------------------------------------------------------------------------ worker state: scratch dir, stderr capture, forked children
static string g_dir;   // this unit's scratch directory (also the cwd)
static int g_capfd = -1, g_saved2 = -1;

static string cap_read()
{
    string out; char buf[65536];
    off_t off = 0; ssize_t n;
    while ((n = pread(g_capfd, buf, sizeof buf, off)) > 0) { out.append(buf, (size_t)n); off += n; }
    return out;
}
static string capture(const std::function<void()> &fn)
{
    if (ftruncate(g_capfd, 0) != 0 || lseek(g_capfd, 0, SEEK_SET) < 0) { perror("cap"); exit(2); }
    dup2(g_capfd, 2);
    fn();
    dup2(g_saved2, 2);
    return cap_read();
}
static void marker_atexit() { if (write(2, MARKER, sizeof MARKER - 1) < 0) { /* nothing to do */ } }   // registered first in every child: runs last, and only if the library left through exit(), not _exit()
// writes = one entry per write(2) the child made to fd 2; filled only when the child ran with ChildOpt::dgram (see below)
struct Child { int code = -1, sig = 0; string err, res; vector<string> writes; };
// dgram: fd 2 of the child is one end of an AF_UNIX SOCK_SEQPACKET socketpair (a message-preserving "datagram" pair: one write = one message, and unlike SOCK_DGRAM it reports EOF when the child ends),
//        so that every write(2) can be counted and its bytes seen; err is then the concatenation of the writes. Used for the candidate only: glibc's unbuffered fprintf (the reference) splits lines over 128 bytes.
// vm_mb: RLIMIT_AS of the child in MB (0 = none). alarm_s: the child is killed by SIGALRM after that many seconds. tag: names the child's scratch files (two children can run at once).
struct ChildOpt { bool dgram = false; unsigned alarm_s = 60; size_t vm_mb = 0; const char *tag = "child"; };
struct ChildH { pid_t pid = -1; string ep, rp; int sock = -1; };
static void drain_writes(int sock, vector<string> &out)
{
    static char buf[1 << 20];
    for (;;) {
        const ssize_t n = recv(sock, buf, sizeof buf, 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) break;   // 0 = every holder of the child's end is gone: the child has ended
        out.emplace_back(buf, (size_t)n);
    }
}
static ChildH child_start(const std::function<int(FILE *)> &fn, const ChildOpt &o = ChildOpt())
{
    ChildH h;
    h.ep = g_dir + "/" + o.tag + ".err"; h.rp = g_dir + "/" + o.tag + ".res";
    int sv[2] = { -1, -1 };
    if (o.dgram && socketpair(AF_UNIX, SOCK_SEQPACKET, 0, sv) != 0) { perror("socketpair"); exit(2); }
    fflush(NULL);
    const pid_t pid = fork();
    if (pid < 0) { perror("fork"); exit(2); }
    if (pid == 0) {
        alarm(o.alarm_s);
        if (o.vm_mb) { const rlimit rl = { (rlim_t)o.vm_mb << 20, (rlim_t)o.vm_mb << 20 }; if (setrlimit(RLIMIT_AS, &rl) != 0) _exit(122); }
        if (o.dgram) { close(sv[1]); if (dup2(sv[0], 2) < 0) _exit(120); close(sv[0]); }
        else {
            const int efd = open(h.ep.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (efd < 0 || dup2(efd, 2) < 0) _exit(120);
            close(efd);
        }
        FILE *res = fopen(h.rp.c_str(), "wb");
        if (!res) _exit(121);
        atexit(marker_atexit);
        const int rc = fn(res);
        fclose(res);
        exit(rc);   // exit, not _exit: atexit handlers (the configure() summary) are part of what is compared
    }
    if (o.dgram) { close(sv[0]); h.sock = sv[1]; }
    h.pid = pid;
    return h;
}
static Child child_finish(ChildH &h)
{
    Child c;
    if (h.sock >= 0) { drain_writes(h.sock, c.writes); close(h.sock); h.sock = -1; for (const string &w : c.writes) c.err += w; }
    int st = 0;
    waitpid(h.pid, &st, 0);
    if (WIFEXITED(st)) c.code = WEXITSTATUS(st); else if (WIFSIGNALED(st)) c.sig = WTERMSIG(st);
    if (c.writes.empty()) read_file(h.ep, c.err);   // a child that never wrote to a socket (or ran without dgram) has its stderr in the file, if any
    read_file(h.rp, c.res);
    return c;
}
static Child run_child(const std::function<int(FILE *)> &fn, const ChildOpt &o = ChildOpt())
{
    ChildH h = child_start(fn, o);
    return child_finish(h);
}
// In-process library calls (E5, E6 messages): run `fn` in a forked child whose fd 2 is the socketpair and return its writes.
static vector<string> capture_writes(const std::function<void()> &fn)
{
    int sv[2];
    if (socketpair(AF_UNIX, SOCK_SEQPACKET, 0, sv) != 0) { perror("socketpair"); exit(2); }
    fflush(NULL);
    const pid_t pid = fork();
    if (pid < 0) { perror("fork"); exit(2); }
    if (pid == 0) { alarm(60); close(sv[1]); if (dup2(sv[0], 2) < 0) _exit(120); close(sv[0]); fn(); _exit(0); }
    close(sv[0]);
    vector<string> w;
    drain_writes(sv[1], w);
    close(sv[1]);
    int st = 0; waitpid(pid, &st, 0);
    return w;
}
static string join_writes(const vector<string> &w) { string s; for (const string &x : w) s += x; return s; }
// "one write per original line" (include/engcfg.h): the candidate's every write(2) must end a line; expect >= 0 also fixes the number of writes (the callers that know how many lines were printed).
// The reference is not held to this: glibc's unbuffered fprintf writes a line of more than 128 bytes in pieces (counted separately, see the writes.* statistics).
static void check_writes(const string &what, const vector<string> &w, long expect, const string &desc)
{
    g_cmp++;
    string shown;
    for (size_t i = 0; i < w.size() && i < 8; i++) shown += "[" + esc(w[i], 100) + "]";
    if (g_flip && g_cmp == g_flip) {   // plumbing check (--flip): pretend there was one write more than there was
        fail(what + ": number of writes (--flip perturbation)", sfmt("expected: %zu writes", w.size()), sfmt("%zu writes: %s", w.size() + 1, shown.c_str()), desc);
        return;
    }
    for (size_t i = 0; i < w.size(); i++)
        if (w[i].empty() || w[i].back() != '\n') {
            fail(what + ": write #" + std::to_string(i) + " does not end a line (a line must go out in one write)", sfmt("expected: %zu writes, each ending with a newline", w.size()), sfmt("%zu writes: %s", w.size(), shown.c_str()), desc);
            return;
        }
    if (expect >= 0 && (long)w.size() != expect)
        fail(what + ": number of writes", sfmt("expected: %ld writes (one per line)", expect), sfmt("%zu writes: %s", w.size(), shown.c_str()), desc);
}
static string child_str(const Child &c) { return sfmt("exit=%d sig=%d stderr=[%s] res=[%s]", c.code, c.sig, esc(c.err).c_str(), esc(c.res).c_str()); }
static void cmp_child(const char *what, const Child &a, const Child &b, const string &desc)
{
    EQI(a.code, b.code, string(what) + ": exit status", desc + "\n  reference " + child_str(a) + "\n  candidate " + child_str(b));
    EQI(a.sig, b.sig, string(what) + ": terminating signal", desc + "\n  reference " + child_str(a) + "\n  candidate " + child_str(b));
    EQS(a.err, b.err, string(what) + ": stderr", desc + "\n  reference " + child_str(a) + "\n  candidate " + child_str(b));
    EQS(a.res, b.res, string(what) + ": result text", desc + "\n  reference " + child_str(a) + "\n  candidate " + child_str(b));
}

// ------------------------------------------------------------------------------------------------ E1 env_int
static void sc_env_int(Api &R, Api &C, Rng &r, int cases)
{
    static const char *names[] = { "ENGCFG_T", "K3GPU_CARDS", "K3GPU_HOST_GB", "", "NO_SUCH_VAR_ENGCFG" };
    for (int i = 0; i < cases && !STOP(); i++) {
        g_cases++;
        const char *name = names[r.u(5)];
        clear_env();
        const Val v = gen_val(r);
        if (name[0] && strcmp(name, "NO_SUCH_VAR_ENGCFG")) put_env(name, v);
        const int dflt = gen_int(r);
        const int a = R.env_int(name, dflt), b = C.env_int(name, dflt);
        stat(!name[0] || !strcmp(name, "NO_SUCH_VAR_ENGCFG") || !v.set || v.s.empty() ? "env_int.default_taken" : "env_int.parsed");
        if (v.set && !v.s.empty() && name[0] && strcmp(name, "NO_SUCH_VAR_ENGCFG")) stat(a < 0 ? "env_int.parsed_negative" : a == 0 ? "env_int.parsed_zero" : "env_int.parsed_positive");
        EQI(a, b, "env_int", sfmt("name [%s] value %s[%s] dflt %d", name, v.set ? "" : "(unset) ", esc(v.s).c_str(), dflt));
        // the same function with the default flipped: unset/empty must follow it, anything else must not
        const int a2 = R.env_int(name, ~dflt), b2 = C.env_int(name, ~dflt);
        EQI(a2, b2, "env_int (other default)", sfmt("name [%s] value [%s] dflt %d", name, esc(v.s).c_str(), ~dflt));
    }
    clear_env();
}

// ------------------------------------------------------------------------------------------------ E2 settings
static string show_settings(const ec_settings &s)
{
    return sfmt("cards %d readers %d verbose %d reserve %zu policy %d chunks %d host_gb %d host_bytes %zu routecache %s[%s] lookahead %d vqueue %d vq_cap %zu vq_from %ld layout %d prefill %d",
                s.cards, s.readers, s.verbose, s.reserve, s.policy, s.chunks, s.host_gb, s.host_bytes, s.routecache ? "" : "NULL", s.routecache ? esc(s.routecache).c_str() : "", s.lookahead, s.vqueue,
                s.vq_cap, s.vq_from, s.layout, s.prefill);
}
static void cmp_settings(const ec_settings &a, const ec_settings &b, const string &desc)
{
    EQI(a.cards, b.cards, "settings.cards", desc); EQI(a.readers, b.readers, "settings.readers", desc); EQI(a.verbose, b.verbose, "settings.verbose", desc);
    EQU(a.reserve, b.reserve, "settings.reserve", desc); EQI(a.policy, b.policy, "settings.policy", desc); EQI(a.chunks, b.chunks, "settings.chunks", desc);
    EQI(a.host_gb, b.host_gb, "settings.host_gb", desc); EQU(a.host_bytes, b.host_bytes, "settings.host_bytes", desc);
    EQI(a.routecache == nullptr, b.routecache == nullptr, "settings.routecache NULL-ness", desc);
    if (a.routecache && b.routecache) EQS(string(a.routecache), string(b.routecache), "settings.routecache string", desc);
    EQI(a.lookahead, b.lookahead, "settings.lookahead", desc); EQI(a.vqueue, b.vqueue, "settings.vqueue", desc); EQU(a.vq_cap, b.vq_cap, "settings.vq_cap", desc);
    EQI(a.vq_from, b.vq_from, "settings.vq_from", desc); EQI(a.layout, b.layout, "settings.layout", desc); EQI(a.prefill, b.prefill, "settings.prefill", desc);
}
static string gen_path(Rng &r)
{
    static const char *c[] = { "/tmp/r.trace", "/home/u/p1d.trace", "relative.trace", "a", "ab", "abc", "/x/y", "-", " ", "/tmp/with space/r", "~/x" };
    switch (r.u(4)) { case 0: return gen_bytes(r, 40); case 1: { string s = "/"; for (int i = (int)r.range(0, 30); i > 0; i--) s += (char)r.range(1, 126); return s; } default: return c[r.u(11)]; }
}
static void set_random_settings_env(Rng &r)
{
    clear_env();
    static const char *common[][8] = {
        { "K3GPU_CARDS", "1", "2", "4", "8", "16", "3", "0" }, { "K3GPU_POLICY", "0", "1", "2", "3", "2", "2", "-1" }, { "K3GPU_VQUEUE", "0", "1", "1", "2", "-1", "1", "x" },
        { "K3GPU_PREFILL", "0", "1", "1", "2", "-1", "x", "0" }, { "K3GPU_LAYOUT", "0", "1", "1", "2", "-1", "x", "1" }, { "K3GPU_HOST_GB", "150", "1", "2", "0", "-1", "50", "2147483647" },
        { "K3GPU_RESERVE_MB", "2500", "0", "1", "-1", "2147483647", "2147483648", "4096" }, { "K3GPU_READERS", "16", "1", "32", "0", "-1", "8", "64" },
        { "K3GPU_LOOKAHEAD", "92", "0", "1", "10", "-1", "184", "x" }, { "K3GPU_VQ_INFLIGHT", "256", "0", "1", "-1", "2147483648", "64", "x" }, { "K3GPU_VQ_FROM", "0", "92", "1", "-1", "2147483648", "100", "x" },
        { "K3GPU_VERBOSE", "0", "1", "2", "-1", "x", "1", "0" }, { "K3GPU_CHUNKS", "0", "1", "4", "8", "-1", "x", "16" } };
    for (auto &row : common) {
        const uint32_t k = r.u(10);
        if (k < 2) continue;
        Val v; v.set = true;
        if (k < 7) v.s = row[1 + r.u(7)]; else v = gen_val(r);
        put_env(row[0], v);
    }
    if (r.p(0.65)) { Val v; v.set = true; v.s = r.p(0.05) ? "" : gen_path(r); put_env("K3GPU_ROUTECACHE", v); }
    static const char *decoys[] = { "K3GPU_TRACE", "K3GPU_COPYQ", "K3GPU_SLOTCACHE", "K3GPU_ROUTETABLE", "K3GPU_COPYQ_LIB", "ENGCFG_T" };
    for (const char *d : decoys) if (r.p(0.3)) put_env(d, gen_val(r));
}
static void sc_settings(Api &R, Api &C, Rng &r, int cases)
{
    static const int devs[] = { 0, 1, 2, 4, 7, 8, 9, 16, -1, -5, INT_MAX, INT_MIN, 3 };
    for (int i = 0; i < cases && !STOP(); i++) {
        g_cases++;
        set_random_settings_env(r);
        const int dev = r.p(0.6) ? devs[r.u(13)] : (int)r.range(-2, 20);
        ec_settings a, b;
        memset(&a, 0xA5, sizeof a); memset(&b, 0x5A, sizeof b);
        R.settings(&a, dev); C.settings(&b, dev);
        const string desc = sfmt("devices %d env:%s\n  reference %s\n  candidate %s", dev, env_dump().c_str(), show_settings(a).c_str(), show_settings(b).c_str());
        cmp_settings(a, b, desc);
        stat("settings.cases"); if (a.routecache) stat("settings.routecache_on"); if (a.vqueue) stat("settings.vqueue_on"); if (a.prefill) stat("settings.prefill_on"); if (a.cards <= 0) stat("settings.cards_le0");
        if (a.policy == 2) { stat("settings.policy2"); if (a.routecache && !a.vqueue) stat("settings.rc_policy2_vqueue_off"); }
        if (a.reserve >= (1ull << 40)) stat("settings.reserve_huge");
        if (a.vq_cap == 256 && !a.routecache) stat("settings.vq_cap_initial");
        g_chk++;
        const char *rc = getenv("K3GPU_ROUTECACHE");
        const bool on = rc && *rc;
        if ((b.routecache != nullptr) != on || (on && b.routecache != rc) || (a.routecache != nullptr) != on || (on && a.routecache != rc)) {
            fail("settings.routecache pointer is not getenv's", show_settings(a), show_settings(b), desc);
        }
        // pend_cap on a spread of slot counts
        for (int k = 0; k < 3; k++) {
            const size_t cap = (size_t)gen_u64(r);
            EQU(R.pend_cap(cap), C.pend_cap(cap), "pend_cap", sfmt("cap_slots %zu", cap));
        }
    }
    clear_env();
}

// ------------------------------------------------------------------------------------------------ E3 card_of
static bool card_trap(int layout, int cards, int L) { return layout == 1 && L != 0 && (cards == 0 || (cards == -1 && L - 1 == INT_MIN) ); }
static void card_cmp(Api &R, Api &C, int layout, int cards, int L)
{
    g_cases++;
    const int a = R.card_of(layout, cards, L), b = C.card_of(layout, cards, L);
    EQI(a, b, "card_of", sfmt("layout %d cards %d layer %d", layout, cards, L));
}
static void sc_card_of(Api &R, Api &C, Rng &r, int cases, bool exhaustive)
{
    if (exhaustive) {
        static const int layouts[] = { 0, 1, 2, -1 };
        for (int layout : layouts) for (int cards = -3; cards <= 12 && !STOP(); cards++) for (int L = -300; L <= 400; L++) {
            if (card_trap(layout, cards, L)) continue;
            card_cmp(R, C, layout, cards, L);
        }
        return;
    }
    static const int layouts[] = { 0, 1, 2, -1, 100, INT_MAX, INT_MIN, 1, 0, 1 };
    static const int ext[] = { INT_MAX, INT_MIN, INT_MIN + 1, INT_MAX - 1, 1 << 30, -(1 << 30), 65536, -65536, 93, 92, 0, 1, -1 };
    for (int i = 0; i < cases && !STOP(); i++) {
        const int layout = layouts[r.u(10)];
        const int cards = r.p(0.5) ? (int)r.range(1, 8) : r.p(0.5) ? (int)r.range(-5, 20) : r.p(0.5) ? ext[r.u(13)] : (int)r.next();
        const int L = r.p(0.5) ? (int)r.range(0, 92) : r.p(0.3) ? (int)r.range(-200, 300) : r.p(0.5) ? ext[r.u(13)] : (int)r.next();
        if (card_trap(layout, cards, L)) continue;
        card_cmp(R, C, layout, cards, L);
    }
}

// ------------------------------------------------------------------------------------------------ E4 force_routes call scripts
struct FCall { size_t pos = 0, size = 0; bool has_cur = false; int layer = 0; vector<int> experts, counts; int L = 0, n = 0, top = 16; int extra = -1; };   // extra: kind of an appended call (see gen_extra_fcalls), -1 for the first 60
enum { SLACK = 512 };

static FCall gen_fcall(Rng &r, int idx)
{
    FCall c;
    static const int tops[] = { 16, 16, 16, 16, 16, 16, 1, 2, 3, 4, 8, 17 };
    static const int ns[] = { 1, 2, 3, 5, 7, 16, 17, 40, 0, 4, 6, 33 };
    c.top = tops[r.u(12)];
    c.n = r.p(0.8) ? (r.p(0.6) ? (int)r.range(1, 24) : ns[r.u(12)]) : (r.p(0.05) ? 200 : (int)r.range(1, 64));
    c.size = r.p(0.1) ? 0 : (size_t)r.range(1, 200);
    switch (r.u(8)) {
    case 0: c.pos = c.size; break; case 1: c.pos = c.size + 1; break; case 2: c.pos = c.size ? c.size - 1 : 0; break; case 3: c.pos = 0; break;
    case 4: c.pos = (size_t)-1; break; default: c.pos = c.size ? r.u((uint32_t)c.size) : 0; break;
    }
    c.layer = r.p(0.7) ? (int)r.range(1, 92) : r.p(0.5) ? 0 : (int)r.next();
    c.L = r.p(0.82) ? c.layer : r.p(0.5) ? (int)r.range(0, 93) : (int)r.next();
    const bool in = c.pos < c.size;
    c.has_cur = in || r.p(0.3);
    // valid counts: each <= n, sum = n * top
    const int n = c.n, top = c.top;
    const int mode = (int)r.u(20);
    if (n >= 1) {
        const int want = n * top;
        int m = (int)r.range(top, std::min(896, std::max(top, top * 3)));
        vector<int> counts(m, 0);
        for (int s = 0; s < want; s++) { int k; do { k = (int)r.u(m); } while (counts[k] >= n); counts[k]++; }
        vector<int> experts(m);
        const bool plain = r.p(0.7);
        for (int i = 0; i < m; i++) experts[i] = plain ? (int)r.range(0, 895) : (int)r.next();
        c.experts = experts; c.counts = counts;
        if (mode == 0) { c.counts.pop_back(); }                                                     // fewer counts than experts
        else if (mode == 1) { c.counts.clear(); }                                                   // no counts at all
        else if (mode == 2) { c.counts[r.u((uint32_t)m)] += 1; }                                    // sum one too high (may also exceed n)
        else if (mode == 3) { const int k = (int)r.u((uint32_t)m); c.counts[k] = n + 1; }           // a count larger than n
        else if (mode == 4) { c.counts[r.u((uint32_t)m)] -= 1; }                                    // sum one too low
        else if (mode == 5) { c.experts.clear(); c.counts.clear(); }                                // empty call
        else if (mode == 6 || mode == 7) {                                                          // sum right, one negative count compensating a raise: writes run past the end of sel
            int D = 0, tries = 0;
            for (int k = 0; k < m && tries < 40 && D < 60; k++, tries++) { const int j = (int)r.u((uint32_t)m); if (c.counts[j] < n) { D += n - c.counts[j]; c.counts[j] = n; } }
            if (D > 0) c.counts[r.u((uint32_t)m)] -= D;
        }
        else if (mode == 8) { c.counts[r.u((uint32_t)m)] = -(int)r.range(0, 5); }                 // negative count, sum usually off
        else if (mode == 9) { std::swap(c.counts[0], c.counts[m - 1]); }                            // still valid, different order
        else if (mode == 10) { c.experts.push_back(c.experts[0]); c.counts.push_back(0); }          // a duplicate expert with count 0
        else if (mode == 11 && n < 1000) { c.n = n + (int)r.range(-1, 1); }                         // n off by one against the counts
    } else if (n == 0) {
        const int m = (int)r.range(0, 4);
        for (int i = 0; i < m; i++) { c.experts.push_back((int)r.range(0, 895)); c.counts.push_back(r.p(0.7) ? 0 : (int)r.range(-2, 2)); }
    }
    (void)idx;
    return c;
}
// Six calls appended to every script, drawn from a generator of their own so that the first 60 calls of a script are the same as before. Each reaches step 2 or 3 of the contract with a call that would be
// valid but for one thing: kind 0 one extra count of 0 (more counts than experts), kind 1 two extra counts +1 and -1 (sum and every count still fine), kind 2 no experts and one count,
// kind 3 a count above n while the sum is still n*top, kind 4 two extra experts with count INT_MIN (the true sum is n*top - 2^32, which a 32-bit sum reads as n*top),
// kind 5 n = 1, top = 1 and 257-400 experts of count 1 plus one compensating negative count (one token gets every deal: a fill counter of 8 bits wraps).
// Kinds 0-4 must leave sel alone in the reference and kind 5 must deal; the driver checks that (an instrument that cannot show the thing proves nothing).
static void gen_extra_fcalls(Rng &x, vector<FCall> &out)
{
    static const int tops[] = { 16, 1, 2, 3, 4, 8, 17 };
    for (int kind = 0; kind < 6; kind++) {
        FCall c; c.extra = kind; c.pos = 0; c.size = 1; c.has_cur = true; c.layer = c.L = (int)x.range(1, 92);
        if (kind == 5) {
            c.n = 1; c.top = 1;
            const int m = (int)x.range(257, 400);
            for (int i = 0; i < m; i++) { c.experts.push_back((int)x.range(0, 895)); c.counts.push_back(1); }
            c.experts.push_back((int)x.range(0, 895)); c.counts.push_back(-(m - 1));
        } else {
            c.top = tops[x.u(7)]; c.n = (int)x.range(1, 40);
            const int want = c.n * c.top, m = (int)x.range(std::max(c.top, 2), std::max(c.top, 2) * 3);
            c.counts.assign(m, 0);
            for (int s = 0; s < want; s++) { int k; do { k = (int)x.u((uint32_t)m); } while (c.counts[k] >= c.n); c.counts[k]++; }
            for (int i = 0; i < m; i++) c.experts.push_back((int)x.range(0, 895));
            if (kind == 0) c.counts.push_back(0);
            else if (kind == 1) { c.counts.push_back(1); c.counts.push_back(-1); }
            else if (kind == 2) { c.experts.clear(); c.counts.assign(1, 0); }
            else if (kind == 3) {
                const int k = (int)x.u((uint32_t)m), j = (k + 1 + (int)x.u((uint32_t)m - 1)) % m;   // j != k
                const int up = c.n + 1 - c.counts[k];
                c.counts[k] = c.n + 1; c.counts[j] -= up;
            } else { for (int i = 0; i < 2; i++) { c.experts.push_back((int)x.range(0, 895)); c.counts.push_back(INT_MIN); } }
        }
        out.push_back(c);
    }
}
static string show_fcall(const FCall &c)
{
    string e, k;
    for (size_t i = 0; i < c.experts.size() && i < 24; i++) e += sfmt("%d,", c.experts[i]);
    for (size_t i = 0; i < c.counts.size() && i < 24; i++) k += sfmt("%d,", c.counts[i]);
    return sfmt("pos %zu size %zu has_cur %d cur.layer %d L %d n %d top %d experts(%zu)[%s%s] counts(%zu)[%s%s]", c.pos, c.size, c.has_cur, c.layer, c.L, c.n, c.top, c.experts.size(), e.c_str(),
                c.experts.size() > 24 ? "..." : "", c.counts.size(), k.c_str(), c.counts.size() > 24 ? "..." : "");
}
static size_t fcall_words(const FCall &c) { return (size_t)std::max(c.n, 0) * (size_t)c.top + SLACK; }
static int run_force_script(const Api &A, const vector<FCall> &sc, FILE *res)
{
    for (size_t i = 0; i < sc.size(); i++) {
        const FCall &c = sc[i];
        ec_call cur;
        cur.layer = c.layer;
        cur.experts = c.experts.empty() ? nullptr : c.experts.data(); cur.n_experts = c.experts.size();
        cur.counts = c.counts.empty() ? nullptr : c.counts.data(); cur.n_counts = c.counts.size();
        vector<int> sel(fcall_words(c), (int)(0x5A5A0000u + i));
        A.force(c.pos, c.size, c.has_cur ? &cur : nullptr, c.L, c.n, sel.data(), c.top);
        fwrite(sel.data(), 4, sel.size(), res);
    }
    return 0;
}
static void sc_force(Api &R, Api &C, Rng &r, int calls)
{
    vector<FCall> sc;
    for (int i = 0; i < calls; i++) sc.push_back(gen_fcall(r, i));
    { Rng x(mix64(g_seed * 7919ull + 0xE4E4E4E4ull)); gen_extra_fcalls(x, sc); }
    g_cases += sc.size();
    const Child a = run_child([&](FILE *res) { return run_force_script(R, sc, res); });
    ChildOpt cand_opt; cand_opt.dgram = true;
    const Child b = run_child([&](FILE *res) { return run_force_script(C, sc, res); }, cand_opt);
    EQI(a.code, b.code, "force_routes script: exit status", sfmt("script of %zu calls", sc.size()));
    EQI(a.sig, b.sig, "force_routes script: signal", sfmt("script of %zu calls", sc.size()));
    EQS(a.err, b.err, "force_routes script: stderr (the once-only warning)", sfmt("script of %zu calls; first call: %s", sc.size(), sc.empty() ? "" : show_fcall(sc[0]).c_str()));
    check_writes("force_routes script (candidate)", b.writes, -1, sfmt("script of %zu calls", sc.size()));
    g_chk++;
    if (a.err.size() > sizeof MARKER - 1) stat("force.scripts_with_warning");
    if (a.code != 0 || a.res.size() == 0) { fail("force_routes script: the reference child did not finish", child_str(a), child_str(b), "script of " + std::to_string(sc.size()) + " calls"); return; }
    // block by block, so a report names the call
    size_t off = 0;
    for (size_t i = 0; i < sc.size(); i++) {
        const size_t bytes = fcall_words(sc[i]) * 4;
        if (off + bytes > a.res.size() || off + bytes > b.res.size()) { fail("force_routes: short result", std::to_string(a.res.size()), std::to_string(b.res.size()), "call #" + std::to_string(i)); return; }
        g_cmp++;
        bool same = memcmp(a.res.data() + off, b.res.data() + off, bytes) == 0;
        {   // what this call did in the reference: wrote inside n*top, wrote into the slack words after it, or nothing
            const int *pa = (const int *)(a.res.data() + off); const size_t body = (size_t)std::max(sc[i].n, 0) * (size_t)sc[i].top, words = bytes / 4; const int pat = (int)(0x5A5A0000u + i);
            bool in_body = false, in_slack = false;
            for (size_t w = 0; w < words; w++) if (pa[w] != pat) { if (w < body) in_body = true; else in_slack = true; }
            stat("force.calls"); if (in_body) stat("force.calls_that_dealt"); if (in_slack) stat("force.calls_that_wrote_past_n_top"); if (!in_body && !in_slack) stat("force.calls_that_left_sel_alone");
            if (sc[i].extra >= 0) {
                const bool wrote = in_body || in_slack;
                stat(sfmt("force.extra_kind%d.%s", sc[i].extra, wrote ? "reference_dealt" : "reference_left_sel_alone"));
                if (wrote != (sc[i].extra == 5)) fail("force_routes: an appended call did not behave as designed in the reference (driver check)", sc[i].extra == 5 ? "deals" : "leaves sel alone", wrote ? "dealt" : "left sel alone", "call #" + std::to_string(i) + ": " + show_fcall(sc[i]));
            }
        }
        if (g_flip && g_cmp == g_flip) same = false;
        if (!same) {
            size_t w = 0; const int *pa = (const int *)(a.res.data() + off), *pb = (const int *)(b.res.data() + off);
            while (w < bytes / 4 && pa[w] == pb[w]) w++;
            fail("force_routes: selected array differs (incl. the slack words after n*top)", sfmt("word %zu = %d", w, w < bytes / 4 ? pa[w] : 0), sfmt("word %zu = %d", w, w < bytes / 4 ? pb[w] : 0),
                 "call #" + std::to_string(i) + ": " + show_fcall(sc[i]));
            return;
        }
        off += bytes;
    }
    EQU(a.res.size(), b.res.size(), "force_routes: total result size", "script");
}

// ------------------------------------------------------------------------------------------------ E5 print_summary
static void gen_summary(Rng &r, ec_summary &s)
{
    memset(&s, 0, sizeof s);
    s.cards = r.p(0.85) ? (int)r.range(0, 8) : (int)r.range(-3, 8);
    s.expert_raw = r.p(0.8) ? 17547264 : gen_int(r);
    const bool want_need = r.p(0.8);
    for (int c = 0; c < EC_MAX_CARDS; c++) {
        ec_card_stats &k = s.card[c];
        k.ready = r.p(0.7) ? (r.p(0.9) ? 1 : (int)r.next() | 1) : 0;
        k.slots = r.p(0.7) ? (int)r.range(0, 4000) : gen_int(r);
        const bool big = r.p(0.25);
        k.needed = want_need && !big ? (uint64_t)r.range(1, 100000) : gen_u64(r);
        k.hit_card = big ? gen_u64(r) : r.u(100000); k.hit_host = big ? gen_u64(r) : r.u(100000); k.from_drive = big ? gen_u64(r) : r.u(100000); k.written_back = gen_u64(r);
        k.t_read = gen_double(r); k.t_copy = gen_double(r);
    }
    if (r.p(0.05)) {   // the sum of `needed` over the ready cards wraps modulo 2^64 to 0 (the original then prints three lines only) or to a small number
        s.cards = (int)r.range(2, 8);
        const uint64_t X = r.next() | 1;
        for (int c = 0; c < EC_MAX_CARDS; c++) s.card[c].needed = 0;
        s.card[0].ready = 1; s.card[1].ready = 1;
        s.card[0].needed = X; s.card[1].needed = r.p(0.6) ? (uint64_t)0 - X : (uint64_t)0 - X + r.u(5);
    }
    s.copyq_rust = r.p(0.5) ? 1 : (int)r.next(); if (r.p(0.4)) s.copyq_rust = 0;
    s.slotcache_rust = r.p(0.5); s.routetable_rust = r.p(0.5);
    for (int i = 0; i < 3; i++) { s.cq_jobs[i] = gen_u64(r); s.sc_calls[i] = gen_u64(r); }
    s.rc_calls = r.p(0.4) ? 0 : (size_t)gen_u64(r); s.rt_nu_calls = gen_u64(r); s.rc_pos = (size_t)gen_u64(r); s.rc_ok = r.p(0.5);
    s.pf_planned = gen_u64(r); s.pf_skipped = gen_u64(r); s.pf_used = gen_u64(r); s.pf_waited = gen_u64(r); s.t_pwait = gen_double(r); s.wb_skipped = gen_u64(r);
    s.vq = r.p(0.5) ? (r.p(0.8) ? 1 : (int)r.next()) : 0;
    s.vq_ram = gen_u64(r); s.vq_drive = gen_u64(r); s.vq_wb = gen_u64(r); s.vq_nowb = gen_u64(r); s.vq_waited = gen_u64(r); s.t_vwait = gen_double(r); s.t_plan = gen_double(r);
    for (int i = 0; i < 3; i++) {
        s.n_calls[i] = r.p(0.3) ? 0.0 : r.p(0.5) ? (double)r.range(1, 100000) : gen_double(r);
        s.t_calls[i] = gen_double(r);
    }
}
static string show_summary(const ec_summary &s)
{
    string o = sfmt("cards %d expert_raw %d copyq_rust %d slotcache_rust %d routetable_rust %d rc_calls %zu rc_pos %zu rc_ok %d vq %d\n", s.cards, s.expert_raw, s.copyq_rust, s.slotcache_rust, s.routetable_rust,
                    s.rc_calls, s.rc_pos, s.rc_ok, s.vq);
    for (int c = 0; c < EC_MAX_CARDS; c++) { const ec_card_stats &k = s.card[c];
        o += sfmt("   card[%d] ready %d slots %d needed %lu hit_card %lu hit_host %lu from_drive %lu written_back %lu t_read %s t_copy %s\n", c, k.ready, k.slots, k.needed, k.hit_card, k.hit_host, k.from_drive,
                  k.written_back, hexd(k.t_read).c_str(), hexd(k.t_copy).c_str()); }
    o += sfmt("   cq %lu %lu %lu sc %lu %lu %lu nu %lu pf %lu %lu %lu %lu t_pwait %s wb_skipped %lu vq %lu %lu %lu %lu %lu t_vwait %s t_plan %s\n", s.cq_jobs[0], s.cq_jobs[1], s.cq_jobs[2], s.sc_calls[0],
              s.sc_calls[1], s.sc_calls[2], s.rt_nu_calls, s.pf_planned, s.pf_skipped, s.pf_used, s.pf_waited, hexd(s.t_pwait).c_str(), s.wb_skipped, s.vq_ram, s.vq_drive, s.vq_wb, s.vq_nowb, s.vq_waited,
              hexd(s.t_vwait).c_str(), hexd(s.t_plan).c_str());
    for (int i = 0; i < 3; i++) o += sfmt("   n_calls[%d] %s t_calls[%d] %s\n", i, hexd(s.n_calls[i]).c_str(), i, hexd(s.t_calls[i]).c_str());
    return o;
}
static void sc_summary(Api &R, Api &C, Rng &r, int cases)
{
    for (int i = 0; i < cases && !STOP(); i++) {
        g_cases++;
        ec_summary s;
        gen_summary(r, s);
        const string a = capture([&] { R.summary(&s); });
        const vector<string> wb = capture_writes([&] { C.summary(&s); });
        const string b = join_writes(wb);
        EQS(a, b, "print_summary stderr", show_summary(s));
        // no line of the summary carries user bytes, so every original line is exactly one newline: the candidate must make exactly one write per line
        const size_t lines = count_of(a, "\n");
        check_writes("print_summary (candidate)", wb, (long)lines, show_summary(s));
        if ((i & 15) == 0) {   // what the reference does for the same case: glibc splits a line of more than 128 bytes (not held against it, only counted)
            const vector<string> wa = capture_writes([&] { R.summary(&s); });
            stat("writes.summary_cases_sampled"); stat("writes.reference_writes", wa.size()); stat("writes.reference_lines", lines);
            if (wa.size() != lines) stat("writes.reference_cases_with_a_split_line");
        }
        stat("summary.cases"); stat("summary.lines", count_of(a, "\n")); stat("summary.experts_line", a.find("K3GPU experts:") != string::npos); stat("summary.card_lines", count_of(a, "K3GPU card ") - count_of(a, "K3GPU card queues"));
        stat("summary.route_cache_line", a.find("K3GPU route cache:") != string::npos); stat("summary.queue_line", a.find("K3GPU card queues:") != string::npos); stat("summary.call_class_lines", count_of(a, "layer calls:"));
        stat("summary.output_has_nan", a.find("nan") != string::npos); stat("summary.output_has_inf", a.find("inf") != string::npos); stat("summary.output_has_negative_zero", a.find("-0.0") != string::npos || a.find(": -0,") != string::npos);
        stat("summary.output_over_1000_bytes", a.size() > 1000);
        g_chk++;
        if (a.empty() || a.find("K3GPU route table") == string::npos) fail("print_summary: reference printed no table line (driver check)", esc(a), "", show_summary(s));
    }
}

// ------------------------------------------------------------------------------------------------ E6 selectors, trace, prefill, messages
static const char *SEL_VALUES[] = { "cpp", "rust", "Rust", "RUST", "CPP", " rust", "rust ", "cpp ", " cpp", "cpp\n", "rust\n", "1", "0", "none", "cp", "rus", "rustx", "cppcpp", "c++", "native", "-" };
static Val gen_sel_val(Rng &r)
{
    Val v; const uint32_t k = r.u(100);
    if (k < 15) return v;
    v.set = true;
    if (k < 23) return v;
    if (k < 38) { v.s = "cpp"; return v; }
    if (k < 53) { v.s = "rust"; return v; }
    if (k < 85) v.s = SEL_VALUES[r.u(sizeof SEL_VALUES / sizeof *SEL_VALUES)]; else if (k < 95) v.s = gen_bytes(r, 20); else v.s = gen_numstr(r);
    return v;
}
// mostly valid, so that the later steps of configure() are reached
static Val gen_sel_val_mostly_valid(Rng &r)
{
    Val v; const uint32_t k = r.u(100);
    if (k < 55) return v;
    v.set = true;
    if (k < 62) return v;
    if (k < 80) v.s = "cpp"; else if (k < 92) v.s = "rust"; else if (k < 97) v.s = SEL_VALUES[r.u(sizeof SEL_VALUES / sizeof *SEL_VALUES)]; else v.s = gen_bytes(r, 20);
    return v;
}
static Val gen_lib_val(Rng &r)
{
    Val v; const uint32_t k = r.u(100);
    if (k < 30) return v;
    v.set = true;
    if (k < 40) return v;
    if (k < 70) v.s = gen_path(r); else if (k < 90) v.s = sfmt("/opt/lib%u.so", r.u(100)); else v.s = gen_bytes(r, 30);
    return v;
}
static const char *SEL_NAMES[3][2] = { { "K3GPU_COPYQ", "K3GPU_COPYQ_LIB" }, { "K3GPU_SLOTCACHE", "K3GPU_SLOTCACHE_LIB" }, { "K3GPU_ROUTETABLE", "K3GPU_ROUTETABLE_LIB" } };

static void e6_selector(Api &R, Api &C, Rng &r)
{
    clear_env();
    const int comp = (int)r.u(3), built = r.p(0.6);
    const Val m = gen_sel_val(r), l = gen_lib_val(r);
    put_env(SEL_NAMES[comp][0], m); put_env(SEL_NAMES[comp][1], l);
    // other components' variables are set to junk: they must not matter
    for (int o = 0; o < 3; o++) if (o != comp && r.p(0.5)) put_env(SEL_NAMES[o][0], gen_sel_val(r));
    ChildOpt cand_opt; cand_opt.dgram = true;
    auto go = [&](const Api &A, const ChildOpt &o) { return run_child([&](FILE *res) {
        const int mode = A.sel_mode(comp, built);
        const char *lib = A.sel_lib(comp);
        fprintf(res, "mode=%d lib=%s\n", mode, lib ? lib : "(null)");
        A.sel_ann(comp, lib);
        return 0; }, o); };
    const Child a = go(R, ChildOpt()), b = go(C, cand_opt);
    const string desc = sfmt("component %d built %d env:%s", comp, built, env_dump().c_str());
    cmp_child("selector", a, b, desc);
    check_writes("selector (candidate)", b.writes, -1, desc);
    stat(a.code == 1 ? "selector.fatal" : "selector.ok"); if (a.res.find("mode=1") != string::npos) stat("selector.mode_rust"); if (a.code == 1) stat("selector.fatal_kind:" + err_kind(a.err));
}
static void e6_trace(Api &R, Api &C, Rng &r)
{
    clear_env();
    const string d = g_dir;
    const char *files[] = { "tr_a", "tr_b" };
    Val t; const uint32_t k = r.u(14);
    t.set = k != 0;
    switch (k) {
    case 0: break; case 1: break;
    case 2: t.s = "/dev/null"; break; case 3: t.s = d + "/tr_a"; break; case 4: t.s = "tr_b"; break; case 5: t.s = "/nonexistent_engcfg_dir/x"; break; case 6: t.s = d; break;
    case 7: t.s = d + "/no_such_subdir/tr_a"; break; case 8: t.s = "/proc/engcfg_cannot_create"; break; case 9: t.s = string(5000, 'x'); break;
    case 10: t.s = d + "/tr_\xff\xfe"; break; case 11: t.s = d + "/tr_a "; break; case 12: t.s = "/"; break; default: t.s = d + "/" + string((size_t)r.range(1, 300), 'q'); break;
    }
    put_env("K3GPU_TRACE", t);
    // a file that is already there with content: "w" truncates it (an append or read-write mutant would keep the bytes)
    string target;
    if (k == 3) target = d + "/tr_a"; else if (k == 4) target = d + "/tr_b"; else if (k == 10) target = d + "/tr_\xff\xfe"; else if (k == 11) target = d + "/tr_a ";
    const bool pre = !target.empty() && r.p(0.5);
    if (pre) stat("trace.preexisting_content");
    auto clean = [&]() { for (const char *f : files) unlink((d + "/" + f).c_str()); unlink((d + "/tr_a ").c_str()); unlink((d + "/tr_\xff\xfe").c_str()); if (t.set && t.s.size() > d.size() + 2 && t.s.compare(0, d.size() + 1, d + "/") == 0) unlink(t.s.c_str()); };
    ChildOpt cand_opt; cand_opt.dgram = true;
    auto go = [&](const Api &A, const ChildOpt &o) {
        clean();
        if (pre) write_file(target, "OLD DATA, 21 bytes.\n");
        Child c = run_child([&](FILE *res) { FILE *f = A.open_trace(); fprintf(res, "%s\n", f ? "stream" : "null"); return 0; }, o);
        struct stat sb; string created = "files:";
        if (t.set && !t.s.empty() && stat(t.s.c_str(), &sb) == 0 && S_ISREG(sb.st_mode)) created += sfmt(" %s size=%lld mode=%o", t.s == "/dev/null" ? "-" : "regular", (long long)sb.st_size, (unsigned)(sb.st_mode & 0777));
        c.res += created;
        clean();
        return c; };
    const Child a = go(R, ChildOpt()), b = go(C, cand_opt);
    cmp_child("open_trace", a, b, "env:" + env_dump());
    check_writes("open_trace (candidate)", b.writes, -1, "env:" + env_dump());
    stat(a.code == 1 ? "trace.fatal" : a.res.find("stream") != string::npos ? "trace.opened" : "trace.null"); if (a.res.find("regular") != string::npos) stat("trace.file_created");
}
static void e6_prefill(Api &R, Api &C, Rng &r)
{
    static const size_t calls[] = { 0, 1, 2, 3, 5612, (size_t)-1, 0, 1 };
    const size_t n = r.p(0.85) ? calls[r.u(8)] : (size_t)gen_u64(r);
    ChildOpt cand_opt; cand_opt.dgram = true;
    auto go = [&](const Api &A, const ChildOpt &o) { return run_child([&](FILE *res) { A.prefill(n); fprintf(res, "continued\n"); return 0; }, o); };
    const Child a = go(R, ChildOpt()), b = go(C, cand_opt);
    cmp_child("prefill_start", a, b, sfmt("calls %zu", n));
    check_writes("prefill_start (candidate)", b.writes, -1, sfmt("calls %zu", n));
    stat(a.code == 1 ? "prefill.fatal" : "prefill.on_line");
}
static void e6_messages(Api &R, Api &C, Rng &r)
{
    ec_settings s; memset(&s, 0, sizeof s);
    string path = r.p(0.15) ? "" : gen_path(r);
    const bool null_rc = r.p(0.1);
    s.cards = gen_int(r); s.readers = gen_int(r); s.verbose = gen_int(r); s.reserve = (size_t)gen_u64(r); s.policy = gen_int(r); s.chunks = gen_int(r); s.host_gb = gen_int(r); s.host_bytes = (size_t)gen_u64(r);
    s.routecache = null_rc ? nullptr : path.c_str(); s.lookahead = gen_int(r); s.vqueue = r.p(0.5) ? (r.p(0.7) ? 1 : gen_int(r)) : 0; s.vq_cap = (size_t)gen_u64(r); s.vq_from = (long)(int64_t)gen_u64(r);
    s.layout = gen_int(r); s.prefill = gen_int(r);
    const size_t calls = (size_t)gen_u64(r), pend = (size_t)gen_u64(r);
    const string desc = sfmt("settings: %s calls %zu pend_cap %zu", show_settings(s).c_str(), calls, pend);
    const string a = capture([&] { R.print_rc(&s, calls, pend); R.start_line(&s); });
    const vector<string> wb = capture_writes([&] { C.print_rc(&s, calls, pend); C.start_line(&s); });
    EQS(a, join_writes(wb), "route cache line + start line", desc);
    check_writes("route cache line + start line (candidate)", wb, 2, desc);   // two fprintf lines, whatever bytes the path holds
    stat("messages.cases"); if (null_rc) stat("messages.null_routecache"); if (s.vqueue) stat("messages.queue_on");
    // announce with arbitrary lib bytes and every component, plus NULL
    const int comp = (int)r.u(3);
    string lib = r.p(0.5) ? gen_path(r) : gen_bytes(r, 40);
    const char *libp = r.p(0.05) ? nullptr : lib.c_str();
    const string a2 = capture([&] { R.sel_ann(comp, libp); });
    const vector<string> wb2 = capture_writes([&] { C.sel_ann(comp, libp); });
    EQS(a2, join_writes(wb2), "selector announce line", sfmt("component %d lib %s[%s]", comp, libp ? "" : "NULL", libp ? esc(libp).c_str() : ""));
    check_writes("selector announce line (candidate)", wb2, 1, sfmt("component %d lib %s[%s]", comp, libp ? "" : "NULL", libp ? esc(libp).c_str() : ""));
}
static void sc_misc(Api &R, Api &C, Rng &r, int cases)
{
    for (int i = 0; i < cases && !STOP(); i++) {
        g_cases++;
        const uint32_t k = r.u(100);
        if (k < 40) e6_selector(R, C, r);
        else if (k < 60) e6_trace(R, C, r);
        else if (k < 70) e6_prefill(R, C, r);
        else e6_messages(R, C, r);
    }
    clear_env();
}

// ------------------------------------------------------------------------------------------------ E7 composite configure()
// Arm A: the reference's verbatim configure() text (ecref_configure). Arms B and C: the documented sequence of contract calls (include/engcfg.h) on the reference library and on the candidate.
static const Api *g_glue_api = nullptr;
static int g_glue_cards = 0; static size_t g_glue_rc = 0;
static void glue_atexit()
{
    ec_summary s; memset(&s, 0, sizeof s);
    s.cards = g_glue_cards; s.expert_raw = 17547264; s.rc_calls = g_glue_rc;
    g_glue_api->summary(&s);   // the original registers print_summary with all counters at their initial values; the C++ adapter will fill the struct from the same globals
}
static void stand_select(const Api &A, int comp, int built)
{
    if (A.sel_mode(comp, built) == EC_MODE_RUST) A.sel_ann(comp, A.sel_lib(comp));
}
static void glue_configure(const Api &A, int devices, const int built[3], ecref_snap &snap)
{
    memset(&snap, 0, sizeof snap);
    g_glue_api = &A;
    ec_settings s; A.settings(&s, devices);
    snap.configured = 1; snap.cards = s.cards; snap.readers = s.readers; snap.verbose = s.verbose; snap.reserve = s.reserve;
    g_glue_cards = s.cards;
    stand_select(A, EC_SLOTCACHE, built[1]);
    snap.policy = s.policy;
    stand_select(A, EC_ROUTETABLE, built[2]);
    snap.chunks = s.chunks;
    FILE *tr = A.open_trace(); snap.trace_open = tr != nullptr;
    snap.host_bytes = s.host_bytes; snap.host_slots = s.host_bytes / 17547264 / 32 * 32;
    fprintf(stderr, "K3GPU: host expert cache of %zu slots (stand-in)\n", snap.host_slots);
    size_t rc_n = 0;
    snap.vq_cap = 256;
    if (s.routecache) {
        rc_n = strlen(s.routecache) % 3;
        snap.lookahead = s.lookahead;
        snap.pend_cap = A.pend_cap(snap.host_slots);
        snap.hook_set = 1; snap.rc_ok = rc_n != 0;
        if (s.policy == 2) snap.nu_set = 1;
        snap.vq = s.vqueue; snap.vq_cap = s.vq_cap; snap.vq_from = s.vq_from;
        A.print_rc(&s, rc_n, snap.pend_cap);
    }
    snap.rc_calls = rc_n; g_glue_rc = rc_n;
    snap.layout = s.layout; snap.prefill = s.prefill;
    if (s.prefill) { A.prefill(rc_n); snap.force_set = 1; }
    stand_select(A, EC_COPYQ, built[0]);
    atexit(glue_atexit);
    A.start_line(&s);
}
static string dump_snap(const ecref_snap &s)
{
    return sfmt("configured=%d cards=%d readers=%d verbose=%d reserve=%zu policy=%d chunks=%d trace_open=%d host_bytes=%zu host_slots=%zu rc_calls=%zu lookahead=%d pend_cap=%zu rc_ok=%d nu_set=%d vq=%d vq_cap=%zu vq_from=%ld hook_set=%d layout=%d prefill=%d force_set=%d",
                s.configured, s.cards, s.readers, s.verbose, s.reserve, s.policy, s.chunks, s.trace_open, s.host_bytes, s.host_slots, s.rc_calls, s.lookahead, s.pend_cap, s.rc_ok, s.nu_set, s.vq, s.vq_cap,
                s.vq_from, s.hook_set, s.layout, s.prefill, s.force_set);
}
static void gen_config_env(Rng &r)
{
    clear_env();
    static const char *rows[][9] = {
        { "K3GPU_CARDS", "1", "2", "4", "8", "16", "3", "0", "-1" }, { "K3GPU_POLICY", "0", "1", "2", "3", "2", "2", "-1", "x" }, { "K3GPU_VQUEUE", "0", "1", "1", "2", "-1", "1", "x", "0" },
        { "K3GPU_PREFILL", "0", "1", "1", "1", "2", "-1", "x", "0" }, { "K3GPU_LAYOUT", "0", "1", "1", "2", "-1", "x", "1", "0" }, { "K3GPU_HOST_GB", "150", "1", "2", "0", "-1", "50", "2147483647", "x" },
        { "K3GPU_RESERVE_MB", "2500", "0", "1", "-1", "2147483647", "2147483648", "4096", "x" }, { "K3GPU_READERS", "16", "1", "32", "0", "-1", "8", "64", "x" },
        { "K3GPU_LOOKAHEAD", "92", "0", "1", "10", "-1", "184", "x", "92" }, { "K3GPU_VQ_INFLIGHT", "256", "0", "1", "-1", "2147483648", "64", "x", "256" },
        { "K3GPU_VQ_FROM", "0", "92", "1", "-1", "2147483648", "100", "x", "0" }, { "K3GPU_VERBOSE", "0", "1", "2", "-1", "x", "1", "0", "0" }, { "K3GPU_CHUNKS", "0", "1", "4", "8", "-1", "x", "16", "0" } };
    for (auto &row : rows) {
        const uint32_t k = r.u(10);
        if (k < 3) continue;
        Val v; v.set = true;
        if (k < 9) v.s = row[1 + r.u(8)]; else v = gen_val(r);
        put_env(row[0], v);
    }
    if (r.p(0.7)) { Val v; v.set = true; v.s = r.p(0.05) ? "" : gen_path(r); put_env("K3GPU_ROUTECACHE", v); }
    for (int c = 0; c < 3; c++) { if (r.p(0.9)) put_env(SEL_NAMES[c][0], gen_sel_val_mostly_valid(r)); if (r.p(0.3)) put_env(SEL_NAMES[c][1], gen_lib_val(r)); }
    if (r.p(0.35)) {
        Val t; t.set = true;
        switch (r.u(6)) { case 0: t.s = ""; break; case 1: t.s = "/dev/null"; break; case 2: t.s = g_dir + "/tr_a"; break; case 3: t.s = "/nonexistent_engcfg_dir/x"; break; case 4: t.s = g_dir; break; default: t.s = "tr_b"; break; }
        put_env("K3GPU_TRACE", t);
    }
}
static void sc_composite(Api &R, Api &C, Rng &r, int cases)
{
    static const int devs[] = { 0, 1, 2, 3, 4, 8, 9, 16, -1, INT_MAX, 5, 6, 7, 8, 8, 8 };
    for (int i = 0; i < cases && !STOP(); i++) {
        g_cases++;
        gen_config_env(r);
        const int dev = devs[r.u(16)];
        int built[3] = { (int)r.p(0.88), (int)r.p(0.88), (int)r.p(0.88) };
        auto clean = [&]() { unlink((g_dir + "/tr_a").c_str()); unlink((g_dir + "/tr_b").c_str()); };
        ecref_snap sa, sb, sc;
        clean();
        const Child a = run_child([&](FILE *res) { R.ref_configure(dev, built); R.ref_snapshot(&sa); fprintf(res, "%s\n", dump_snap(sa).c_str()); return 0; });
        clean();
        const Child b = run_child([&](FILE *res) { glue_configure(R, dev, built, sb); fprintf(res, "%s\n", dump_snap(sb).c_str()); return 0; });
        clean();
        ChildOpt cand_opt; cand_opt.dgram = true;
        const Child c = run_child([&](FILE *res) { glue_configure(C, dev, built, sc); fprintf(res, "%s\n", dump_snap(sc).c_str()); return 0; }, cand_opt);
        clean();
        (void)sa; (void)sb; (void)sc;
        const string desc = sfmt("devices %d built copyq=%d slotcache=%d routetable=%d env:%s", dev, built[0], built[1], built[2], env_dump().c_str());
        cmp_child("composite: verbatim configure() vs contract sequence on the reference lib", a, b, desc);
        cmp_child("composite: verbatim configure() vs contract sequence on the candidate", a, c, desc);
        check_writes("composite (candidate)", c.writes, -1, desc);
        stat("composite.cases"); stat(a.code == 0 ? "composite.exit0" : "composite.exit" + std::to_string(a.code));
        if (a.code == 1) stat("composite.fatal_kind:" + err_kind(a.err));
        stat("composite.route_cache_line", a.err.find("K3GPU route cache:") != string::npos); stat("composite.prefill_on_line", a.err.find("PREFILL ON") != string::npos);
        stat("composite.rust_announce_lines", count_of(a.err, " rust (")); stat("composite.cpp_selector_lines", count_of(a.err, " cpp\n")); stat("composite.exit_summary_present", a.err.find("K3GPU copy queue") != string::npos);
        stat("composite.start_line", a.err.find(" drive readers\n") != string::npos); stat("composite.trace_opened", a.res.find("trace_open=1") != string::npos);
    }
    clear_env();
}

// ------------------------------------------------------------------------------------------------ E8 out-of-contract inputs: both libraries must die (the signal may differ) and say what they are documented to say
// Besides the signal, the text left on stderr is compared with what each library is designed to leave: the reference's own death (card_of: SIGFPE leaves no text, except that the UBSan build of the reference
// that asan.sh uses first prints its own "runtime error: division ..." line; force_routes: libstdc++'s terminate message naming
// std::length_error), the candidate's explicit std::process::abort() (nothing at all: no Rust panic text). The last pin is not a sentence of include/engcfg.h, which only says "the Rust library aborts";
// it is the crate's own design (NOTES.md: "abort without the libstdc++ message"; the explicit checks in card_of / force_routes), and it is what tells an explicit check from an implicit panic (the `%` itself, the vector size).
static void sc_deaths(Api &R, Api &C)
{
    struct Case { const char *name; std::function<int(const Api &, FILE *)> fn; const char *ref_err_has; };
    ec_summary s; memset(&s, 0, sizeof s);
    s.cards = 9;
    int e16[16] = { 0 }, c16[16]; for (int &x : c16) x = -1;
    ec_call neg = { 0, e16, 16, c16, 16 };
    int sel1[1] = { 0 };
    const vector<Case> cases = {
        { "card_of layout 1, cards 0, layer 5", [&](const Api &A, FILE *) { return A.card_of(1, 0, 5); }, "" },
        { "card_of layout 1, cards -1, layer INT_MIN+1", [&](const Api &A, FILE *) { return A.card_of(1, -1, INT_MIN + 1); }, "" },
        { "force_routes n = -1 reaching the vector", [&](const Api &A, FILE *) { A.force(0, 1, &neg, 0, -1, sel1, 16); return 0; }, "std::length_error" },
    };
    (void)s;
    for (const Case &c : cases) {
        g_cases++;
        const Child a = run_child([&](FILE *res) { return c.fn(R, res); }), b = run_child([&](FILE *res) { return c.fn(C, res); });
        g_cmp++;
        const bool both_died = a.sig != 0 && b.sig != 0;
        if (!both_died) fail(string("out of contract: both must die by a signal: ") + c.name, child_str(a), child_str(b), c.name);
        else printf("  E8 %s: reference died with signal %d, candidate with signal %d\n", c.name, a.sig, b.sig);
        g_cmp++;
        const bool ref_text_ok = c.ref_err_has[0] ? a.err.find(c.ref_err_has) != string::npos : (a.err.empty() || a.err.find("runtime error: division") != string::npos);   // the second form: the UBSan build of the reference (asan.sh) names the trap before SIGFPE
        if (!ref_text_ok) fail(string("out of contract: the reference died, but not with the text it is designed to leave (driver check): ") + c.name, c.ref_err_has[0] ? string("stderr containing ") + c.ref_err_has : "empty stderr", "[" + esc(a.err) + "]", c.name);
        g_cmp++;
        if (!b.err.empty()) fail(string("out of contract: the candidate must abort silently (explicit abort, no panic text): ") + c.name, "empty stderr", "[" + esc(b.err) + "]", c.name);
    }
}

// ------------------------------------------------------------------------------------------------ E9 force_routes at sizes E4 cannot reach (one fixed scenario per unit; a fresh process per library, so the once-only warning is the first thing seen)
// E4 draws n <= 200, top <= 17 and at most about 900 experts, so no integer wider than a byte, a short or an int is ever tested in the dealing. Each case below is a call of the contract that
// reaches such a width, built so that its cost is bounded (the largest needs two buffers of 16 MB and about 2.15e9 loop iterations; nothing allocates gigabytes). `warn` = the reason the call must be
// rejected with (reference and candidate must print it; sel must stay untouched), else the call deals and the reference's whole buffer is first checked against an analytic expectation, then the candidate's
// buffer against the reference's. The child of each library runs with RLIMIT_AS = --e9-vm-mb (default 2048): a candidate that accepts a call with 2^30 tokens (a 4 GB fill vector) fails to allocate and
// dies, which the comparison sees, instead of taking real memory. Not reachable inside that bound (stated, not hidden): a VALID call with n * top or with a sum of counts above 2^31 - 1 needs a buffer of
// n * top words (more than 8 GB) -- the accepting direction of the 32-bit mutants is tested, the rejecting direction (a valid call refused because the product wrapped negative) is not.
static size_t g_e9_vm_mb = 2048;
static const int E9_PAT = 0x5A5A5A5A;
enum { E9_CASES = 13 };
struct E9Case { const char *name = ""; int n = 1, top = 1; vector<int> experts, counts; size_t words = 64; bool heavy = false; const char *warn = nullptr; };
static const char *E9_NOFIT = "counts do not fit this call", *E9_NOCOUNTS = "the route cache has no counts";
static bool e9_heavy(int idx) { return idx == 0; }
static void e9_build(int idx, E9Case &c)
{
    switch (idx) {
    case 0:   // the dealing counter passes 2^31: 513 experts of count n = 2^22 plus one count of -2^31 (sum n = n * top), every token gets 513 deals, 513 * 2^22 = 2,151,677,952 deals
        c.name = "s32-heavy"; c.heavy = true; c.n = 1 << 22; c.top = 1;
        for (int e = 0; e <= 512; e++) { c.experts.push_back(1000 + e); c.counts.push_back(c.n); }
        c.experts.push_back(5); c.counts.push_back(INT_MIN);
        c.words = (size_t)c.n + 1024; break;
    case 1:   // sum of counts 5 * 2^30 = 2^32 + 2^30: a 32-bit sum reads 2^30 = n * top and accepts (then needs a 4 GB fill vector)
        c.name = "total32"; c.n = 1 << 30; c.top = 1; c.warn = E9_NOFIT;
        for (int e = 0; e < 5; e++) { c.experts.push_back(1 + e); c.counts.push_back(1 << 30); }
        break;
    case 2:   // sum of counts -2^32 reads 0 in 32 bits, equal to n * top = 0
        c.name = "total32neg"; c.n = 0; c.top = 16; c.warn = E9_NOFIT; c.experts = { 1, 2 }; c.counts = { INT_MIN, INT_MIN }; break;
    case 3:   // n * top = 2^32 reads 0 in 32 bits, equal to the sum 0
        c.name = "ntop32"; c.n = 4096; c.top = 1 << 20; c.warn = E9_NOFIT; c.experts = { 1, 2 }; c.counts = { 0, 0 }; break;
    case 4:   // n * top = 2147488281 reads -2147479015 in 32 bits, equal to the sum
        c.name = "ntop32neg"; c.n = 46341; c.top = 46341; c.warn = E9_NOFIT; c.experts = { 1 }; c.counts = { -2147479015 }; break;
    case 5: case 6: {   // one token receives every deal (n = 1): 300 / 70,000 deals, so a counter of 8 / 16 bits wraps
        const int m = idx == 5 ? 300 : 70000;
        c.name = idx == 5 ? "u8" : "u16"; c.n = 1; c.top = 1;
        for (int e = 0; e < m; e++) { c.experts.push_back(1000 + e); c.counts.push_back(1); }
        c.experts.push_back(5); c.counts.push_back(-(m - 1));
        c.words = (size_t)m + 1024; break; }
    case 7: c.name = "morecounts"; c.warn = E9_NOCOUNTS; c.experts = { 7 }; c.counts = { 1, 0 }; break;
    case 8: c.name = "morecounts-sum-ok"; c.warn = E9_NOCOUNTS; c.experts = { 7 }; c.counts = { 1, 1, -1 }; break;
    case 9: c.name = "counts-without-experts"; c.warn = E9_NOCOUNTS; c.counts = { 0 }; break;
    case 10: c.name = "count-above-n"; c.n = 3; c.top = 2; c.warn = E9_NOFIT; c.experts = { 10, 11, 12 }; c.counts = { 4, 1, 1 }; break;
    case 11: c.name = "count-above-n-negative"; c.n = 2; c.top = 1; c.warn = E9_NOFIT; c.experts = { 10, 11 }; c.counts = { 3, -1 }; break;
    case 12: c.name = "valid-control"; c.n = 3; c.top = 2; c.experts = { 10, 11, 12 }; c.counts = { 3, 2, 1 }; break;   // hand-dealt: sel = 10 11 10 11 10 12
    }
}
// what sel must hold afterwards, worked out by hand (not by either library)
static void e9_expected(const E9Case &c, vector<int> &sel)
{
    sel.assign(c.words, E9_PAT);
    if (c.warn) return;
    if (c.heavy) {   // expert e's d-th deal goes to token d (n divides e * n) at index d + e; the last writer of index j is the largest e <= min(512, j)
        for (size_t j = 0; j < (size_t)c.n + 512; j++) sel[j] = 1000 + (int)std::min<size_t>(512, j);
        return;
    }
    if (!strcmp(c.name, "valid-control")) { const int v[6] = { 10, 11, 10, 11, 10, 12 }; for (int j = 0; j < 6; j++) sel[j] = v[j]; return; }
    for (size_t e = 0; e + 1 < c.experts.size(); e++) sel[e] = c.experts[e];   // u8 / u16: n = 1, top = 1, the e-th deal goes to sel[e]
}
static string e9_outcome(const Child &c) { return c.sig ? sfmt("signal %d", c.sig) : sfmt("exit %d", c.code); }
static void sc_e9(Api &R, Api &C, int idx)
{
    E9Case k; e9_build(idx, k);
    g_cases++;
    auto make = [&k](const Api *A) { return std::function<int(FILE *)>([&k, A](FILE *res) {
        ec_call cur; cur.layer = 3;
        cur.experts = k.experts.empty() ? nullptr : k.experts.data(); cur.n_experts = k.experts.size();
        cur.counts = k.counts.empty() ? nullptr : k.counts.data(); cur.n_counts = k.counts.size();
        vector<int> sel(k.words, E9_PAT);
        A->force(0, 1, &cur, 3, k.n, sel.data(), k.top);
        fwrite(sel.data(), 4, sel.size(), res);
        return 0; }); };
    ChildOpt oa, ob; oa.vm_mb = ob.vm_mb = g_e9_vm_mb; oa.alarm_s = ob.alarm_s = 900; oa.tag = "ref"; ob.tag = "cand";
    ChildH ha = child_start(make(&R), oa), hb = child_start(make(&C), ob);   // both at once: the heavy case takes seconds per library
    const Child a = child_finish(ha), b = child_finish(hb);
    const string desc = sfmt("case %s: n %d top %d, %zu experts, %zu counts, sel of %zu words", k.name, k.n, k.top, k.experts.size(), k.counts.size(), k.words);
    printf("  E9 %s: reference %s, candidate %s\n", k.name, e9_outcome(a).c_str(), e9_outcome(b).c_str());
    // the instrument: the reference must have done what the case is designed to show
    vector<int> want; e9_expected(k, want);
    const string want_err = (k.warn ? sfmt("K3GPU prefill: call 0 (layer 3): %s; the computed routes are used from here\n", k.warn) : string()) + MARKER;
    size_t w = 0; const size_t nw = a.res.size() / 4; const int *pa = (const int *)a.res.data();
    while (w < nw && w < want.size() && pa[w] == want[w]) w++;
    g_cmp++;
    const bool ref_ok = a.code == 0 && a.sig == 0 && a.err == want_err && a.res.size() == want.size() * 4 && w == want.size();
    if (!ref_ok) fail("E9: the reference did not do what the case is designed to show (driver check)", sfmt("exit 0, stderr [%s], sel as worked out by hand", esc(want_err).c_str()),
                      sfmt("%s, stderr [%s], %zu words, first word that differs from the hand-worked buffer: %zu (%d vs %d)", e9_outcome(a).c_str(), esc(a.err).c_str(), nw, w, w < nw ? pa[w] : 0, w < want.size() ? want[w] : 0), desc);
    else stat(string("e9.") + k.name + ".reference_as_designed");
    // the comparison
    EQI(a.code, b.code, string("E9 ") + k.name + ": exit status", desc);
    EQI(a.sig, b.sig, string("E9 ") + k.name + ": terminating signal", desc);
    EQS(a.err, b.err, string("E9 ") + k.name + ": stderr", desc);
    g_cmp++;
    if (a.res != b.res) {
        const int *pb = (const int *)b.res.data(); const size_t nb = b.res.size() / 4;
        size_t d = 0; while (d < nw && d < nb && pa[d] == pb[d]) d++;
        fail(string("E9 ") + k.name + ": selected array differs", sfmt("%zu words, word %zu = %d", nw, d, d < nw ? pa[d] : 0), sfmt("%zu words, word %zu = %d", nb, d, d < nb ? pb[d] : 0), desc);
    }
    g_chk++;
}

// ------------------------------------------------------------------------------------------------ units, workers, scheduling
struct Unit { string scen; uint64_t seed; int cases; };
struct Opts { string ref, cand, out = "out", only; int jobs = 24; double scale = 1.0; uint64_t seed0 = 1; };

static int run_unit(const Opts &o, const Unit &u)
{
    Api R, C;
    if (!load_api(R, o.ref.c_str(), true) || !load_api(C, o.cand.c_str(), false)) return 2;
    g_scen = u.scen; g_seed = u.seed; g_cmp = g_chk = g_cases = 0; g_failed = false; g_nfail = 0; g_report.clear();
    g_dir = o.out + "/work/" + u.scen + "-" + std::to_string(u.seed);
    mkdir((o.out + "/work").c_str(), 0755); mkdir(g_dir.c_str(), 0755);
    char cwd0[4096]; if (!getcwd(cwd0, sizeof cwd0)) return 2;
    if (g_dir[0] != '/') g_dir = string(cwd0) + "/" + g_dir;
    if (chdir(g_dir.c_str()) != 0) { perror("chdir"); return 2; }
    g_capfd = open((g_dir + "/cap.tmp").c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    g_saved2 = dup(2);
    if (g_capfd < 0 || g_saved2 < 0) return 2;
    Rng r(mix64(u.seed * 1000003ull + std::hash<string>()(u.scen)));
    clear_env();
    if (u.scen == "E1") sc_env_int(R, C, r, u.cases);
    else if (u.scen == "E2") sc_settings(R, C, r, u.cases);
    else if (u.scen == "E3") sc_card_of(R, C, r, u.cases, u.seed == 0);
    else if (u.scen == "E4") sc_force(R, C, r, u.cases);
    else if (u.scen == "E5") sc_summary(R, C, r, u.cases);
    else if (u.scen == "E6") sc_misc(R, C, r, u.cases);
    else if (u.scen == "E7") sc_composite(R, C, r, u.cases);
    else if (u.scen == "E8") sc_deaths(R, C);
    else if (u.scen == "E9") sc_e9(R, C, (int)(u.seed - o.seed0));
    const string resl = sfmt("UNIT %s %llu %llu %llu %llu %s\n", u.scen.c_str(), (unsigned long long)u.seed, (unsigned long long)g_cmp, (unsigned long long)g_chk, (unsigned long long)g_cases,
                             g_failed ? "FAIL" : "PASS");
    string stats_text;
    for (auto &kv : g_stats) stats_text += sfmt("STAT %s %llu %s %llu\n", u.scen.c_str(), (unsigned long long)u.seed, kv.first.c_str(), (unsigned long long)kv.second);
    if (chdir(cwd0) != 0) return 2;
    write_file(g_dir + ".res", resl + stats_text);
    if (g_failed) { mkdir((o.out + "/fail").c_str(), 0755); write_file(o.out + "/fail/" + u.scen + "-" + std::to_string(u.seed) + ".txt", g_report); return 1; }
    // a passing unit leaves nothing behind
    unlink((g_dir + "/cap.tmp").c_str()); unlink((g_dir + "/child.err").c_str()); unlink((g_dir + "/child.res").c_str()); rmdir(g_dir.c_str());
    return 0;
}

int main(int argc, char **argv)
{
    Opts o;
    vector<string> pos;
    for (int i = 1; i < argc; i++) {
        const string a = argv[i];
        auto next = [&]() -> string { if (i + 1 >= argc) { fprintf(stderr, "missing value after %s\n", a.c_str()); exit(2); } return argv[++i]; };
        if (a == "--only") o.only = next(); else if (a == "--jobs") o.jobs = atoi(next().c_str()); else if (a == "--scale") o.scale = atof(next().c_str());
        else if (a == "--out") o.out = next(); else if (a == "--keep-going") g_keep_going = true; else if (a == "--flip") g_flip = strtoull(next().c_str(), nullptr, 10);
        else if (a == "--seed0") o.seed0 = strtoull(next().c_str(), nullptr, 10);
        else if (a == "--e9-vm-mb") g_e9_vm_mb = (size_t)strtoull(next().c_str(), nullptr, 10);
        else if (a.size() > 1 && a[0] == '-' && a[1] == '-') { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; } else pos.push_back(a);
    }
    if (pos.size() != 2) { fprintf(stderr, "usage: diff_driver REF.so CAND.so [options]\n"); return 2; }
    o.ref = pos[0]; o.cand = pos[1];
    if (o.jobs > 24) o.jobs = 24;
    mkdir(o.out.c_str(), 0755);
    auto want = [&](const char *s) { return o.only.empty() || ("," + o.only + ",").find(string(",") + s + ",") != string::npos; };
    auto sc = [&](int n) { return std::max(1, (int)(n * o.scale)); };
    vector<Unit> units;
    const uint64_t s0 = o.seed0;
    if (want("E8")) units.push_back({ "E8", s0, 0 });
    if (want("E3")) { units.push_back({ "E3", 0, 0 }); for (int i = 1; i <= 24; i++) units.push_back({ "E3", s0 + i, sc(100000) }); }
    if (want("E7")) for (int i = 0; i < 100; i++) units.push_back({ "E7", s0 + i, sc(1000) });
    // E9 after E7: a mutant caught by E8, E3 or E7 stops the scheduling before the heavy case starts. Scaled (smoke) runs leave the heavy case out.
    if (want("E9")) for (int i = 0; i < E9_CASES; i++) if (!(o.scale < 1.0 && e9_heavy(i))) units.push_back({ "E9", s0 + i, 0 });
    if (want("E4")) for (int i = 0; i < 4000; i++) units.push_back({ "E4", s0 + i, 60 });
    if (want("E6")) for (int i = 0; i < 60; i++) units.push_back({ "E6", s0 + i, sc(1000) });
    if (want("E5")) for (int i = 0; i < 100; i++) units.push_back({ "E5", s0 + i, sc(3000) });
    if (want("E2")) for (int i = 0; i < 40; i++) units.push_back({ "E2", s0 + i, sc(4000) });
    if (want("E1")) for (int i = 0; i < 60; i++) units.push_back({ "E1", s0 + i, sc(5000) });
    if (o.scale < 1.0) { vector<Unit> keep; std::map<string, int> seen; const int cap = std::max(1, (int)(100 * o.scale)); for (auto &u : units) if (seen[u.scen]++ < cap || u.scen == "E8" || u.scen == "E9") keep.push_back(u); units = keep; }

    FILE *log = fopen((o.out + "/units.log").c_str(), "w");
    if (!log) { perror("units.log"); return 2; }
    size_t next = 0, running = 0, done = 0; int bad = 0;
    std::map<pid_t, Unit> pids;
    std::map<string, std::pair<uint64_t, uint64_t>> tot, totc;   // scen -> (units pass, units fail), (cmp, cases)
    std::map<string, uint64_t> totchk, allstats;
    printf("diff_driver: %zu units, %d jobs, reference %s, candidate %s%s\n", units.size(), o.jobs, o.ref.c_str(), o.cand.c_str(), g_flip ? " (flip active)" : "");
    fflush(stdout);
    while (done < units.size()) {
        while (running < (size_t)o.jobs && next < units.size() && (bad == 0 || g_keep_going)) {
            const Unit u = units[next++];
            fprintf(log, "BEGIN %s %llu\n", u.scen.c_str(), (unsigned long long)u.seed); fflush(log);
            fflush(NULL);
            const pid_t p = fork();
            if (p < 0) { perror("fork"); return 2; }
            if (p == 0) { fclose(log); const int rc = run_unit(o, u); fflush(stdout); _exit(rc); }
            pids[p] = u; running++;
        }
        if (running == 0) break;
        int st = 0;
        const pid_t p = waitpid(-1, &st, 0);
        if (p < 0) break;
        auto it = pids.find(p); if (it == pids.end()) continue;
        const Unit u = it->second; pids.erase(it); running--; done++;
        string line;
        read_file(o.out + "/work/" + u.scen + "-" + std::to_string(u.seed) + ".res", line);
        unlink((o.out + "/work/" + u.scen + "-" + std::to_string(u.seed) + ".res").c_str());
        if (WIFSIGNALED(st) || line.empty()) {
            line = sfmt("UNIT %s %llu 0 0 0 CRASH sig=%d exit=%d\n", u.scen.c_str(), (unsigned long long)u.seed, WIFSIGNALED(st) ? WTERMSIG(st) : 0, WIFEXITED(st) ? WEXITSTATUS(st) : -1);
            bad++; tot[u.scen].second++;
        } else {
            char sname[16]; unsigned long long sd, cm, ck, cs; char res[16];
            if (sscanf(line.c_str(), "UNIT %15s %llu %llu %llu %llu %15s", sname, &sd, &cm, &ck, &cs, res) == 6) {
                totc[u.scen].first += cm; totc[u.scen].second += cs; totchk[u.scen] += ck;
                if (!strcmp(res, "PASS")) tot[u.scen].first++; else { tot[u.scen].second++; bad++; }
                for (size_t p = line.find("STAT "); p != string::npos; p = line.find("\nSTAT ", p + 1) == string::npos ? string::npos : line.find("\nSTAT ", p + 1) + 1) {
                    char sn[16], key[200]; unsigned long long sd2, val;
                    if (sscanf(line.c_str() + p, "STAT %15s %llu %199s %llu", sn, &sd2, key, &val) == 4) allstats[string(sn) + " " + key] += val;
                }
            } else { bad++; tot[u.scen].second++; }
        }
        fputs(line.c_str(), log); fflush(log);
    }
    fclose(log);
    // wait for stragglers when stopping early
    while (running > 0) { int st; pid_t p = waitpid(-1, &st, 0); if (p < 0) break; running--; }
    uint64_t allc = 0, allcases = 0, allunits = 0, allfail = 0;
    for (auto &kv : tot) {
        printf("%s: %llu units PASS, %llu FAIL/CRASH, %llu comparisons, %llu single-library checks, %llu cases\n", kv.first.c_str(), (unsigned long long)kv.second.first, (unsigned long long)kv.second.second,
               (unsigned long long)totc[kv.first].first, (unsigned long long)totchk[kv.first], (unsigned long long)totc[kv.first].second);
        allc += totc[kv.first].first; allcases += totc[kv.first].second; allunits += kv.second.first + kv.second.second; allfail += kv.second.second;
    }
    printf("TOTAL: %llu units (%llu failed), %llu comparisons, %llu cases%s\n", (unsigned long long)allunits, (unsigned long long)allfail, (unsigned long long)allc, (unsigned long long)allcases,
           bad ? "  -> FAIL (see fail/)" : "  -> all PASS");
    printf("branch statistics (sum over units; STAT lines in units.log):\n");
    for (auto &kv : allstats) printf("  %-70s %llu\n", kv.first.c_str(), (unsigned long long)kv.second);
    return bad ? 1 : 0;
}
