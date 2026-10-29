// Differential driver of the pool start-up (include/poolinit.h): the C++ reference (ref/libpoolinit_ref.so, the original text behind a harness) against the Rust crate (libpoolinit.so), same C ABI.
// One binary, three modes:
//   child : runs ONE unit (a seeded list of 1..3 init calls) against ONE library inside this process and writes a canonical log; its stderr (the library's messages) goes to a file.
//   unit  : generates the unit, starts a child per library (in parallel), compares the logs byte for byte and the exit statuses, prints one result line.
//   run   : schedules many units over many processes and prints one summary line per scenario.
// Inside a child nothing is faked except CUDA (the pi_ops table: a recording fake that returns configured results, addresses from fake arenas, errors at a configured call) and the
// slot caches' spares, which go to the REAL libslotcache.so (or, in a quarter of the steps, to a recording stub with arbitrary cache pointers, NULL included). The real system calls of the
// host cache (mmap, madvise, the zeroing, the 15 helper threads) really run, with small sizes: this executable is linked -rdynamic and defines mmap / madvise / pthread_create, which both
// libraries reach through the dynamic linker; the definitions record the calls, can make the k-th mmap fail, and (poison mode) fill each new mapping with 0xA5 so that the library's
// zeroing is observable. In the other half of the steps the zeroing is observed through mincore (all pages resident).
// Every child also checks its library against a MODEL written here from decode.cuh@4d8d891 (expected ops-call list, out struct, stderr text, exit); a disagreement is a VIOLATION (exit 3).
// Scenarios: H host cache | HX host cache with a failing op or mmap | P card pool | PX pool boundaries (62..65 slots, wrapping slot counts) and failing ops | PR the engine's real constants
// (fake ops only: nothing is mapped) | M two or three calls in one process | A (not in the default run: --scn A) refused arguments: NULL pointers, NULL ops, expert_raw 0.
// Memory: a host step maps at most 48 MiB (real); the pool steps map nothing.
// DRIVER v2 (closes the blind spots listed in mutants/RESULTS.md; the canonical logs and the configurations of the six default scenarios are unchanged):
//   * the seconds of the summary line are checked against a virtual clock (clock_gettime is defined here too; see "the clock of a host step");
//   * gates on the host workers: a barrier (>= 16 chunks: all 16 workers, the calling thread included, must map) and a serial mode (the calling thread maps every chunk, so the i-th
//     registered address must be the i-th mapped one: checks the registration order in a single unit);
//   * the number of error_string calls (one per failed CHECK) and of write() calls on stderr (one per message; fd 2 of a child is a SOCK_SEQPACKET socket);
//   * a hard cap on the event recorder, and prctl(PR_SET_DUMPABLE, 0) in the child (no crash handler per crash);
//   * scenario A: the library's own refusals (NULL ops / cfg / out, a NULL function pointer, expert_raw 0) against the reference's, byte for byte.
#include "poolinit.h"
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <climits>
#include <map>
#include <mutex>
#include <string>
#include <thread>
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
    uint64_t r(uint64_t n) { return n ? u() % n : 0; }
    bool p(double q) { return (double)(u() >> 11) * (1.0 / 9007199254740992.0) < q; }
    long range(long a, long b) { return a + (long)r((uint64_t)(b - a + 1)); }
};
static std::string sfmt(const char *f, ...)
{
    char b[2048]; va_list ap; va_start(ap, f); vsnprintf(b, sizeof b, f, ap); va_end(ap); return b;
}
static std::string esc(const std::string &s)
{
    std::string o;
    for (unsigned char c : s) { if (c == '\n') o += "\\n"; else if (c < 32 || c > 126) o += sfmt("\\x%02x", c); else o.push_back((char)c); }
    return o;
}
static std::string read_all(const std::string &p)
{
    std::string s; FILE *f = fopen(p.c_str(), "rb"); if (!f) return s;
    char b[65536]; size_t n; while ((n = fread(b, 1, sizeof b, f)) > 0) s.append(b, n); fclose(f); return s;
}

// ------------------------------------------------------------------------------------------------ the interposed system calls (armed only around a library call)
static std::atomic<int> g_armed{ 0 };
static std::atomic<uint64_t> g_seq{ 0 };
struct MmapRec { uintptr_t ret; size_t len; int prot, flags, fd; long off; uintptr_t addr; uint64_t seq; long tid; };
struct MadvRec { uintptr_t addr; size_t len; int adv; int first_byte; uint64_t seq; };
static const unsigned REC_MAX = 4096;
static MmapRec g_mm[REC_MAX]; static std::atomic<unsigned> g_mm_n{ 0 }, g_mm_calls{ 0 };
static MadvRec g_ma[REC_MAX]; static std::atomic<unsigned> g_ma_n{ 0 };
static std::atomic<unsigned> g_thr_n{ 0 };
static std::atomic<int> g_mmap_fail_at{ 0 }, g_poison{ 0 };
static std::atomic<uint64_t> g_first_op_seq{ ~0ULL };

// driver v2, the clock of a host step. While g_vclock is set, clock_gettime(CLOCK_MONOTONIC) returns the real reading plus g_vns virtual nanoseconds, and g_vns grows by a fixed amount
// at every mmap (0.3 s), host_register (0.2 s), set_device (0.1 s) and every 32nd sc_add_spare (0.1 s): the "seconds" of the summary line then have a known value, 0.6 * n_chunks + 0.1 plus the real
// time, whatever the threads do and without sleeping. Every other clock id is shifted by +1000 s per reading, so a library that reads another clock prints about 1000 s.
static std::atomic<int> g_vclock{ 0 };
static std::atomic<long long> g_vns{ 0 };
static std::atomic<int> g_clk_other{ 0 };
static const long long VNS_MMAP = 300000000LL, VNS_REGISTER = 200000000LL, VNS_SETDEV = 100000000LL, VNS_SPARE32 = 100000000LL;
static long long expected_vns(size_t n_chunks) { return (long long)n_chunks * (VNS_MMAP + VNS_REGISTER + VNS_SPARE32) + VNS_SETDEV; }
static double real_mono() { struct timespec t; syscall(SYS_clock_gettime, CLOCK_MONOTONIC, &t); return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec; }
extern "C" int clock_gettime(clockid_t c, struct timespec *t) noexcept
{
    const int r = (int)syscall(SYS_clock_gettime, c, t);
    if (r != 0 || !g_vclock.load(std::memory_order_acquire)) return r;
    if (c == CLOCK_MONOTONIC) {
        const long long ns = (long long)t->tv_nsec + g_vns.load(std::memory_order_acquire);
        t->tv_sec += (time_t)(ns / 1000000000LL); t->tv_nsec = (long)(ns % 1000000000LL);
    } else t->tv_sec += 1000L * (g_clk_other.fetch_add(1) + 1);
    return r;
}

// driver v2, the gates of a host step (both release by themselves after GATE_TIMEOUT_MS and then record a timeout, which the step turns into a violation).
//   gate 1 (barrier, n_chunks >= 16): the first mmap of each thread waits until 16 distinct threads are inside mmap, so each of the 16 workers holds one chunk and the calling thread must be one of them.
//   gate 2 (serial, n_chunks >= 2): the 15 helpers do not start (a pthread_create trampoline holds them) until the calling thread has made its n-th mmap, so the calling thread maps every chunk,
//          the i-th mmap is chunk number i, and the i-th host_register must be the address of the i-th mmap.
static std::atomic<int> g_gate_mode{ 0 }, g_gate_open{ 0 }, g_gate_timeout{ 0 };
static std::atomic<unsigned> g_gate_need{ 0 }, g_gate_arrived{ 0 }, g_gate_n{ 0 }, g_epoch{ 0 };
static std::atomic<long> g_caller_tid{ 0 };
static thread_local unsigned t_epoch = 0;
static const double GATE_TIMEOUT_S = 5.0;
static void gate_wait()
{
    const double t0 = real_mono();
    while (!g_gate_open.load(std::memory_order_acquire)) {
        if (real_mono() - t0 > GATE_TIMEOUT_S) { g_gate_timeout.store(1); g_gate_open.store(1); break; }
        struct timespec ts = { 0, 50000 }; nanosleep(&ts, nullptr);
    }
}
static void gate_barrier()
{
    const unsigned ep = g_epoch.load();
    if (t_epoch == ep) return;
    t_epoch = ep;
    if (g_gate_arrived.fetch_add(1) + 1 >= g_gate_need.load()) { g_gate_open.store(1); return; }
    gate_wait();
}
struct Boot { void *(*f)(void *); void *arg; };
static void *boot_tramp(void *p)
{
    const Boot b = *(Boot *)p; delete (Boot *)p;
    gate_wait();
    return b.f(b.arg);
}

extern "C" void *mmap(void *addr, size_t len, int prot, int flags, int fd, off_t off) noexcept
{
    if (!g_armed.load(std::memory_order_acquire)) return (void *)syscall(SYS_mmap, addr, len, prot, flags, fd, off);
    const unsigned call = g_mm_calls.fetch_add(1) + 1;
    if (g_gate_mode.load(std::memory_order_acquire) == 1) gate_barrier();
    void *r;
    if (g_mmap_fail_at.load() && (int)call == g_mmap_fail_at.load()) { errno = ENOMEM; r = MAP_FAILED; }
    else {
        r = (void *)syscall(SYS_mmap, addr, len, prot, flags, fd, off);
        if (r != MAP_FAILED && g_poison.load()) memset(r, 0xA5, len);
    }
    if (g_vclock.load(std::memory_order_acquire)) g_vns.fetch_add(VNS_MMAP);
    const unsigned i = g_mm_n.fetch_add(1);
    if (i < REC_MAX) g_mm[i] = MmapRec{ (uintptr_t)r, len, prot, flags, fd, (long)off, (uintptr_t)addr, g_seq.fetch_add(1), (long)syscall(SYS_gettid) };
    if (g_gate_mode.load(std::memory_order_acquire) == 2 && call >= g_gate_n.load()) g_gate_open.store(1);
    return r;
}
extern "C" int madvise(void *addr, size_t len, int advice) noexcept
{
    if (!g_armed.load(std::memory_order_acquire)) return (int)syscall(SYS_madvise, addr, len, advice);
    const int fb = (addr && g_poison.load()) ? *(volatile unsigned char *)addr : -1;
    const unsigned i = g_ma_n.fetch_add(1);
    if (i < REC_MAX) g_ma[i] = MadvRec{ (uintptr_t)addr, len, advice, fb, g_seq.fetch_add(1) };
    return (int)syscall(SYS_madvise, addr, len, advice);
}
static int (*real_pthread_create)(pthread_t *, const pthread_attr_t *, void *(*)(void *), void *) = nullptr;
extern "C" int pthread_create(pthread_t *t, const pthread_attr_t *a, void *(*f)(void *), void *arg) noexcept
{
    if (!real_pthread_create) { void *s = dlsym(RTLD_NEXT, "pthread_create"); memcpy((void *)&real_pthread_create, &s, sizeof s); }
    if (g_armed.load(std::memory_order_acquire)) {
        g_thr_n.fetch_add(1);
        if (g_gate_mode.load(std::memory_order_acquire) == 2) return real_pthread_create(t, a, boot_tramp, new Boot{ f, arg });
    }
    return real_pthread_create(t, a, f, arg);
}

// ------------------------------------------------------------------------------------------------ unit configuration (a pure function of (scenario, seed))
struct Step {
    char kind = 'H';   // 'H' host cache, 'P' card pool, 'A' refused arguments (NULL pointers, NULL ops, expert_raw 0)
    size_t bytes = 0, raw = 0, reserve = 0, freeb = 0, total = 0;
    int dev = 0, readers = 0, experts = 0;
    std::string file; bool file_null = false;
    long fail_at = 0; int fail_code = 2; bool null_err = false; std::string errtext;
    int mmap_fail_at = 0; bool poison = false;
    bool fake_cache = false; int cache_id = 0; bool cache_null = false;
    uint64_t darena = 0, harena = 0, hnd = 0;
    int gate = 0;                   // host steps: 0 free running, 1 barrier, 2 serial (see the gates above); drawn from its own stream, not part of the configuration line
    bool apool = false; unsigned amask = 0;   // kind 'A': pool or host entry point; the set bits of ARG_HOST / ARG_POOL are the defects of the call
};
struct UCfg { std::string scn; int seed = 0; std::vector<Step> steps; };

static const int FAIL_CODES[] = { 1, 2, 3, 35, 100, 719, 999, 100000, INT_MAX, -1, -5, INT_MIN };
static Step host_step(Rng &R, int fault)   // fault: 0 none, 1 random fault
{
    Step s; s.kind = 'H'; s.cache_id = 0;
    static const size_t raws[] = { 2048, 4096, 8192, 12288, 16384, 32768, 61440, 65536, 65536, 100000, 131072 };
    s.raw = raws[R.r(11)];
    const size_t chunk = 32 * s.raw;
    const size_t nmax = std::min<size_t>(24, ((size_t)48 << 20) / chunk);
    const size_t n = R.p(0.12) ? 0 : (size_t)R.range(1, (long)nmax);
    size_t extra;
    switch (R.r(7)) { case 0: extra = 0; break; case 1: extra = 1; break; case 2: extra = s.raw - 1; break; case 3: extra = s.raw; break; case 4: extra = 31 * s.raw; break; case 5: extra = chunk - 1; break; default: extra = (size_t)R.r(chunk); }
    if (extra >= chunk) extra = chunk - 1;
    s.bytes = n * chunk + extra;
    switch (R.r(5)) { case 0: s.file_null = true; break; case 1: s.file = "decode.cuh"; break; case 2: s.file = "/home/u/src/code/gpu/decode.cuh"; break; case 3: s.file = ""; break; default: s.file = std::string((size_t)R.range(100, 300), 'a') + "%s%n"; }
    s.poison = R.p(0.5);
    s.errtext = R.p(0.5) ? "out of memory" : "100% bad %s %d %n \\ \xc3\xa9";
    s.null_err = R.p(0.04);
    s.fail_code = FAIL_CODES[R.r(sizeof FAIL_CODES / sizeof *FAIL_CODES)];
    if (fault) {
        const int k = (int)R.r(4);
        if (k == 0) s.fail_at = 1;
        else if (k == 1 && n >= 1) s.fail_at = 1 + R.range(1, (long)n);
        else if (k == 2 && n >= 1) s.mmap_fail_at = (int)R.range(1, (long)n);
        else s.fail_at = 1;
    }
    return s;
}
static Step pool_step(Rng &R, int mode, int fault)   // mode 0 ordinary, 1 boundary, 2 the engine's constants
{
    Step s; s.kind = 'P';
    static const int devs[] = { 0, 1, 2, 3, 5, 7, 255, 1000000, -1, INT_MAX };
    static const int readers_l[] = { -5, -1, 0, 1, 2, 3, 16, 31, 32, 33, 40, 1000, INT_MIN, INT_MAX, 16, 8 };
    s.dev = devs[R.r(10)]; s.readers = readers_l[R.r(16)];
    long k;   // slots in the true quotient
    size_t rem = 0;
    if (mode == 2) {
        s.raw = 17547264; s.experts = 896; s.reserve = (size_t)2500 << 20; s.total = (size_t)24 << 30;
        static const long ks[] = { 63, 64, 65, 300, 800, 1104, 1300, 0, 64 };
        k = ks[R.r(9)]; if (R.p(0.5)) s.readers = (int)R.range(1, 40);
    } else {
        static const size_t raws[] = { 1, 64, 4096, 65536, 65536, 100000, 1 << 20 };
        static const int exps[] = { 896, 896, 0, 1, 17, 4096 };
        s.raw = raws[R.r(7)]; s.experts = exps[R.r(6)];
        s.reserve = R.p(0.3) ? 0 : (R.p(0.5) ? ((size_t)1 << 20) : (size_t)R.r(((size_t)4 << 30)));
        s.total = (size_t)R.r(((size_t)1 << 40));
        static const long kb[] = { 62, 63, 63, 64, 64, 65, 66 };
        if (mode == 1 && R.p(0.8)) k = kb[R.r(7)];
        else { static const long ka[] = { 0, 1, 63, 64, 65, 100, 127, 128, 129, 500, 1104, 2999, 4000 }; k = ka[R.r(13)]; }
        if (mode == 1 && s.raw <= (1 << 20) && R.p(0.15)) k += (long)((uint64_t)R.range(1, 3) << 32);   // the slot count wraps in the cast to int
        if (mode == 1 && s.raw > 1 && R.p(0.1)) k = (long)(((uint64_t)1 << 31) + (uint64_t)R.range(0, 4));   // 2^31 + e: negative after the cast
    }
    switch (R.r(4)) { case 0: rem = 0; break; case 1: rem = s.raw > 1 ? s.raw - 1 : 0; break; case 2: rem = (size_t)R.r(s.raw); break; default: rem = 1 % s.raw; }
    s.freeb = s.reserve + (size_t)k * s.raw + rem;
    if (mode != 2 && R.p(0.07)) s.freeb = s.reserve ? (size_t)R.r(s.reserve) : 0;   // free below the reserve (or equal to 0)
    switch (R.r(5)) { case 0: s.file_null = true; break; case 1: s.file = "decode.cuh"; break; case 2: s.file = "/home/u/src/code/gpu/decode.cuh"; break; case 3: s.file = ""; break; default: s.file = std::string((size_t)R.range(100, 300), 'b') + "%s%n"; }
    s.errtext = R.p(0.5) ? "invalid device ordinal" : "100% bad %s %d %n \\ \xc3\xa9";
    s.null_err = R.p(0.04);
    s.fail_code = FAIL_CODES[R.r(sizeof FAIL_CODES / sizeof *FAIL_CODES)];
    if (fault) { const long re = std::max(0, std::min(s.readers, 32)); s.fail_at = R.range(1, 6 + 2 * re + 2); }   // beyond the last call = no failure
    s.darena = 0x700000000000ULL + ((uint64_t)R.r(64) << 32) + 512 * R.r(1000);
    s.harena = 0x7e0000000000ULL + ((uint64_t)R.r(64) << 32) + 64 * R.r(1000);
    s.hnd = 0x1000 + 16 * R.r(100000);
    return s;
}
static const char *ARG_HOST[] = { "ops is NULL", "cfg is NULL", "out is NULL", "ops->set_device is NULL", "ops->host_register is NULL", "ops->error_string is NULL", "ops->sc_add_spare is NULL", "expert_raw is 0" };
static const char *ARG_POOL[] = { "ops is NULL", "cfg is NULL", "out is NULL", "ops->set_device is NULL", "ops->mem_get_info is NULL", "ops->device_malloc is NULL", "ops->host_alloc is NULL",
                                  "ops->stream_create is NULL", "ops->event_create is NULL", "ops->error_string is NULL", "ops->sc_add_spare is NULL", "expert_raw is 0" };
static const int N_ARG_HOST = 8, N_ARG_POOL = 12;
static Step arg_step(Rng &R, int seed)   // seeds 1..20: each single defect of the header's list once; later seeds: one to three defects at once (the first in the library's check order is the one reported)
{
    Step s; s.kind = 'A';
    if (seed >= 1 && seed <= N_ARG_HOST + N_ARG_POOL) { s.apool = seed > N_ARG_HOST; s.amask = 1u << (s.apool ? seed - N_ARG_HOST - 1 : seed - 1); }
    else { s.apool = R.p(0.5); const int nb = s.apool ? N_ARG_POOL : N_ARG_HOST, k = (int)R.range(1, 3); for (int i = 0; i < k; i++) s.amask |= 1u << R.r((uint64_t)nb); }
    s.raw = 65536; s.bytes = 3 * 32 * s.raw;   // valid otherwise: a library that skips a check would really map 3 chunks / make 100 slots
    s.dev = 3; s.reserve = 0; s.freeb = 100 * s.raw; s.total = (size_t)1 << 40; s.readers = 2; s.experts = 896;
    s.file = "decode.cuh";
    s.darena = 0x700000000000ULL; s.harena = 0x7e0000000000ULL; s.hnd = 0x1000;
    return s;
}
static UCfg make_cfg(const std::string &scn, int seed)
{
    UCfg c; c.scn = scn; c.seed = seed;
    Rng R(mix64((uint64_t)seed) ^ fnv(scn));
    if (scn == "H") c.steps.push_back(host_step(R, 0));
    else if (scn == "HX") c.steps.push_back(host_step(R, 1));
    else if (scn == "P") c.steps.push_back(pool_step(R, 0, 0));
    else if (scn == "PX") c.steps.push_back(pool_step(R, 1, R.p(0.5) ? 1 : 0));
    else if (scn == "PR") { c.steps.push_back(pool_step(R, 2, R.p(0.2) ? 1 : 0)); Step h = host_step(R, 0); h.raw = 17547264; h.bytes = (size_t)R.r(32 * h.raw); h.poison = false; c.steps.push_back(h); }
    else if (scn == "A") c.steps.push_back(arg_step(R, seed));
    else {   // M
        const int n = (int)R.range(2, 3);
        for (int i = 0; i < n; i++) {
            const bool host = R.p(0.45);
            Step s = host ? host_step(R, (i == n - 1 && R.p(0.15)) ? 1 : 0) : pool_step(R, (int)R.r(2), (i == n - 1 && R.p(0.15)) ? 1 : 0);
            if (host) { s.bytes = std::min(s.bytes, 16 * 32 * s.raw); }
            c.steps.push_back(s);
        }
    }
    int hp = 0;
    for (Step &s : c.steps) {
        s.fake_cache = R.p(0.25);
        s.cache_id = s.kind == 'H' ? 0 : 1 + (hp++ % 3);
        s.cache_null = s.fake_cache && R.p(0.2);
        if (s.kind == 'H') { s.darena = 0; s.harena = 0; s.hnd = 0; }
    }
    // driver v2: the gate of each host step comes from a stream of its own, so every draw above (the configurations of the 3,000-unit run) is unchanged
    for (size_t i = 0; i < c.steps.size(); i++) {
        Step &s = c.steps[i]; if (s.kind != 'H') continue;
        Rng Q(mix64((uint64_t)seed * 977ULL + i + 12345ULL) ^ fnv(scn) ^ 0x6A7E5ULL);
        const bool serial = Q.p(0.3);
        const size_t n = (s.bytes / s.raw) / 32;
        if (s.mmap_fail_at) continue;   // the failing mmap ends the process from a worker: nothing to hold back
        if (serial && n >= 2) s.gate = 2; else if (n >= 16) s.gate = 1;
    }
    return c;
}
static std::string cfg_str(const UCfg &c)
{
    std::string o = sfmt("CFG scn=%s seed=%d steps=%zu", c.scn.c_str(), c.seed, c.steps.size());
    for (const Step &s : c.steps) {
        if (s.kind == 'H') o += sfmt(" | H bytes=%zu raw=%zu poison=%d fail_at=%ld mmap_fail=%d code=%d nullerr=%d file=%s fake=%d", s.bytes, s.raw, (int)s.poison, s.fail_at, s.mmap_fail_at, s.fail_code, (int)s.null_err, s.file_null ? "NULL" : ("len" + std::to_string(s.file.size())).c_str(), (int)s.fake_cache);
        else if (s.kind == 'A') o += sfmt(" | A pool=%d defects=0x%x", (int)s.apool, s.amask);
        else o += sfmt(" | P dev=%d free=%zu reserve=%zu readers=%d raw=%zu experts=%d fail_at=%ld code=%d nullerr=%d file=%s fake=%d", s.dev, s.freeb, s.reserve, s.readers, s.raw, s.experts, s.fail_at, s.fail_code, (int)s.null_err, s.file_null ? "NULL" : ("len" + std::to_string(s.file.size())).c_str(), (int)s.fake_cache);
    }
    return o;
}

// ------------------------------------------------------------------------------------------------ canonical log
static int g_logfd = -1;
static std::string g_errpath;
static std::string g_buf;
static std::mutex g_logmu;
static void flush_locked()
{
    size_t off = 0;
    while (off < g_buf.size()) { const ssize_t w = write(g_logfd, g_buf.data() + off, g_buf.size() - off); if (w <= 0) _exit(5); off += (size_t)w; }
    g_buf.clear();
}
static void flush_log() { std::lock_guard<std::mutex> g(g_logmu); flush_locked(); }
static void LOG(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void LOG(const char *fmt, ...)
{
    char b[4096]; va_list ap; va_start(ap, fmt); const int n = vsnprintf(b, sizeof b, fmt, ap); va_end(ap);
    std::lock_guard<std::mutex> g(g_logmu);
    g_buf.append(b, (size_t)std::min<int>(n, (int)sizeof b - 1)); g_buf.push_back('\n');
    if (g_buf.size() >= 65536) flush_locked();
}
[[noreturn]] static void viol(const std::string &m)
{
    LOG("VIOLATION %s", m.c_str()); flush_log(); _exit(3);
}

// ------------------------------------------------------------------------------------------------ events: the ordered ops-call record, built by the fakes and by the model
struct Ev {
    char t = 0; long long a = 0, b = 0, c = 0, k = 0, z = 0, al = 0;
    bool operator==(const Ev &o) const { return t == o.t && a == o.a && b == o.b && c == o.c && k == o.k && z == o.z && al == o.al; }
};
static std::string fmt_ev(const Ev &e)
{
    switch (e.t) {
    case 'D': return sfmt("set_device %lld -> %lld", e.a, e.k);
    case 'I': return sfmt("mem_get_info -> free %lld total %lld code %lld", e.a, e.b, e.k);
    case 'M': return sfmt("malloc %lld -> %llx code %lld", e.a, (unsigned long long)e.b, e.k);
    case 'R': return sfmt("register #%lld len %lld flags %lld -> %lld zero %lld aligned %lld", e.a, e.b, e.c, e.k, e.z, e.al);
    case 'A': return sfmt("host_alloc %lld flags %lld -> %llx code %lld", e.a, e.b, (unsigned long long)e.c, e.k);
    case 'S': return sfmt("stream flags %lld -> %llx code %lld", e.a, (unsigned long long)e.b, e.k);
    case 'E': return sfmt("event flags %lld -> %llx code %lld", e.a, (unsigned long long)e.b, e.k);
    default: return sfmt("add_spare cache %lld slot %lld ok %lld", e.a, e.b, e.al);
    }
}
static void log_events(const std::vector<Ev> &ev)   // consecutive spares of one cache with slot numbers going down by one are written as one run (injective)
{
    for (size_t i = 0; i < ev.size();) {
        if (ev[i].t != 'P') { LOG("c %s", fmt_ev(ev[i]).c_str()); i++; continue; }
        size_t j = i + 1;
        while (j < ev.size() && ev[j].t == 'P' && ev[j].a == ev[i].a && ev[j].al == ev[i].al && ev[j].b == ev[j - 1].b - 1) j++;
        LOG("c add_spare x%zu cache %lld slots %lld..%lld ok %lld", j - i, ev[i].a, ev[i].b, ev[j - 1].b, ev[i].al);
        i = j;
    }
}

// ------------------------------------------------------------------------------------------------ the libraries
static struct { sc_cache *(*nw)(); void (*add_spare)(sc_cache *, unsigned char *); size_t (*snap_spare)(const sc_cache *, unsigned char **, size_t); } SC;
static struct { void (*host)(const pi_ops *, const pi_host_cfg *, pi_host_out *); void (*pool)(const pi_ops *, const pi_pool_cfg *, pi_pool_out *); } PI;
static void *must(void *h, const char *lib, const char *name)
{
    void *s = dlsym(h, name);
    if (!s) { fprintf(stderr, "missing symbol %s in %s\n", name, lib); exit(2); }
    return s;
}
#define BIND(st, field, h, lib, name) do { void *s_ = must(h, lib, name); memcpy((void *)&st.field, &s_, sizeof s_); } while (0)
static void load_libs(const char *sc, const char *lib)
{
    void *h = dlopen(sc, RTLD_NOW | RTLD_LOCAL); if (!h) { fprintf(stderr, "dlopen %s: %s\n", sc, dlerror()); exit(2); }
    BIND(SC, nw, h, sc, "sc_new"); BIND(SC, add_spare, h, sc, "sc_add_spare"); BIND(SC, snap_spare, h, sc, "sc_snapshot_spare");
    h = dlopen(lib, RTLD_NOW | RTLD_LOCAL); if (!h) { fprintf(stderr, "dlopen %s: %s\n", lib, dlerror()); exit(2); }
    BIND(PI, host, h, lib, "pi_host_init"); BIND(PI, pool, h, lib, "pi_pool_init");
}

// ------------------------------------------------------------------------------------------------ the fakes (library side) and the state of the step being run
static sc_cache *g_real[4];                                     // real caches: 0 host, 1..3 pools
static std::vector<unsigned char *> g_acc[4];                   // every spare each real cache was given, in order, over the whole unit
static std::atomic<int> g_errcalls{ 0 };                        // calls of the error_string op in the current step
static const size_t EV_CAP = 200000;                            // hard cap of the recorder: a legitimate step makes at most about 4,200 ops / spares calls
static struct Cur {
    Step s;
    std::vector<Ev> events;
    std::vector<uintptr_t> regs;
    uintptr_t slab = 0; bool have_slab = false;
    long ord = 0; unsigned long nspare = 0;
    uint64_t darena = 0, harena = 0, hnd = 0;
    std::thread::id tid;
    std::string errstr;
} X;
static int cache_id_of(sc_cache *c)
{
    for (int i = 0; i < 4; i++) if (c && c == g_real[i]) return i;
    if (!c) return 100;
    return 10 + (int)(((uintptr_t)c - 0x5000) / 16);
}
static void chk_thread(const char *what) { if (std::this_thread::get_id() != X.tid) viol(std::string("op called on a thread other than the caller's: ") + what); }
static void first_op() { uint64_t e = ~0ULL; g_first_op_seq.compare_exchange_strong(e, g_seq.fetch_add(1)); }
static void push_ev(const Ev &e)   // every record of a call goes through here: past the cap the unit fails cleanly instead of growing the recorder (and the real cache) without bound
{
    if (X.events.size() >= EV_CAP) viol(sfmt("event recorder cap: more than %zu ops / spares in one step (a library that makes this many calls does not follow the contract)", EV_CAP));
    X.events.push_back(e);
}
static void vadvance(long long ns) { if (g_vclock.load(std::memory_order_acquire)) g_vns.fetch_add(ns); }
static int fail_code_now() { return (X.s.fail_at && X.ord == X.s.fail_at) ? X.s.fail_code : 0; }
static int f_set_device(int d) { chk_thread("set_device"); first_op(); X.ord++; vadvance(VNS_SETDEV); const int k = fail_code_now(); Ev e; e.t = 'D'; e.a = d; e.k = k; push_ev(e); return k; }
static int f_mem_get_info(size_t *f, size_t *t) { chk_thread("mem_get_info"); first_op(); X.ord++; const int k = fail_code_now(); *f = X.s.freeb; *t = X.s.total; Ev e; e.t = 'I'; e.a = (long long)*f; e.b = (long long)*t; e.k = k; push_ev(e); return k; }
static int f_device_malloc(void **p, size_t n)
{
    chk_thread("device_malloc"); first_op(); X.ord++; const int k = fail_code_now();
    *p = (void *)X.darena; Ev e; e.t = 'M'; e.a = (long long)n; e.b = (long long)X.darena; e.k = k; push_ev(e);
    if (!X.have_slab) { X.have_slab = true; X.slab = X.darena; }
    X.darena += (n + 511) & ~(size_t)511;
    return k;
}
static bool in_mapped(uintptr_t a, size_t n)
{
    const unsigned cnt = std::min<unsigned>(g_mm_n.load(), REC_MAX);
    for (unsigned i = 0; i < cnt; i++) if (g_mm[i].ret != (uintptr_t)MAP_FAILED && a >= g_mm[i].ret && a + n <= g_mm[i].ret + g_mm[i].len) return true;
    return false;
}
static int f_host_register(void *p, size_t n, unsigned fl)
{
    chk_thread("host_register"); first_op(); X.ord++; vadvance(VNS_REGISTER); const int k = fail_code_now();
    Ev e; e.t = 'R'; e.a = (long long)X.regs.size(); e.b = (long long)n; e.c = fl; e.k = k;
    const uintptr_t a = (uintptr_t)p;
    bool ok = a % 4096 == 0;
    for (uintptr_t q : X.regs) ok = ok && (a + n <= q || q + n <= a);
    e.al = ok ? 1 : 0;
    e.z = -1;
    if (X.s.poison) {
        e.z = 0;
        if (in_mapped(a, n)) { const unsigned char *b = (const unsigned char *)p; size_t i = 0; while (i < n && b[i] == 0) i++; e.z = i == n ? 1 : 0; }
    }
    X.regs.push_back(a); push_ev(e);
    return k;
}
static int f_host_alloc(void **p, size_t n, unsigned fl)
{
    chk_thread("host_alloc"); first_op(); X.ord++; const int k = fail_code_now();
    *p = (void *)X.harena; Ev e; e.t = 'A'; e.a = (long long)n; e.b = fl; e.c = (long long)X.harena; e.k = k; push_ev(e);
    X.harena += (n + 63) & ~(size_t)63;
    return k;
}
static int f_stream_create(void **s, unsigned fl) { chk_thread("stream_create"); first_op(); X.ord++; const int k = fail_code_now(); *s = (void *)X.hnd; Ev e; e.t = 'S'; e.a = fl; e.b = (long long)X.hnd; e.k = k; push_ev(e); X.hnd += 16; return k; }
static int f_event_create(void **s, unsigned fl) { chk_thread("event_create"); first_op(); X.ord++; const int k = fail_code_now(); *s = (void *)X.hnd; Ev e; e.t = 'E'; e.a = fl; e.b = (long long)X.hnd; e.k = k; push_ev(e); X.hnd += 16; return k; }
static const char *f_error_string(int c)
{
    chk_thread("error_string");
    g_errcalls.fetch_add(1);
    if (X.s.null_err) return nullptr;
    X.errstr = sfmt("E%d %s", c, X.s.errtext.c_str());
    return X.errstr.c_str();
}
static std::vector<std::pair<int, unsigned char *>> g_spare_log;   // (cache id, slot) of the current step
static void f_add_spare(sc_cache *c, unsigned char *p)
{
    chk_thread("sc_add_spare");
    Ev e; e.t = 'P'; e.a = cache_id_of(c);
    const uintptr_t a = (uintptr_t)p;
    if (X.s.kind == 'H') {
        e.al = 0; e.b = (long long)(a - (X.regs.empty() ? 0 : X.regs[0]));
        for (size_t i = 0; i < X.regs.size(); i++) {
            const uintptr_t r = X.regs[i];
            if (a >= r && a < r + 32 * X.s.raw) { const uintptr_t off = a - r; if (off % X.s.raw == 0 && off / X.s.raw < 32) { e.b = (long long)(i * 32 + off / X.s.raw); e.al = 1; } break; }
        }
    } else {
        e.al = 0; e.b = (long long)(a - X.slab);
        if (X.have_slab && a >= X.slab && (a - X.slab) % X.s.raw == 0) { e.b = (long long)((a - X.slab) / X.s.raw); e.al = 1; }
    }
    push_ev(e);
    if (++X.nspare % 32 == 0) vadvance(VNS_SPARE32);
    const int id = (int)e.a;
    if (id >= 0 && id < 4) g_acc[id].push_back(p);
    if (!X.s.fake_cache && c) SC.add_spare(c, p);
}
static const pi_ops OPS = { f_set_device, f_mem_get_info, f_device_malloc, f_host_register, f_host_alloc, f_stream_create, f_event_create, f_error_string, f_add_spare };

// ------------------------------------------------------------------------------------------------ the model: what the original does, written from decode.cuh@4d8d891
struct Model {
    std::vector<Ev> events;
    bool fatal = false; std::string errtext;   // the whole stderr text of the step when fatal (or normal, with the seconds as T)
    bool host_mmap_fail = false;
    bool via_check = false;                    // the step ends in a failed CHECK: error_string is called exactly once
    pi_pool_out po; pi_host_out ho;
};
static const unsigned long long SENT = 0xA5A5A5A5A5A5A5A5ULL;
static void sentinel_pool_out(pi_pool_out &o)
{
    memset(&o, 0, sizeof o);
    for (int i = 0; i < PI_MAX_READERS; i++) { o.rstream[i] = (void *)SENT; o.bounce[i] = (unsigned char *)SENT; }
    pi_expert_stats st = { 7, 8, 9, 10, 11, 1.5, 2.5, 3.5 }; o.st = st;
}
static std::string check_text(const Step &s, int line, int code)
{
    const std::string file = s.file_null ? std::string("decode.cuh") : s.file;
    return file + sfmt(":%d: ", line) + (s.null_err ? std::string("(null)") : sfmt("E%d %s", code, s.errtext.c_str())) + "\n";
}
static Model model_host(const Step &s)
{
    Model m; memset(&m.ho, 0, sizeof m.ho);
    const size_t cap0 = s.bytes / s.raw, chunk = 32 * s.raw, n = cap0 / 32;
    m.ho.cap_slots = cap0; m.ho.ready = 1;
    if (s.mmap_fail_at >= 1 && (size_t)s.mmap_fail_at <= n) { m.fatal = true; m.host_mmap_fail = true; m.errtext = sfmt("host expert cache: cannot map %zu bytes\n", chunk); return m; }
    long ord = 0;
    auto op = [&](Ev e) { ord++; e.k = (s.fail_at && ord == s.fail_at) ? s.fail_code : 0; m.events.push_back(e); return (int)e.k; };
    { Ev e; e.t = 'D'; e.a = 0; if (op(e)) { m.fatal = true; m.via_check = true; m.errtext = check_text(s, 428, (int)m.events.back().k); return m; } }
    for (size_t i = 0; i < n; i++) {
        Ev e; e.t = 'R'; e.a = (long long)i; e.b = (long long)chunk; e.c = 1; e.z = s.poison ? 1 : -1; e.al = 1;
        if (op(e)) { m.fatal = true; m.via_check = true; m.errtext = check_text(s, 429, (int)m.events.back().k); return m; }
    }
    for (size_t i = n; i-- > 0;) for (int sl = 32; sl-- > 0;) { Ev e; e.t = 'P'; e.a = s.cache_null ? 100 : (s.fake_cache ? 10 + s.cache_id : s.cache_id); e.b = (long long)(i * 32 + (size_t)sl); e.al = 1; m.events.push_back(e); }
    m.ho.cap_slots = n * 32;
    m.errtext = sfmt("K3GPU: host expert cache of %zu slots (%.1f GB) pinned in T s\n", n * 32, (double)(n * 32) * (double)s.raw / 1e9);
    return m;
}
static Model model_pool(const Step &s)
{
    Model m; sentinel_pool_out(m.po);
    long ord = 0; uint64_t darena = s.darena, harena = s.harena, hnd = s.hnd;
    auto op = [&](Ev e) { ord++; e.k = (s.fail_at && ord == s.fail_at) ? s.fail_code : 0; m.events.push_back(e); return (int)e.k; };
    auto bad = [&](int line) { m.fatal = true; m.via_check = true; m.errtext = check_text(s, line, (int)m.events.back().k); };
    const long long cache = s.cache_null ? 100 : (s.fake_cache ? 10 + (long long)s.cache_id : s.cache_id);
    m.po.dev = s.dev; m.po.readers = std::min(s.readers, 32);
    { Ev e; e.t = 'D'; e.a = s.dev; if (op(e)) { bad(454); return m; } }
    { Ev e; e.t = 'I'; e.a = (long long)s.freeb; e.b = (long long)s.total; if (op(e)) { bad(455); return m; } }
    const size_t usable = s.freeb > s.reserve ? s.freeb - s.reserve : 0;
    const int slots = (int)(usable / s.raw);
    m.po.slots = slots;
    if (slots < 64) { m.fatal = true; m.errtext = sfmt("card %d: only %d expert slots fit after the reserve\n", s.dev, slots); return m; }
    const uint64_t slab = darena;
    { Ev e; e.t = 'M'; e.a = (long long)((size_t)slots * s.raw); e.b = (long long)darena; darena += ((size_t)slots * s.raw + 511) & ~(size_t)511; m.po.base = (unsigned char *)slab; if (op(e)) { bad(459); return m; } }
    for (int i = slots; i-- > 0;) { Ev e; e.t = 'P'; e.a = cache; e.b = i; e.al = 1; m.events.push_back(e); }
    { Ev e; e.t = 'S'; e.a = 1; e.b = (long long)hnd; m.po.copy = (void *)hnd; hnd += 16; if (op(e)) { bad(461); return m; } }
    for (int i = 0; i < m.po.readers; i++) {
        { Ev e; e.t = 'S'; e.a = 1; e.b = (long long)hnd; m.po.rstream[i] = (void *)hnd; hnd += 16; if (op(e)) { bad(463); return m; } }
        { Ev e; e.t = 'A'; e.a = (long long)s.raw; e.b = 0; e.c = (long long)harena; m.po.bounce[i] = (unsigned char *)harena; harena += (s.raw + 63) & ~(size_t)63; if (op(e)) { bad(464); return m; } }
    }
    { Ev e; e.t = 'E'; e.a = 2; e.b = (long long)hnd; m.po.written_back = (void *)hnd; hnd += 16; if (op(e)) { bad(466); return m; } }
    { Ev e; e.t = 'M'; e.a = (long long)((size_t)(long)s.experts * sizeof(void *)); e.b = (long long)darena; m.po.d_eptr = (const unsigned char **)darena; darena += ((size_t)(long)s.experts * sizeof(void *) + 511) & ~(size_t)511; if (op(e)) { bad(467); return m; } }
    m.po.ready = 1;
    return m;
}
static Model model_arg(const Step &s)   // refused before anything else is done: the library's own message, exit(1), no op, no mmap / madvise / thread, the out struct untouched
{
    Model m; sentinel_pool_out(m.po); memset(&m.ho, 0, sizeof m.ho);
    const int nb = s.apool ? N_ARG_POOL : N_ARG_HOST;
    for (int i = 0; i < nb; i++) if (s.amask >> i & 1) { m.fatal = true; m.errtext = std::string("poolinit: ") + (s.apool ? ARG_POOL[i] : ARG_HOST[i]) + "\n"; break; }
    return m;
}

// ------------------------------------------------------------------------------------------------ the step run inside a child
static pi_pool_out G_PO; static pi_host_out G_HO;
static Model g_model;
static bool g_in_call = false;
static int g_step_no = 0;
static std::string mask_time(const std::string &t, bool *ok)   // "... pinned in 12.3 s\n" -> "... pinned in T s\n"; *ok = the number has the shape %.1f
{
    *ok = true;
    const std::string key = "pinned in ";
    const size_t p = t.find(key);
    if (p == std::string::npos) return t;
    size_t q = p + key.size(), d = q;
    while (d < t.size() && t[d] >= '0' && t[d] <= '9') d++;
    const bool shape = d > q && d + 2 < t.size() && t[d] == '.' && t[d + 1] >= '0' && t[d + 1] <= '9' && t.compare(d + 2, std::string::npos, " s\n") == 0;
    if (!shape) { *ok = false; return t; }
    return t.substr(0, q) + "T s\n";
}
// driver v2: fd 2 of the child is one end of a SOCK_SEQPACKET socketpair (non-blocking, so a library that writes a lot never blocks); every write() the library makes on stderr, from the
// C library's fprintf as from Rust's std, arrives as one record, so the number of writes is observable. What arrives is also appended to the err file the parent created (the artefact of v1).
static int g_errsock = -1, g_errfd = -1;
static double g_tcall = 0;                      // wall time (real) of the library call of the current step
static size_t g_nwrites = 0;                    // stderr records since the last drain
static std::string new_stderr()
{
    std::string all; char b[70000]; g_nwrites = 0;
    for (;;) {
        const ssize_t n = recv(g_errsock, b, sizeof b, MSG_DONTWAIT);
        if (n < 0) break;
        g_nwrites++; all.append(b, (size_t)n);
        if (g_errfd >= 0 && n > 0) { const ssize_t w = write(g_errfd, b, (size_t)n); (void)w; }
    }
    return all;
}
static bool is_pool(const Step &s) { return s.kind == 'P' || (s.kind == 'A' && s.apool); }
static bool out_equal_pool(const pi_pool_out &a, const pi_pool_out &b)
{
    if (a.dev != b.dev || a.slots != b.slots || a.readers != b.readers || a.ready != b.ready || a.base != b.base || a.copy != b.copy || a.written_back != b.written_back || a.d_eptr != b.d_eptr) return false;
    for (int i = 0; i < PI_MAX_READERS; i++) if (a.rstream[i] != b.rstream[i] || a.bounce[i] != b.bounce[i]) return false;
    return !memcmp(&a.st, &b.st, sizeof a.st);
}
static bool st_untouched(const pi_pool_out &o) { pi_pool_out t; sentinel_pool_out(t); return !memcmp(&o.st, &t.st, sizeof o.st); }
static void log_out_pool(const pi_pool_out &o)
{
    LOG("OUT dev=%d slots=%d readers=%d ready=%d base=%llx copy=%llx written_back=%llx d_eptr=%llx", o.dev, o.slots, o.readers, o.ready, (unsigned long long)(uintptr_t)o.base, (unsigned long long)(uintptr_t)o.copy, (unsigned long long)(uintptr_t)o.written_back, (unsigned long long)(uintptr_t)o.d_eptr);
    unsigned long long hr = 1469598103934665603ULL, hb = 1469598103934665603ULL; int nr = 0, nb = 0;
    for (int i = 0; i < PI_MAX_READERS; i++) {
        hr = (hr ^ (unsigned long long)(uintptr_t)o.rstream[i]) * 1099511628211ULL; hb = (hb ^ (unsigned long long)(uintptr_t)o.bounce[i]) * 1099511628211ULL;
        nr += o.rstream[i] != (void *)SENT; nb += o.bounce[i] != (unsigned char *)SENT;
    }
    LOG("OUT rstream_set=%d bounce_set=%d rstream_hash=%016llx bounce_hash=%016llx st_unchanged=%d", nr, nb, hr, hb, (int)st_untouched(o));
}
static void check_seconds(const std::string &raw, size_t n_chunks)   // the printed seconds lie between what the virtual clock added and that plus the real wall time of the call
{
    const size_t p = raw.find("pinned in ");
    if (p == std::string::npos) return;
    const double secs = atof(raw.c_str() + p + 10), v = (double)expected_vns(n_chunks) / 1e9;
    auto r1 = [](double x) { char b[64]; snprintf(b, sizeof b, "%.1f", x); return atof(b); };
    const double lo = r1(v), hi = r1(v + g_tcall);
    if (secs < lo - 1e-9 || secs > hi + 1e-9)
        viol(sfmt("the seconds of the summary line: printed %.1f s, expected %.1f to %.1f s (virtual clock %.1f s for %zu chunks, real wall time of the call %.3f s)", secs, lo, hi, v, n_chunks, g_tcall));
}
static void compare_step(const char *where)
{
    // events
    if (X.events.size() != g_model.events.size()) viol(sfmt("%s: %zu ops/spares recorded, the model expects %zu", where, X.events.size(), g_model.events.size()));
    for (size_t i = 0; i < X.events.size(); i++)
        if (!(X.events[i] == g_model.events[i])) viol(sfmt("%s: call %zu is [%s], the model expects [%s]", where, i, fmt_ev(X.events[i]).c_str(), fmt_ev(g_model.events[i]).c_str()));
    // out struct
    if (is_pool(X.s)) { if (!out_equal_pool(G_PO, g_model.po)) viol(std::string(where) + ": the out struct differs from the model"); }
    else if (G_HO.cap_slots != g_model.ho.cap_slots || G_HO.ready != g_model.ho.ready) viol(sfmt("%s: host out cap_slots %zu ready %d, model %zu %d", where, G_HO.cap_slots, G_HO.ready, g_model.ho.cap_slots, g_model.ho.ready));
    // stderr
    bool shape = true;
    const std::string raw_err = new_stderr();
    const size_t nwrites = g_nwrites;
    const std::string got = mask_time(raw_err, &shape);
    if (!shape) viol("the seconds in the host cache line do not have the shape %.1f: " + esc(got));
    if (got != g_model.errtext) viol(std::string(where) + ": stderr [" + esc(got) + "], the model expects [" + esc(g_model.errtext) + "]");
    LOG("STDERR %s", esc(got).c_str());
    // driver v2: how the text was produced and what the seconds are. The original's fprintf on the unbuffered stderr makes one write() for a short message, but glibc 2.39 flushes about every 128
    // bytes (measured: a 100-byte file name gives 1 write, 110 gives 2, 240 gives 3), so "one write per message" is the original's behaviour only for a text up to 100 bytes; longer texts: at least one.
    const bool short_msg = g_model.errtext.size() <= 100;
    if (g_model.errtext.empty() ? nwrites != 0 : (short_msg ? nwrites != 1 : nwrites < 1))
        viol(sfmt("%s: the stderr text (%zu bytes) was written with %zu write() calls, expected %s", where, g_model.errtext.size(), nwrites, g_model.errtext.empty() ? "none" : (short_msg ? "one (a text up to 100 bytes)" : "at least one")));
    const int want_e = g_model.via_check ? 1 : 0;
    if (g_errcalls.load() != want_e) viol(sfmt("%s: error_string was called %d times, %d expected (once per failed CHECK, otherwise never)", where, g_errcalls.load(), want_e));
    if (X.s.kind == 'H' && !g_model.fatal) check_seconds(raw_err, (X.s.bytes / X.s.raw) / 32);
}
static void check_gate(const Step &s)   // the gates of driver v2 (valid at the return and at an exit after the mapping phase: every mmap has been made before the first op)
{
    if (s.kind != 'H' || !s.gate) return;
    const unsigned nm = std::min<unsigned>(g_mm_n.load(), REC_MAX);
    const long caller = g_caller_tid.load();
    if (g_gate_timeout.load())
        viol(sfmt("gate %d (%s) timed out after %.0f s: %u threads had reached mmap (%u chunks); %s", s.gate, s.gate == 1 ? "barrier of the 16 workers" : "helpers held until the calling thread has mapped every chunk", GATE_TIMEOUT_S, g_gate_arrived.load(), g_gate_n.load(),
                  s.gate == 1 ? "the calling thread did not take part in the mapping" : "the calling thread did not map all the chunks"));
    if (s.gate == 1) {
        std::vector<long> t; for (unsigned i = 0; i < nm; i++) t.push_back(g_mm[i].tid);
        std::sort(t.begin(), t.end()); t.erase(std::unique(t.begin(), t.end()), t.end());
        if (t.size() != 16 || !std::binary_search(t.begin(), t.end(), caller)) viol(sfmt("with %u chunks (>= 16) and all workers waiting in mmap, %zu distinct threads mapped (expect 16, the calling thread among them: %d)", g_gate_n.load(), t.size(), (int)std::binary_search(t.begin(), t.end(), caller)));
    } else {
        for (unsigned i = 0; i < nm; i++) if (g_mm[i].tid != caller) viol(sfmt("serial mode: mmap call %u was made by a helper, but the helpers had not started", i));
        for (size_t i = 0; i < X.regs.size() && i < nm; i++)
            if (X.regs[i] != g_mm[i].ret) viol(sfmt("serial mode: host_register #%zu got %llx but chunk number %zu is mmap call number %zu (%llx): the chunks are registered in a different order from the one they were taken from the counter in", i, (unsigned long long)X.regs[i], i, i + 1, (unsigned long long)g_mm[i].ret));
    }
}
static void final_handler()   // runs at exit(): if the library ended the process inside a call, that has to be what the model says
{
    if (!g_in_call) { flush_log(); return; }
    g_in_call = false;
    g_armed.store(0); g_vclock.store(0);
    if (!g_model.fatal) viol(sfmt("step %d: the library exited the process, the model expects it to return", g_step_no));
    LOG("FATAL step %d", g_step_no);
    if (!g_model.host_mmap_fail) { log_events(X.events); }
    compare_step("fatal exit");
    if (X.s.kind == 'A' && (g_mm_n.load() || g_ma_n.load() || g_thr_n.load())) viol("a refused call made an mmap / madvise / thread");
    if (X.s.kind == 'H' && !g_model.host_mmap_fail) check_gate(X.s);
    if (is_pool(X.s)) log_out_pool(G_PO); else LOG("OUT cap_slots=%zu ready=%d", G_HO.cap_slots, G_HO.ready);
    LOG("EXIT_IN_LIBRARY ok");
    flush_log();
}
static void check_memory(const Step &s)
{
    const size_t n = (s.bytes / s.raw) / 32, chunk = 32 * s.raw;
    const unsigned nm = std::min<unsigned>(g_mm_n.load(), REC_MAX), na = std::min<unsigned>(g_ma_n.load(), REC_MAX);
    bool args = nm == n, set = true, adv = na == n, order = true, mem = true;
    std::vector<uintptr_t> mapped, advised;
    uint64_t maxseq = 0;
    for (unsigned i = 0; i < nm; i++) {
        const MmapRec &r = g_mm[i];
        args = args && r.addr == 0 && r.len == chunk && r.prot == (PROT_READ | PROT_WRITE) && r.flags == (MAP_PRIVATE | MAP_ANONYMOUS) && r.fd == -1 && r.off == 0 && r.ret != (uintptr_t)MAP_FAILED;
        mapped.push_back(r.ret); maxseq = std::max(maxseq, r.seq);
    }
    for (unsigned i = 0; i < na; i++) {
        const MadvRec &r = g_ma[i];
        adv = adv && r.adv == MADV_HUGEPAGE && r.len == chunk && (!s.poison || r.first_byte == 0xA5);
        advised.push_back(r.addr); maxseq = std::max(maxseq, r.seq);
    }
    std::vector<uintptr_t> regs = X.regs;
    std::sort(mapped.begin(), mapped.end()); std::sort(advised.begin(), advised.end()); std::sort(regs.begin(), regs.end());
    set = mapped == regs && advised == regs && std::adjacent_find(regs.begin(), regs.end()) == regs.end();
    order = nm == 0 && na == 0 ? true : maxseq < g_first_op_seq.load();
    const unsigned thr = g_thr_n.load();
    for (uintptr_t a : regs) {
        if (s.poison) { const unsigned char *b = (const unsigned char *)a; for (size_t i = 0; i < chunk; i++) if (b[i]) { mem = false; break; } }
        else {
            const size_t pg = 4096, np = (chunk + pg - 1) / pg; std::vector<unsigned char> v(np);
            if (mincore((void *)a, chunk, v.data()) != 0) mem = false; else for (size_t i = 0; i < np; i++) if (!(v[i] & 1)) { mem = false; break; }
        }
    }
    LOG("MM mmaps=%u madvises=%u threads=%u args_ok=%d set_ok=%d madvise_ok=%d order_ok=%d memory_ok=%d", nm, na, thr, (int)args, (int)set, (int)adv, (int)order, (int)mem);
    if (!(args && set && adv && order && mem && thr == 15 && nm == n && na == n)) viol(sfmt("host memory checks failed: args=%d set=%d madvise=%d order=%d memory=%d threads=%u (expect 15) mmaps=%u madvises=%u (expect %zu)", (int)args, (int)set, (int)adv, (int)order, (int)mem, thr, nm, na, n));
    check_gate(s);
    for (uintptr_t a : regs) syscall(SYS_munmap, a, chunk);
}
static void run_step(const Step &s, int no)
{
    g_step_no = no;
    X = Cur(); X.s = s; X.tid = std::this_thread::get_id();
    X.darena = s.darena; X.harena = s.harena; X.hnd = s.hnd;
    g_mm_n.store(0); g_ma_n.store(0); g_thr_n.store(0); g_mm_calls.store(0); g_first_op_seq.store(~0ULL);
    g_mmap_fail_at.store(s.mmap_fail_at); g_poison.store(s.poison ? 1 : 0);
    g_errcalls.store(0); g_tcall = 0;
    sc_cache *cache = s.cache_null ? nullptr : (s.fake_cache ? (sc_cache *)(uintptr_t)(0x5000 + 16 * s.cache_id) : g_real[s.cache_id]);
    g_model = s.kind == 'H' ? model_host(s) : s.kind == 'A' ? model_arg(s) : model_pool(s);
    const char *file = s.file_null ? nullptr : s.file.c_str();
    LOG("STEP %d kind %c", no, s.kind);
    new_stderr();   // nothing of an earlier step is left in the socket
    g_in_call = true; g_armed.store(1, std::memory_order_release);
    if (s.kind == 'H') {
        pi_host_cfg cfg = { s.bytes, s.raw, cache, file }; memset(&G_HO, 0, sizeof G_HO);
        g_epoch.fetch_add(1); g_gate_arrived.store(0); g_gate_open.store(0); g_gate_timeout.store(0);
        g_gate_need.store(16); g_gate_n.store((unsigned)((s.bytes / s.raw) / 32)); g_caller_tid.store((long)syscall(SYS_gettid));
        g_vns.store(0); g_clk_other.store(0); g_gate_mode.store(s.gate); g_vclock.store(1);
        const double tc0 = real_mono();
        PI.host(&OPS, &cfg, &G_HO);
        g_tcall = real_mono() - tc0;
        g_vclock.store(0); g_gate_mode.store(0);
    } else if (s.kind == 'A') {
        pi_ops ops = OPS; const unsigned m = s.amask;
        if (!s.apool) {
            if (m >> 3 & 1) ops.set_device = nullptr;
            if (m >> 4 & 1) ops.host_register = nullptr;
            if (m >> 5 & 1) ops.error_string = nullptr;
            if (m >> 6 & 1) ops.sc_add_spare = nullptr;
            pi_host_cfg cfg = { s.bytes, (m >> 7 & 1) ? (size_t)0 : s.raw, cache, file }; memset(&G_HO, 0, sizeof G_HO);
            PI.host((m & 1) ? nullptr : &ops, (m >> 1 & 1) ? nullptr : &cfg, (m >> 2 & 1) ? nullptr : &G_HO);
        } else {
            if (m >> 3 & 1) ops.set_device = nullptr;
            if (m >> 4 & 1) ops.mem_get_info = nullptr;
            if (m >> 5 & 1) ops.device_malloc = nullptr;
            if (m >> 6 & 1) ops.host_alloc = nullptr;
            if (m >> 7 & 1) ops.stream_create = nullptr;
            if (m >> 8 & 1) ops.event_create = nullptr;
            if (m >> 9 & 1) ops.error_string = nullptr;
            if (m >> 10 & 1) ops.sc_add_spare = nullptr;
            pi_pool_cfg cfg = { s.dev, s.reserve, s.readers, (m >> 11 & 1) ? (size_t)0 : s.raw, s.experts, cache, file }; sentinel_pool_out(G_PO);
            PI.pool((m & 1) ? nullptr : &ops, (m >> 1 & 1) ? nullptr : &cfg, (m >> 2 & 1) ? nullptr : &G_PO);
        }
    } else {
        pi_pool_cfg cfg = { s.dev, s.reserve, s.readers, s.raw, s.experts, cache, file }; sentinel_pool_out(G_PO);
        PI.pool(&OPS, &cfg, &G_PO);
    }
    g_armed.store(0, std::memory_order_release); g_in_call = false;
    if (g_model.fatal) viol(sfmt("step %d: the library returned, the model expects the process to end", no));
    log_events(X.events);
    compare_step("return");
    if (s.kind == 'P') {
        const unsigned nm = g_mm_n.load(), na = g_ma_n.load(), thr = g_thr_n.load();
        LOG("MM mmaps=%u madvises=%u threads=%u", nm, na, thr);
        if (nm || na || thr) viol("the card pool made an mmap / madvise / thread");
        log_out_pool(G_PO);
    } else {
        check_memory(s);
        LOG("OUT cap_slots=%zu ready=%d", G_HO.cap_slots, G_HO.ready);
    }
    // the real slot caches: their spare vectors hold exactly what the library handed to sc_add_spare, in order
    if (!s.fake_cache && !s.cache_null) {
        const int id = s.cache_id;
        std::vector<unsigned char *> snap(g_acc[id].size() + 1);
        const size_t tot = SC.snap_spare(g_real[id], snap.data(), snap.size());
        bool ok = tot == g_acc[id].size();
        for (size_t i = 0; ok && i < tot; i++) ok = snap[i] == g_acc[id][i];
        LOG("SNAP cache %d spares %zu equal_to_the_adds=%d", id, tot, (int)ok);
        if (!ok) viol("the real cache's spare vector differs from the slots added");
    }
}
static int run_child(const UCfg &cfg, const std::string &lib, const std::string &sc, const std::string &logp, const std::string &errp)
{
    g_logfd = open(logp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (g_logfd < 0) { perror("log"); return 2; }
    g_errpath = errp;
    prctl(PR_SET_DUMPABLE, 0, 0, 0, 0);   // a crashing library must not start the distribution's crash handler (apport, about 8 s per crash); the signal is still delivered
    load_libs(sc.c_str(), lib.c_str());
    for (int i = 0; i < 4; i++) g_real[i] = SC.nw();
    void *s_ = dlsym(RTLD_NEXT, "pthread_create"); memcpy((void *)&real_pthread_create, &s_, sizeof s_);
    {
        int sv[2];
        if (socketpair(AF_UNIX, SOCK_SEQPACKET, 0, sv) != 0) { perror("socketpair"); return 2; }
        fcntl(sv[1], F_SETFL, O_NONBLOCK);
        g_errsock = sv[0]; dup2(sv[1], 2); close(sv[1]);
        g_errfd = open(errp.c_str(), O_WRONLY | O_APPEND);
    }
    atexit(final_handler);
    LOG("%s", cfg_str(cfg).c_str());
    int no = 0;
    for (const Step &s : cfg.steps) run_step(s, ++no);
    LOG("END ok");
    flush_log();
    return 0;
}

// ------------------------------------------------------------------------------------------------ parent: units and runs
struct Opts {
    std::string mode, scn, ref, rust, sc, out, lib, log, err;
    int seed = 0, jobs = 16, keep_raw = 2; bool keep_going = false;
    std::map<std::string, int> n; std::vector<std::string> scns;
};
static std::string g_self;
static void mkdir_p(const std::string &p)
{
    std::string cur;
    for (size_t i = 0; i <= p.size(); i++) { if (i == p.size() || p[i] == '/') { if (!cur.empty()) mkdir(cur.c_str(), 0755); } if (i < p.size()) cur.push_back(p[i]); }
}
static pid_t spawn_child(const Opts &o, const std::string &lib, const std::string &log, const std::string &err)
{
    std::vector<std::string> a = { g_self, "child", "--scn", o.scn, "--seed", std::to_string(o.seed), "--lib", lib, "--sc", o.sc, "--log", log, "--err", err };
    const pid_t p = fork();
    if (p == 0) {
        const int dn = open("/dev/null", O_RDWR); dup2(dn, 0); dup2(dn, 1);
        const int ef = open(err.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644); dup2(ef, 2);
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
static void count_file(const std::string &p, uint64_t &lines, uint64_t &ops)
{
    lines = ops = 0; FILE *f = fopen(p.c_str(), "rb"); if (!f) return;
    char *b = nullptr; size_t cap = 0; ssize_t n;
    while ((n = getline(&b, &cap, f)) > 0) { lines++; if (n > 2 && b[0] == 'c' && b[1] == ' ') ops++; }
    free(b); fclose(f);
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
        FILE *g = fopen(p.c_str(), "rb"); char *buf = nullptr; size_t cap = 0; uint64_t ln = 0; ssize_t n;
        const uint64_t from = line >= 4 ? line - 4 : 0;
        out += sfmt("--- %s: first difference at byte %llu, line %llu\n", p.c_str(), (unsigned long long)off, (unsigned long long)line + 1);
        while ((n = getline(&buf, &cap, g)) > 0) {
            if (ln >= from && ln <= line + 2) { std::string s(buf, (size_t)n); if (s.size() > 400) s = s.substr(0, 400) + "...\n"; out += sfmt("%s%llu: %s", ln == line ? ">>" : "  ", (unsigned long long)ln + 1, s.c_str()); }
            if (ln > line + 2) break;
            ln++;
        }
        free(buf); fclose(g);
    }
    return out;
}
static bool has_violation(const std::string &log) { return read_all(log).find("VIOLATION") != std::string::npos; }

static int do_unit(const Opts &o)
{
    UCfg c = make_cfg(o.scn, o.seed);
    const std::string dir = o.out + "/" + o.scn; mkdir_p(dir);
    const std::string pre = dir + "/u" + std::to_string(o.seed);
    struct R { std::string lib, log, err; pid_t pid; int st; };
    std::vector<R> rs = { { o.ref, pre + ".ref.log", pre + ".ref.err", 0, 0 }, { o.rust, pre + ".rust.log", pre + ".rust.err", 0, 0 } };
    for (auto &r : rs) r.pid = spawn_child(o, r.lib, r.log, r.err);
    for (auto &r : rs) { int st = 0; waitpid(r.pid, &st, 0); r.st = st; }
    bool pass = true; std::string reason = "-";
    uint64_t la = 0, oa = 0, lb = 0, ob = 0;
    count_file(rs[0].log, la, oa); count_file(rs[1].log, lb, ob);
    uint64_t off = 0;
    const bool same = same_files(rs[0].log, rs[1].log, off);
    const std::string s0 = status_str(rs[0].st), s1 = status_str(rs[1].st);
    if (!same) { pass = false; reason = "logs differ"; const std::string d = diff_report(rs[0].log, rs[1].log, off); FILE *f = fopen((pre + ".diff").c_str(), "wb"); if (f) { fwrite(d.data(), 1, d.size(), f); fclose(f); } }
    else if (s0 != s1) { pass = false; reason = "status differs " + s0 + " vs " + s1; }
    else if (s0 != "exit0" && s0 != "exit1") { pass = false; reason = "both children ended with " + s0; }
    else if (has_violation(rs[0].log) || has_violation(rs[1].log)) { pass = false; reason = "VIOLATION in a log"; }
    std::string extra = "ref=" + s0 + " rust=" + s1;
    {
        FILE *f = fopen((pre + ".res").c_str(), "wb");
        if (f) { fprintf(f, "%s %s %d %llu %llu %llu %llu %s %s\n", pass ? "PASS" : "FAIL", o.scn.c_str(), o.seed, (unsigned long long)la, (unsigned long long)oa, (unsigned long long)lb, (unsigned long long)ob, extra.c_str(), reason.c_str()); fclose(f); }
    }
    if (pass && o.seed >= o.keep_raw) for (auto &r : rs) { unlink(r.log.c_str()); unlink(r.err.c_str()); }
    printf("UNIT %s %d %s lines=%llu/%llu %s %s\n", o.scn.c_str(), o.seed, pass ? "PASS" : "FAIL", (unsigned long long)la, (unsigned long long)lb, extra.c_str(), reason.c_str());
    return pass ? 0 : 1;
}

static int do_run(Opts o)
{
    if (o.scns.empty()) o.scns = { "H", "HX", "P", "PX", "PR", "M" };
    std::map<std::string, int> dflt = { { "H", 600 }, { "HX", 400 }, { "P", 600 }, { "PX", 800 }, { "PR", 200 }, { "M", 400 }, { "A", 60 } };   // A (refused arguments) is not in the default list: --scn A
    for (auto &kv : dflt) if (!o.n.count(kv.first)) o.n[kv.first] = kv.second;
    struct Spec { std::string scn; int seed; };
    std::vector<Spec> specs;
    for (auto &s : o.scns) for (int i = 0; i < o.n[s]; i++) specs.push_back({ s, i + 1 });
    mkdir_p(o.out);
    const time_t t0 = time(nullptr);
    std::map<pid_t, Spec> act; size_t next = 0, done = 0; int fails = 0; bool stop = false;
    while (next < specs.size() || !act.empty()) {
        while (!stop && next < specs.size() && (int)act.size() < o.jobs) {
            const Spec sp = specs[next++];
            const pid_t p = fork();
            if (p == 0) {
                std::vector<std::string> a = { g_self, "unit", "--scn", sp.scn, "--seed", std::to_string(sp.seed), "--ref", o.ref, "--rust", o.rust, "--sc", o.sc, "--out", o.out, "--keep-raw", std::to_string(o.keep_raw) };
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
        if (done % 200 == 0) { printf("[%zu/%zu done, %d fail, %lds]\n", done, specs.size(), fails, (long)(time(nullptr) - t0)); fflush(stdout); }
    }
    std::string summary; int rc = fails ? 1 : 0;
    for (auto &s : o.scns) {
        int units = 0, pass = 0, fail = 0, exit1 = 0; unsigned long long la = 0, oa = 0, lb = 0, ob = 0;
        for (int i = 0; i < o.n[s]; i++) {
            FILE *f = fopen((o.out + "/" + s + "/u" + std::to_string(i + 1) + ".res").c_str(), "rb"); if (!f) continue;
            char line[1024]; if (!fgets(line, sizeof line, f)) { fclose(f); continue; } fclose(f);
            char res[16], scn[16]; int sd; unsigned long long a1, a2, b1, b2;
            if (sscanf(line, "%15s %15s %d %llu %llu %llu %llu", res, scn, &sd, &a1, &a2, &b1, &b2) != 7) continue;
            units++; if (!strcmp(res, "PASS")) pass++; else fail++;
            la += a1; oa += a2; lb += b1; ob += b2;
            if (strstr(line, "ref=exit1")) exit1++;
        }
        std::string l = sfmt("%s units=%d pass=%d fail=%d logged_event_lines_ref=%llu logged_event_lines_rust=%llu (lines %llu/%llu) units_ending_in_fatal_exit1=%d", s.c_str(), units, pass, fail, oa, ob, la, lb, exit1);
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
    if (argc < 2) { fprintf(stderr, "usage: pi_driver run|unit|child|cfg --ref LIB --rust LIB --sc LIBSLOTCACHE [--out DIR --n H=600,... --scn H,P --jobs N --keep-going]\n"); return 2; }
    Opts o; o.mode = argv[1];
    for (int i = 2; i < argc; i++) {
        const std::string k = argv[i]; auto val = [&]() -> std::string { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", k.c_str()); exit(2); } return argv[++i]; };
        if (k == "--scn") { const std::string v = val(); if (o.mode == "run") { size_t p = 0; while (p <= v.size()) { size_t q = v.find(',', p); if (q == std::string::npos) q = v.size(); if (q > p) o.scns.push_back(v.substr(p, q - p)); p = q + 1; } } else o.scn = v; }
        else if (k == "--seed") o.seed = atoi(val().c_str());
        else if (k == "--ref") o.ref = val(); else if (k == "--rust") o.rust = val(); else if (k == "--sc") o.sc = val();
        else if (k == "--out") o.out = val(); else if (k == "--lib") o.lib = val(); else if (k == "--log") o.log = val(); else if (k == "--err") o.err = val();
        else if (k == "--jobs") o.jobs = atoi(val().c_str());
        else if (k == "--keep-raw") o.keep_raw = atoi(val().c_str()); else if (k == "--keep-going") o.keep_going = true;
        else if (k == "--n") { const std::string v = val(); size_t p = 0; while (p < v.size()) { size_t q = v.find(',', p); if (q == std::string::npos) q = v.size(); const std::string t = v.substr(p, q - p); const size_t e = t.find('='); if (e != std::string::npos) o.n[t.substr(0, e)] = atoi(t.c_str() + e + 1); p = q + 1; } }
        else { fprintf(stderr, "unknown option %s\n", k.c_str()); return 2; }
    }
    if (o.mode == "child") { const UCfg c = make_cfg(o.scn, o.seed); return run_child(c, o.lib, o.sc, o.log, o.err); }
    if (o.mode == "cfg") { const UCfg c = make_cfg(o.scn, o.seed); std::string g; for (const Step &s : c.steps) if (s.kind == 'H') g += sfmt(" gate=%d", s.gate); printf("%s%s\n", cfg_str(c).c_str(), g.c_str()); return 0; }
    if (o.mode == "unit") return do_unit(o);
    if (o.mode == "run") return do_run(o);
    fprintf(stderr, "unknown mode %s\n", o.mode.c_str()); return 2;
}
