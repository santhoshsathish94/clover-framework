// Differential driver of the card supply (include/supply.h): the C++ reference (ref/libsupply_ref.so, the original text of pool_ensure behind a harness) against the Rust crate
// (libsupply.so), same C ABI. One binary, three modes:
//   child : runs ONE unit (a seeded world and a sequence of sy_ensure calls) against ONE supply library inside this process and writes a canonical log (the library's stderr goes into the same file).
//   unit  : generates the unit's data files, starts a child per library, compares the logs byte for byte, writes one result line.
//   run   : schedules many units over many processes and prints one summary line per scenario.
// Inside a child everything but the supply is real or modelled: the slot caches are the real libslotcache.so (both arms use the same one), the drive reads are real pread calls on files made
// of known bytes, the pinned bounce slots, the card pool slots and the RAM cache slots are real memory. NO CUDA call exists: the sy_ops table is a set of recording fakes; a "copy" is a memcpy
// done when the call is made. The planner's pend_take / arr_wait / host_acquire are modelled (pending and arriving entries have POISON bytes until the call completes them).
// Absolute checks (they hold for BOTH arms independently of each other): every requested expert is on the card with the right bytes; every cache entry that is not pending / arriving holds its
// expert's bytes; counters equal the counts derived from the recorded calls; the set of reader copies equals the set the model derives (own chunk arithmetic); ordering rules between the
// calls (event recorded before a reader waits for it, victim bytes complete when copied down, host slot released only after the final sync, ...).
// Scenarios: Y1 all hits on the card | Y2 all from RAM | Y3 all from the drive | Y4 mix | Y5 write-back / not worth / capacity 0 / NULL host cache / host full of pending entries |
//            Y6 prefill with a slab | Y7 chunking (few experts, many readers, forced chunk counts) | Y8 counts, duplicates, layers and ekey | Y9 reader counts 1..16 |
//            YF failure paths (short file, bad fd, injected CUDA status, slot cache fatal) | T1 real threads (jitter, two runs per arm).
// Driver v2 (blind spots found by the mutation check, verify/mutants/RESULTS.md) adds: Y10 real-size records (offsets above 2^31, chunk cap) | Y11 short reads, EOF, errno from pread |
//            Y12 forced chunk counts | Y13 NULL RAM cache | Y14 a second sy_init | YG injected CUDA status on every op kind, `check` that returns or not | T2 real threads with a barrier
//            (the number of reader threads that start must be min(pool_readers, jobs)). The driver executable defines pread() (built with -rdynamic, so both libraries bind to it): file descriptors it
//            registered are virtual files (bytes generated, seeded short counts, EOF cut, errno), every other fd goes to the kernel. The status given to exit() (EXIT line) and the number of pread()
//            calls and short returns (PREAD line) are part of every log.
#include "supply.h"
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
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <map>
#include <mutex>
#include <set>
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
static uint64_t hash_bytes(const unsigned char *p, size_t n)
{
    uint64_t h = 1469598103934665603ULL; size_t i = 0;
    for (; i + 8 <= n; i += 8) { uint64_t w; memcpy(&w, p + i, 8); h = (h ^ w) * 0x100000001B3ULL; h ^= h >> 29; }
    for (; i < n; i++) h = (h ^ p[i]) * 0x100000001B3ULL;
    return h;
}
// the bytes of a file (or of a slab) are a pure function of (base, byte offset): byte o is byte (o & 7) of mix64(base + (o >> 3))
static void gen_bytes(unsigned char *d, uint64_t base, uint64_t off, size_t n)
{
    for (size_t i = 0; i < n;) {
        const uint64_t o = off + i, w = mix64(base + (o >> 3));
        for (unsigned sh = (unsigned)(o & 7); sh < 8 && i < n; sh++, i++) d[i] = (unsigned char)(w >> (8 * sh));
    }
}
static uint64_t file_base(int fseed) { return mix64(0xF11E0000ULL + (uint64_t)fseed); }
static uint64_t slab_base(int layer) { return mix64(0x51AB0000ULL + (uint64_t)layer); }
static int fseed_of(int layer) { return layer % 4; }

// ------------------------------------------------------------------------------------------------ unit configuration (a pure function of (scenario, seed))
struct UCfg {
    std::string scn; int seed = 0;
    int E = 32; long raw = 8192; int nlayers = 1; int layers[6]; int warm_layers[4]; int nwarm = 0;
    int Np = 16, Nh = 8; long cap = 8; bool host_null = false;
    int policy = 1, install_nu = 0, nu_mode = 0;
    int vq = 0, prefill = 0, chunks = 0, PR = 4, dev = 0;
    int steps = 20, gmin = 1, gmax = 8, dup_pct = 0, pop = 12, rd_lo = 1, rd_hi = 16;
    double p_pool = 0, p_host = 0, p_pend = 0.3, p_arrive = 0.3, p_noise = 0.2;
    bool warm = true, hostfull = false, fd_unused = false, threads = false;
    int fdmode = 0, fail_op = -1, fail_occ = 1;
    // driver v2: virtual files (pread interposer), seeded short reads, EOF cut (percent of the file), errno for the records e with e % err_mod == err_r, `check` that returns, thread barrier,
    // experts drawn from hi_min (the first expert whose offset e * raw is >= 2^31), a second sy_init at step `reinit`
    bool vfile = false, check_ret = false, barrier = false;
    int short_pct = 0, cut_pct = 100, err_mod = 0, err_r = 0, err_no = 0, hi_min = 0, reinit = -1;
    bool ext() const { return vfile || check_ret || barrier || short_pct || cut_pct != 100 || err_mod || hi_min || reinit >= 0; }
    UCfg() { memset(layers, 0, sizeof layers); memset(warm_layers, 0, sizeof warm_layers); }
};
static const int LAYER_POOL[] = { 0, 1, 5, 17, 50, 91, 92 };

static UCfg make_cfg(const std::string &scn, int seed)
{
    UCfg c; c.scn = scn; c.seed = seed;
    Rng R(mix64((uint64_t)seed) ^ fnv(scn));
    auto pk = [&](std::initializer_list<long> l) { return *(l.begin() + R.r((uint32_t)l.size())); };
    auto pkd = [&](std::initializer_list<double> l) { return *(l.begin() + R.r((uint32_t)l.size())); };
    c.raw = pk({ 4096, 5000, 8192, 12345, 20480, 4097, 8192, 20480 });
    c.E = (int)pk({ 17, 32, 64, 64, 17 });
    if (R.p(0.08)) { c.E = 896; c.raw = 4096; }
    c.nlayers = R.range(1, 4);
    { std::vector<int> pool(std::begin(LAYER_POOL), std::end(LAYER_POOL)); for (int i = 0; i < c.nlayers; i++) { const size_t k = R.r((uint32_t)pool.size()); c.layers[i] = pool[k]; pool.erase(pool.begin() + (long)k); } }
    c.nwarm = R.range(1, 3); for (int i = 0; i < c.nwarm; i++) c.warm_layers[i] = (int)R.r(93);
    c.Np = (int)pk({ 6, 10, 16, 24, 40 }); c.Nh = (int)pk({ 8, 12, 30, 80 }); c.cap = c.Nh;
    c.policy = (int)pk({ 0, 1, 2, 2 }); c.install_nu = c.policy == 2 ? (R.p(0.9) ? 1 : 0) : (R.p(0.2) ? 1 : 0); c.nu_mode = (int)R.r(3);
    c.vq = R.p(0.5) ? 1 : 0; c.prefill = R.p(0.08) ? 1 : 0; c.chunks = R.p(0.2) ? (int)pk({ 1, 2, 3, 5, 8, 11, 64 }) : 0;
    c.PR = (int)pk({ 1, 2, 3, 4, 8, 16, 32 }); c.dev = (int)R.r(8);
    c.steps = R.range(15, 45); c.pop = R.range(6, 24); c.dup_pct = (int)pk({ 0, 0, 0, 10 });
    c.p_noise = pkd({ 0.0, 0.2, 0.5 });
    c.fd_unused = R.p(0.5);
    if (c.E == 1) { c.gmax = 1; }
    if (R.p(0.04)) { c.E = 1; }
    auto fix_g = [&]() { c.gmax = std::min(c.gmax, std::min(c.Np, c.E)); c.gmin = std::min(c.gmin, c.gmax); };
    if (scn == "Y1") { c.p_pool = pkd({ 1.0, 1.0, 0.9 }); c.p_arrive = pkd({ 0.0, 0.5, 1.0 }); c.gmax = c.Np; }
    else if (scn == "Y2") { c.p_host = pkd({ 1.0, 1.0, 0.95 }); c.p_pend = pkd({ 0.0, 0.5, 1.0 }); c.Nh = (int)pk({ 30, 80 }); c.cap = c.Nh; c.gmax = std::min(c.Np, 12); }
    else if (scn == "Y3") { c.gmax = c.Np; c.p_noise = pkd({ 0.0, 0.2 }); }
    else if (scn == "Y4") { c.p_pool = 0.3; c.p_host = 0.3; c.gmax = c.Np; }
    else if (scn == "Y5") {
        c.policy = R.p(0.8) ? 2 : (int)R.r(2); c.install_nu = c.policy == 2 ? 1 : 0; c.nu_mode = (int)R.r(3); c.Nh = (int)pk({ 2, 4, 8, 30 }); c.cap = c.Nh;
        c.Np = (int)pk({ 6, 10, 16 }); c.p_pool = 0.2; c.p_host = 0.2; c.p_noise = 0.5; c.gmax = c.Np; c.prefill = R.p(0.05) ? 1 : 0;
        const double w = R.p(0.5) ? 0.0 : 1.0;
        if (w > 0) { const int k = (int)R.r(10); if (k < 2) c.cap = 0; else if (k < 4) { c.host_null = true; c.cap = 0; } else if (k < 5) c.hostfull = true; }
    }
    else if (scn == "Y6") { c.prefill = 1; c.gmax = c.Np; c.p_pool = 0.2; c.p_host = 0.3; }
    else if (scn == "Y7") { c.gmin = 1; c.gmax = 4; c.PR = (int)pk({ 4, 8, 16, 32 }); c.rd_lo = 4; c.rd_hi = 16; c.p_host = 0.1; c.chunks = R.p(0.3) ? (int)pk({ 1, 2, 3, 5, 8, 11, 64 }) : 0; }
    else if (scn == "Y8") { c.dup_pct = (int)pk({ 30, 40 }); c.p_pool = 0.3; c.p_host = 0.2; c.gmax = c.Np; if (R.p(0.3)) { c.E = (int)pk({ 1, 17, 896 }); if (c.E == 896) c.raw = 4096; } }
    else if (scn == "Y9") { c.gmax = c.Np; c.p_pool = 0.2; c.p_host = 0.2; c.rd_lo = -2; c.rd_hi = 16; }   // reader counts -2..16: 0 and negative run everything on the calling thread
    else if (scn == "YF") {
        c.PR = 1; c.steps = R.range(8, 25); c.gmax = c.Np; c.p_pool = 0.2; c.p_host = 0.2; c.p_noise = 0.3; c.rd_hi = 16;
        const int k = (int)R.r(10);
        if (k < 3) c.fdmode = 1; else if (k < 5) c.fdmode = 2; else if (k < 8) { c.fail_op = (int)R.r(10); c.fail_occ = R.range(1, 3); } else c.hostfull = true;
        c.prefill = c.fdmode ? 0 : c.prefill;
    }
    else if (scn == "Y10") {   // real-size records: the file offset e * raw exceeds 2^31 (a 32-bit product wraps), and 8 / 7 / 9 chunks of one record differ in size
        const bool real = R.p(0.25);
        c.vfile = true; c.prefill = 0; c.fd_unused = false; c.fdmode = 0; c.fail_op = -1; c.raw = real ? 17547264L : 4198401L; c.E = (int)pk({ 896, 896, 700 });
        c.Np = real ? 4 : (int)pk({ 4, 6, 10 }); c.Nh = c.Np + 8; c.cap = c.Nh; c.PR = real ? 16 : (int)pk({ 16, 32 }); c.rd_lo = 2; c.rd_hi = 32;
        c.gmin = 1; c.gmax = 2; c.steps = R.range(8, real ? 12 : 20); c.p_pool = 0.1; c.p_host = 0.05; c.p_noise = 0.1; c.dup_pct = 0; c.chunks = R.p(0.15) ? (int)pk({ 1, 2, 5 }) : 0;
        c.hi_min = (int)((2147483648LL + c.raw - 1) / c.raw);
    }
    else if (scn == "Y11") {   // short reads (seeded counts), then, with one reader thread: EOF in the middle of the files or an errno for some records
        c.vfile = true; c.prefill = 0; c.fd_unused = false; c.fdmode = 0; c.short_pct = (int)pk({ 30, 60, 90 }); c.gmax = c.Np; c.p_pool = 0.2; c.p_host = 0.2;
        const int k = (int)R.r(10);
        if (k >= 5) {
            c.PR = 1; c.rd_lo = 1; c.rd_hi = 16;
            if (k < 8) c.cut_pct = (int)pk({ 30, 50, 70 }); else { c.err_mod = (int)pk({ 8, 16 }); c.err_r = (int)R.r((uint32_t)c.err_mod); c.err_no = (int)pk({ EIO, EINVAL, ENOMEM, EBADF }); }
        }
    }
    else if (scn == "Y12") {   // forced chunk counts, mostly 1, with few experts and many readers (the computed count would be above 1)
        c.gmin = 1; c.gmax = 4; c.PR = (int)pk({ 4, 8, 16, 32 }); c.rd_lo = 4; c.rd_hi = 16; c.p_host = 0.1; c.chunks = R.p(0.7) ? 1 : (int)pk({ 0, 2, 3 });
    }
    else if (scn == "Y13") {   // NULL RAM cache: allowed with capacity 0, or with prefill
        c.host_null = true; c.gmax = c.Np; c.p_pool = 0.3; c.p_host = 0.3;
        if (R.p(0.5)) c.cap = 0; else c.prefill = 1;
    }
    else if (scn == "Y14") {   // a second sy_init in the middle of the unit: the g_wb_skipped counter starts again at 0
        c.policy = 2; c.install_nu = 1; c.nu_mode = 1; c.prefill = 0; c.Np = (int)pk({ 6, 10 }); c.p_pool = 0.2; c.p_host = 0.2; c.p_noise = 0.5; c.gmax = c.Np; c.steps = R.range(20, 40); c.reinit = c.steps / 2;
    }
    else if (scn == "YG") {   // injected CUDA status: every op kind in turn (seed % 10), occurrence 1..3, one reader thread; `check` exits or returns
        c.PR = 1; c.prefill = 0; c.policy = (int)pk({ 0, 1, 1 }); c.install_nu = 0; c.Np = (int)pk({ 6, 10 }); c.steps = R.range(12, 30); c.gmax = c.Np; c.p_pool = 0.2; c.p_host = 0.4; c.p_noise = 0.3;
        c.rd_hi = 16; c.fdmode = 0; c.fail_op = seed % 10; c.fail_occ = R.range(1, 3); c.check_ret = R.p(0.5);
    }
    else if (scn == "T1" || scn == "T2") {   // T2 is T1 with a barrier: the first min(pool_readers, jobs) reader jobs of a call all start before one of them goes on
        c.barrier = scn == "T2";
        c.threads = true; c.PR = (int)pk({ 8, 16, 32 }); c.rd_lo = 8; c.rd_hi = 16; c.Np = (int)pk({ 24, 40 }); c.E = 64; c.raw = pk({ 20480, 12345, 8192 }); c.steps = R.range(12, 30);
        c.gmin = c.Np / 2; c.gmax = c.Np; c.p_host = 0.2; c.p_pool = 0.1; c.prefill = R.p(0.1) ? 1 : 0; c.dup_pct = 0; c.Nh = 80; c.cap = 80; c.chunks = R.p(0.3) ? (int)pk({ 2, 3, 8 }) : 0;
    }
    if (c.E < 1) c.E = 1;
    // the RAM cache must hold the experts one call takes out of it plus the pending ones plus room to write down into, or the slot cache's fatal exit ends the unit early (Y5 and the host-full units do that on purpose)
    if (scn != "Y5") { const bool had_cap = c.cap > 0; c.Nh = std::max(c.Nh, c.Np + 8); if (had_cap) c.cap = c.Nh; }
    fix_g();
    if (c.gmax < 1) c.gmax = 1;
    return c;
}

static std::string cfg_str(const UCfg &c)
{
    std::string s = sfmt("CFG scn=%s seed=%d E=%d raw=%ld layers=%d Np=%d Nh=%d cap=%ld hostnull=%d policy=%d nu=%d/%d vq=%d prefill=%d chunks=%d PR=%d dev=%d steps=%d g=%d..%d dup=%d pool=%.2f host=%.2f pend=%.2f arr=%.2f noise=%.2f "
                         "hostfull=%d fdunused=%d thr=%d fdmode=%d fail=%d/%d rd=%d..%d",
                         c.scn.c_str(), c.seed, c.E, c.raw, c.nlayers, c.Np, c.Nh, c.cap, (int)c.host_null, c.policy, c.install_nu, c.nu_mode, c.vq, c.prefill, c.chunks, c.PR, c.dev, c.steps, c.gmin, c.gmax, c.dup_pct,
                         c.p_pool, c.p_host, c.p_pend, c.p_arrive, c.p_noise, (int)c.hostfull, (int)c.fd_unused, (int)c.threads, c.fdmode, c.fail_op, c.fail_occ, c.rd_lo, c.rd_hi);
    if (c.ext()) s += sfmt(" vfile=%d short=%d cut=%d err=%d/%d/%d chk=%d bar=%d hi=%d reinit=%d", (int)c.vfile, c.short_pct, c.cut_pct, c.err_mod, c.err_r, c.err_no, (int)c.check_ret, (int)c.barrier, c.hi_min, c.reinit);
    return s;
}

// data files: layer L reads f<L % 4>_<E>_<raw>.bin (E * raw bytes of file_base(L % 4)); s<..> is the same cut at 40 % for the short-file failure
static std::string data_path(const std::string &dir, char kind, int fs, const UCfg &c) { return sfmt("%s/%c%d_%d_%ld.bin", dir.c_str(), kind, fs, c.E, c.raw); }
static void mkdir_p(const std::string &p)
{
    std::string cur;
    for (size_t i = 0; i <= p.size(); i++) { if (i == p.size() || p[i] == '/') { if (!cur.empty()) mkdir(cur.c_str(), 0755); } if (i < p.size()) cur.push_back(p[i]); }
}
static void make_file(const std::string &path, int fs, uint64_t bytes)
{
    struct stat st; if (stat(path.c_str(), &st) == 0 && (uint64_t)st.st_size == bytes) return;
    const std::string tmp = path + sfmt(".tmp%d", (int)getpid());
    FILE *f = fopen(tmp.c_str(), "wb"); if (!f) { perror("data file"); exit(2); }
    std::vector<unsigned char> b(1 << 16);
    for (uint64_t off = 0; off < bytes; off += b.size()) { const size_t n = (size_t)std::min<uint64_t>(b.size(), bytes - off); gen_bytes(b.data(), file_base(fs), off, n); fwrite(b.data(), 1, n, f); }
    fclose(f); rename(tmp.c_str(), path.c_str());
}
static void make_data(const std::string &dir, const UCfg &c)
{
    if (c.vfile) return;   // no files: the pread interposer generates the bytes
    mkdir_p(dir);
    for (int i = 0; i < c.nlayers; i++) {
        const int fs = fseed_of(c.layers[i]);
        make_file(data_path(dir, 'f', fs, c), fs, (uint64_t)c.E * (uint64_t)c.raw);
        if (c.fdmode == 1) make_file(data_path(dir, 's', fs, c), fs, (uint64_t)c.E * (uint64_t)c.raw * 4 / 10);
    }
}

// ------------------------------------------------------------------------------------------------ canonical log
static int g_logfd = -1;
static std::string g_buf;
static std::mutex g_logmu;
static std::atomic<unsigned long> g_pread_calls{ 0 }, g_pread_short{ 0 };   // pread() calls of the unit and how many returned fewer bytes than the file had to give (see the interposer below); logged at exit
static bool g_sync_log = false;   // flush after every line (failure units), so a message written straight to fd 2 lands in chronological order
// coverage: 0 steps, 1 card hits, 2 acquires, 3 RAM hits, 4 write-backs, 5 not worth, 6 arr_wait that completed an arrival, 7 pend_take that completed a read, 8 reader copies, 9 steps with > 1 chunk per expert, 10 steps that needed nothing, 11 steps with a duplicate expert
static unsigned long g_cov[12]; static std::string g_covpath;
static unsigned long g_nlines = 0, g_nz = 0;
static void flush_locked()
{
    size_t off = 0;
    while (off < g_buf.size()) { const ssize_t w = write(g_logfd, g_buf.data() + off, g_buf.size() - off); if (w <= 0) _exit(5); off += (size_t)w; }
    g_buf.clear();
}
static void flush_log() { std::lock_guard<std::mutex> g(g_logmu); flush_locked(); }
static void logs(const std::string &s, bool z)
{
    std::lock_guard<std::mutex> g(g_logmu);
    g_buf += s; g_buf.push_back('\n'); g_nlines++; if (z) g_nz++;
    if (g_sync_log || g_buf.size() >= 65536) flush_locked();   // flush points depend only on the content, so they are the same in both children
}
static void LOG(const std::string &s) { logs(s, false); }
static void LOGZ(const std::string &s) { logs(s, true); }
static void viol(const std::string &m)
{
    LOG("VIOLATION " + m); flush_log(); exit(3);
}
// registered with on_exit: `status` is what exit() was given by whoever called it (the library's fatal path, the fake `check`, a violation, main's return). It is written into the log (EXIT line) and is the
// status the child leaves with, so a library that exits with 2 where the original exits with 1 shows in the log and in the parent's status comparison.
static void at_exit_flush(int status, void *)
{
    {
        std::lock_guard<std::mutex> g(g_logmu);
        g_buf += sfmt("PREAD calls=%lu short=%lu\nEXIT %d\n", g_pread_calls.load(), g_pread_short.load(), status); g_nlines += 2;
        flush_locked();
    }
    if (!g_covpath.empty()) {   // coverage counters of the unit (not compared between the arms; the parent adds them up)
        FILE *f = fopen(g_covpath.c_str(), "wb");
        if (f) { for (int i = 0; i < 12; i++) fprintf(f, "%lu ", g_cov[i]); fprintf(f, "\n"); fclose(f); }
    }
    _exit(status);
}
static void on_alarm(int)
{
    if (g_logfd >= 0 && !g_buf.empty()) { const ssize_t w = write(g_logfd, g_buf.data(), g_buf.size()); (void)w; }
    static const char m[] = "TIMEOUT\n"; const ssize_t w = write(g_logfd >= 0 ? g_logfd : 2, m, sizeof m - 1); (void)w;
    _exit(4);
}

// ------------------------------------------------------------------------------------------------ the libraries
static struct {
    void (*set_policy)(int); void (*set_next_use)(sc_next_use_fn); sc_cache *(*nw)(); void (*add_spare)(sc_cache *, unsigned char *); void (*release)(sc_cache *, unsigned char *);
    int (*contains)(const sc_cache *, int); unsigned char *(*find)(sc_cache *, int); void (*touch)(sc_cache *, int, float); int (*worth)(const sc_cache *, int);
    unsigned char *(*acquire)(sc_cache *, int, const int *, size_t, float, int *, float *); unsigned char *(*remove)(sc_cache *, int);
    size_t (*snap_lru)(const sc_cache *, sc_entry *, size_t); size_t (*snap_spare)(const sc_cache *, unsigned char **, size_t);
} SC;
static struct { void (*init)(const sy_ops *, const sy_config *); void (*ensure)(sy_pool *, sy_host *, int, int, const int *, size_t, const int *, int); unsigned long (*wbs)(void); } SY;
static bool g_symfail = false;
static void *must(void *h, const char *lib, const char *name)
{
    void *s = dlsym(h, name);
    if (!s) { fprintf(stderr, "missing symbol %s in %s\n", name, lib); g_symfail = true; }
    return s;
}
#define BIND(st, field, h, lib, name) do { void *s_ = must(h, lib, name); memcpy((void *)&st.field, &s_, sizeof s_); } while (0)
static void load_libs(const char *sc, const char *sy)
{
    void *h = dlopen(sc, RTLD_NOW | RTLD_LOCAL); if (!h) { fprintf(stderr, "dlopen %s: %s\n", sc, dlerror()); exit(2); }
    BIND(SC, set_policy, h, sc, "sc_set_policy"); BIND(SC, set_next_use, h, sc, "sc_set_next_use"); BIND(SC, nw, h, sc, "sc_new"); BIND(SC, add_spare, h, sc, "sc_add_spare");
    BIND(SC, release, h, sc, "sc_release"); BIND(SC, contains, h, sc, "sc_contains"); BIND(SC, find, h, sc, "sc_find"); BIND(SC, touch, h, sc, "sc_touch"); BIND(SC, worth, h, sc, "sc_worth");
    BIND(SC, acquire, h, sc, "sc_acquire"); BIND(SC, remove, h, sc, "sc_remove"); BIND(SC, snap_lru, h, sc, "sc_snapshot_lru"); BIND(SC, snap_spare, h, sc, "sc_snapshot_spare");
    h = dlopen(sy, RTLD_NOW | RTLD_LOCAL); if (!h) { fprintf(stderr, "dlopen %s: %s\n", sy, dlerror()); exit(2); }
    BIND(SY, init, h, sy, "sy_init"); BIND(SY, ensure, h, sy, "sy_ensure"); BIND(SY, wbs, h, sy, "sy_wb_skipped");
    if (g_symfail) exit(2);
}

// ------------------------------------------------------------------------------------------------ the world
static UCfg C;
static Rng RD;
static sc_cache *g_pc = nullptr, *g_hc = nullptr;
static unsigned char *g_parena, *g_harena, *g_bounce[SY_MAX_READERS], *g_slab[93];
static const unsigned char *g_slab_tab[93];
static int g_fd[93];
static char g_copy, g_rs[SY_MAX_READERS], g_ev;
static sy_pool g_pool; static sy_host g_host; static std::vector<const unsigned char *> g_eptr;
static unsigned long long g_nuc = 0; static int g_clock = 0;

// ---- pread: defined by this executable (linked with -rdynamic), so the supply libraries, which import pread, bind to it. A descriptor registered in g_vf is a VIRTUAL file: its bytes are generated
// (gen_bytes of the same base as the real data files), the size may be cut (EOF), reads may return seeded short counts (a function of unit seed, offset and length only, so both arms and every thread
// order see the same counts), and the records e % err_mod == err_r fail with errno err_no. A negative offset is EINVAL, as for a regular file. Any other descriptor goes to the kernel.
struct VFile { bool on = false; uint64_t base = 0, size = 0; int err_mod = 0, err_r = 0, err_no = 0; };
static VFile g_vf[4096];
extern "C" ssize_t pread(int fd, void *buf, size_t n, off_t off)
{
    const unsigned long calls = ++g_pread_calls;
    if (calls > 400000) viol(sfmt("pread called %lu times: a read loop that does not end", calls));
    if (fd < 0 || fd >= 4096 || !g_vf[fd].on) return (ssize_t)syscall(SYS_pread64, fd, buf, n, off);
    const VFile &f = g_vf[fd];
    if (off < 0) { errno = EINVAL; return -1; }
    const uint64_t o = (uint64_t)off;
    if (f.err_mod && (int)((o / (uint64_t)C.raw) % (uint64_t)f.err_mod) == f.err_r) { errno = f.err_no; return -1; }
    if (n == 0 || o >= f.size) return 0;
    size_t r = (size_t)std::min<uint64_t>(n, f.size - o);
    if (C.short_pct) {
        const uint64_t h = mix64((uint64_t)C.seed * 0x9E3779B97F4A7C15ULL ^ (o * 0xD6E8FEB86659FD93ULL) ^ ((uint64_t)n << 20));
        if ((int)(h % 100) < C.short_pct) { const size_t s = 1 + (size_t)((h >> 8) % r); if (s < r) g_pread_short++; r = s; }
    }
    gen_bytes((unsigned char *)buf, f.base, o, r);
    return (ssize_t)r;
}

static std::string idname(const void *q)
{
    const unsigned char *p = (const unsigned char *)q;
    if (!p) return "-";
    const size_t raw = (size_t)C.raw;
    if (p >= g_parena && p < g_parena + (size_t)C.Np * raw) { const size_t o = (size_t)(p - g_parena); return o % raw ? sfmt("P%zu+%zu", o / raw, o % raw) : sfmt("P%zu", o / raw); }
    if (g_harena && p >= g_harena && p < g_harena + (size_t)C.Nh * raw) { const size_t o = (size_t)(p - g_harena); return o % raw ? sfmt("H%zu+%zu", o / raw, o % raw) : sfmt("H%zu", o / raw); }
    for (int k = 0; k < C.PR; k++) if (p >= g_bounce[k] && p < g_bounce[k] + raw) { const size_t o = (size_t)(p - g_bounce[k]); return o ? sfmt("B%d+%zu", k, o) : sfmt("B%d", k); }
    for (int L = 0; L < 93; L++) if (g_slab[L] && p >= g_slab[L] && p < g_slab[L] + 4 * raw) return sfmt("S%d+%zu", L, (size_t)(p - g_slab[L]));
    return "?";
}
static int pool_index(const void *q)   // slot number if q is exactly the start of a pool slot, else -1
{
    const unsigned char *p = (const unsigned char *)q;
    if (!p || p < g_parena || p >= g_parena + (size_t)C.Np * (size_t)C.raw) return -1;
    const size_t o = (size_t)(p - g_parena); return o % (size_t)C.raw ? -1 : (int)(o / (size_t)C.raw);
}

static void fill_expected(unsigned char *slot, int key)
{
    const int layer = key / C.E, e = key % C.E;
    if (C.prefill) memcpy(slot, g_slab[layer] + (size_t)(e & 3) * (size_t)C.raw, (size_t)C.raw);
    else gen_bytes(slot, file_base(fseed_of(layer)), (uint64_t)e * (uint64_t)C.raw, (size_t)C.raw);
}
static uint64_t exp_hash(int key)
{
    static std::unordered_map<int, uint64_t> memo;
    auto f = memo.find(key); if (f != memo.end()) return f->second;
    std::vector<unsigned char> b((size_t)C.raw); fill_expected(b.data(), key);
    const uint64_t h = hash_bytes(b.data(), b.size()); memo[key] = h; return h;
}

// model of the planner's bookkeeping and of the supply's calls
static std::map<int, unsigned char *> g_pending, g_arriving;   // host key -> slot (read not finished); pool key -> slot (copy to the card not finished)
static int g_slot_key[4096], g_slot_prev[4096];                // pool slot -> key the cache gave it, key it held before (the victim)
static std::set<unsigned char *> g_held;                       // host slots taken out of the cache by the call, not yet released
struct NeedRec { int e, key; unsigned char *dst; };
static std::vector<NeedRec> g_need; static std::map<int, unsigned char *> g_host_taken;
static unsigned long c_hit_card, c_acq, c_host_hit, c_wb, c_skipped; static unsigned long wbs_before;
static unsigned char *g_last_ha_slot; static int g_last_ha_key;
static std::atomic<int> g_phase{ 0 }; static std::atomic<bool> g_ev_recorded{ false }; static bool g_sync_done = false;
static std::atomic<int> g_active_jobs{ 0 };
static int g_cur_readers = 1, g_cur_layer = 0;
static std::mutex g_jobmu; static std::vector<std::string> g_jobs; static std::vector<int> g_job_k;
static std::atomic<unsigned> g_jit{ 0 };
static unsigned long g_il_threaded = 0, g_il_interleaved = 0, g_big = 0, g_multi_k = 0;
static int g_occ[16];
// T2 barrier: the first g_bar_expected reader jobs of a call (one per thread: a thread blocked here cannot take another job) wait until all of them have started; a supply that starts fewer threads times out
static std::mutex g_barmu; static std::condition_variable g_barcv; static int g_bar_arrived = 0, g_bar_expected = 0;
struct JobTL { bool active = false; std::string head, ck; int k = -1, stage = 0; int dev = 0; std::string dst, src; size_t n = 0; uint64_t h = 0; };
static thread_local JobTL tl;

static int nu_fn_impl(int key)
{
    g_nuc++;
    const uint64_t m = mix64((uint64_t)key * 7 + 13 + (uint64_t)C.seed);
    if (C.nu_mode == 1 && (key % 4) == 3) return 100000 + (int)(m % 13);   // some experts are needed far in the future: not worth a RAM slot
    if (C.nu_mode == 2) return g_clock + 1 + (int)(m % 5);
    return g_clock + 1 + (int)(m % 40);
}
static int nu_fn(int key) { return nu_fn_impl(key); }

static void maybe_delay()
{
    if (!C.threads) return;
    const uint64_t r = mix64((uint64_t)g_jit.fetch_add(1) * 0x9E3779B1ULL + (uint64_t)C.seed + (uint64_t)(uintptr_t)pthread_self());
    if (r % 3 == 0) usleep((useconds_t)(r % 400));
}
static int maybe_fail(int kind)
{
    if (!C.threads && C.fail_op == kind && ++g_occ[kind] == C.fail_occ) return 700 + kind;
    return 0;
}
static const char *SITE_TEXT[11][2] = {
    { "cudaStreamSynchronize(0)", "k0" }, { "cudaDeviceSynchronize()", "kD" }, { "cudaSetDevice(pool.dev)", "kS" },
    { "cudaMemcpyAsync(h, v.src, EXPERT_RAW, cudaMemcpyDeviceToHost, pool.copy)", "kW" }, { "cudaEventRecord(pool.written_back, pool.copy)", "kE" },
    { "cudaMemcpyAsync(n.dst, g_prefill ? g_slab[layer] + (size_t)(n.e & 3) * EXPERT_RAW : n.host, EXPERT_RAW, cudaMemcpyHostToDevice, pool.copy)", "kH" },
    { "cudaSetDevice(dev)", "kd" }, { "cudaStreamWaitEvent(pool.rstream[tid], pool.written_back, 0)", "kw" },
    { "cudaMemcpyAsync(dev_dst, b, len, cudaMemcpyHostToDevice, pool.rstream[tid])", "kc" }, { "cudaStreamSynchronize(pool.rstream[tid])", "ks" }, { "cudaStreamSynchronize(pool.copy)", "kC" } };

// ---- the chunk model (written separately from the supply's arithmetic): the expected reader copies of one ensure
static void model_chunks(long raw, int chunks_cfg, int n_drive, int pool_readers, std::vector<std::pair<long, long>> &out)
{
    int nchunk;
    if (chunks_cfg > 0) nchunk = chunks_cfg;
    else if (n_drive == 0 || n_drive >= pool_readers) nchunk = 1;
    else { nchunk = (pool_readers + n_drive - 1) / n_drive; if (nchunk > 8) nchunk = 8; }
    const long per = (raw + nchunk - 1) / nchunk, chunk = (per + 4095) / 4096 * 4096;
    long co = 0;
    for (int k = 0; k < nchunk && co < raw; k++, co += chunk) out.push_back({ co, std::min(chunk, raw - co) });
}
static std::string job_text(int dev, const std::string &dst, size_t n, const std::string &src, uint64_t h, int k, const std::string &ck, bool seq)
{
    std::string s = sfmt("J d=%d dst=%s n=%zu src=%s h=%016llx", dev, dst.c_str(), n, src.c_str(), (unsigned long long)h);
    if (seq) s += sfmt(" k=%d", k);
    return s + " ck=" + ck;
}
static size_t model_njobs()   // reader jobs the model expects for the call in progress (needs and host takes are complete once the event is recorded)
{
    int n_drive = 0; for (auto &n : g_need) if (!g_host_taken.count(n.key)) n_drive++;
    std::vector<std::pair<long, long>> ch; model_chunks(C.raw, C.chunks, n_drive, std::min(g_cur_readers, C.PR), ch);
    return (size_t)n_drive * ch.size();
}
static void barrier_wait()
{
    std::unique_lock<std::mutex> lk(g_barmu);
    const int idx = g_bar_arrived++;
    if (idx >= g_bar_expected) return;   // a later job of the same call
    g_barcv.notify_all();
    if (!g_barcv.wait_for(lk, std::chrono::seconds(10), [] { return g_bar_arrived >= g_bar_expected; })) {
        const int a = g_bar_arrived, e = g_bar_expected; lk.unlock();
        viol(sfmt("reader barrier timed out: %d of %d reader threads started a job (the supply must run min(pool_readers, jobs) threads)", a, e));
    }
}
static void final_sync_flush()   // main thread, at the end of the readers' phase: model check, then the job records go into the log
{
    if (g_active_jobs.load() != 0) viol("final stream_sync(pool.copy) while a reader job was still running");
    std::vector<std::string> recs; std::vector<int> ks;
    { std::lock_guard<std::mutex> g(g_jobmu); recs.swap(g_jobs); ks.swap(g_job_k); }
    const int layer = g_cur_layer;
    std::vector<std::pair<NeedRec, bool>> drive;
    int n_drive = 0; for (auto &n : g_need) { const bool h = g_host_taken.count(n.key) != 0; if (!h) n_drive++; drive.push_back({ n, h }); }
    const int pool_readers = std::min(g_cur_readers, C.PR);
    std::vector<std::pair<long, long>> ch; model_chunks(C.raw, C.chunks, n_drive, pool_readers, ch);
    const size_t njobs = (size_t)n_drive * ch.size();
    g_cov[8] += njobs; if (njobs && ch.size() > 1) g_cov[9]++;
    const bool threaded = pool_readers > 1 && njobs >= 2;
    if (C.barrier && threaded) LOG(sfmt("BAR %d", (int)std::min<size_t>((size_t)pool_readers, njobs)));
    std::vector<std::string> expv;
    for (auto &d : drive) {
        if (d.second) continue;
        for (auto &c : ch) {
            std::string src; uint64_t h;
            std::vector<unsigned char> tmp((size_t)c.second);
            if (C.prefill) { src = sfmt("S%d+%ld", layer, (long)(d.first.e & 3) * C.raw + c.first); h = hash_bytes(g_slab[layer] + (size_t)(d.first.e & 3) * (size_t)C.raw + (size_t)c.first, (size_t)c.second); }
            else { src = "B"; gen_bytes(tmp.data(), file_base(fseed_of(layer)), (uint64_t)d.first.e * (uint64_t)C.raw + (uint64_t)c.first, tmp.size()); h = hash_bytes(tmp.data(), tmp.size()); }
            expv.push_back(job_text(C.dev, idname(d.first.dst + c.first), (size_t)c.second, src, h, 0, "kd,kw,kc,ks", false));
        }
    }
    if (recs.size() != expv.size()) viol(sfmt("%zu reader copies recorded, the model expects %zu (n_drive=%d, chunks per expert=%zu, pool_readers=%d)", recs.size(), expv.size(), n_drive, ch.size(), pool_readers));
    // the records carry k (the reader's stream index) for the check below; the compared text has it only in sequential mode
    std::vector<std::string> shown;
    for (size_t i = 0; i < recs.size(); i++) {
        if (ks[i] < 0 || ks[i] >= std::max(1, std::min(pool_readers, (int)std::min<size_t>(njobs, 64)))) {
            if (!(pool_readers <= 0 && ks[i] == 0)) viol(sfmt("reader job on stream index %d with pool_readers %d and %zu jobs", ks[i], pool_readers, njobs));
        }
        if (!threaded && ks[i] != 0) viol(sfmt("sequential section but a job ran on stream index %d", ks[i]));
    }
    std::vector<std::string> a = recs;   // the format of the model has no k; strip it from the records for the comparison
    for (auto &s : a) { const size_t p = s.find(" k="); if (p != std::string::npos) { const size_t q = s.find(' ', p + 1); s.erase(p, q - p); } }
    if (threaded) {
        std::vector<std::string> sa = a, se = expv; std::sort(sa.begin(), sa.end()); std::sort(se.begin(), se.end());
        if (sa != se) { for (size_t i = 0; i < sa.size(); i++) if (sa[i] != se[i]) viol(sfmt("reader copy set differs from the model at sorted index %zu: got [%s] want [%s]", i, sa[i].c_str(), se[i].c_str())); }
        g_il_threaded++; if (a != expv) g_il_interleaved++;   // the commit order of the jobs differed from the job list's order: real concurrency happened
        if (njobs >= 8) { g_big++; std::set<int> dk(ks.begin(), ks.end()); if (dk.size() >= 2) g_multi_k++; }
        for (auto &s : sa) LOG(s);
    } else {
        for (size_t i = 0; i < a.size(); i++) if (a[i] != expv[i]) viol(sfmt("reader copy %zu differs from the model: got [%s] want [%s]", i, a[i].c_str(), expv[i].c_str()));
        for (size_t i = 0; i < recs.size(); i++) { std::string s = a[i]; const size_t p = s.find(" ck="); s.insert(p, sfmt(" k=%d", ks[i])); LOG(s); }
    }
}

// ---- the sy_ops fakes (CUDA effects: recorded, copies done at call time)
static int o_device_sync(void)
{
    if (g_phase.load() != 0) viol("device_sync inside the readers' phase");
    LOG("DS"); return maybe_fail(0);
}
static int o_stream_sync(void *s)
{
    if (!s) { LOG("SS(0)"); return maybe_fail(0); }
    if (s == &g_copy) {
        if (g_phase.load() != 1) viol("stream_sync(pool.copy) without a recorded event before it");
        g_phase.store(0);   // the jobs are over: the main thread is the only one left
        const int f = maybe_fail(9);
        final_sync_flush();
        LOG("SS(c)"); g_sync_done = true; return f;
    }
    int k = -1; for (int i = 0; i < SY_MAX_READERS; i++) if (s == &g_rs[i]) k = i;
    if (k < 0) viol("stream_sync on an unknown stream");
    if (!tl.active || tl.stage != 3 || tl.k != k) viol("reader stream_sync out of order for its job");
    tl.stage = 4; maybe_delay(); return maybe_fail(8);
}
static int o_set_device(int d)
{
    if (d != C.dev) viol(sfmt("set_device(%d), the pool's device is %d", d, C.dev));
    if (g_phase.load() == 1) {   // a reader job starts
        if (tl.active) viol("a reader job started inside a reader job");
        tl = JobTL(); tl.active = true; tl.stage = 1; tl.dev = d; g_active_jobs.fetch_add(1);
        if (C.barrier) barrier_wait();
        maybe_delay(); return maybe_fail(5);
    }
    LOG(sfmt("SD(%d)", d)); return maybe_fail(1);
}
static int o_wait_event(void *s, void *e, unsigned flags)
{
    int k = -1; for (int i = 0; i < SY_MAX_READERS; i++) if (s == &g_rs[i]) k = i;
    if (k < 0 || !tl.active || tl.stage != 1) viol("stream_wait_event outside a reader job or on a wrong stream");
    if (e != &g_ev || flags != 0) viol("stream_wait_event with a wrong event or flags");
    if (!g_ev_recorded.load()) viol("a reader waits for pool.written_back before it was recorded");
    tl.k = k; tl.stage = 2; maybe_delay(); return maybe_fail(6);
}
static int o_d2h(void *dst, const void *src, size_t n, void *s)
{
    if (s != &g_copy || g_phase.load() != 0) viol("device-to-host copy not on pool.copy before the event");
    const int j = pool_index(src); if (j < 0 || n != (size_t)C.raw) viol("device-to-host copy does not read a whole pool slot");
    if (dst != g_last_ha_slot) viol("device-to-host copy into a slot other than the one host_acquire just returned");
    const uint64_t h = hash_bytes((const unsigned char *)src, n);
    if (g_slot_prev[j] < 0 || h != exp_hash(g_slot_prev[j])) viol(sfmt("victim key %d copied down while its bytes are incomplete (arr_wait missing?)", g_slot_prev[j]));
    memcpy(dst, src, n);
    LOG(sfmt("D2H %s %s %zu c h=%016llx", idname(dst).c_str(), idname(src).c_str(), n, (unsigned long long)h)); return maybe_fail(2);
}
static int o_h2d(void *dst, const void *src, size_t n, void *s)
{
    if (s == &g_copy) {
        if (g_phase.load() != 1) viol("main host-to-device copy before the event was recorded");
        const uint64_t h = hash_bytes((const unsigned char *)src, n);
        LOG(sfmt("H2D %s %s %zu c h=%016llx", idname(dst).c_str(), idname(src).c_str(), n, (unsigned long long)h));
        memcpy(dst, src, n); return maybe_fail(4);
    }
    if (!tl.active || tl.stage != 2) viol("reader host-to-device copy without a preceding wait on the event");
    int k = -1; for (int i = 0; i < SY_MAX_READERS; i++) if (s == &g_rs[i]) k = i;
    if (k != tl.k) viol("reader copy on a stream other than the one its job waited on");
    std::string src_name;
    if (!C.prefill) { if (src != g_bounce[k]) viol(sfmt("reader copy on stream %d does not come from that thread's bounce slot (%s)", k, idname(src).c_str())); src_name = "B"; }
    else { src_name = idname(src); if (src_name[0] != 'S') viol("prefill reader copy not from the slab"); }
    const uint64_t h = hash_bytes((const unsigned char *)src, n);
    maybe_delay();
    memcpy(dst, src, n);
    tl.dst = idname(dst); tl.src = src_name; tl.n = n; tl.h = h; tl.stage = 3; return maybe_fail(7);
}
static int o_event_record(void *e, void *s)
{
    if (e != &g_ev || s != &g_copy || g_phase.load() != 0) viol("event_record on a wrong event / stream / phase");
    g_ev_recorded.store(true); g_phase.store(1);
    if (C.barrier) {
        const size_t nj = model_njobs(); const int pr = std::min(g_cur_readers, C.PR);
        std::lock_guard<std::mutex> g(g_barmu); g_bar_arrived = 0; g_bar_expected = (pr > 1 && nj >= 2) ? (int)std::min<size_t>((size_t)pr, nj) : 0;
    }
    LOG("ER"); return maybe_fail(3);
}
static void o_check(int status, const char *expr)
{
    const char *code = nullptr;
    for (int i = 0; i < 11; i++) if (!strcmp(expr, SITE_TEXT[i][0])) code = SITE_TEXT[i][1];
    if (!code) viol(std::string("check called with an unknown site text: ") + expr);
    const std::string cs = status ? sfmt("%s!%d", code, status) : std::string(code);
    if (tl.active) {
        if (!tl.ck.empty()) tl.ck += ",";
        tl.ck += cs;
        if (!strcmp(code, "ks")) {   // the last call of a job: commit its record
            if (tl.stage != 4) viol("job ended out of order");
            std::string rec = sfmt("J d=%d dst=%s n=%zu src=%s h=%016llx k=%d ck=%s", tl.dev, tl.dst.c_str(), tl.n, tl.src.c_str(), (unsigned long long)tl.h, tl.k, tl.ck.c_str());
            { std::lock_guard<std::mutex> g(g_jobmu); g_jobs.push_back(rec); g_job_k.push_back(tl.k); }
            tl.active = false; g_active_jobs.fetch_sub(1);
        }
    } else LOG("CK " + cs);
    if (status) {   // check_ret: the fake returns (the contract says it must not; the supply must then exit(1) itself, at once, without another call)
        flush_log(); fprintf(stderr, "cuda error %d at %s\n", status, expr);
        if (!C.check_ret) exit(1);
    }
}

// ---- the sy_ops fakes (slot caches: log, check the planner's rules, forward to the real functions)
static int cn(const sc_cache *c) { return c == g_pc ? 0 : (c == g_hc ? 1 : 2); }   // 0 pool, 1 host
static std::string protsig(const int *p, size_t n)
{
    std::vector<int> v(p, p + n); std::sort(v.begin(), v.end()); v.erase(std::unique(v.begin(), v.end()), v.end());
    uint64_t h = 0; for (int k : v) h += mix64((uint64_t)(int64_t)k);
    return sfmt("%zu/%zu:%016llx", n, v.size(), (unsigned long long)h);
}
static unsigned char *o_sc_find(sc_cache *c, int key)
{
    unsigned char *s = SC.find(c, key);
    LOG(sfmt("%s F %d %s", cn(c) ? "H" : "P", key, idname(s).c_str()));
    if (cn(c) == 0 && s) c_hit_card++;
    return s;
}
static void o_sc_touch(sc_cache *c, int key, float w)
{
    if (C.vq && g_arriving.count(key)) viol(sfmt("touch of key %d, whose copy to the card is not complete (arr_wait missing)", key));
    SC.touch(c, key, w); LOG(sfmt("P T %d %08x", key, fbits(w)));
}
static int o_sc_worth(const sc_cache *c, int key)
{
    if (C.cap == 0 || C.prefill) viol("worth asked although the RAM cache has no capacity or prefill is on");
    const int r = SC.worth(c, key); LOG(sfmt("H W %d %d", key, r)); if (!r) c_skipped++; return r;
}
static unsigned char *o_sc_acquire(sc_cache *c, int key, const int *prot, size_t np, float h0, int *vk, float *vh)
{
    if (cn(c) != 0) viol("sc_acquire on the RAM cache called directly (it must go through host_acquire)");
    unsigned char *s = SC.acquire(c, key, prot, np, h0, vk, vh);
    const int v = *vk, j = pool_index(s);
    LOG(sfmt("P A %d %08x %s %s %d %08x", key, fbits(h0), protsig(prot, np).c_str(), idname(s).c_str(), v, v >= 0 ? fbits(*vh) : 0u));
    if (j < 0) viol("sc_acquire returned a pointer that is not a pool slot");
    g_slot_prev[j] = v; g_slot_key[j] = key; c_acq++;
    g_need.push_back(NeedRec{ key % C.E, key, s });
    return s;
}
static unsigned char *o_sc_remove(sc_cache *c, int key)
{
    unsigned char *s = SC.remove(c, key);
    LOG(sfmt("H R %d %s", key, idname(s).c_str()));
    if (cn(c) != 1) viol("sc_remove on the pool cache");
    if (s) {
        if (g_pending.count(key)) viol(sfmt("host key %d taken out of the RAM cache before pend_take", key));
        c_host_hit++; g_host_taken[key] = s; g_held.insert(s);
    }
    return s;
}
static void o_sc_release(sc_cache *c, unsigned char *s)
{
    if (!g_sync_done) viol("a host slot released before the final stream_sync(pool.copy)");
    if (!g_held.count(s)) viol("release of a host slot that was not held");
    g_held.erase(s); LOG(sfmt("H L %s", idname(s).c_str())); SC.release(c, s);
}
static void o_pend_take(int key)
{
    auto f = g_pending.find(key);
    LOG(sfmt("PN %d %s", key, f == g_pending.end() ? "none" : "done"));
    if (f != g_pending.end()) { fill_expected(f->second, key); g_pending.erase(f); g_cov[7]++; }
}
static void o_arr_wait(int key)
{
    auto f = g_arriving.find(key);
    LOG(sfmt("AW %d %s", key, f == g_arriving.end() ? "none" : "done"));
    if (f != g_arriving.end()) { fill_expected(f->second, key); g_arriving.erase(f); g_cov[6]++; }
}
static unsigned char *o_host_acquire(int key, float heat, int *vk, float *vh)
{
    if (C.cap == 0 || C.prefill || !g_hc) viol("host_acquire although the RAM cache has no capacity, prefill is on, or there is no RAM cache");
    std::vector<int> prot; for (auto &kv : g_pending) prot.push_back(kv.first);
    unsigned char *s = SC.acquire(g_hc, key, prot.data(), prot.size(), heat, vk, vh);
    const int v = *vk;
    LOG(sfmt("H A %d %08x %s %s %d %08x", key, fbits(heat), protsig(prot.data(), prot.size()).c_str(), idname(s).c_str(), v, v >= 0 ? fbits(*vh) : 0u));
    if (g_held.count(s)) viol("host_acquire handed out a slot that is still held");
    c_wb++; g_last_ha_slot = s; g_last_ha_key = key;
    return s;
}
static sy_ops make_ops()
{
    sy_ops o; memset(&o, 0, sizeof o);
    o.device_sync = o_device_sync; o.stream_sync = o_stream_sync; o.set_device = o_set_device; o.memcpy_d2h_async = o_d2h; o.memcpy_h2d_async = o_h2d; o.event_record = o_event_record;
    o.stream_wait_event = o_wait_event; o.check = o_check; o.sc_find = o_sc_find; o.sc_touch = o_sc_touch; o.sc_worth = o_sc_worth; o.sc_acquire = o_sc_acquire; o.sc_remove = o_sc_remove;
    o.sc_release = o_sc_release; o.pend_take = o_pend_take; o.arr_wait = o_arr_wait; o.host_acquire = o_host_acquire;
    return o;
}

// ---- the driver's own use of the caches (world building), logged with a 'w' prefix
static void place_pool(int key, bool arriving, const std::vector<int> &prot)
{
    if (SC.contains(g_pc, key) || (g_hc && SC.contains(g_hc, key))) return;   // the pool and the RAM cache never hold the same expert (decode.cuh:536-540); acquiring a held key is outside slotcache.h
    int vk = -1; float vh = 0;
    unsigned char *s = SC.acquire(g_pc, key, prot.data(), prot.size(), (float)(1 + RD.r(5)), &vk, &vh);
    const int j = pool_index(s);
    if (j < 0) viol("world: pool acquire returned a bad slot");
    if (vk >= 0) { g_arriving.erase(vk); }
    g_slot_prev[j] = vk; g_slot_key[j] = key;
    if (arriving) { memset(s, 0xEE, (size_t)C.raw); g_arriving[key] = s; } else fill_expected(s, key);
    LOG(sfmt("wP %d %s arr=%d vk=%d", key, idname(s).c_str(), (int)arriving, vk));
}
static void place_host(int key, bool pending)
{
    if (!g_hc || SC.contains(g_hc, key) || SC.contains(g_pc, key)) return;
    if (pending && !C.hostfull && (long)g_pending.size() >= std::max(0L, (long)C.Nh - (long)C.Np - 2)) pending = false;   // room for the experts the call takes out, plus the write-downs, unless this is the host-full unit
    std::vector<int> prot; for (auto &kv : g_pending) prot.push_back(kv.first);
    int vk = -1; float vh = 0;
    unsigned char *s = SC.acquire(g_hc, key, prot.data(), prot.size(), (float)(1 + RD.r(5)), &vk, &vh);
    if (pending) { memset(s, 0xEE, (size_t)C.raw); g_pending[key] = s; } else fill_expected(s, key);
    LOG(sfmt("wH %d %s pend=%d vk=%d", key, idname(s).c_str(), (int)pending, vk));
}

static void dump_state(int step)
{
    // the full state of both caches and the arenas: a hash per piece, compared between the arms
    auto sid = [&](const unsigned char *p) -> uint64_t {   // slot number inside its arena (addresses differ between processes)
        if (p >= g_parena && p < g_parena + (size_t)C.Np * (size_t)C.raw) return (uint64_t)(p - g_parena) / (uint64_t)C.raw;
        if (g_harena && p >= g_harena && p < g_harena + (size_t)C.Nh * (size_t)C.raw) return 100000 + (uint64_t)(p - g_harena) / (uint64_t)C.raw;
        return 999999;
    };
    auto cache_sig = [&](sc_cache *c, bool is_pool, uint64_t &lh, uint64_t &sh, size_t &n, size_t &ns) {
        std::vector<sc_entry> e(SC.snap_lru(c, nullptr, 0)); const size_t tot = e.size(); SC.snap_lru(c, e.data(), tot);
        lh = 1469598103934665603ULL;
        for (auto &x : e) {
            lh = (lh ^ (uint64_t)(uint32_t)x.key) * 1099511628211ULL; lh = (lh ^ sid(x.slot)) * 1099511628211ULL; lh = (lh ^ fbits(x.heat)) * 1099511628211ULL; lh = (lh ^ (uint64_t)(uint32_t)x.nu) * 1099511628211ULL;
            const bool incomplete = is_pool ? g_arriving.count(x.key) != 0 : g_pending.count(x.key) != 0;
            if (!incomplete && hash_bytes(x.slot, (size_t)C.raw) != exp_hash(x.key)) viol(sfmt("step %d: %s cache entry key %d does not hold its expert's bytes", step, is_pool ? "pool" : "host", x.key));
            if (is_pool && g_slot_key[pool_index(x.slot)] != x.key) viol(sfmt("step %d: model slot owner of key %d is wrong", step, x.key));
        }
        std::vector<unsigned char *> sp(SC.snap_spare(c, nullptr, 0)); ns = sp.size(); SC.snap_spare(c, sp.data(), ns);
        sh = 1469598103934665603ULL; for (auto *p : sp) sh = (sh ^ sid(p)) * 1099511628211ULL;
        n = tot;
    };
    uint64_t pl, ps, hl = 0, hs = 0; size_t pn, pns, hn = 0, hns = 0;
    cache_sig(g_pc, true, pl, ps, pn, pns);
    if (g_hc) cache_sig(g_hc, false, hl, hs, hn, hns);
    const uint64_t ph = hash_bytes(g_parena, (size_t)C.Np * (size_t)C.raw), hh = g_harena ? hash_bytes(g_harena, (size_t)C.Nh * (size_t)C.raw) : 0;
    const sy_stats &s = g_pool.st;
    LOGZ(sfmt("Z %d st=%lu,%lu,%lu,%lu,%lu wbs=%lu pool=%zu/%zu:%016llx,%016llx host=%zu/%zu:%016llx,%016llx mem=%016llx,%016llx arr=%zu pend=%zu nuc=%llu", step, s.needed, s.hit_card, s.hit_host, s.from_drive, s.written_back, SY.wbs(),
              pn, pns, (unsigned long long)pl, (unsigned long long)ps, hn, hns, (unsigned long long)hl, (unsigned long long)hs, (unsigned long long)ph, (unsigned long long)hh, g_arriving.size(), g_pending.size(), g_nuc));
}

// ------------------------------------------------------------------------------------------------ one unit in one child
static int run_child(const UCfg &cfg, const std::string &sylib, const std::string &sclib, const std::string &datadir, const std::string &logp)
{
    C = cfg;
    g_covpath = logp + ".cov";
    g_logfd = open(logp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (g_logfd < 0) { perror("open log"); return 2; }
    dup2(g_logfd, 2);   // the library's and the slot cache's stderr text lands in the same file
    on_exit(at_exit_flush, nullptr);
    signal(SIGALRM, on_alarm); alarm(120);
    load_libs(sclib.c_str(), sylib.c_str());
    g_sync_log = C.scn == "YF" || C.scn == "YG" || C.scn == "Y11" || C.hostfull;
    RD = Rng(mix64((uint64_t)C.seed * 2654435761ULL + 99));
    LOG(cfg_str(C));
    const size_t raw = (size_t)C.raw;

    // memory: pool slots ("device"), RAM cache slots, one bounce slot per reader slot, a slab for every layer that can appear (records of 1 MiB and more: zero pages from calloc, no slab, to keep the unit light)
    const bool lazy = C.raw >= (1L << 20);
    auto arena = [&](size_t n, int fill) { if (lazy) return (unsigned char *)calloc(1, n); unsigned char *p = (unsigned char *)malloc(n); memset(p, fill, n); return p; };
    g_parena = arena((size_t)C.Np * raw, 0xCC);
    g_harena = C.Nh > 0 && !C.host_null ? arena((size_t)C.Nh * raw, 0xDD) : nullptr;
    for (int k = 0; k < SY_MAX_READERS; k++) g_bounce[k] = nullptr;
    for (int k = 0; k < C.PR; k++) g_bounce[k] = arena(raw, 0xBB);
    std::set<int> need_slab; for (int i = 0; i < C.nlayers; i++) need_slab.insert(C.layers[i]); for (int i = 0; i < C.nwarm; i++) need_slab.insert(C.warm_layers[i]);
    if (!lazy) for (int L : need_slab) { g_slab[L] = (unsigned char *)malloc(4 * raw); gen_bytes(g_slab[L], slab_base(L), 0, 4 * raw); g_slab_tab[L] = g_slab[L]; }
    for (int i = 0; i < 93; i++) g_fd[i] = -1;
    int bad_layer = C.layers[0];
    for (int i = 0; i < C.nlayers; i++) {
        const int L = C.layers[i], fs = fseed_of(L);
        if (C.vfile) {   // virtual file: /dev/zero only gives a descriptor number; every read of it goes to pread() above
            const int v = open("/dev/zero", O_RDONLY); if (v < 0 || v >= 4096) { fprintf(stderr, "open /dev/zero failed\n"); exit(2); }
            VFile &f = g_vf[v]; f.on = true; f.base = file_base(fs); f.size = (uint64_t)C.E * (uint64_t)C.raw * (uint64_t)C.cut_pct / 100; f.err_mod = C.err_mod; f.err_r = C.err_r; f.err_no = C.err_no;
            g_fd[L] = v; continue;
        }
        const std::string p = data_path(datadir, (C.fdmode == 1 && L == bad_layer) ? 's' : 'f', fs, C);
        g_fd[L] = open(p.c_str(), O_RDONLY); if (g_fd[L] < 0) { fprintf(stderr, "open %s failed\n", p.c_str()); exit(2); }
        if (C.fdmode == 2 && L == bad_layer) { close(g_fd[L]); g_fd[L] = -1; }
    }
    g_pc = SC.nw(); g_hc = (g_harena ? SC.nw() : nullptr);
    SC.set_policy(C.policy); LOG(sfmt("dY policy %d", C.policy));
    if (C.install_nu) { SC.set_next_use(nu_fn); LOG("dX fn"); }
    for (int i = C.Np; i-- > 0;) SC.add_spare(g_pc, g_parena + (size_t)i * raw);
    if (g_hc) for (int i = C.Nh; i-- > 0;) SC.add_spare(g_hc, g_harena + (size_t)i * raw);
    for (int i = 0; i < 4096; i++) g_slot_key[i] = g_slot_prev[i] = -1;

    memset(&g_pool, 0, sizeof g_pool);
    g_eptr.assign((size_t)C.E, nullptr);
    g_pool.dev = C.dev; g_pool.readers = C.PR; g_pool.cache = g_pc; g_pool.copy = &g_copy; g_pool.written_back = &g_ev; g_pool.h_eptr = g_eptr.data();
    for (int k = 0; k < SY_MAX_READERS; k++) { g_pool.rstream[k] = &g_rs[k]; g_pool.bounce[k] = g_bounce[k]; }
    g_host.cache = g_hc; g_host.cap_slots = (size_t)C.cap;
    sy_config scfg; memset(&scfg, 0, sizeof scfg);
    scfg.experts = C.E; scfg.expert_raw = raw; scfg.vq = C.vq; scfg.prefill = C.prefill; scfg.chunks = C.chunks; scfg.slab = g_slab_tab;
    const sy_ops ops = make_ops();
    SY.init(&ops, &scfg); LOG("dY init");

    // warm-up: fill the pool and the RAM cache with complete entries of layers other than (mostly) the called ones
    if (C.warm) {
        const int np = (int)(C.Np * (0.5 + 0.5 * (double)RD.r(100) / 100.0));
        for (int i = 0; i < np; i++) { const int L = C.warm_layers[RD.r((uint32_t)C.nwarm)]; place_pool(L * C.E + (int)RD.r((uint32_t)C.E), false, {}); }
        if (g_hc) {
            const int nh = C.hostfull ? C.Nh : (int)(C.Nh * (double)RD.r(80) / 100.0);
            for (int i = 0; i < nh; i++) { const int L = C.warm_layers[RD.r((uint32_t)C.nwarm)]; place_host(L * C.E + (int)RD.r((uint32_t)C.E), C.hostfull || RD.p(0.2)); }
        }
    }
    std::vector<std::vector<int>> popular((size_t)C.nlayers);
    for (int i = 0; i < C.nlayers; i++) { Rng rp(mix64((uint64_t)C.seed * 31 + (uint64_t)i)); for (int k = 0; k < C.pop; k++) popular[(size_t)i].push_back(C.hi_min ? C.hi_min + (int)rp.r((uint32_t)(C.E - C.hi_min)) : (int)rp.r((uint32_t)C.E)); }

    for (int step = 0; step < C.steps; step++) {
        g_clock = step;
        const int li = (int)RD.r((uint32_t)C.nlayers), layer = C.layers[li];
        std::vector<int> ex;
        const int g = std::max(1, std::min(C.E, C.gmin + (int)RD.r((uint32_t)(C.gmax - C.gmin + 1))));
        for (int tries = 0; (int)ex.size() < g && tries < 200; tries++) {
            const int e = RD.p(0.7) ? popular[(size_t)li][RD.r((uint32_t)popular[(size_t)li].size())] : (int)RD.r((uint32_t)C.E);
            if (std::find(ex.begin(), ex.end(), e) == ex.end()) ex.push_back(e);
        }
        const size_t ndistinct = ex.size();
        if (C.dup_pct && !ex.empty() && RD.p(C.dup_pct / 100.0)) { const int nd = 1 + (int)RD.r(3); for (int k = 0; k < nd && (int)ex.size() < C.Np; k++) ex.push_back(ex[RD.r((uint32_t)ndistinct)]); }
        std::vector<int> counts((size_t)C.E);
        for (int e = 0; e < C.E; e++) counts[(size_t)e] = RD.p(0.1) ? 0 : 1 + (int)RD.r(9);
        // world events before the call: some of its experts are already on the card (maybe still arriving) or in RAM (maybe still being read), and noise
        std::vector<int> call_keys; for (size_t i = 0; i < ndistinct; i++) call_keys.push_back(layer * C.E + ex[i]);
        for (size_t i = 0; i < ndistinct; i++) {
            const int key = call_keys[i];
            if (RD.p(C.p_pool)) place_pool(key, C.vq && RD.p(C.p_arrive), call_keys);
            else if (RD.p(C.p_host)) place_host(key, RD.p(C.p_pend));
        }
        if (RD.p(C.p_noise)) {
            const int nn = 1 + (int)RD.r(4);
            for (int i = 0; i < nn; i++) {
                const int L = C.warm_layers[RD.r((uint32_t)C.nwarm)], key = L * C.E + (int)RD.r((uint32_t)C.E);
                if (RD.p(0.5)) { if ((int)ndistinct < C.Np) place_pool(key, C.vq && RD.p(C.p_arrive), call_keys); }   // a full pool of this call's own keys has no slot left for noise
                else place_host(key, RD.p(C.p_pend));
            }
        }
        const int readers = C.rd_lo + (int)RD.r((uint32_t)(C.rd_hi - C.rd_lo + 1));
        int fd = g_fd[layer]; if (C.prefill && C.fd_unused) fd = -1;
        LOG(sfmt("STEP %d L=%d fd=%s n=%zu rd=%d ex=%s", step, layer, fd < 0 ? "none" : "ok", ex.size(), readers, protsig(ex.data(), ex.size()).c_str()));
        // per-call bookkeeping
        if (step == C.reinit) {   // a second sy_init replaces the ops and the config and sets the g_wb_skipped counter to 0
            const unsigned long before = SY.wbs(); SY.init(&ops, &scfg); const unsigned long after = SY.wbs();
            LOG(sfmt("dY reinit wbs %lu -> %lu", before, after));
            if (after != 0) viol(sfmt("a second sy_init left the g_wb_skipped counter at %lu", after));
        }
        g_need.clear(); g_host_taken.clear(); g_held.clear(); c_hit_card = c_acq = c_host_hit = c_wb = c_skipped = 0; wbs_before = SY.wbs();
        g_phase.store(0); g_ev_recorded.store(false); g_sync_done = false; g_cur_readers = readers; g_cur_layer = layer;
        const sy_stats before = g_pool.st;
        SY.ensure(&g_pool, &g_host, layer, fd, ex.data(), ex.size(), counts.data(), readers);
        // absolute checks
        if (g_phase.load() != 0) viol("the call ended inside the readers' phase");
        if (!g_need.empty() && !g_sync_done) viol("the call ended without the final stream_sync(pool.copy)");
        g_cov[0]++; g_cov[1] += c_hit_card; g_cov[2] += c_acq; g_cov[3] += c_host_hit; g_cov[4] += c_wb; g_cov[5] += c_skipped; if (g_need.empty()) g_cov[10]++; if (ex.size() > ndistinct) g_cov[11]++;
        if (!g_held.empty()) viol("host slots still held after the call");
        const sy_stats &s = g_pool.st;
        auto eq = [&](const char *what, unsigned long got, unsigned long want) { if (got != want) viol(sfmt("step %d: stats %s = %lu, derived from the calls %lu", step, what, got, want)); };
        eq("needed", s.needed - before.needed, ex.size()); eq("hit_card", s.hit_card - before.hit_card, c_hit_card); eq("hit_host", s.hit_host - before.hit_host, c_host_hit);
        eq("from_drive", s.from_drive - before.from_drive, c_acq - c_host_hit); eq("written_back", s.written_back - before.written_back, c_wb); eq("wb_skipped", SY.wbs() - wbs_before, c_skipped);
        eq("find hits + acquires", c_hit_card + c_acq, ex.size());
        const double dre = s.t_read - before.t_read, dco = s.t_copy - before.t_copy, den = s.t_ensure - before.t_ensure;
        if (g_need.empty() && (dre != 0 || dco != 0)) viol("t_read / t_copy changed in a call that needed nothing");
        if (den < 0 || dre < 0 || dco < 0 || den + 1e-9 < dre + dco) viol(sfmt("step %d: times t_ensure %g < t_read %g + t_copy %g", step, den, dre, dco));
        std::string hp;
        for (size_t i = 0; i < ndistinct; i++) {
            const int e = ex[i], key = layer * C.E + e, j = pool_index(g_eptr[(size_t)e]);
            if (j < 0) viol(sfmt("step %d: h_eptr[%d] is not a pool slot", step, e));
            if (hash_bytes(g_parena + (size_t)j * raw, raw) != exp_hash(key)) viol(sfmt("step %d: expert %d (key %d) is not on the card with the right bytes", step, e, key));
            hp += sfmt(" %d=P%d", e, j);
        }
        LOG("HP" + hp);   // where the caller will find each expert
        dump_state(step);
    }
    LOGZ(sfmt("END steps=%d", C.steps));
    // a threaded unit must show that several reader threads really took jobs (a supply that ran every job on the calling thread would pass every other check)
    if (C.threads && g_big >= 4 && g_multi_k * 2 < g_big) viol(sfmt("only %lu of %lu calls with >= 8 reader jobs used more than one thread", g_multi_k, g_big));
    flush_log();
    if (C.threads) { FILE *f = fopen((logp + ".il").c_str(), "wb"); if (f) { fprintf(f, "%lu %lu\n", g_il_threaded, g_il_interleaved); fclose(f); } }
    return 0;
}

// ------------------------------------------------------------------------------------------------ parent: units and runs
struct Opts {
    std::string mode, scn, ref, rust, sc, out, log, data;
    int seed = 0, jobs = 16, keep_raw = 2; bool keep_going = false;
    std::map<std::string, int> n; std::vector<std::string> scns;
};
static std::string g_self;
static pid_t spawn_child(const Opts &o, const std::string &lib, const std::string &log)
{
    std::vector<std::string> a = { g_self, "child", "--scn", o.scn, "--seed", std::to_string(o.seed), "--lib", lib, "--sc", o.sc, "--data", o.data, "--log", log };
    const pid_t p = fork();
    if (p == 0) {
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
        FILE *g = fopen(p.c_str(), "rb"); char *buf = nullptr; size_t cap = 0; uint64_t ln = 0; ssize_t n;
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
static std::string slurp_tail(const std::string &p, size_t cap = 1 << 16)
{
    FILE *f = fopen(p.c_str(), "rb"); if (!f) return ""; fseek(f, 0, SEEK_END); long sz = ftell(f); long from = sz > (long)cap ? sz - (long)cap : 0; fseek(f, from, SEEK_SET);
    std::string s(sz - from, '\0'); size_t r = fread(&s[0], 1, s.size(), f); s.resize(r); fclose(f); return s;
}
static std::string fatal_kind(const std::string &log)
{
    const std::string t = slurp_tail(log);
    if (t.find("every slot is in use by one call") != std::string::npos) return "slotcache";
    if (t.find("reading expert ") != std::string::npos && t.find(" failed: ") != std::string::npos) return "pread";
    if (t.find("cuda error ") != std::string::npos) return "cuda";
    return "";
}

static bool thr_scn(const std::string &s) { return s == "T1" || s == "T2"; }   // threaded scenarios: two runs per arm

static int do_unit(const Opts &o)
{
    UCfg c = make_cfg(o.scn, o.seed);
    const std::string dir = o.out + "/" + o.scn; mkdir_p(dir);
    const std::string pre = dir + "/u" + std::to_string(o.seed);
    make_data(o.data, c);
    struct R { std::string lib, log; pid_t pid; int st; };
    std::vector<R> rs;
    if (thr_scn(o.scn)) rs = { { o.ref, pre + ".ref1.log", 0, 0 }, { o.ref, pre + ".ref2.log", 0, 0 }, { o.rust, pre + ".rust1.log", 0, 0 }, { o.rust, pre + ".rust2.log", 0, 0 } };
    else rs = { { o.ref, pre + ".ref.log", 0, 0 }, { o.rust, pre + ".rust.log", 0, 0 } };
    for (auto &r : rs) r.pid = spawn_child(o, r.lib, r.log);
    for (auto &r : rs) { int st = 0; waitpid(r.pid, &st, 0); r.st = st; }
    bool pass = true; std::string reason = "-", extra;
    uint64_t la = 0, za = 0, lb = 0, zb = 0;
    count_file(rs[0].log, la, za); count_file(rs[thr_scn(o.scn) ? 2 : 1].log, lb, zb);
    uint64_t off = 0;
    const size_t other = thr_scn(o.scn) ? 2 : 1;
    const std::string s0 = status_str(rs[0].st), s1 = status_str(rs[other].st);
    if (!same_files(rs[0].log, rs[other].log, off)) { pass = false; reason = "logs differ"; const std::string d = diff_report(rs[0].log, rs[other].log, off); FILE *f = fopen((pre + ".diff").c_str(), "wb"); if (f) { fwrite(d.data(), 1, d.size(), f); fclose(f); } }
    else if (s0 != s1) { pass = false; reason = "status differs " + s0 + " vs " + s1; }
    else if (s0 != "exit0" && s0 != "exit1") { pass = false; reason = "both children ended with " + s0; }
    std::string fk;
    if (pass && s0 == "exit1") { fk = fatal_kind(rs[0].log); if (fk.empty()) { pass = false; reason = "exit 1 without a known fatal message"; } }
    if (thr_scn(o.scn)) {
        for (auto &r : rs) if (!(WIFEXITED(r.st) && WEXITSTATUS(r.st) == 0)) { pass = false; reason = r.log + ": " + status_str(r.st); break; }
        uint64_t o1 = 0, o2 = 0;
        if (pass && !same_files(rs[0].log, rs[1].log, o1)) { pass = false; reason = "ref run 1 and ref run 2 differ"; }
        if (pass && !same_files(rs[2].log, rs[3].log, o2)) { pass = false; reason = "rust run 1 and rust run 2 differ"; }
        unsigned long thr = 0, il = 0;
        for (auto &r : rs) { FILE *f = fopen((r.log + ".il").c_str(), "rb"); if (f) { unsigned long a = 0, b = 0; if (fscanf(f, "%lu %lu", &a, &b) == 2) { thr += a; il += b; } fclose(f); } }
        extra += sfmt("threaded_ensures=%lu interleaved=%lu ", thr, il);
    }
    extra += "ref=" + s0 + " rust=" + s1; if (!fk.empty()) extra += " fatal=" + fk;
    {
        std::string cov; FILE *f = fopen((rs[0].log + ".cov").c_str(), "rb");
        if (f) { char b[256]; if (fgets(b, sizeof b, f)) cov = b; fclose(f); }
        for (char &ch : cov) if (ch == ' ') ch = ',';
        while (!cov.empty() && (cov.back() == ',' || cov.back() == '\n')) cov.pop_back();
        for (auto &r : rs) unlink((r.log + ".cov").c_str());
        extra += " cov=" + (cov.empty() ? std::string("none") : cov);
    }
    {
        FILE *f = fopen((pre + ".res").c_str(), "wb");
        if (f) { fprintf(f, "%s %s %d %llu %llu %llu %llu %s %s\n", pass ? "PASS" : "FAIL", o.scn.c_str(), o.seed, (unsigned long long)la, (unsigned long long)za, (unsigned long long)lb, (unsigned long long)zb, extra.c_str(), reason.c_str()); fclose(f); }
    }
    if (pass) for (auto &r : rs) {
        if (o.seed < o.keep_raw) { const std::string cmd = "gzip -1 -f '" + r.log + "'; rm -f '" + r.log + ".il'"; if (system(cmd.c_str()) != 0) fprintf(stderr, "gzip failed\n"); }
        else { unlink(r.log.c_str()); unlink((r.log + ".il").c_str()); }
    }
    printf("UNIT %s %d %s lines=%llu/%llu %s %s\n", o.scn.c_str(), o.seed, pass ? "PASS" : "FAIL", (unsigned long long)la, (unsigned long long)lb, extra.c_str(), reason.c_str());
    return pass ? 0 : 1;
}

static int do_run(Opts o)
{
    if (o.scns.empty()) o.scns = { "Y1", "Y2", "Y3", "Y4", "Y5", "Y6", "Y7", "Y8", "Y9", "YF", "T1", "Y10", "Y11", "Y12", "Y13", "Y14", "YG", "T2" };
    std::map<std::string, int> dflt = { { "Y1", 60 }, { "Y2", 60 }, { "Y3", 60 }, { "Y4", 60 }, { "Y5", 80 }, { "Y6", 60 }, { "Y7", 60 }, { "Y8", 60 }, { "Y9", 60 }, { "YF", 80 }, { "T1", 60 },
                                        { "Y10", 40 }, { "Y11", 80 }, { "Y12", 60 }, { "Y13", 60 }, { "Y14", 40 }, { "YG", 80 }, { "T2", 60 } };
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
                std::vector<std::string> a = { g_self, "unit", "--scn", sp.scn, "--seed", std::to_string(sp.seed), "--ref", o.ref, "--rust", o.rust, "--sc", o.sc, "--out", o.out, "--data", o.data, "--keep-raw", std::to_string(o.keep_raw) };
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
    std::string summary; int rc = fails ? 1 : 0;
    for (auto &s : o.scns) {
        int units = 0, pass = 0, fail = 0, exit1 = 0, k_sc = 0, k_pr = 0, k_cu = 0; unsigned long long la = 0, za = 0, lb = 0, zb = 0, thr = 0, il = 0, cov[12] = { 0 };
        for (int i = 0; i < o.n[s]; i++) {
            FILE *f = fopen((o.out + "/" + s + "/u" + std::to_string(i + 1) + ".res").c_str(), "rb"); if (!f) continue;
            char line[2048]; if (!fgets(line, sizeof line, f)) { fclose(f); continue; } fclose(f);
            char res[16], scn[16]; int sd; unsigned long long a1, a2, b1, b2;
            if (sscanf(line, "%15s %15s %d %llu %llu %llu %llu", res, scn, &sd, &a1, &a2, &b1, &b2) != 7) continue;
            units++; if (!strcmp(res, "PASS")) pass++; else fail++;
            la += a1; za += a2; lb += b1; zb += b2;
            if (strstr(line, "ref=exit1")) exit1++;
            if (strstr(line, "fatal=slotcache")) { k_sc++; }
            if (strstr(line, "fatal=pread")) { k_pr++; }
            if (strstr(line, "fatal=cuda")) { k_cu++; }
            const char *t = strstr(line, "threaded_ensures="); unsigned long long x = 0, y = 0; if (t && sscanf(t, "threaded_ensures=%llu interleaved=%llu", &x, &y) == 2) { thr += x; il += y; }
            const char *cv = strstr(line, "cov=");
            if (cv) { unsigned long long v[12]; if (sscanf(cv, "cov=%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7], &v[8], &v[9], &v[10], &v[11]) == 12) for (int q = 0; q < 12; q++) cov[q] += v[q]; }
        }
        const unsigned long long opa = la - za - 4ULL * units, opb = lb - zb - 4ULL * units;   // minus the Z lines and the CFG / END / PREAD / EXIT line of each log
        std::string l = sfmt("%s units=%d pass=%d fail=%d logged_lines_ref=%llu logged_lines_rust=%llu (all lines %llu/%llu, Z lines %llu/%llu) units_ending_in_fatal_exit1=%d (slotcache %d, pread %d, cuda %d)", s.c_str(), units, pass, fail, opa, opb, la, lb, za, zb, exit1, k_sc, k_pr, k_cu);
        if (thr_scn(s)) l += sfmt(" threaded_ensures=%llu of_which_reader_copy_commit_order_differed_from_job_list_order=%llu", thr, il);
        l += (fail == 0 && units == o.n[s]) ? " PASS" : " FAIL";
        l += sfmt("\n    coverage (reference arm, sums over the units): ensure_calls=%llu card_hits=%llu pool_acquires=%llu ram_hits=%llu writebacks=%llu not_worth=%llu arr_wait_completed=%llu pend_take_completed=%llu reader_copies=%llu calls_with_chunks_gt1=%llu calls_needing_nothing=%llu calls_with_duplicate_expert=%llu", cov[0], cov[1], cov[2], cov[3], cov[4], cov[5], cov[6], cov[7], cov[8], cov[9], cov[10], cov[11]);
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
    if (argc < 2) { fprintf(stderr, "usage: sy_driver run|unit|child --ref LIB --rust LIB --sc LIB --data DIR [--out DIR --n Y1=60,... --scn Y1,Y2 --jobs N --keep-going]\n"); return 2; }
    Opts o; o.mode = argv[1];
    std::string lib;
    for (int i = 2; i < argc; i++) {
        const std::string k = argv[i]; auto val = [&]() -> std::string { if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", k.c_str()); exit(2); } return argv[++i]; };
        if (k == "--scn") { const std::string v = val(); if (o.mode == "run") { size_t p = 0; while (p <= v.size()) { size_t q = v.find(',', p); if (q == std::string::npos) q = v.size(); if (q > p) o.scns.push_back(v.substr(p, q - p)); p = q + 1; } } else o.scn = v; }
        else if (k == "--seed") o.seed = atoi(val().c_str());
        else if (k == "--ref") o.ref = val(); else if (k == "--rust") o.rust = val(); else if (k == "--sc") o.sc = val(); else if (k == "--out") o.out = val(); else if (k == "--data") o.data = val();
        else if (k == "--lib") lib = val(); else if (k == "--log") o.log = val(); else if (k == "--jobs") o.jobs = atoi(val().c_str());
        else if (k == "--keep-raw") o.keep_raw = atoi(val().c_str()); else if (k == "--keep-going") o.keep_going = true;
        else if (k == "--n") { const std::string v = val(); size_t p = 0; while (p < v.size()) { size_t q = v.find(',', p); if (q == std::string::npos) q = v.size(); const std::string t = v.substr(p, q - p); const size_t e = t.find('='); if (e != std::string::npos) o.n[t.substr(0, e)] = atoi(t.c_str() + e + 1); p = q + 1; } }
        else { fprintf(stderr, "unknown option %s\n", k.c_str()); return 2; }
    }
    if (o.data.empty()) o.data = o.out + "/data";
    if (o.mode == "child") { const UCfg c = make_cfg(o.scn, o.seed); return run_child(c, lib, o.sc, o.data, o.log); }
    if (o.mode == "unit") return do_unit(o);
    if (o.mode == "run") return do_run(o);
    fprintf(stderr, "unknown mode %s\n", o.mode.c_str()); return 2;
}
