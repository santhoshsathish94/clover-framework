// Differential verifier for copyq.h: dlopen()s one library (argv[1]) and runs one scenario (argv[2]).
// stdout = canonical output (deterministic from the seed: no times, no addresses); stderr = diagnostics.
// Exit code: 0 PASS, 2 FAIL, 3 driver/infrastructure error.
// Build (box): nvcc -O2 -std=c++17 -I../include diff_driver.cpp -o diff_driver -ldl -lpthread
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "copyq.h"

static const size_t RAW = 17547264;
static const int NCARDS = 8;

#define CU(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "DRIVER cuda error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); _exit(3); } } while (0)
#define DIE(...) do { fprintf(stderr, "DRIVER: " __VA_ARGS__); fprintf(stderr, "\n"); _exit(3); } while (0)

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// ---------------------------------------------------------------- sha256
static const uint32_t K256[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
static inline uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
static void sha_block(uint32_t h[8], const uint8_t *p)
{
    uint32_t w[64];
    for (int i = 0; i < 16; i++) w[i] = (uint32_t)p[4*i] << 24 | (uint32_t)p[4*i+1] << 16 | (uint32_t)p[4*i+2] << 8 | p[4*i+3];
    for (int i = 16; i < 64; i++) {
        const uint32_t s0 = rotr(w[i-15], 7) ^ rotr(w[i-15], 18) ^ (w[i-15] >> 3), s1 = rotr(w[i-2], 17) ^ rotr(w[i-2], 19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; i++) {
        const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25), ch = (e & f) ^ (~e & g), t1 = hh + S1 + ch + K256[i] + w[i];
        const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22), mj = (a & b) ^ (a & c) ^ (b & c), t2 = S0 + mj;
        hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
}
static std::string sha256_hex(const uint8_t *d, size_t n)
{
    uint32_t h[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    size_t i = 0;
    for (; i + 64 <= n; i += 64) sha_block(h, d + i);
    uint8_t tail[128] = {0}; const size_t rem = n - i; memcpy(tail, d + i, rem); tail[rem] = 0x80;
    const size_t tl = rem + 1 + 8 <= 64 ? 64 : 128; const uint64_t bits = (uint64_t)n * 8;
    for (int k = 0; k < 8; k++) tail[tl - 1 - k] = (uint8_t)(bits >> (8 * k));
    sha_block(h, tail); if (tl == 128) sha_block(h, tail + 64);
    char out[65]; for (int k = 0; k < 8; k++) snprintf(out + 8 * k, 9, "%08x", h[k]);
    return std::string(out);
}

// ---------------------------------------------------------------- records: bytes = hash(record id, word index); negative ids = poison fills
static inline uint64_t smix(uint64_t x) { x += 0x9e3779b97f4a7c15ULL; x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL; x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL; return x ^ (x >> 31); }
static void gen_record(int id, uint8_t *b)
{
    if (id == -1) { memset(b, 0x5A, RAW); return; }   // kind 0 slot initial
    if (id == -2) { memset(b, 0xA5, RAW); return; }   // card destination initial
    if (id == -3) { memset(b, 0xEE, RAW); return; }   // victim buffer initial
    uint64_t *w = (uint64_t *)b; for (size_t i = 0; i < RAW / 8; i++) w[i] = smix(((uint64_t)(uint32_t)id << 40) ^ i);
}
struct Rec { uint8_t *buf; std::string sha; };
static std::map<int, Rec> g_rec; static std::mutex g_recmu;
static const Rec &rec(int id)
{
    std::lock_guard<std::mutex> g(g_recmu);
    auto it = g_rec.find(id); if (it != g_rec.end()) return it->second;
    Rec r; r.buf = (uint8_t *)aligned_alloc(4096, RAW); if (!r.buf) DIE("alloc");
    gen_record(id, r.buf); r.sha = sha256_hex(r.buf, RAW);
    return g_rec.emplace(id, r).first->second;
}
static uint8_t *pinned(size_t n) { uint8_t *p = nullptr; CU(cudaHostAlloc((void **)&p, n, cudaHostAllocPortable)); return p; }
static std::map<int, uint8_t *> g_pin;   // pinned copies of source records for kind 2 host_src (filled by the main thread only)
static const uint8_t *pinned_rec(int id) { auto it = g_pin.find(id); if (it != g_pin.end()) return it->second; uint8_t *p = pinned(RAW); memcpy(p, rec(id).buf, RAW); g_pin[id] = p; return p; }

struct Rng { uint64_t s; uint64_t next() { return smix(s++); } int below(int n) { return (int)(next() % (uint64_t)n); } };

// ---------------------------------------------------------------- flags and jobs
struct alignas(64) Flags { int arr, rel, wb, p; char pad[48]; };
static inline int ld(const int *p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }

struct Job {
    int id = 0, kind = 0, card = 0, dev_field = 0, fd = -1, layer = 0, filerec = 0;
    bool victim = false, rel = false;
    int srcid = -1;                 // record id the destination (kind>=1) or slot (kind 0) must hold at the end
    int oldid = -2;                 // record id the card destination held before the job (the victim copy must equal it)
    uint8_t *dst = nullptr, *slot = nullptr, *vict = nullptr; const uint8_t *host_src = nullptr;
    Flags *f = nullptr; bool chain = false; std::atomic<int> enq{0};
    std::string dst_hex, vic_hex, slot_hex, note; int dst_ok = -1, vic_ok = -1, slot_ok = -1; bool timeout = false;
    int fin_arr = 0, fin_rel = 0, fin_wb = 0, fin_p = 0;
    Job() {}
    Job(Job &&o) noexcept { *this = std::move(o); }
    Job &operator=(Job &&o) noexcept {
        id = o.id; kind = o.kind; card = o.card; dev_field = o.dev_field; fd = o.fd; layer = o.layer; filerec = o.filerec; victim = o.victim; rel = o.rel;
        srcid = o.srcid; oldid = o.oldid; dst = o.dst; slot = o.slot; vict = o.vict; host_src = o.host_src; f = o.f; chain = o.chain; enq = o.enq.load();
        dst_hex = o.dst_hex; vic_hex = o.vic_hex; slot_hex = o.slot_hex; note = o.note; dst_ok = o.dst_ok; vic_ok = o.vic_ok; slot_ok = o.slot_ok; timeout = o.timeout;
        fin_arr = o.fin_arr; fin_rel = o.fin_rel; fin_wb = o.fin_wb; fin_p = o.fin_p; return *this;
    }
};
static void exp_flags(const Job &j, int &a, int &r, int &w, int &p) { a = j.kind >= 1; r = j.kind >= 1 && j.rel; w = j.kind >= 1 && j.victim; p = j.kind == 0; }

typedef void (*init_fn)(const cq_config *); typedef void (*enq_fn)(const cq_job *);
static init_fn p_init; static enq_fn p_enq;
static void load_lib(const char *path)
{
    void *h = dlopen(path, RTLD_NOW | RTLD_LOCAL); if (!h) DIE("dlopen %s: %s", path, dlerror());
    p_init = (init_fn)dlsym(h, "cq_init"); p_enq = (enq_fn)dlsym(h, "cq_enqueue"); if (!p_init || !p_enq) DIE("missing cq_init/cq_enqueue in %s", path);
}
static void enqueue(Job &j)
{
    cq_job c; memset(&c, 0, sizeof c);
    c.kind = j.kind; c.dev = j.dev_field; c.fd = j.fd; c.off = (off_t)j.filerec * (off_t)RAW; c.layer = j.layer; c.slot = j.slot;
    c.p_state = &j.f->p; c.dst = j.dst; c.host_src = j.host_src; c.victim_host = j.vict;
    c.wb_state = &j.f->wb; c.arr_state = &j.f->arr; c.rel_state = j.rel ? &j.f->rel : nullptr;   // every *_state non-NULL except rel: a flag not belonging to the kind must stay 0
    j.enq = 1; p_enq(&c);
}
static void alloc_mem(Job &j, bool pageable_only = false)
{
    if (j.kind >= 1) {
        CU(cudaSetDevice(j.card)); CU(cudaMalloc((void **)&j.dst, RAW));
        if (j.victim && j.oldid >= 0) CU(cudaMemcpy(j.dst, rec(j.oldid).buf, RAW, cudaMemcpyHostToDevice)); else CU(cudaMemset(j.dst, 0xA5, RAW));
        if (j.victim) { j.vict = (!pageable_only && j.id % 2 == 0) ? pinned(RAW) : (uint8_t *)aligned_alloc(4096, RAW); memset(j.vict, 0xEE, RAW); }
    } else { j.slot = (!pageable_only && j.id % 2 == 0) ? pinned(RAW) : (uint8_t *)aligned_alloc(4096, RAW); memset(j.slot, 0x5A, RAW); }
}

static void check_bytes(const uint8_t *buf, int id, bool fast, std::string &hex, int &ok, std::string &note)
{
    const Rec &e = rec(id);
    if (fast && !memcmp(buf, e.buf, RAW)) { hex = e.sha; ok = 1; return; }
    hex = sha256_hex(buf, RAW); ok = hex == e.sha;
    if (!ok) { size_t i = 0, nd = 0; while (i < RAW && buf[i] == e.buf[i]) i++; for (size_t k = 0; k < RAW; k++) nd += buf[k] != e.buf[k]; note += " firstdiff=" + std::to_string(i) + " ndiff=" + std::to_string(nd); }
}

// ---------------------------------------------------------------- observer + waiters
struct Watch {
    std::vector<Job> *jobs = nullptr; std::atomic<bool> stop{false}; std::thread obs; std::vector<std::thread> waiters; std::vector<uint8_t> vio; bool fast = false; double deadline = 0;
};
static void observer(Watch *w)
{
    std::vector<Job> &J = *w->jobs;
    while (!w->stop.load(std::memory_order_relaxed)) {
        for (size_t i = 0; i < J.size(); i++) {
            Job &j = J[i]; Flags *f = j.f; uint8_t &v = w->vio[i];
            if (j.kind >= 1) {
                const int a = ld(&f->arr);
                if (a == 1) { const int r = ld(&f->rel), wb = ld(&f->wb); if (j.rel && r != 1) v |= 1; if (j.victim && wb != 1) v |= 2; }
                if (j.victim && !(v & 24) && ld(&f->wb) == 1) { if (memcmp(j.vict + RAW - 64, rec(j.oldid).buf + RAW - 64, 64)) v |= 8; else v |= 16; }   // 16 = checked, fine
            } else if (!(v & 36) && ld(&f->p) == 1) { if (memcmp(j.slot + RAW - 64, rec(j.srcid).buf + RAW - 64, 64)) v |= 4; else v |= 32; }
        }
    }
}
static void waiter(Watch *w, std::vector<Job *> jobs, int card)
{
    uint8_t *buf = nullptr; cudaStream_t st = nullptr;
    if (card >= 0) { CU(cudaSetDevice(card)); buf = pinned(RAW); CU(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking)); }
    std::vector<Job *> pend = jobs;
    while (!pend.empty()) {
        bool prog = false;
        for (size_t i = 0; i < pend.size();) {
            Job *j = pend[i]; const int *flag = j->kind >= 1 ? &j->f->arr : &j->f->p;
            if (ld(flag) == 1) {
                if (j->kind >= 1) {
                    CU(cudaMemcpyAsync(buf, j->dst, RAW, cudaMemcpyDeviceToHost, st)); CU(cudaStreamSynchronize(st));
                    check_bytes(buf, j->srcid, w->fast, j->dst_hex, j->dst_ok, j->note);
                    if (j->vict) check_bytes(j->vict, j->oldid, w->fast, j->vic_hex, j->vic_ok, j->note);
                } else check_bytes(j->slot, j->srcid, w->fast, j->slot_hex, j->slot_ok, j->note);
                pend.erase(pend.begin() + i); prog = true;
            } else i++;
        }
        if (now_s() > w->deadline) { for (Job *j : pend) j->timeout = true; break; }
        if (!prog) std::this_thread::yield();
    }
}
static void watch_start(Watch &w, std::vector<Job> &jobs, bool fast, double timeout_s, bool waiters)
{
    w.jobs = &jobs; w.fast = fast; w.deadline = now_s() + timeout_s; w.vio.assign(jobs.size(), 0);
    w.obs = std::thread(observer, &w);
    if (!waiters) return;
    std::vector<Job *> by[NCARDS + 1];
    for (Job &j : jobs) by[j.kind >= 1 ? j.card : NCARDS].push_back(&j);
    for (int c = 0; c <= NCARDS; c++) if (!by[c].empty()) w.waiters.emplace_back(waiter, &w, by[c], c == NCARDS ? -1 : c);
}
static void watch_join_waiters(Watch &w) { for (auto &t : w.waiters) t.join(); w.waiters.clear(); }
static void watch_stop(Watch &w) { w.stop = true; w.obs.join(); }

// ---------------------------------------------------------------- reporting
static int report(const char *scen, std::vector<Job> &jobs, Watch &w, const std::string &extra_fail)
{
    usleep(300 * 1000);   // settle: a flag that is stored late or by the wrong kind shows up here
    watch_stop(w);
    int bad = 0, nviol_rel = 0, nviol_wb = 0, nviol_p = 0, nviol_wbd = 0, nto = 0, nflag = 0;
    for (size_t i = 0; i < jobs.size(); i++) {
        Job &j = jobs[i]; Flags *f = j.f; j.fin_arr = ld(&f->arr); j.fin_rel = ld(&f->rel); j.fin_wb = ld(&f->wb); j.fin_p = ld(&f->p);
        int a, r, wb, p; exp_flags(j, a, r, wb, p);
        const bool fl_ok = j.fin_arr == a && j.fin_rel == r && j.fin_wb == wb && j.fin_p == p;
        const bool jb = j.timeout || !fl_ok || j.dst_ok == 0 || j.vic_ok == 0 || j.slot_ok == 0;
        bad += jb; nto += j.timeout; nflag += !fl_ok; const uint8_t v = w.vio[i]; nviol_rel += !!(v & 1); nviol_wb += !!(v & 2); nviol_p += !!(v & 4); nviol_wbd += !!(v & 8);
        printf("J %d kind=%d dev=%d card=%d fd=%s rec=%d layer=%d victim=%d rel=%d src=%d | dst=%s %s vic=%s %s slot=%s %s | flags arr=%d rel=%d wb=%d p=%d %s%s%s\n",
               j.id, j.kind, j.dev_field, j.kind >= 1 ? j.card : -1, j.fd >= 0 ? "ok" : "bad", j.filerec, j.layer, (int)j.victim, (int)j.rel, j.srcid,
               j.dst_hex.empty() ? "-" : j.dst_hex.c_str(), j.dst_ok < 0 ? "-" : (j.dst_ok ? "OK" : "BAD"), j.vic_hex.empty() ? "-" : j.vic_hex.c_str(), j.vic_ok < 0 ? "-" : (j.vic_ok ? "OK" : "BAD"),
               j.slot_hex.empty() ? "-" : j.slot_hex.c_str(), j.slot_ok < 0 ? "-" : (j.slot_ok ? "OK" : "BAD"), j.fin_arr, j.fin_rel, j.fin_wb, j.fin_p, fl_ok ? "flags_OK" : "flags_BAD", j.timeout ? " TIMEOUT" : "", j.note.c_str());
    }
    const long viol = (long)nviol_rel + nviol_wb + nviol_p + nviol_wbd;
    printf("OBSERVER arr1_without_rel=%d arr1_without_wb=%d p1_slot_not_final=%d wb1_victim_not_final=%d\n", nviol_rel, nviol_wb, nviol_p, nviol_wbd);
    const bool pass = bad == 0 && viol == 0 && extra_fail.empty();
    printf("SUMMARY %s jobs=%zu bad_jobs=%d timeouts=%d bad_flags=%d observer_violations=%ld%s\n", scen, jobs.size(), bad, nto, nflag, viol, extra_fail.c_str());
    printf("RESULT %s\n", pass ? "PASS" : "FAIL");
    fflush(stdout);
    return pass ? 0 : 2;
}

static std::string g_files = "/dev/shm/copyq_verify";
static std::string g_tfile = "/home/usergpu/rustmig/copyq/verify/data/rec8.bin";
static int open_file(const std::string &p) { int fd = open(p.c_str(), O_RDONLY); if (fd < 0) DIE("open %s: run mkfiles first", p.c_str()); return fd; }

static void init_lib(int prefill, const unsigned char *const *slab, const int *hold)
{
    cq_config cfg; memset(&cfg, 0, sizeof cfg); cfg.expert_raw = RAW; cfg.readers = 16; cfg.cards = NCARDS; cfg.prefill = prefill; cfg.slab = slab; cfg.act_hold = hold;
    for (int c = 0; c < NCARDS; c++) { CU(cudaSetDevice(c)); CU(cudaFree(0)); }
    p_init(&cfg);
}
static void shuffle_order(std::vector<int> &o, Rng &r) { for (size_t i = o.size(); i > 1; i--) std::swap(o[i - 1], o[r.below((int)i)]); }

// ---------------------------------------------------------------- S1 mixed
static std::vector<Job> make_s1(Rng &r, int fd, Flags *&fl)
{
    std::vector<Job> jobs; jobs.reserve(200); int id = 0;
    auto add = [&](int kind, int card, int rc) {
        jobs.emplace_back(); Job &j = jobs.back(); j.id = id++; j.kind = kind; j.card = card; j.dev_field = card; j.filerec = rc; j.victim = r.below(2); j.rel = r.below(2); j.srcid = rc;
        j.oldid = j.victim ? 100 + r.below(8) : -2;
        if (kind == 1) j.fd = fd; if (kind == 2) j.host_src = pinned_rec(rc);
        if (kind == 0) { j.victim = false; j.oldid = -2; j.dev_field = r.below(2) ? 99 : r.below(NCARDS); }
        return &j;
    };
    for (int c = 0; c < NCARDS; c++) { for (int t = 0; t < 4; t++) add(1, c, r.below(8)); for (int t = 0; t < 4; t++) add(2, c, r.below(8)); }
    for (int t = 0; t < 32; t++) add(0, 0, r.below(8))->fd = fd;
    const int nbase = (int)jobs.size();
    for (int t = 0; t < 32; t++) { const int src_slot_job = 64 + t; Job *c = add(2, t % NCARDS, jobs[src_slot_job].filerec); c->chain = true; c->srcid = jobs[src_slot_job].srcid; c->filerec = jobs[src_slot_job].filerec; }
    (void)nbase;
    fl = (Flags *)aligned_alloc(64, jobs.size() * sizeof(Flags)); memset(fl, 0, jobs.size() * sizeof(Flags));
    for (size_t i = 0; i < jobs.size(); i++) { jobs[i].f = &fl[i]; alloc_mem(jobs[i]); }
    return jobs;
}
static int run_s1(uint64_t seed)
{
    Rng r{seed}; Flags *fl; int fd = open_file(g_files + "/rec8.bin");
    for (int i = 0; i < 8; i++) { rec(i); rec(100 + i); } rec(-1); rec(-2); rec(-3);
    std::vector<Job> jobs = make_s1(r, fd, fl);
    printf("SCENARIO S1 seed=%llu jobs=%zu\n", (unsigned long long)seed, jobs.size());
    init_lib(0, nullptr, nullptr);
    Watch w; watch_start(w, jobs, false, 240, true);
    std::vector<int> order; for (size_t i = 0; i < jobs.size(); i++) if (!jobs[i].chain) order.push_back((int)i); shuffle_order(order, r);
    for (int i : order) enqueue(jobs[i]);
    std::string extra;
    for (size_t k = 0; k < 32; k++) {   // chain: kind 0 slot -> kind 2 source, once the slot flag is set
        Job &s = jobs[64 + k]; const double dl = now_s() + 240;
        while (ld(&s.f->p) != 1 && now_s() < dl) std::this_thread::yield();
        Job &c = jobs[96 + k]; c.host_src = s.slot; if (ld(&s.f->p) != 1) { extra += " chain_wait_timeout"; continue; }
        enqueue(c);
    }
    watch_join_waiters(w);
    return report("S1", jobs, w, extra);
}

// ---------------------------------------------------------------- S2 prefill
static int run_s2(uint64_t seed)
{
    Rng r{seed}; Flags *fl; int fd = open_file(g_files + "/rec8.bin");
    for (int i = 0; i < 8; i++) { rec(i); rec(100 + i); } rec(-1); rec(-2); rec(-3);
    std::vector<Job> jobs; jobs.reserve(80); int id = 0;
    auto add = [&](int kind, int card) {
        jobs.emplace_back(); Job &j = jobs.back(); j.id = id++; j.kind = kind; j.card = card; j.dev_field = card; j.filerec = r.below(8); j.layer = r.below(93);
        if (id == 1) j.layer = 0; if (id == 2) j.layer = 92;
        j.victim = kind >= 1 && r.below(2); j.rel = r.below(2); j.oldid = j.victim ? 100 + r.below(8) : -2;
        j.fd = r.below(2) ? fd : -1;                                         // no drive read must happen: -1 must be as good as a valid fd
        if (kind == 2) j.host_src = r.below(2) ? pinned_rec(j.filerec) : nullptr;   // ignored under prefill: a wrong source would show as record `filerec`
        if (kind == 0) { j.srcid = -1; j.dev_field = r.below(2) ? 99 : card; } else j.srcid = 1000 + j.layer * 4 + (j.filerec & 3);
    };
    for (int c = 0; c < NCARDS; c++) { for (int t = 0; t < 3; t++) add(1, c); for (int t = 0; t < 3; t++) add(2, c); }
    for (int t = 0; t < 16; t++) add(0, 0);
    static uint8_t *slabp[93]; static const unsigned char *slabc[93]; static uint8_t *poison = (uint8_t *)aligned_alloc(4096, 4 * RAW); memset(poison, 0xEE, 4 * RAW);
    std::vector<char> used(93, 0); for (Job &j : jobs) if (j.kind >= 1) used[j.layer] = 1;
    for (int l = 0; l < 93; l++) {
        if (!used[l]) { slabp[l] = poison; slabc[l] = poison; continue; }
        slabp[l] = (uint8_t *)aligned_alloc(4096, 4 * RAW); for (int q = 0; q < 4; q++) gen_record(1000 + l * 4 + q, slabp[l] + (size_t)q * RAW); slabc[l] = slabp[l];
    }
    for (Job &j : jobs) if (j.kind >= 1) rec(j.srcid);
    fl = (Flags *)aligned_alloc(64, jobs.size() * sizeof(Flags)); memset(fl, 0, jobs.size() * sizeof(Flags));
    for (size_t i = 0; i < jobs.size(); i++) { jobs[i].f = &fl[i]; alloc_mem(jobs[i]); }
    printf("SCENARIO S2 seed=%llu jobs=%zu prefill=1\n", (unsigned long long)seed, jobs.size());
    init_lib(1, slabc, nullptr);
    Watch w; watch_start(w, jobs, false, 240, true);
    std::vector<int> order; for (size_t i = 0; i < jobs.size(); i++) order.push_back((int)i); shuffle_order(order, r);
    for (int i : order) enqueue(jobs[i]);
    watch_join_waiters(w);
    return report("S2", jobs, w, "");
}

// ---------------------------------------------------------------- S3 act_hold
static int run_s3(uint64_t seed)
{
    Rng r{seed}; Flags *fl; int fd = open_file(g_files + "/rec8.bin");
    for (int i = 0; i < 8; i++) { rec(i); rec(100 + i); } rec(-1); rec(-2); rec(-3);
    std::vector<Job> jobs; jobs.reserve(64); int id = 0;
    auto add = [&](int kind, int card) {
        jobs.emplace_back(); Job &j = jobs.back(); j.id = id++; j.kind = kind; j.card = card; j.dev_field = card; j.filerec = r.below(8); j.srcid = j.filerec; j.victim = kind >= 1 && r.below(2); j.rel = r.below(2);
        j.oldid = j.victim ? 100 + r.below(8) : -2; if (kind == 1) j.fd = fd; if (kind == 2) j.host_src = pinned_rec(j.filerec); if (kind == 0) { j.fd = fd; j.dev_field = 99; }
    };
    for (int c = 0; c < NCARDS; c++) { add(1, c); add(1, c); add(2, c); add(2, c); }
    for (int t = 0; t < 8; t++) add(0, 0);
    fl = (Flags *)aligned_alloc(64, jobs.size() * sizeof(Flags)); memset(fl, 0, jobs.size() * sizeof(Flags));
    for (size_t i = 0; i < jobs.size(); i++) { jobs[i].f = &fl[i]; alloc_mem(jobs[i]); }
    printf("SCENARIO S3 seed=%llu jobs=%zu\n", (unsigned long long)seed, jobs.size());
    static int hold __attribute__((aligned(64))); __atomic_store_n(&hold, 1, __ATOMIC_SEQ_CST);
    init_lib(0, nullptr, &hold);
    Watch w; watch_start(w, jobs, false, 240, false);
    for (Job &j : jobs) enqueue(j);
    int arr_set = 0, rel_set = 0;
    for (int ms = 0; ms < 300; ms++) { for (Job &j : jobs) if (j.kind >= 1) { arr_set += ld(&j.f->arr); rel_set += ld(&j.f->rel); } usleep(1000); }
    // Model of the original: a card queue has max(2, readers/cards) = 2 threads, each takes one job in FIFO order and stops in the gate after its victim copy, so while the hold
    // is on exactly the first 2 jobs of each card queue (enqueue order) have done their victim copy (wb == 1) and no later job has (wb == 0). Kind 0 is not gated.
    std::vector<int> qidx(jobs.size(), -1); { int cnt[NCARDS] = {0}; for (size_t i = 0; i < jobs.size(); i++) if (jobs[i].kind >= 1) qidx[i] = cnt[jobs[i].card]++; }
    int nk0 = 0, p_done = 0, wb_ok = 0, ngated0 = 0; const double dl = now_s() + 20;
    for (Job &j : jobs) { nk0 += j.kind == 0; ngated0 += j.kind >= 1; }
    auto wb_model = [&]() { int ok = 0; for (size_t i = 0; i < jobs.size(); i++) if (jobs[i].kind >= 1) ok += (ld(&jobs[i].f->wb) == 1) == (jobs[i].victim && qidx[i] < 2); return ok; };
    for (;;) {
        p_done = 0; for (Job &j : jobs) if (j.kind == 0) p_done += ld(&j.f->p) == 1;
        wb_ok = wb_model();
        if ((p_done == nk0 && wb_ok == ngated0) || now_s() > dl) break; usleep(1000);
    }
    for (Job &j : jobs) if (j.kind >= 1) { arr_set += ld(&j.f->arr); rel_set += ld(&j.f->rel); }
    int unchanged = 0, ngated = 0; uint8_t *buf = pinned(RAW); cudaStream_t st; CU(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    for (Job &j : jobs) if (j.kind >= 1) {
        ngated++; CU(cudaSetDevice(j.card)); CU(cudaMemcpyAsync(buf, j.dst, RAW, cudaMemcpyDeviceToHost, st)); CU(cudaStreamSynchronize(st));
        std::string h, n; int ok; check_bytes(buf, j.oldid >= 0 ? j.oldid : -2, false, h, ok, n); unchanged += ok;
    }
    for (Job &j : jobs) if (j.kind >= 1) arr_set += ld(&j.f->arr);
    wb_ok = wb_model();   // evaluated again at the end of the hold window: the later jobs must still be at wb == 0
    printf("HOLD arr_or_rel_set_during_hold=%d gated_jobs=%d dst_unchanged_during_hold=%d/%d kind0_done_during_hold=%d/%d victim_wb_as_model_during_hold=%d/%d\n", arr_set + rel_set, ngated, unchanged, ngated, p_done, nk0, wb_ok, ngated0);
    std::string extra; if (arr_set + rel_set) extra += " flag_set_during_hold"; if (unchanged != ngated) extra += " dst_changed_during_hold"; if (p_done != nk0) extra += " kind0_blocked_by_hold"; if (wb_ok != ngated0) extra += " victim_wb_not_as_model";
    __atomic_store_n(&hold, 0, __ATOMIC_SEQ_CST);
    // after release: same checks as S1
    std::vector<Job *> by[NCARDS + 1]; for (Job &j : jobs) by[j.kind >= 1 ? j.card : NCARDS].push_back(&j);
    w.deadline = now_s() + 120; for (int c = 0; c <= NCARDS; c++) if (!by[c].empty()) w.waiters.emplace_back(waiter, &w, by[c], c == NCARDS ? -1 : c);
    watch_join_waiters(w);
    return report("S3", jobs, w, extra);
}

// ---------------------------------------------------------------- S4 stress (4 producers)
static int run_s4(uint64_t seed)
{
    Rng r{seed}; Flags *fl; int fd = open_file(g_files + "/rec8.bin");
    for (int i = 0; i < 8; i++) { rec(i); rec(100 + i); } rec(-1); rec(-2); rec(-3);
    const int P = 4, PER = 500; std::vector<Job> jobs; jobs.reserve(P * PER); std::vector<int> mine[P];
    for (int p = 0; p < P; p++) for (int k = 0; k < PER; k++) {
        jobs.emplace_back(); Job &j = jobs.back(); j.id = p * PER + k; const int m = k % 5; j.kind = m == 0 ? 0 : (m == 1 || m == 4) ? 1 : 2; j.card = r.below(NCARDS); j.dev_field = j.card;
        j.filerec = r.below(8); j.srcid = j.filerec; j.victim = j.kind >= 1 && r.below(4) == 0; j.rel = r.below(2); j.oldid = j.victim ? 100 + r.below(8) : -2;
        if (j.kind != 2) j.fd = fd; if (j.kind == 2) j.host_src = pinned_rec(j.filerec); if (j.kind == 0) j.dev_field = 99;
        mine[p].push_back((int)jobs.size() - 1);
    }
    fl = (Flags *)aligned_alloc(64, jobs.size() * sizeof(Flags)); memset(fl, 0, jobs.size() * sizeof(Flags));
    for (size_t i = 0; i < jobs.size(); i++) { jobs[i].f = &fl[i]; alloc_mem(jobs[i], true); }
    printf("SCENARIO S4 seed=%llu producers=%d jobs=%zu\n", (unsigned long long)seed, P, jobs.size());
    init_lib(0, nullptr, nullptr);
    Watch w; watch_start(w, jobs, true, 900, true);
    std::atomic<int> go{0}; std::vector<std::thread> prod;
    for (int p = 0; p < P; p++) prod.emplace_back([&, p]() { while (!go.load()) std::this_thread::yield(); for (int i : mine[p]) enqueue(jobs[i]); });
    go = 1; for (auto &t : prod) t.join();
    watch_join_waiters(w);
    return report("S4", jobs, w, "");
}

// ---------------------------------------------------------------- S5 error paths (each case in a child process)
static int s5_child(const char *cs)
{
    const std::string c = cs; cq_config cfg; memset(&cfg, 0, sizeof cfg); cfg.expert_raw = RAW; cfg.readers = 16; cfg.cards = NCARDS;
    CU(cudaSetDevice(0)); CU(cudaFree(0));
    if (c == "init_badraw") { cfg.expert_raw = 1234; p_init(&cfg); fprintf(stderr, "S5child: cq_init returned\n"); return 0; }
    p_init(&cfg);
    static Flags f; memset(&f, 0, sizeof f); cq_job j; memset(&j, 0, sizeof j); j.p_state = &f.p; j.arr_state = &f.arr; j.wb_state = &f.wb;
    int fd = -1; if (c.find("badfd") != std::string::npos) fd = -1; else fd = open_file(g_files + "/short.bin");
    j.fd = fd; j.off = 0;
    if (c == "k0_badfd" || c == "k0_eof") { j.kind = 0; j.slot = (uint8_t *)aligned_alloc(4096, RAW); }
    else if (c == "k1_badfd" || c == "k1_eof") { j.kind = 1; j.dev = 0; CU(cudaMalloc((void **)&j.dst, RAW)); }
    else if (c == "k2_baddst") { j.kind = 2; j.dev = 0; j.dst = (unsigned char *)0x10; j.host_src = pinned_rec(0); }
    else DIE("unknown S5 case %s", cs);
    p_enq(&j);
    const double dl = now_s() + 30; while (now_s() < dl) { if (ld(&f.p) || ld(&f.arr)) break; usleep(1000); }
    fprintf(stderr, "S5child: no exit; flags p=%d arr=%d\n", ld(&f.p), ld(&f.arr)); return 0;
}
static int run_s5(const char *self, const char *lib)
{
    static const char *cases[] = {"k0_badfd", "k1_badfd", "k0_eof", "k1_eof", "k2_baddst", "init_badraw"};   // first two are the required ones, the rest are extras
    int bad = 0;
    for (const char *cs : cases) {
        int pf[2]; if (pipe(pf)) DIE("pipe");
        const pid_t pid = fork(); if (pid < 0) DIE("fork");
        if (pid == 0) { dup2(pf[1], 2); close(pf[0]); close(pf[1]); const int dn = open("/dev/null", O_WRONLY); dup2(dn, 1); execl(self, self, lib, "S5child", cs, (char *)nullptr); _exit(127); }
        close(pf[1]); std::string err; char b[512]; const double dl = now_s() + 60; bool killed = false;
        for (;;) {
            struct pollfd p = {pf[0], POLLIN, 0}; const int pr = poll(&p, 1, 200);
            if (pr > 0) { const ssize_t n = read(pf[0], b, sizeof b); if (n <= 0) break; err.append(b, (size_t)n); }
            if (now_s() > dl) { kill(pid, SIGKILL); killed = true; break; }
        }
        int status = 0; waitpid(pid, &status, 0); close(pf[0]);
        std::string e; for (char ch : err) { if (ch == '\n') e += "\\n"; else e += ch; }
        printf("S5 %s exit=%s%d stderr=[%s]\n", cs, WIFSIGNALED(status) ? "signal" : "", WIFSIGNALED(status) ? WTERMSIG(status) : WEXITSTATUS(status), e.c_str());
        const bool want1 = WIFEXITED(status) && WEXITSTATUS(status) == 1 && !killed; if (!want1) bad++;
    }
    printf("SUMMARY S5 cases_not_exit_1=%d\n", bad); printf("RESULT %s\n", bad ? "FAIL" : "PASS"); return bad ? 2 : 0;
}

// ---------------------------------------------------------------- timing T1/T2/T3
static int run_timing(const char *name, const std::string &tag)
{
    const int T = name[1] - '0'; const int PER = 120, WARM = 4; const int TOTW = T == 3 ? 32 : NCARDS * WARM;
    int fd = -1; if (T >= 2) fd = open_file(g_tfile);
    for (int i = 0; i < 8; i++) rec(i);
    std::vector<Job> jobs; jobs.reserve(NCARDS * (PER + WARM) + 64);
    uint8_t *slab[NCARDS] = {0};
    if (T == 1) for (int c = 0; c < NCARDS; c++) { CU(cudaSetDevice(c)); slab[c] = pinned(4 * RAW); for (int q = 0; q < 4; q++) memcpy(slab[c] + (size_t)q * RAW, rec(q).buf, RAW); }
    int id = 0; std::vector<int> warm, timed;
    auto add = [&](int kind, int card, int k, bool w) {
        jobs.emplace_back(); Job &j = jobs.back(); j.id = id++; j.kind = kind; j.card = card; j.dev_field = card; j.filerec = T == 1 ? k % 4 : k % 8; j.srcid = j.filerec;
        if (kind == 1) j.fd = fd; if (kind == 0) { j.fd = fd; j.dev_field = card; }
        if (kind == 2) j.host_src = slab[card] + (size_t)j.filerec * RAW;
        (w ? warm : timed).push_back((int)jobs.size() - 1);
    };
    if (T == 3) { for (int k = 0; k < TOTW; k++) add(0, 0, k, true); for (int k = 0; k < PER; k++) for (int c = 0; c < NCARDS; c++) add(0, c, k, false); }
    else { for (int k = 0; k < WARM; k++) for (int c = 0; c < NCARDS; c++) add(T == 1 ? 2 : 1, c, k, true); for (int k = 0; k < PER; k++) for (int c = 0; c < NCARDS; c++) add(T == 1 ? 2 : 1, c, k, false); }
    Flags *fl = (Flags *)aligned_alloc(64, jobs.size() * sizeof(Flags)); memset(fl, 0, jobs.size() * sizeof(Flags));
    for (size_t i = 0; i < jobs.size(); i++) { jobs[i].f = &fl[i]; alloc_mem(jobs[i], true); }
    init_lib(0, nullptr, nullptr);
    auto wait_all = [&](const std::vector<int> &v) { const double dl = now_s() + 300; for (int i : v) { Job &j = jobs[i]; const int *fg = j.kind >= 1 ? &j.f->arr : &j.f->p; while (ld(fg) != 1) { if (now_s() > dl) DIE("timeout"); usleep(100); } } };
    const double w0 = now_s(); for (int i : warm) enqueue(jobs[i]); wait_all(warm); const double warm_s = now_s() - w0;
    struct rusage a, b; getrusage(RUSAGE_SELF, &a); const double t0 = now_s();
    for (int i : timed) enqueue(jobs[i]);
    wait_all(timed); const double t1 = now_s(); getrusage(RUSAGE_SELF, &b);
    auto tv = [](const struct timeval &x) { return (double)x.tv_sec + 1e-6 * (double)x.tv_usec; };
    // verification (not timed): every destination holds its record
    int bad = 0; uint8_t *buf = pinned(RAW); cudaStream_t st; CU(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
    for (Job &j : jobs) {
        if (j.kind >= 1) { CU(cudaSetDevice(j.card)); CU(cudaMemcpyAsync(buf, j.dst, RAW, cudaMemcpyDeviceToHost, st)); CU(cudaStreamSynchronize(st)); bad += memcmp(buf, rec(j.srcid).buf, RAW) != 0; }
        else bad += memcmp(j.slot, rec(j.srcid).buf, RAW) != 0;
    }
    const double wall = t1 - t0, bytes = (double)timed.size() * (double)RAW;
    printf("TIMING %s lib=%s jobs=%zu bytes=%.0f wall_s=%.4f gbps=%.3f cpu_user_s=%.3f cpu_sys_s=%.3f warm_s=%.4f verify_bad=%d\n", name, tag.c_str(), timed.size(), bytes, wall, bytes / 1e9 / wall,
           tv(b.ru_utime) - tv(a.ru_utime), tv(b.ru_stime) - tv(a.ru_stime), warm_s, bad);
    fflush(stdout); return bad ? 2 : 0;
}

// ---------------------------------------------------------------- files / selftest
static int run_mkfiles()
{
    mkdir(g_files.c_str(), 0755); const std::string d2 = g_tfile.substr(0, g_tfile.rfind('/')); { std::string cmd = "mkdir -p " + d2; if (system(cmd.c_str())) DIE("mkdir"); }
    for (const std::string &path : {g_files + "/rec8.bin", g_tfile}) {
        FILE *f = fopen(path.c_str(), "wb"); if (!f) DIE("create %s", path.c_str());
        for (int i = 0; i < 8; i++) if (fwrite(rec(i).buf, 1, RAW, f) != RAW) DIE("write");
        fclose(f);
        int fd = open_file(path); uint8_t *b = (uint8_t *)aligned_alloc(4096, RAW);
        for (int i = 0; i < 8; i++) { size_t got = 0; while (got < RAW) { ssize_t n = pread(fd, b + got, RAW - got, (off_t)i * (off_t)RAW + (off_t)got); if (n <= 0) DIE("readback"); got += (size_t)n; } if (memcmp(b, rec(i).buf, RAW)) DIE("file %s record %d differs from generator", path.c_str(), i); }
        close(fd); printf("file %s: 8 records written and read back equal to the generator\n", path.c_str());
    }
    FILE *f = fopen((g_files + "/short.bin").c_str(), "wb"); uint8_t z[100]; memset(z, 7, 100); fwrite(z, 1, 100, f); fclose(f); printf("file short.bin: 100 bytes\n"); return 0;
}
static int run_selftest()
{
    const char *abc = "abc"; const std::string a = sha256_hex((const uint8_t *)abc, 3); const std::string e = sha256_hex((const uint8_t *)"", 0);
    uint8_t m[1000]; memset(m, 'a', sizeof m); const std::string m1 = sha256_hex(m, 1000);
    printf("sha256(abc)=%s %s\nsha256()=%s %s\nsha256(a*1000)=%s\n", a.c_str(), a == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" ? "OK" : "WRONG", e.c_str(), e == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" ? "OK" : "WRONG", m1.c_str());
    for (int i : {0, 3}) printf("rec %d sha256=%s\n", i, rec(i).sha.c_str());
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: %s <lib.so|-> <scenario> [--seed N] [--tag T]\n", argv[0]); return 3; }
    uint64_t seed = 1; std::string tag = argv[1]; const char *scen = argv[2];
    for (int i = 3; i < argc; i++) { if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], 0, 10); else if (!strcmp(argv[i], "--tag") && i + 1 < argc) tag = argv[++i]; }
    if (getenv("CQ_FILES")) g_files = getenv("CQ_FILES"); if (getenv("CQ_TFILE")) g_tfile = getenv("CQ_TFILE");
    const std::string s = scen;
    if (s == "mkfiles") return run_mkfiles();
    if (s == "selftest") return run_selftest();
    if (strcmp(argv[1], "-")) load_lib(argv[1]); else DIE("a library is required for %s", scen);
    if (s == "S1") return run_s1(seed);
    if (s == "S2") return run_s2(seed);
    if (s == "S3") return run_s3(seed);
    if (s == "S4") return run_s4(seed);
    if (s == "S5") return run_s5(argv[0], argv[1]);
    if (s == "S5child") return s5_child(argc > 3 ? argv[3] : "");
    if (s == "T1" || s == "T2" || s == "T3") return run_timing(scen, tag);
    DIE("unknown scenario %s", scen);
}
