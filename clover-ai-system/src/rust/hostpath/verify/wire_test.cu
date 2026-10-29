// Wiring test of the Rust host path inside the engine's own code: backend-gpu.cu is included, so configure(), pool_open(), route_hook, pool_ensure, loader_submit/loader_wait,
// print_summary and every K3GPU_* selector are the ones the engine runs. Real CUDA (one card), a synthetic experts file, no model and no whole-model driver.
//
//   wire_test cfg                          configure() and the exit summary only (compare the stderr of K3GPU_ENGCFG=cpp and =rust)
//   wire_test ensure FILE CALLS SEED       CALLS scripted calls: pool_ensure on card 0, the bytes of every expert checked against the file, one result line
//   wire_test plan FILE CALLS SEED         the same with a route cache made from the script, route_hook before every call (the planner, the copy queue, the card queues)
//   wire_test loader                       jobs on four cards through loader_submit / loader_wait: totals and per-card order
// Settings as the engine's (K3GPU_HOST_GB, K3GPU_RESERVE_MB, K3GPU_POLICY, K3GPU_VQUEUE, K3GPU_COPYQ ... K3GPU_ENGCFG, ...). The experts file is created if missing (48 records of 17,547,264 bytes).
// build (box): nvcc -O3 --fmad=false -arch=sm_86 -std=c++17 -DEXPERT_V2 -DINT8_PLANES -I<gpu> -I<common> -I<crate include dirs> wire_test.cu -lcublas -ldl -lpthread -o wire_test
#include "../../../code/gpu/backend-gpu.cu"
#include <sys/stat.h>
#include <atomic>
#include <chrono>

enum { NE = 48 };
static uint64_t mix(uint64_t x) { x += 0x9e3779b97f4a7c15ULL; x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL; x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL; return x ^ (x >> 31); }
static void expected(unsigned char *buf, int e, size_t off, size_t len)
{
    for (size_t k = 0; k < len; k++) { const size_t p = off + k; const uint64_t w = mix((uint64_t)e * 1000003ULL + p / 8); buf[k] = (unsigned char)(w >> (8 * (p % 8))); }
}
static void make_file(const char *path)
{
    struct stat st;
    if (stat(path, &st) == 0 && (size_t)st.st_size == (size_t)NE * EXPERT_RAW) return;
    FILE *f = fopen(path, "wb"); if (!f) { perror(path); exit(2); }
    std::vector<uint64_t> w(EXPERT_RAW / 8);
    for (int e = 0; e < NE; e++) {
        for (size_t i = 0; i < w.size(); i++) w[i] = mix((uint64_t)e * 1000003ULL + i);
        if (fwrite(w.data(), 8, w.size(), f) != w.size()) { perror("write"); exit(2); }
    }
    fclose(f);
}

struct Rng { uint64_t s; unsigned next() { s = mix(s); return (unsigned)(s >> 33); } };
struct ScriptCall { int layer; std::vector<int> uniq; std::vector<int> count; };   // count[e], EXPERTS entries
static std::vector<ScriptCall> make_script(unsigned seed, int ncalls)
{
    Rng r{ 0x1234567ULL + seed }; std::vector<ScriptCall> v;
    for (int i = 0; i < ncalls; i++) {
        ScriptCall c; c.layer = 1 + (i % 2); c.count.assign(EXPERTS, 0);
        const int n = 1 + (int)(r.next() % 12);
        // a mix of recent experts (hits) and new ones, so every path runs: card hits, host hits, drive reads, victims
        while ((int)c.uniq.size() < n) {
            int e = (r.next() % 3 == 0 && !v.empty()) ? v[r.next() % v.size()].uniq[0] % NE : (int)(r.next() % NE);
            if (!c.count[e]) { c.count[e] = 1 + (int)(r.next() % 8); c.uniq.push_back(e); }
        }
        std::sort(c.uniq.begin(), c.uniq.end());
        v.push_back(std::move(c));
    }
    return v;
}

static int run_calls(const char *file, int ncalls, unsigned seed, bool plan)
{
    const std::vector<ScriptCall> script = make_script(seed, ncalls);
    make_file(file);
    if (plan) {   // the route cache is the script itself (what an earlier run of the same request would have traced)
        const char *rc = "/tmp/wt-routecache.txt"; FILE *f = fopen(rc, "w"); if (!f) { perror(rc); return 2; }
        for (auto &c : script) { fprintf(f, "%d 1", c.layer); for (int e : c.uniq) fprintf(f, " %d:%d", e, c.count[e]); fputc('\n', f); }
        fclose(f); setenv("K3GPU_ROUTECACHE", rc, 1);
    }
    configure();
    const int fd = open(file, O_RDONLY | O_DIRECT); if (fd < 0) { perror(file); return 2; }
    static Handle hs[3];
    for (int L = 1; L <= 2; L++) { hs[L].L = L; hs[L].card = 0; hs[L].fd = fd; g_handles[L] = &hs[L]; plan_note_layer(&hs[L]); }
    pool_open(0);
    CardPool &pool = g_pool[0];
    uint64_t hash = 0; long bad = 0; std::vector<unsigned char> got(4096), want(4096);
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < ncalls; i++) {
        const ScriptCall &c = script[i];
        if (g_route_hook) g_route_hook(c.layer, c.uniq);
        pool_ensure(pool, g_host, c.layer, fd, c.uniq, c.count.data(), g_readers);
        for (int e : c.uniq) {
            const unsigned char *p = pool.h_eptr[e];
            hash = mix(hash ^ ((uint64_t)i << 40 ^ (uint64_t)e << 20 ^ (uint64_t)((p - pool.base) / EXPERT_RAW)));
            for (int k = 0; k <= 32; k++) {
                const size_t off = k == 32 ? (size_t)EXPERT_RAW - 4096 : (size_t)k * (EXPERT_RAW / 32 / 8 * 8);
                CHECK(cudaMemcpy(got.data(), p + off, 4096, cudaMemcpyDeviceToHost));
                expected(want.data(), e, off, 4096);
                if (memcmp(got.data(), want.data(), 4096)) { if (bad++ < 5) fprintf(stderr, "MISMATCH call %d layer %d expert %d window %d\n", i, c.layer, e, k); }
            }
        }
    }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    planner_sync();
    const ExpertStats &s = pool.st;
    printf("RESULT calls=%d hash=%016llx mismatches=%ld needed=%lu hit_card=%lu hit_host=%lu from_drive=%lu written_back=%lu wb_skipped=%lu planned=%lu skipped=%lu pf_used=%lu vq_ram=%lu vq_drive=%lu vq_wb=%lu vq_nowb=%lu rc_ok=%d pos=%zu (%.1f s)\n",
           ncalls, (unsigned long long)hash, bad, s.needed, s.hit_card, s.hit_host, s.from_drive, s.written_back, wb_skipped_total(), g_pf_planned, g_pf_skipped, g_pf_used,
           g_vq_ram, g_vq_drive, g_vq_wb, g_vq_nowb, g_rc_ok ? 1 : 0, g_rc_pos, secs);
    fflush(stdout);
    return bad ? 1 : 0;
}

static int run_loader()
{
    configure();
    enum { CARDS = 4, JOBS = 4000 };
    static std::vector<int> order[CARDS]; static std::atomic<long> total(0); static std::atomic<int> threads_seen[CARDS];
    static std::atomic<unsigned long> tid_of[CARDS];
    const std::hash<std::thread::id> hh;
    for (int i = 0; i < JOBS; i++) {
        const int card = i % CARDS;
        loader_submit(card, [card, i]() {
            const unsigned long t = (unsigned long)std::hash<std::thread::id>()(std::this_thread::get_id());
            unsigned long z = 0; if (!tid_of[card].compare_exchange_strong(z, t) && z != t) threads_seen[card]++;
            if (i % 50 == 0) usleep(200);
            order[card].push_back(i); total++;
        });
        if (i % 1000 == 999) loader_wait();   // a wait in the middle: it returns only when every job so far has finished
        if (i % 1000 == 999 && total.load() != i + 1) { printf("LOADER wait returned with %ld of %d jobs done\n", total.load(), i + 1); return 1; }
    }
    loader_wait();
    long viol = 0, extra_threads = 0;
    for (int c = 0; c < CARDS; c++) { for (size_t k = 1; k < order[c].size(); k++) if (order[c][k] <= order[c][k - 1]) viol++; extra_threads += threads_seen[c].load(); }
    (void)hh;
    printf("LOADER total=%ld order_violations=%ld extra_threads=%ld per_card=%zu/%zu/%zu/%zu\n", total.load(), viol, extra_threads, order[0].size(), order[1].size(), order[2].size(), order[3].size());
    return viol || total.load() != JOBS || extra_threads ? 1 : 0;
}

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: wire_test cfg | ensure FILE CALLS SEED | plan FILE CALLS SEED | loader\n"); return 2; }
    const std::string mode = argv[1];
    if (mode == "cfg") { configure(); return 0; }
    if (mode == "loader") return run_loader();
    if ((mode == "ensure" || mode == "plan") && argc >= 5) return run_calls(argv[2], atoi(argv[3]), (unsigned)atoi(argv[4]), mode == "plan");
    fprintf(stderr, "bad arguments\n"); return 2;
}
