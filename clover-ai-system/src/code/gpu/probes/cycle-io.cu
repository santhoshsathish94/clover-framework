// The data path of the eight-card cycle with no computing: layers' experts (and optionally their trunk) go disk -> pinned RAM slots -> the card that owns the layer,
// one layer after another, while a stand-in for the computing takes COMPUTE_MS per layer in layer order. Measures what the cards would wait for.
//
//   cycle-io DATASET_DIR LAYERS SLOTS READERS COMPUTE_MS [TRUNK=0|1] [FIRST_LAYER=1]
// Environment: CYCLE_DIRECT=1  read the experts files with O_DIRECT (no page cache)
//              CYCLE_PASSES=2  with SLOTS >= LAYERS the first pass fills the RAM slots from disk and keeps them; the later passes take every layer from RAM only
//
// Layer L goes to card (L mod 8). A card's expert buffer is free again when the stand-in has finished the layer that used it.
// Reported per pass: the time to the first layer, the time per layer after it, and what the stand-in waited for the experts (the cards' idle time).
//
// build: nvcc -O3 -std=c++14 -arch=sm_86 -o cycle-io cycle-io.cu -lpthread
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static const size_t LAYER_BYTES = (size_t)896 * 17547264;

static size_t read_all(const std::string &path, unsigned char *dst, size_t cap, bool direct)
{
    int fd = open(path.c_str(), O_RDONLY | (direct ? O_DIRECT : 0));
    if (fd < 0) return 0;
    struct stat st; fstat(fd, &st);
    size_t n = (size_t)st.st_size < cap ? (size_t)st.st_size : cap, got = 0;
    while (got < n) { ssize_t k = pread(fd, dst + got, n - got < (64u << 20) ? n - got : (64u << 20), (off_t)got); if (k <= 0) break; got += (size_t)k; }
    close(fd);
    return got;
}

int main(int argc, char **argv)
{
    if (argc < 6) { fprintf(stderr, "usage: cycle-io DATASET_DIR LAYERS SLOTS READERS COMPUTE_MS [TRUNK] [FIRST_LAYER]\n"); return 2; }
    const std::string dataset = argv[1];
    const int K = atoi(argv[2]), SLOTS = atoi(argv[3]), READERS = atoi(argv[4]);
    const double compute = atof(argv[5]) / 1000.0;
    const bool with_trunk = argc > 6 && atoi(argv[6]);
    const int first = argc > 7 ? atoi(argv[7]) : 1;
    const bool direct = getenv("CYCLE_DIRECT") && atoi(getenv("CYCLE_DIRECT"));
    const int passes = getenv("CYCLE_PASSES") ? atoi(getenv("CYCLE_PASSES")) : 1;
    const bool resident = passes > 1;
    if (resident && SLOTS < K) { fprintf(stderr, "CYCLE_PASSES > 1 needs SLOTS >= LAYERS\n"); return 2; }
    const int G = 8;
    std::vector<unsigned char *> dev(G), dev_trunk(G);
    const size_t trunk_cap = 700u << 20;
    for (int g = 0; g < G; g++) { CHECK(cudaSetDevice(g)); CHECK(cudaMalloc(&dev[g], LAYER_BYTES)); CHECK(cudaMalloc(&dev_trunk[g], trunk_cap)); }
    std::vector<unsigned char *> slot_buf(SLOTS), slot_trunk(SLOTS);
    const double tp = now();
    for (int s = 0; s < SLOTS; s++) { CHECK(cudaMallocHost(&slot_buf[s], LAYER_BYTES)); CHECK(cudaMallocHost(&slot_trunk[s], trunk_cap)); }
    printf("%d pinned slots: %.1f s\n", SLOTS, now() - tp);

    std::mutex mu; std::condition_variable cv;
    std::vector<int> slot_layer(SLOTS, -1); std::vector<char> slot_ready(SLOTS, 0);
    std::vector<size_t> slot_trunk_bytes(SLOTS, 0);
    for (int pass = 0; pass < passes; pass++) {
        const bool from_disk = !(resident && pass > 0);
        std::vector<char> exp_ready(K, 0), buf_free(G, 1);
        std::vector<double> t_ready(K, 0), t_read_start(K, 0), t_read_end(K, 0), waited(K, 0.0);
        std::atomic<int> next_read(0);
        std::atomic<double> trunk_bytes_total(0);
        const double t0 = now();
        std::vector<std::thread> th;
        if (from_disk) for (int r = 0; r < READERS; r++) th.emplace_back([&]() {
            for (;;) {
                const int i = next_read++; if (i >= K) return;
                const int L = first + i; int s = -1;
                { std::unique_lock<std::mutex> lk(mu);
                  if (resident) s = i;
                  else {
                      cv.wait(lk, [&]() { for (int j = 0; j < SLOTS; j++) if (slot_layer[j] == -1) return true; return false; });
                      for (int j = 0; j < SLOTS; j++) if (slot_layer[j] == -1) { s = j; break; }
                  }
                  slot_layer[s] = -2; }
                t_read_start[i] = now() - t0;
                char p[4096]; snprintf(p, sizeof p, "%s/expert-%d/experts.direct", dataset.c_str(), L);
                if (read_all(p, slot_buf[s], LAYER_BYTES, direct) != LAYER_BYTES) { fprintf(stderr, "read of %s failed\n", p); exit(1); }
                size_t tb = 0;
                if (with_trunk) {
                    const char *names[5] = { "common.bin", "attention.bin", "routed-shared.bin", "operator.bin", "index.bin" };
                    for (int f = 0; f < 5; f++) { snprintf(p, sizeof p, "%s/trunk-%d/%s", dataset.c_str(), L, names[f]); tb += read_all(p, slot_trunk[s] + tb, trunk_cap - tb, false); }
                }
                t_read_end[i] = now() - t0;
                { std::lock_guard<std::mutex> lk(mu); slot_layer[s] = L; slot_ready[s] = 1; slot_trunk_bytes[s] = tb; }
                cv.notify_all();
            }
        });
        for (int g = 0; g < G; g++) th.emplace_back([&, g]() {
            CHECK(cudaSetDevice(g));
            cudaStream_t st; CHECK(cudaStreamCreate(&st));
            for (int i = 0; i < K; i++) {
                const int L = first + i; if (L % G != g) continue;
                int s = -1;
                { std::unique_lock<std::mutex> lk(mu);
                  cv.wait(lk, [&]() { if (!buf_free[g]) return false; for (int j = 0; j < SLOTS; j++) if (slot_layer[j] == L && slot_ready[j]) return true; return false; });
                  for (int j = 0; j < SLOTS; j++) if (slot_layer[j] == L && slot_ready[j]) { s = j; break; }
                  buf_free[g] = 0; }
                const size_t chunk = 256u << 20;
                for (size_t off = 0; off < LAYER_BYTES; off += chunk) CHECK(cudaMemcpyAsync(dev[g] + off, slot_buf[s] + off, LAYER_BYTES - off < chunk ? LAYER_BYTES - off : chunk, cudaMemcpyHostToDevice, st));
                if (slot_trunk_bytes[s]) CHECK(cudaMemcpyAsync(dev_trunk[g], slot_trunk[s], slot_trunk_bytes[s], cudaMemcpyHostToDevice, st));
                CHECK(cudaStreamSynchronize(st));
                { std::lock_guard<std::mutex> lk(mu); trunk_bytes_total = trunk_bytes_total + (double)slot_trunk_bytes[s];
                  if (!resident) { slot_layer[s] = -1; slot_ready[s] = 0; }
                  exp_ready[i] = 1; t_ready[i] = now() - t0; }
                cv.notify_all();
            }
        });
        double t_stand0 = 0, t_stand1 = 0;
        th.emplace_back([&]() {   // the stand-in for the computing: layers in order, COMPUTE_MS each, once the layer's experts are on the card
            for (int i = 0; i < K; i++) {
                const double asked = now();
                { std::unique_lock<std::mutex> lk(mu); cv.wait(lk, [&]() { return exp_ready[i]; }); }
                if (i == 0) t_stand0 = now() - t0;
                waited[i] = now() - asked;
                std::this_thread::sleep_for(std::chrono::duration<double>(compute));
                { std::lock_guard<std::mutex> lk(mu); buf_free[(first + i) % G] = 1; }
                cv.notify_all();
            }
            t_stand1 = now() - t0;
        });
        for (auto &t : th) t.join();
        const double total = now() - t0;
        double wait_sum = 0; for (int i = 1; i < K; i++) wait_sum += waited[i];
        printf("\npass %d (%s): %d layers (%d..%d)%s, %d slots, %d reader(s)%s, stand-in compute %.0f ms per layer\n", pass + 1, from_disk ? "disk to RAM to cards" : "RAM to cards only", K, first, first + K - 1,
               with_trunk ? " with their trunk" : "", SLOTS, READERS, direct ? ", O_DIRECT" : ", buffered", compute * 1000);
        printf("first layer's experts on its card after %.2f s; all %d layers done at %.2f s (%.2f s per layer after the first); %.1f GB moved = %.2f GB/s\n", t_stand0, K, t_stand1,
               K > 1 ? (t_stand1 - t_stand0) / (K - 1) : 0.0, (K * (double)LAYER_BYTES + trunk_bytes_total) / 1e9, (K * (double)LAYER_BYTES + trunk_bytes_total) / 1e9 / total);
        printf("the stand-in waited for experts %.2f s of the %.2f s after the first layer: it computed %.0f%% of the time\n", wait_sum, t_stand1 - t_stand0, 100.0 * (K - 1) * compute / (t_stand1 - t_stand0));
        if (K <= 16) {
            printf("layer: read started, read done, on the card, waited (s):");
            for (int i = 0; i < K; i++) { if (i % 4 == 0) printf("\n "); printf(" %d: %.1f %.1f %.1f %.2f |", first + i, t_read_start[i], t_read_end[i], t_ready[i], waited[i]); }
            printf("\n");
        }
    }
    return 0;
}
