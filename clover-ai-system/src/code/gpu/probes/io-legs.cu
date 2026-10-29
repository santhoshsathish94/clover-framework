// The bus legs: host to card, card to host, card to card, for one card or several at the same time.
//
//   io-legs h2d CARDS pinned|pageable [GIB=2]     each card copies GIB gibibytes from host memory, 64 MiB at a time, two copies in flight
//   io-legs d2h CARDS [GIB=2]                     the other way, from the card to pinned host memory
//   io-legs p2p                                   card to card: 0->2 (NVLink pair), 0->1 (same PCIe switch), 0->4 (across the host bridge), then all cards at once in a ring
//
// Run under numactl to place the host memory: numactl --cpunodebind=0 --membind=0 io-legs h2d 8 pinned
// build: nvcc -O3 -std=c++14 -arch=sm_86 -o io-legs io-legs.cu -lpthread
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static const size_t CHUNK = 64u << 20;

static void host_device(const char *mode, int cards, bool pinned, double gib)
{
    const size_t total = (size_t)(gib * 1073741824.0) / CHUNK * CHUNK;
    std::vector<double> secs(cards, 0.0);
    std::atomic<int> ready(0), go(0);
    std::vector<std::thread> th;
    for (int c = 0; c < cards; c++) th.emplace_back([&, c]() {
        CHECK(cudaSetDevice(c));
        unsigned char *dev; CHECK(cudaMalloc(&dev, total));
        unsigned char *host[2];
        for (int i = 0; i < 2; i++) { if (pinned) CHECK(cudaMallocHost(&host[i], CHUNK)); else host[i] = (unsigned char *)malloc(CHUNK); memset(host[i], 3, CHUNK); }
        cudaStream_t s[2]; for (int i = 0; i < 2; i++) CHECK(cudaStreamCreate(&s[i]));
        const bool h2d = !strcmp(mode, "h2d");
        ready++; while (!go) std::this_thread::yield();
        const double t0 = now();
        for (size_t off = 0, k = 0; off < total; off += CHUNK, k++) {
            const int i = (int)(k & 1);
            CHECK(cudaStreamSynchronize(s[i]));
            if (h2d) CHECK(cudaMemcpyAsync(dev + off, host[i], CHUNK, cudaMemcpyHostToDevice, s[i]));
            else CHECK(cudaMemcpyAsync(host[i], dev + off, CHUNK, cudaMemcpyDeviceToHost, s[i]));
        }
        for (int i = 0; i < 2; i++) CHECK(cudaStreamSynchronize(s[i]));
        secs[c] = now() - t0;
    });
    while (ready < cards) std::this_thread::yield();
    const double t0 = now(); go = 1;
    for (auto &t : th) t.join();
    const double all = now() - t0;
    std::string per;
    for (int c = 0; c < cards; c++) { char b[32]; snprintf(b, sizeof b, "%s%.2f", c ? " " : "", total / 1e9 / secs[c]); per += b; }
    printf("%s, %d card(s), %s host memory: per card GB/s [%s], in all %.2f GB/s\n", mode, cards, pinned ? "pinned" : "pageable", per.c_str(), cards * (double)total / 1e9 / all);
}

static double peer(int from, int to, size_t bytes)
{
    CHECK(cudaSetDevice(from)); unsigned char *a; CHECK(cudaMalloc(&a, bytes));
    CHECK(cudaSetDevice(to)); unsigned char *b; CHECK(cudaMalloc(&b, bytes));
    CHECK(cudaSetDevice(from));
    cudaMemcpyPeer(b, to, a, from, bytes); CHECK(cudaDeviceSynchronize());
    const double t0 = now();
    for (int i = 0; i < 5; i++) CHECK(cudaMemcpyPeer(b, to, a, from, bytes));
    CHECK(cudaDeviceSynchronize());
    const double t = (now() - t0) / 5;
    cudaFree(a); CHECK(cudaSetDevice(to)); cudaFree(b);
    return bytes / 1e9 / t;
}

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: io-legs h2d CARDS pinned|pageable [GIB] | d2h CARDS [GIB] | p2p\n"); return 2; }
    const std::string cmd = argv[1];
    int have = 0; CHECK(cudaGetDeviceCount(&have));
    if (cmd == "h2d" && argc >= 4) host_device("h2d", atoi(argv[2]), !strcmp(argv[3], "pinned"), argc > 4 ? atof(argv[4]) : 2.0);
    else if (cmd == "d2h" && argc >= 3) host_device("d2h", atoi(argv[2]), true, argc > 3 ? atof(argv[3]) : 2.0);
    else if (cmd == "p2p") {
        const size_t bytes = 512u << 20;
        printf("card to card, %zu MiB, one pair at a time: 0->2 %.1f GB/s, 0->1 %.1f GB/s, 0->4 %.1f GB/s, 0->7 %.1f GB/s\n", bytes >> 20, peer(0, 2, bytes), peer(0, 1, bytes), peer(0, 4, bytes), peer(0, 7, bytes));
        std::vector<std::thread> th; std::vector<double> r(have);
        std::atomic<int> go(0);
        for (int c = 0; c < have; c++) th.emplace_back([&, c]() { while (!go) std::this_thread::yield(); r[c] = peer(c, (c + 1) % have, bytes); });
        go = 1; for (auto &t : th) t.join();
        printf("all %d cards at once, each to the next in a ring:", have); for (int c = 0; c < have; c++) printf(" %.1f", r[c]); printf(" GB/s\n");
    } else { fprintf(stderr, "bad arguments\n"); return 2; }
    return 0;
}
