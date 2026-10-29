// How fast one layer's 896 experts (15.72 GB) can be supplied to the cards: the cost of the cycle "each card gets the next layer's experts when it has finished its layer".
//
//   expert-supply MODE CARDS READERS DATASET_DIR [FIRST_LAYER]
//
// MODE     disk   read experts.direct with O_DIRECT (no page cache) from the NVMe, copy to the card
//          cache  buffered read (whatever of the file is in the page cache is read from memory), copy to the card
//          pcie   no file: copy from pinned host memory to the card, to see the bus alone
// CARDS    how many cards load at the same time; card c takes layer FIRST_LAYER + c (default FIRST_LAYER 1), its own file
// READERS  reader threads per card, each with its own 64 MiB pinned buffer and stream
//
// Prints each card's time and the total. Reads files, writes nothing.
// build: nvcc -O3 -std=c++14 -arch=sm_86 -o expert-supply expert-supply.cu -lpthread
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <fcntl.h>
#include <unistd.h>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

static const size_t LAYER_BYTES = (size_t)896 * 17547264, CHUNK = 64u << 20;
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char **argv)
{
    if (argc < 5) { fprintf(stderr, "usage: expert-supply disk|cache|pcie CARDS READERS DATASET_DIR [FIRST_LAYER]\n"); return 2; }
    const std::string mode = argv[1];
    const int cards = atoi(argv[2]), readers = atoi(argv[3]), first = argc > 5 ? atoi(argv[5]) : 1;
    int have = 0; CHECK(cudaGetDeviceCount(&have));
    if (cards < 1 || cards > have || readers < 1) { fprintf(stderr, "bad cards or readers (%d cards present)\n", have); return 2; }
    std::vector<double> seconds(cards, 0.0);
    std::vector<unsigned char *> dev(cards);
    for (int c = 0; c < cards; c++) { CHECK(cudaSetDevice(c)); CHECK(cudaMalloc(&dev[c], LAYER_BYTES)); }
    std::atomic<int> ready(0), go(0);
    auto card_main = [&](int c) {
        CHECK(cudaSetDevice(c));
        char path[4096]; snprintf(path, sizeof path, "%s/expert-%d/experts.direct", argv[4], first + c);
        int fd = -1;
        if (mode != "pcie") {
            fd = open(path, O_RDONLY | (mode == "disk" ? O_DIRECT : 0));
            if (fd < 0) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
        }
        std::atomic<size_t> next(0);
        const size_t chunks = LAYER_BYTES / CHUNK + (LAYER_BYTES % CHUNK != 0);
        std::vector<std::thread> workers;
        ready++;
        while (!go) std::this_thread::yield();
        const double t0 = now();
        for (int r = 0; r < readers; r++) workers.emplace_back([&, r]() {
            CHECK(cudaSetDevice(c));
            unsigned char *pinned; CHECK(cudaMallocHost(&pinned, CHUNK));
            cudaStream_t s; CHECK(cudaStreamCreate(&s));
            for (size_t i; (i = next++) < chunks; ) {
                const size_t off = i * CHUNK, n = LAYER_BYTES - off < CHUNK ? LAYER_BYTES - off : CHUNK;
                if (fd >= 0) { size_t got = 0; while (got < n) { ssize_t k = pread(fd, pinned + got, n - got, (off_t)(off + got)); if (k <= 0) { fprintf(stderr, "read failed at %zu\n", off + got); exit(1); } got += (size_t)k; } }
                CHECK(cudaMemcpyAsync(dev[c] + off, pinned, n, cudaMemcpyHostToDevice, s));
                CHECK(cudaStreamSynchronize(s));
            }
            cudaStreamDestroy(s); cudaFreeHost(pinned);
        });
        for (auto &w : workers) w.join();
        seconds[c] = now() - t0;
        if (fd >= 0) close(fd);
    };
    std::vector<std::thread> threads;
    for (int c = 0; c < cards; c++) threads.emplace_back(card_main, c);
    while (ready < cards) std::this_thread::yield();
    const double t0 = now(); go = 1;
    for (auto &t : threads) t.join();
    const double total = now() - t0;
    double slowest = 0;
    for (int c = 0; c < cards; c++) { printf("card %d: %.2f s (%.2f GB/s)\n", c, seconds[c], LAYER_BYTES / 1e9 / seconds[c]); if (seconds[c] > slowest) slowest = seconds[c]; }
    printf("%s, %d card(s), %d reader(s) per card: %.2f s for %d layer(s) = %.2f GB/s in all, %.2f s per layer on average\n", mode.c_str(), cards, readers, total, cards,
           cards * LAYER_BYTES / 1e9 / total, total / cards);
    return 0;
}
