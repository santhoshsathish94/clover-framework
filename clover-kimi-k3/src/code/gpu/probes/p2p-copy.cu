// Card-to-card and host-to-card copy rates on the gpu box. Every copy is checked on the destination.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

static const size_t SZ = 256ull << 20;
static const size_t NW = SZ / 4;
static const int WARM = 2, REPS = 10;
static int n;
static std::vector<unsigned *> src, dst;
static std::vector<cudaStream_t> st;
static unsigned *hsrc;

__global__ void fill(unsigned *p, size_t n, unsigned seed) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        p[i] = (unsigned)(i * 2654435761u) ^ (seed * 0x9E3779B9u);
}
__global__ void check(const unsigned *p, size_t n, unsigned seed, unsigned long long *bad) {
    unsigned long long b = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        b += p[i] != ((unsigned)(i * 2654435761u) ^ (seed * 0x9E3779B9u));
    if (b) atomicAdd(bad, b);
}

static unsigned long long verify(int dev, unsigned seed) {
    CK(cudaSetDevice(dev));
    unsigned long long *b;
    CK(cudaMalloc(&b, 8));
    CK(cudaMemset(b, 0, 8));
    check<<<256, 256>>>(dst[dev], NW, seed, b);
    unsigned long long h;
    CK(cudaMemcpy(&h, b, 8, cudaMemcpyDeviceToHost));
    CK(cudaFree(b));
    CK(cudaMemset(dst[dev], 0, SZ));
    return h;
}

// a copy job: s = -1 means host pinned buffer; d = destination card
struct Job { int s, d; };

// runs all jobs concurrently (one stream each, on the destination card for H2D, the source card for D2D), REPS copies per job
// returns aggregate GB/s by wall clock and fills per-job GB/s
static double run(const std::vector<Job> &jobs, std::vector<double> *per = nullptr) {
    std::vector<cudaStream_t> ss(jobs.size());
    std::vector<cudaEvent_t> e0(jobs.size()), e1(jobs.size());
    for (size_t k = 0; k < jobs.size(); k++) {
        int owner = jobs[k].s < 0 ? jobs[k].d : jobs[k].s;
        CK(cudaSetDevice(owner));
        CK(cudaStreamCreateWithFlags(&ss[k], cudaStreamNonBlocking));
        CK(cudaEventCreate(&e0[k])); CK(cudaEventCreate(&e1[k]));
    }
    auto issue = [&](int reps, bool timed) {
        for (size_t k = 0; k < jobs.size(); k++) {
            const Job &j = jobs[k];
            if (timed) CK(cudaEventRecord(e0[k], ss[k]));
            for (int r = 0; r < reps; r++) {
                if (j.s < 0) CK(cudaMemcpyAsync(dst[j.d], hsrc, SZ, cudaMemcpyHostToDevice, ss[k]));
                else CK(cudaMemcpyPeerAsync(dst[j.d], j.d, src[j.s], j.s, SZ, ss[k]));
            }
            if (timed) CK(cudaEventRecord(e1[k], ss[k]));
        }
        for (auto &s : ss) CK(cudaStreamSynchronize(s));
    };
    issue(WARM, false);
    auto t0 = std::chrono::steady_clock::now();
    issue(REPS, true);
    double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (per) per->assign(jobs.size(), 0.0);
    for (size_t k = 0; k < jobs.size(); k++) {
        float ms; CK(cudaEventElapsedTime(&ms, e0[k], e1[k]));
        if (per) (*per)[k] = REPS * SZ / (ms * 1e-3) / 1e9;
        CK(cudaStreamDestroy(ss[k])); CK(cudaEventDestroy(e0[k])); CK(cudaEventDestroy(e1[k]));
    }
    return jobs.size() * REPS * SZ / wall / 1e9;
}

static void matrix(const char *title) {
    printf("\n== %s: one copy at a time, %d x %zu MiB, GB/s (row = source, column = destination), bad words after each\n", title, REPS, SZ >> 20);
    unsigned long long badsum = 0;
    printf("      "); for (int j = 0; j < n; j++) printf("   c%d  ", j); printf("\n");
    for (int i = 0; i < n; i++) {
        printf("c%d    ", i);
        for (int j = 0; j < n; j++) {
            if (i == j) { printf("   -   "); continue; }
            double g = run({{i, j}});
            badsum += verify(j, i);
            printf("%6.2f ", g);
        }
        printf("\n");
    }
    printf("bad words over the whole matrix: %llu\n", badsum);
}

static void report(const char *name, const std::vector<Job> &jobs) {
    std::vector<double> per;
    double agg = run(jobs, &per);
    unsigned long long bad = 0;
    // a card that receives from two jobs would make the check meaningless; tests below never do that
    for (auto &j : jobs) bad += verify(j.d, j.s < 0 ? 1000u : (unsigned)j.s);
    printf("%-44s aggregate %6.2f GB/s  per job:", name, agg);
    for (size_t k = 0; k < jobs.size(); k++) printf(" %s%d->%d %.2f", jobs[k].s < 0 ? "h" : "c", jobs[k].s < 0 ? 0 : jobs[k].s, jobs[k].d, per[k]);
    printf("  bad %llu\n", bad);
}

int main() {
    CK(cudaGetDeviceCount(&n));
    printf("cards %d, copy size %zu MiB, %d timed copies per job after %d warm-up\n", n, SZ >> 20, REPS, WARM);
    src.resize(n); dst.resize(n);
    for (int d = 0; d < n; d++) {
        CK(cudaSetDevice(d));
        CK(cudaMalloc(&src[d], SZ)); CK(cudaMalloc(&dst[d], SZ));
        fill<<<256, 256>>>(src[d], NW, d);
        CK(cudaMemset(dst[d], 0, SZ));
        CK(cudaDeviceSynchronize());
    }
    CK(cudaHostAlloc(&hsrc, SZ, cudaHostAllocPortable));
    for (size_t i = 0; i < NW; i++) hsrc[i] = (unsigned)(i * 2654435761u) ^ (1000u * 0x9E3779B9u);

    printf("\n== cudaDeviceCanAccessPeer (row = from, column = to)\n      ");
    for (int j = 0; j < n; j++) printf("c%d ", j); printf("\n");
    std::vector<std::vector<int>> can(n, std::vector<int>(n, 0));
    for (int i = 0; i < n; i++) {
        printf("c%d    ", i);
        for (int j = 0; j < n; j++) { if (i != j) CK(cudaDeviceCanAccessPeer(&can[i][j], i, j)); printf(" %s ", i == j ? "-" : (can[i][j] ? "1" : "0")); }
        printf("\n");
    }

    matrix("peer access OFF (driver stages through host memory)");

    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) if (i != j && can[i][j]) { CK(cudaSetDevice(i)); CK(cudaDeviceEnablePeerAccess(j, 0)); }
    matrix("peer access ON where cudaDeviceCanAccessPeer = 1");

    printf("\n== concurrent patterns, peer access ON where possible\n");
    report("NVLink pair 0->2", {{0, 2}});
    report("NVLink pair 0<->2 both directions", {{0, 2}, {2, 0}});
    report("same switch 0->1", {{0, 1}});
    report("same switch 0<->1 both directions", {{0, 1}, {1, 0}});
    report("across switches 0->4", {{0, 4}});
    report("4 NVLink pairs one direction", {{0, 2}, {1, 3}, {4, 6}, {5, 7}});
    report("4 NVLink pairs both directions", {{0, 2}, {2, 0}, {1, 3}, {3, 1}, {4, 6}, {6, 4}, {5, 7}, {7, 5}});
    report("host -> 1 card (c0)", {{-1, 0}});
    report("host -> 4 cards (0,1,4,5)", {{-1, 0}, {-1, 1}, {-1, 4}, {-1, 5}});
    report("host -> 8 cards", {{-1, 0}, {-1, 1}, {-1, 2}, {-1, 3}, {-1, 4}, {-1, 5}, {-1, 6}, {-1, 7}});
    // the broadcast idea: host feeds one card of each pair while each fed card passes its previous layer to its NVLink partner
    report("host->0,1,4,5 + NVLink 0->2,1->3,4->6,5->7", {{-1, 0}, {-1, 1}, {-1, 4}, {-1, 5}, {0, 2}, {1, 3}, {4, 6}, {5, 7}});
    printf("\ndone\n");
    return 0;
}
