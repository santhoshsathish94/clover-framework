/* Does the 48.43 GB/s ceiling survive at trunk scale?

   The first measurement used a 4.3 GB buffer. The trunk is 52 GB. At 4 KB pages
   that is ~13 million pages against ~1 million, so the earlier number may have
   been measured in a regime the real workload never operates in - a ratio from
   one regime is a coincidence until a second regime fails to break it.

   Sweeps buffer size and compares default pages against MADV_HUGEPAGE, which is
   the difference the engine's own unused trunk-to-RAM path contemplated
   ("hugetlb 2M" / "THP" / "4K") and never tested.

   cc -O3 -march=native -fopenmp membw-scale.c -o membw-scale
*/
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <sys/mman.h>
#include <omp.h>

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static double read_bw(const unsigned char *buf, size_t n, int threads, int reps)
{
    double best = 0.0;
    for (int rep = 0; rep < reps; rep++) {
        double t0 = now_s();
        uint64_t acc = 0;
#pragma omp parallel for num_threads(threads) schedule(static) reduction(+:acc)
        for (size_t i = 0; i < n / 64; i++) {
            const uint64_t *p = (const uint64_t *)(buf + i * 64);
            acc += p[0] + p[1] + p[2] + p[3] + p[4] + p[5] + p[6] + p[7];
        }
        double dt = now_s() - t0;
        if (acc == 0x5555555555555555ULL) putchar(' ');  /* keep acc live */
        double gbs = n / 1e9 / dt;
        if (gbs > best) best = gbs;
    }
    return best;
}

static double run_one(size_t n, int threads, int huge, int reps)
{
    unsigned char *buf = mmap(NULL, n, PROT_READ | PROT_WRITE,
                              MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (buf == MAP_FAILED) return -1.0;
    if (madvise(buf, n, huge ? MADV_HUGEPAGE : MADV_NOHUGEPAGE)) { /* advisory */ }
    memset(buf, 1, n);                 /* fault in before timing */
    double g = read_bw(buf, n, threads, reps);
    munmap(buf, n);
    return g;
}

int main(int argc, char **argv)
{
    int threads = (argc > 1) ? atoi(argv[1]) : 4;
    printf("threads %d (peak thread count from the 4.3 GB sweep)\n\n", threads);
    printf("  %-10s %12s %12s %10s\n", "buffer GB", "4K GB/s", "THP GB/s", "THP gain");

    size_t sizes[] = { 4, 8, 16, 32, 48, 56 };
    for (unsigned i = 0; i < sizeof sizes / sizeof *sizes; i++) {
        size_t n = sizes[i] << 30;
        int reps = (sizes[i] >= 32) ? 2 : 3;
        double g4 = run_one(n, threads, 0, reps);
        double gh = run_one(n, threads, 1, reps);
        if (g4 < 0 || gh < 0) { printf("  %-10zu  allocation failed\n", sizes[i]); break; }
        printf("  %-10zu %12.2f %12.2f %9.2fx\n", sizes[i], g4, gh, gh / g4);
        fflush(stdout);
    }
    printf("\nthe trunk is 52.39 GB; the engine achieves 39.5 GB/s on it\n");
    return 0;
}
