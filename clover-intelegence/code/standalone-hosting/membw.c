/* What is the memory bandwidth ceiling of this box, and how close does the
   trunk read get to it?
 
   The trunk is 52.39 GB streamed per token at 2 FLOP/byte and the engine
   achieves 39.5 GB/s on it. That number is meaningless on its own: it is either
   at the ceiling, in which case only fewer bytes can help, or well under it, in
   which case the way the read is issued is costing time that no quantisation
   would need to recover. Nothing should be proposed before this is known.

   Three measurements, all on one buffer larger than the 128 MiB L3 so nothing
   is served from cache:
     read    - pure streaming load, the upper bound on any weight read
     qkernel - the engine's own Q shape, int8 rows of 4-byte scale + `in` bytes,
               so the comparison is against the real access pattern rather than
               an idealised one
   Thread counts sweep 1..32 because the box is 16 physical cores with 2 threads
   each, and a memory-bound kernel usually peaks at or below the physical count.

   cc -O3 -march=native -fopenmp membw.c -o membw
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <omp.h>

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

/* Pure streaming read: the ceiling any weight-reading kernel is bounded by. */
static double read_bw(const unsigned char *buf, size_t n, int threads)
{
    double best = 0.0;
    for (int rep = 0; rep < 3; rep++) {
        double t0 = now_s();
        uint64_t acc = 0;
#pragma omp parallel for num_threads(threads) schedule(static) reduction(+:acc)
        for (size_t i = 0; i < n / 64; i++) {
            const uint64_t *p = (const uint64_t *)(buf + i * 64);
            acc += p[0] + p[1] + p[2] + p[3] + p[4] + p[5] + p[6] + p[7];
        }
        double dt = now_s() - t0;
        if (acc == 0x5555555555555555ULL) fprintf(stderr, "");  /* keep acc live */
        double gbs = n / 1e9 / dt;
        if (gbs > best) best = gbs;
    }
    return best;
}

/* The engine's Q: one float32 scale per row, then `in` int8 weights, dot with x.
   Same row layout and same OpenMP schedule as clover-one.c. */
static double q_bw(const unsigned char *W, const float *x, float *y,
                   int in, int out, int threads)
{
    const size_t rowb = (size_t)4 + (size_t)in;
    double best = 0.0;
    for (int rep = 0; rep < 3; rep++) {
        double t0 = now_s();
#pragma omp parallel for num_threads(threads) schedule(static)
        for (int o = 0; o < out; o++) {
            const unsigned char *row = W + (size_t)o * rowb;
            float scale;
            memcpy(&scale, row, 4);
            const int8_t *w = (const int8_t *)(row + 4);
            float a = 0.0f;
            for (int i = 0; i < in; i++) a += (float)w[i] * x[i];
            y[o] = a * scale;
        }
        double dt = now_s() - t0;
        double gbs = (double)out * rowb / 1e9 / dt;
        if (gbs > best) best = gbs;
    }
    return best;
}

int main(int argc, char **argv)
{
    size_t gb = (argc > 1) ? (size_t)atoi(argv[1]) : 4;
    size_t n = gb << 30;

    unsigned char *buf = aligned_alloc(4096, n);
    if (!buf) { fprintf(stderr, "alloc %zu failed\n", n); return 1; }
    memset(buf, 1, n);   /* fault every page in before timing */

    printf("buffer %.1f GB, %d logical cpus\n", n / 1e9, omp_get_max_threads());
    printf("\npure streaming read\n  %-8s %10s\n", "threads", "GB/s");
    int tc[] = { 1, 2, 4, 8, 12, 16, 24, 32 };
    double peak = 0.0;
    for (unsigned i = 0; i < sizeof tc / sizeof *tc; i++) {
        double g = read_bw(buf, n, tc[i]);
        if (g > peak) peak = g;
        printf("  %-8d %10.2f\n", tc[i], g);
    }

    /* The real shape: 12288 x 7168, the largest trunk slot (G, Q, K, V). */
    const int in = 7168, out = 12288;
    const size_t rowb = 4 + in;
    unsigned char *W = aligned_alloc(4096, (size_t)out * rowb);
    float *x = aligned_alloc(64, (size_t)in * sizeof(float));
    float *y = aligned_alloc(64, (size_t)out * sizeof(float));
    if (!W || !x || !y) { fprintf(stderr, "alloc W/x/y failed\n"); return 1; }
    memset(W, 3, (size_t)out * rowb);
    for (int i = 0; i < in; i++) x[i] = 0.001f * (float)(i & 255);

    printf("\nQ kernel, 12288x7168 int8 (%.1f MB, the G/Q/K/V shape)\n",
           (double)out * rowb / 1e6);
    printf("  %-8s %10s %10s\n", "threads", "GB/s", "%% of peak");
    for (unsigned i = 0; i < sizeof tc / sizeof *tc; i++) {
        double g = q_bw(W, x, y, in, out, tc[i]);
        printf("  %-8d %10.2f %9.1f%%\n", tc[i], g, 100.0 * g / peak);
    }

    printf("\nengine achieves 39.5 GB/s on the trunk; peak streaming here is %.2f GB/s\n", peak);
    return 0;
}
