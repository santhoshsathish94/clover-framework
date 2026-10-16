/* Would 4-bit weights actually be faster here, or does the unpack eat the gain?
 
   The trunk is 52.4 GB of int8 and the engine projects it at 37.9 GB/s against a
   measured 47.5 GB/s streaming ceiling, so it is close to bandwidth-bound. MXFP4
   would cut the bytes to ~27.8 GB. That only helps if the 4-bit kernel sustains
   more than 20.1 GB/s; below that it is slower than what we already have despite
   moving half the data.

   Reading that off OPSTAT is not possible - the mxfp4 op does not account its
   weight bytes, and the experts it runs on stream from NVMe at 12.85 GB/s, so its
   timing is confounded by I/O the trunk does not have. This measures both kernels
   on the same shape, from RAM, on this CPU.

   MXFP4 is E2M1 with a shared 8-bit exponent per 32 values: 16 bytes of packed
   nibbles plus 1 scale byte per block, 4.25 bits per weight. Decoding is a LUT
   lookup per nibble, which is the cost in question.

   cc -O3 -march=native -fopenmp fp4-vs-int8.c -o fp4-vs-int8
*/
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <omp.h>

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

/* E2M1 magnitudes, the 16 values a 4-bit code can take */
static float FP4[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
   -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

/* int8: a block of row scales, then row-major int8, exactly the trunk's layout */
static double bench_int8(const float *scales, const int8_t *W, const float *x,
                         float *y, int in, int out, int threads, int reps)
{
    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        double t0 = now_s();
#pragma omp parallel for num_threads(threads) schedule(static)
        for (int o = 0; o < out; o++) {
            const int8_t *w = W + (size_t)o * in;
            float a = 0.0f;
            for (int i = 0; i < in; i++) a += (float)w[i] * x[i];
            y[o] = a * scales[o];
        }
        double dt = now_s() - t0;
        if (dt < best) best = dt;
    }
    return best;
}

/* mxfp4: per 32 values, one uint8 exponent then 16 bytes of packed nibbles */
static double bench_fp4(const uint8_t *E, const uint8_t *W, const float *x,
                        float *y, int in, int out, int threads, int reps)
{
    const int nblk = in / 32;
    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        double t0 = now_s();
#pragma omp parallel for num_threads(threads) schedule(static)
        for (int o = 0; o < out; o++) {
            const uint8_t *e = E + (size_t)o * nblk;
            const uint8_t *w = W + (size_t)o * nblk * 16;
            float a = 0.0f;
            for (int b = 0; b < nblk; b++) {
                const float s = ldexpf(1.0f, (int)e[b] - 127);
                const uint8_t *p = w + b * 16;
                const float *xb = x + b * 32;
                float acc = 0.0f;
                for (int i = 0; i < 16; i++) {
                    const uint8_t byte = p[i];
                    acc += FP4[byte & 15] * xb[2 * i];
                    acc += FP4[byte >> 4]  * xb[2 * i + 1];
                }
                a += acc * s;
            }
            y[o] = a;
        }
        double dt = now_s() - t0;
        if (dt < best) best = dt;
    }
    return best;
}

int main(int argc, char **argv)
{
    const int in = 7168, out = 12288;      /* the G/Q/K/V shape */
    int threads = (argc > 1) ? atoi(argv[1]) : 16;
    const int reps = 5;

    size_t i8_bytes = (size_t)out * in + (size_t)out * 4;
    size_t nblk = in / 32;
    size_t f4_bytes = (size_t)out * nblk * 16 + (size_t)out * nblk;

    float  *scales = aligned_alloc(64, (size_t)out * sizeof(float));
    int8_t *W8     = aligned_alloc(64, (size_t)out * in);
    uint8_t *E     = aligned_alloc(64, (size_t)out * nblk);
    uint8_t *W4    = aligned_alloc(64, (size_t)out * nblk * 16);
    float  *x      = aligned_alloc(64, (size_t)in * sizeof(float));
    float  *y      = aligned_alloc(64, (size_t)out * sizeof(float));
    if (!scales || !W8 || !E || !W4 || !x || !y) { fprintf(stderr, "alloc failed\n"); return 1; }

    for (int o = 0; o < out; o++) scales[o] = 0.001f;
    memset(W8, 3, (size_t)out * in);
    memset(E, 127, (size_t)out * nblk);
    memset(W4, 0x42, (size_t)out * nblk * 16);
    for (int i = 0; i < in; i++) x[i] = 0.001f * (float)(i & 255);

    printf("shape %dx%d, %d threads, best of %d\n\n", out, in, threads, reps);
    printf("  %-10s %12s %10s %12s\n", "kernel", "MB", "seconds", "GB/s");

    double t8 = bench_int8(scales, W8, x, y, in, out, threads, reps);
    printf("  %-10s %12.1f %10.4f %12.2f\n", "int8", i8_bytes / 1e6, t8, i8_bytes / 1e9 / t8);

    double t4 = bench_fp4(E, W4, x, y, in, out, threads, reps);
    printf("  %-10s %12.1f %10.4f %12.2f\n", "mxfp4", f4_bytes / 1e6, t4, f4_bytes / 1e9 / t4);

    printf("\nper-matrix speedup from 4 bits: %.3fx\n", t8 / t4);
    double trunk8 = 52.39, trunk4 = 52.39 * (double)f4_bytes / (double)i8_bytes;
    printf("extrapolated to the whole trunk:\n");
    printf("  int8   %.2f GB at %.2f GB/s = %.3f s\n",
           trunk8, i8_bytes / 1e9 / t8, trunk8 / (i8_bytes / 1e9 / t8));
    printf("  mxfp4  %.2f GB at %.2f GB/s = %.3f s\n",
           trunk4, f4_bytes / 1e9 / t4, trunk4 / (f4_bytes / 1e9 / t4));
    printf("the engine's own Q does 53.83 GB in 1.42 s = 37.9 GB/s\n");
    return 0;
}
