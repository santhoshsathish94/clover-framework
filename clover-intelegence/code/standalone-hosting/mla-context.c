/* One MLA layer, N cached positions: what a single attention step costs.
 *
 * The engine stores K and V expanded, 98,560 B a position a layer, and the
 * question is what that costs when the context is long rather than 256. The
 * kernel here mirrors clover-one.c: head-parallel, each head walking the whole
 * sequence with an H*QN stride, double accumulation in the dot. The access
 * pattern is the thing being measured, so it is copied rather than idealised.
 *
 * scv/ex live on the heap here. In the engine they are stack arrays sized
 * GENERATION_CAPACITY, which is the reason the context cap is 256.
 *
 *   gcc -O3 -march=native -fopenmp mla-context.c -lm -o bin/mla-context
 *   ./bin/mla-context <positions>
 */
#define _GNU_SOURCE
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <time.h>

#define H  96
#define QN 128
#define QR 64
#define QH 192
#define VH 128

static double now_s(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

static float *big(size_t bytes)
{
    void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (p == MAP_FAILED) return NULL;
    (void)madvise(p, bytes, MADV_HUGEPAGE);
    return (float *)p;
}

int main(int argc, char **argv)
{
    size_t N = argc > 1 ? strtoull(argv[1], NULL, 10) : 4096;
    /* 0 reproduces the engine's position-major K/V; 1 gives each head a
       contiguous slice, which is the same bytes under a different stride. */
    int head_major = argc > 2 ? atoi(argv[2]) : 0;
    if (!N) return 1;

    size_t kb = N * H * QN * sizeof(float);
    size_t vb = N * H * VH * sizeof(float);
    size_t pb = N * QR * sizeof(float);
    size_t sb = (size_t)H * 2 * N * sizeof(float);

    float *keys = big(kb), *values = big(vb), *positions = big(pb), *scratch = big(sb);
    if (!keys || !values || !positions || !scratch) {
        fprintf(stderr, "allocation failed for %zu positions (%.1f GB)\n",
                N, (double)(kb + vb + pb + sb) / 1e9);
        return 1;
    }

    /* Fault in and fill. Without this the first step would time page faults. */
    double t0 = now_s();
#pragma omp parallel for schedule(static)
    for (size_t s = 0; s < N; s++) {
        for (int h = 0; h < H; h++) {
            size_t ko = head_major ? ((size_t)h * N + s) * QN : ((size_t)s * H + h) * QN;
            size_t vo = head_major ? ((size_t)h * N + s) * VH : ((size_t)s * H + h) * VH;
            for (int i = 0; i < QN; i++) keys[ko + i] = 0.001f * (float)((s + h + i) & 1023);
            for (int i = 0; i < VH; i++) values[vo + i] = 0.001f * (float)((s * 3 + h + i) & 1023);
        }
        float *p = positions + s * QR;
        for (size_t i = 0; i < QR; i++) p[i] = 0.001f * (float)((s + i) & 255);
    }
    double fill = now_s() - t0;

    float *q = aligned_alloc(64, (size_t)H * QH * sizeof(float));
    float *acc = aligned_alloc(64, (size_t)H * VH * sizeof(float));
    if (!q || !acc) return 1;
    for (size_t i = 0; i < (size_t)H * QH; i++) q[i] = 0.001f * (float)(i & 511);

    const float msc = 1.0f / sqrtf(192.0f);
    const size_t last = N - 1;
    double best = 1e18;

    for (int rep = 0; rep < 3; rep++) {
        double t = now_s();
#pragma omp parallel for schedule(static)
        for (int h = 0; h < H; h++) {
            const float *qh = q + (size_t)h * QH;
            float *scv = scratch + (size_t)h * 2 * N, *ex = scv + N;
            for (size_t s = 0; s <= last; s++) {
                const float *kl = keys + (head_major ? ((size_t)h * N + s) : ((size_t)s * H + h)) * QN;
                const float *kr = positions + (size_t)s * QR;
                double d = 0.0;
                for (int i = 0; i < QN; i++) d += (double)qh[i] * (double)kl[i];
                for (int i = 0; i < QR; i++) d += (double)qh[QN + i] * (double)kr[i];
                scv[s] = (float)d * msc;
            }
            float m = scv[0];
            for (size_t s = 1; s <= last; s++) if (scv[s] > m) m = scv[s];
            double z = 0.0;
            for (size_t s = 0; s <= last; s++) { ex[s] = expf(scv[s] - m); z += (double)ex[s]; }
            float *o = acc + (size_t)h * VH;
            for (int i = 0; i < VH; i++) o[i] = 0.0f;
            for (size_t s = 0; s <= last; s++) {
                const float pr = (float)((double)ex[s] / z);
                const float *vs = values + (head_major ? ((size_t)h * N + s) : ((size_t)s * H + h)) * VH;
                for (int i = 0; i < VH; i++) o[i] = o[i] + pr * vs[i];
            }
        }
        double dt = now_s() - t;
        if (dt < best) best = dt;
    }

    double gb = (double)(kb + vb + pb) / 1e9;
    double checksum = 0.0;
    for (int i = 0; i < H * VH; i++) checksum += acc[i];

    printf("MLA_JSON {\"positions\":%zu,\"layout\":\"%s\",\"expanded_gb\":%.3f,\"latent_gb\":%.3f,"
           "\"fill_s\":%.3f,\"step_s\":%.6f,\"read_gb_per_s\":%.2f,"
           "\"all_24_layers_s\":%.3f,\"checksum\":%.6f}\n",
           N, head_major ? "head-major" : "position-major", gb, (double)N * 576 * 4 / 1e9,
           fill, best, gb / best, best * 24.0, checksum);
    return 0;
}
