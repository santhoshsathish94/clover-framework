#define _GNU_SOURCE
/* Is any 7168-value vector the model computes already a row of the 163,840-row embedding table?
   Step 1: hash every table row and every computed vector; count exact (bit-for-bit) matches.
   Step 2: for a sample of vectors, the nearest table row by cosine, and the token's own row.
   The embedding vectors themselves are the positive control: they must match exactly. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "numeric-table.h"

enum { W = 7168, V = 163840, SLOTS = 1 << 19 };

static uint64_t fnv(const float *v)
{
    const unsigned char *p = (const unsigned char *)v;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < (size_t)W * 4; i++) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

static float dot(const float *a, const float *b)
{
    float l[8] = {0};
    for (unsigned i = 0; i < W; i += 8) for (int k = 0; k < 8; k++) l[k] += a[i + k] * b[i + k];
    return ((l[0] + l[4]) + (l[1] + l[5])) + ((l[2] + l[6]) + (l[3] + l[7]));
}

static int cmpf(const void *a, const void *b) { float x = *(const float *)a, y = *(const float *)b; return (x > y) - (x < y); }

int main(int argc, char **argv)
{
    if (argc < 5) { fputs("usage: vec-match SEED_BIN IDS_FILE N label=file.f32 ...\n", stderr); return 2; }
    const unsigned N = (unsigned)atoi(argv[3]);
    uint32_t *ids = malloc(N * sizeof *ids);
    FILE *f = fopen(argv[2], "r");
    unsigned have = 0; unsigned long v;
    while (f && have < N && fscanf(f, "%lu", &v) == 1) { ids[have++] = (uint32_t)v; int ch = fgetc(f); if (ch != ',' && ch != ' ' && ch != '\n' && ch != EOF) ungetc(ch, f); }
    if (!f || have != N) { fputs("ids file problem\n", stderr); return 1; }
    fclose(f);

    ServerInput *in = server_input_open(argv[1]);
    if (!in) { fputs("seed open failed\n", stderr); return 1; }
    float *tab = malloc((size_t)V * W * sizeof *tab);
    double t0 = omp_get_wtime();
    for (unsigned id = 0; id < V; id++) if (!server_input(in, id, tab + (size_t)id * W)) { fprintf(stderr, "lookup %u failed\n", id); return 1; }
    printf("read %d table rows in %.1f s\n", V, omp_get_wtime() - t0);

    int32_t *slot = malloc(SLOTS * sizeof *slot);
    uint64_t *hv = malloc((size_t)V * sizeof *hv);
    for (int i = 0; i < SLOTS; i++) slot[i] = -1;
#pragma omp parallel for schedule(static)
    for (int id = 0; id < V; id++) hv[id] = fnv(tab + (size_t)id * W);
    unsigned dup_rows = 0;
    for (int id = 0; id < V; id++) {
        uint64_t s = hv[id] & (SLOTS - 1);
        while (slot[s] >= 0) { if (hv[slot[s]] == hv[id] && !memcmp(tab + (size_t)slot[s] * W, tab + (size_t)id * W, W * 4)) dup_rows++; s = (s + 1) & (SLOTS - 1); }
        slot[s] = id;
    }
    printf("table rows that are bit-identical to an earlier row: %u of %d\n", dup_rows, V);

    float *rnorm = malloc((size_t)V * sizeof *rnorm);
#pragma omp parallel for schedule(static)
    for (int id = 0; id < V; id++) rnorm[id] = sqrtf(dot(tab + (size_t)id * W, tab + (size_t)id * W));
    float *sim = malloc((size_t)V * sizeof *sim);
    float *vec = malloc((size_t)N * W * sizeof *vec);
    const unsigned stride = N >= 100 ? N / 100 : 1;
    float *best = malloc(N * sizeof *best), *own = malloc(N * sizeof *own);

    for (int a = 4; a < argc; a++) {
        char *eq = strchr(argv[a], '=');
        if (!eq) { fputs("argument must be label=path\n", stderr); return 1; }
        *eq = 0;
        const char *label = argv[a], *path = eq + 1;
        f = fopen(path, "rb");
        if (!f || fread(vec, sizeof(float), (size_t)N * W, f) != (size_t)N * W) { fprintf(stderr, "cannot read %s\n", path); return 1; }
        fclose(f);
        unsigned exact = 0, exact_own = 0;
        for (unsigned i = 0; i < N; i++) {
            const float *x = vec + (size_t)i * W;
            uint64_t h = fnv(x), s = h & (SLOTS - 1);
            int found = 0;
            while (slot[s] >= 0) {
                int r = slot[s];
                if (hv[r] == h && !memcmp(tab + (size_t)r * W, x, W * 4)) { found = 1; exact_own += (uint32_t)r == ids[i]; }
                s = (s + 1) & (SLOTS - 1);
            }
            exact += found;
        }
        unsigned sampled = 0, nearest_is_own = 0;
        for (unsigned i = 0; i < N; i += stride) {
            const float *x = vec + (size_t)i * W;
            float xn = sqrtf(dot(x, x));
#pragma omp parallel for schedule(static)
            for (int r = 0; r < V; r++) sim[r] = rnorm[r] > 0 && xn > 0 ? dot(x, tab + (size_t)r * W) / (xn * rnorm[r]) : -2.0f;
            int arg = 0;
            for (int r = 1; r < V; r++) if (sim[r] > sim[arg]) arg = r;
            best[sampled] = sim[arg]; own[sampled] = sim[ids[i]];
            nearest_is_own += (uint32_t)arg == ids[i];
            sampled++;
        }
        float bs[1000], os[1000];
        memcpy(bs, best, sampled * sizeof *bs); memcpy(os, own, sampled * sizeof *os);
        qsort(bs, sampled, sizeof *bs, cmpf); qsort(os, sampled, sizeof *os, cmpf);
        printf("\n%-24s exact table-row matches: %u of %u (of which the token's own row: %u)\n", label, exact, N, exact_own);
        printf("%-24s nearest row, %u sampled: cosine min %.4f median %.4f max %.4f; nearest row is the token's own: %u of %u\n",
            "", sampled, bs[0], bs[sampled / 2], bs[sampled - 1], nearest_is_own, sampled);
        printf("%-24s cosine to the token's own row: min %.4f median %.4f max %.4f\n", "", os[0], os[sampled / 2], os[sampled - 1]);
        fflush(stdout);
    }
    server_input_close(in);
    return 0;
}
