/* With a layer resident, does applying each expert to MORE positions pay?
 *
 * layer-resident.c measured residency at one position per expert: 16 routed
 * experts cost 7.5 ms resident against 63.1 ms cold, 8.4x. That is the decode
 * case. Prefill is different - 8192 positions through one layer means each of
 * the 896 experts serves about 146 positions - and the row count was never
 * swept, so whether the weight read amortises across positions is unmeasured.
 *
 * It matters because the prior cycle's blocker was stated for decode: loading
 * 15.72 GB inside a 36.5 ms window is 32x short. At 8192 positions that window
 * is ~47 s and the same load is covered 40x over. If per-position cost also
 * falls with rows, layer-at-a-time prefill wins twice.
 *
 * Faults the layer in, confirms with mincore, then times the engine's own
 * root_project_rows at increasing row counts against disjoint expert windows
 * so nothing is served from L3.
 *
 * Read only: projects into scratch and discards.
 *
 * cc -O3 -march=native -fopenmp -DNPOS_SLOTS=8 -DROOT_MAX_ROWS=64 layer-rows.c -lm
 */
#define _GNU_SOURCE
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <sys/mman.h>
#define main standalone_program_main
#include "clover-one.c"
#undef main

enum { EXPERTS = 896, SELECTED = 16, PAGE = 4096 };

/* 53 is coprime with 896, so this is a permutation; disjoint windows keep a
   repeated measurement out of the 128 MB L3. */
static void expert_window(unsigned *into, unsigned window)
{
    for (unsigned i = 0; i < SELECTED; i++)
        into[i] = ((window * SELECTED + i) * 53 + 11) % EXPERTS;
}

static double resident_fraction(const unsigned char *base, size_t bytes, unsigned char *vec)
{
    size_t pages = (bytes + PAGE - 1) / PAGE, present = 0;
    if (mincore((void *)base, bytes, vec)) return -1.0;
    for (size_t p = 0; p < pages; p++) if (vec[p] & 1) present++;
    return (double)present / (double)pages;
}

int main(int argc, char **argv)
{
    assert(argc == 3);
    unsigned layer = (unsigned)strtoul(argv[2], NULL, 10);
    if (layer < 1 || layer > 92) die("layer must be 1..92");

    assert(resident_read_config());
    resident_configure(argv[1]);
    result_options();
    load_index(getenv("K3_INDEX"));
    prepared_init();
    resident_startup(0);

    Root *root = resident_roots[layer];
    if (!root) die("root unavailable for this layer");
    if (!root->direct) die("experts.direct not mapped");
    RootScratch *scratch = resident_root_scratch;

    const size_t layer_bytes = (size_t)EXPERTS * ROOT_EXPERT_RAW;
    unsigned char *vec = malloc((layer_bytes + PAGE - 1) / PAGE);
    if (!vec) die("mincore vector allocation failed");

    static float in[ROOT_MAX_ROWS][3584], out[ROOT_MAX_ROWS][3584];
    const float *inp[ROOT_MAX_ROWS]; float *outp[ROOT_MAX_ROWS];
    for (int t = 0; t < ROOT_MAX_ROWS; t++) {
        for (int i = 0; i < 3584; i++) in[t][i] = 0.001f * (float)((i * 7 + t) % 97);
        inp[t] = in[t]; outp[t] = out[t];
    }

    printf("layer %u, %.2f GB resident set, %d threads, ROOT_MAX_ROWS=%d\n",
           layer, layer_bytes / 1e9, omp_get_max_threads(), ROOT_MAX_ROWS);

    double started = now_s();
    volatile unsigned long long sink = 0;
    for (size_t off = 0; off < layer_bytes; off += PAGE) sink += root->direct[off];
    double fault = now_s() - started;
    (void)sink;
    printf("faulted %.2f GB in %.3f s = %.2f GB/s; resident %.1f%%\n\n",
           layer_bytes / 1e9, fault, layer_bytes / fault / 1e9,
           100 * resident_fraction(root->direct, layer_bytes, vec));

    /* gate + up + down for one position */
    const double macs = 2.0 * 3072 * 3584 + 3584.0 * 3072;
    const double expert_bytes = (double)ROOT_EXPERT_RAW;

    printf("  %-5s %12s %14s %12s %12s %9s\n",
           "rows", "s/16-expert", "s per position", "GB/s", "GFLOP/s", "vs rows=1");
    double base = 0.0;
    for (unsigned rows = 1; rows <= (unsigned)ROOT_MAX_ROWS; rows *= 2) {
        unsigned window[SELECTED];
        expert_window(window, 1);
        for (unsigned i = 0; i < SELECTED; i++)          /* warm, untimed */
            for (unsigned m = 0; m < 3; m++)
                root_project_rows(root, scratch, window[i], m, inp, outp, rows);

        const unsigned reps = 8;
        double t0 = now_s();
        for (unsigned r = 0; r < reps; r++) {
            expert_window(window, 2 + r);
            for (unsigned i = 0; i < SELECTED; i++)
                for (unsigned m = 0; m < 3; m++)
                    if (!root_project_rows(root, scratch, window[i], m, inp, outp, rows))
                        die("projection failed");
        }
        double per = (now_s() - t0) / reps;
        double perpos = per / rows;
        if (!base) base = perpos;
        printf("  %-5u %12.5f %14.6f %12.2f %12.1f %8.2fx\n",
               rows, per, perpos,
               SELECTED * expert_bytes / per / 1e9,
               2.0 * SELECTED * macs * rows / per / 1e9,
               base / perpos);
    }

    printf("\nIf s per position falls with rows, a resident expert serving many\n");
    printf("positions is cheaper per token, and layer-at-a-time prefill pays twice:\n");
    printf("once on the read it avoids, once on the intensity it gains.\n");

    /* One layer at real prefill scale, rather than extrapolating from a rate.
       8192 positions x 16 experts = 131,072 expert-position pairs, processed
       expert-major in groups of 8 because 8 measured as the optimum above. */
    {
        const unsigned positions = 8192, topk = 16, group = 8;
        const unsigned long pairs = (unsigned long)positions * topk;
        const unsigned per_expert = (unsigned)(pairs / EXPERTS);      /* ~146 */
        const unsigned groups = (per_expert + group - 1) / group;
        double t0 = now_s();
        for (unsigned e = 0; e < EXPERTS; e++)
            for (unsigned g = 0; g < groups; g++)
                for (unsigned m = 0; m < 3; m++)
                    if (!root_project_rows(root, scratch, e, m, inp, outp, group))
                        die("projection failed");
        double full = now_s() - t0;
        printf("\n  one layer at prefill scale: %u positions, %lu expert-position pairs\n",
               positions, pairs);
        printf("  %u experts x %u groups of %u rows x 3 matrices = %u calls\n",
               EXPERTS, groups, group, EXPERTS * groups * 3);
        printf("  measured %.2f s a layer  ->  %.0f s for 93 layers = %.2f hours\n",
               full, full * 93, full * 93 / 3600.0);
        printf("  against the 2.6 hours observed today: %.2fx\n", 9451.0 / (full * 93));
    }

    free(vec);
    resident_shutdown();
    return 0;
}
