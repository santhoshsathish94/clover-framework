/* What does ONE layer cost when the WHOLE layer is in RAM?
 *
 * The chain architecture rests on a claim that has only ever been derived: a
 * unit holding its entire layer pays no read stall, because nothing arrives
 * from disk. `root->direct` is already an mmap of the layer's complete
 * 896 x 17,547,264 B = 15.72 GB of experts, so residency is a page-cache
 * question rather than a new data path. Fault every page, confirm with
 * mincore, then time the same projections cold and resident.
 *
 * Second arm answers the opposite proposal: compute all 896 experts rather
 * than the routed 16, trading 56x the bytes for parallelism a wide machine
 * could actually use. Both arms read the same resident mapping, so the ratio
 * is the honest cost of dense rather than a disk artefact.
 *
 * Read only: projects into scratch and discards. No engine state is touched.
 *
 *   gcc -O3 -march=native -ffp-contract=off -fno-fast-math -fopenmp \
 *       -DNPOS_SLOTS=8 layer-resident.c -lm -lpthread -o bin/layer-resident
 *   OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores \
 *       ./bin/layer-resident <absolute dataset path> <layer 1..92>
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

/* 53 is coprime with 896, so this is a permutation. Routing picks 16 of 896
   with no locality, and a contiguous run would collect readahead the real
   engine never gets. Taking disjoint windows of 16 also keeps a repeated
   measurement out of the 128 MB L3 -- 16 experts are 280 MB, so reusing one
   set would measure cache rather than memory. */
static void expert_window(unsigned *into, unsigned window)
{
    for (unsigned i = 0; i < SELECTED; i++)
        into[i] = ((window * SELECTED + i) * 53 + 11) % EXPERTS;
}

static double project_set(Root *root, RootScratch *scratch, const unsigned *experts,
                          unsigned count, const float *const *inp, float *const *outp)
{
    double started = now_s();
    for (unsigned i = 0; i < count; i++)
        for (unsigned matrix = 0; matrix < 3; matrix++)
            if (!root_project_rows(root, scratch, experts[i], matrix, inp, outp, 1))
                die("projection failed");
    return now_s() - started;
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
    assert(argc == 3 && omp_get_max_threads() <= 128);
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
    if (!root->direct) die("experts.direct not mapped; this harness needs the direct mapping");
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

    unsigned chosen[SELECTED]; expert_window(chosen, 0);
    unsigned all[EXPERTS]; for (unsigned e = 0; e < EXPERTS; e++) all[e] = e;

    const double expert_bytes = (double)ROOT_EXPERT_RAW;
    const double macs = 2.0 * 3072 * 3584 + 3584.0 * 3072;   /* gate + up + down, one position */

    printf("layer %u, %u experts, %.2f GB resident set, %d threads\n\n",
           layer, EXPERTS, layer_bytes / 1e9, omp_get_max_threads());

    printf("  before: %.1f%% of the layer is resident\n", 100 * resident_fraction(root->direct, layer_bytes, vec));

    /* A. the routed 16, cold -- this is what a layer pays today */
    double cold = project_set(root, scratch, chosen, SELECTED, inp, outp);
    printf("  A  16 routed experts, COLD       %8.4f s   %6.2f GB/s   %7.1f ms/layer\n",
           cold, SELECTED * expert_bytes / cold / 1e9, cold * 1000);

    /* B. make the whole layer resident, the way a unit would hold it */
    double started = now_s();
    volatile unsigned long long sink = 0;
    for (size_t off = 0; off < layer_bytes; off += PAGE) sink += root->direct[off];
    double fault = now_s() - started;
    (void)sink;
    printf("  B  fault in all %.2f GB          %8.4f s   %6.2f GB/s\n",
           layer_bytes / 1e9, fault, layer_bytes / fault / 1e9);
    printf("     after: %.1f%% of the layer is resident\n\n",
           100 * resident_fraction(root->direct, layer_bytes, vec));

    /* C. the routed 16, now resident -- what a unit would pay. A fresh window of
       16 every rep, so the bytes are in RAM but never in L3. */
    const unsigned reps = 20;
    unsigned window[SELECTED];
    expert_window(window, 1);
    project_set(root, scratch, window, SELECTED, inp, outp);            /* warm, not timed */
    started = now_s();
    for (unsigned r = 0; r < reps; r++) {
        expert_window(window, 2 + r);
        project_set(root, scratch, window, SELECTED, inp, outp);
    }
    double warm = (now_s() - started) / reps;
    printf("  C  16 routed experts, RESIDENT   %8.4f s   %6.2f GB/s   %7.1f ms/layer   %.2fx vs cold\n",
           warm, SELECTED * expert_bytes / warm / 1e9, warm * 1000, cold / warm);
    printf("     (%u disjoint windows of 16, %.2f GB touched, so no L3 reuse)\n",
           reps, reps * SELECTED * expert_bytes / 1e9);

    /* D. all 896 -- the dense proposal, same mapping, no disk in the path */
    double dense = project_set(root, scratch, all, EXPERTS, inp, outp);
    printf("  D  ALL %u experts, RESIDENT     %8.4f s   %6.2f GB/s   %7.1f ms/layer   %.1fx vs C\n",
           EXPERTS, dense, EXPERTS * expert_bytes / dense / 1e9, dense * 1000, dense / warm);

    printf("\n  arithmetic: sparse %.1f GFLOP/s, dense %.1f GFLOP/s\n",
           2.0 * SELECTED * macs / warm / 1e9, 2.0 * EXPERTS * macs / dense / 1e9);
    printf("  one token, 92 MoE layers: sparse %.2f s, dense %.2f s\n",
           92 * warm, 92 * dense);

    free(vec);
    resident_shutdown();
    return 0;
}
