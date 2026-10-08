/* Is the expert projection bound by decoding weights or by arithmetic?
 *
 * root_project_rows rebuilds 16 doubles from 8 packed bytes through a byte-to-
 * float-pair table for every 16 coordinates, then multiplies them into `count`
 * position vectors. The decode is paid once per weight whatever count is; the
 * arithmetic is paid per position. So:
 *
 *   time flat as count grows      -> decode dominates, and only fewer distinct
 *                                    experts or a cheaper decode will help
 *   time proportional to count    -> arithmetic dominates, and the decode is
 *                                    already free
 *
 * One expert, repeated, so the bytes stay in cache and this measures compute
 * rather than the read path. Read only: projects into scratch and discards.
 */
#define _GNU_SOURCE
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#define main standalone_program_main
#include "clover-one.c"
#undef main

int main(int argc, char **argv)
{
    assert(argc == 2 && omp_get_max_threads() <= 128);
    assert(resident_read_config());
    resident_configure(argv[1]);
    result_options();
    load_index(getenv("K3_INDEX"));
    prepared_init();
    resident_startup(0);

    Root *root = resident_roots[1];
    if (!root) die("layer 1 root unavailable");
    RootScratch *scratch = resident_root_scratch;

    static float in[NPOS_SLOTS][3584], out[NPOS_SLOTS][3584];
    const float *inp[NPOS_SLOTS];
    float *outp[NPOS_SLOTS];
    for (int t = 0; t < NPOS_SLOTS; t++) {
        for (int i = 0; i < 3584; i++) in[t][i] = 0.001f * (float)((i * 7 + t) % 97);
        inp[t] = in[t]; outp[t] = out[t];
    }

    const unsigned expert = 498;
    if (!root_project_rows(root, scratch, expert, 0, inp, outp, 1))
        die("projection unavailable: expert bytes not reachable");

    /* one expert is gate + up + down: 3072x3584, 3072x3584, 3584x3072 */
    const double macs_per_position = 2.0 * 3072 * 3584 + 3584.0 * 3072;
    const double bytes_per_expert = (double)ROOT_EXPERT_RAW;
    const unsigned reps = 60;

    printf("  layer 1 root: %u palettes, %u maps, %u templates, %u references\n",
           root->palette_count, root->map_count, root->template_count, root->reference_count);
    printf("  pair tables: %u x 256 x 2 float = %.1f KB (as double, %.1f KB)\n\n",
           root->reference_count, root->reference_count * 2048 / 1024.0,
           root->reference_count * 4096 / 1024.0);

    printf("  count   wall_s   s/position   GFLOP/s    GB/s   vs count=1\n");
    double base = 0, t[9] = {0};
    for (unsigned count = 1; count <= NPOS_SLOTS; count *= 2) {
        for (unsigned matrix = 0; matrix < 3; matrix++)      /* warm, not timed */
            root_project_rows(root, scratch, expert, matrix, inp, outp, count);
        double started = now_s();
        for (unsigned r = 0; r < reps; r++)
            for (unsigned matrix = 0; matrix < 3; matrix++)
                if (!root_project_rows(root, scratch, expert, matrix, inp, outp, count))
                    die("projection failed");
        double elapsed = now_s() - started;
        double flop = 2.0 * reps * count * macs_per_position;
        double bytes = (double)reps * bytes_per_expert;
        if (count == 1) base = elapsed;
        t[count] = elapsed;
        printf("  %5u  %7.4f  %11.5f  %8.1f  %6.1f   %7.2fx\n",
               count, elapsed, elapsed / count, flop / elapsed / 1e9,
               bytes / elapsed / 1e9, elapsed / base);
    }
    /* wall = fixed + per_position * count, fitted on the two widest points */
    double slope = (t[8] - t[2]) / 6.0, fixed = t[2] - 2 * slope;
    printf("\n  fit over count 2..8:  wall = %.4f + %.4f * count  (per %u reps)\n",
           fixed, slope, reps);
    printf("  one expert, one position: %.3f ms decode+stream + %.3f ms arithmetic = %.0f%% fixed\n",
           fixed / reps * 1000, slope / reps * 1000, 100 * fixed / (fixed + slope));
    printf("  decode+stream alone would run at %.1f GB/s\n",
           bytes_per_expert * reps / fixed / 1e9);
    resident_shutdown();
    return 0;
}
