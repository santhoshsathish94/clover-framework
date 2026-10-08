/* Is an expert matrix low-rank enough to store as factors instead of in full?
 *
 * The only currency left is bytes read: the projection is matrix-vector, so
 * every element must be read at least once, and the kernel already runs at
 * 80-90% of memory bandwidth. Fewer multiplications would not help; fewer
 * bytes would.
 *
 * Storing gate as rank-r factors costs r*(3072+3584)*2 bytes against
 * 3072*3584/2 = 5.5 MB for the 4-bit form, so it only pays below r ~ 413 of
 * 3072, about 13% of full rank.
 *
 * This extracts a real expert matrix exactly, by projecting unit basis vectors
 * through the engine's own kernel, so the singular values are measured on the
 * weights the engine actually uses rather than on a re-implementation.
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
    assert(argc == 4);
    unsigned layer = (unsigned)atoi(argv[2]);
    unsigned expert = (unsigned)atoi(argv[3]);
    assert(resident_read_config());
    resident_configure(argv[1]);
    result_options();
    load_index(getenv("K3_INDEX"));
    prepared_init();
    resident_startup(0);

    Root *root = resident_roots[layer];
    if (!root) die("layer root unavailable");
    RootScratch *scratch = resident_root_scratch;

    /* matrix 0 is gate: 3584 in, 3072 out */
    const unsigned width = 3584, rows = 3072, step = 8;
    static float basis[8][3584], out[8][3584];
    const float *inp[8]; float *outp[8];
    for (unsigned k = 0; k < step; k++) { inp[k] = basis[k]; outp[k] = out[k]; }

    char path[512];
    snprintf(path, sizeof path, "/tmp/expert-L%u-E%u-gate.f32", layer, expert);
    FILE *f = fopen(path, "wb");
    if (!f) die("matrix dump could not be opened");

    double started = now_s();
    for (unsigned start = 0; start < width; start += step) {
        unsigned n = width - start < step ? width - start : step;
        memset(basis, 0, sizeof basis);
        for (unsigned k = 0; k < n; k++) basis[k][start + k] = 1.0f;
        if (!root_project_rows(root, scratch, expert, 0, inp, outp, n))
            die("projection failed");
        for (unsigned k = 0; k < n; k++)
            if (fwrite(out[k], 4, rows, f) != rows) die("short write");
    }
    fclose(f);
    printf("wrote %s : %u columns x %u rows, %.1f MB, %.2f s\n",
           path, width, rows, width * rows * 4.0 / 1e6, now_s() - started);
    resident_shutdown();
    return 0;
}
