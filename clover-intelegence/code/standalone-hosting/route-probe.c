/* Is the expert route predictable?
 *
 * A layer cannot hide its own expert reads: the router must run before the ids
 * are known, leaving only the shared expert and the projections themselves as
 * cover. Prefetching needs the ids earlier than the router can give them, so
 * the question is whether something already known predicts them.
 *
 * This drives the engine over one prompt and several output tokens and dumps
 * every (layer, position) top-16 through the sel_fp hook. Read only: it changes
 * no weights and no outputs, it only records what the router chose.
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
    assert(argc == 3);
    unsigned want = (unsigned)atoi(argv[2]);
    assert(resident_read_config());
    resident_configure(argv[1]);
    result_options();
    load_index(getenv("K3_INDEX"));
    prepared_init();
    resident_startup(0);

    sel_fp = fopen("/tmp/selections.tsv", "w");
    if (!sel_fp) die("selection dump could not be opened");
    fputs("#layer\tposition\ttoken\te0..e15\n", sel_fp);

    static const unsigned prompt[] = {1008,10484,318,15383,387};   /* The capital of France is */
    const unsigned count = (unsigned)(sizeof prompt / sizeof *prompt);
    lane_sequence_clear(0);
    active_lane = 0; batch_decode = 0;

    unsigned next = evaluate_tokens(prompt, count, 0, 1);
    printf("STEP 0 prompt pass -> %u\n", next); fflush(stdout);
    for (unsigned step = 0; step < want; step++) {
        unsigned token = next;
        next = evaluate_tokens(&token, 1, count + step, 1);
        printf("STEP %u token %u -> %u\n", step + 1, token, next); fflush(stdout);
    }
    fclose(sel_fp); sel_fp = NULL;
    resident_shutdown();
    puts("PASS: selections dumped to /tmp/selections.tsv");
    return 0;
}
