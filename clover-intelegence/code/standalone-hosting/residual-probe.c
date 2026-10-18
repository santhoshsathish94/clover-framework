/* Are consecutive tokens' per-layer states similar enough to reuse?
 *
 * The engine is already incremental in attention: resident_sequences carries the
 * KV cache and the KDA recurrent state, so tokens 1..n are never recomputed.
 * What is recomputed is the new token's own journey through 93 layers.
 *
 * If token n and n+1 arrive at layer L in nearly the same state, there might be
 * structure to exploit - compute a correction rather than the whole layer. If
 * they are far apart, there is nothing there and the repetition is the function
 * meeting new data, not waste.
 *
 * Drives the engine's own dumplay hook, which is declared and written to but
 * never assigned. Read only: records layer inputs, changes nothing.
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

    dumplay = fopen("/tmp/layer-inputs.f32", "wb");
    if (!dumplay) die("layer input dump could not be opened");

    static const unsigned prompt[] = {1008,10484,318,15383,387};
    const unsigned count = (unsigned)(sizeof prompt / sizeof *prompt);
    lane_sequence_clear(0);
    active_lane = 0; batch_decode = 0;

    unsigned next = evaluate_tokens(prompt, count, 0, 1);
    printf("prefill -> %u  (%u positions x %d layers dumped)\n", next, count, NLAY);
    for (unsigned step = 0; step < want; step++) {
        unsigned token = next;
        next = evaluate_tokens(&token, 1, count + step, 1);
        printf("decode %u: %u -> %u\n", step + 1, token, next);
    }
    fclose(dumplay); dumplay = NULL;
    printf("E=%d NLAY=%d prompt_positions=%u decode_steps=%u\n", E, NLAY, count, want);
    resident_shutdown();
    puts("PASS: layer inputs dumped to /tmp/layer-inputs.f32");
    return 0;
}
