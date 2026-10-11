/* Does the stage path depend on the data? Drives the stage machine directly for several
   different inputs and compares the exact sequence of stages each one visits. */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TRANSFORMER_NO_MAIN
#include TRANSFORMER_SOURCE

enum { TRACE_MAX = 4096 };

static unsigned trace_run(Transformer *transformer, TransformerSequence *sequence,
    const float *input, const float *snapshots, int *trace)
{
    static float output[TRANSFORMER_WIDTH];
    static float output_snapshots[9 * TRANSFORMER_WIDTH];
    transformer_sequence_reset(sequence);
    memcpy(sequence->snapshots, snapshots,
        (size_t)TRANSFORMER_INPUT_SNAPSHOTS * TRANSFORMER_WIDTH * sizeof(float));
    sequence->snapshot_count = TRANSFORMER_INPUT_SNAPSHOTS;
    sequence->stage = 1;
    unsigned visited = 0;
    while (sequence->stage && visited < TRACE_MAX) {
        trace[visited++] = sequence->stage;
        if (!transformer_step(transformer, sequence, input, output, output_snapshots)) {
            printf("  step failed at stage %d\n", sequence->stage);
            return visited;
        }
    }
    return visited;
}

int main(int argc, char **argv)
{
    assert(argc == 2);
    Transformer *transformer = NULL;
    assert(transformer_open(argv[1], &transformer));
    assert(transformer);
    TransformerSequence *sequence = transformer_sequence_create(transformer);
    assert(sequence);

    static float input[TRANSFORMER_WIDTH];
    static float snapshots[9 * TRANSFORMER_WIDTH];
    static int first[TRACE_MAX], other[TRACE_MAX];
    unsigned first_length = 0;
    int identical = 1;

    /* Deliberately unlike each other: different magnitudes, signs and shapes, so the
       router picks different experts for each. */
    for (unsigned probe = 0; probe < 6; probe++) {
        for (unsigned i = 0; i < TRANSFORMER_WIDTH; i++) {
            float base = (float)((i * 2654435761u + probe * 40503u) % 2048) / 2048.0f - 0.5f;
            input[i] = base * (float)(probe + 1) * 0.1f;
        }
        for (unsigned s = 0; s < TRANSFORMER_INPUT_SNAPSHOTS; s++)
            for (unsigned i = 0; i < TRANSFORMER_WIDTH; i++)
                snapshots[s * TRANSFORMER_WIDTH + i] = input[i] * (0.5f + 0.1f * (float)s);

        int *trace = probe ? other : first;
        unsigned length = trace_run(transformer, sequence, input, snapshots, trace);
        unsigned routes[16] = {0};
        memcpy(routes, sequence->selected, sizeof routes);

        if (!probe) {
            first_length = length;
            printf("probe 0: %u stages, first experts %u %u %u\n", length, routes[0], routes[1], routes[2]);
        } else {
            int same = length == first_length && !memcmp(first, other, length * sizeof(int));
            if (!same) identical = 0;
            printf("probe %u: %u stages, first experts %u %u %u -> stage path %s\n",
                probe, length, routes[0], routes[1], routes[2], same ? "IDENTICAL" : "DIFFERS");
        }
    }

    printf("%s: stage path is %s across differing inputs\n",
        identical ? "PASS" : "FAIL", identical ? "identical" : "data dependent");
    transformer_sequence_close(sequence);
    transformer_close(transformer);
    return identical ? 0 : 1;
}
