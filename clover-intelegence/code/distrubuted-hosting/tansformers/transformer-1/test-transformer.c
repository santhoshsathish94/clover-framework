#define TRANSFORMER_NO_MAIN
#include "transformer-1.c"
#include <assert.h>

int main(int argc, char **argv)
{
    if (argc != 2) return 2;
    Transformer *transformer = NULL;
    assert(transformer_open(argv[1], &transformer));
    unsigned count = 0;
    for (unsigned id = 0; id < TRANSFORMER_RECORDS; id++) count += transformer->records[id].data != NULL;
    assert(count == 19 && transformer->records[5].data && transformer->records[29].kind == 3);
    assert(!transformer->records[26].data && !transformer->records[27].data && !transformer->records[28].data);
    TransformerSequence *sequence = transformer_sequence_create(transformer);
    TransformerSequence *independent = transformer_sequence_create(transformer);
    assert(sequence && independent);
    float input[7168] = {0}, snapshot[7168] = {0}, output[7168];
    assert(transformer_process(transformer, sequence, input, snapshot, output));
    for (unsigned coordinate = 0; coordinate < 7168; coordinate++) assert(output[coordinate] == 0.0f);
    assert(sequence->positions == 1 && !independent->positions);
    assert(!memcmp(sequence->snapshot, snapshot, sizeof snapshot));
    for (unsigned rank = 0; rank < 16; rank++) {
        assert(sequence->selected[rank] < 896 && isfinite(sequence->weights[rank]));
        for (unsigned earlier = 0; earlier < rank; earlier++) assert(sequence->selected[rank] != sequence->selected[earlier]);
    }
    input[0] = NAN;
    assert(!transformer_process(transformer, sequence, input, snapshot, output));
    input[0] = 0;
    snapshot[0] = INFINITY;
    assert(!transformer_process(transformer, sequence, input, snapshot, output));
    assert(!transformer_process(NULL, sequence, input, snapshot, output));
    assert(!transformer_process(transformer, NULL, input, snapshot, output));
    assert(sequence->positions == 1);
    transformer_sequence_reset(sequence);
    assert(!sequence->positions && !sequence->failed && sequence->owner == transformer);
    transformer_sequence_close(independent);
    transformer_sequence_close(sequence);
    transformer_close(transformer);
    puts("PASS: standalone layer-1 zero-input computation, 19 records, distinct live routes, independent state, reset and invalid inputs");
    return 0;
}