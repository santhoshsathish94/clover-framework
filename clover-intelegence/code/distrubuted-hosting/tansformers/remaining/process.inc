int transformer_process(Transformer *transformer, TransformerSequence *sequence,
    const float input[TRANSFORMER_WIDTH], const float *snapshots, unsigned snapshot_count,
    float output[TRANSFORMER_WIDTH], float *output_snapshots)
{
    if (!transformer || !sequence || sequence->owner != transformer || !input || !output || !snapshots || !output_snapshots ||
        snapshot_count != TRANSFORMER_INPUT_SNAPSHOTS || sequence->failed || sequence->positions == SIZE_MAX ||
        fegetround() != FE_TONEAREST || FLT_EVAL_METHOD != 0 || !transformer_finite(input, TRANSFORMER_WIDTH) ||
        !transformer_finite(snapshots, (size_t)snapshot_count * TRANSFORMER_WIDTH)) return 0;
    memcpy(sequence->snapshots, snapshots, (size_t)snapshot_count * TRANSFORMER_WIDTH * sizeof(float));
    sequence->snapshot_count = snapshot_count;
    memcpy(sequence->incoming, input, sizeof sequence->incoming);
    memcpy(sequence->residual, input, sizeof sequence->residual);
    transformer_aggregate(transformer, sequence, 37);
    if (TRANSFORMER_LAYER % 12 == 0) {
        memcpy(sequence->snapshots[sequence->snapshot_count], input, TRANSFORMER_WIDTH * sizeof(float));
        sequence->snapshot_count++;
    }
    transformer_normalize(sequence->normalized, sequence->aggregate, transformer->input_gains, TRANSFORMER_WIDTH);
#if TRANSFORMER_MLA
    if (!transformer_mla(transformer, sequence)) { sequence->failed = 1; return 0; }
#else
    transformer_qkv(transformer, sequence);
    transformer_decay(transformer, sequence);
    transformer_update_attention(sequence);
    transformer_attention_output(transformer, sequence);
#endif
    if (TRANSFORMER_LAYER % 12 != 0) for (unsigned coordinate = 0; coordinate < TRANSFORMER_WIDTH; coordinate++)
        sequence->residual[coordinate] = sequence->incoming[coordinate] + sequence->residual[coordinate];
    transformer_aggregate(transformer, sequence, 38);
    if (!transformer_moe(transformer, sequence)) { sequence->failed = 1; return 0; }
    if (!transformer_finite(sequence->residual, TRANSFORMER_WIDTH)) { sequence->failed = 1; return 0; }
    memcpy(output, sequence->residual, sizeof sequence->residual);
    memcpy(output_snapshots, sequence->snapshots, (size_t)sequence->snapshot_count * TRANSFORMER_WIDTH * sizeof(float));
    sequence->positions++;
    return 1;
}

int transformer_routes(const TransformerSequence *sequence, unsigned selected[16])
{
    if (!sequence || !selected || !sequence->positions || sequence->failed) return 0;
    memcpy(selected, sequence->selected, sizeof sequence->selected);
    return 1;
}

#ifndef TRANSFORMER_NO_MAIN
static int transformer_receive(float *input, float *snapshots)
{
    int status = scanf("%f", input);
    if (status == EOF && !ferror(stdin)) return 0;
    if (status != 1) return -1;
    for (unsigned coordinate = 1; coordinate < TRANSFORMER_WIDTH; coordinate++) if (scanf("%f", input + coordinate) != 1) return -1;
    for (unsigned coordinate = 0; coordinate < TRANSFORMER_INPUT_SNAPSHOTS * TRANSFORMER_WIDTH; coordinate++)
        if (scanf("%f", snapshots + coordinate) != 1) return -1;
    return 1;
}

static int transformer_send(const float *output, const float *snapshots)
{
    for (unsigned row = 0; row <= TRANSFORMER_OUTPUT_SNAPSHOTS; row++) {
        const float *values = row ? snapshots + (row - 1) * TRANSFORMER_WIDTH : output;
        for (unsigned coordinate = 0; coordinate < TRANSFORMER_WIDTH; coordinate++)
            printf("%a%c", (double)values[coordinate], coordinate + 1 == TRANSFORMER_WIDTH ? '\n' : ' ');
    }
    return fflush(stdout) == 0 && !ferror(stdout);
}

static int transformer_stream(Transformer *transformer)
{
    TransformerSequence *sequence = transformer_sequence_create(transformer);
    if (!sequence) return 1;
    float input[7168], snapshots[8 * 7168], output[7168], output_snapshots[8 * 7168];
    int result = 0;
    for (;;) {
        int status = transformer_receive(input, snapshots);
        if (!status) break;
        if (status < 0) { result = 2; break; }
        if (!transformer_process(transformer, sequence, input, snapshots, TRANSFORMER_INPUT_SNAPSHOTS, output, output_snapshots) ||
            !transformer_send(output, output_snapshots)) { result = 1; break; }
    }
    if (result) fputs("transformer: incomplete, invalid or failed vector bundle\n", stderr);
    transformer_sequence_close(sequence);
    return result;
}

static void transformer_inspect(const Transformer *transformer)
{
    unsigned count = 0;
    for (unsigned id = 0; id < TRANSFORMER_RECORDS; id++) count += transformer->records[id].data != NULL;
    printf("layer %u: %s, %u additional records, 896 experts, snapshots %u -> %u\n",
        TRANSFORMER_LAYER, TRANSFORMER_MLA ? "MLA" : "KDA", count, TRANSFORMER_INPUT_SNAPSHOTS, TRANSFORMER_OUTPUT_SNAPSHOTS);
}

int main(int argc, char **argv)
{
    if ((argc != 2 && argc != 3) || (strcmp(argv[1], "--inspect") && strcmp(argv[1], "--stream"))) {
        fputs("usage: transformer-N --inspect|--stream [DATASET_DIRECTORY]\n", stderr);
        return 2;
    }
    Transformer *transformer = NULL;
    if (!transformer_open(argc == 3 ? argv[2] : "dataset", &transformer)) {
        fputs("transformer: invalid or unreadable layer-specific datasets\n", stderr);
        return 1;
    }
    int result = 0;
    if (!strcmp(argv[1], "--stream")) result = transformer_stream(transformer);
    else transformer_inspect(transformer);
    transformer_close(transformer);
    return result;
}
#endif