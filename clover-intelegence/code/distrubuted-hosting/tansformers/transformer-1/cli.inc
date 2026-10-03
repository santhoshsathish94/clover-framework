#ifndef TRANSFORMER_NO_MAIN
static int transformer_receive(float *input, float *snapshot)
{
    int status = scanf("%f", input);
    if (status == EOF && !ferror(stdin)) return 0;
    if (status != 1) return -1;
    for (unsigned coordinate = 1; coordinate < TRANSFORMER_WIDTH; coordinate++)
        if (scanf("%f", input + coordinate) != 1) return -1;
    for (unsigned coordinate = 0; coordinate < TRANSFORMER_WIDTH; coordinate++)
        if (scanf("%f", snapshot + coordinate) != 1) return -1;
    return 1;
}

static int transformer_send(const float *output, const float *snapshot)
{
    for (unsigned coordinate = 0; coordinate < TRANSFORMER_WIDTH; coordinate++)
        printf("%a%c", (double)output[coordinate], coordinate + 1 == TRANSFORMER_WIDTH ? '\n' : ' ');
    for (unsigned coordinate = 0; coordinate < TRANSFORMER_WIDTH; coordinate++)
        printf("%a%c", (double)snapshot[coordinate], coordinate + 1 == TRANSFORMER_WIDTH ? '\n' : ' ');
    return fflush(stdout) == 0 && !ferror(stdout);
}

static int transformer_stream(Transformer *transformer)
{
    TransformerSequence *sequence = transformer_sequence_create(transformer);
    if (!sequence) return 1;
    float input[TRANSFORMER_WIDTH], snapshot[TRANSFORMER_WIDTH], output[TRANSFORMER_WIDTH];
    int result = 0;
    for (;;) {
        int status = transformer_receive(input, snapshot);
        if (!status) break;
        if (status < 0) { result = 2; break; }
        if (!transformer_process(transformer, sequence, input, snapshot, output) ||
            !transformer_send(output, snapshot)) { result = 1; break; }
    }
    if (result) fputs("transformer-1: incomplete, invalid or failed vector pair\n", stderr);
    transformer_sequence_close(sequence);
    return result;
}

static void transformer_inspect(const Transformer *transformer)
{
    unsigned count = 0;
    for (unsigned id = 0; id < TRANSFORMER_RECORDS; id++) count += transformer->records[id].data != NULL;
    printf("layer 1: KDA operator, %u additional trunk records, 896 computed-value experts\n", count);
}

int main(int argc, char **argv)
{
    if ((argc != 2 && argc != 3) || (strcmp(argv[1], "--inspect") && strcmp(argv[1], "--stream"))) {
        fputs("usage: transformer-1 --inspect|--stream [DATASET_DIRECTORY]\n", stderr);
        return 2;
    }
    Transformer *transformer = NULL;
    if (!transformer_open(argc == 3 ? argv[2] : "dataset", &transformer)) {
        fputs("transformer-1: invalid or unreadable layer-1 datasets\n", stderr);
        return 1;
    }
    int result = 0;
    if (!strcmp(argv[1], "--stream")) result = transformer_stream(transformer);
    else transformer_inspect(transformer);
    transformer_close(transformer);
    return result;
}