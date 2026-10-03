#define _GNU_SOURCE
#define TRANSFORMER_NO_MAIN
#include "transformer-1.c"
#include <assert.h>
#define main reference_main
#include "reference-one.inc"
#undef main

static float test_inputs[5][7168];
static unsigned test_routes[5][16];
static unsigned char test_records[80][51204];
static float test_latents[80][3584];
static unsigned test_expert_positions[3];
static const char *test_dataset;

static void test_capture_input(int layer)
{
    if (layer == 1) memcpy(test_inputs, resid, sizeof test_inputs);
}

static void test_capture_route(int layer, int position, const int *ids)
{
    assert(layer == 1 && position >= 0 && position < 5);
    for (unsigned rank = 0; rank < 16; rank++) test_routes[position][rank] = (unsigned)ids[rank];
}

static void test_observed_expert(int layer, int expert, const char *part,
    float *const *output, const float *const *inputs, int positions, int width, int rows)
{
    assert(layer == 1 && expert >= 0 && expert < 896);
    unsigned matrix = !strcmp(part, "gate") ? 0 : !strcmp(part, "up") ? 1 : 2;
    assert((matrix == 2 && width == 3072 && rows == 3584) || (matrix < 2 && width == 3584 && rows == 3072));
    for (int position = 0; position < positions; position++) {
        int found = -1;
        for (unsigned record = 0; record < 80; record++) {
            if (root_u32(test_records[record]) != (unsigned)expert) continue;
            const void *expected = matrix == 2 ? test_records[record] + 4 + 2 * 3072 * 4 : (void *)test_latents[record];
            if (!memcmp(inputs[position], expected, (size_t)width * 4)) { found = (int)record; break; }
        }
        if (found < 0) { fprintf(stderr, "reference expert input missing: %d %s\n", expert, part); abort(); }
        size_t offset = matrix == 2 ? 4 + 3 * 3072 * 4 : 4 + matrix * 3072 * 4;
        memcpy(output[position], test_records[found] + offset, (size_t)rows * 4);
        test_expert_positions[matrix]++;
    }
}

static int test_compare_layer(void)
{
    assert(cur_L == 1 && nsnap == 1);
    for (unsigned matrix = 0; matrix < 3; matrix++) assert(test_expert_positions[matrix] == 80);
    Transformer *transformer = NULL;
    assert(transformer_open(test_dataset, &transformer));
    TransformerSequence *sequence = transformer_sequence_create(transformer);
    assert(sequence);
    float output[7168];
    for (unsigned position = 0; position < 5; position++) {
        assert(transformer_process(transformer, sequence, test_inputs[position], snap[position][0], output));
        size_t differences = 0;
        for (unsigned coordinate = 0; coordinate < 7168; coordinate++)
            if (memcmp(output + coordinate, resid[position] + coordinate, 4)) differences++;
        if (differences) fprintf(stderr, "position %u: %zu differing layer-1 residual values\n", position, differences);
        assert(!differences);
        assert(!memcmp(sequence->selected, test_routes[position], sizeof sequence->selected));
        assert(!memcmp(sequence->snapshot, snap[position][0], sizeof sequence->snapshot));
        printf("PASS: position %u, layer-1 residual, 16 routes and S0 exact\n", position);
        fflush(stdout);
    }
    transformer_sequence_reset(sequence);
    assert(!sequence->positions && !sequence->failed);
    assert(transformer_process(transformer, sequence, test_inputs[0], snap[0][0], output));
    assert(!memcmp(output, resid[0], sizeof output));
    float invalid[7168] = {NAN};
    assert(!transformer_process(transformer, sequence, invalid, snap[0][0], output));
    assert(sequence->positions == 1);
    transformer_sequence_close(sequence);
    transformer_close(transformer);
    puts("PASS: live layer-1 against original arithmetic and exact-input-bound original expert observations; reset and invalid inputs");
    return 0;
}

int main(int argc, char **argv)
{
    if (argc != 3) return 2;
    test_dataset = argv[1];
    char path[4096];
    int length = snprintf(path, sizeof path, "%s/root-1/observations/%s/results.bin", argv[1], argv[2]);
    assert(length > 0 && (size_t)length < sizeof path);
    FILE *file = fopen(path, "rb");
    assert(file && fread(test_records, 1, sizeof test_records, file) == sizeof test_records);
    assert(fgetc(file) == EOF && !ferror(file));
    assert(!fclose(file));
    length = snprintf(path, sizeof path, "%s/root-1/observations/%s/inputs.f32", argv[1], argv[2]);
    assert(length > 0 && (size_t)length < sizeof path);
    file = fopen(path, "rb");
    assert(file && fread(test_latents, 1, sizeof test_latents, file) == sizeof test_latents);
    assert(fgetc(file) == EOF && !ferror(file));
    assert(!fclose(file));
    char *args[] = {"original-layer-one", "1", NULL};
    return reference_main(2, args);
}