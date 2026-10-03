#define _GNU_SOURCE
#define SERVER_NO_MAIN
#include "server.c"
#include <assert.h>

#define main original_layer_zero_main
#include "reference-zero.inc"
#undef main

static const char *test_qkv_path;
static const char *test_trunk_directory;

static int test_equal(const float *actual, const float *expected, size_t count, const char *label)
{
    size_t differences = 0;
    for (size_t coordinate = 0; coordinate < count; coordinate++)
        if (memcmp(actual + coordinate, expected + coordinate, sizeof(float))) differences++;
    if (differences) fprintf(stderr, "%s: %zu of %zu values differ\n", label, differences, count);
    return differences == 0;
}

static int test_compare_reference(void)
{
    assert(cur_L == 0 && nsnap == 1 && have_prefix == 1);
    Server *server = NULL;
    assert(server_open(test_qkv_path, test_trunk_directory, &server));
    ServerSequence *sequence = server_sequence_create(server);
    ServerSequence *independent = server_sequence_create(server);
    assert(sequence && independent);
    float output[SERVER_WIDTH], saved[SERVER_WIDTH], bad_input[SERVER_WIDTH] = {0};
    for (unsigned position = 0; position < NPOS; position++) {
        assert(server_process(server, sequence, snap[position][0], output, saved));
        assert(test_equal(output, resid[position], E, "layer-0 residual"));
        assert(test_equal(saved, snap[position][0], E, "snapshot S0"));
    }
    assert(sequence->positions == NPOS && independent->positions == 0);
    assert(server_process(server, independent, snap[0][0], output, saved));
    assert(test_equal(output, resid[0], E, "independent sequence"));
    server_sequence_reset(sequence);
    assert(sequence->positions == 0);
    assert(server_process(server, sequence, snap[0][0], output, saved));
    assert(test_equal(output, resid[0], E, "reset sequence"));
    bad_input[0] = NAN;
    assert(!server_process(server, sequence, bad_input, output, saved));
    assert(!server_process(NULL, sequence, snap[0][0], output, saved));
    assert(!server_process(server, NULL, snap[0][0], output, saved));
    assert(!server_process(server, sequence, NULL, output, saved));
    assert(!server_process(server, sequence, snap[0][0], output, output));
    assert(sequence->positions == 1 && !sequence->failed);
    server_sequence_close(independent);
    server_sequence_close(sequence);
    server_close(server);
    assert(!server_open(NULL, test_trunk_directory, &server) && server == NULL);
    assert(!server_open(test_qkv_path, test_trunk_directory, NULL));
    printf("PASS: original layer-0 residual and S0 exact at %d positions; independent/reset sequences and invalid inputs\n", NPOS);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc != 3) return 2;
    test_qkv_path = argv[1];
    test_trunk_directory = argv[2];
    char *reference_args[] = {"reference-layer-zero", "0", NULL};
    return original_layer_zero_main(2, reference_args);
}