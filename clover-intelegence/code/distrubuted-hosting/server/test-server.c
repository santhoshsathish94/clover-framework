#define SERVER_NO_MAIN
#include "server.c"
#include <assert.h>

int main(int argc, char **argv)
{
    if (argc != 3) return 2;
    float palette[256];
    unsigned char ids[16];
    float input_row[16];
    for (unsigned code = 0; code < 256; code++) palette[code] = (float)code;
    for (unsigned lane = 0; lane < 16; lane++) { ids[lane] = (unsigned char)lane; input_row[lane] = 1.0f; }
    assert(server_project_row(input_row, ids, 16, palette, 2.0f) == 240.0f);
    assert(server_sigmoid(0.0f) == 0.5f);
    const unsigned char taps[] = {0,0,128,63, 0,0,0,64, 0,0,64,64, 0,0,128,64};
    float history[3] = {2, 3, 4};
    assert(server_convolve(1, taps, history) == 30.0f);
    assert(history[0] == 3 && history[1] == 4 && history[2] == 1);

    Server *server = NULL;
    assert(server_open(argv[1], argv[2], &server));
    unsigned records = 0;
    for (unsigned id = 0; id < SERVER_RECORDS; id++) records += server->records[id].data != NULL;
    assert(records == 13 && server->records[5].data && !server->records[1].data);
    ServerSequence *sequence = server_sequence_create(server);
    assert(sequence && !sequence->positions && !sequence->failed);
    float input[SERVER_WIDTH] = {0}, output[SERVER_WIDTH], snapshot[SERVER_WIDTH];
    for (unsigned position = 0; position < 2; position++) {
        assert(server_process(server, sequence, input, output, snapshot));
        for (unsigned coordinate = 0; coordinate < SERVER_WIDTH; coordinate++) assert(output[coordinate] == 0.0f);
        assert(!memcmp(input, snapshot, sizeof input));
    }
    assert(sequence->positions == 2);
    input[0] = INFINITY;
    assert(!server_process(server, sequence, input, output, snapshot));
    assert(sequence->positions == 2);
    server_sequence_reset(sequence);
    assert(sequence->positions == 0 && sequence->owner == server);
    server_sequence_close(sequence);
    server_close(server);
    puts("PASS: standalone layer-0 primitives, 13 bound records, two zero-input steps, reset and nonfinite rejection");
    return 0;
}