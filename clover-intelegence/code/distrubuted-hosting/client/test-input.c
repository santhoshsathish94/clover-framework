#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L
#define CLIENT_NO_MAIN
#include "client.c"
#include <assert.h>

static int input_reference_seek(FILE *file, uint64_t offset)
{
#ifdef _WIN32
    return _fseeki64(file, (int64_t)offset, SEEK_SET) == 0;
#else
    return fseeko(file, (off_t)offset, SEEK_SET) == 0;
#endif
}

int main(int argc, char **argv)
{
    if (argc < 4) return 2;
    FILE *reference = fopen(argv[2], "rb");
    ClientInput *input = client_input_open(argv[1]);
    assert(reference && input);
    float vector[CLIENT_WIDTH];
    unsigned char raw[CLIENT_WIDTH * 2];
    unsigned checked = 0;
    for (int argument = 3; argument < argc; argument++) {
        uint32_t token;
        assert(decimal_id((const unsigned char *)argv[argument], strlen(argv[argument]), &token));
        assert(client_input(input, token, vector));
        assert(input_reference_seek(reference, UINT64_C(2348810824) + (uint64_t)token * sizeof raw));
        assert(fread(raw, 1, sizeof raw, reference) == sizeof raw);
        for (unsigned coordinate = 0; coordinate < CLIENT_WIDTH; coordinate++) {
            uint32_t bits = ((uint32_t)raw[coordinate * 2] | ((uint32_t)raw[coordinate * 2 + 1] << 8)) << 16;
            assert(!memcmp(vector + coordinate, &bits, sizeof bits));
        }
        checked++;
        printf("INPUT_MATCH position=%u token=%u dimensions=%u\n", checked - 1, token, CLIENT_WIDTH);
    }
    assert(!fclose(reference));
    client_input_close(input);
    printf("PASS: %u token vectors, %u float32 coordinates exact against original embedding rows; no layer execution\n",
        checked, checked * CLIENT_WIDTH);
    return 0;
}