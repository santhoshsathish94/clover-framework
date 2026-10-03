#include "decode.h"
#include <stdio.h>
#include <stdlib.h>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

static unsigned test_u32(const unsigned char *bytes)
{
    return (unsigned)bytes[0] | ((unsigned)bytes[1] << 8) |
        ((unsigned)bytes[2] << 16) | ((unsigned)bytes[3] << 24);
}

static int test_batch(void)
{
    unsigned char header[8], input[250000], output[200001];
    for (;;) {
        size_t received = fread(header, 1, sizeof header, stdin);
        if (!received && feof(stdin) && !ferror(stdin)) break;
        if (received != sizeof header) return 2;
        unsigned input_bytes = test_u32(header), output_bytes = test_u32(header + 4);
        if (input_bytes > sizeof input || output_bytes > 200000 ||
            fread(input, 1, input_bytes, stdin) != input_bytes) return 2;
        int valid = decode_zlib(input, input_bytes, output, output_bytes);
        if (putchar(valid ? 0 : 1) == EOF) return 2;
        if (valid && fwrite(output, 1, output_bytes, stdout) != output_bytes) return 2;
    }
    return fflush(stdout) || ferror(stdout) ? 2 : 0;
}

int main(int argc, char **argv)
{
#ifdef _WIN32
    if (_setmode(_fileno(stdin), _O_BINARY) < 0 || _setmode(_fileno(stdout), _O_BINARY) < 0) return 2;
#endif
    if (argc != 2) return 2;
    if (!strcmp(argv[1], "--batch")) return test_batch();
    char *end;
    unsigned long expected = strtoul(argv[1], &end, 10);
    if (*end || expected > 200000) return 2;
    unsigned char input[250000], output[200001];
    size_t bytes = fread(input, 1, sizeof input, stdin);
    if (ferror(stdin) || !feof(stdin)) return 2;
    if (!decode_zlib(input, bytes, output, expected)) return 1;
    return fwrite(output, 1, expected, stdout) == expected ? 0 : 2;
}