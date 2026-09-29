#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static void die(const char *message)
{
    fprintf(stderr, "%s\n", message);
    exit(2);
}

#include "seed-reader.h"

int main(int argc, char **argv)
{
    if (argc != 3) return 2;
    SeedInput seed;
    seed_open(&seed, argv[1]);
    if (!strcmp(argv[2], "all")) {
        for (unsigned block = 0; block < seed.blocks; block++) {
            seed_load_block(&seed, block);
            const unsigned count = seed_u32(seed.index + (size_t)block * 56 + 12);
            if (fwrite(seed.raw, 2, (size_t)count * seed.width, stdout) != (size_t)count * seed.width)
                die("test output failed");
        }
    } else {
        char *end;
        errno = 0;
        unsigned long row = strtoul(argv[2], &end, 10);
        if (errno || *end || row > UINT32_MAX) die("test row invalid");
        uint16_t *words = malloc(seed.width * sizeof(*words));
        if (!words) die("test row allocation");
        seed_read_row(&seed, (unsigned)row, words);
        for (unsigned index = 0; index < seed.width; index++) {
            const unsigned char bytes[2] = {words[index], words[index] >> 8};
            if (fwrite(bytes, 1, 2, stdout) != 2) die("test row output");
        }
        free(words);
    }
    seed_close(&seed);
    return 0;
}