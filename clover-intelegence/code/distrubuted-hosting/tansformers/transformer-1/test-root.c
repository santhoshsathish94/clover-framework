#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L
#include "root.h"
#include <assert.h>

int main(int argc, char **argv)
{
    if (argc != 2) return 2;
    Root *root = root_open(argv[1]);
    RootScratch *scratch = malloc(sizeof *scratch);
    assert(root && scratch);
    unsigned checked = 0;
    const unsigned experts[] = {0, 498, 895};
    for (unsigned index = 0; index < 3; index++)
        for (unsigned matrix = 0; matrix < 3; matrix++) {
            unsigned blocks = matrix == 2 ? 56 : 48;
            for (unsigned sample = 0; sample < 3; sample++) {
                unsigned block = sample == 0 ? 0 : sample == 1 ? blocks / 2 : blocks - 1;
                assert(root_block(root, scratch, experts[index], matrix, block));
                checked++;
            }
        }
    float input[3584] = {0}, output[3584];
    for (unsigned matrix = 0; matrix < 3; matrix++) {
        assert(root_project(root, scratch, 0, matrix, input, output));
        unsigned rows = matrix == 2 ? 3584 : 3072;
        for (unsigned row = 0; row < rows; row++) assert(output[row] == 0.0f);
    }
    assert(!root_block(root, scratch, 896, 0, 0));
    assert(!root_block(root, scratch, 0, 3, 0));
    assert(!root_project(root, scratch, 0, 0, NULL, output));
    printf("PASS: %u distributed real block checksums; complete expert-0 zero projections; address controls\n", checked);
    free(scratch);
    root_close(root);
    return 0;
}