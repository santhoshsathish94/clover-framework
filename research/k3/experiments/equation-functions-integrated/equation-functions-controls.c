#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "root-function.h"

typedef struct { unsigned rows, width; } SeedInput;
static unsigned seed_reads, projection_calls;

static void die(const char *message) {
    fprintf(stderr, "%s\n", message);
    exit(1);
}

static void seed_read_row(SeedInput *reader, unsigned token, uint16_t *output) {
    if (!reader || token >= reader->rows) die("invalid underlying seed read");
    seed_reads++;
    for (unsigned coordinate = 0; coordinate < reader->width; coordinate++)
        output[coordinate] = (uint16_t)((token + coordinate) & 65535);
}

#include "seed-function.h"

void l1_compact_project(float *const *outputs, const float *const *inputs,
                        int positions, int expert, int matrix) {
    if (!outputs || !inputs || positions != 1 || expert != 0 || matrix != (int)(projection_calls % 3))
        die("root projection call contract");
    unsigned rows = matrix == 2 ? 3584 : 3072;
    if (matrix == 2) {
        for (unsigned coordinate = 0; coordinate < 3072; coordinate++)
            if (inputs[0][coordinate] != 0.0f) die("root SiTU zero activation differs");
    }
    for (unsigned row = 0; row < rows; row++) outputs[0][row] = 0.0f;
    projection_calls++;
}

static unsigned observed;
static void observer(void *context, const char *stage, float *const *values, int positions, int width) {
    const char *names[] = {"gate", "up", "situ", "down"};
    if (context != &observed || observed >= 4 || strcmp(stage, names[observed]) ||
        positions != 1 || width != (observed == 3 ? 3584 : 3072)) die("root observer contract");
    for (int coordinate = 0; coordinate < width; coordinate++)
        if (values[0][coordinate] != 0.0f) die("root observer value differs");
    observed++;
}

int main(int argc, char **argv) {
    SeedInput reader = {163840, 7168};
    uint16_t embedding[7168];
    float input[3584] = {0}, output[3584];
    const float *inputs[] = {input};
    float *outputs[] = {output};
    if (argc == 2) {
        if (!strcmp(argv[1], "seed-token")) seed(&reader, 163840, embedding);
        else if (!strcmp(argv[1], "seed-shape")) { reader.width = 3; seed(&reader, 0, embedding); }
        else if (!strcmp(argv[1], "seed-null")) seed(NULL, 0, embedding);
        else if (!strcmp(argv[1], "root-layer")) root(2, 0, 1, inputs, outputs, NULL, NULL);
        else if (!strcmp(argv[1], "root-expert")) root(1, 896, 1, inputs, outputs, NULL, NULL);
        else if (!strcmp(argv[1], "root-positions")) root(1, 0, 0, inputs, outputs, NULL, NULL);
        else if (!strcmp(argv[1], "root-null")) root(1, 0, 1, NULL, outputs, NULL, NULL);
        else die("unknown control");
        die("invalid control unexpectedly returned");
    }
    seed(&reader, 0, embedding);
    if (embedding[0] != 0 || embedding[7167] != 7167) die("seed first token differs");
    seed(&reader, 163839, embedding);
    if (embedding[0] != (163839 & 65535) || embedding[7167] != ((163839 + 7167) & 65535) ||
        seed_reads != 2 || seed_function_calls != 2) die("seed last token differs");
    root(1, 0, 1, inputs, outputs, observer, &observed);
    if (observed != 4 || projection_calls != 3) die("root observed stage coverage");
    root(1, 0, 1, inputs, outputs, NULL, NULL);
    if (projection_calls != 6) die("root without observer incomplete");
    for (unsigned coordinate = 0; coordinate < 3584; coordinate++)
        if (output[coordinate] != 0.0f) die("root output differs");
    puts("PASS: seed boundary tokens and root full stage order, with and without observer.");
    return 0;
}