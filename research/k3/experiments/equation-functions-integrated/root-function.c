#include "root-function.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef NPOS
#define NPOS 5
#endif

void l1_compact_project(float *const *outputs, const float *const *inputs,
                        int positions, int expert, int matrix);

void root(int layer, int expert, int positions, const float *const *input,
          float *const *output, RootObserver observer, void *context) {
    if (layer != 1 || expert < 0 || expert >= 896 || positions < 1 || positions > NPOS || !input || !output) {
        fprintf(stderr, "root: unsupported layer or invalid expert/input dimensions\n");
        exit(1);
    }
    float gate[NPOS][3072], up[NPOS][3072];
    float *gate_rows[NPOS], *up_rows[NPOS];
    const float *hidden_rows[NPOS];
    for (int position = 0; position < positions; position++) {
        if (!input[position] || !output[position]) {
            fprintf(stderr, "root: null input/output row\n");
            exit(1);
        }
        gate_rows[position] = gate[position];
        up_rows[position] = up[position];
        hidden_rows[position] = gate[position];
    }
    l1_compact_project(gate_rows, input, positions, expert, 0);
    if (observer) observer(context, "gate", gate_rows, positions, 3072);
    l1_compact_project(up_rows, input, positions, expert, 1);
    if (observer) observer(context, "up", up_rows, positions, 3072);
    for (int position = 0; position < positions; position++) {
        for (int coordinate = 0; coordinate < 3072; coordinate++) {
            const float raw_gate = gate[position][coordinate];
            const float sigmoid = 1.0f / (1.0f + expf(-raw_gate));
            const float activated = (4.0f * tanhf(raw_gate / 4.0f)) * sigmoid;
            const float capped_up = 25.0f * tanhf(up[position][coordinate] / 25.0f);
            gate[position][coordinate] = activated * capped_up;
        }
    }
    if (observer) observer(context, "situ", gate_rows, positions, 3072);
    l1_compact_project(output, hidden_rows, positions, expert, 2);
    if (observer) observer(context, "down", output, positions, 3584);
}