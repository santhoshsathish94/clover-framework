#ifndef K3_EQUATION_FUNCTIONS_MODEL_H
#define K3_EQUATION_FUNCTIONS_MODEL_H

#include "root-function.h"
int l1_compact_enabled(void);
void l1_compact_finish(void);

static float root_expected_gate[NPOS][3072], root_expected_up[NPOS][3072];
static unsigned long long root_function_calls, root_positions;
static unsigned long long root_stage_values[4];

static void root_oracle_Xm(float *const *outputs, const float *const *inputs, int positions,
                           const unsigned char *packed, const unsigned char *scales,
                           int width, int rows, const unsigned char *next_packed,
                           const unsigned char *next_scales) {
    Xm(outputs, inputs, positions, packed, scales, width, rows, next_packed, next_scales);
    if (cur_L != 1 || !l1_compact_enabled()) return;
    if (strcmp(cur_part, "gate") && strcmp(cur_part, "up")) return;
    if (positions < 1 || positions > NPOS || width != 3584 || rows != 3072)
        die("root oracle dimensions");
    for (int position = 0; position < positions; position++) {
        float *destination = !strcmp(cur_part, "gate") ? root_expected_gate[position] : root_expected_up[position];
        memcpy(destination, outputs[position], 3072 * sizeof(float));
    }
}

typedef struct {
    float *const *hidden;
    float (*down)[3584];
    unsigned stages;
} RootComparison;

static void root_compare(void *context, const char *stage, float *const *values, int positions, int width) {
    RootComparison *comparison = context;
    unsigned ordinal = !strcmp(stage, "gate") ? 0 : !strcmp(stage, "up") ? 1 : !strcmp(stage, "situ") ? 2 : !strcmp(stage, "down") ? 3 : 4;
    if (ordinal != comparison->stages || ordinal > 3 || positions < 1 || positions > NPOS ||
        width != (ordinal == 3 ? 3584 : 3072)) die("root observation order/dimensions");
    for (int position = 0; position < positions; position++) {
        const float *expected = ordinal == 0 ? root_expected_gate[position] : ordinal == 1 ? root_expected_up[position] :
                                ordinal == 2 ? comparison->hidden[position] : comparison->down[position];
        if (memcmp(values[position], expected, (size_t)width * sizeof(float))) die("root function stage mismatch");
    }
    root_stage_values[ordinal] += (unsigned long long)positions * (unsigned)width;
    comparison->stages++;
}

static void root_checked(int layer, int expert, int positions, const float *const *inputs,
                         float *const *hidden, float *const *outputs) {
    if (layer != 1 || !l1_compact_enabled()) return;
    float down[NPOS][3584];
    for (int position = 0; position < positions; position++)
        memcpy(down[position], outputs[position], 3584 * sizeof(float));
    RootComparison comparison = {hidden, down, 0};
    root(layer, expert, positions, inputs, outputs, root_compare, &comparison);
    if (comparison.stages != 4) die("root function not fully exercised");
    root_function_calls++;
    root_positions += (unsigned)positions;
}

static void equation_functions_report(void) {
    const char *path = getenv("K3_FUNCTIONS_REPORT");
    if (!path) die("K3_FUNCTIONS_REPORT required");
    FILE *report = fopen(path, "wx");
    if (!report) die("equation function report open");
    if (seed_model_enabled() && seed_function_calls != NPOS) die("seed function not fully exercised");
    if (fruit_enabled() && fruit_pass_count != 2) die("fruit function not fully exercised");
    if (l1_compact_enabled() && (!root_function_calls || root_positions != (unsigned long long)NPOS * 16))
        die("root function not fully exercised");
    fprintf(report, "{\"gate\":\"PASS\",\"seed_calls\":%llu,\"fruit_calls\":%u,\"root_calls\":%llu,"
            "\"root_positions\":%llu,\"root_stage_values\":[%llu,%llu,%llu,%llu],"
            "\"root_activation_scratch_bytes\":%zu,\"root_weight_array_bytes\":0,"
            "\"root_output_replaces_oracle\":true,\"root_layer\":1}\n",
            seed_function_calls, fruit_pass_count, root_function_calls, root_positions,
            root_stage_values[0], root_stage_values[1], root_stage_values[2], root_stage_values[3],
            2 * (size_t)NPOS * 3072 * sizeof(float));
    fclose(report);
    l1_compact_finish();
}

#define Xm root_oracle_Xm
#endif