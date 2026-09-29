#ifndef K3_LAYER1_COMPACT_MODEL_H
#define K3_LAYER1_COMPACT_MODEL_H

int l1_compact_enabled(void);
void l1_compact_project(float *const *outputs, const float *const *inputs,
                        int positions, int expert, int matrix);
void l1_compact_finish(void);

static float l1_oracle_gate[NPOS][3072];
static float l1_oracle_up[NPOS][3072];
static uint64_t l1_checked_projection_values, l1_checked_situ_values;
static unsigned l1_projection_calls, l1_situ_position;

static void l1_checked_Xm(float *const *outputs, const float *const *inputs, int positions,
                          const unsigned char *packed, const unsigned char *scales,
                          int width, int rows, const unsigned char *next_packed,
                          const unsigned char *next_scales) {
    if (cur_L != 1 || !l1_compact_enabled()) {
        Xm(outputs, inputs, positions, packed, scales, width, rows, next_packed, next_scales);
        return;
    }
    int matrix = !strcmp(cur_part, "gate") ? 0 : !strcmp(cur_part, "up") ? 1 : !strcmp(cur_part, "down") ? 2 : -1;
    if (matrix < 0 || width != (matrix == 2 ? 3072 : 3584) || rows != (matrix == 2 ? 3584 : 3072))
        die("layer1 compact projection geometry");
    float reference[NPOS][3584];
    float *oracle[NPOS];
    for (int position = 0; position < positions; position++) oracle[position] = reference[position];
    Xm(oracle, inputs, positions, packed, scales, width, rows, next_packed, next_scales);
    l1_compact_project(outputs, inputs, positions, cur_e, matrix);
    for (int position = 0; position < positions; position++) {
        if (memcmp(outputs[position], oracle[position], (size_t)rows * sizeof(float)))
            die("layer1 compact projection mismatch");
        if (matrix == 0) memcpy(l1_oracle_gate[position], oracle[position], (size_t)rows * sizeof(float));
        if (matrix == 1) memcpy(l1_oracle_up[position], oracle[position], (size_t)rows * sizeof(float));
    }
    if (matrix == 0) l1_situ_position = 0;
    l1_checked_projection_values += (uint64_t)positions * (unsigned)rows;
    l1_projection_calls++;
}

static void l1_checked_situ(float *output, const float *gate, const float *up, int count) {
    if (cur_L != 1 || count != 3072 || !l1_compact_enabled()) {
        situ(output, gate, up, count);
        return;
    }
    if (count != 3072 || l1_situ_position >= NPOS) die("layer1 compact SiTU address");
    float expected[3072];
    situ(expected, l1_oracle_gate[l1_situ_position], l1_oracle_up[l1_situ_position], count);
    situ(output, gate, up, count);
    if (memcmp(output, expected, sizeof(expected))) die("layer1 compact SiTU mismatch");
    l1_checked_situ_values += (unsigned)count;
    l1_situ_position++;
}

static void l1_model_finish(void) {
    if (!l1_compact_enabled()) return;
    if (!l1_projection_calls || !l1_checked_situ_values) die("layer1 compact checks missing");
    const char *path = getenv("K3_L1_CHECK");
    if (!path) die("layer1 compact check report required");
    FILE *report = fopen(path, "wx");
    if (!report) die("layer1 compact report open");
    fprintf(report, "{\"gate\":\"PASS\",\"layer\":1,\"projection_calls\":%u,"
            "\"projection_values_checked\":%llu,\"situ_values_checked\":%llu,"
            "\"compact_outputs_feed_model\":true,\"original_packed_data_retained_for_oracle\":true}\n",
            l1_projection_calls, (unsigned long long)l1_checked_projection_values,
            (unsigned long long)l1_checked_situ_values);
    fclose(report);
    l1_compact_finish();
}

#define Xm l1_checked_Xm
#define situ l1_checked_situ
#endif