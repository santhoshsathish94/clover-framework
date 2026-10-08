/* Read-only: does a stored observation exist for each live expert step, and is the live
   input the same vector that produced it? Decides whether the stored gate/up/activation/
   down can be looked up by expert id instead of recomputed. Changes no engine behaviour. */
#define _GNU_SOURCE
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#define CLOVER_CHECK_EXPERT
#define main standalone_program_main
static void check_live_expert(unsigned layer, unsigned expert, const float *input, const float *output);
#include "clover-one.c"
#undef main

#define PROBE_RECORDS 80
#define PROBE_STRIDE 51204

static const unsigned char *probe_results[NLAY];
static const float *probe_inputs[NLAY];
static unsigned long long probe_steps, probe_id_found, probe_input_exact;
static double probe_input_worst = -1, probe_down_exact = -1, probe_down_idonly = -1;
static unsigned short probe_seen[NLAY][896], probe_hit[NLAY][896];
static unsigned probe_layer_steps[NLAY], probe_layer_byid[NLAY], probe_layer_exact[NLAY];

static void probe_open(unsigned layer)
{
    if (probe_results[layer]) return;
    char path[4096];
    size_t bytes;
    const char *root = getenv("K3_PREPARED_DATA");
    snprintf(path, sizeof path, "%s/root-%u/observations/france/results.bin", root, layer);
    probe_results[layer] = map_file(path, &bytes);
    snprintf(path, sizeof path, "%s/root-%u/observations/france/inputs.f32", root, layer);
    probe_inputs[layer] = (const float *)map_file(path, &bytes);
}

static void check_live_expert(unsigned layer, unsigned expert, const float *input, const float *output)
{
    if (layer < 1 || layer >= NLAY) return;
    probe_open(layer);
    probe_steps++;
    probe_layer_steps[layer]++;
    if (expert < 896) probe_seen[layer][expert]++;
    double best = 1e30;
    unsigned chosen = PROBE_RECORDS;
    for (unsigned record = 0; record < PROBE_RECORDS; record++) {
        int32_t identity;
        memcpy(&identity, probe_results[layer] + (size_t)record * PROBE_STRIDE, sizeof identity);
        if ((unsigned)identity != expert) continue;
        double worst = 0;
        const float *stored = probe_inputs[layer] + (size_t)record * LAT;
        for (unsigned k = 0; k < LAT; k++) {
            double d = fabs((double)input[k] - (double)stored[k]);
            if (d > worst) worst = d;
        }
        if (worst < best) { best = worst; chosen = record; }
    }
    if (chosen == PROBE_RECORDS) return;
    probe_id_found++;
    probe_layer_byid[layer]++;
    if (best > probe_input_worst) probe_input_worst = best;
    const float *down = (const float *)(probe_results[layer] + (size_t)chosen * PROBE_STRIDE + 4 + 3 * I_ * 4);
    double worst = 0;
    for (unsigned k = 0; k < LAT; k++) {
        double d = fabs((double)output[k] - (double)down[k]);
        if (d > worst) worst = d;
    }
    if (best == 0.0) {
        probe_input_exact++;
        probe_layer_exact[layer]++;
        if (expert < 896) probe_hit[layer][expert]++;
        if (worst > probe_down_exact) probe_down_exact = worst;
    } else if (worst > probe_down_idonly) {
        probe_down_idonly = worst;
    }
}

int main(int argc, char **argv)
{
    assert(argc == 2);
    assert(resident_read_config());
    resident_configure(argv[1]);
    result_options();
    load_index(getenv("K3_INDEX"));
    prepared_init();
    resident_startup(0);
    resident_sequence_clear();
    unsigned ids[5] = {1008, 10484, 318, 15383, 387};
    for (unsigned position = 0; position < 5; position++)
        evaluate_token(ids[position], position, position > 0);
    printf("expert steps            %llu\n", probe_steps);
    printf("stored record by id     %llu\n", probe_id_found);
    printf("live input bit-exact    %llu\n", probe_input_exact);
    printf("worst input |diff|      %.6g\n", probe_input_worst);
    printf("worst down |diff| exact %.6g\n", probe_down_exact);
    printf("worst down |diff| idonly %.6g\n", probe_down_idonly);
    unsigned long long pairs = 0, whole = 0;
    for (unsigned layer = 1; layer < NLAY; layer++)
        for (unsigned expert = 0; expert < 896; expert++) {
            if (!probe_seen[layer][expert]) continue;
            pairs++;
            if (probe_hit[layer][expert] == probe_seen[layer][expert]) whole++;
        }
    printf("expert reads needed     %llu\n", pairs);
    printf("expert reads avoidable  %llu  (%.1f%%, %.2f GB of %.2f GB)\n", whole,
           pairs ? 100.0 * (double)whole / (double)pairs : 0.0,
           (double)whole * 17547264.0 / 1e9, (double)pairs * 17547264.0 / 1e9);
    printf("\nper layer: steps / found-by-id / input-exact\n");
    for (unsigned layer = 1; layer < NLAY; layer++)
        printf("L%-3u %3u %3u %3u%s", layer, probe_layer_steps[layer], probe_layer_byid[layer],
               probe_layer_exact[layer], layer % 6 == 0 ? "\n" : "   ");
    printf("\n");
    return 0;
}
