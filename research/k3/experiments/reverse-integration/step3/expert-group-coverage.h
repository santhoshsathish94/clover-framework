#include "expert-group-order.h"
#include "expert-joint-codec.h"
#include "expert-group-equation.h"
#include "expert-scale-codec.h"

typedef struct {
    int index;
    unsigned scale;
    unsigned char codes[32];
    uint32_t masks[4];
} CoverageGroup;

static FILE *coverage_log;
static int coverage_first = -1;
static unsigned char coverage_seen[NLAY][NEXP];
static unsigned long long coverage_groups, coverage_weights, coverage_outputs, coverage_pairs;

#include "reverse-codec.h"
#include "reverse-histogram.h"
#include "reverse-order.h"
#include "reverse-mask.h"
#include "reverse-four-mask.h"

static int coverage_mode(void)
{
    static int mode = -1;
    if (mode < 0) {
        const char *setting = getenv("K3_GROUP_COVERAGE");
        mode = setting ? atoi(setting) : 0;
        if (mode < 0 || mode > 2) die("K3_GROUP_COVERAGE must be 0, 1 or 2");
    }
    return mode;
}

static void coverage_start(void)
{
    if (coverage_log) return;
    const char *path = getenv("K3_GROUP_COVERAGE_REPORT");
    if (!path) die("K3_GROUP_COVERAGE_REPORT required");
    coverage_log = fopen(path, "wb");
    if (!coverage_log) die("coverage report open");
    group_order_selfcheck();
    expert_group_selfcheck();
    if (joint_enabled()) joint_selfcheck();
}

static CoverageGroup coverage_decode(const unsigned char *packed, unsigned scale,
                                    int layer, int expert, int part, int row,
                                    int index, int width, int rows, int positions)
{
    unsigned char original[32];
    if (!joint_enabled()) die("reverse step11 requires joint code consumer");
    for (unsigned coordinate = 0; coordinate < 32; coordinate++)
        original[coordinate] = (packed[coordinate / 2] >> (4 * (coordinate % 2))) & 15;
    unsigned counts[16];
    GroupOrderInteger rank, ways;
    if (!group_order_encode(original, 32, counts, &rank) || !group_order_ways(counts, 32, &ways))
        die("coverage arrangement encode failed");
    unsigned rank_bits = 0;
    for (GroupOrderInteger remaining = ways - 1; remaining; remaining >>= 1) rank_bits++;
    unsigned char payload[29] = {0};
    unsigned offset = 0;
    group_order_put(payload, &offset, scale, 8);
    joint_store(payload, &offset, counts, rank, rank_bits);
    const unsigned payload_bits = offset;
    const unsigned payload_bytes = (offset + 7) / 8;
    if (payload_bytes > sizeof(payload)) die("coverage payload overflow");

    CoverageGroup decoded = {0};
    decoded.index = index;
    offset = 0;
    decoded.scale = (unsigned)group_order_get(payload, &offset, 8);
    unsigned restored_counts[16];
    GroupOrderInteger restored_rank = joint_load(payload, &offset, restored_counts);
    unsigned char histogram_payload[5];
    reverse_histogram(restored_counts, histogram_payload);
    unsigned char order_payload[29];
    unsigned order_bits;
    reverse_order(&decoded.scale, restored_counts, &restored_rank, order_payload, &order_bits);
    if (offset != payload_bits || !group_order_decode(restored_counts, 32, restored_rank, decoded.codes))
        die("coverage arrangement decode failed");
    if (decoded.scale != scale || restored_rank != rank || memcmp(counts, restored_counts, sizeof(counts)))
        die("coverage stored fields differ");
    for (unsigned coordinate = 0; coordinate < 32; coordinate++) {
        if (decoded.codes[coordinate] != original[coordinate]) die("coverage decoded code differs");
        const float value = expert_group_weight(decoded.codes[coordinate], decoded.scale);
        if (memcmp(&value, &DQ[scale][original[coordinate]], sizeof(value))) die("coverage decoded weight differs");
    }
    unsigned char mask_payload[5];
    unsigned mask_bits;
    reverse_mask(decoded.codes, mask_payload, &mask_bits);
    unsigned char four_mask_payload[17];
    reverse_four_mask(&decoded, four_mask_payload);
    char rank_text[40];
    group_order_decimal(rank, rank_text);
    fprintf(coverage_log,
        "{\"type\":\"group\",\"layer\":%d,\"expert\":%d,\"part\":%d,"
        "\"row\":%d,\"group\":%d,\"width\":%d,\"rows\":%d,\"positions\":%d,"
        "\"scale\":%u,\"rank\":\"%s\",\"payload_bits\":%u,"
        "\"joint_mode\":%d,"
        "\"code_changes\":0,\"weight_changes\":0,\"packed\":[",
        layer, expert, part, row, index, width, rows, positions, scale, rank_text, payload_bits,
        joint_enabled());
    for (unsigned byte = 0; byte < 16; byte++) fprintf(coverage_log, "%s%u", byte ? "," : "", packed[byte]);
    fprintf(coverage_log, "],\"histogram_payload\":[");
    for (unsigned byte = 0; byte < 5; byte++) fprintf(coverage_log, "%s%u", byte ? "," : "", histogram_payload[byte]);
    fprintf(coverage_log, "],\"order_bits\":%u,\"order_payload\":[", order_bits);
    for (unsigned byte = 0; byte < (order_bits + 7) / 8; byte++)
        fprintf(coverage_log, "%s%u", byte ? "," : "", order_payload[byte]);
    fprintf(coverage_log, "],\"mask_bits\":%u,\"mask_payload\":[", mask_bits);
    for (unsigned byte = 0; byte < (mask_bits + 7) / 8; byte++)
        fprintf(coverage_log, "%s%u", byte ? "," : "", mask_payload[byte]);
    fprintf(coverage_log, "],\"four_mask_payload\":[");
    for (unsigned byte = 0; byte < 17; byte++)
        fprintf(coverage_log, "%s%u", byte ? "," : "", four_mask_payload[byte]);
    fprintf(coverage_log, "],\"payload\":[");
    for (unsigned byte = 0; byte < payload_bytes; byte++) fprintf(coverage_log, "%s%u", byte ? "," : "", payload[byte]);
    fprintf(coverage_log, "]}\n");
    coverage_groups++;
    coverage_weights += 32;
    return decoded;
}

static float coverage_project(const float *input, const unsigned char *packed,
                              const unsigned char *scales, int width,
                              const CoverageGroup groups[3])
{
    double lanes[4][4] = {{0}};
    for (int block = 0; block < width; block += 16) {
        for (int vector = 0; vector < 4; vector++) {
            for (int lane = 0; lane < 4; lane++) {
                const int coordinate = block + vector * 4 + lane;
                const int group = coordinate / 32;
                int selected = -1;
                for (int candidate = 0; candidate < 3; candidate++)
                    if (groups[candidate].index == group) selected = candidate;
                double weight;
                if (selected >= 0) {
                    weight = (double)expert_group_weight(expert_group_code(groups[selected].masks, coordinate % 32),
                                                         groups[selected].scale);
                } else {
                    const unsigned code = (packed[coordinate / 2] >> (4 * (coordinate % 2))) & 15;
                    weight = DQd[scales[group]][code];
                }
                lanes[vector][lane] += weight * (double)input[coordinate];
            }
        }
    }
    double partial[4];
    for (int lane = 0; lane < 4; lane++)
        partial[lane] = (lanes[0][lane] + lanes[2][lane]) + (lanes[1][lane] + lanes[3][lane]);
    return (float)((partial[0] + partial[2]) + (partial[1] + partial[3]));
}

static void coverage_replace(float *const *output, const float *const *input,
                             int positions, const unsigned char *packed,
                             const unsigned char *scales, int width, int rows,
                             int layer, int expert, int part, const int *position_ids, const int *ranks)
{
    const int mode = coverage_mode();
    if (!mode || (mode == 1 && layer != 1) ||
        (mode == 2 && layer != 1 && layer != 48 && layer != 92)) return;
    if (mode == 1) {
        if (coverage_first < 0) coverage_first = expert;
        if (expert != coverage_first) return;
    }
    if (expert < 0 || expert >= NEXP || part < 0 || part > 2 || positions < 1 || positions > NPOS)
        die("coverage site dimensions");
    if ((part < 2 && (width != 3584 || rows != 3072)) ||
        (part == 2 && (width != 3072 || rows != 3584))) die("coverage matrix shape");
    if (coverage_seen[layer][expert] & (1U << part)) die("coverage duplicate expert part");
    coverage_start();
    if (!coverage_seen[layer][expert]) coverage_pairs++;
    coverage_seen[layer][expert] |= 1U << part;
    const int selected_rows[3] = {0, rows / 2, rows - 1};
    const int group_count = width / 32;
    const int selected_groups[3] = {0, group_count / 2, group_count - 1};
    for (int row_index = 0; row_index < 3; row_index++) {
        const int row = selected_rows[row_index];
        const unsigned char *row_codes = packed + (size_t)row * (width / 2);
        const unsigned char *row_scales = scales + (size_t)row * group_count;
        unsigned char restored_scales[112];
        if (scale_row_enabled()) {
            scale_prepare(row_scales, row_codes, group_count, restored_scales,
                coverage_log, layer, expert, part, row, rows, positions);
            row_scales = restored_scales;
        }
        CoverageGroup decoded[3];
        for (int group = 0; group < 3; group++)
            decoded[group] = coverage_decode(row_codes + selected_groups[group] * 16,
                row_scales[selected_groups[group]], layer, expert, part, row,
                selected_groups[group], width, rows, positions);
        for (int position = 0; position < positions; position++) {
            const float original = output[position][row];
            const float candidate = coverage_project(input[position], row_codes, row_scales, width, decoded);
            uint32_t original_bits, candidate_bits;
            memcpy(&original_bits, &original, sizeof(original_bits));
            memcpy(&candidate_bits, &candidate, sizeof(candidate_bits));
            fprintf(coverage_log,
                "{\"type\":\"projection\",\"layer\":%d,\"expert\":%d,\"part\":%d,"
                "\"row\":%d,\"position\":%d,\"rank\":%d,\"reference_bits\":%u,\"candidate_bits\":%u}\n",
                layer, expert, part, row, position_ids[position], ranks[position], original_bits, candidate_bits);
            if (candidate_bits != original_bits) die("coverage projection differs");
            output[position][row] = candidate;
            coverage_outputs++;
        }
    }
}

static void coverage_finish(void)
{
    const int mode = coverage_mode();
    if (!mode) return;
    if (!coverage_log || !coverage_pairs) die("coverage never executed");
    unsigned pairs_by_layer[NLAY] = {0};
    for (int layer = 0; layer < NLAY; layer++)
        for (int expert = 0; expert < NEXP; expert++) {
            if (!coverage_seen[layer][expert]) continue;
            if (coverage_seen[layer][expert] != 7) die("coverage missing expert matrix");
            pairs_by_layer[layer]++;
        }
    if (coverage_groups != coverage_pairs * 27 || coverage_weights != coverage_groups * 32)
        die("coverage group count mismatch");
    scale_finish(coverage_log, coverage_pairs);
    if ((mode == 1 && (coverage_pairs != 1 || pairs_by_layer[1] != 1)) ||
        (mode == 2 && (!pairs_by_layer[1] || !pairs_by_layer[48] || !pairs_by_layer[92])))
        die("coverage layer count mismatch");
    fprintf(coverage_log,
        "{\"type\":\"summary\",\"mode\":%d,\"pairs\":%llu,\"groups\":%llu,"
        "\"weights\":%llu,\"projection_outputs\":%llu,\"decoder_checks\":%llu,"
        "\"invalid_inputs_rejected\":%llu,\"decode_pairs_checked\":4096,\"mask_coordinates_checked\":512,"
        "\"joint_mode\":%d,\"joint_decoder_checks\":%llu,\"joint_invalid_inputs_rejected\":%llu,"
        "\"layer_pairs\":{\"1\":%u,\"48\":%u,\"92\":%u},\"gate\":\"PASS\"}\n",
        mode, coverage_pairs, coverage_groups, coverage_weights, coverage_outputs,
        group_order_roundtrips, group_order_invalid, joint_enabled(), joint_decoder_checks,
        joint_invalid_checks, pairs_by_layer[1], pairs_by_layer[48], pairs_by_layer[92]);
    if (fclose(coverage_log) != 0) die("coverage report close");
    coverage_log = NULL;
}