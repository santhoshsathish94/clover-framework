static void expert_group_masks(const unsigned char *packed, uint32_t masks[4])
{
    for (int bit = 0; bit < 4; bit++) masks[bit] = 0;
    for (int coordinate = 0; coordinate < 32; coordinate++) {
        const unsigned code = (packed[coordinate / 2] >> (4 * (coordinate % 2))) & 15;
        for (int bit = 0; bit < 4; bit++)
            masks[bit] |= ((code >> bit) & 1U) << coordinate;
    }
}

static unsigned expert_group_code(const uint32_t masks[4], int coordinate)
{
    unsigned code = 0;
    for (int bit = 0; bit < 4; bit++)
        code |= ((masks[bit] >> coordinate) & 1U) << bit;
    return code;
}

static float expert_group_weight(unsigned code, unsigned scale)
{
    const unsigned exponent = (code >> 1) & 3;
    const unsigned mantissa = code & 1;
    const float magnitude = exponent == 0 ? 0.5f * mantissa :
        (1.0f + 0.5f * mantissa) * (float)(1U << (exponent - 1));
    const float signed_value = copysignf(magnitude, (code & 8) ? -1.0f : 1.0f);
    return scale == 255 ? copysignf(0.0f, signed_value) :
        scalbnf(signed_value, (int)scale - 127);
}

static void expert_group_selfcheck(void)
{
    for (unsigned scale = 0; scale < 256; scale++) {
        for (unsigned code = 0; code < 16; code++) {
            const float generated = expert_group_weight(code, scale);
            if (memcmp(&generated, &DQ[scale][code], sizeof(float)))
                die("expert group equation differs from reference decode");
        }
    }
    for (unsigned code = 0; code < 16; code++) {
        unsigned char packed[16];
        uint32_t masks[4];
        memset(packed, (int)(code | (code << 4)), sizeof(packed));
        expert_group_masks(packed, masks);
        for (int coordinate = 0; coordinate < 32; coordinate++)
            if (expert_group_code(masks, coordinate) != code)
                die("expert group mask coordinate differs");
    }
}

static float expert_group_projection(const float *input, const unsigned char *packed,
                                     const unsigned char *scales, int width,
                                     const uint32_t masks[4], unsigned group_scale)
{
    double lanes[4][4] = {{0}};
    for (int block = 0; block < width; block += 16) {
        for (int vector = 0; vector < 4; vector++) {
            for (int lane = 0; lane < 4; lane++) {
                const int coordinate = block + vector * 4 + lane;
                double weight;
                if (coordinate < 32) {
                    weight = (double)expert_group_weight(expert_group_code(masks, coordinate), group_scale);
                } else {
                    const unsigned code = (packed[coordinate / 2] >> (4 * (coordinate % 2))) & 15;
                    weight = DQd[scales[coordinate / 32]][code];
                }
                lanes[vector][lane] += weight * (double)input[coordinate];
            }
        }
    }
    double partial[4];
    for (int lane = 0; lane < 4; lane++)
        partial[lane] = (lanes[0][lane] + lanes[2][lane]) +
                        (lanes[1][lane] + lanes[3][lane]);
    return (float)((partial[0] + partial[2]) + (partial[1] + partial[3]));
}

static void expert_group_replace(float *const *output, const float *const *input,
                                 int positions, const unsigned char *packed,
                                 const unsigned char *scales, int width, int rows,
                                 int expert, const int *position_ids, const int *ranks)
{
    const char *setting = getenv("K3_EXPERT_GROUP");
    const int mode = setting ? atoi(setting) : 0;
    if (mode == 0) return;
    if (mode != 1) die("K3_EXPERT_GROUP must be 0 or 1");
    if (width != 3584 || rows != 3072 || positions < 1 || positions > NPOS ||
        position_ids[0] != 0 || ranks[0] != 0)
        die("expert group unexpected selected site");
    const char *path = getenv("K3_EXPERT_GROUP_REPORT");
    if (!path) die("K3_EXPERT_GROUP_REPORT required");
    expert_group_selfcheck();

    uint32_t masks[4];
    const unsigned scale = scales[0];
    expert_group_masks(packed, masks);
    unsigned char regenerated[16] = {0};
    float decoded[32];
    int changed_weights = 0;
    for (int coordinate = 0; coordinate < 32; coordinate++) {
        const unsigned code = expert_group_code(masks, coordinate);
        const unsigned original = (packed[coordinate / 2] >> (4 * (coordinate % 2))) & 15;
        regenerated[coordinate / 2] |= code << (4 * (coordinate % 2));
        decoded[coordinate] = expert_group_weight(code, scale);
        changed_weights += memcmp(decoded + coordinate, &DQ[scale][original], sizeof(float)) != 0;
    }
    const int changed_packed = memcmp(regenerated, packed, sizeof(regenerated)) != 0;
    float reference[NPOS], candidate[NPOS];
    int changed_outputs = 0;
    for (int position = 0; position < positions; position++) {
        reference[position] = output[position][0];
        candidate[position] = expert_group_projection(input[position], packed, scales, width, masks, scale);
        changed_outputs += memcmp(reference + position, candidate + position, sizeof(float)) != 0;
        output[position][0] = candidate[position];
    }

    FILE *report = fopen(path, "wb");
    if (!report) die("expert group report open");
    fprintf(report,
        "{\n  \"layer\": 1,\n  \"expert_id\": %d,\n  \"tensor\": \"w1/gate\",\n"
        "  \"row\": 0,\n  \"group\": 0,\n  \"input_width\": %d,\n  \"rows\": %d,\n"
        "  \"positions\": %d,\n  \"decode_pairs_checked\": 4096,\n"
        "  \"mask_coordinates_checked\": 512,\n  \"group_weights_checked\": 32,\n"
        "  \"group_weights_changed\": %d,\n  \"packed_roundtrip_exact\": %s,\n"
        "  \"projection_outputs_changed\": %d,\n  \"scale_byte\": %u,\n"
        "  \"masks\": [%u, %u, %u, %u],\n  \"packed_bytes\": [",
        expert, width, rows, positions, changed_weights,
        changed_packed ? "false" : "true", changed_outputs, scale,
        masks[0], masks[1], masks[2], masks[3]);
    for (int byte = 0; byte < 16; byte++) fprintf(report, "%s%u", byte ? ", " : "", packed[byte]);
    fprintf(report, "],\n  \"codes\": [");
    for (int coordinate = 0; coordinate < 32; coordinate++)
        fprintf(report, "%s%u", coordinate ? ", " : "", expert_group_code(masks, coordinate));
    fprintf(report, "],\n  \"decoded_weights\": [");
    for (int coordinate = 0; coordinate < 32; coordinate++)
        fprintf(report, "%s%.9g", coordinate ? ", " : "", decoded[coordinate]);
    fprintf(report, "],\n  \"projection_outputs\": [\n");
    for (int position = 0; position < positions; position++)
        fprintf(report, "    %s{\"position\": %d, \"rank\": %d, \"reference\": %.9g, \"candidate\": %.9g}\n",
            position ? "," : "", position_ids[position], ranks[position], reference[position], candidate[position]);
    fprintf(report, "  ],\n  \"gate\": \"%s\"\n}\n",
        changed_weights || changed_packed || changed_outputs ? "FAIL" : "PASS");
    if (fclose(report) != 0) die("expert group report close");
    if (changed_weights || changed_packed || changed_outputs) die("expert group same-result check failed");
    printf("expert group replacement: layer 1 expert %d gate row 0 group 0, %d position(s), PASS\n",
           expert, positions);
}