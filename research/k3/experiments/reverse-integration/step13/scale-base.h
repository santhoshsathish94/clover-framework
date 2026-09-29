static unsigned long long scale_rows_checked, scale_values_checked, scale_weights_checked;
static unsigned long long scale_payload_bytes, scale_control_checks, scale_invalid_checks;

static int scale_row_enabled(void)
{
    static int enabled = -1;
    if (enabled < 0) {
        const char *setting = getenv("K3_SCALE_ROWS");
        enabled = setting ? atoi(setting) : 0;
        if (enabled < 0 || enabled > 1) die("K3_SCALE_ROWS must be 0 or 1");
    }
    return enabled;
}

static int scale_encode(const unsigned char *scales, unsigned count, unsigned char payload[114], unsigned *size)
{
    if (!count || count > 112) return 0;
    unsigned base = 255, maximum = 0;
    for (unsigned group = 0; group < count; group++) {
        if (scales[group] < base) base = scales[group];
        if (scales[group] > maximum) maximum = scales[group];
    }
    unsigned width = 0;
    for (unsigned range = maximum - base; range; range >>= 1) width++;
    memset(payload, 0, 114);
    payload[0] = (unsigned char)base;
    payload[1] = (unsigned char)width;
    for (unsigned group = 0; group < count; group++) {
        const unsigned delta = scales[group] - base;
        for (unsigned bit = 0; bit < width; bit++) {
            const unsigned position = 12 + group * width + bit;
            payload[position / 8] |= ((delta >> bit) & 1U) << (position % 8);
        }
    }
    *size = (12 + count * width + 7) / 8;
    return 1;
}

static int scale_decode(const unsigned char *payload, unsigned size, unsigned count, unsigned char *scales)
{
    if (!count || count > 112 || size < 2) return 0;
    const unsigned base = payload[0], width = payload[1] & 15;
    if (width > 8) return 0;
    const unsigned bits = 12 + count * width;
    if (size != (bits + 7) / 8) return 0;
    if ((bits % 8) && (payload[size - 1] >> (bits % 8))) return 0;
    for (unsigned group = 0; group < count; group++) {
        unsigned delta = 0;
        for (unsigned bit = 0; bit < width; bit++) {
            const unsigned position = 12 + group * width + bit;
            delta |= ((payload[position / 8] >> (position % 8)) & 1U) << bit;
        }
        if (base + delta > 255) return 0;
        scales[group] = (unsigned char)(base + delta);
    }
    return 1;
}

static void scale_control(const unsigned char *values, unsigned count)
{
    unsigned char payload[114], decoded[112];
    unsigned size;
    if (!scale_encode(values, count, payload, &size) ||
        !scale_decode(payload, size, count, decoded) || memcmp(values, decoded, count))
        die("scale row roundtrip failed");
    scale_control_checks++;
}

static void scale_selfcheck(void)
{
    for (unsigned width = 0; width <= 8; width++) {
        const unsigned char values[4] = {0, (1U << width) - 1, 0, (1U << width) - 1};
        scale_control(values, 4);
    }
    const unsigned constants[3] = {0, 121, 255};
    for (unsigned index = 0; index < 3; index++) {
        unsigned char values[112];
        memset(values, constants[index], sizeof(values));
        scale_control(values, 112);
    }
    unsigned char payload[114] = {0}, decoded[112];
    if (scale_decode(payload, 1, 112, decoded)) die("short scale descriptor accepted");
    scale_invalid_checks++;
    payload[1] = 9;
    if (scale_decode(payload, 2, 112, decoded)) die("invalid scale width accepted");
    scale_invalid_checks++;
    payload[1] = 16;
    if (scale_decode(payload, 2, 112, decoded)) die("nonzero scale padding accepted");
    scale_invalid_checks++;
    payload[0] = 255; payload[1] = 17;
    if (scale_decode(payload, 2, 4, decoded)) die("scale byte overflow accepted");
    scale_invalid_checks++;
    if (scale_decode(payload, 2, 0, decoded)) die("zero scale count accepted");
    scale_invalid_checks++;
    if (scale_decode(payload, 2, 113, decoded)) die("oversized scale row accepted");
    scale_invalid_checks++;
}

static void scale_prepare(const unsigned char *original, const unsigned char *codes, unsigned groups,
                          unsigned char restored[112], FILE *trace, int layer, int expert,
                          int part, int row, int rows, int positions)
{
    if (!scale_rows_checked) scale_selfcheck();
    unsigned char payload[114];
    unsigned size;
    if (!scale_encode(original, groups, payload, &size) ||
        !scale_decode(payload, size, groups, restored) || memcmp(original, restored, groups))
        die("model scale row differs");
    for (unsigned coordinate = 0; coordinate < groups * 32; coordinate++) {
        const unsigned code = (codes[coordinate / 2] >> (4 * (coordinate % 2))) & 15;
        const float weight = expert_group_weight(code, restored[coordinate / 32]);
        if (memcmp(&weight, &DQ[original[coordinate / 32]][code], sizeof(weight)))
            die("model scale row weight differs");
    }
    fprintf(trace,
        "{\"type\":\"scale_row\",\"layer\":%d,\"expert\":%d,\"part\":%d,\"row\":%d,"
        "\"rows\":%d,\"width\":%u,\"positions\":%d,\"groups\":%u,\"scale_changes\":0,"
        "\"weights_checked\":%u,\"weight_changes\":0,\"base\":%u,\"offset_width\":%u,"
        "\"payload_bits\":%u,\"original_scales\":[",
        layer, expert, part, row, rows, groups * 32, positions, groups,
        groups * 32, payload[0], payload[1] & 15, 12 + groups * (payload[1] & 15));
    for (unsigned group = 0; group < groups; group++) fprintf(trace, "%s%u", group ? "," : "", original[group]);
    fprintf(trace, "],\"payload\":[");
    for (unsigned byte = 0; byte < size; byte++) fprintf(trace, "%s%u", byte ? "," : "", payload[byte]);
    fprintf(trace, "]}\n");
    scale_rows_checked++;
    scale_values_checked += groups;
    scale_weights_checked += groups * 32;
    scale_payload_bytes += size;
}

static void scale_finish(FILE *trace, unsigned long long pairs)
{
    if (!scale_row_enabled()) return;
    if (scale_rows_checked != pairs * 9 || scale_values_checked != pairs * 960 ||
        scale_weights_checked != scale_values_checked * 32)
        die("scale row coverage incomplete");
    fprintf(trace,
        "{\"type\":\"scale_summary\",\"rows\":%llu,\"scales\":%llu,\"weights\":%llu,"
        "\"payload_bytes\":%llu,\"control_checks\":%llu,\"invalid_inputs_rejected\":%llu,\"gate\":\"PASS\"}\n",
        scale_rows_checked, scale_values_checked, scale_weights_checked,
        scale_payload_bytes, scale_control_checks, scale_invalid_checks);
}