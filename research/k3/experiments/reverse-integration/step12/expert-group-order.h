typedef unsigned __int128 GroupOrderInteger;

static unsigned long long group_order_roundtrips;
static unsigned long long group_order_invalid;

static GroupOrderInteger group_order_factorial(unsigned count)
{
    GroupOrderInteger result = 1;
    for (unsigned factor = 2; factor <= count; factor++) result *= factor;
    return result;
}

static int group_order_ways(const unsigned counts[16], unsigned length, GroupOrderInteger *ways)
{
    if (length > 32) return 0;
    unsigned total = 0;
    GroupOrderInteger denominator = 1;
    for (unsigned code = 0; code < 16; code++) {
        if (counts[code] > length) return 0;
        total += counts[code];
        if (total > length) return 0;
        denominator *= group_order_factorial(counts[code]);
    }
    if (total != length) return 0;
    *ways = group_order_factorial(length) / denominator;
    return 1;
}

static int group_order_encode(const unsigned char *codes, unsigned length,
                               unsigned counts[16], GroupOrderInteger *rank)
{
    if (length > 32) return 0;
    memset(counts, 0, 16 * sizeof(unsigned));
    for (unsigned position = 0; position < length; position++) {
        if (codes[position] >= 16) return 0;
        counts[codes[position]]++;
    }
    unsigned remaining[16];
    memcpy(remaining, counts, sizeof(remaining));
    *rank = 0;
    for (unsigned position = 0; position < length; position++) {
        for (unsigned code = 0; code < codes[position]; code++) {
            if (!remaining[code]) continue;
            remaining[code]--;
            GroupOrderInteger ways;
            if (!group_order_ways(remaining, length - position - 1, &ways)) return 0;
            *rank += ways;
            remaining[code]++;
        }
        remaining[codes[position]]--;
    }
    return 1;
}

static int group_order_decode(const unsigned counts[16], unsigned length,
                               GroupOrderInteger rank, unsigned char *codes)
{
    GroupOrderInteger total;
    if (!group_order_ways(counts, length, &total) || rank >= total) return 0;
    unsigned remaining[16];
    memcpy(remaining, counts, sizeof(remaining));
    for (unsigned position = 0; position < length; position++) {
        int found = 0;
        for (unsigned code = 0; code < 16; code++) {
            if (!remaining[code]) continue;
            remaining[code]--;
            GroupOrderInteger ways;
            if (!group_order_ways(remaining, length - position - 1, &ways)) return 0;
            if (rank < ways) {
                codes[position] = (unsigned char)code;
                found = 1;
                break;
            }
            rank -= ways;
            remaining[code]++;
        }
        if (!found) return 0;
    }
    return rank == 0;
}

static void group_order_check(const unsigned char *codes, unsigned length)
{
    unsigned counts[16];
    GroupOrderInteger rank;
    unsigned char decoded[32];
    if (!group_order_encode(codes, length, counts, &rank) ||
        !group_order_decode(counts, length, rank, decoded) || memcmp(codes, decoded, length))
        die("group arrangement roundtrip failed");
    group_order_roundtrips++;
}

static void group_order_selfcheck(void)
{
    unsigned char codes[32];
    for (unsigned sequence = 0; sequence < 4096; sequence++) {
        for (unsigned position = 0; position < 6; position++)
            codes[position] = (sequence >> (position * 2)) & 3;
        group_order_check(codes, 6);
    }
    for (unsigned first = 0; first < 16; first++) {
        memset(codes, first, sizeof(codes));
        group_order_check(codes, 32);
        for (unsigned second = 0; second < 16; second++) {
            for (unsigned position = 0; position < 32; position++)
                codes[position] = (position & 1) ? second : first;
            group_order_check(codes, 32);
        }
    }
    for (unsigned reverse = 0; reverse < 2; reverse++) {
        for (unsigned position = 0; position < 32; position++)
            codes[position] = reverse ? 15 - position / 2 : position / 2;
        group_order_check(codes, 32);
    }
    unsigned counts[16];
    for (unsigned code = 0; code < 16; code++) counts[code] = 2;
    GroupOrderInteger ways;
    if (!group_order_ways(counts, 32, &ways)) die("balanced group counts invalid");
    const GroupOrderInteger ranks[3] = {0, ways / 2, ways - 1};
    for (unsigned sample = 0; sample < 3; sample++) {
        if (!group_order_decode(counts, 32, ranks[sample], codes)) die("group rank boundary decode");
        unsigned again_counts[16];
        GroupOrderInteger again_rank;
        if (!group_order_encode(codes, 32, again_counts, &again_rank) ||
            memcmp(counts, again_counts, sizeof(counts)) || again_rank != ranks[sample])
            die("group rank boundary mismatch");
        group_order_roundtrips++;
    }
    if (group_order_decode(counts, 32, ways, codes)) die("out-of-range group rank accepted");
    group_order_invalid++;
    counts[0] = 3;
    if (group_order_decode(counts, 32, 0, codes)) die("excess group count accepted");
    group_order_invalid++;
    counts[0] = 1;
    if (group_order_decode(counts, 32, 0, codes)) die("short group count accepted");
    group_order_invalid++;
    counts[0] = 33;
    if (group_order_decode(counts, 32, 0, codes)) die("oversized group count accepted");
    group_order_invalid++;
    memset(codes, 0, sizeof(codes));
    codes[0] = 16;
    GroupOrderInteger ignored;
    if (group_order_encode(codes, 32, counts, &ignored)) die("invalid code accepted");
    group_order_invalid++;
}

static void group_order_decimal(GroupOrderInteger value, char result[40])
{
    char reverse[40];
    unsigned count = 0;
    do {
        reverse[count++] = (char)('0' + value % 10);
        value /= 10;
    } while (value);
    for (unsigned index = 0; index < count; index++) result[index] = reverse[count - 1 - index];
    result[count] = '\0';
}

static void group_order_put(unsigned char *payload, unsigned *offset,
                             GroupOrderInteger value, unsigned bits)
{
    for (unsigned bit = 0; bit < bits; bit++, (*offset)++)
        payload[*offset / 8] |= (unsigned char)(((value >> bit) & 1) << (*offset % 8));
}

static GroupOrderInteger group_order_get(const unsigned char *payload, unsigned *offset, unsigned bits)
{
    GroupOrderInteger value = 0;
    for (unsigned bit = 0; bit < bits; bit++, (*offset)++)
        value |= (GroupOrderInteger)((payload[*offset / 8] >> (*offset % 8)) & 1) << bit;
    return value;
}

static void group_order_replace(uint32_t masks[4], unsigned *scale)
{
    const char *setting = getenv("K3_GROUP_ORDER");
    const int mode = setting ? atoi(setting) : 0;
    if (!mode) return;
    if (mode != 1) die("K3_GROUP_ORDER must be 0 or 1");
    const char *path = getenv("K3_GROUP_ORDER_REPORT");
    if (!path) die("K3_GROUP_ORDER_REPORT required");
    static int called;
    if (called++) die("arrangement replacement called twice");
    group_order_selfcheck();

    unsigned char original[32], decoded[32];
    const unsigned original_scale = *scale;
    for (unsigned position = 0; position < 32; position++) {
        original[position] = 0;
        for (unsigned bit = 0; bit < 4; bit++)
            original[position] |= ((masks[bit] >> position) & 1U) << bit;
    }
    unsigned counts[16];
    GroupOrderInteger rank, ways;
    if (!group_order_encode(original, 32, counts, &rank) || !group_order_ways(counts, 32, &ways))
        die("selected group encode failed");
    unsigned rank_bits = 0;
    for (GroupOrderInteger remaining = ways - 1; remaining; remaining >>= 1) rank_bits++;
    unsigned char payload[29] = {0};
    unsigned offset = 0;
    group_order_put(payload, &offset, *scale, 8);
    for (unsigned code = 0; code < 16; code++) group_order_put(payload, &offset, counts[code], 6);
    group_order_put(payload, &offset, rank, rank_bits);
    const unsigned payload_bits = offset;
    const unsigned payload_bytes = (offset + 7) / 8;
    if (payload_bytes > sizeof(payload)) die("group order payload overflow");

    memset(masks, 0, 4 * sizeof(uint32_t));
    offset = 0;
    *scale = (unsigned)group_order_get(payload, &offset, 8);
    unsigned stored_counts[16];
    for (unsigned code = 0; code < 16; code++) stored_counts[code] = (unsigned)group_order_get(payload, &offset, 6);
    GroupOrderInteger stored_ways;
    if (!group_order_ways(stored_counts, 32, &stored_ways)) die("serialized group counts invalid");
    unsigned stored_rank_bits = 0;
    for (GroupOrderInteger remaining = stored_ways - 1; remaining; remaining >>= 1) stored_rank_bits++;
    const GroupOrderInteger stored_rank = group_order_get(payload, &offset, stored_rank_bits);
    if (offset != payload_bits || !group_order_decode(stored_counts, 32, stored_rank, decoded))
        die("selected group arrangement decode failed");
    unsigned changed = 0;
    for (unsigned position = 0; position < 32; position++) {
        changed += original[position] != decoded[position];
        for (unsigned bit = 0; bit < 4; bit++)
            masks[bit] |= ((decoded[position] >> bit) & 1U) << position;
    }
    char rank_text[40], ways_text[40];
    group_order_decimal(stored_rank, rank_text);
    group_order_decimal(stored_ways, ways_text);
    const int passed = changed == 0 && *scale == original_scale &&
        stored_rank == rank && memcmp(counts, stored_counts, sizeof(counts)) == 0;
    FILE *report = fopen(path, "wb");
    if (!report) die("group arrangement report open");
    fprintf(report,
        "{\n  \"width\": 32,\n  \"alphabet\": 16,\n  \"scale_byte\": %u,\n"
        "  \"rank\": \"%s\",\n  \"arrangements\": \"%s\",\n"
        "  \"rank_bits\": %u,\n  \"count_bits\": 96,\n  \"payload_bits\": %u,\n"
        "  \"decoder_checks\": %llu,\n  \"invalid_inputs_rejected\": %llu,\n"
        "  \"codes_checked\": 32,\n  \"codes_changed\": %u,\n  \"counts\": [",
        *scale, rank_text, ways_text, rank_bits, payload_bits,
        group_order_roundtrips, group_order_invalid, changed);
    for (unsigned code = 0; code < 16; code++) fprintf(report, "%s%u", code ? ", " : "", stored_counts[code]);
    fprintf(report, "],\n  \"decoded_codes\": [");
    for (unsigned position = 0; position < 32; position++) fprintf(report, "%s%u", position ? ", " : "", decoded[position]);
    fprintf(report, "],\n  \"payload_bytes\": [");
    for (unsigned byte = 0; byte < payload_bytes; byte++) fprintf(report, "%s%u", byte ? ", " : "", payload[byte]);
    fprintf(report, "],\n  \"gate\": \"%s\"\n}\n", passed ? "PASS" : "FAIL");
    if (fclose(report) != 0) die("group arrangement report close");
    if (!passed) die("selected group arrangement differs");
}