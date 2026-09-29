typedef struct {
    unsigned count;
    uint64_t rank;
} ExpertMaskRank;

static uint64_t expert_mask_choose(unsigned total, unsigned count)
{
    if (count > total) return 0;
    if (count > total - count) count = total - count;
    uint64_t value = 1;
    for (unsigned factor = 1; factor <= count; factor++)
        value = value * (total - count + factor) / factor;
    return value;
}

static ExpertMaskRank expert_mask_encode(uint32_t mask, unsigned width)
{
    ExpertMaskRank encoded = {0, 0};
    for (unsigned position = 0; position < width; position++) {
        if (!(mask & (UINT32_C(1) << position))) continue;
        encoded.count++;
        encoded.rank += expert_mask_choose(position, encoded.count);
    }
    return encoded;
}

static int expert_mask_decode(ExpertMaskRank encoded, unsigned width, uint32_t *output)
{
    if (width > 32 || encoded.count > width ||
        encoded.rank >= expert_mask_choose(width, encoded.count)) return 0;
    uint32_t mask = 0;
    int upper = (int)width - 1;
    for (unsigned ordinal = encoded.count; ordinal > 0; ordinal--) {
        int position = upper;
        while (position >= 0 && expert_mask_choose((unsigned)position, ordinal) > encoded.rank)
            position--;
        if (position < 0) return 0;
        mask |= UINT32_C(1) << (unsigned)position;
        encoded.rank -= expert_mask_choose((unsigned)position, ordinal);
        upper = position - 1;
    }
    if (encoded.rank != 0) return 0;
    *output = mask;
    return 1;
}

static unsigned long long expert_mask_roundtrip_checks;
static unsigned long long expert_mask_invalid_checks;

static void expert_mask_check_value(uint32_t value, unsigned width)
{
    const ExpertMaskRank encoded = expert_mask_encode(value, width);
    uint32_t decoded;
    if (!expert_mask_decode(encoded, width, &decoded) || decoded != value)
        die("count/rank mask roundtrip failed");
    expert_mask_roundtrip_checks++;
}

static void expert_mask_selfcheck(void)
{
    for (uint32_t value = 0; value < 4096; value++) expert_mask_check_value(value, 12);
    expert_mask_check_value(0, 32);
    expert_mask_check_value(UINT32_MAX, 32);
    for (unsigned first = 0; first < 32; first++) {
        const uint32_t single = UINT32_C(1) << first;
        expert_mask_check_value(single, 32);
        expert_mask_check_value(~single, 32);
        for (unsigned second = first + 1; second < 32; second++) {
            const uint32_t pair = single | (UINT32_C(1) << second);
            expert_mask_check_value(pair, 32);
            expert_mask_check_value(~pair, 32);
        }
    }
    for (unsigned count = 0; count <= 32; count++) {
        const uint64_t limit = expert_mask_choose(32, count);
        const uint64_t ranks[3] = {0, limit / 2, limit - 1};
        for (unsigned sample = 0; sample < 3; sample++) {
            if ((sample > 0 && ranks[sample] == ranks[0]) ||
                (sample > 1 && ranks[sample] == ranks[1])) continue;
            const ExpertMaskRank encoded = {count, ranks[sample]};
            uint32_t decoded;
            if (!expert_mask_decode(encoded, 32, &decoded)) die("mask boundary decode failed");
            const ExpertMaskRank again = expert_mask_encode(decoded, 32);
            if (again.count != count || again.rank != ranks[sample]) die("mask boundary rank failed");
            expert_mask_roundtrip_checks++;
        }
        const ExpertMaskRank invalid_rank = {count, limit};
        uint32_t ignored;
        if (expert_mask_decode(invalid_rank, 32, &ignored)) die("invalid mask rank accepted");
        expert_mask_invalid_checks++;
    }
    const ExpertMaskRank invalid_count = {33, 0};
    uint32_t ignored;
    if (expert_mask_decode(invalid_count, 32, &ignored)) die("invalid mask count accepted");
    expert_mask_invalid_checks++;
}

static void expert_mask_replace_plane(uint32_t masks[4])
{
    const char *setting = getenv("K3_MASK_RANK");
    const int mode = setting ? atoi(setting) : 0;
    if (mode == 0) return;
    if (mode != 1) die("K3_MASK_RANK must be 0 or 1");
    const char *path = getenv("K3_MASK_RANK_REPORT");
    if (!path) die("K3_MASK_RANK_REPORT required");
    static int applied;
    if (applied) die("count/rank replacement called twice");
    expert_mask_selfcheck();

    const uint32_t original = masks[2];
    const ExpertMaskRank encoded = expert_mask_encode(original, 32);
    unsigned rank_bits = 0;
    for (uint64_t remaining = expert_mask_choose(32, encoded.count) - 1; remaining; remaining >>= 1)
        rank_bits++;
    const unsigned payload_bits = 6 + rank_bits;
    const unsigned payload_bytes = (payload_bits + 7) / 8;
    const uint64_t packed = (encoded.rank << 6) | encoded.count;
    unsigned char payload[5] = {0};
    if (payload_bytes > sizeof(payload)) die("mask rank payload exceeds format bound");
    for (unsigned byte = 0; byte < payload_bytes; byte++)
        payload[byte] = (unsigned char)(packed >> (byte * 8));
    uint64_t loaded = 0;
    for (unsigned byte = 0; byte < payload_bytes; byte++)
        loaded |= (uint64_t)payload[byte] << (byte * 8);
    const ExpertMaskRank from_payload = {(unsigned)(loaded & 63), loaded >> 6};
    masks[2] = 0;
    if (!expert_mask_decode(from_payload, 32, &masks[2])) die("selected mask decoding failed");
    const int matches = masks[2] == original;
    applied = 1;

    FILE *report = fopen(path, "wb");
    if (!report) die("mask rank report open");
    fprintf(report,
        "{\n  \"plane\": 2,\n  \"original_mask\": %u,\n  \"count\": %u,\n"
        "  \"rank\": %llu,\n  \"regenerated_mask\": %u,\n"
        "  \"decoder_roundtrip_checks\": %llu,\n  \"invalid_inputs_rejected\": %llu,\n"
        "  \"mask_bits_checked\": 32,\n  \"mask_bits_changed\": %u,\n"
        "  \"payload_bits\": %u,\n  \"payload_bytes\": [",
        original, encoded.count, (unsigned long long)encoded.rank, masks[2],
        expert_mask_roundtrip_checks, expert_mask_invalid_checks,
        (unsigned)__builtin_popcount(original ^ masks[2]), payload_bits);
    for (unsigned byte = 0; byte < payload_bytes; byte++)
        fprintf(report, "%s%u", byte ? ", " : "", payload[byte]);
    fprintf(report, "],\n  \"gate\": \"%s\"\n}\n", matches ? "PASS" : "FAIL");
    if (fclose(report) != 0) die("mask rank report close");
    if (!matches) die("selected mask differs after count/rank decoding");
}