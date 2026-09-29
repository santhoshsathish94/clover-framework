static unsigned long long joint_decoder_checks, joint_invalid_checks;

static int joint_enabled(void)
{
    static int enabled = -1;
    if (enabled < 0) {
        const char *setting = getenv("K3_JOINT_RANK");
        enabled = setting ? atoi(setting) : 0;
        if (enabled < 0 || enabled > 1) die("K3_JOINT_RANK must be 0 or 1");
    }
    return enabled;
}

static GroupOrderInteger joint_multiply(GroupOrderInteger first, GroupOrderInteger second)
{
    GroupOrderInteger product;
    if (__builtin_mul_overflow(first, second, &product)) die("joint rank multiplication overflow");
    return product;
}

static GroupOrderInteger joint_add(GroupOrderInteger first, GroupOrderInteger second)
{
    GroupOrderInteger sum;
    if (__builtin_add_overflow(first, second, &sum)) die("joint rank addition overflow");
    return sum;
}

static GroupOrderInteger joint_power(unsigned base, unsigned exponent)
{
    GroupOrderInteger power = 1;
    for (unsigned factor = 0; factor < exponent; factor++) power = joint_multiply(power, base);
    return power;
}

static uint64_t joint_choose(unsigned total, unsigned count)
{
    if (count > total) return 0;
    if (count > total - count) count = total - count;
    uint64_t value = 1;
    for (unsigned factor = 1; factor <= count; factor++)
        value = value * (total - count + factor) / factor;
    return value;
}

static int joint_encode(const unsigned counts[16], unsigned total, unsigned alphabet,
                        GroupOrderInteger arrangement, GroupOrderInteger *encoded)
{
    if (total > 32 || alphabet < 2 || alphabet > 16) return 0;
    for (unsigned code = alphabet; code < 16; code++) if (counts[code]) return 0;
    GroupOrderInteger ways;
    if (!group_order_ways(counts, total, &ways) || arrangement >= ways) return 0;
    unsigned remaining = total;
    GroupOrderInteger multiplicity = 1, offset = 0;
    for (unsigned code = 0; code + 1 < alphabet; code++) {
        const unsigned free_symbols = alphabet - code - 1;
        for (unsigned alternative = 0; alternative < counts[code]; alternative++) {
            GroupOrderInteger block = joint_multiply(multiplicity, joint_choose(remaining, alternative));
            block = joint_multiply(block, joint_power(free_symbols, remaining - alternative));
            offset = joint_add(offset, block);
        }
        multiplicity = joint_multiply(multiplicity, joint_choose(remaining, counts[code]));
        remaining -= counts[code];
    }
    if (remaining != counts[alphabet - 1] || multiplicity != ways) die("joint histogram partition inconsistent");
    *encoded = joint_add(offset, arrangement);
    return 1;
}

static int joint_decode(GroupOrderInteger encoded, unsigned total, unsigned alphabet,
                        unsigned counts[16], GroupOrderInteger *arrangement)
{
    if (total > 32 || alphabet < 2 || alphabet > 16) return 0;
    if (!(total == 32 && alphabet == 16) && encoded >= joint_power(alphabet, total)) return 0;
    memset(counts, 0, 16 * sizeof(unsigned));
    unsigned remaining = total;
    GroupOrderInteger multiplicity = 1;
    for (unsigned code = 0; code + 1 < alphabet; code++) {
        const unsigned free_symbols = alphabet - code - 1;
        int found = 0;
        for (unsigned count = 0; count <= remaining; count++) {
            const GroupOrderInteger placed = joint_multiply(multiplicity, joint_choose(remaining, count));
            const GroupOrderInteger block = joint_multiply(placed, joint_power(free_symbols, remaining - count));
            if (encoded < block) {
                counts[code] = count;
                multiplicity = placed;
                remaining -= count;
                found = 1;
                break;
            }
            encoded -= block;
        }
        if (!found) return 0;
    }
    counts[alphabet - 1] = remaining;
    GroupOrderInteger ways;
    if (!group_order_ways(counts, total, &ways) || ways != multiplicity || encoded >= ways) return 0;
    *arrangement = encoded;
    return 1;
}

static void joint_check_sequence(const unsigned char *codes, unsigned total, unsigned alphabet)
{
    unsigned counts[16], restored_counts[16];
    GroupOrderInteger arrangement, encoded, restored_arrangement;
    unsigned char decoded[32];
    if (!group_order_encode(codes, total, counts, &arrangement) ||
        !joint_encode(counts, total, alphabet, arrangement, &encoded) ||
        !joint_decode(encoded, total, alphabet, restored_counts, &restored_arrangement) ||
        memcmp(counts, restored_counts, sizeof(counts)) || arrangement != restored_arrangement ||
        !group_order_decode(restored_counts, total, restored_arrangement, decoded) || memcmp(codes, decoded, total))
        die("joint sequence check failed");
    joint_decoder_checks++;
}

static void joint_selfcheck(void)
{
    unsigned char codes[32];
    for (unsigned sequence = 0; sequence < 81; sequence++) {
        unsigned value = sequence;
        for (unsigned position = 0; position < 4; position++) {
            codes[position] = value % 3;
            value /= 3;
        }
        joint_check_sequence(codes, 4, 3);
    }
    for (unsigned code = 0; code < 16; code++) {
        memset(codes, code, sizeof(codes));
        joint_check_sequence(codes, 32, 16);
    }
    unsigned counts[16];
    for (unsigned code = 0; code < 16; code++) counts[code] = 2;
    GroupOrderInteger ways;
    if (!group_order_ways(counts, 32, &ways)) die("joint balanced counts failed");
    const GroupOrderInteger ranks[3] = {0, ways / 2, ways - 1};
    for (unsigned sample = 0; sample < 3; sample++) {
        if (!group_order_decode(counts, 32, ranks[sample], codes)) die("joint balanced decode failed");
        joint_check_sequence(codes, 32, 16);
    }
    const GroupOrderInteger extremes[2] = {0, ~(GroupOrderInteger)0};
    for (unsigned sample = 0; sample < 2; sample++) {
        GroupOrderInteger arrangement, encoded;
        if (!joint_decode(extremes[sample], 32, 16, counts, &arrangement) ||
            !joint_encode(counts, 32, 16, arrangement, &encoded) || encoded != extremes[sample])
            die("joint full-width boundary failed");
        joint_decoder_checks++;
    }
    GroupOrderInteger ignored;
    if (joint_decode(81, 4, 3, counts, &ignored)) die("joint small-domain invalid rank accepted");
    joint_invalid_checks++;
    if (joint_decode(0, 33, 16, counts, &ignored)) die("joint excessive length accepted");
    joint_invalid_checks++;
    if (joint_decode(0, 32, 1, counts, &ignored)) die("joint invalid alphabet accepted");
    joint_invalid_checks++;
    memset(counts, 0, sizeof(counts));
    counts[0] = 32;
    if (joint_encode(counts, 32, 16, 1, &ignored)) die("joint invalid arrangement accepted");
    joint_invalid_checks++;
    counts[0] = 31;
    if (joint_encode(counts, 32, 16, 0, &ignored)) die("joint short histogram accepted");
    joint_invalid_checks++;
}

static void joint_store(unsigned char *payload, unsigned *offset, const unsigned counts[16],
                        GroupOrderInteger arrangement, unsigned arrangement_bits)
{
    if (joint_enabled()) {
        GroupOrderInteger encoded;
        if (!joint_encode(counts, 32, 16, arrangement, &encoded)) die("selected joint encode failed");
        group_order_put(payload, offset, encoded, 128);
    } else {
        for (unsigned code = 0; code < 16; code++) group_order_put(payload, offset, counts[code], 6);
        group_order_put(payload, offset, arrangement, arrangement_bits);
    }
}

static GroupOrderInteger joint_load(const unsigned char *payload, unsigned *offset, unsigned counts[16])
{
    if (joint_enabled()) {
        const GroupOrderInteger encoded = group_order_get(payload, offset, 128);
        GroupOrderInteger arrangement;
        if (!joint_decode(encoded, 32, 16, counts, &arrangement)) die("selected joint decode failed");
        return arrangement;
    }
    for (unsigned code = 0; code < 16; code++) counts[code] = (unsigned)group_order_get(payload, offset, 6);
    GroupOrderInteger ways;
    if (!group_order_ways(counts, 32, &ways)) die("legacy counts invalid");
    unsigned bits = 0;
    for (GroupOrderInteger remaining = ways - 1; remaining; remaining >>= 1) bits++;
    return group_order_get(payload, offset, bits);
}