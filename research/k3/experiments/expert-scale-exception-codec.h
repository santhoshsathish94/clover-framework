#include <gmp.h>

static unsigned long long exception_control_checks, exception_invalid_checks;
static unsigned long long exception_rows, fixed_offset_rows;

static int exception_enabled(void)
{
    static int enabled = -1;
    if (enabled < 0) {
        const char *setting = getenv("K3_SCALE_EXCEPTIONS");
        enabled = setting ? atoi(setting) : 0;
        if (enabled < 0 || enabled > 1) die("K3_SCALE_EXCEPTIONS must be 0 or 1");
    }
    return enabled;
}

static unsigned exception_bits(mpz_srcptr value)
{
    return mpz_sgn(value) ? (unsigned)mpz_sizeinbase(value, 2) : 0;
}

static unsigned exception_count_bits(unsigned count)
{
    unsigned bits = 0;
    for (; count; count >>= 1) bits++;
    return bits;
}

static void exception_put(mpz_t payload, unsigned *offset, mpz_srcptr value, unsigned bits)
{
    if (mpz_sgn(value) < 0 || exception_bits(value) > bits || *offset + bits > 1024)
        die("exception field exceeds format");
    mpz_t shifted;
    mpz_init(shifted);
    mpz_mul_2exp(shifted, value, *offset);
    mpz_ior(payload, payload, shifted);
    mpz_clear(shifted);
    *offset += bits;
}

static void exception_put_small(mpz_t payload, unsigned *offset, unsigned value, unsigned bits)
{
    mpz_t field;
    mpz_init_set_ui(field, value);
    exception_put(payload, offset, field, bits);
    mpz_clear(field);
}

static int exception_take(mpz_srcptr payload, unsigned *offset, unsigned limit, mpz_t value, unsigned bits)
{
    if (*offset + bits > limit) return 0;
    mpz_fdiv_q_2exp(value, payload, *offset);
    mpz_fdiv_r_2exp(value, value, bits);
    *offset += bits;
    return 1;
}

static int exception_take_small(mpz_srcptr payload, unsigned *offset, unsigned limit,
                                unsigned *value, unsigned bits)
{
    mpz_t field;
    mpz_init(field);
    const int valid = exception_take(payload, offset, limit, field, bits);
    if (valid) *value = (unsigned)mpz_get_ui(field);
    mpz_clear(field);
    return valid;
}

static int exception_encode(const unsigned char *scales, unsigned count, unsigned format,
                            unsigned char bytes[128], unsigned *size, unsigned *bit_count,
                            unsigned *chosen)
{
    if (!count || count > 112 || format > 2) return 0;
    unsigned base = 255, maximum = 0;
    for (unsigned index = 0; index < count; index++) {
        if (scales[index] < base) base = scales[index];
        if (scales[index] > maximum) maximum = scales[index];
    }
    const unsigned width = exception_count_bits(maximum - base);
    unsigned frequencies[256] = {0}, positions[112], usual = 0, exceptions = 0;
    for (unsigned index = 0; index < count; index++) frequencies[scales[index] - base]++;
    for (unsigned value = 1; value < (1U << width); value++)
        if (frequencies[value] > frequencies[usual]) usual = value;
    for (unsigned index = 0; index < count; index++)
        if (scales[index] - base != usual) positions[exceptions++] = index;
    const unsigned radix = (1U << width) - 1;
    mpz_t position_rank, value_rank, possibilities, term, payload;
    mpz_inits(position_rank, value_rank, possibilities, term, payload, NULL);
    for (unsigned ordinal = 1; ordinal <= exceptions; ordinal++) {
        mpz_bin_uiui(term, positions[ordinal - 1], ordinal);
        mpz_add(position_rank, position_rank, term);
    }
    for (unsigned ordinal = exceptions; ordinal > 0; ordinal--) {
        const unsigned value = scales[positions[ordinal - 1]] - base;
        const unsigned digit = value < usual ? value : value - 1;
        mpz_mul_ui(value_rank, value_rank, radix);
        mpz_add_ui(value_rank, value_rank, digit);
    }
    mpz_bin_uiui(possibilities, count, exceptions);
    mpz_sub_ui(term, possibilities, 1);
    const unsigned position_bits = exception_bits(term);
    mpz_ui_pow_ui(possibilities, radix, exceptions);
    mpz_sub_ui(term, possibilities, 1);
    const unsigned value_bits = exception_bits(term);
    const unsigned fixed_bits = 12 + count * width;
    const unsigned encoded_bits = 12 + width + exception_count_bits(count) + position_bits + value_bits;
    const unsigned use_exceptions = format == 1 || (format == 2 && encoded_bits < fixed_bits);
    unsigned offset = 0;
    if (format == 2) exception_put_small(payload, &offset, use_exceptions, 1);
    exception_put_small(payload, &offset, base, 8);
    exception_put_small(payload, &offset, width, 4);
    if (use_exceptions) {
        exception_put_small(payload, &offset, usual, width);
        exception_put_small(payload, &offset, exceptions, exception_count_bits(count));
        exception_put(payload, &offset, position_rank, position_bits);
        exception_put(payload, &offset, value_rank, value_bits);
    } else {
        for (unsigned index = 0; index < count; index++)
            exception_put_small(payload, &offset, scales[index] - base, width);
    }
    memset(bytes, 0, 128);
    size_t written = 0;
    mpz_export(bytes, &written, -1, 1, 0, 0, payload);
    *size = (offset + 7) / 8;
    *bit_count = offset;
    *chosen = use_exceptions;
    if (written > *size || *size > 128) die("exception serialized size inconsistent");
    mpz_clears(position_rank, value_rank, possibilities, term, payload, NULL);
    return 1;
}

static int exception_decode(const unsigned char *bytes, unsigned size, unsigned count,
                            unsigned format, unsigned char *scales)
{
    if (!count || count > 112 || size < 2 || size > 128 || format > 2) return 0;
    mpz_t payload, position_rank, value_rank, possibilities, term;
    mpz_inits(payload, position_rank, value_rank, possibilities, term, NULL);
    mpz_import(payload, size, -1, 1, 0, 0, bytes);
    unsigned offset = 0, use_exceptions = format == 1, base, width, usual, exceptions;
    unsigned values[112], positions[112];
    int valid = 0;
    if (format == 2 && !exception_take_small(payload, &offset, size * 8, &use_exceptions, 1)) goto finish;
    if (!exception_take_small(payload, &offset, size * 8, &base, 8) ||
        !exception_take_small(payload, &offset, size * 8, &width, 4) || width > 8) goto finish;
    if (!use_exceptions) {
        for (unsigned index = 0; index < count; index++)
            if (!exception_take_small(payload, &offset, size * 8, &values[index], width)) goto finish;
    } else {
        if (!exception_take_small(payload, &offset, size * 8, &usual, width) ||
            !exception_take_small(payload, &offset, size * 8, &exceptions, exception_count_bits(count)) ||
            exceptions > count || (width == 0 && exceptions != 0)) goto finish;
        mpz_bin_uiui(possibilities, count, exceptions);
        mpz_sub_ui(term, possibilities, 1);
        if (!exception_take(payload, &offset, size * 8, position_rank, exception_bits(term)) ||
            mpz_cmp(position_rank, possibilities) >= 0) goto finish;
        int upper = (int)count - 1;
        for (unsigned ordinal = exceptions; ordinal > 0; ordinal--) {
            int position = upper;
            for (; position >= 0; position--) {
                mpz_bin_uiui(term, (unsigned)position, ordinal);
                if (mpz_cmp(term, position_rank) <= 0) break;
            }
            if (position < 0) goto finish;
            positions[ordinal - 1] = (unsigned)position;
            mpz_sub(position_rank, position_rank, term);
            upper = position - 1;
        }
        if (mpz_sgn(position_rank)) goto finish;
        const unsigned radix = (1U << width) - 1;
        mpz_ui_pow_ui(possibilities, radix, exceptions);
        mpz_sub_ui(term, possibilities, 1);
        if (!exception_take(payload, &offset, size * 8, value_rank, exception_bits(term)) ||
            mpz_cmp(value_rank, possibilities) >= 0) goto finish;
        for (unsigned index = 0; index < count; index++) values[index] = usual;
        for (unsigned ordinal = 0; ordinal < exceptions; ordinal++) {
            const unsigned digit = (unsigned)mpz_fdiv_q_ui(value_rank, value_rank, radix);
            values[positions[ordinal]] = digit < usual ? digit : digit + 1;
        }
        if (mpz_sgn(value_rank)) goto finish;
    }
    if (size != (offset + 7) / 8) goto finish;
    mpz_fdiv_q_2exp(term, payload, offset);
    if (mpz_sgn(term)) goto finish;
    for (unsigned index = 0; index < count; index++) {
        if (base + values[index] > 255) goto finish;
        scales[index] = (unsigned char)(base + values[index]);
    }
    valid = 1;
finish:
    mpz_clears(payload, position_rank, value_rank, possibilities, term, NULL);
    return valid;
}

static void exception_control(const unsigned char *values, unsigned count)
{
    for (unsigned format = 0; format < 3; format++) {
        unsigned char payload[128], decoded[112];
        unsigned size, bits, chosen;
        if (!exception_encode(values, count, format, payload, &size, &bits, &chosen) ||
            !exception_decode(payload, size, count, format, decoded) || memcmp(values, decoded, count))
            die("exception codec roundtrip failed");
        exception_control_checks++;
    }
}

static void exception_selfcheck(void)
{
    for (unsigned mask = 0; mask < 256; mask++) {
        unsigned char values[8];
        for (unsigned position = 0; position < 8; position++) values[position] = 120 + ((mask >> position) & 1);
        exception_control(values, 8);
    }
    for (unsigned width = 0; width <= 8; width++) {
        const unsigned char values[4] = {0, (1U << width) - 1, 0, (1U << width) - 1};
        exception_control(values, 4);
    }
    const unsigned constants[3] = {0, 121, 255};
    for (unsigned index = 0; index < 3; index++) {
        unsigned char values[112];
        memset(values, constants[index], sizeof(values));
        exception_control(values, 112);
    }
    unsigned char wide[112];
    for (unsigned index = 0; index < 112; index++) wide[index] = index == 111 ? 255 : index;
    exception_control(wide, 112);
    unsigned char invalid[128] = {0}, decoded[112];
    if (exception_decode(invalid, 1, 112, 2, decoded)) die("short exception header accepted");
    exception_invalid_checks++;
    if (exception_decode(invalid, 2, 0, 2, decoded)) die("zero exception row length accepted");
    exception_invalid_checks++;
    invalid[1] = 18;
    if (exception_decode(invalid, 2, 112, 2, decoded)) die("invalid exception width accepted");
    exception_invalid_checks++;
    invalid[1] = 128;
    if (exception_decode(invalid, 2, 112, 2, decoded)) die("exception padding accepted");
    exception_invalid_checks++;
    invalid[0] = 0; invalid[1] = 0;
    if (exception_decode(invalid, 2, 113, 2, decoded)) die("oversized exception row accepted");
    exception_invalid_checks++;
}

static void exception_prepare(const unsigned char *original, const unsigned char *codes, unsigned groups,
                              unsigned char restored[112], FILE *trace, int layer, int expert,
                              int part, int row, int rows, int positions)
{
    if (!scale_rows_checked) exception_selfcheck();
    unsigned char payload[128];
    unsigned size, bits, chosen;
    if (!exception_encode(original, groups, 2, payload, &size, &bits, &chosen) ||
        !exception_decode(payload, size, groups, 2, restored) || memcmp(original, restored, groups))
        die("model exception scale row differs");
    for (unsigned coordinate = 0; coordinate < groups * 32; coordinate++) {
        const unsigned code = (codes[coordinate / 2] >> (4 * (coordinate % 2))) & 15;
        const float weight = expert_group_weight(code, restored[coordinate / 32]);
        if (memcmp(&weight, &DQ[original[coordinate / 32]][code], sizeof(weight)))
            die("model exception weight differs");
    }
    fprintf(trace,
        "{\"type\":\"scale_row\",\"layer\":%d,\"expert\":%d,\"part\":%d,\"row\":%d,"
        "\"rows\":%d,\"width\":%u,\"positions\":%d,\"groups\":%u,\"scale_changes\":0,"
        "\"weights_checked\":%u,\"weight_changes\":0,\"scale_format\":\"adaptive-exceptions\","
        "\"uses_exceptions\":%s,\"payload_bits\":%u,\"original_scales\":[",
        layer, expert, part, row, rows, groups * 32, positions, groups, groups * 32,
        chosen ? "true" : "false", bits);
    for (unsigned group = 0; group < groups; group++) fprintf(trace, "%s%u", group ? "," : "", original[group]);
    fprintf(trace, "],\"payload\":[");
    for (unsigned byte = 0; byte < size; byte++) fprintf(trace, "%s%u", byte ? "," : "", payload[byte]);
    fprintf(trace, "]}\n");
    scale_rows_checked++;
    scale_values_checked += groups;
    scale_weights_checked += groups * 32;
    scale_payload_bytes += size;
    if (chosen) exception_rows++; else fixed_offset_rows++;
}

static void exception_finish(FILE *trace)
{
    if (!exception_enabled()) return;
    if (exception_rows + fixed_offset_rows != scale_rows_checked) die("exception row count mismatch");
    fprintf(trace,
        "{\"type\":\"exception_summary\",\"exception_rows\":%llu,\"fixed_rows\":%llu,"
        "\"control_checks\":%llu,\"invalid_inputs_rejected\":%llu,\"gate\":\"PASS\"}\n",
        exception_rows, fixed_offset_rows, exception_control_checks, exception_invalid_checks);
}