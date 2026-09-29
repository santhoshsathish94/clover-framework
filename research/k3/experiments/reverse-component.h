static int reverse_component_encode(const float values[E], unsigned char **output, size_t *size)
{
    uint32_t words[E];
    long integers[E];
    int exponents[E], minimum = 0, maximum = 0;
    unsigned code_bits = 0;
    memcpy(words, values, sizeof(words));
    for (unsigned index = 0; index < E; index++) {
        if (((words[index] >> 23) & 255) == 255) return 0;
        reverse_dyadic_parts(words[index], &integers[index], &exponents[index]);
        if (!index || exponents[index] < minimum) minimum = exponents[index];
        if (!index || exponents[index] > maximum) maximum = exponents[index];
        const unsigned magnitude = integers[index] < 0 ? -integers[index] : integers[index];
        unsigned bits = 0;
        for (unsigned code = magnitude ? (magnitude - 1) / 2 : 0; code; code >>= 1) bits++;
        if (bits > code_bits) code_bits = bits;
    }
    unsigned exponent_bits = 0;
    for (unsigned range = maximum - minimum; range; range >>= 1) exponent_bits++;
    *size = 8 + (E * (2 + exponent_bits + code_bits) + 7) / 8;
    unsigned char *payload = calloc(1, *size);
    if (!payload) die("reverse component allocation failed");
    payload[0] = 'O'; payload[1] = '1';
    reverse_put16(payload + 2, E);
    reverse_put16(payload + 4, (unsigned)minimum & 65535);
    payload[6] = exponent_bits; payload[7] = code_bits;
    unsigned offset = 64;
    for (unsigned index = 0; index < E; index++) {
        const unsigned magnitude = integers[index] < 0 ? -integers[index] : integers[index];
        group_order_put(payload, &offset, words[index] >> 31, 1);
        group_order_put(payload, &offset, magnitude == 0, 1);
        group_order_put(payload, &offset, exponents[index] - minimum, exponent_bits);
        group_order_put(payload, &offset, magnitude ? (magnitude - 1) / 2 : 0, code_bits);
    }
    if ((offset + 7) / 8 != *size) die("reverse component encoded length differs");
    *output = payload;
    return 1;
}

static int reverse_component_decode(const unsigned char *payload, size_t size, float values[E])
{
    if (size < 8 || payload[0] != 'O' || payload[1] != '1' || reverse_get16(payload + 2) != E) return 0;
    const unsigned stored = reverse_get16(payload + 4);
    const int minimum = stored < 32768 ? (int)stored : (int)stored - 65536;
    const unsigned exponent_bits = payload[6], code_bits = payload[7];
    if (minimum < -149 || minimum > 127 || exponent_bits > 9 || code_bits > 23) return 0;
    const unsigned bits = 64 + E * (2 + exponent_bits + code_bits);
    if (size != (bits + 7) / 8 || ((bits % 8) && (payload[size - 1] >> (bits % 8)))) return 0;
    unsigned offset = 64;
    for (unsigned index = 0; index < E; index++) {
        const unsigned negative = group_order_get(payload, &offset, 1);
        const unsigned zero = group_order_get(payload, &offset, 1);
        const int exponent = minimum + (int)group_order_get(payload, &offset, exponent_bits);
        const unsigned code = group_order_get(payload, &offset, code_bits);
        if (exponent < -149 || exponent > 127 || (zero && (code || exponent))) return 0;
        const float magnitude = zero ? 0.0f : scalbnf((float)(2 * code + 1), exponent);
        if (!isfinite(magnitude) || (!zero && magnitude == 0)) return 0;
        values[index] = copysignf(magnitude, negative ? -1.0f : 1.0f);
    }
    return offset == bits;
}

static void reverse_component(float values[E], FILE *trace)
{
    static int controls_checked;
    if (!controls_checked) {
        uint32_t words[E];
        float controls[E], restored[E];
        for (unsigned index = 0; index < E; index++)
            words[index] = ((index % 2) << 31) | (((index / 2) % 255) << 23) | ((index * 7919U) & 0x7fffff);
        words[0] = 0; words[1] = 0x80000000;
        memcpy(controls, words, sizeof(words));
        unsigned char *payload;
        size_t size;
        if (!reverse_component_encode(controls, &payload, &size) ||
            !reverse_component_decode(payload, size, restored) || memcmp(restored, words, sizeof(words)))
            die("reverse component finite controls differ");
        if (reverse_component_decode(payload, size - 1, restored)) die("reverse component short payload accepted");
        payload[7] = 24;
        if (reverse_component_decode(payload, size, restored)) die("reverse component invalid width accepted");
        free(payload);
        words[0] = 0x7f800000;
        memcpy(controls, words, sizeof(words));
        if (reverse_component_encode(controls, &payload, &size)) die("reverse component nonfinite value accepted");
        controls_checked = 1;
    }
    unsigned char original[E * 4], *payload;
    memcpy(original, values, sizeof(original));
    size_t size;
    if (!reverse_component_encode(values, &payload, &size)) die("reverse component input unsupported");
    memset(values, 0, sizeof(original));
    if (!reverse_component_decode(payload, size, values) || memcmp(original, values, sizeof(original)))
        die("reverse component consumed values differ");
    reverse_vector_record(trace, 5, payload, (uint32_t)size);
    free(payload);
}