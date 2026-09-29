static void reverse_put16(unsigned char *bytes, unsigned value)
{
    bytes[0] = value;
    bytes[1] = value >> 8;
}

static unsigned reverse_get16(const unsigned char *bytes)
{
    return bytes[0] | ((unsigned)bytes[1] << 8);
}

static void reverse_dyadic_parts(uint32_t word, long *integer, int *exponent)
{
    const unsigned stored_exponent = (word >> 23) & 255;
    unsigned significand = word & 0x7fffff;
    *exponent = -149;
    if (stored_exponent) {
        significand |= 1U << 23;
        *exponent = (int)stored_exponent - 150;
    }
    if (!significand) {
        *integer = 0;
        *exponent = 0;
        return;
    }
    const unsigned shift = __builtin_ctz(significand);
    *exponent += shift;
    *integer = significand >> shift;
    if (word >> 31) *integer = -*integer;
}

static unsigned char *reverse_dyadic_codec(float values[E], unsigned block_size, size_t *size)
{
    if (block_size != 32 && block_size != 128) die("reverse dyadic block size invalid");
    const size_t capacity = 8 + (E / block_size) * 4 + E * 40;
    unsigned char *payload = calloc(1, capacity);
    if (!payload) die("reverse dyadic allocation failed");
    memcpy(payload, "DYA1", 4);
    reverse_put16(payload + 4, E);
    reverse_put16(payload + 6, block_size);
    size_t offset = 8;
    mpz_t number, piece, packed;
    mpz_inits(number, piece, packed, NULL);
    for (unsigned start = 0; start < E; start += block_size) {
        uint32_t words[128];
        memcpy(words, values + start, block_size * 4);
        int raw = 0, minimum = 0, nonzero = 0;
        long integers[128];
        int exponents[128];
        for (unsigned index = 0; index < block_size; index++) {
            if (words[index] == 0x80000000 || ((words[index] >> 23) & 255) == 255) raw = 1;
            reverse_dyadic_parts(words[index], &integers[index], &exponents[index]);
            if (integers[index] && (!nonzero || exponents[index] < minimum)) minimum = exponents[index];
            nonzero |= integers[index] != 0;
        }
        if (raw) {
            if (offset + 4 + block_size * 4 > capacity) die("reverse dyadic raw overflow");
            offset += 4;
            memcpy(payload + offset, words, block_size * 4);
            offset += block_size * 4;
            continue;
        }
        unsigned bits = 1;
        for (unsigned index = 0; index < block_size; index++) {
            mpz_set_si(number, integers[index]);
            if (integers[index]) mpz_mul_2exp(number, number, exponents[index] - minimum);
            if (mpz_sgn(number) < 0) mpz_com(piece, number); else mpz_set(piece, number);
            const unsigned width = (mpz_sgn(piece) ? mpz_sizeinbase(piece, 2) : 0) + 1;
            if (width > bits) bits = width;
        }
        const size_t bytes = (block_size * bits + 7) / 8;
        if (bits > 320 || offset + 4 + bytes > capacity) die("reverse dyadic payload bound");
        reverse_put16(payload + offset, (unsigned)minimum & 65535);
        reverse_put16(payload + offset + 2, bits);
        offset += 4;
        mpz_set_ui(packed, 0);
        for (unsigned index = 0; index < block_size; index++) {
            mpz_set_si(number, integers[index]);
            if (integers[index]) mpz_mul_2exp(number, number, exponents[index] - minimum);
            mpz_fdiv_r_2exp(piece, number, bits);
            mpz_mul_2exp(piece, piece, index * bits);
            mpz_ior(packed, packed, piece);
        }
        size_t written = 0;
        mpz_export(payload + offset, &written, -1, 1, 0, 0, packed);
        if (written > bytes) die("reverse dyadic integer export overflow");
        offset += bytes;
    }
    *size = offset;
    memset(values, 0, E * 4);
    offset = 8;
    for (unsigned start = 0; start < E; start += block_size) {
        if (offset + 4 > *size) die("reverse dyadic short block");
        const unsigned stored = reverse_get16(payload + offset);
        const int exponent = stored < 32768 ? (int)stored : (int)stored - 65536;
        const unsigned bits = reverse_get16(payload + offset + 2);
        offset += 4;
        const size_t bytes = bits ? (block_size * bits + 7) / 8 : block_size * 4;
        if (bits > 320 || offset + bytes > *size) die("reverse dyadic invalid block");
        if (!bits) memcpy(values + start, payload + offset, bytes);
        else {
            mpz_import(packed, bytes, -1, 1, 0, 0, payload + offset);
            for (unsigned index = 0; index < block_size; index++) {
                mpz_fdiv_q_2exp(number, packed, index * bits);
                mpz_fdiv_r_2exp(number, number, bits);
                if (mpz_tstbit(number, bits - 1)) {
                    mpz_set_ui(piece, 1);
                    mpz_mul_2exp(piece, piece, bits);
                    mpz_sub(number, number, piece);
                }
                values[start + index] = (float)ldexp(mpz_get_d(number), exponent);
            }
        }
        offset += bytes;
    }
    mpz_clears(number, piece, packed, NULL);
    if (offset != *size) die("reverse dyadic trailing data");
    return payload;
}

static void reverse_dyadic(float values[E], unsigned block_size, FILE *trace, unsigned kind)
{
    static int controls_checked;
    if (!controls_checked) {
        const uint32_t patterns[2][8] = {
            {0, 1, 0x80000001, 0x7f7fffff, 0xff7fffff, 0x3f800000, 0xbf800000, 0x00800000},
            {0x80000000, 0x7f800000, 0xff800000, 0x7fc12345, 1, 0x80000001, 0, 0x3f800000}
        };
        for (unsigned mode = 0; mode < 2; mode++) {
            for (unsigned width = 32; width <= 128; width *= 4) {
                uint32_t expected[E];
                float restored[E];
                for (unsigned index = 0; index < E; index++) expected[index] = patterns[mode][index % 8];
                memcpy(restored, expected, sizeof(expected));
                size_t size;
                unsigned char *payload = reverse_dyadic_codec(restored, width, &size);
                if (memcmp(restored, expected, sizeof(expected))) die("reverse dyadic control differs");
                free(payload);
            }
        }
        controls_checked = 1;
    }
    unsigned char expected[E * 4];
    memcpy(expected, values, sizeof(expected));
    size_t size;
    unsigned char *payload = reverse_dyadic_codec(values, block_size, &size);
    if (memcmp(expected, values, sizeof(expected))) die("reverse dyadic consumed vector differs");
    reverse_vector_record(trace, kind, payload, (uint32_t)size);
    free(payload);
}