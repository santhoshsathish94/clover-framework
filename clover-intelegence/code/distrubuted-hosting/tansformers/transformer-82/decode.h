#ifndef TRANSFORMER_DECODE_H
#define TRANSFORMER_DECODE_H
#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef struct {
    const unsigned char *data;
    size_t size, position;
    unsigned bit;
    int failed;
} DecodeBits;

typedef struct {
    unsigned counts[16];
    unsigned symbols[288];
} DecodeTree;

static unsigned decode_bits(DecodeBits *stream, unsigned count)
{
    unsigned value = 0;
    for (unsigned offset = 0; offset < count; offset++) {
        if (stream->position == stream->size) { stream->failed = 1; return 0; }
        value |= ((stream->data[stream->position] >> stream->bit) & 1U) << offset;
        if (++stream->bit == 8) { stream->bit = 0; stream->position++; }
    }
    return value;
}

static int decode_tree(DecodeTree *tree, const unsigned *lengths, unsigned count, int code_lengths)
{
    unsigned offsets[16] = {0}, maximum = 0;
    int remaining = 1;
    memset(tree, 0, sizeof *tree);
    if (count > 288) return 0;
    for (unsigned symbol = 0; symbol < count; symbol++) {
        if (lengths[symbol] > 15) return 0;
        tree->counts[lengths[symbol]]++;
        if (lengths[symbol] > maximum) maximum = lengths[symbol];
    }
    if (!maximum) return !code_lengths;
    for (unsigned length = 1; length <= 15; length++) {
        remaining = remaining * 2 - (int)tree->counts[length];
        if (remaining < 0) return 0;
    }
    if (remaining && (code_lengths || maximum != 1)) return 0;
    for (unsigned length = 1; length < 15; length++) offsets[length + 1] = offsets[length] + tree->counts[length];
    for (unsigned symbol = 0; symbol < count; symbol++)
        if (lengths[symbol]) tree->symbols[offsets[lengths[symbol]]++] = symbol;
    return 1;
}

static int decode_symbol(DecodeBits *stream, const DecodeTree *tree)
{
    unsigned code = 0, first = 0, index = 0;
    for (unsigned length = 1; length <= 15; length++) {
        code |= decode_bits(stream, 1);
        if (stream->failed) return -1;
        unsigned count = tree->counts[length];
        if (code >= first && code - first < count) return (int)tree->symbols[index + code - first];
        index += count;
        first = (first + count) << 1;
        code <<= 1;
    }
    stream->failed = 1;
    return -1;
}

static int decode_tables(DecodeBits *stream, unsigned type, DecodeTree *literal, DecodeTree *distance)
{
    unsigned lengths[320] = {0};
    if (type == 1) {
        for (unsigned symbol = 0; symbol < 288; symbol++)
            lengths[symbol] = symbol < 144 ? 8 : symbol < 256 ? 9 : symbol < 280 ? 7 : 8;
        for (unsigned symbol = 288; symbol < 320; symbol++) lengths[symbol] = 5;
        return decode_tree(literal, lengths, 288, 0) && decode_tree(distance, lengths + 288, 32, 0);
    }
    static const unsigned order[19] = {16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15};
    unsigned literal_count = decode_bits(stream, 5) + 257;
    unsigned distance_count = decode_bits(stream, 5) + 1;
    unsigned length_count = decode_bits(stream, 4) + 4;
    if (literal_count > 286 || distance_count > 32 || stream->failed) return 0;
    unsigned codes[19] = {0};
    for (unsigned index = 0; index < length_count; index++) codes[order[index]] = decode_bits(stream, 3);
    DecodeTree code_tree;
    if (stream->failed || !decode_tree(&code_tree, codes, 19, 1)) return 0;
    unsigned count = literal_count + distance_count, index = 0;
    while (index < count) {
        int symbol = decode_symbol(stream, &code_tree);
        if (symbol < 0 || symbol > 18) return 0;
        if (symbol < 16) { lengths[index++] = (unsigned)symbol; continue; }
        unsigned value = 0, repeat;
        if (symbol == 16) {
            if (!index) return 0;
            value = lengths[index - 1];
            repeat = decode_bits(stream, 2) + 3;
        } else if (symbol == 17) repeat = decode_bits(stream, 3) + 3;
        else repeat = decode_bits(stream, 7) + 11;
        if (stream->failed || repeat > count - index) return 0;
        while (repeat--) lengths[index++] = value;
    }
    return lengths[256] && decode_tree(literal, lengths, literal_count, 0) &&
        decode_tree(distance, lengths + literal_count, distance_count, 0);
}

static uint32_t decode_adler(const unsigned char *data, size_t length)
{
    uint32_t first = 1, second = 0;
    for (size_t index = 0; index < length; index++) {
        first = (first + data[index]) % 65521U;
        second = (second + first) % 65521U;
    }
    return (second << 16) | first;
}

static int decode_zlib(const unsigned char *input, size_t input_bytes,
    unsigned char *output, size_t output_bytes)
{
    static const unsigned length_base[29] = {3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59,67,83,99,115,131,163,195,227,258};
    static const unsigned length_extra[29] = {0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0};
    static const unsigned distance_base[30] = {1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,257,385,513,769,1025,1537,2049,3073,4097,6145,8193,12289,16385,24577};
    static const unsigned distance_extra[30] = {0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13};
    if (!input || !output || input_bytes < 6 || (input[0] & 15) != 8 ||
        (input[0] >> 4) > 7 || (((unsigned)input[0] << 8) | input[1]) % 31 || (input[1] & 32)) return 0;
    DecodeBits stream = {input + 2, input_bytes - 6, 0, 0, 0};
    size_t produced = 0;
    unsigned final = 0, window = 1U << ((input[0] >> 4) + 8);
    while (!final) {
        final = decode_bits(&stream, 1);
        unsigned type = decode_bits(&stream, 2);
        if (stream.failed || type == 3) return 0;
        if (type == 0) {
            if (stream.bit) { stream.bit = 0; stream.position++; }
            unsigned length = decode_bits(&stream, 16), inverse = decode_bits(&stream, 16);
            if (stream.failed || (length ^ inverse) != 65535 || length > output_bytes - produced ||
                length > stream.size - stream.position) return 0;
            memcpy(output + produced, stream.data + stream.position, length);
            stream.position += length;
            produced += length;
            continue;
        }
        DecodeTree literal, distance;
        if (!decode_tables(&stream, type, &literal, &distance)) return 0;
        for (;;) {
            int symbol = decode_symbol(&stream, &literal);
            if (symbol < 0) return 0;
            if (symbol < 256) {
                if (produced == output_bytes) return 0;
                output[produced++] = (unsigned char)symbol;
            } else if (symbol == 256) break;
            else {
                if (symbol > 285) return 0;
                unsigned ordinal = (unsigned)symbol - 257;
                unsigned length = length_base[ordinal] + decode_bits(&stream, length_extra[ordinal]);
                int distance_symbol = decode_symbol(&stream, &distance);
                if (stream.failed || distance_symbol < 0 || distance_symbol >= 30) return 0;
                unsigned back = distance_base[distance_symbol] + decode_bits(&stream, distance_extra[distance_symbol]);
                if (stream.failed || back > produced || back > window || length > output_bytes - produced) return 0;
                for (unsigned offset = 0; offset < length; offset++) { output[produced] = output[produced - back]; produced++; }
            }
        }
    }
    size_t consumed = stream.position + (stream.bit != 0);
    const unsigned char *checksum = input + input_bytes - 4;
    uint32_t expected = ((uint32_t)checksum[0] << 24) | ((uint32_t)checksum[1] << 16) |
        ((uint32_t)checksum[2] << 8) | checksum[3];
    return !stream.failed && consumed == stream.size && produced == output_bytes &&
        decode_adler(output, output_bytes) == expected;
}
#endif