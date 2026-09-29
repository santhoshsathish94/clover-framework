#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>
#include "../expert-constant-palette.h"
#include "expert-map-constants.h"

enum { MATRIX_COUNT = 2688, BLOCK_COUNT = 136192, ROWS_PER_BLOCK = 64,
       INDEX_BYTES = MATRIX_COUNT * 2 + BLOCK_COUNT * 8, PAYLOAD_START = 32 + INDEX_BYTES };
typedef struct {
    FILE *file;
    unsigned char *index;
    uint64_t *offsets;
} MapStore;

static void fail(const char *message) {
    fprintf(stderr, "map store: %s\n", message);
    exit(1);
}

static uint16_t read16(const unsigned char *data) {
    return (uint16_t)((uint16_t)data[0] | (uint16_t)data[1] << 8);
}

static uint32_t read32(const unsigned char *data) {
    return (uint32_t)data[0] | (uint32_t)data[1] << 8 | (uint32_t)data[2] << 16 | (uint32_t)data[3] << 24;
}

static uint64_t read64(const unsigned char *data) {
    return read32(data) | (uint64_t)read32(data + 4) << 32;
}

static void read_at(FILE *file, uint64_t offset, unsigned char *data, size_t size) {
    if (offset > INT64_MAX || fseeko(file, (off_t)offset, SEEK_SET) || fread(data, 1, size, file) != size)
        fail("invalid offset or short file");
}

static size_t serialize_constants(unsigned char *output) {
    size_t cursor = 0;
    for (unsigned index = 0; index < 66; index++) {
        output[cursor++] = (unsigned char)k3_l1_value_bits[index];
        output[cursor++] = (unsigned char)(k3_l1_value_bits[index] >> 8);
    }
    memcpy(output + cursor, k3_group_maps, sizeof(k3_group_maps));
    cursor += sizeof(k3_group_maps);
    for (unsigned index = 0; index <= K3_TEMPLATE_COUNT; index++) {
        output[cursor++] = (unsigned char)k3_template_offsets[index];
        output[cursor++] = (unsigned char)(k3_template_offsets[index] >> 8);
    }
    for (unsigned index = 0; index < K3_TEMPLATE_REFS; index++) {
        output[cursor++] = (unsigned char)k3_template_maps[index];
        output[cursor++] = (unsigned char)(k3_template_maps[index] >> 8);
    }
    return cursor;
}

static uint32_t constants_crc(void) {
    unsigned char bytes[sizeof(k3_l1_value_bits) + sizeof(k3_group_maps) +
                        sizeof(k3_template_offsets) + sizeof(k3_template_maps)];
    size_t size = serialize_constants(bytes);
    if (size != sizeof(bytes)) fail("constant serialization size differs");
    uint32_t checksum = (uint32_t)crc32(0, bytes, (uInt)size);
    if (checksum != K3_CONSTANTS_CRC) fail("compiled constants differ");
    return checksum;
}

static unsigned first_block(unsigned record) {
    unsigned matrix = record % 3;
    return (record / 3) * 152 + matrix * 48;
}

static uint16_t weight_bits(unsigned template_id, unsigned selector, unsigned code) {
    if (template_id >= K3_TEMPLATE_COUNT || code >= 16) fail("template or code out of range");
    unsigned start = k3_template_offsets[template_id];
    unsigned end = k3_template_offsets[template_id + 1];
    if (end > K3_TEMPLATE_REFS || selector >= end - start) fail("selector out of range");
    unsigned map = k3_template_maps[start + selector];
    if (map >= K3_MAP_COUNT) fail("compiled map out of range");
    unsigned reference = k3_group_maps[map][code];
    if (reference >= 66) fail("absent value reference");
    return k3_l1_value_bits[reference];
}

static MapStore open_store(const char *path) {
    MapStore store = {0};
    unsigned char header[32];
    store.file = fopen(path, "rb");
    if (!store.file) fail("cannot open dataset");
    read_at(store.file, 0, header, sizeof(header));
    if (memcmp(header, "K3MAPS01", 8) || read32(header + 8) != constants_crc() ||
        read64(header + 16) != INDEX_BYTES) fail("wrong format or constant identity");
    uint64_t payload_size = read64(header + 24);
    if (payload_size > INT64_MAX - PAYLOAD_START || fseeko(store.file, 0, SEEK_END)) fail("invalid file length");
    off_t size = ftello(store.file);
    if (size < 0 || (uint64_t)size != PAYLOAD_START + payload_size) fail("file length differs");
    store.index = malloc(INDEX_BYTES);
    store.offsets = malloc((BLOCK_COUNT + 1) * sizeof(*store.offsets));
    if (!store.index || !store.offsets) fail("index allocation failed");
    read_at(store.file, 32, store.index, INDEX_BYTES);
    if ((uint32_t)crc32(0, store.index, INDEX_BYTES) != read32(header + 12)) fail("index checksum differs");
    for (unsigned record = 0; record < MATRIX_COUNT; record++)
        if (read16(store.index + record * 2) >= K3_TEMPLATE_COUNT) fail("matrix template out of range");
    store.offsets[0] = PAYLOAD_START;
    for (unsigned block = 0; block < BLOCK_COUNT; block++) {
        unsigned width = block % 152 < 96 ? 3584 : 3072;
        uint32_t length = read32(store.index + MATRIX_COUNT * 2 + block * 8);
        if (!length || length > ROWS_PER_BLOCK * width) fail("invalid compressed block length");
        store.offsets[block + 1] = store.offsets[block] + length;
    }
    if (store.offsets[BLOCK_COUNT] != PAYLOAD_START + payload_size) fail("block lengths do not cover file");
    return store;
}

static size_t decode_block(MapStore *store, unsigned record, unsigned block, unsigned char *output) {
    if (record >= MATRIX_COUNT || block >= (record % 3 == 2 ? 56U : 48U)) fail("block address out of range");
    unsigned width = record % 3 == 2 ? 3072 : 3584;
    unsigned count = width * ROWS_PER_BLOCK, raw_size = count / 2 + count / 32;
    unsigned block_id = first_block(record) + block;
    unsigned template_id = read16(store->index + record * 2);
    const unsigned char *entry = store->index + MATRIX_COUNT * 2 + block_id * 8;
    uint32_t length = read32(entry);
    unsigned char *compressed = malloc(length), *raw = malloc(raw_size);
    if (!compressed || !raw) fail("block allocation failed");
    read_at(store->file, store->offsets[block_id], compressed, length);
    z_stream decoder = {0};
    decoder.next_in = compressed;
    decoder.avail_in = length;
    decoder.next_out = raw;
    decoder.avail_out = raw_size;
    if (inflateInit(&decoder) != Z_OK) fail("zlib initialization failed");
    int result = inflate(&decoder, Z_FINISH);
    int valid = result == Z_STREAM_END && decoder.total_in == length && decoder.total_out == raw_size;
    inflateEnd(&decoder);
    if (!valid) fail("invalid compressed frame");
    for (unsigned coordinate = 0; coordinate < count; coordinate++) {
        unsigned selector = raw[count / 2 + coordinate / 32];
        unsigned code = (raw[coordinate / 2] >> (coordinate % 2 * 4)) & 15;
        uint16_t bits = weight_bits(template_id, selector, code);
        output[coordinate * 2] = (unsigned char)bits;
        output[coordinate * 2 + 1] = (unsigned char)(bits >> 8);
    }
    if ((uint32_t)crc32(0, output, count * 2) != read32(entry + 4)) fail("decoded block checksum differs");
    free(compressed);
    free(raw);
    return count * 2;
}

static unsigned argument(const char *text, unsigned limit) {
    char *end = NULL;
    errno = 0;
    unsigned long value = strtoul(text, &end, 10);
    if (errno || !*text || *end || value >= limit) fail("invalid numeric argument");
    return (unsigned)value;
}

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "constants")) {
        unsigned char bytes[sizeof(k3_l1_value_bits) + sizeof(k3_group_maps) +
                            sizeof(k3_template_offsets) + sizeof(k3_template_maps)];
        size_t size = serialize_constants(bytes);
        constants_crc();
        return fwrite(bytes, 1, size, stdout) != size || fflush(stdout);
    }
    if (argc != 3 && argc != 6) fail("usage: reader constants | reader verify FILE | reader row FILE EXPERT MATRIX ROW");
    int whole = argc == 3 && !strcmp(argv[1], "verify");
    if (!whole && (argc != 6 || strcmp(argv[1], "row"))) fail("invalid command");
    MapStore store = open_store(argv[2]);
    unsigned char *output = malloc(ROWS_PER_BLOCK * 3584 * 2);
    if (!output) fail("output allocation failed");
    if (whole) {
        for (unsigned record = 0; record < MATRIX_COUNT; record++) {
            for (unsigned block = 0; block < (record % 3 == 2 ? 56U : 48U); block++) {
                size_t size = decode_block(&store, record, block, output);
                if (fwrite(output, 1, size, stdout) != size) fail("output failed");
            }
        }
    } else {
        unsigned expert = argument(argv[3], 896);
        unsigned matrix = !strcmp(argv[4], "w1") ? 0 : !strcmp(argv[4], "w3") ? 1 : !strcmp(argv[4], "w2") ? 2 : 3;
        if (matrix == 3) fail("invalid matrix");
        unsigned row = argument(argv[5], matrix == 2 ? 3584 : 3072);
        unsigned width = matrix == 2 ? 3072 : 3584;
        decode_block(&store, expert * 3 + matrix, row / ROWS_PER_BLOCK, output);
        const unsigned char *values = output + (row % ROWS_PER_BLOCK) * width * 2;
        for (unsigned coordinate = 0; coordinate < width; coordinate++) {
            unsigned char bytes[4] = {0, 0, values[coordinate * 2], values[coordinate * 2 + 1]};
            if (fwrite(bytes, 1, 4, stdout) != 4) fail("output failed");
        }
    }
    free(output);
    free(store.offsets);
    free(store.index);
    fclose(store.file);
    return fflush(stdout) != 0;
}