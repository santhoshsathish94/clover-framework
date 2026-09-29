#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <zlib.h>
#include "../expert-constant-palette.h"

enum { RECORDS = 2688, BLOCKS = 136192, STRIDE = 24, BLOCK_ROWS = 64 };
typedef struct {
    FILE *file;
    unsigned char *index;
    uint64_t payload_start;
    uint64_t payload_size;
} ConstantStore;

static void fail(const char *message) {
    fprintf(stderr, "constant store: %s\n", message);
    exit(1);
}

static uint32_t read32(const unsigned char *data) {
    return (uint32_t)data[0] | (uint32_t)data[1] << 8 |
           (uint32_t)data[2] << 16 | (uint32_t)data[3] << 24;
}

static uint64_t read64(const unsigned char *data) {
    return read32(data) | (uint64_t)read32(data + 4) << 32;
}

static void read_at(FILE *file, uint64_t offset, unsigned char *data, size_t size) {
    if (offset > INT64_MAX || fseeko(file, (off_t)offset, SEEK_SET) || fread(data, 1, size, file) != size)
        fail("short read or invalid offset");
}

static void palette_bytes(unsigned char output[132]) {
    for (unsigned index = 0; index < 66; index++) {
        output[index * 2] = (unsigned char)k3_l1_value_bits[index];
        output[index * 2 + 1] = (unsigned char)(k3_l1_value_bits[index] >> 8);
    }
}

static ConstantStore open_store(const char *path) {
    ConstantStore store = {0};
    unsigned char header[64], palette[132];
    store.file = fopen(path, "rb");
    if (!store.file) fail("cannot open dataset");
    read_at(store.file, 0, header, sizeof(header));
    palette_bytes(palette);
    uint64_t index_size = (uint64_t)(RECORDS + BLOCKS) * STRIDE;
    if (memcmp(header, "K3CONST1", 8) || read32(header + 8) != 1 || read32(header + 12) != 1 ||
        read32(header + 16) != RECORDS || read32(header + 20) != BLOCKS ||
        read32(header + 24) != STRIDE || read32(header + 28) != STRIDE ||
        read32(header + 32) != (uint32_t)crc32(0, palette, sizeof(palette)) ||
        read64(header + 40) != index_size || read64(header + 56) != 0)
        fail("unsupported header or wrong compiled palette");
    store.payload_start = sizeof(header) + index_size;
    store.payload_size = read64(header + 48);
    if (store.payload_size > INT64_MAX - store.payload_start || fseeko(store.file, 0, SEEK_END))
        fail("invalid file length");
    off_t file_size = ftello(store.file);
    if (file_size < 0 || (uint64_t)file_size != store.payload_start + store.payload_size)
        fail("file length differs");
    store.index = malloc((size_t)index_size);
    if (!store.index) fail("index allocation failed");
    read_at(store.file, sizeof(header), store.index, (size_t)index_size);
    if (read32(header + 36) != (uint32_t)crc32(0, store.index, (uInt)index_size))
        fail("index checksum differs");
    uint64_t cursor = 0;
    uint32_t next_block = 0;
    for (unsigned record = 0; record < RECORDS; record++) {
        const unsigned char *entry = store.index + record * STRIDE;
        uint32_t groups = read32(entry + 8);
        uint32_t block_count = record % 3 == 2 ? 56 : 48;
        uint32_t width = record % 3 == 2 ? 3072 : 3584;
        if (read64(entry) != cursor || groups < 1 || groups > 256 ||
            read32(entry + 12) != next_block || read32(entry + 16) != block_count || read32(entry + 20))
            fail("invalid matrix record");
        cursor += (uint64_t)groups * 16;
        for (unsigned block = 0; block < block_count; block++, next_block++) {
            const unsigned char *item = store.index + (RECORDS + next_block) * STRIDE;
            uint32_t size = read32(item + 8), codec = read32(item + 12);
            if (read64(item) != cursor || !size || size > BLOCK_ROWS * width || codec > 1 || read32(item + 20))
                fail("invalid block record");
            cursor += size;
        }
    }
    if (cursor != store.payload_size || next_block != BLOCKS) fail("incomplete payload");
    return store;
}

static size_t decode_block(ConstantStore *store, unsigned record, unsigned block, unsigned char *output) {
    if (record >= RECORDS) fail("expert or matrix out of range");
    const unsigned char *entry = store->index + record * STRIDE;
    if (block >= read32(entry + 16)) fail("block out of range");
    const unsigned char *item = store->index + (RECORDS + read32(entry + 12) + block) * STRIDE;
    unsigned width = record % 3 == 2 ? 3072 : 3584;
    unsigned count = BLOCK_ROWS * width, raw_size = count / 2 + count / 32;
    unsigned char mapping[4096];
    unsigned char *compressed = malloc(read32(item + 8));
    unsigned char *raw = malloc(raw_size);
    if (!compressed || !raw) fail("block allocation failed");
    read_at(store->file, store->payload_start + read64(entry), mapping, read32(entry + 8) * 16);
    read_at(store->file, store->payload_start + read64(item), compressed, read32(item + 8));
    if (read32(item + 12) == 0) {
        if (read32(item + 8) != raw_size) fail("raw block length differs");
        memcpy(raw, compressed, raw_size);
    } else {
        z_stream decoder = {0};
        decoder.next_in = compressed;
        decoder.avail_in = read32(item + 8);
        decoder.next_out = raw;
        decoder.avail_out = raw_size;
        if (inflateInit(&decoder) != Z_OK) fail("zlib initialization failed");
        int result = inflate(&decoder, Z_FINISH);
        int valid = result == Z_STREAM_END && decoder.total_in == read32(item + 8) && decoder.total_out == raw_size;
        inflateEnd(&decoder);
        if (!valid) fail("invalid compressed frame");
    }
    for (unsigned coordinate = 0; coordinate < count; coordinate++) {
        unsigned selector = raw[count / 2 + coordinate / 32];
        unsigned code = (raw[coordinate / 2] >> (coordinate % 2 * 4)) & 15;
        if (selector >= read32(entry + 8)) fail("group selector outside mapping");
        unsigned reference = mapping[selector * 16 + code];
        if (reference >= 66) fail("value reference outside compiled palette");
        uint16_t bits = k3_l1_value_bits[reference];
        output[coordinate * 2] = (unsigned char)bits;
        output[coordinate * 2 + 1] = (unsigned char)(bits >> 8);
    }
    if ((uint32_t)crc32(0, output, count * 2) != read32(item + 16)) fail("decoded block checksum differs");
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
    if (argc == 2 && !strcmp(argv[1], "palette")) {
        unsigned char bytes[132];
        palette_bytes(bytes);
        return fwrite(bytes, 1, sizeof(bytes), stdout) != sizeof(bytes);
    }
    if (argc != 3 && argc != 6) fail("usage: reader palette | reader verify FILE | reader row FILE EXPERT MATRIX ROW");
    int whole = argc == 3 && !strcmp(argv[1], "verify");
    if (!whole && (argc != 6 || strcmp(argv[1], "row"))) fail("invalid command");
    ConstantStore store = open_store(argv[2]);
    unsigned char *output = malloc(BLOCK_ROWS * 3584 * 2);
    if (!output) fail("output allocation failed");
    if (whole) {
        for (unsigned record = 0; record < RECORDS; record++) {
            unsigned block_count = record % 3 == 2 ? 56 : 48;
            for (unsigned block = 0; block < block_count; block++) {
                size_t size = decode_block(&store, record, block, output);
                if (fwrite(output, 1, size, stdout) != size) fail("output write failed");
            }
        }
    } else {
        unsigned expert = argument(argv[3], 896);
        unsigned matrix = !strcmp(argv[4], "w1") ? 0 : !strcmp(argv[4], "w3") ? 1 : !strcmp(argv[4], "w2") ? 2 : 3;
        if (matrix == 3) fail("invalid matrix");
        unsigned row = argument(argv[5], matrix == 2 ? 3584 : 3072);
        unsigned width = matrix == 2 ? 3072 : 3584;
        decode_block(&store, expert * 3 + matrix, row / BLOCK_ROWS, output);
        unsigned char *values = output + (row % BLOCK_ROWS) * width * 2;
        for (unsigned coordinate = 0; coordinate < width; coordinate++) {
            unsigned char bytes[4] = {0, 0, values[coordinate * 2], values[coordinate * 2 + 1]};
            if (fwrite(bytes, 1, 4, stdout) != 4) fail("row write failed");
        }
    }
    free(output);
    free(store.index);
    fclose(store.file);
    return fflush(stdout) != 0;
}