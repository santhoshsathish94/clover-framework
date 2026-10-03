#ifndef TRANSFORMER_ROOT_H
#define TRANSFORMER_ROOT_H
#include "decode.h"
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

enum { ROOT_MATRICES = 2688, ROOT_BLOCKS = 136192, ROOT_INDEX_BYTES = 1094912,
    ROOT_CONSTANT_BYTES = 768, ROOT_PALETTE = 64, ROOT_MAPS = 13,
    ROOT_TEMPLATES = 29, ROOT_REFS = 186, ROOT_RAW = 121856, ROOT_COMPRESSED = 229376 };

typedef struct {
    FILE *file;
    unsigned char constants[ROOT_CONSTANT_BYTES];
    unsigned char *index;
    uint64_t *offsets;
    uint32_t crc_table[256];
    double values[ROOT_PALETTE];
} Root;

typedef struct {
    unsigned char compressed[ROOT_COMPRESSED], raw[ROOT_RAW];
} RootScratch;

static unsigned root_u16(const unsigned char *data)
{
    return (unsigned)data[0] | ((unsigned)data[1] << 8);
}

static uint32_t root_u32(const unsigned char *data)
{
    return (uint32_t)root_u16(data) | ((uint32_t)root_u16(data + 2) << 16);
}

static uint64_t root_u64(const unsigned char *data)
{
    return root_u32(data) | ((uint64_t)root_u32(data + 4) << 32);
}

static int root_seek(FILE *file, uint64_t offset)
{
    if (offset > INT64_MAX) return 0;
#ifdef _WIN32
    return _fseeki64(file, (int64_t)offset, SEEK_SET) == 0;
#else
    return fseeko(file, (off_t)offset, SEEK_SET) == 0;
#endif
}

static int root_length(FILE *file, uint64_t expected)
{
#ifdef _WIN32
    if (_fseeki64(file, 0, SEEK_END)) return 0;
    int64_t length = _ftelli64(file);
#else
    if (fseeko(file, 0, SEEK_END)) return 0;
    off_t length = ftello(file);
#endif
    return length >= 0 && (uint64_t)length == expected && root_seek(file, 0);
}

static uint32_t root_crc(const Root *root, const unsigned char *data, size_t count)
{
    uint32_t crc = UINT32_MAX;
    for (size_t index = 0; index < count; index++) crc = root->crc_table[(crc ^ data[index]) & 255] ^ (crc >> 8);
    return crc ^ UINT32_MAX;
}

static void root_close(Root *root)
{
    if (!root) return;
    if (root->file) fclose(root->file);
    free(root->index);
    free(root->offsets);
    free(root);
}

static Root *root_open(const char *directory)
{
    char path[4096];
    Root *root = calloc(1, sizeof *root);
    if (!root) return NULL;
    for (unsigned byte = 0; byte < 256; byte++) {
        uint32_t crc = byte;
        for (unsigned bit = 0; bit < 8; bit++) crc = (crc >> 1) ^ ((crc & 1) ? UINT32_C(0xedb88320) : 0);
        root->crc_table[byte] = crc;
    }
    int length = snprintf(path, sizeof path, "%s/constants.bin", directory);
    if (length < 0 || (size_t)length >= sizeof path) { root_close(root); return NULL; }
    FILE *constants = fopen(path, "rb");
    if (!constants) { root_close(root); return NULL; }
    int valid = root_length(constants, ROOT_CONSTANT_BYTES) &&
        fread(root->constants, 1, ROOT_CONSTANT_BYTES, constants) == ROOT_CONSTANT_BYTES;
    if (fclose(constants)) valid = 0;
    if (!valid) { root_close(root); return NULL; }
    for (unsigned index = 0; index < ROOT_PALETTE; index++) {
        unsigned bits = root_u16(root->constants + index * 2);
        if ((bits & 0x7f80U) == 0x7f80U) { root_close(root); return NULL; }
        uint32_t widened = (uint32_t)bits << 16;
        float value;
        memcpy(&value, &widened, 4);
        root->values[index] = (double)value;
    }
    const unsigned char *maps = root->constants + ROOT_PALETTE * 2;
    const unsigned char *offsets = maps + ROOT_MAPS * 16;
    const unsigned char *refs = offsets + (ROOT_TEMPLATES + 1) * 2;
    for (unsigned index = 0; index < ROOT_MAPS * 16; index++)
        if (maps[index] != 255 && maps[index] >= ROOT_PALETTE) valid = 0;
    if (root_u16(offsets) || root_u16(offsets + ROOT_TEMPLATES * 2) != ROOT_REFS) valid = 0;
    for (unsigned index = 0; index < ROOT_TEMPLATES; index++) {
        unsigned begin = root_u16(offsets + index * 2), end = root_u16(offsets + (index + 1) * 2);
        if (begin >= end || end > ROOT_REFS || end - begin > 256) valid = 0;
    }
    for (unsigned index = 0; index < ROOT_REFS; index++) if (root_u16(refs + index * 2) >= ROOT_MAPS) valid = 0;
    length = snprintf(path, sizeof path, "%s/experts.bin", directory);
    if (!valid || length < 0 || (size_t)length >= sizeof path) { root_close(root); return NULL; }
    root->file = fopen(path, "rb");
    unsigned char header[32];
    if (!root->file || fread(header, 1, sizeof header, root->file) != sizeof header ||
        memcmp(header, "K3MAPS01", 8) || root_u64(header + 16) != ROOT_INDEX_BYTES ||
        root_u32(header + 8) != root_crc(root, root->constants, ROOT_CONSTANT_BYTES)) { root_close(root); return NULL; }
    root->index = malloc(ROOT_INDEX_BYTES);
    root->offsets = malloc((ROOT_BLOCKS + 1) * sizeof *root->offsets);
    if (!root->index || !root->offsets || fread(root->index, 1, ROOT_INDEX_BYTES, root->file) != ROOT_INDEX_BYTES ||
        root_crc(root, root->index, ROOT_INDEX_BYTES) != root_u32(header + 12)) { root_close(root); return NULL; }
    for (unsigned matrix = 0; matrix < ROOT_MATRICES; matrix++)
        if (root_u16(root->index + matrix * 2) >= ROOT_TEMPLATES) valid = 0;
    root->offsets[0] = 32 + ROOT_INDEX_BYTES;
    for (unsigned block = 0; block < ROOT_BLOCKS; block++) {
        unsigned bytes = root_u32(root->index + ROOT_MATRICES * 2 + block * 8);
        if (!bytes || bytes > ROOT_COMPRESSED) valid = 0;
        root->offsets[block + 1] = root->offsets[block] + bytes;
    }
    if (!valid || root->offsets[ROOT_BLOCKS] - root->offsets[0] != root_u64(header + 24) ||
        !root_length(root->file, root->offsets[ROOT_BLOCKS])) { root_close(root); return NULL; }
    return root;
}

static int root_block(Root *root, RootScratch *scratch, unsigned expert, unsigned matrix, unsigned block)
{
    unsigned width = matrix == 2 ? 3072 : 3584, blocks = matrix == 2 ? 56 : 48;
    if (!root || !scratch || expert >= 896 || matrix >= 3 || block >= blocks) return 0;
    unsigned ordinal = expert * 152 + matrix * 48 + block;
    size_t bytes = (size_t)(root->offsets[ordinal + 1] - root->offsets[ordinal]);
    if (!root_seek(root->file, root->offsets[ordinal]) ||
        fread(scratch->compressed, 1, bytes, root->file) != bytes ||
        !decode_zlib(scratch->compressed, bytes, scratch->raw, 64 * width / 2 + 64 * width / 32)) return 0;
    const unsigned char *maps = root->constants + ROOT_PALETTE * 2;
    const unsigned char *offsets = maps + ROOT_MAPS * 16;
    const unsigned char *refs = offsets + (ROOT_TEMPLATES + 1) * 2;
    unsigned template_id = root_u16(root->index + (expert * 3 + matrix) * 2);
    unsigned begin = root_u16(offsets + template_id * 2), end = root_u16(offsets + (template_id + 1) * 2);
    uint32_t crc = UINT32_MAX;
    for (unsigned group = 0; group < 64 * width / 32; group++) {
        unsigned selector = scratch->raw[64 * width / 2 + group];
        if (selector >= end - begin) return 0;
        const unsigned char *map = maps + root_u16(refs + (begin + selector) * 2) * 16;
        for (unsigned coordinate = 0; coordinate < 32; coordinate++) {
            unsigned code = (scratch->raw[group * 16 + coordinate / 2] >> ((coordinate & 1) * 4)) & 15;
            unsigned value_id = map[code];
            if (value_id >= ROOT_PALETTE) return 0;
            for (unsigned byte = 0; byte < 2; byte++)
                crc = root->crc_table[(crc ^ root->constants[value_id * 2 + byte]) & 255] ^ (crc >> 8);
        }
    }
    return (crc ^ UINT32_MAX) == root_u32(root->index + ROOT_MATRICES * 2 + ordinal * 8 + 4);
}

static int root_project(Root *root, RootScratch *scratch, unsigned expert, unsigned matrix,
    const float *input, float *output)
{
    if (!root || !scratch || !input || !output || expert >= 896 || matrix >= 3) return 0;
    unsigned width = matrix == 2 ? 3072 : 3584, rows = matrix == 2 ? 3584 : 3072;
    const unsigned char *maps = root->constants + ROOT_PALETTE * 2;
    const unsigned char *offsets = maps + ROOT_MAPS * 16;
    const unsigned char *refs = offsets + (ROOT_TEMPLATES + 1) * 2;
    unsigned template_id = root_u16(root->index + (expert * 3 + matrix) * 2);
    unsigned begin = root_u16(offsets + template_id * 2);
    for (unsigned block = 0; block < rows / 64; block++) {
        if (!root_block(root, scratch, expert, matrix, block)) return 0;
        for (unsigned row = 0; row < 64; row++) {
            const unsigned char *codes = scratch->raw + row * width / 2;
            const unsigned char *selectors = scratch->raw + 64 * width / 2 + row * width / 32;
            double lanes[16] = {0};
            for (unsigned coordinate = 0; coordinate < width; coordinate += 16) {
                unsigned map_id = root_u16(refs + (begin + selectors[coordinate / 32]) * 2);
                const unsigned char *map = maps + map_id * 16;
                for (unsigned lane = 0; lane < 16; lane++) {
                    unsigned column = coordinate + lane;
                    unsigned code = (codes[column / 2] >> ((column & 1) * 4)) & 15;
                    lanes[lane] = lanes[lane] + root->values[map[code]] * (double)input[column];
                }
            }
            double sums[4];
            for (unsigned lane = 0; lane < 4; lane++)
                sums[lane] = (lanes[lane] + lanes[lane + 8]) + (lanes[lane + 4] + lanes[lane + 12]);
            output[block * 64 + row] = (float)((sums[0] + sums[2]) + (sums[1] + sums[3]));
        }
    }
    return 1;
}
#endif