#define main l1_unused_map_cli
#include "expert-map-store.c"
#undef main
#include <omp.h>

#ifndef NPOS
#define NPOS 5
#endif

static MapStore l1_store;
static unsigned char *l1_packed;
static unsigned char *l1_compressed;
static size_t l1_compressed_capacity;
static uint64_t l1_blocks, l1_bytes, l1_weights, l1_projection_values;
static FILE *l1_trace;
static int l1_open;

int l1_compact_enabled(void) {
    const char *path = getenv("K3_L1_COMPACT");
    return path && *path;
}

static void l1_initialize(void) {
    if (l1_open) return;
    const char *trace = getenv("K3_L1_TRACE");
    if (!l1_compact_enabled() || !trace) fail("compact dataset/trace not configured");
    l1_store = open_store(getenv("K3_L1_COMPACT"));
    for (unsigned block = 0; block < BLOCK_COUNT; block++) {
        uint32_t length = read32(l1_store.index + MATRIX_COUNT * 2 + block * 8);
        if (length > l1_compressed_capacity) l1_compressed_capacity = length;
    }
    l1_packed = malloc(121856);
    l1_compressed = malloc(l1_compressed_capacity);
    l1_trace = fopen(trace, "wx");
    if (!l1_packed || !l1_compressed || !l1_trace) fail("compact buffer/trace initialization failed");
    l1_open = 1;
}

void l1_compact_project(float *const *outputs, const float *const *inputs,
                        int positions, int expert, int matrix) {
    if (positions < 1 || positions > NPOS || expert < 0 || expert >= 896 || matrix < 0 || matrix >= 3)
        fail("compact projection address invalid");
    l1_initialize();
    unsigned width = matrix == 2 ? 3072 : 3584;
    unsigned rows = matrix == 2 ? 3584 : 3072;
    unsigned record = (unsigned)expert * 3 + (unsigned)matrix;
    unsigned template_id = read16(l1_store.index + record * 2);
    unsigned count = width * ROWS_PER_BLOCK;
    unsigned raw_size = count / 2 + count / 32;
    const z_crc_t *crc_table = get_crc_table();
    uint64_t bytes = 0;
    uint32_t row_crc[ROWS_PER_BLOCK];
    for (unsigned block = 0; block < rows / ROWS_PER_BLOCK; block++) {
        unsigned block_id = first_block(record) + block;
        const unsigned char *entry = l1_store.index + MATRIX_COUNT * 2 + block_id * 8;
        uint32_t length = read32(entry);
        read_at(l1_store.file, l1_store.offsets[block_id], l1_compressed, length);
        z_stream decoder = {0};
        decoder.next_in = l1_compressed;
        decoder.avail_in = length;
        decoder.next_out = l1_packed;
        decoder.avail_out = raw_size;
        if (inflateInit(&decoder) != Z_OK) fail("compact inflate initialization failed");
        int result = inflate(&decoder, Z_FINISH);
        int valid = result == Z_STREAM_END && decoder.total_in == length && decoder.total_out == raw_size;
        inflateEnd(&decoder);
        if (!valid) fail("compact block framing differs");
#pragma omp parallel for schedule(static)
        for (unsigned row = 0; row < ROWS_PER_BLOCK; row++) {
            double lanes[NPOS][16] = {{0}};
            uint32_t crc = UINT32_MAX;
            const unsigned char *packed = l1_packed + row * width / 2;
            const unsigned char *selectors = l1_packed + count / 2 + row * width / 32;
            for (unsigned coordinate = 0; coordinate < width; coordinate++) {
                unsigned code = (packed[coordinate / 2] >> (coordinate % 2 * 4)) & 15;
                uint16_t bits = weight_bits(template_id, selectors[coordinate / 32], code);
                crc = (crc >> 8) ^ (uint32_t)crc_table[(crc ^ bits) & 255];
                crc = (crc >> 8) ^ (uint32_t)crc_table[(crc ^ (bits >> 8)) & 255];
                uint32_t wide = (uint32_t)bits << 16;
                float weight;
                memcpy(&weight, &wide, sizeof(weight));
                for (int position = 0; position < positions; position++) {
                    double product = (double)weight * (double)inputs[position][coordinate];
                    lanes[position][coordinate % 16] += product;
                }
            }
            row_crc[row] = crc ^ UINT32_MAX;
            for (int position = 0; position < positions; position++) {
                double partial[4];
                for (unsigned lane = 0; lane < 4; lane++)
                    partial[lane] = (lanes[position][lane] + lanes[position][lane + 8]) +
                                    (lanes[position][lane + 4] + lanes[position][lane + 12]);
                outputs[position][block * ROWS_PER_BLOCK + row] =
                    (float)((partial[0] + partial[2]) + (partial[1] + partial[3]));
            }
        }
        uLong checksum = 0;
        for (unsigned row = 0; row < ROWS_PER_BLOCK; row++)
            checksum = crc32_combine(checksum, row_crc[row], width * 2);
        if ((uint32_t)checksum != read32(entry + 4)) fail("compact streamed weight checksum differs");
        l1_blocks++;
        bytes += length;
    }
    l1_bytes += bytes;
    l1_weights += (uint64_t)rows * width;
    l1_projection_values += (uint64_t)positions * rows;
    fprintf(l1_trace, "{\"type\":\"projection\",\"layer\":1,\"expert\":%d,\"matrix\":%d,"
            "\"positions\":%d,\"rows\":%u,\"width\":%u,\"blocks\":%u,\"compressed_bytes\":%llu,"
            "\"crc_pass\":true}\n", expert, matrix, positions, rows, width, rows / ROWS_PER_BLOCK,
            (unsigned long long)bytes);
    fflush(l1_trace);
}

void l1_compact_finish(void) {
    if (!l1_open) {
        if (l1_compact_enabled()) fail("compact path never consumed");
        return;
    }
    fprintf(l1_trace, "{\"type\":\"summary\",\"blocks\":%llu,\"compressed_bytes\":%llu,"
            "\"weights_consumed\":%llu,\"projection_values\":%llu,\"packed_buffer_bytes\":121856,"
            "\"compressed_buffer_bytes\":%zu,\"index_bytes\":%u,\"derived_offset_bytes\":%zu,"
            "\"weight_array_bytes\":0,\"checksum_array_bytes\":%zu,"
            "\"double_accumulator_bytes_per_worker\":%zu,\"positions_capacity\":%d}\n",
            (unsigned long long)l1_blocks, (unsigned long long)l1_bytes,
            (unsigned long long)l1_weights, (unsigned long long)l1_projection_values,
            l1_compressed_capacity, INDEX_BYTES, (BLOCK_COUNT + 1) * sizeof(uint64_t),
            ROWS_PER_BLOCK * sizeof(uint32_t), NPOS * 16 * sizeof(double), NPOS);
    fclose(l1_trace);
    fclose(l1_store.file);
    free(l1_store.index);
    free(l1_store.offsets);
    free(l1_packed);
    free(l1_compressed);
    l1_open = 0;
}