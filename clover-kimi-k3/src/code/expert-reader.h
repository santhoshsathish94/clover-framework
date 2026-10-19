#ifndef TRANSFORMER_EXPERT_READER_H
#define TRANSFORMER_EXPERT_READER_H
#include "inflate-fast.h"
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <omp.h>
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#include "expert-metadata.h"

#ifndef EXPERT_STORE_PHASE_BEGIN
#define EXPERT_STORE_PHASE_BEGIN(name)
#define EXPERT_STORE_PHASE_END(name,phase)
#endif

enum { EXPERT_STORE_MATRICES = 2688, EXPERT_STORE_BLOCKS = 136192, EXPERT_STORE_INDEX_BYTES = 1094912,
    EXPERT_STORE_RAW = 121856, EXPERT_STORE_COMPRESSED = 229376, EXPERT_STORE_DOWN_RAW = 104448, EXPERT_RAW_BYTES = 17547264 };

typedef struct {
    FILE *file;
    unsigned char *constants;
    unsigned constant_bytes, palette_count, map_count, template_count, reference_count;
    unsigned char *index;
    uint64_t *offsets;
    uint32_t crc_table[256];
    uint32_t crc_wide[7][256];
    double *values;
    float (*pairs)[256][2];
    const unsigned char *direct;
    size_t direct_bytes;
    const unsigned char *prefetched;
    size_t prefetched_bytes;
    unsigned prefetched_expert;
} ExpertStore;

typedef struct {
    unsigned char compressed[EXPERT_STORE_COMPRESSED], raw[EXPERT_STORE_RAW];
} ExpertStoreScratch;

static unsigned expert_store_u16(const unsigned char *data)
{
    return (unsigned)data[0] | ((unsigned)data[1] << 8);
}

static uint32_t expert_store_u32(const unsigned char *data)
{
    return (uint32_t)expert_store_u16(data) | ((uint32_t)expert_store_u16(data + 2) << 16);
}

static uint64_t expert_store_u64(const unsigned char *data)
{
    return expert_store_u32(data) | ((uint64_t)expert_store_u32(data + 4) << 32);
}

static int expert_store_seek(FILE *file, uint64_t offset)
{
    if (offset > INT64_MAX) return 0;
#ifdef _WIN32
    return _fseeki64(file, (int64_t)offset, SEEK_SET) == 0;
#else
    return fseeko(file, (off_t)offset, SEEK_SET) == 0;
#endif
}

static int expert_store_length(FILE *file, uint64_t expected)
{
#ifdef _WIN32
    if (_fseeki64(file, 0, SEEK_END)) return 0;
    int64_t length = _ftelli64(file);
#else
    if (fseeko(file, 0, SEEK_END)) return 0;
    off_t length = ftello(file);
#endif
    return length >= 0 && (uint64_t)length == expected && expert_store_seek(file, 0);
}

static uint32_t expert_store_crc(const ExpertStore *store, const unsigned char *data, size_t count)
{
    uint32_t crc = UINT32_MAX;
    for (size_t index = 0; index < count; index++) crc = store->crc_table[(crc ^ data[index]) & 255] ^ (crc >> 8);
    return crc ^ UINT32_MAX;
}

static void expert_store_close(ExpertStore *store)
{
    if (!store) return;
    if (store->file) fclose(store->file);
    if (store->direct) munmap((void *)store->direct, store->direct_bytes);
    free(store->pairs);
    free(store->constants);
    free(store->values);
    free(store->index);
    free(store->offsets);
    free(store);
}

static ExpertStore *expert_store_open(const char *directory, unsigned layer)
{
    char path[4096];
    ExpertStore *store = calloc(1, sizeof *store);
    if (!store) return NULL;
    unsigned counts[4] = {0};
    struct stat constants_info;
    int constants_length = snprintf(path, sizeof path, "%s/constants.bin", directory);
    if (constants_length < 0 || (size_t)constants_length >= sizeof path || stat(path, &constants_info) || constants_info.st_size <= 0) { expert_store_close(store); return NULL; }
    int metadata_length = snprintf(path, sizeof path, "%s/maps.json", directory);
    if (metadata_length < 0 || (size_t)metadata_length >= sizeof path || !expert_store_metadata(path, layer, (size_t)constants_info.st_size, counts)) { expert_store_close(store); return NULL; }
    store->palette_count = counts[0]; store->map_count = counts[1];
    store->template_count = counts[2]; store->reference_count = counts[3];
    store->constant_bytes = counts[0]*2 + counts[1]*16 + (counts[2]+1)*2 + counts[3]*2;
    store->constants = malloc(store->constant_bytes);
    store->values = malloc(store->palette_count * sizeof *store->values);
    if (!store->constants || !store->values) { expert_store_close(store); return NULL; }
    for (unsigned byte = 0; byte < 256; byte++) {
        uint32_t crc = byte;
        for (unsigned bit = 0; bit < 8; bit++) crc = (crc >> 1) ^ ((crc & 1) ? UINT32_C(0xedb88320) : 0);
        store->crc_table[byte] = crc;
    }
    for (unsigned byte=0;byte<256;byte++) {
        uint32_t crc=store->crc_table[byte];
        for (unsigned depth=0;depth<7;depth++) {
            crc=store->crc_table[crc&255]^(crc>>8);
            store->crc_wide[depth][byte]=crc;
        }
    }
    int length = snprintf(path, sizeof path, "%s/constants.bin", directory);
    if (length < 0 || (size_t)length >= sizeof path) { expert_store_close(store); return NULL; }
    FILE *constants = fopen(path, "rb");
    if (!constants) { expert_store_close(store); return NULL; }
    int valid = expert_store_length(constants, store->constant_bytes) &&
        fread(store->constants, 1, store->constant_bytes, constants) == store->constant_bytes;
    if (fclose(constants)) valid = 0;
    if (!valid) { expert_store_close(store); return NULL; }
    for (unsigned index = 0; index < store->palette_count; index++) {
        unsigned bits = expert_store_u16(store->constants + index * 2);
        if ((bits & 0x7f80U) == 0x7f80U) { expert_store_close(store); return NULL; }
        uint32_t widened = (uint32_t)bits << 16;
        float value;
        memcpy(&value, &widened, 4);
        store->values[index] = (double)value;
    }
    const unsigned char *maps = store->constants + store->palette_count * 2;
    const unsigned char *offsets = maps + store->map_count * 16;
    const unsigned char *refs = offsets + (store->template_count + 1) * 2;
    for (unsigned index = 0; index < store->map_count * 16; index++)
        if (maps[index] != 255 && maps[index] >= store->palette_count) valid = 0;
    if (expert_store_u16(offsets) || expert_store_u16(offsets + store->template_count * 2) != store->reference_count) valid = 0;
    for (unsigned index = 0; index < store->template_count; index++) {
        unsigned begin = expert_store_u16(offsets + index * 2), end = expert_store_u16(offsets + (index + 1) * 2);
        if (begin >= end || end > store->reference_count || end - begin > 256) valid = 0;
    }
    for (unsigned index = 0; index < store->reference_count; index++) if (expert_store_u16(refs + index * 2) >= store->map_count) valid = 0;
    store->pairs = malloc((size_t)store->map_count * sizeof *store->pairs);
    if (!store->pairs) { expert_store_close(store); return NULL; }
    for (unsigned entry = 0; entry < store->map_count; entry++) {
        const unsigned char *map = maps + entry * 16;
        for (unsigned byte = 0; byte < 256; byte++) {
            unsigned low = map[byte & 15], high = map[byte >> 4];
            store->pairs[entry][byte][0] = low < store->palette_count ? (float)store->values[low] : 0.0f;
            store->pairs[entry][byte][1] = high < store->palette_count ? (float)store->values[high] : 0.0f;
        }
    }
    length = snprintf(path, sizeof path, "%s/experts.bin", directory);
    if (!valid || length < 0 || (size_t)length >= sizeof path) { expert_store_close(store); return NULL; }
    store->file = fopen(path, "rb");
    unsigned char header[32];
    if (!store->file || fread(header, 1, sizeof header, store->file) != sizeof header ||
        memcmp(header, "K3MAPS01", 8) || expert_store_u64(header + 16) != EXPERT_STORE_INDEX_BYTES ||
        expert_store_u32(header + 8) != expert_store_crc(store, store->constants, store->constant_bytes)) { expert_store_close(store); return NULL; }
    store->index = malloc(EXPERT_STORE_INDEX_BYTES);
    store->offsets = malloc((EXPERT_STORE_BLOCKS + 1) * sizeof *store->offsets);
    if (!store->index || !store->offsets || fread(store->index, 1, EXPERT_STORE_INDEX_BYTES, store->file) != EXPERT_STORE_INDEX_BYTES ||
        expert_store_crc(store, store->index, EXPERT_STORE_INDEX_BYTES) != expert_store_u32(header + 12)) { expert_store_close(store); return NULL; }
    for (unsigned matrix = 0; matrix < EXPERT_STORE_MATRICES; matrix++)
        if (expert_store_u16(store->index + matrix * 2) >= store->template_count) valid = 0;
    store->offsets[0] = 32 + EXPERT_STORE_INDEX_BYTES;
    for (unsigned block = 0; block < EXPERT_STORE_BLOCKS; block++) {
        unsigned bytes = expert_store_u32(store->index + EXPERT_STORE_MATRICES * 2 + block * 8);
        if (!bytes || bytes > EXPERT_STORE_COMPRESSED) valid = 0;
        store->offsets[block + 1] = store->offsets[block] + bytes;
    }
    int direct_length = snprintf(path, sizeof path, "%s/experts.direct", directory);
    if (direct_length > 0 && (size_t)direct_length < sizeof path) {
        int handle = open(path, O_RDONLY);
        if (handle >= 0) {
            struct stat direct_info;
            size_t bytes = (size_t)896 * EXPERT_RAW_BYTES;
            if (!fstat(handle, &direct_info) && (size_t)direct_info.st_size == bytes) {
                void *mapped = mmap(NULL, bytes, PROT_READ, MAP_SHARED, handle, 0);
                if (mapped != MAP_FAILED) { store->direct = mapped; store->direct_bytes = bytes; }
            }
            close(handle);
        }
    }
    if (!valid || store->offsets[EXPERT_STORE_BLOCKS] - store->offsets[0] != expert_store_u64(header + 24) ||
        (!store->direct && !expert_store_length(store->file, store->offsets[EXPERT_STORE_BLOCKS]))) { expert_store_close(store); return NULL; }
    return store;
}

#include "expert-validation.h"

static int expert_store_block(ExpertStore *store, ExpertStoreScratch *scratch, unsigned expert, unsigned matrix, unsigned block)
{
    unsigned width = matrix == 2 ? 3072 : 3584, blocks = matrix == 2 ? 56 : 48;
    if (!store || !scratch || expert >= 896 || matrix >= 3 || block >= blocks) return 0;
    unsigned ordinal = expert * 152 + matrix * 48 + block;
    size_t bytes = (size_t)(store->offsets[ordinal + 1] - store->offsets[ordinal]);
    EXPERT_STORE_PHASE_BEGIN(read_started);
    const unsigned char *compressed=scratch->compressed;
    if (store->prefetched && store->prefetched_expert==expert) {
        uint64_t offset=store->offsets[ordinal]-store->offsets[expert*152];
        if (offset>store->prefetched_bytes || bytes>store->prefetched_bytes-offset) return 0;
        compressed=store->prefetched+(size_t)offset;
    } else if (!expert_store_read_at(store->file, scratch->compressed, bytes, store->offsets[ordinal])) return 0;
    EXPERT_STORE_PHASE_END(read_started,0);
    EXPERT_STORE_PHASE_BEGIN(decode_started);
    if (!decode_zlib(compressed, bytes, scratch->raw, 64 * width / 2 + 64 * width / 32)) return 0;
    EXPERT_STORE_PHASE_END(decode_started,1);
    EXPERT_STORE_PHASE_BEGIN(check_started);
    const unsigned char *maps = store->constants + store->palette_count * 2;
    const unsigned char *offsets = maps + store->map_count * 16;
    const unsigned char *refs = offsets + (store->template_count + 1) * 2;
    unsigned template_id = expert_store_u16(store->index + (expert * 3 + matrix) * 2);
    unsigned begin = expert_store_u16(offsets + template_id * 2), end = expert_store_u16(offsets + (template_id + 1) * 2);
    uint32_t crc = UINT32_MAX;
    for (unsigned group = 0; group < 64 * width / 32; group++) {
        unsigned selector = scratch->raw[64 * width / 2 + group];
        if (selector >= end - begin) return 0;
        const unsigned char *map = maps + expert_store_u16(refs + (begin + selector) * 2) * 16;
        if (!expert_store_validate_group(store,scratch->raw+group*16,map,&crc)) return 0;
    }
    int valid=(crc ^ UINT32_MAX) == expert_store_u32(store->index + EXPERT_STORE_MATRICES * 2 + ordinal * 8 + 4);
    EXPERT_STORE_PHASE_END(check_started,2);
    return valid;
}

static int expert_store_project(ExpertStore *store, ExpertStoreScratch *scratch, unsigned expert, unsigned matrix,
    const float *input, float *output)
{
    if (!store || !scratch || !input || !output || expert >= 896 || matrix >= 3) return 0;
    unsigned width = matrix == 2 ? 3072 : 3584, rows = matrix == 2 ? 3584 : 3072;
    const unsigned char *maps = store->constants + store->palette_count * 2;
    const unsigned char *offsets = maps + store->map_count * 16;
    const unsigned char *refs = offsets + (store->template_count + 1) * 2;
    unsigned template_id = expert_store_u16(store->index + (expert * 3 + matrix) * 2);
    unsigned begin = expert_store_u16(offsets + template_id * 2);
    int valid = 1;
    ExpertStoreScratch *pool = scratch;
    double converted[3584];
    for (unsigned coordinate = 0; coordinate < width; coordinate++) converted[coordinate] = (double)input[coordinate];
#pragma omp parallel for schedule(static) reduction(&:valid)
    for (unsigned block = 0; block < rows / 64; block++) {
        ExpertStoreScratch *scratch = pool + omp_get_thread_num();
        const unsigned char *data;
        if (store->direct) data = store->direct + (size_t)expert * EXPERT_RAW_BYTES +
            (matrix < 2 ? (size_t)(matrix * 48 + block) * EXPERT_STORE_RAW
                        : (size_t)96 * EXPERT_STORE_RAW + (size_t)block * EXPERT_STORE_DOWN_RAW);
        else if (expert_store_block(store, scratch, expert, matrix, block)) data = scratch->raw;
        else { valid = 0; continue; }
        EXPERT_STORE_PHASE_BEGIN(math_started);
        for (unsigned row = 0; row < 64; row++) {
            const unsigned char *codes = data + row * width / 2;
            const unsigned char *selectors = data + 64 * width / 2 + row * width / 32;
            double sums[4];
#if defined(__AVX2__)
            __m256d lane0 = _mm256_setzero_pd(), lane4 = _mm256_setzero_pd();
            __m256d lane8 = _mm256_setzero_pd(), lane12 = _mm256_setzero_pd();
            for (unsigned coordinate = 0; coordinate < width; coordinate += 16) {
                const float (*pair)[2] = store->pairs[expert_store_u16(refs + (begin + selectors[coordinate / 32]) * 2)];
                const unsigned char *packed = codes + coordinate / 2;
                const double *source = converted + coordinate;
                __m256d w0 = _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd((const double *)pair[packed[0]]), _mm_load_sd((const double *)pair[packed[1]]))));
                __m256d w1 = _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd((const double *)pair[packed[2]]), _mm_load_sd((const double *)pair[packed[3]]))));
                __m256d w2 = _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd((const double *)pair[packed[4]]), _mm_load_sd((const double *)pair[packed[5]]))));
                __m256d w3 = _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd((const double *)pair[packed[6]]), _mm_load_sd((const double *)pair[packed[7]]))));
                lane0 = _mm256_add_pd(lane0, _mm256_mul_pd(w0, _mm256_loadu_pd(source)));
                lane4 = _mm256_add_pd(lane4, _mm256_mul_pd(w1, _mm256_loadu_pd(source + 4)));
                lane8 = _mm256_add_pd(lane8, _mm256_mul_pd(w2, _mm256_loadu_pd(source + 8)));
                lane12 = _mm256_add_pd(lane12, _mm256_mul_pd(w3, _mm256_loadu_pd(source + 12)));
            }
            _mm256_storeu_pd(sums, _mm256_add_pd(_mm256_add_pd(lane0, lane8), _mm256_add_pd(lane4, lane12)));
#else
            double lanes[16] = {0};
            for (unsigned coordinate = 0; coordinate < width; coordinate += 16) {
                const float (*pair)[2] = store->pairs[expert_store_u16(refs + (begin + selectors[coordinate / 32]) * 2)];
                for (unsigned lane = 0; lane < 16; lane++) {
                    unsigned column = coordinate + lane;
                    lanes[lane] = lanes[lane] + (double)pair[codes[column / 2]][column & 1] * converted[column];
                }
            }
            for (unsigned lane = 0; lane < 4; lane++)
                sums[lane] = (lanes[lane] + lanes[lane + 8]) + (lanes[lane + 4] + lanes[lane + 12]);
#endif
            output[block * 64 + row] = (float)((sums[0] + sums[2]) + (sums[1] + sums[3]));
        }
        EXPERT_STORE_PHASE_END(math_started,3);
    }
    return valid;
}

enum { EXPERT_STORE_MAX_ROWS = 8 };

/* Asks the kernel to start reading an expert before the maths needs it. Expert reads are
   demand page faults on the compute thread, so without this the thread stalls once per
   expert and the device never sees more than one request in flight. */
static void expert_store_prefetch(const ExpertStore *store, unsigned expert)
{
    if (!store || !store->direct || expert >= 896) return;
    (void)posix_madvise((void *)(store->direct + (size_t)expert * EXPERT_RAW_BYTES), EXPERT_RAW_BYTES, POSIX_MADV_WILLNEED);
}

static int expert_store_project_rows(ExpertStore *store, ExpertStoreScratch *scratch, unsigned expert, unsigned matrix,
    const float *const *inputs, float *const *outputs, unsigned count)
{
    if (!store || !scratch || !inputs || !outputs || !count || count > EXPERT_STORE_MAX_ROWS ||
        expert >= 896 || matrix >= 3) return 0;
    unsigned width = matrix == 2 ? 3072 : 3584, rows = matrix == 2 ? 3584 : 3072;
    const unsigned char *maps = store->constants + store->palette_count * 2;
    const unsigned char *offsets = maps + store->map_count * 16;
    const unsigned char *refs = offsets + (store->template_count + 1) * 2;
    unsigned template_id = expert_store_u16(store->index + (expert * 3 + matrix) * 2);
    unsigned begin = expert_store_u16(offsets + template_id * 2);
    int valid = 1;
    ExpertStoreScratch *pool = scratch;
    static double converted[EXPERT_STORE_MAX_ROWS][3584];
    for (unsigned entry = 0; entry < count; entry++) {
        if (!inputs[entry] || !outputs[entry]) return 0;
        for (unsigned coordinate = 0; coordinate < width; coordinate++)
            converted[entry][coordinate] = (double)inputs[entry][coordinate];
    }
#pragma omp parallel for schedule(static) reduction(&:valid)
    for (unsigned block = 0; block < rows / 64; block++) {
        ExpertStoreScratch *scratch = pool + omp_get_thread_num();
        const unsigned char *data;
        if (store->direct) data = store->direct + (size_t)expert * EXPERT_RAW_BYTES +
            (matrix < 2 ? (size_t)(matrix * 48 + block) * EXPERT_STORE_RAW
                        : (size_t)96 * EXPERT_STORE_RAW + (size_t)block * EXPERT_STORE_DOWN_RAW);
        else if (expert_store_block(store, scratch, expert, matrix, block)) data = scratch->raw;
        else { valid = 0; continue; }
        EXPERT_STORE_PHASE_BEGIN(math_started);
        for (unsigned row = 0; row < 64; row++) {
            const unsigned char *codes = data + row * width / 2;
            const unsigned char *selectors = data + 64 * width / 2 + row * width / 32;
            double sums[EXPERT_STORE_MAX_ROWS][4];
#if defined(__AVX2__)
            __m256d lane0[EXPERT_STORE_MAX_ROWS], lane4[EXPERT_STORE_MAX_ROWS];
            __m256d lane8[EXPERT_STORE_MAX_ROWS], lane12[EXPERT_STORE_MAX_ROWS];
            for (unsigned entry = 0; entry < count; entry++) {
                lane0[entry] = _mm256_setzero_pd(); lane4[entry] = _mm256_setzero_pd();
                lane8[entry] = _mm256_setzero_pd(); lane12[entry] = _mm256_setzero_pd();
            }
            for (unsigned coordinate = 0; coordinate < width; coordinate += 16) {
                const float (*pair)[2] = store->pairs[expert_store_u16(refs + (begin + selectors[coordinate / 32]) * 2)];
                const unsigned char *packed = codes + coordinate / 2;
                __m256d w0 = _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd((const double *)pair[packed[0]]), _mm_load_sd((const double *)pair[packed[1]]))));
                __m256d w1 = _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd((const double *)pair[packed[2]]), _mm_load_sd((const double *)pair[packed[3]]))));
                __m256d w2 = _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd((const double *)pair[packed[4]]), _mm_load_sd((const double *)pair[packed[5]]))));
                __m256d w3 = _mm256_cvtps_pd(_mm_castpd_ps(_mm_unpacklo_pd(_mm_load_sd((const double *)pair[packed[6]]), _mm_load_sd((const double *)pair[packed[7]]))));
                for (unsigned entry = 0; entry < count; entry++) {
                    const double *source = converted[entry] + coordinate;
                    lane0[entry] = _mm256_add_pd(lane0[entry], _mm256_mul_pd(w0, _mm256_loadu_pd(source)));
                    lane4[entry] = _mm256_add_pd(lane4[entry], _mm256_mul_pd(w1, _mm256_loadu_pd(source + 4)));
                    lane8[entry] = _mm256_add_pd(lane8[entry], _mm256_mul_pd(w2, _mm256_loadu_pd(source + 8)));
                    lane12[entry] = _mm256_add_pd(lane12[entry], _mm256_mul_pd(w3, _mm256_loadu_pd(source + 12)));
                }
            }
            for (unsigned entry = 0; entry < count; entry++)
                _mm256_storeu_pd(sums[entry], _mm256_add_pd(_mm256_add_pd(lane0[entry], lane8[entry]),
                                                            _mm256_add_pd(lane4[entry], lane12[entry])));
#else
            double lanes[EXPERT_STORE_MAX_ROWS][16] = {{0}};
            for (unsigned coordinate = 0; coordinate < width; coordinate += 16) {
                const float (*pair)[2] = store->pairs[expert_store_u16(refs + (begin + selectors[coordinate / 32]) * 2)];
                for (unsigned entry = 0; entry < count; entry++)
                    for (unsigned lane = 0; lane < 16; lane++) {
                        unsigned column = coordinate + lane;
                        lanes[entry][lane] = lanes[entry][lane] + (double)pair[codes[column / 2]][column & 1] * converted[entry][column];
                    }
            }
            for (unsigned entry = 0; entry < count; entry++)
                for (unsigned lane = 0; lane < 4; lane++)
                    sums[entry][lane] = (lanes[entry][lane] + lanes[entry][lane + 8]) + (lanes[entry][lane + 4] + lanes[entry][lane + 12]);
#endif
            for (unsigned entry = 0; entry < count; entry++)
                outputs[entry][block * 64 + row] = (float)((sums[entry][0] + sums[entry][2]) + (sums[entry][1] + sums[entry][3]));
        }
        EXPERT_STORE_PHASE_END(math_started,3);
    }
    return valid;
}
#endif