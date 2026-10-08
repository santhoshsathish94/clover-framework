#ifndef TRANSFORMER_ROOT_H
#define TRANSFORMER_ROOT_H
#include "decode.h"
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
#include "root-metadata.h"

#ifndef ROOT_PHASE_BEGIN
#define ROOT_PHASE_BEGIN(name)
#define ROOT_PHASE_END(name,phase)
#endif

enum { ROOT_MATRICES = 2688, ROOT_BLOCKS = 136192, ROOT_INDEX_BYTES = 1094912,
    ROOT_RAW = 121856, ROOT_COMPRESSED = 229376, ROOT_DOWN_RAW = 104448, ROOT_EXPERT_RAW = 17547264 };

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
    int direct_fd;
    const unsigned char *prefetched;
    size_t prefetched_bytes;
    unsigned prefetched_expert;
    unsigned layer;
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
    if (root->direct) munmap((void *)root->direct, root->direct_bytes);
    if (root->direct_fd > 0) close(root->direct_fd);
    free(root->pairs);
    free(root->constants);
    free(root->values);
    free(root->index);
    free(root->offsets);
    free(root);
}

static Root *root_open(const char *directory, unsigned layer)
{
    char path[4096];
    Root *root = calloc(1, sizeof *root);
    if (!root) return NULL;
    root->layer = layer;
    unsigned counts[4] = {0};
    struct stat constants_info;
    int constants_length = snprintf(path, sizeof path, "%s/constants.bin", directory);
    if (constants_length < 0 || (size_t)constants_length >= sizeof path || stat(path, &constants_info) || constants_info.st_size <= 0) { root_close(root); return NULL; }
    int metadata_length = snprintf(path, sizeof path, "%s/maps.json", directory);
    if (metadata_length < 0 || (size_t)metadata_length >= sizeof path || !root_metadata(path, layer, (size_t)constants_info.st_size, counts)) { root_close(root); return NULL; }
    root->palette_count = counts[0]; root->map_count = counts[1];
    root->template_count = counts[2]; root->reference_count = counts[3];
    root->constant_bytes = counts[0]*2 + counts[1]*16 + (counts[2]+1)*2 + counts[3]*2;
    root->constants = malloc(root->constant_bytes);
    root->values = malloc(root->palette_count * sizeof *root->values);
    if (!root->constants || !root->values) { root_close(root); return NULL; }
    for (unsigned byte = 0; byte < 256; byte++) {
        uint32_t crc = byte;
        for (unsigned bit = 0; bit < 8; bit++) crc = (crc >> 1) ^ ((crc & 1) ? UINT32_C(0xedb88320) : 0);
        root->crc_table[byte] = crc;
    }
    for (unsigned byte=0;byte<256;byte++) {
        uint32_t crc=root->crc_table[byte];
        for (unsigned depth=0;depth<7;depth++) {
            crc=root->crc_table[crc&255]^(crc>>8);
            root->crc_wide[depth][byte]=crc;
        }
    }
    int length = snprintf(path, sizeof path, "%s/constants.bin", directory);
    if (length < 0 || (size_t)length >= sizeof path) { root_close(root); return NULL; }
    FILE *constants = fopen(path, "rb");
    if (!constants) { root_close(root); return NULL; }
    int valid = root_length(constants, root->constant_bytes) &&
        fread(root->constants, 1, root->constant_bytes, constants) == root->constant_bytes;
    if (fclose(constants)) valid = 0;
    if (!valid) { root_close(root); return NULL; }
    for (unsigned index = 0; index < root->palette_count; index++) {
        unsigned bits = root_u16(root->constants + index * 2);
        if ((bits & 0x7f80U) == 0x7f80U) { root_close(root); return NULL; }
        uint32_t widened = (uint32_t)bits << 16;
        float value;
        memcpy(&value, &widened, 4);
        root->values[index] = (double)value;
    }
    const unsigned char *maps = root->constants + root->palette_count * 2;
    const unsigned char *offsets = maps + root->map_count * 16;
    const unsigned char *refs = offsets + (root->template_count + 1) * 2;
    for (unsigned index = 0; index < root->map_count * 16; index++)
        if (maps[index] != 255 && maps[index] >= root->palette_count) valid = 0;
    if (root_u16(offsets) || root_u16(offsets + root->template_count * 2) != root->reference_count) valid = 0;
    for (unsigned index = 0; index < root->template_count; index++) {
        unsigned begin = root_u16(offsets + index * 2), end = root_u16(offsets + (index + 1) * 2);
        if (begin >= end || end > root->reference_count || end - begin > 256) valid = 0;
    }
    for (unsigned index = 0; index < root->reference_count; index++) if (root_u16(refs + index * 2) >= root->map_count) valid = 0;
    root->pairs = malloc((size_t)root->map_count * sizeof *root->pairs);
    if (!root->pairs) { root_close(root); return NULL; }
    for (unsigned entry = 0; entry < root->map_count; entry++) {
        const unsigned char *map = maps + entry * 16;
        for (unsigned byte = 0; byte < 256; byte++) {
            unsigned low = map[byte & 15], high = map[byte >> 4];
            root->pairs[entry][byte][0] = low < root->palette_count ? (float)root->values[low] : 0.0f;
            root->pairs[entry][byte][1] = high < root->palette_count ? (float)root->values[high] : 0.0f;
        }
    }
    length = snprintf(path, sizeof path, "%s/experts.bin", directory);
    if (!valid || length < 0 || (size_t)length >= sizeof path) { root_close(root); return NULL; }
    root->file = fopen(path, "rb");
    unsigned char header[32];
    if (!root->file || fread(header, 1, sizeof header, root->file) != sizeof header ||
        memcmp(header, "K3MAPS01", 8) || root_u64(header + 16) != ROOT_INDEX_BYTES ||
        root_u32(header + 8) != root_crc(root, root->constants, root->constant_bytes)) { root_close(root); return NULL; }
    root->index = malloc(ROOT_INDEX_BYTES);
    root->offsets = malloc((ROOT_BLOCKS + 1) * sizeof *root->offsets);
    if (!root->index || !root->offsets || fread(root->index, 1, ROOT_INDEX_BYTES, root->file) != ROOT_INDEX_BYTES ||
        root_crc(root, root->index, ROOT_INDEX_BYTES) != root_u32(header + 12)) { root_close(root); return NULL; }
    for (unsigned matrix = 0; matrix < ROOT_MATRICES; matrix++)
        if (root_u16(root->index + matrix * 2) >= root->template_count) valid = 0;
    root->offsets[0] = 32 + ROOT_INDEX_BYTES;
    for (unsigned block = 0; block < ROOT_BLOCKS; block++) {
        unsigned bytes = root_u32(root->index + ROOT_MATRICES * 2 + block * 8);
        if (!bytes || bytes > ROOT_COMPRESSED) valid = 0;
        root->offsets[block + 1] = root->offsets[block] + bytes;
    }
    int direct_length = snprintf(path, sizeof path, "%s/experts.direct", directory);
    if (direct_length > 0 && (size_t)direct_length < sizeof path) {
        int handle = open(path, O_RDONLY);
        if (handle >= 0) {
            struct stat direct_info;
            size_t bytes = (size_t)896 * ROOT_EXPERT_RAW;
            if (!fstat(handle, &direct_info) && (size_t)direct_info.st_size == bytes) {
                void *mapped = mmap(NULL, bytes, PROT_READ, MAP_SHARED, handle, 0);
                if (mapped != MAP_FAILED) { root->direct = mapped; root->direct_bytes = bytes; }
            }
            /* Held open rather than closed: root_pull reads the experts through it.
               O_DIRECT lands an expert straight in the arena; the page-cache copy is
               what holds the pread path near 6 GB/s on an array that fio runs at 13.5.
               It also means these pages never become resident, so the mincore path
               stops short-cutting warm requests -- hence opt-in rather than default. */
            if (root->direct) {
                const char *direct_io = getenv("K3_EXPERT_DIRECT");
                if (direct_io && !strcmp(direct_io, "1")) {
                    int flags = fcntl(handle, F_GETFL);
                    if (flags >= 0) (void)fcntl(handle, F_SETFL, flags | O_DIRECT);
                }
                root->direct_fd = handle;
            } else close(handle);
        }
    }
    if (!valid || root->offsets[ROOT_BLOCKS] - root->offsets[0] != root_u64(header + 24) ||
        (!root->direct && !root_length(root->file, root->offsets[ROOT_BLOCKS]))) { root_close(root); return NULL; }
    return root;
}

#include "root-validation.h"

static const unsigned char *root_arena_find(unsigned expert);

static int root_block(Root *root, RootScratch *scratch, unsigned expert, unsigned matrix, unsigned block)
{
    unsigned width = matrix == 2 ? 3072 : 3584, blocks = matrix == 2 ? 56 : 48;
    if (!root || !scratch || expert >= 896 || matrix >= 3 || block >= blocks) return 0;
    unsigned ordinal = expert * 152 + matrix * 48 + block;
    size_t bytes = (size_t)(root->offsets[ordinal + 1] - root->offsets[ordinal]);
    ROOT_PHASE_BEGIN(read_started);
    const unsigned char *compressed=scratch->compressed;
    if (root->prefetched && root->prefetched_expert==expert) {
        uint64_t offset=root->offsets[ordinal]-root->offsets[expert*152];
        if (offset>root->prefetched_bytes || bytes>root->prefetched_bytes-offset) return 0;
        compressed=root->prefetched+(size_t)offset;
    } else if (!root_read_at(root->file, scratch->compressed, bytes, root->offsets[ordinal])) return 0;
    ROOT_PHASE_END(read_started,0);
    ROOT_PHASE_BEGIN(decode_started);
    if (!decode_zlib(compressed, bytes, scratch->raw, 64 * width / 2 + 64 * width / 32)) return 0;
    ROOT_PHASE_END(decode_started,1);
    ROOT_PHASE_BEGIN(check_started);
    const unsigned char *maps = root->constants + root->palette_count * 2;
    const unsigned char *offsets = maps + root->map_count * 16;
    const unsigned char *refs = offsets + (root->template_count + 1) * 2;
    unsigned template_id = root_u16(root->index + (expert * 3 + matrix) * 2);
    unsigned begin = root_u16(offsets + template_id * 2), end = root_u16(offsets + (template_id + 1) * 2);
    uint32_t crc = UINT32_MAX;
    for (unsigned group = 0; group < 64 * width / 32; group++) {
        unsigned selector = scratch->raw[64 * width / 2 + group];
        if (selector >= end - begin) return 0;
        const unsigned char *map = maps + root_u16(refs + (begin + selector) * 2) * 16;
        if (!root_validate_group(root,scratch->raw+group*16,map,&crc)) return 0;
    }
    int valid=(crc ^ UINT32_MAX) == root_u32(root->index + ROOT_MATRICES * 2 + ordinal * 8 + 4);
    ROOT_PHASE_END(check_started,2);
    return valid;
}

static int root_project(Root *root, RootScratch *scratch, unsigned expert, unsigned matrix,
    const float *input, float *output)
{
    if (!root || !scratch || !input || !output || expert >= 896 || matrix >= 3) return 0;
    unsigned width = matrix == 2 ? 3072 : 3584, rows = matrix == 2 ? 3584 : 3072;
    const unsigned char *maps = root->constants + root->palette_count * 2;
    const unsigned char *offsets = maps + root->map_count * 16;
    const unsigned char *refs = offsets + (root->template_count + 1) * 2;
    unsigned template_id = root_u16(root->index + (expert * 3 + matrix) * 2);
    unsigned begin = root_u16(offsets + template_id * 2);
    int valid = 1;
    RootScratch *pool = scratch;
    double converted[3584];
    for (unsigned coordinate = 0; coordinate < width; coordinate++) converted[coordinate] = (double)input[coordinate];
#pragma omp parallel for schedule(static) reduction(&:valid)
    for (unsigned block = 0; block < rows / 64; block++) {
        RootScratch *scratch = pool + omp_get_thread_num();
        const unsigned char *data;
        const unsigned char *held = root_arena_find(expert);
        if (held) data = held +
            (matrix < 2 ? (size_t)(matrix * 48 + block) * ROOT_RAW
                        : (size_t)96 * ROOT_RAW + (size_t)block * ROOT_DOWN_RAW);
        else if (root->direct) data = root->direct + (size_t)expert * ROOT_EXPERT_RAW +
            (matrix < 2 ? (size_t)(matrix * 48 + block) * ROOT_RAW
                        : (size_t)96 * ROOT_RAW + (size_t)block * ROOT_DOWN_RAW);
        else if (root_block(root, scratch, expert, matrix, block)) data = scratch->raw;
        else { valid = 0; continue; }
        ROOT_PHASE_BEGIN(math_started);
        for (unsigned row = 0; row < 64; row++) {
            const unsigned char *codes = data + row * width / 2;
            const unsigned char *selectors = data + 64 * width / 2 + row * width / 32;
            double sums[4];
#if defined(__AVX2__)
            __m256d lane0 = _mm256_setzero_pd(), lane4 = _mm256_setzero_pd();
            __m256d lane8 = _mm256_setzero_pd(), lane12 = _mm256_setzero_pd();
            for (unsigned coordinate = 0; coordinate < width; coordinate += 16) {
                const float (*pair)[2] = root->pairs[root_u16(refs + (begin + selectors[coordinate / 32]) * 2)];
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
                const float (*pair)[2] = root->pairs[root_u16(refs + (begin + selectors[coordinate / 32]) * 2)];
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
        ROOT_PHASE_END(math_started,3);
    }
    return valid;
}

enum { ROOT_MAX_ROWS = 8 };

static unsigned long long root_pull_bytes;
static double root_pull_seconds;

/* live-root.h is included before now_s() is defined, so it keeps its own clock. */
static double root_now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* The arithmetic reads from here, not from the file mapping. Faulting 25 GB in
   through a file-backed mmap costs about six million minor faults, which no
   amount of readahead removes: measured cold the expert projection takes 22.4 s
   against 1.3 s once resident. An anonymous arena is huge-page eligible, so the
   same bytes cost a few thousand faults, and the reads that fill it are large and
   concurrent, which is the pattern fio measures at 13.5 GB/s. */
enum { ROOT_ARENA_MAX = 160 };
static unsigned char *root_arena;
static size_t root_arena_bytes;
static unsigned root_arena_expert[ROOT_ARENA_MAX];
static unsigned root_arena_count;
static unsigned char *root_pull_data[ROOT_ARENA_MAX];   /* where each entry's bytes live */

/* ----------------------------------------------- resident per-layer experts
   A token visits a layer once, takes 16 experts, and does not come back until
   the next token, 92 layers and ~1500 other keys later. One shared pool is
   therefore swept clean before the reuse arrives, which is why a global cache
   measured the same hit rate at every capacity. Giving each layer its own set
   removes the competition: the only traffic through layer L's set is layer L's.
   Held bytes are projected in place, never copied, because the weight sweep
   already runs near RAM bandwidth. */
static unsigned root_cache_capacity, root_cache_layers;
static unsigned char *root_cache_store;
static unsigned *root_cache_expert;
static unsigned long long *root_cache_stamp;
static unsigned char *root_cache_busy;
static unsigned root_cache_slot[ROOT_ARENA_MAX];
static unsigned long long root_cache_clock, root_cache_hits, root_cache_misses;

static int root_cache_open(unsigned layers, unsigned capacity)
{
    if (!layers || !capacity) return 0;
    size_t slots = (size_t)layers * capacity;
    size_t bytes = slots * ROOT_EXPERT_RAW;
    size_t rounded = (bytes + (2u << 20) - 1) & ~(size_t)((2u << 20) - 1);
    root_cache_store = aligned_alloc(2u << 20, rounded);
    root_cache_expert = malloc(slots * sizeof *root_cache_expert);
    root_cache_stamp = calloc(slots, sizeof *root_cache_stamp);
    root_cache_busy = calloc(slots, 1);
    if (!root_cache_store || !root_cache_expert || !root_cache_stamp || !root_cache_busy) {
        free(root_cache_store); free(root_cache_expert);
        free(root_cache_stamp); free(root_cache_busy);
        root_cache_store = NULL; return 0;
    }
#ifdef MADV_HUGEPAGE
    (void)madvise(root_cache_store, rounded, MADV_HUGEPAGE);
#endif
    /* First touch of a fresh mapping costs a fault and a zero-fill of the whole
       huge page. Left alone that lands inside the first request and makes it
       slower than no cache at all, so it is paid here instead, once. */
    for (size_t at = 0; at < rounded; at += (size_t)(2u << 20)) root_cache_store[at] = 0;
    for (size_t slot = 0; slot < slots; slot++) root_cache_expert[slot] = 0xffffffffu;
    root_cache_layers = layers; root_cache_capacity = capacity;
    return 1;
}

static void root_cache_close(void)
{
    free(root_cache_store); free(root_cache_expert);
    free(root_cache_stamp); free(root_cache_busy);
    root_cache_store = NULL; root_cache_expert = NULL;
    root_cache_stamp = NULL; root_cache_busy = NULL;
    root_cache_capacity = root_cache_layers = 0;
}

static const unsigned char *root_arena_find(unsigned expert)
{
    for (unsigned slot = 0; slot < root_arena_count; slot++)
        if (root_arena_expert[slot] == expert)
            return root_pull_data[slot];
    return NULL;
}

/* A pread copies 17.5 MB whether or not the kernel already holds the pages, while
   the mapping costs nothing once resident. That is the whole difference between a
   cold run and a warm one, so each expert is asked before it is fetched. */
static int root_resident(const Root *root, unsigned expert)
{
    if (!root->direct || expert >= 896) return 0;
    unsigned char vec[(ROOT_EXPERT_RAW + 4095) / 4096];
    if (mincore((void *)(root->direct + (size_t)expert * ROOT_EXPERT_RAW),
                ROOT_EXPERT_RAW, vec)) return 0;
    size_t seen = 0, have = 0;
    for (size_t page = 0; page < sizeof vec; page += 64) { seen++; have += vec[page] & 1; }
    return seen && have * 10 >= seen * 9;
}

/* Reading and multiplying are otherwise strictly serialised: the pull is about
   8.9 s of a 15.5 s run while the arithmetic is 6.7 s, yet expert n's maths needs
   only expert n's bytes. Readers publish each slot as it lands so the arithmetic
   starts on the first expert while the rest are still arriving.
   These are dedicated threads, never OMP: an earlier attempt used an OMP region
   here and its team competed with the compute team, taking the expert bucket from
   4.3 s to 9.6 s. Readers sit blocked in pread and cost no CPU. */
enum { ROOT_READERS = 16 };
static pthread_t root_reader[ROOT_READERS];
static int root_readers_started;
static const Root *root_pull_root;
static unsigned root_pull_list[ROOT_ARENA_MAX];
static unsigned root_pull_count;
static volatile int root_arena_ready[ROOT_ARENA_MAX];
static int root_pull_fetch[ROOT_ARENA_MAX];
static unsigned root_pull_cursor;
static unsigned long long root_pull_batch;
static double root_pull_started;
static double root_wait_seconds;   /* compute blocked on bytes that have not landed */
static double root_drain_seconds;  /* reads still outstanding when the layer is done */
static pthread_mutex_t root_pull_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t root_pull_wake = PTHREAD_COND_INITIALIZER;

static void *root_reader_main(void *unused)
{
    (void)unused;
    unsigned long long seen = 0;
    for (;;) {
        pthread_mutex_lock(&root_pull_lock);
        while (root_pull_batch == seen) pthread_cond_wait(&root_pull_wake, &root_pull_lock);
        seen = root_pull_batch;
        pthread_mutex_unlock(&root_pull_lock);
        for (;;) {
            unsigned entry = __atomic_fetch_add(&root_pull_cursor, 1, __ATOMIC_ACQ_REL);
            if (entry >= root_pull_count) break;
            size_t got = 0;
            if (root_pull_fetch[entry]) {
                unsigned char *into = root_pull_data[entry];
                off_t at = (off_t)root_pull_list[entry] * ROOT_EXPERT_RAW;
                while (got < ROOT_EXPERT_RAW) {
                    ssize_t n = pread(root_pull_root->direct_fd, into + got,
                                      ROOT_EXPERT_RAW - got, at + (off_t)got);
                    if (n <= 0) break;
                    got += (size_t)n;
                }
                if (got == ROOT_EXPERT_RAW) {
                    root_arena_expert[entry] = root_pull_list[entry];
                    if (root_cache_slot[entry] != 0xffffffffu)
                        root_cache_expert[root_cache_slot[entry]] = root_pull_list[entry];
                }
                __atomic_fetch_add(&root_pull_bytes, got, __ATOMIC_RELAXED);
            }
            __atomic_store_n(&root_arena_ready[entry], 1, __ATOMIC_RELEASE);
        }
    }
    return NULL;
}

static void root_pull_begin(const Root *root, const unsigned *experts, unsigned count)
{
    root_arena_count = 0;
    if (!root || root->direct_fd < 0 || !experts || !count || count > ROOT_ARENA_MAX) return;
    /* The cache can only serve a batch narrower than one layer's set, otherwise a
       miss would have to evict an expert this same batch still needs. */
    unsigned held = (root_cache_store && root->layer < root_cache_layers &&
                     count <= root_cache_capacity) ? root_cache_capacity : 0;
    unsigned base = held ? root->layer * held : 0;
    if (!held) {
        size_t need = (size_t)count * ROOT_EXPERT_RAW;
        if (need > root_arena_bytes) {
            size_t rounded = (need + (2u << 20) - 1) & ~(size_t)((2u << 20) - 1);
            free(root_arena);
            root_arena = aligned_alloc(2u << 20, rounded);
            root_arena_bytes = root_arena ? rounded : 0;
#ifdef MADV_HUGEPAGE
            if (root_arena) (void)madvise(root_arena, rounded, MADV_HUGEPAGE);
#endif
        }
        if (!root_arena) return;
    }
    for (unsigned entry = 0; entry < count; entry++) {
        root_pull_list[entry] = experts[entry];
        root_arena_expert[entry] = 0xffffffffu;
        root_cache_slot[entry] = 0xffffffffu;
        root_pull_fetch[entry] = experts[entry] < 896 && !root_resident(root, experts[entry]);
        root_pull_data[entry] = held ? NULL : root_arena + (size_t)entry * ROOT_EXPERT_RAW;
        __atomic_store_n(&root_arena_ready[entry], 0, __ATOMIC_RELEASE);
    }
    if (held) {
        /* hits first, so a miss never evicts something this batch will ask for */
        for (unsigned entry = 0; entry < count; entry++) {
            if (!root_pull_fetch[entry]) continue;
            for (unsigned slot = 0; slot < held; slot++)
                if (root_cache_expert[base + slot] == experts[entry]) {
                    root_cache_busy[base + slot] = 1;
                    root_cache_stamp[base + slot] = ++root_cache_clock;
                    root_cache_slot[entry] = base + slot;
                    root_pull_data[entry] = root_cache_store + (size_t)(base + slot) * ROOT_EXPERT_RAW;
                    root_arena_expert[entry] = experts[entry];
                    root_pull_fetch[entry] = 0;
                    root_cache_hits++;
                    __atomic_store_n(&root_arena_ready[entry], 1, __ATOMIC_RELEASE);
                    break;
                }
        }
        for (unsigned entry = 0; entry < count; entry++) {
            if (!root_pull_fetch[entry]) continue;
            unsigned victim = held;
            for (unsigned slot = 0; slot < held; slot++) {
                if (root_cache_busy[base + slot]) continue;
                if (victim == held || root_cache_stamp[base + slot] < root_cache_stamp[base + victim])
                    victim = slot;
            }
            if (victim == held) { root_pull_fetch[entry] = 0; continue; }
            root_cache_busy[base + victim] = 1;
            root_cache_stamp[base + victim] = ++root_cache_clock;
            root_cache_expert[base + victim] = 0xffffffffu;
            root_cache_slot[entry] = base + victim;
            root_pull_data[entry] = root_cache_store + (size_t)(base + victim) * ROOT_EXPERT_RAW;
            root_cache_misses++;
        }
    }
    root_pull_root = root;
    root_pull_count = count;
    root_arena_count = count;
    root_pull_started = root_now_s();
    __atomic_store_n(&root_pull_cursor, 0, __ATOMIC_RELEASE);
    if (!root_readers_started) {
        for (unsigned reader = 0; reader < ROOT_READERS; reader++)
            if (pthread_create(&root_reader[reader], NULL, root_reader_main, NULL)) break;
        root_readers_started = 1;
    }
    pthread_mutex_lock(&root_pull_lock);
    root_pull_batch++;
    pthread_cond_broadcast(&root_pull_wake);
    pthread_mutex_unlock(&root_pull_lock);
}

static void root_arena_wait(unsigned slot)
{
    if (slot >= root_arena_count) return;
    if (__atomic_load_n(&root_arena_ready[slot], __ATOMIC_ACQUIRE)) return;
    double stall = root_now_s();
    while (!__atomic_load_n(&root_arena_ready[slot], __ATOMIC_ACQUIRE)) sched_yield();
    root_wait_seconds += root_now_s() - stall;
}

static void root_pull_end(void)
{
    double drain = root_now_s();
    double before = root_wait_seconds;
    for (unsigned slot = 0; slot < root_arena_count; slot++) root_arena_wait(slot);
    root_wait_seconds = before;
    for (unsigned slot = 0; slot < root_arena_count; slot++)
        if (root_cache_slot[slot] != 0xffffffffu) root_cache_busy[root_cache_slot[slot]] = 0;
    root_drain_seconds += root_now_s() - drain;
    root_pull_seconds += root_now_s() - root_pull_started;
}


static int root_project_rows(Root *root, RootScratch *scratch, unsigned expert, unsigned matrix,
    const float *const *inputs, float *const *outputs, unsigned count)
{
    if (!root || !scratch || !inputs || !outputs || !count || count > ROOT_MAX_ROWS ||
        expert >= 896 || matrix >= 3) return 0;
    unsigned width = matrix == 2 ? 3072 : 3584, rows = matrix == 2 ? 3584 : 3072;
    const unsigned char *maps = root->constants + root->palette_count * 2;
    const unsigned char *offsets = maps + root->map_count * 16;
    const unsigned char *refs = offsets + (root->template_count + 1) * 2;
    unsigned template_id = root_u16(root->index + (expert * 3 + matrix) * 2);
    unsigned begin = root_u16(offsets + template_id * 2);
    int valid = 1;
    RootScratch *pool = scratch;
    static double converted[ROOT_MAX_ROWS][3584];
    for (unsigned entry = 0; entry < count; entry++) {
        if (!inputs[entry] || !outputs[entry]) return 0;
        for (unsigned coordinate = 0; coordinate < width; coordinate++)
            converted[entry][coordinate] = (double)inputs[entry][coordinate];
    }
#pragma omp parallel for schedule(static) reduction(&:valid)
    for (unsigned block = 0; block < rows / 64; block++) {
        RootScratch *scratch = pool + omp_get_thread_num();
        const unsigned char *data;
        const unsigned char *held = root_arena_find(expert);
        if (held) data = held +
            (matrix < 2 ? (size_t)(matrix * 48 + block) * ROOT_RAW
                        : (size_t)96 * ROOT_RAW + (size_t)block * ROOT_DOWN_RAW);
        else if (root->direct) data = root->direct + (size_t)expert * ROOT_EXPERT_RAW +
            (matrix < 2 ? (size_t)(matrix * 48 + block) * ROOT_RAW
                        : (size_t)96 * ROOT_RAW + (size_t)block * ROOT_DOWN_RAW);
        else if (root_block(root, scratch, expert, matrix, block)) data = scratch->raw;
        else { valid = 0; continue; }
        ROOT_PHASE_BEGIN(math_started);
        for (unsigned row = 0; row < 64; row++) {
            const unsigned char *codes = data + row * width / 2;
            const unsigned char *selectors = data + 64 * width / 2 + row * width / 32;
            double sums[ROOT_MAX_ROWS][4];
#if defined(__AVX2__)
            __m256d lane0[ROOT_MAX_ROWS], lane4[ROOT_MAX_ROWS];
            __m256d lane8[ROOT_MAX_ROWS], lane12[ROOT_MAX_ROWS];
            for (unsigned entry = 0; entry < count; entry++) {
                lane0[entry] = _mm256_setzero_pd(); lane4[entry] = _mm256_setzero_pd();
                lane8[entry] = _mm256_setzero_pd(); lane12[entry] = _mm256_setzero_pd();
            }
            for (unsigned coordinate = 0; coordinate < width; coordinate += 16) {
                const float (*pair)[2] = root->pairs[root_u16(refs + (begin + selectors[coordinate / 32]) * 2)];
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
            double lanes[ROOT_MAX_ROWS][16] = {{0}};
            for (unsigned coordinate = 0; coordinate < width; coordinate += 16) {
                const float (*pair)[2] = root->pairs[root_u16(refs + (begin + selectors[coordinate / 32]) * 2)];
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
        ROOT_PHASE_END(math_started,3);
    }
    return valid;
}
#endif