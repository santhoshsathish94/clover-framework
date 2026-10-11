#define _FILE_OFFSET_BITS 64
#define _POSIX_C_SOURCE 200809L
#include <float.h>
#include <fenv.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "live-root.h"
#include "stage.h"
#include "trace.h"
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

enum {
    TRANSFORMER_WIDTH = 7168, TRANSFORMER_ROWS = 12288, TRANSFORMER_HEAD = 128,
    TRANSFORMER_HEADS = 96, TRANSFORMER_LATENT = 3584, TRANSFORMER_EXPERT = 3072,
    TRANSFORMER_LAYER = 1,
    TRANSFORMER_SHARED = 6144, TRANSFORMER_RECORDS = 40,
    TRANSFORMER_QKV_PREFIX = 64 + 24 + TRANSFORMER_WIDTH * 4 + 1024,
    TRANSFORMER_QKV_COMPONENT = TRANSFORMER_WIDTH + 20
};

typedef struct {
    uint32_t id, group, kind, rows, columns, count;
    uint64_t offset, length;
    unsigned char *data;
} TransformerRecord;

typedef struct {
    unsigned char *qkv;
    TransformerRecord records[TRANSFORMER_RECORDS];
    float palette[256];
    size_t qkv_bytes;
    unsigned char *group_map[3];
    size_t group_bytes[3];
    Root *root;
} Transformer;

typedef struct {
    float history[3][TRANSFORMER_ROWS][3];
    float recurrent[TRANSFORMER_HEADS][TRANSFORMER_HEAD][TRANSFORMER_HEAD];
    float snapshot[TRANSFORMER_WIDTH], normalized[TRANSFORMER_WIDTH];
    float qkv[3][TRANSFORMER_ROWS], beta[TRANSFORMER_HEADS], decay_hidden[TRANSFORMER_HEAD];
    float decay[TRANSFORMER_ROWS], gate[TRANSFORMER_ROWS], attention[TRANSFORMER_ROWS];
    float residual[TRANSFORMER_WIDTH], aggregate[TRANSFORMER_WIDTH], postnorm[TRANSFORMER_WIDTH];
    float incoming[TRANSFORMER_WIDTH], latent[TRANSFORMER_LATENT], mixture[TRANSFORMER_LATENT];
    float expert_gate[TRANSFORMER_EXPERT], expert_up[TRANSFORMER_EXPERT], expert_down[TRANSFORMER_LATENT];
    float latent_norm[TRANSFORMER_LATENT], routed[TRANSFORMER_WIDTH];
    float shared_gate[TRANSFORMER_SHARED], shared_up[TRANSFORMER_SHARED], shared_output[TRANSFORMER_WIDTH];
    unsigned selected[16];
    float weights[16];
    RootScratch *root_scratch;
    const Transformer *owner;
    size_t positions;
    int stage;
    int failed;
} TransformerSequence;

typedef struct {
    unsigned id, group, kind, rows, columns;
} TransformerLayout;

static const TransformerLayout transformer_layout[] = {
    {0, 0, 2, 7168, 1}, {1, 0, 2, 7168, 1}, {2, 0, 2, 7168, 1},
    {3, 0, 2, 7168, 1}, {4, 0, 2, 7168, 1}, {5, 0, 2, 7168, 1},
    {6, 0, 1, 12288, 7168}, {7, 0, 1, 7168, 12288},
    {8, 1, 1, 12288, 7168}, {9, 1, 1, 12288, 7168},
    {10, 1, 1, 12288, 7168}, {11, 1, 1, 96, 7168},
    {12, 1, 1, 128, 7168}, {13, 1, 1, 12288, 128},
    {14, 1, 2, 49152, 1}, {15, 1, 2, 49152, 1}, {16, 1, 2, 49152, 1},
    {17, 1, 2, 128, 1}, {18, 1, 2, 12288, 1}, {19, 1, 2, 128, 1},
    {29, 2, 3, 896, 7168}, {30, 2, 2, 896, 1},
    {31, 2, 1, 3584, 7168}, {32, 2, 1, 7168, 3584},
    {33, 2, 2, 3584, 1}, {34, 2, 1, 6144, 7168},
    {35, 2, 1, 6144, 7168}, {36, 2, 1, 7168, 6144}, {37, 0, 2, 7168, 1},
    {38, 0, 2, 7168, 1}, {39, 1, 2, 96, 1}
};

static uint32_t transformer_u32(const unsigned char *bytes)
{
    return (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) |
        ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
}

static uint64_t transformer_u64(const unsigned char *bytes)
{
    return transformer_u32(bytes) | ((uint64_t)transformer_u32(bytes + 4) << 32);
}

static float transformer_f32(const unsigned char *bytes)
{
    uint32_t bits = transformer_u32(bytes);
    float value;
    memcpy(&value, &bits, sizeof value);
    return value;
}

static int transformer_path(char *output, size_t capacity, const char *directory, const char *name)
{
    int length = snprintf(output, capacity, "%s/%s", directory, name);
    return length >= 0 && (size_t)length < capacity;
}

static FILE *transformer_open_sized(const char *path, uint64_t expected)
{
    FILE *file = fopen(path, "rb");
    if (!file) return NULL;
    if (expected > LONG_MAX || fseek(file, 0, SEEK_END)) { fclose(file); return NULL; }
    long length = ftell(file);
    if (length < 0 || (uint64_t)length != expected || fseek(file, 0, SEEK_SET)) {
        fclose(file);
        return NULL;
    }
    return file;
}

static int transformer_read(FILE *file, uint64_t offset, size_t length, unsigned char **output)
{
    if (offset > LONG_MAX || fseek(file, (long)offset, SEEK_SET)) return 0;
    unsigned char *data = malloc(length);
    if (!data) return 0;
    if (fread(data, 1, length, file) != length) { free(data); return 0; }
    *output = data;
    return 1;
}

/* Weights are mapped, not copied, so the bytes live once in the page cache and stay
   reclaimable. Copying them made 93 resident stages hold 54 GB of unreclaimable heap.
   The trunk is read at every layer of every token, so it is also locked resident unless
   CLOVER_LOCK_TRUNK=0; the experts stay unlocked and let the page cache keep the hot ones. */
static const unsigned char *transformer_map(const char *path, size_t expected)
{
    int handle = open(path, O_RDONLY);
    if (handle < 0) return NULL;
    struct stat info;
    const unsigned char *mapped = NULL;
    if (!fstat(handle, &info) && (size_t)info.st_size == expected) {
        void *address = mmap(NULL, expected, PROT_READ, MAP_SHARED, handle, 0);
        if (address != MAP_FAILED) {
            const char *lock = getenv("CLOVER_LOCK_TRUNK");
            if ((!lock || strcmp(lock, "0")) && mlock(address, expected))
                fprintf(stderr, "trunk not locked (%s): %s\n", path, strerror(errno));
            mapped = address;
        }
    }
    close(handle);
    return mapped;
}

static int transformer_needs(unsigned id)
{
    switch (id) {
    case 5: case 6: case 7: case 11: case 12: case 13: case 18:
    case 19: case 29: case 30: case 31: case 32: case 33: case 34:
    case 35: case 36: case 37: case 38: case 39: return 1;
    default: return 0;
    }
}

static int transformer_load_qkv(Transformer *transformer, const char *path)
{
    const size_t bytes = TRANSFORMER_QKV_PREFIX + (size_t)TRANSFORMER_ROWS * 3 * TRANSFORMER_QKV_COMPONENT;
    FILE *file = transformer_open_sized(path, bytes);
    if (!file) return 0;
    if (fclose(file)) return 0;
    transformer->qkv = (unsigned char *)transformer_map(path, bytes);
    if (!transformer->qkv) return 0;
    transformer->qkv_bytes = bytes;
    const unsigned char *data = transformer->qkv;
    const uint32_t fields[] = {1, 1, TRANSFORMER_WIDTH, TRANSFORMER_ROWS, TRANSFORMER_HEAD, 3, 4, 6, TRANSFORMER_QKV_COMPONENT, 0};
    if (memcmp(data, "K3QKV001", 8)) return 0;
    for (size_t field = 0; field < sizeof fields / sizeof *fields; field++)
        if (transformer_u32(data + 8 + field * 4) != fields[field]) return 0;
    if (transformer_f32(data + 48) != 1e-5f || transformer_f32(data + 52) != 1e-6f || transformer_u64(data + 56) != bytes) return 0;
    for (unsigned stage = 0; stage < 6; stage++) if (transformer_u32(data + 64 + stage * 4) != stage + 1) return 0;
    for (unsigned coordinate = 0; coordinate < TRANSFORMER_WIDTH; coordinate++)
        if (!isfinite(transformer_f32(data + 88 + coordinate * 4))) return 0;
    for (unsigned code = 0; code < 256; code++) {
        float expected = (float)(code < 128 ? (int)code : (int)code - 256);
        float stored = transformer_f32(data + 88 + TRANSFORMER_WIDTH * 4 + code * 4);
        if (memcmp(&stored, &expected, sizeof stored)) return 0;
        transformer->palette[code] = stored;
    }
    for (unsigned row = 0; row < TRANSFORMER_ROWS; row++)
        for (unsigned component = 0; component < 3; component++) {
            const unsigned char *record = data + TRANSFORMER_QKV_PREFIX + ((size_t)row * 3 + component) * TRANSFORMER_QKV_COMPONENT;
            for (unsigned field = 0; field < 5; field++) if (!isfinite(transformer_f32(record + field * 4))) return 0;
        }
    return 1;
}

static int transformer_load_trunk(Transformer *transformer, const char *directory)
{
    const size_t count = sizeof transformer_layout / sizeof *transformer_layout;
    const size_t bytes = 1168 + count * 72;
    char path[4096];
    unsigned char *index = NULL;
    if (!transformer_path(path, sizeof path, directory, "index.bin")) return 0;
    FILE *file = transformer_open_sized(path, bytes);
    if (!file) return 0;
    int loaded = transformer_read(file, 0, bytes, &index);
    if (fclose(file)) loaded = 0;
    if (!loaded) { free(index); return 0; }
    int valid = !memcmp(index, "K3TRK001", 8) && transformer_u32(index + 8) == 1 &&
        transformer_u32(index + 12) == 1 && transformer_u32(index + 16) == count && !transformer_u32(index + 20);
    uint64_t lengths[3], ends[3] = {0};
    for (unsigned group = 0; group < 3; group++) lengths[group] = transformer_u64(index + 24 + group * 8);
    for (unsigned code = 0; code < 256; code++) {
        float value = transformer_f32(index + 144 + code * 4);
        if (memcmp(&value, &transformer->palette[code], sizeof value)) valid = 0;
    }
    for (size_t position = 0; valid && position < count; position++) {
        const unsigned char *encoded = index + 1168 + position * 72;
        const TransformerLayout *layout = &transformer_layout[position];
        TransformerRecord record = {transformer_u32(encoded), transformer_u32(encoded + 4), transformer_u32(encoded + 8),
            transformer_u32(encoded + 12), transformer_u32(encoded + 16), transformer_u32(encoded + 20),
            transformer_u64(encoded + 24), transformer_u64(encoded + 32), NULL};
        uint64_t elements = (uint64_t)layout->rows * layout->columns;
        uint64_t wanted = layout->kind == 1 ? layout->rows * 4ULL + elements : elements * 4;
        if (record.id != layout->id || record.group != layout->group || record.kind != layout->kind ||
            record.rows != layout->rows || record.columns != layout->columns || record.count != elements ||
            record.length != wanted || record.offset != ends[layout->group] ||
            record.offset > lengths[layout->group] || record.length > lengths[layout->group] - record.offset ||
            record.length > SIZE_MAX) { valid = 0; break; }
        ends[record.group] += record.length;
        transformer->records[record.id] = record;
    }
    for (unsigned group = 0; group < 3; group++) if (ends[group] != lengths[group]) valid = 0;
    free(index);
    if (!valid) return 0;
    const char *names[] = {"common.bin", "attention.bin", "routed-shared.bin"};
    for (unsigned group = 0; group < 3; group++) {
        if (!transformer_path(path, sizeof path, directory, names[group])) return 0;
        file = transformer_open_sized(path, lengths[group]);
        if (!file) return 0;
        if (fclose(file)) return 0;
        file = NULL;
        transformer->group_map[group] = (unsigned char *)transformer_map(path, (size_t)lengths[group]);
        if (!transformer->group_map[group]) return 0;
        transformer->group_bytes[group] = (size_t)lengths[group];
        for (unsigned id = 0; valid && id < TRANSFORMER_RECORDS; id++) {
            TransformerRecord *record = &transformer->records[id];
            if (!transformer_needs(id) || record->group != group) continue;
            record->data = transformer->group_map[group] + record->offset;
            unsigned floats = record->kind == 1 ? record->rows : record->count;
            for (unsigned coordinate = 0; coordinate < floats; coordinate++)
                if (!isfinite(transformer_f32(record->data + (size_t)coordinate * 4))) { valid = 0; break; }
        }
        if (!valid) return 0;
    }
    return 1;
}

void transformer_close(Transformer *transformer)
{
    if (!transformer) return;
    for (unsigned group = 0; group < 3; group++)
        if (transformer->group_map[group]) munmap(transformer->group_map[group], transformer->group_bytes[group]);
    root_close(transformer->root);
    if (transformer->qkv) munmap(transformer->qkv, transformer->qkv_bytes);
    free(transformer);
}

int transformer_open(const char *dataset, Transformer **result)
{
    if (!result) return 0;
    *result = NULL;
    char qkv_path[4096], trunk_directory[4096], root_directory[4096];
    if (!dataset || !transformer_path(qkv_path, sizeof qkv_path, dataset, "operators/qkv-all/layer-1/operator.bin") ||
        !transformer_path(trunk_directory, sizeof trunk_directory, dataset, "trunk-1") ||
        !transformer_path(root_directory, sizeof root_directory, dataset, "root-1") ||
        sizeof(float) != 4 || sizeof(double) != 8 ||
        FLT_RADIX != 2 || FLT_MANT_DIG != 24 || DBL_MANT_DIG != 53) return 0;
    Transformer *transformer = calloc(1, sizeof *transformer);
    if (!transformer) return 0;
    if (!transformer_load_qkv(transformer, qkv_path) || !transformer_load_trunk(transformer, trunk_directory) || !(transformer->root = root_open(root_directory, TRANSFORMER_LAYER))) {
        transformer_close(transformer);
        return 0;
    }
    *result = transformer;
    return 1;
}

TransformerSequence *transformer_sequence_create(const Transformer *transformer)
{
    if (!transformer) return NULL;
    TransformerSequence *sequence = calloc(1, sizeof *sequence);
    if (!sequence) return NULL;
    /* One scratch per thread: the reader indexes this pool by OpenMP thread number. */
    sequence->root_scratch = calloc((size_t)omp_get_max_threads(), sizeof *sequence->root_scratch);
    if (!sequence->root_scratch) { free(sequence); return NULL; }
    sequence->owner = transformer;
    return sequence;
}

void transformer_sequence_reset(TransformerSequence *sequence)
{
    if (!sequence) return;
    const Transformer *owner = sequence->owner;
    RootScratch *scratch = sequence->root_scratch;
    memset(sequence, 0, sizeof *sequence);
    sequence->owner = owner;
    sequence->root_scratch = scratch;
}

void transformer_sequence_close(TransformerSequence *sequence)
{
    if (sequence) free(sequence->root_scratch);
    free(sequence);
}

static int transformer_finite(const float *values, size_t count)
{
    for (size_t coordinate = 0; coordinate < count; coordinate++)
        if (!isfinite(values[coordinate])) return 0;
    return 1;
}

static float transformer_inverse_rms(const float *input, unsigned width)
{
    double squares = 0.0;
    for (unsigned coordinate = 0; coordinate < width; coordinate++)
        squares += (double)input[coordinate] * (double)input[coordinate];
    return (float)(1.0 / sqrt(squares / (double)width + (double)1e-5f));
}

static void transformer_normalize(float *output, const float *input, const unsigned char *gains, unsigned width)
{
    float inverse = transformer_inverse_rms(input, width);
    for (unsigned coordinate = 0; coordinate < width; coordinate++)
        output[coordinate] = (transformer_f32(gains + coordinate * 4) * input[coordinate]) * inverse;
}

static float transformer_project_row(const float *input, const unsigned char *ids,
    unsigned width, const float *palette, float scale)
{
    float lanes[16] = {0}, combined[8];
    for (unsigned coordinate = 0; coordinate < width; coordinate += 16)
        for (unsigned lane = 0; lane < 16; lane++)
            lanes[lane] = fmaf(palette[ids[coordinate + lane]], input[coordinate + lane], lanes[lane]);
    for (unsigned lane = 0; lane < 8; lane++) combined[lane] = lanes[lane] + lanes[lane + 8];
    float sum0 = combined[0] + combined[4], sum1 = combined[1] + combined[5];
    float sum2 = combined[2] + combined[6], sum3 = combined[3] + combined[7];
    return ((sum0 + sum2) + (sum1 + sum3)) * scale;
}

static void transformer_project(const Transformer *transformer, unsigned id, const float *input, float *output)
{
    const TransformerRecord *record = &transformer->records[id];
    const unsigned char *ids = record->data + (size_t)record->rows * 4;
#pragma omp parallel for schedule(static)
    for (unsigned row = 0; row < record->rows; row++)
        output[row] = transformer_project_row(input, ids + (size_t)row * record->columns,
            record->columns, transformer->palette, transformer_f32(record->data + row * 4));
}

static float transformer_sigmoid(float value)
{
    return 1.0f / (1.0f + expf(-value));
}

static float transformer_convolve(float raw, const unsigned char *taps, float *history)
{
    float value = transformer_f32(taps) * raw;
    for (unsigned previous = 0; previous < 3; previous++)
        value = value + transformer_f32(taps + (previous + 1) * 4) * history[previous];
    history[0] = history[1];
    history[1] = history[2];
    history[2] = raw;
    return value;
}

static void transformer_l2_heads(float *values)
{
    for (unsigned head = 0; head < TRANSFORMER_HEADS; head++) {
        float *block = values + head * TRANSFORMER_HEAD;
        double squares = 0.0;
        for (unsigned coordinate = 0; coordinate < TRANSFORMER_HEAD; coordinate++)
            squares += (double)block[coordinate] * (double)block[coordinate];
        float inverse = (float)(1.0 / sqrt(squares + (double)1e-6f));
        for (unsigned coordinate = 0; coordinate < TRANSFORMER_HEAD; coordinate++) block[coordinate] *= inverse;
    }
}

static void transformer_qkv(const Transformer *transformer, TransformerSequence *sequence)
{
    /* Each row owns its output and its own convolution history, so rows are independent. */
#pragma omp parallel for schedule(static)
    for (unsigned row = 0; row < TRANSFORMER_ROWS; row++)
        for (unsigned component = 0; component < 3; component++) {
            const unsigned char *record = transformer->qkv + TRANSFORMER_QKV_PREFIX +
                ((size_t)row * 3 + component) * TRANSFORMER_QKV_COMPONENT;
            float raw = transformer_project_row(sequence->normalized, record + 20,
                TRANSFORMER_WIDTH, transformer->palette, transformer_f32(record));
            float convolved = transformer_convolve(raw, record + 4, sequence->history[component][row]);
            sequence->qkv[component][row] = convolved * transformer_sigmoid(convolved);
        }
    transformer_l2_heads(sequence->qkv[0]);
    transformer_l2_heads(sequence->qkv[1]);
}

static void transformer_decay(const Transformer *transformer, TransformerSequence *sequence)
{
    transformer_project(transformer, 11, sequence->normalized, sequence->beta);
    transformer_project(transformer, 12, sequence->normalized, sequence->decay_hidden);
    transformer_project(transformer, 13, sequence->decay_hidden, sequence->decay);
    for (unsigned head = 0; head < TRANSFORMER_HEADS; head++) {
        sequence->beta[head] = transformer_sigmoid(sequence->beta[head]);
        float exponent = transformer_f32(transformer->records[39].data + head * 4);
        for (unsigned coordinate = 0; coordinate < TRANSFORMER_HEAD; coordinate++) {
            unsigned index = head * TRANSFORMER_HEAD + coordinate;
            float value = exponent * (sequence->decay[index] + transformer_f32(transformer->records[18].data + index * 4));
            sequence->decay[index] = expf(-5.0f * transformer_sigmoid(value));
        }
    }
}

static void transformer_update_attention(TransformerSequence *sequence)
{
    const float query_scale = 1.0f / sqrtf(128.0f);
    for (unsigned head = 0; head < TRANSFORMER_HEADS; head++) {
        float prediction[TRANSFORMER_HEAD] = {0};
        float *output = sequence->attention + head * TRANSFORMER_HEAD;
        const float *query = sequence->qkv[0] + head * TRANSFORMER_HEAD;
        const float *key = sequence->qkv[1] + head * TRANSFORMER_HEAD;
        const float *value = sequence->qkv[2] + head * TRANSFORMER_HEAD;
        const float *decay = sequence->decay + head * TRANSFORMER_HEAD;
        for (unsigned row = 0; row < TRANSFORMER_HEAD; row++)
            for (unsigned column = 0; column < TRANSFORMER_HEAD; column++) {
                float state = decay[row] * sequence->recurrent[head][row][column];
                sequence->recurrent[head][row][column] = state;
                prediction[column] = prediction[column] + state * key[row];
            }
        memset(output, 0, TRANSFORMER_HEAD * sizeof *output);
        for (unsigned row = 0; row < TRANSFORMER_HEAD; row++) {
            float key_beta = key[row] * sequence->beta[head];
            float scaled_query = query[row] * query_scale;
            for (unsigned column = 0; column < TRANSFORMER_HEAD; column++) {
                float state = sequence->recurrent[head][row][column] + key_beta * (value[column] - prediction[column]);
                sequence->recurrent[head][row][column] = state;
                output[column] = output[column] + state * scaled_query;
            }
        }
    }
}

static void transformer_attention_output(const Transformer *transformer, TransformerSequence *sequence)
{
    transformer_project(transformer, 6, sequence->normalized, sequence->gate);
    for (unsigned head = 0; head < TRANSFORMER_HEADS; head++) {
        float *block = sequence->attention + head * TRANSFORMER_HEAD;
        transformer_normalize(block, block, transformer->records[19].data, TRANSFORMER_HEAD);
    }
    for (unsigned coordinate = 0; coordinate < TRANSFORMER_ROWS; coordinate++)
        sequence->gate[coordinate] = sequence->attention[coordinate] * transformer_sigmoid(sequence->gate[coordinate]);
    transformer_project(transformer, 7, sequence->gate, sequence->residual);
}

/* A comma list of numbers and low-high ranges, or "all". */
static int transformer_listed(const char *list, int value)
{
    if (!strcmp(list, "all")) return 1;
    while (*list) {
        char *after;
        long low = strtol(list, &after, 10), high = low;
        if (after == list) return 0;
        if (*after == '-') { list = after + 1; high = strtol(list, &after, 10); }
        if (value >= low && value <= high) return 1;
        list = *after == ',' ? after + 1 : after;
    }
    return 0;
}

static int transformer_fold_residual(unsigned fold)
{
    static int decided, layer_ok, stage3, stage21;
    if (!decided) {
        const char *setting = getenv("CLOVER_FOLD");
        const char *layers = getenv("CLOVER_FOLD_LAYERS");
        const char *stages = getenv("CLOVER_FOLD_STAGES");
        if (!layers || !*layers)
            layers = setting && !strcmp(setting, "residual") ? "all" : "";
        layer_ok = *layers && transformer_listed(layers, TRANSFORMER_LAYER);
        stages = stages && *stages ? stages : "3,21";
        stage3 = transformer_listed(stages, 3);
        stage21 = transformer_listed(stages, 21);
        decided = 1;
    }
    return layer_ok && (fold == 37 ? stage3 : stage21);
}

/* Layer 1 holds one slot, the server's, passed beside the residual rather than in an
   array. Dropping it leaves the residual alone, which is the same thing the residual
   fold does. */
static int transformer_dropped(unsigned slot)
{
    static int decided;
    static const char *list;
    if (!decided) {
        list = getenv("CLOVER_DROP_SLOTS");
        if (!list) list = "";
        decided = 1;
    }
    return *list && transformer_listed(list, (int)slot);
}

static void transformer_aggregate(const Transformer *transformer, TransformerSequence *sequence, unsigned fold)
{
    const float *sources[] = {sequence->snapshot, sequence->residual};
    if (transformer_fold_residual(fold) || transformer_dropped(0)) {
        memcpy(sequence->aggregate, sequence->residual, TRANSFORMER_WIDTH * sizeof(float));
        return;
    }
    float scores[2], exponentials[2], weights[2];
    for (unsigned source = 0; source < 2; source++) {
        float inverse = transformer_inverse_rms(sources[source], TRANSFORMER_WIDTH);
        double score = 0.0;
        for (unsigned coordinate = 0; coordinate < TRANSFORMER_WIDTH; coordinate++) {
            float normalized = sources[source][coordinate] * inverse;
            score += (double)normalized * (double)transformer_f32(transformer->records[fold].data + coordinate * 4);
        }
        scores[source] = (float)score;
    }
    float maximum = scores[1] > scores[0] ? scores[1] : scores[0];
    double total = 0.0;
    for (unsigned source = 0; source < 2; source++) {
        exponentials[source] = expf(scores[source] - maximum);
        total += (double)exponentials[source];
    }
    for (unsigned source = 0; source < 2; source++) weights[source] = (float)((double)exponentials[source] / total);
    for (unsigned coordinate = 0; coordinate < TRANSFORMER_WIDTH; coordinate++) {
        float value = 0.0f;
        for (unsigned source = 0; source < 2; source++) value = value + weights[source] * sources[source][coordinate];
        sequence->aggregate[coordinate] = value;
    }
}

static void transformer_activation(float *gate, const float *up, unsigned width)
{
    for (unsigned coordinate = 0; coordinate < width; coordinate++) {
        float activated = (4.0f * tanhf(gate[coordinate] / 4.0f)) * transformer_sigmoid(gate[coordinate]);
        float capped = 25.0f * tanhf(up[coordinate] / 25.0f);
        gate[coordinate] = activated * capped;
    }
}

static int transformer_route(const Transformer *transformer, TransformerSequence *sequence)
{
    float scores[896], best[16];
    unsigned count = 0;
    for (unsigned expert = 0; expert < 896; expert++) {
        const unsigned char *row = transformer->records[29].data + (size_t)expert * TRANSFORMER_WIDTH * 4;
        double score = 0.0;
        for (unsigned coordinate = 0; coordinate < TRANSFORMER_WIDTH; coordinate++)
            score += (double)transformer_f32(row + coordinate * 4) * (double)sequence->postnorm[coordinate];
        scores[expert] = transformer_sigmoid((float)score);
        float choice = scores[expert] + transformer_f32(transformer->records[30].data + expert * 4);
        if (!isfinite(choice)) return 0;
        if (count == 16 && !(choice > best[15])) continue;
        unsigned position = count < 16 ? count : 15;
        while (position && choice > best[position - 1]) {
            best[position] = best[position - 1];
            sequence->selected[position] = sequence->selected[position - 1];
            position--;
        }
        best[position] = choice;
        sequence->selected[position] = expert;
        if (count < 16) count++;
    }
    double total = 0.0;
    for (unsigned rank = 0; rank < 16; rank++) total += (double)scores[sequence->selected[rank]];
    float inverse = (float)(1.0 / (total + 1e-20));
    for (unsigned rank = 0; rank < 16; rank++) sequence->weights[rank] = scores[sequence->selected[rank]] * inverse;
    return 1;
}

static int transformer_expert(Transformer *transformer, TransformerSequence *sequence, unsigned expert)
{
    if (!root_project(transformer->root, sequence->root_scratch, expert, 0, sequence->latent, sequence->expert_gate) ||
        !root_project(transformer->root, sequence->root_scratch, expert, 1, sequence->latent, sequence->expert_up)) return 0;
    transformer_activation(sequence->expert_gate, sequence->expert_up, TRANSFORMER_EXPERT);
    return root_project(transformer->root, sequence->root_scratch, expert, 2, sequence->expert_gate, sequence->expert_down);
}

static int transformer_mix_experts(Transformer *transformer, TransformerSequence *sequence)
{
    memset(sequence->mixture, 0, sizeof sequence->mixture);
    for (unsigned rank = 0; rank < 16; rank++) root_prefetch(transformer->root, sequence->selected[rank]);
    for (unsigned rank = 0; rank < 16; rank++) {
        if (!transformer_expert(transformer, sequence, sequence->selected[rank])) return 0;
        for (unsigned coordinate = 0; coordinate < TRANSFORMER_LATENT; coordinate++)
            sequence->mixture[coordinate] = sequence->mixture[coordinate] + sequence->weights[rank] * sequence->expert_down[coordinate];
    }
    transformer_normalize(sequence->latent_norm, sequence->mixture, transformer->records[33].data, TRANSFORMER_LATENT);
    transformer_project(transformer, 32, sequence->latent_norm, sequence->routed);
    return 1;
}

static void transformer_shared(const Transformer *transformer, TransformerSequence *sequence)
{
    transformer_project(transformer, 34, sequence->postnorm, sequence->shared_gate);
    transformer_project(transformer, 35, sequence->postnorm, sequence->shared_up);
    transformer_activation(sequence->shared_gate, sequence->shared_up, TRANSFORMER_SHARED);
    transformer_project(transformer, 36, sequence->shared_gate, sequence->shared_output);
}

static int transformer_moe(Transformer *transformer, TransformerSequence *sequence)
{
    transformer_normalize(sequence->postnorm, sequence->aggregate, transformer->records[5].data, TRANSFORMER_WIDTH);
    if (!transformer_finite(sequence->postnorm, TRANSFORMER_WIDTH) || !transformer_route(transformer, sequence)) return 0;
    transformer_project(transformer, 31, sequence->postnorm, sequence->latent);
    if (!transformer_mix_experts(transformer, sequence)) return 0;
    transformer_shared(transformer, sequence);
    for (unsigned coordinate = 0; coordinate < TRANSFORMER_WIDTH; coordinate++) {
        float mixed = sequence->routed[coordinate] + sequence->shared_output[coordinate];
        sequence->residual[coordinate] = sequence->residual[coordinate] + mixed;
    }
    return 1;
}
/* One condition per stage, one unit of work, one return. The numbering is the shared
   one, so a caller can tell layer 1 stage 7 from layer 46 stage 7. */
enum {
    TRANSFORMER_STAGE_BASE = CLOVER_STAGE_LAYER_BASE + (TRANSFORMER_LAYER - 1) * CLOVER_STAGE_STRIDE,
    TRANSFORMER_STAGE_EXPERT_FIRST = 30,
    TRANSFORMER_STAGE_EXPERT_LAST = 109
};

int transformer_stage(const TransformerSequence *sequence)
{
    return sequence ? TRANSFORMER_STAGE_BASE + sequence->stage : 0;
}

static int transformer_step(Transformer *transformer, TransformerSequence *sequence,
    const float *input, float output[TRANSFORMER_WIDTH])
{
    const int stage = sequence->stage;

    if (stage == 1) { memcpy(sequence->incoming, input, sizeof sequence->incoming); sequence->stage++; return 1; }
    if (stage == 2) { memcpy(sequence->residual, input, sizeof sequence->residual); sequence->stage++; return 1; }
    if (stage == 3) { transformer_aggregate(transformer, sequence, 37); sequence->stage++; return 1; }
    if (stage == 4) { sequence->stage++; return 1; }
    if (stage == 5) {
        transformer_normalize(sequence->normalized, sequence->aggregate, transformer->qkv + 88, TRANSFORMER_WIDTH);
        sequence->stage++; return 1;
    }
    if (stage == 6) { transformer_qkv(transformer, sequence);              sequence->stage++; return 1; }
    if (stage == 7) { transformer_decay(transformer, sequence);            sequence->stage++; return 1; }
    if (stage == 8) { transformer_update_attention(sequence);              sequence->stage++; return 1; }
    if (stage == 9) { transformer_attention_output(transformer, sequence); sequence->stage = 20; return 1; }
    if (stage == 20) {
        for (unsigned coordinate = 0; coordinate < TRANSFORMER_WIDTH; coordinate++)
            sequence->residual[coordinate] = sequence->incoming[coordinate] + sequence->residual[coordinate];
        sequence->stage++; return 1;
    }
    if (stage == 21) { transformer_aggregate(transformer, sequence, 38); sequence->stage++; return 1; }
    if (stage == 22) {
        transformer_normalize(sequence->postnorm, sequence->aggregate, transformer->records[5].data, TRANSFORMER_WIDTH);
        sequence->stage++; return 1;
    }
    if (stage == 23) { if (!transformer_finite(sequence->postnorm, TRANSFORMER_WIDTH)) return 0; sequence->stage++; return 1; }
    if (stage == 24) { if (!transformer_route(transformer, sequence)) return 0; sequence->stage++; return 1; }
    if (stage == 25) { transformer_project(transformer, 31, sequence->postnorm, sequence->latent); sequence->stage++; return 1; }
    if (stage == 26) { memset(sequence->mixture, 0, sizeof sequence->mixture); sequence->stage++; return 1; }
    if (stage == 27) {
        for (unsigned rank = 0; rank < 16; rank++) root_prefetch(transformer->root, sequence->selected[rank]);
        sequence->stage = TRANSFORMER_STAGE_EXPERT_FIRST; return 1;
    }
    if (stage >= TRANSFORMER_STAGE_EXPERT_FIRST && stage <= TRANSFORMER_STAGE_EXPERT_LAST) {
        const unsigned rank = (unsigned)(stage - TRANSFORMER_STAGE_EXPERT_FIRST) / 5;
        const int step = (stage - TRANSFORMER_STAGE_EXPERT_FIRST) % 5;
        const unsigned expert = sequence->selected[rank];
        if (step == 0) {
            if (!root_project(transformer->root, sequence->root_scratch, expert, 0, sequence->latent, sequence->expert_gate)) return 0;
            sequence->stage++; return 1;
        }
        if (step == 1) {
            if (!root_project(transformer->root, sequence->root_scratch, expert, 1, sequence->latent, sequence->expert_up)) return 0;
            sequence->stage++; return 1;
        }
        if (step == 2) {
            transformer_activation(sequence->expert_gate, sequence->expert_up, TRANSFORMER_EXPERT);
            sequence->stage++; return 1;
        }
        if (step == 3) {
            if (!root_project(transformer->root, sequence->root_scratch, expert, 2, sequence->expert_gate, sequence->expert_down)) return 0;
            sequence->stage++; return 1;
        }
        if (step == 4) {
            for (unsigned coordinate = 0; coordinate < TRANSFORMER_LATENT; coordinate++)
                sequence->mixture[coordinate] = sequence->mixture[coordinate] + sequence->weights[rank] * sequence->expert_down[coordinate];
            sequence->stage++; return 1;
        }
    }
    if (stage == 110) {
        transformer_normalize(sequence->latent_norm, sequence->mixture, transformer->records[33].data, TRANSFORMER_LATENT);
        sequence->stage++; return 1;
    }
    if (stage == 111) { transformer_project(transformer, 32, sequence->latent_norm, sequence->routed); sequence->stage++; return 1; }
    if (stage == 112) { transformer_project(transformer, 34, sequence->postnorm, sequence->shared_gate); sequence->stage++; return 1; }
    if (stage == 113) { transformer_project(transformer, 35, sequence->postnorm, sequence->shared_up); sequence->stage++; return 1; }
    if (stage == 114) { transformer_activation(sequence->shared_gate, sequence->shared_up, TRANSFORMER_SHARED); sequence->stage++; return 1; }
    if (stage == 115) { transformer_project(transformer, 36, sequence->shared_gate, sequence->shared_output); sequence->stage++; return 1; }
    if (stage == 116) {
        for (unsigned coordinate = 0; coordinate < TRANSFORMER_WIDTH; coordinate++)
            sequence->residual[coordinate] = sequence->residual[coordinate] +
                (sequence->routed[coordinate] + sequence->shared_output[coordinate]);
        sequence->stage++; return 1;
    }
    if (stage == 117) {
        if (!transformer_finite(sequence->residual, TRANSFORMER_WIDTH) ||
            !transformer_finite(&sequence->recurrent[0][0][0], TRANSFORMER_HEADS * TRANSFORMER_HEAD * TRANSFORMER_HEAD) ||
            !transformer_finite(&sequence->history[0][0][0], 3 * TRANSFORMER_ROWS * 3)) return 0;
        sequence->stage++; return 1;
    }
    if (stage == 118) { memcpy(output, sequence->residual, sizeof sequence->residual); sequence->stage++; return 1; }
    if (stage == 119) { sequence->stage++; return 1; }
    if (stage == 120) { sequence->positions++; sequence->stage = 0; return 1; }

    sequence->stage++;
    return 1;
}

int transformer_process(Transformer *transformer, TransformerSequence *sequence,
    const float input[TRANSFORMER_WIDTH], const float snapshot[TRANSFORMER_WIDTH], float output[TRANSFORMER_WIDTH])
{
    if (!transformer || !sequence || sequence->owner != transformer || !input || !output || !snapshot ||
        sequence->failed || sequence->positions == SIZE_MAX ||
        fegetround() != FE_TONEAREST || FLT_EVAL_METHOD != 0 || !transformer_finite(input, TRANSFORMER_WIDTH) || !transformer_finite(snapshot, TRANSFORMER_WIDTH)) return 0;
    memcpy(sequence->snapshot, snapshot, sizeof sequence->snapshot);
    sequence->stage = 1;
    clover_trace_open();
    /* Layer 1 is the only owner whose residual moves with the prompt, and it was the
       one owner absent from every capture. */
    clover_vectors_open(TRANSFORMER_LAYER);
    clover_vectors_write(TRANSFORMER_LAYER, 0, (unsigned long)sequence->positions,
        input, TRANSFORMER_WIDTH);
    while (sequence->stage) {
        const int executed = sequence->stage;
        const double began = clover_trace_active() ? clover_trace_clock() : 0.0;
        if (!transformer_step(transformer, sequence, input, output)) {
            sequence->failed = 1;
            return 0;
        }
        if (clover_trace_active())
            clover_trace_stage(TRANSFORMER_LAYER, executed, (unsigned long)sequence->positions,
                clover_trace_clock() - began,
                clover_trace_hash(sequence->residual, sizeof sequence->residual),
                executed >= TRANSFORMER_STAGE_EXPERT_FIRST && executed <= TRANSFORMER_STAGE_EXPERT_LAST
                    ? (long)sequence->selected[(executed - TRANSFORMER_STAGE_EXPERT_FIRST) / 5] : -1);
        clover_vectors_write(TRANSFORMER_LAYER, executed, (unsigned long)sequence->positions,
            sequence->residual, TRANSFORMER_WIDTH);
        if (clover_vectors_active(TRANSFORMER_LAYER) && (executed == 3 || executed == 21))
            clover_vectors_write(TRANSFORMER_LAYER, 4000 + executed, (unsigned long)sequence->positions,
                sequence->aggregate, TRANSFORMER_WIDTH);
    }
    clover_vectors_write(TRANSFORMER_LAYER, 1000, (unsigned long)sequence->positions,
        sequence->snapshot, TRANSFORMER_WIDTH);
    return 1;
}

/* The transformer owns the way in: the token, its embedding and trunk 0. The way out
   belongs to layer 93, which holds final normalisation, the head and the address the
   answer goes to, so neither end has to wait on the other. */

#ifndef TRANSFORMER_NO_MAIN
static int transformer_receive(float *input, float *snapshot)
{
    int status = scanf("%f", input);
    if (status == EOF && !ferror(stdin)) return 0;
    if (status != 1) return -1;
    for (unsigned coordinate = 1; coordinate < TRANSFORMER_WIDTH; coordinate++)
        if (scanf("%f", input + coordinate) != 1) return -1;
    for (unsigned coordinate = 0; coordinate < TRANSFORMER_WIDTH; coordinate++)
        if (scanf("%f", snapshot + coordinate) != 1) return -1;
    return 1;
}

static int transformer_send(const float *output, const float *snapshot)
{
    for (unsigned coordinate = 0; coordinate < TRANSFORMER_WIDTH; coordinate++)
        printf("%a%c", (double)output[coordinate], coordinate + 1 == TRANSFORMER_WIDTH ? '\n' : ' ');
    for (unsigned coordinate = 0; coordinate < TRANSFORMER_WIDTH; coordinate++)
        printf("%a%c", (double)snapshot[coordinate], coordinate + 1 == TRANSFORMER_WIDTH ? '\n' : ' ');
    return fflush(stdout) == 0 && !ferror(stdout);
}

static int transformer_stream(Transformer *transformer)
{
    TransformerSequence *sequence = transformer_sequence_create(transformer);
    if (!sequence) return 1;
    float input[TRANSFORMER_WIDTH], snapshot[TRANSFORMER_WIDTH], output[TRANSFORMER_WIDTH];
    int result = 0;
    for (;;) {
        int status = transformer_receive(input, snapshot);
        if (!status) break;
        if (status < 0) { result = 2; break; }
        if (!transformer_process(transformer, sequence, input, snapshot, output) ||
            !transformer_send(output, snapshot)) { result = 1; break; }
    }
    if (result) fputs("transformer-1: incomplete, invalid or failed vector pair\n", stderr);
    transformer_sequence_close(sequence);
    return result;
}

static void transformer_inspect(const Transformer *transformer)
{
    unsigned count = 0;
    for (unsigned id = 0; id < TRANSFORMER_RECORDS; id++) count += transformer->records[id].data != NULL;
    printf("layer 1: KDA operator, %u additional trunk records, 896 computed-value experts\n", count);
}

int main(int argc, char **argv)
{
    if ((argc != 2 && argc != 3) || (strcmp(argv[1], "--inspect") && strcmp(argv[1], "--stream"))) {
        fputs("usage: transformer-1 --inspect|--stream [DATASET_DIRECTORY]\n", stderr);
        return 2;
    }
    Transformer *transformer = NULL;
    if (!transformer_open(argc == 3 ? argv[2] : "dataset", &transformer)) {
        fputs("transformer-1: invalid or unreadable layer-1 datasets\n", stderr);
        return 1;
    }
    int result = 0;
    if (!strcmp(argv[1], "--stream")) result = transformer_stream(transformer);
    else transformer_inspect(transformer);
    transformer_close(transformer);
    return result;
}
#endif