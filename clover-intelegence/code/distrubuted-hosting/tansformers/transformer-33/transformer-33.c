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
#include "root.h"
#define TRANSFORMER_LAYER 33
#define TRANSFORMER_MLA 0
#define TRANSFORMER_INPUT_SNAPSHOTS 3
#define TRANSFORMER_OUTPUT_SNAPSHOTS 3

enum {
    TRANSFORMER_WIDTH = 7168, TRANSFORMER_ROWS = 12288, TRANSFORMER_HEAD = 128,
    TRANSFORMER_HEADS = 96, TRANSFORMER_LATENT = 3584, TRANSFORMER_EXPERT = 3072,
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
    TransformerRecord operator_records[40];
    const unsigned char *input_gains;
    Root *root;
} Transformer;

typedef struct {
    float history[3][TRANSFORMER_ROWS][3];
    float recurrent[TRANSFORMER_HEADS][TRANSFORMER_HEAD][TRANSFORMER_HEAD];
    float snapshots[8][TRANSFORMER_WIDTH], normalized[TRANSFORMER_WIDTH];
    unsigned snapshot_count;
    float qkv[3][TRANSFORMER_ROWS], beta[TRANSFORMER_HEADS], decay_hidden[TRANSFORMER_HEAD];
    float decay[TRANSFORMER_ROWS], gate[TRANSFORMER_ROWS], attention[TRANSFORMER_ROWS];
    float residual[TRANSFORMER_WIDTH], aggregate[TRANSFORMER_WIDTH], postnorm[TRANSFORMER_WIDTH];
    float incoming[TRANSFORMER_WIDTH], latent[TRANSFORMER_LATENT], mixture[TRANSFORMER_LATENT];
    float expert_gate[TRANSFORMER_EXPERT], expert_up[TRANSFORMER_EXPERT], expert_down[TRANSFORMER_LATENT];
    float latent_norm[TRANSFORMER_LATENT], routed[TRANSFORMER_WIDTH];
    float shared_gate[TRANSFORMER_SHARED], shared_up[TRANSFORMER_SHARED], shared_output[TRANSFORMER_WIDTH];
    unsigned selected[16];
    float weights[16];
    RootScratch root_scratch;
    float *mla_cache;
    size_t mla_capacity;
    float mla_query_latent[1536], mla_query[18432], mla_kv_latent[576], mla_expanded[24576];
    const Transformer *owner;
    size_t positions;
    int failed;
} TransformerSequence;

typedef struct {
    unsigned id, group, kind, rows, columns;
} TransformerLayout;

static const TransformerLayout transformer_layout[] = {
    {0, 0, 2, 7168, 1},
    {1, 0, 2, 7168, 1},
    {2, 0, 2, 7168, 1},
    {3, 0, 2, 7168, 1},
    {4, 0, 2, 7168, 1},
    {5, 0, 2, 7168, 1},
    {6, 0, 1, 12288, 7168},
    {7, 0, 1, 7168, 12288},
    {8, 1, 1, 12288, 7168},
    {9, 1, 1, 12288, 7168},
    {10, 1, 1, 12288, 7168},
    {11, 1, 1, 96, 7168},
    {12, 1, 1, 128, 7168},
    {13, 1, 1, 12288, 128},
    {14, 1, 2, 49152, 1},
    {15, 1, 2, 49152, 1},
    {16, 1, 2, 49152, 1},
    {17, 1, 2, 128, 1},
    {18, 1, 2, 12288, 1},
    {19, 1, 2, 128, 1},
    {29, 2, 3, 896, 7168},
    {30, 2, 2, 896, 1},
    {31, 2, 1, 3584, 7168},
    {32, 2, 1, 7168, 3584},
    {33, 2, 2, 3584, 1},
    {34, 2, 1, 6144, 7168},
    {35, 2, 1, 6144, 7168},
    {36, 2, 1, 7168, 6144},
    {37, 0, 2, 7168, 1},
    {38, 0, 2, 7168, 1},
    {39, 1, 2, 96, 1}
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

static int transformer_needs(unsigned id)
{
    if (TRANSFORMER_MLA && id >= 8 && id < 29) return 0;
    switch (id) {
    case 5: case 6: case 7: case 11: case 12: case 13: case 18:
    case 19: case 29: case 30: case 31: case 32: case 33: case 34:
    case 35: case 36: case 37: case 38: case 39: return 1;
    default: return 0;
    }
}

static int transformer_load_qkv(Transformer *transformer, const char *path)
{
    const size_t bytes = TRANSFORMER_MLA ? 56251936 :
        TRANSFORMER_QKV_PREFIX + (size_t)TRANSFORMER_ROWS * 3 * TRANSFORMER_QKV_COMPONENT;
    FILE *file = transformer_open_sized(path, bytes);
    if (!file) return 0;
    int loaded = transformer_read(file, 0, bytes, &transformer->qkv);
    if (fclose(file)) loaded = 0;
    if (!loaded) return 0;
    const unsigned char *data = transformer->qkv;
    const uint32_t fields[] = {1, TRANSFORMER_LAYER, TRANSFORMER_WIDTH,
        TRANSFORMER_MLA ? 18432 : TRANSFORMER_ROWS, TRANSFORMER_MLA ? 192 : 128,
        3, TRANSFORMER_MLA ? 0 : 4, TRANSFORMER_MLA ? 7 : 6,
        TRANSFORMER_MLA ? 32 : TRANSFORMER_QKV_COMPONENT, 0};
    if (memcmp(data, TRANSFORMER_MLA ? "K3MLA001" : "K3QKV001", 8)) return 0;
    for (unsigned field = 0; field < 10; field++) if (transformer_u32(data + 8 + field * 4) != fields[field]) return 0;
    if (transformer_f32(data + 48) != 1e-5f || transformer_f32(data + 52) != 1e-6f || transformer_u64(data + 56) != bytes) return 0;
    size_t palette_offset = TRANSFORMER_MLA ? 64 + 7 * 32 : 88 + TRANSFORMER_WIDTH * 4;
    for (unsigned code = 0; code < 256; code++) {
        float expected = (float)(code < 128 ? (int)code : (int)code - 256);
        float stored = transformer_f32(data + palette_offset + code * 4);
        if (memcmp(&stored, &expected, 4)) return 0;
        transformer->palette[code] = stored;
    }
    if (!TRANSFORMER_MLA) {
        for (unsigned stage = 0; stage < 6; stage++) if (transformer_u32(data + 64 + stage * 4) != stage + 1) return 0;
        transformer->input_gains = data + 88;
        for (unsigned coordinate = 0; coordinate < TRANSFORMER_WIDTH; coordinate++)
            if (!isfinite(transformer_f32(data + 88 + coordinate * 4))) return 0;
        for (unsigned row = 0; row < TRANSFORMER_ROWS; row++) for (unsigned component = 0; component < 3; component++) {
            const unsigned char *record = data + TRANSFORMER_QKV_PREFIX + ((size_t)row * 3 + component) * TRANSFORMER_QKV_COMPONENT;
            for (unsigned field = 0; field < 5; field++) if (!isfinite(transformer_f32(record + field * 4))) return 0;
        }
    } else {
        const unsigned schema[7][4] = {{4,2,7168,1},{20,1,1536,7168},{21,2,1536,1},
            {22,1,18432,1536},{23,1,576,7168},{24,2,512,1},{25,1,24576,512}};
        size_t next = 64 + 7 * 32 + 1024;
        for (unsigned index = 0; index < 7; index++) {
            const unsigned char *encoded = data + 64 + index * 32;
            for (unsigned field = 0; field < 4; field++) if (transformer_u32(encoded + field * 4) != schema[index][field]) return 0;
            unsigned slot = schema[index][0], kind = schema[index][1], rows = schema[index][2], columns = schema[index][3];
            size_t length = kind == 1 ? rows * 4ULL + (size_t)rows * columns : (size_t)rows * columns * 4;
            if (transformer_u64(encoded + 16) != next || transformer_u64(encoded + 24) != length || length > bytes - next) return 0;
            TransformerRecord *record = &transformer->operator_records[slot];
            *record = (TransformerRecord){slot, 1, kind, rows, columns, rows * columns, next, length, transformer->qkv + next};
            unsigned floats = kind == 1 ? rows : rows * columns;
            for (unsigned coordinate = 0; coordinate < floats; coordinate++)
                if (!isfinite(transformer_f32(record->data + (size_t)coordinate * 4))) return 0;
            next += length;
        }
        if (next != bytes) return 0;
        transformer->input_gains = transformer->operator_records[4].data;
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
        transformer_u32(index + 12) == TRANSFORMER_LAYER && transformer_u32(index + 16) == count && !transformer_u32(index + 20);
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
        for (unsigned id = 0; valid && id < TRANSFORMER_RECORDS; id++) {
            TransformerRecord *record = &transformer->records[id];
            if (!transformer_needs(id) || record->group != group) continue;
            if (!transformer_read(file, record->offset, (size_t)record->length, &record->data)) { valid = 0; break; }
            unsigned floats = record->kind == 1 ? record->rows : record->count;
            for (unsigned coordinate = 0; coordinate < floats; coordinate++)
                if (!isfinite(transformer_f32(record->data + (size_t)coordinate * 4))) { valid = 0; break; }
        }
        if (fclose(file)) valid = 0;
        if (!valid) return 0;
    }
    return 1;
}

void transformer_close(Transformer *transformer)
{
    if (!transformer) return;
    for (unsigned id = 0; id < TRANSFORMER_RECORDS; id++) free(transformer->records[id].data);
    root_close(transformer->root);
    free(transformer->qkv);
    free(transformer);
}

int transformer_open(const char *dataset, Transformer **result)
{
    if (!result) return 0;
    *result = NULL;
    char qkv_path[4096], trunk_directory[4096], root_directory[4096];
    if (!dataset || !transformer_path(qkv_path, sizeof qkv_path, dataset, "operators/qkv-all/layer-33/operator.bin") ||
        !transformer_path(trunk_directory, sizeof trunk_directory, dataset, "trunk-33") ||
        !transformer_path(root_directory, sizeof root_directory, dataset, "root-33") ||
        sizeof(float) != 4 || sizeof(double) != 8 ||
        FLT_RADIX != 2 || FLT_MANT_DIG != 24 || DBL_MANT_DIG != 53) return 0;
    Transformer *transformer = calloc(1, sizeof *transformer);
    if (!transformer) return 0;
    if (!transformer_load_qkv(transformer, qkv_path) || !transformer_load_trunk(transformer, trunk_directory) || !(transformer->root = root_open(root_directory))) {
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
    if (sequence) sequence->owner = transformer;
    return sequence;
}

void transformer_sequence_reset(TransformerSequence *sequence)
{
    if (!sequence) return;
    const Transformer *owner = sequence->owner;
    free(sequence->mla_cache);
    memset(sequence, 0, sizeof *sequence);
    sequence->owner = owner;
}

void transformer_sequence_close(TransformerSequence *sequence)
{
    if (sequence) free(sequence->mla_cache);
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
    const TransformerRecord *record = transformer->operator_records[id].data ? &transformer->operator_records[id] : &transformer->records[id];
    const unsigned char *ids = record->data + (size_t)record->rows * 4;
    for (unsigned row = 0; row < record->rows; row++)
        output[row] = transformer_project_row(input, ids + (size_t)row * record->columns,
            record->columns, transformer->palette, transformer_f32(record->data + row * 4));
}

static float transformer_sigmoid(float value)
{
    return 1.0f / (1.0f + expf(-value));
}

#if !TRANSFORMER_MLA
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

#endif

#if TRANSFORMER_MLA
static int transformer_mla(Transformer *transformer, TransformerSequence *sequence)
{
    const size_t stride = 2 * TRANSFORMER_ROWS + 64;
    size_t count = sequence->positions + 1;
    if (count > SIZE_MAX / stride / sizeof(float) || count > SIZE_MAX / (2 * sizeof(float))) return 0;
    if (count > sequence->mla_capacity) {
        size_t capacity = sequence->mla_capacity ? sequence->mla_capacity : 8;
        while (capacity < count) {
            if (capacity > SIZE_MAX / 2) return 0;
            capacity *= 2;
        }
        if (capacity > SIZE_MAX / stride / sizeof(float)) return 0;
        float *cache = realloc(sequence->mla_cache, capacity * stride * sizeof(float));
        if (!cache) return 0;
        sequence->mla_cache = cache;
        sequence->mla_capacity = capacity;
    }
    float *scores = malloc(count * 2 * sizeof(float));
    if (!scores) return 0;
    float *exponentials = scores + count;
    transformer_project(transformer, 20, sequence->normalized, sequence->mla_query_latent);
    transformer_normalize(sequence->mla_query_latent, sequence->mla_query_latent,
        transformer->operator_records[21].data, 1536);
    transformer_project(transformer, 22, sequence->mla_query_latent, sequence->mla_query);
    transformer_project(transformer, 23, sequence->normalized, sequence->mla_kv_latent);
    transformer_normalize(sequence->mla_kv_latent, sequence->mla_kv_latent,
        transformer->operator_records[24].data, 512);
    transformer_project(transformer, 25, sequence->mla_kv_latent, sequence->mla_expanded);
    float *current = sequence->mla_cache + sequence->positions * stride;
    for (unsigned head = 0; head < TRANSFORMER_HEADS; head++) {
        memcpy(current + head * 128, sequence->mla_expanded + head * 256, 128 * sizeof(float));
        memcpy(current + TRANSFORMER_ROWS + head * 128, sequence->mla_expanded + head * 256 + 128, 128 * sizeof(float));
    }
    memcpy(current + 2 * TRANSFORMER_ROWS, sequence->mla_kv_latent + 512, 64 * sizeof(float));
    const float scale = 1.0f / sqrtf(192.0f);
    for (unsigned head = 0; head < TRANSFORMER_HEADS; head++) {
        const float *query = sequence->mla_query + head * 192;
        for (size_t position = 0; position < count; position++) {
            const float *cached = sequence->mla_cache + position * stride;
            double score = 0.0;
            for (unsigned coordinate = 0; coordinate < 128; coordinate++)
                score += (double)query[coordinate] * (double)cached[head * 128 + coordinate];
            for (unsigned coordinate = 0; coordinate < 64; coordinate++)
                score += (double)query[128 + coordinate] * (double)cached[2 * TRANSFORMER_ROWS + coordinate];
            scores[position] = (float)score * scale;
        }
        float maximum = scores[0];
        for (size_t position = 1; position < count; position++) if (scores[position] > maximum) maximum = scores[position];
        double total = 0.0;
        for (size_t position = 0; position < count; position++) {
            exponentials[position] = expf(scores[position] - maximum);
            total += (double)exponentials[position];
        }
        float *output = sequence->attention + head * 128;
        memset(output, 0, 128 * sizeof(float));
        for (size_t position = 0; position < count; position++) {
            float weight = (float)((double)exponentials[position] / total);
            const float *value = sequence->mla_cache + position * stride + TRANSFORMER_ROWS + head * 128;
            for (unsigned coordinate = 0; coordinate < 128; coordinate++) output[coordinate] = output[coordinate] + weight * value[coordinate];
        }
    }
    free(scores);
    transformer_project(transformer, 6, sequence->normalized, sequence->gate);
    for (unsigned coordinate = 0; coordinate < TRANSFORMER_ROWS; coordinate++)
        sequence->gate[coordinate] = sequence->attention[coordinate] * transformer_sigmoid(sequence->gate[coordinate]);
    transformer_project(transformer, 7, sequence->gate, sequence->residual);
    return 1;
}
#endif

static void transformer_aggregate(const Transformer *transformer, TransformerSequence *sequence, unsigned fold)
{
    const float *sources[9];
    unsigned count = sequence->snapshot_count + 1;
    for (unsigned source = 0; source < sequence->snapshot_count; source++) sources[source] = sequence->snapshots[source];
    sources[sequence->snapshot_count] = sequence->residual;
    float scores[9], exponentials[9], weights[9];
    for (unsigned source = 0; source < count; source++) {
        float inverse = transformer_inverse_rms(sources[source], TRANSFORMER_WIDTH);
        double score = 0.0;
        for (unsigned coordinate = 0; coordinate < TRANSFORMER_WIDTH; coordinate++) {
            float normalized = sources[source][coordinate] * inverse;
            score += (double)normalized * (double)transformer_f32(transformer->records[fold].data + coordinate * 4);
        }
        scores[source] = (float)score;
    }
    float maximum = scores[0];
    for (unsigned source = 1; source < count; source++) if (scores[source] > maximum) maximum = scores[source];
    double total = 0.0;
    for (unsigned source = 0; source < count; source++) {
        exponentials[source] = expf(scores[source] - maximum);
        total += (double)exponentials[source];
    }
    for (unsigned source = 0; source < count; source++) weights[source] = (float)((double)exponentials[source] / total);
    for (unsigned coordinate = 0; coordinate < TRANSFORMER_WIDTH; coordinate++) {
        float value = 0.0f;
        for (unsigned source = 0; source < count; source++) value = value + weights[source] * sources[source][coordinate];
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
    if (!root_project(transformer->root, &sequence->root_scratch, expert, 0, sequence->latent, sequence->expert_gate) ||
        !root_project(transformer->root, &sequence->root_scratch, expert, 1, sequence->latent, sequence->expert_up)) return 0;
    transformer_activation(sequence->expert_gate, sequence->expert_up, TRANSFORMER_EXPERT);
    return root_project(transformer->root, &sequence->root_scratch, expert, 2, sequence->expert_gate, sequence->expert_down);
}

static int transformer_mix_experts(Transformer *transformer, TransformerSequence *sequence)
{
    memset(sequence->mixture, 0, sizeof sequence->mixture);
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
int transformer_process(Transformer *transformer, TransformerSequence *sequence,
    const float input[TRANSFORMER_WIDTH], const float *snapshots, unsigned snapshot_count,
    float output[TRANSFORMER_WIDTH], float *output_snapshots)
{
    if (!transformer || !sequence || sequence->owner != transformer || !input || !output || !snapshots || !output_snapshots ||
        snapshot_count != TRANSFORMER_INPUT_SNAPSHOTS || sequence->failed || sequence->positions == SIZE_MAX ||
        fegetround() != FE_TONEAREST || FLT_EVAL_METHOD != 0 || !transformer_finite(input, TRANSFORMER_WIDTH) ||
        !transformer_finite(snapshots, (size_t)snapshot_count * TRANSFORMER_WIDTH)) return 0;
    memcpy(sequence->snapshots, snapshots, (size_t)snapshot_count * TRANSFORMER_WIDTH * sizeof(float));
    sequence->snapshot_count = snapshot_count;
    memcpy(sequence->incoming, input, sizeof sequence->incoming);
    memcpy(sequence->residual, input, sizeof sequence->residual);
    transformer_aggregate(transformer, sequence, 37);
    if (TRANSFORMER_LAYER % 12 == 0) {
        memcpy(sequence->snapshots[sequence->snapshot_count], input, TRANSFORMER_WIDTH * sizeof(float));
        sequence->snapshot_count++;
    }
    transformer_normalize(sequence->normalized, sequence->aggregate, transformer->input_gains, TRANSFORMER_WIDTH);
#if TRANSFORMER_MLA
    if (!transformer_mla(transformer, sequence)) { sequence->failed = 1; return 0; }
#else
    transformer_qkv(transformer, sequence);
    transformer_decay(transformer, sequence);
    transformer_update_attention(sequence);
    transformer_attention_output(transformer, sequence);
#endif
    if (TRANSFORMER_LAYER % 12 != 0) for (unsigned coordinate = 0; coordinate < TRANSFORMER_WIDTH; coordinate++)
        sequence->residual[coordinate] = sequence->incoming[coordinate] + sequence->residual[coordinate];
    transformer_aggregate(transformer, sequence, 38);
    if (!transformer_moe(transformer, sequence)) { sequence->failed = 1; return 0; }
    if (!transformer_finite(sequence->residual, TRANSFORMER_WIDTH)) { sequence->failed = 1; return 0; }
    memcpy(output, sequence->residual, sizeof sequence->residual);
    memcpy(output_snapshots, sequence->snapshots, (size_t)sequence->snapshot_count * TRANSFORMER_WIDTH * sizeof(float));
    sequence->positions++;
    return 1;
}

int transformer_routes(const TransformerSequence *sequence, unsigned selected[16])
{
    if (!sequence || !selected || !sequence->positions || sequence->failed) return 0;
    memcpy(selected, sequence->selected, sizeof sequence->selected);
    return 1;
}

#ifndef TRANSFORMER_NO_MAIN
static int transformer_receive(float *input, float *snapshots)
{
    int status = scanf("%f", input);
    if (status == EOF && !ferror(stdin)) return 0;
    if (status != 1) return -1;
    for (unsigned coordinate = 1; coordinate < TRANSFORMER_WIDTH; coordinate++) if (scanf("%f", input + coordinate) != 1) return -1;
    for (unsigned coordinate = 0; coordinate < TRANSFORMER_INPUT_SNAPSHOTS * TRANSFORMER_WIDTH; coordinate++)
        if (scanf("%f", snapshots + coordinate) != 1) return -1;
    return 1;
}

static int transformer_send(const float *output, const float *snapshots)
{
    for (unsigned row = 0; row <= TRANSFORMER_OUTPUT_SNAPSHOTS; row++) {
        const float *values = row ? snapshots + (row - 1) * TRANSFORMER_WIDTH : output;
        for (unsigned coordinate = 0; coordinate < TRANSFORMER_WIDTH; coordinate++)
            printf("%a%c", (double)values[coordinate], coordinate + 1 == TRANSFORMER_WIDTH ? '\n' : ' ');
    }
    return fflush(stdout) == 0 && !ferror(stdout);
}

static int transformer_stream(Transformer *transformer)
{
    TransformerSequence *sequence = transformer_sequence_create(transformer);
    if (!sequence) return 1;
    float input[7168], snapshots[8 * 7168], output[7168], output_snapshots[8 * 7168];
    int result = 0;
    for (;;) {
        int status = transformer_receive(input, snapshots);
        if (!status) break;
        if (status < 0) { result = 2; break; }
        if (!transformer_process(transformer, sequence, input, snapshots, TRANSFORMER_INPUT_SNAPSHOTS, output, output_snapshots) ||
            !transformer_send(output, output_snapshots)) { result = 1; break; }
    }
    if (result) fputs("transformer: incomplete, invalid or failed vector bundle\n", stderr);
    transformer_sequence_close(sequence);
    return result;
}

static void transformer_inspect(const Transformer *transformer)
{
    unsigned count = 0;
    for (unsigned id = 0; id < TRANSFORMER_RECORDS; id++) count += transformer->records[id].data != NULL;
    printf("layer %u: %s, %u additional records, 896 experts, snapshots %u -> %u\n",
        TRANSFORMER_LAYER, TRANSFORMER_MLA ? "MLA" : "KDA", count, TRANSFORMER_INPUT_SNAPSHOTS, TRANSFORMER_OUTPUT_SNAPSHOTS);
}

int main(int argc, char **argv)
{
    if ((argc != 2 && argc != 3) || (strcmp(argv[1], "--inspect") && strcmp(argv[1], "--stream"))) {
        fputs("usage: transformer-N --inspect|--stream [DATASET_DIRECTORY]\n", stderr);
        return 2;
    }
    Transformer *transformer = NULL;
    if (!transformer_open(argc == 3 ? argv[2] : "dataset", &transformer)) {
        fputs("transformer: invalid or unreadable layer-specific datasets\n", stderr);
        return 1;
    }
    int result = 0;
    if (!strcmp(argv[1], "--stream")) result = transformer_stream(transformer);
    else transformer_inspect(transformer);
    transformer_close(transformer);
    return result;
}
#endif