#include <float.h>
#include <fenv.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    SERVER_WIDTH = 7168, SERVER_ROWS = 12288, SERVER_HEAD = 128,
    SERVER_HEADS = 96, SERVER_DENSE = 33792, SERVER_RECORDS = 40,
    SERVER_QKV_PREFIX = 64 + 24 + SERVER_WIDTH * 4 + 1024,
    SERVER_QKV_COMPONENT = SERVER_WIDTH + 20
};

typedef struct {
    uint32_t id, group, kind, rows, columns, count;
    uint64_t offset, length;
    unsigned char *data;
} ServerRecord;

typedef struct {
    unsigned char *qkv;
    ServerRecord records[SERVER_RECORDS];
    float palette[256];
} Server;

typedef struct {
    float history[3][SERVER_ROWS][3];
    float recurrent[SERVER_HEADS][SERVER_HEAD][SERVER_HEAD];
    float snapshot[SERVER_WIDTH], normalized[SERVER_WIDTH];
    float qkv[3][SERVER_ROWS], beta[SERVER_HEADS], decay_hidden[SERVER_HEAD];
    float decay[SERVER_ROWS], gate[SERVER_ROWS], attention[SERVER_ROWS];
    float residual[SERVER_WIDTH], aggregate[SERVER_WIDTH], postnorm[SERVER_WIDTH];
    float dense_gate[SERVER_DENSE], dense_up[SERVER_DENSE], dense_output[SERVER_WIDTH];
    const Server *owner;
    size_t positions;
    int failed;
} ServerSequence;

typedef struct {
    unsigned id, group, kind, rows, columns;
} ServerLayout;

static const ServerLayout server_layout[] = {
    {0, 0, 2, 7168, 1}, {1, 0, 2, 7168, 1}, {2, 0, 2, 7168, 1},
    {3, 0, 2, 7168, 1}, {4, 0, 2, 7168, 1}, {5, 0, 2, 7168, 1},
    {6, 0, 1, 12288, 7168}, {7, 0, 1, 7168, 12288},
    {8, 1, 1, 12288, 7168}, {9, 1, 1, 12288, 7168},
    {10, 1, 1, 12288, 7168}, {11, 1, 1, 96, 7168},
    {12, 1, 1, 128, 7168}, {13, 1, 1, 12288, 128},
    {14, 1, 2, 49152, 1}, {15, 1, 2, 49152, 1}, {16, 1, 2, 49152, 1},
    {17, 1, 2, 128, 1}, {18, 1, 2, 12288, 1}, {19, 1, 2, 128, 1},
    {26, 2, 1, 33792, 7168}, {27, 2, 1, 33792, 7168},
    {28, 2, 1, 7168, 33792}, {37, 0, 2, 7168, 1},
    {38, 0, 2, 7168, 1}, {39, 1, 2, 96, 1}
};

static uint32_t server_u32(const unsigned char *bytes)
{
    return (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) |
        ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
}

static uint64_t server_u64(const unsigned char *bytes)
{
    return server_u32(bytes) | ((uint64_t)server_u32(bytes + 4) << 32);
}

static float server_f32(const unsigned char *bytes)
{
    uint32_t bits = server_u32(bytes);
    float value;
    memcpy(&value, &bits, sizeof value);
    return value;
}

static int server_path(char *output, size_t capacity, const char *directory, const char *name)
{
    int length = snprintf(output, capacity, "%s/%s", directory, name);
    return length >= 0 && (size_t)length < capacity;
}

static FILE *server_open_sized(const char *path, uint64_t expected)
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

static int server_read(FILE *file, uint64_t offset, size_t length, unsigned char **output)
{
    if (offset > LONG_MAX || fseek(file, (long)offset, SEEK_SET)) return 0;
    unsigned char *data = malloc(length);
    if (!data) return 0;
    if (fread(data, 1, length, file) != length) { free(data); return 0; }
    *output = data;
    return 1;
}

static int server_needs(unsigned id)
{
    switch (id) {
    case 5: case 6: case 7: case 11: case 12: case 13: case 18:
    case 19: case 26: case 27: case 28: case 38: case 39: return 1;
    default: return 0;
    }
}

static int server_load_qkv(Server *server, const char *path)
{
    const size_t bytes = SERVER_QKV_PREFIX + (size_t)SERVER_ROWS * 3 * SERVER_QKV_COMPONENT;
    FILE *file = server_open_sized(path, bytes);
    if (!file) return 0;
    int loaded = server_read(file, 0, bytes, &server->qkv);
    if (fclose(file)) loaded = 0;
    if (!loaded) return 0;
    const unsigned char *data = server->qkv;
    const uint32_t fields[] = {1, 0, SERVER_WIDTH, SERVER_ROWS, SERVER_HEAD, 3, 4, 6, SERVER_QKV_COMPONENT, 0};
    if (memcmp(data, "K3QKV001", 8)) return 0;
    for (size_t field = 0; field < sizeof fields / sizeof *fields; field++)
        if (server_u32(data + 8 + field * 4) != fields[field]) return 0;
    if (server_f32(data + 48) != 1e-5f || server_f32(data + 52) != 1e-6f || server_u64(data + 56) != bytes) return 0;
    for (unsigned stage = 0; stage < 6; stage++) if (server_u32(data + 64 + stage * 4) != stage + 1) return 0;
    for (unsigned coordinate = 0; coordinate < SERVER_WIDTH; coordinate++)
        if (!isfinite(server_f32(data + 88 + coordinate * 4))) return 0;
    for (unsigned code = 0; code < 256; code++) {
        float expected = (float)(code < 128 ? (int)code : (int)code - 256);
        float stored = server_f32(data + 88 + SERVER_WIDTH * 4 + code * 4);
        if (memcmp(&stored, &expected, sizeof stored)) return 0;
        server->palette[code] = stored;
    }
    for (unsigned row = 0; row < SERVER_ROWS; row++)
        for (unsigned component = 0; component < 3; component++) {
            const unsigned char *record = data + SERVER_QKV_PREFIX + ((size_t)row * 3 + component) * SERVER_QKV_COMPONENT;
            for (unsigned field = 0; field < 5; field++) if (!isfinite(server_f32(record + field * 4))) return 0;
        }
    return 1;
}

static int server_load_trunk(Server *server, const char *directory)
{
    const size_t count = sizeof server_layout / sizeof *server_layout;
    const size_t bytes = 1168 + count * 72;
    char path[4096];
    unsigned char *index = NULL;
    if (!server_path(path, sizeof path, directory, "index.bin")) return 0;
    FILE *file = server_open_sized(path, bytes);
    if (!file) return 0;
    int loaded = server_read(file, 0, bytes, &index);
    if (fclose(file)) loaded = 0;
    if (!loaded) { free(index); return 0; }
    int valid = !memcmp(index, "K3TRK001", 8) && server_u32(index + 8) == 1 &&
        !server_u32(index + 12) && server_u32(index + 16) == count && !server_u32(index + 20);
    uint64_t lengths[3], ends[3] = {0};
    for (unsigned group = 0; group < 3; group++) lengths[group] = server_u64(index + 24 + group * 8);
    for (unsigned code = 0; code < 256; code++) {
        float value = server_f32(index + 144 + code * 4);
        if (memcmp(&value, &server->palette[code], sizeof value)) valid = 0;
    }
    for (size_t position = 0; valid && position < count; position++) {
        const unsigned char *encoded = index + 1168 + position * 72;
        const ServerLayout *layout = &server_layout[position];
        ServerRecord record = {server_u32(encoded), server_u32(encoded + 4), server_u32(encoded + 8),
            server_u32(encoded + 12), server_u32(encoded + 16), server_u32(encoded + 20),
            server_u64(encoded + 24), server_u64(encoded + 32), NULL};
        uint64_t elements = (uint64_t)layout->rows * layout->columns;
        uint64_t wanted = layout->kind == 1 ? layout->rows * 4ULL + elements : elements * 4;
        if (record.id != layout->id || record.group != layout->group || record.kind != layout->kind ||
            record.rows != layout->rows || record.columns != layout->columns || record.count != elements ||
            record.length != wanted || record.offset != ends[layout->group] ||
            record.offset > lengths[layout->group] || record.length > lengths[layout->group] - record.offset ||
            record.length > SIZE_MAX) { valid = 0; break; }
        ends[record.group] += record.length;
        server->records[record.id] = record;
    }
    for (unsigned group = 0; group < 3; group++) if (ends[group] != lengths[group]) valid = 0;
    free(index);
    if (!valid) return 0;
    const char *names[] = {"common.bin", "kda.bin", "dense.bin"};
    for (unsigned group = 0; group < 3; group++) {
        if (!server_path(path, sizeof path, directory, names[group])) return 0;
        file = server_open_sized(path, lengths[group]);
        if (!file) return 0;
        for (unsigned id = 0; valid && id < SERVER_RECORDS; id++) {
            ServerRecord *record = &server->records[id];
            if (!server_needs(id) || record->group != group) continue;
            if (!server_read(file, record->offset, (size_t)record->length, &record->data)) { valid = 0; break; }
            unsigned floats = record->kind == 1 ? record->rows : record->count;
            for (unsigned coordinate = 0; coordinate < floats; coordinate++)
                if (!isfinite(server_f32(record->data + (size_t)coordinate * 4))) { valid = 0; break; }
        }
        if (fclose(file)) valid = 0;
        if (!valid) return 0;
    }
    return 1;
}

void server_close(Server *server)
{
    if (!server) return;
    for (unsigned id = 0; id < SERVER_RECORDS; id++) free(server->records[id].data);
    free(server->qkv);
    free(server);
}

int server_open(const char *qkv_path, const char *trunk_directory, Server **result)
{
    if (!result) return 0;
    *result = NULL;
    if (!qkv_path || !trunk_directory || sizeof(float) != 4 || sizeof(double) != 8 ||
        FLT_RADIX != 2 || FLT_MANT_DIG != 24 || DBL_MANT_DIG != 53) return 0;
    Server *server = calloc(1, sizeof *server);
    if (!server) return 0;
    if (!server_load_qkv(server, qkv_path) || !server_load_trunk(server, trunk_directory)) {
        server_close(server);
        return 0;
    }
    *result = server;
    return 1;
}

ServerSequence *server_sequence_create(const Server *server)
{
    if (!server) return NULL;
    ServerSequence *sequence = calloc(1, sizeof *sequence);
    if (sequence) sequence->owner = server;
    return sequence;
}

void server_sequence_reset(ServerSequence *sequence)
{
    if (!sequence) return;
    const Server *owner = sequence->owner;
    memset(sequence, 0, sizeof *sequence);
    sequence->owner = owner;
}

void server_sequence_close(ServerSequence *sequence)
{
    free(sequence);
}

static int server_finite(const float *values, size_t count)
{
    for (size_t coordinate = 0; coordinate < count; coordinate++)
        if (!isfinite(values[coordinate])) return 0;
    return 1;
}

static float server_inverse_rms(const float *input, unsigned width)
{
    double squares = 0.0;
    for (unsigned coordinate = 0; coordinate < width; coordinate++)
        squares += (double)input[coordinate] * (double)input[coordinate];
    return (float)(1.0 / sqrt(squares / (double)width + (double)1e-5f));
}

static void server_normalize(float *output, const float *input, const unsigned char *gains, unsigned width)
{
    float inverse = server_inverse_rms(input, width);
    for (unsigned coordinate = 0; coordinate < width; coordinate++)
        output[coordinate] = (server_f32(gains + coordinate * 4) * input[coordinate]) * inverse;
}

static float server_project_row(const float *input, const unsigned char *ids,
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

static void server_project(const Server *server, unsigned id, const float *input, float *output)
{
    const ServerRecord *record = &server->records[id];
    const unsigned char *ids = record->data + (size_t)record->rows * 4;
    for (unsigned row = 0; row < record->rows; row++)
        output[row] = server_project_row(input, ids + (size_t)row * record->columns,
            record->columns, server->palette, server_f32(record->data + row * 4));
}

static float server_sigmoid(float value)
{
    return 1.0f / (1.0f + expf(-value));
}

static float server_convolve(float raw, const unsigned char *taps, float *history)
{
    float value = server_f32(taps) * raw;
    for (unsigned previous = 0; previous < 3; previous++)
        value = value + server_f32(taps + (previous + 1) * 4) * history[previous];
    history[0] = history[1];
    history[1] = history[2];
    history[2] = raw;
    return value;
}

static void server_l2_heads(float *values)
{
    for (unsigned head = 0; head < SERVER_HEADS; head++) {
        float *block = values + head * SERVER_HEAD;
        double squares = 0.0;
        for (unsigned coordinate = 0; coordinate < SERVER_HEAD; coordinate++)
            squares += (double)block[coordinate] * (double)block[coordinate];
        float inverse = (float)(1.0 / sqrt(squares + (double)1e-6f));
        for (unsigned coordinate = 0; coordinate < SERVER_HEAD; coordinate++) block[coordinate] *= inverse;
    }
}

static void server_qkv(const Server *server, ServerSequence *sequence)
{
    for (unsigned row = 0; row < SERVER_ROWS; row++)
        for (unsigned component = 0; component < 3; component++) {
            const unsigned char *record = server->qkv + SERVER_QKV_PREFIX +
                ((size_t)row * 3 + component) * SERVER_QKV_COMPONENT;
            float raw = server_project_row(sequence->normalized, record + 20,
                SERVER_WIDTH, server->palette, server_f32(record));
            float convolved = server_convolve(raw, record + 4, sequence->history[component][row]);
            sequence->qkv[component][row] = convolved * server_sigmoid(convolved);
        }
    server_l2_heads(sequence->qkv[0]);
    server_l2_heads(sequence->qkv[1]);
}

static void server_decay(const Server *server, ServerSequence *sequence)
{
    server_project(server, 11, sequence->normalized, sequence->beta);
    server_project(server, 12, sequence->normalized, sequence->decay_hidden);
    server_project(server, 13, sequence->decay_hidden, sequence->decay);
    for (unsigned head = 0; head < SERVER_HEADS; head++) {
        sequence->beta[head] = server_sigmoid(sequence->beta[head]);
        float exponent = server_f32(server->records[39].data + head * 4);
        for (unsigned coordinate = 0; coordinate < SERVER_HEAD; coordinate++) {
            unsigned index = head * SERVER_HEAD + coordinate;
            float value = exponent * (sequence->decay[index] + server_f32(server->records[18].data + index * 4));
            sequence->decay[index] = expf(-5.0f * server_sigmoid(value));
        }
    }
}

static void server_update_attention(ServerSequence *sequence)
{
    const float query_scale = 1.0f / sqrtf(128.0f);
    for (unsigned head = 0; head < SERVER_HEADS; head++) {
        float prediction[SERVER_HEAD] = {0};
        float *output = sequence->attention + head * SERVER_HEAD;
        const float *query = sequence->qkv[0] + head * SERVER_HEAD;
        const float *key = sequence->qkv[1] + head * SERVER_HEAD;
        const float *value = sequence->qkv[2] + head * SERVER_HEAD;
        const float *decay = sequence->decay + head * SERVER_HEAD;
        for (unsigned row = 0; row < SERVER_HEAD; row++)
            for (unsigned column = 0; column < SERVER_HEAD; column++) {
                float state = decay[row] * sequence->recurrent[head][row][column];
                sequence->recurrent[head][row][column] = state;
                prediction[column] = prediction[column] + state * key[row];
            }
        memset(output, 0, SERVER_HEAD * sizeof *output);
        for (unsigned row = 0; row < SERVER_HEAD; row++) {
            float key_beta = key[row] * sequence->beta[head];
            float scaled_query = query[row] * query_scale;
            for (unsigned column = 0; column < SERVER_HEAD; column++) {
                float state = sequence->recurrent[head][row][column] + key_beta * (value[column] - prediction[column]);
                sequence->recurrent[head][row][column] = state;
                output[column] = output[column] + state * scaled_query;
            }
        }
    }
}

static void server_attention_output(const Server *server, ServerSequence *sequence)
{
    server_project(server, 6, sequence->normalized, sequence->gate);
    for (unsigned head = 0; head < SERVER_HEADS; head++) {
        float *block = sequence->attention + head * SERVER_HEAD;
        server_normalize(block, block, server->records[19].data, SERVER_HEAD);
    }
    for (unsigned coordinate = 0; coordinate < SERVER_ROWS; coordinate++)
        sequence->gate[coordinate] = sequence->attention[coordinate] * server_sigmoid(sequence->gate[coordinate]);
    server_project(server, 7, sequence->gate, sequence->residual);
}

static void server_aggregate(const Server *server, ServerSequence *sequence)
{
    const float *sources[] = {sequence->snapshot, sequence->residual};
    float scores[2], exponentials[2], weights[2];
    for (unsigned source = 0; source < 2; source++) {
        float inverse = server_inverse_rms(sources[source], SERVER_WIDTH);
        double score = 0.0;
        for (unsigned coordinate = 0; coordinate < SERVER_WIDTH; coordinate++) {
            float normalized = sources[source][coordinate] * inverse;
            score += (double)normalized * (double)server_f32(server->records[38].data + coordinate * 4);
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
    for (unsigned coordinate = 0; coordinate < SERVER_WIDTH; coordinate++) {
        float value = 0.0f;
        for (unsigned source = 0; source < 2; source++) value = value + weights[source] * sources[source][coordinate];
        sequence->aggregate[coordinate] = value;
    }
}

static void server_dense_activation(float *gate, const float *up)
{
    for (unsigned coordinate = 0; coordinate < SERVER_DENSE; coordinate++) {
        float activated = (4.0f * tanhf(gate[coordinate] / 4.0f)) * server_sigmoid(gate[coordinate]);
        float capped_up = 25.0f * tanhf(up[coordinate] / 25.0f);
        gate[coordinate] = activated * capped_up;
    }
}

static void server_dense(const Server *server, ServerSequence *sequence)
{
    server_normalize(sequence->postnorm, sequence->aggregate, server->records[5].data, SERVER_WIDTH);
    server_project(server, 26, sequence->postnorm, sequence->dense_gate);
    server_project(server, 27, sequence->postnorm, sequence->dense_up);
    server_dense_activation(sequence->dense_gate, sequence->dense_up);
    server_project(server, 28, sequence->dense_gate, sequence->dense_output);
    for (unsigned coordinate = 0; coordinate < SERVER_WIDTH; coordinate++)
        sequence->residual[coordinate] = sequence->residual[coordinate] + sequence->dense_output[coordinate];
}

int server_process(const Server *server, ServerSequence *sequence,
    const float input[SERVER_WIDTH], float output[SERVER_WIDTH], float snapshot[SERVER_WIDTH])
{
    if (!server || !sequence || sequence->owner != server || !input || !output || !snapshot ||
        output == snapshot || sequence->failed || sequence->positions == SIZE_MAX ||
        fegetround() != FE_TONEAREST || FLT_EVAL_METHOD != 0 || !server_finite(input, SERVER_WIDTH)) return 0;
    memcpy(sequence->snapshot, input, sizeof sequence->snapshot);
    server_normalize(sequence->normalized, sequence->snapshot, server->qkv + 88, SERVER_WIDTH);
    server_qkv(server, sequence);
    server_decay(server, sequence);
    server_update_attention(sequence);
    server_attention_output(server, sequence);
    server_aggregate(server, sequence);
    server_dense(server, sequence);
    if (!server_finite(sequence->residual, SERVER_WIDTH) ||
        !server_finite(&sequence->recurrent[0][0][0], SERVER_HEADS * SERVER_HEAD * SERVER_HEAD) ||
        !server_finite(&sequence->history[0][0][0], 3 * SERVER_ROWS * 3)) {
        sequence->failed = 1;
        return 0;
    }
    memcpy(output, sequence->residual, sizeof sequence->residual);
    memcpy(snapshot, sequence->snapshot, sizeof sequence->snapshot);
    sequence->positions++;
    return 1;
}

#ifndef SERVER_NO_MAIN
static int server_stream(Server *server)
{
    ServerSequence *sequence = server_sequence_create(server);
    if (!sequence) return 1;
    float input[SERVER_WIDTH], output[SERVER_WIDTH], snapshot[SERVER_WIDTH];
    int result = 0;
    for (;;) {
        int scanned = scanf("%f", &input[0]);
        if (scanned == EOF && !ferror(stdin)) break;
        if (scanned != 1) { result = 2; break; }
        for (unsigned coordinate = 1; coordinate < SERVER_WIDTH; coordinate++)
            if (scanf("%f", &input[coordinate]) != 1) { result = 2; break; }
        if (result) break;
        if (!server_process(server, sequence, input, output, snapshot)) { result = 1; break; }
        for (unsigned coordinate = 0; coordinate < SERVER_WIDTH; coordinate++)
            printf("%a%c", (double)output[coordinate], coordinate + 1 == SERVER_WIDTH ? '\n' : ' ');
        for (unsigned coordinate = 0; coordinate < SERVER_WIDTH; coordinate++)
            printf("%a%c", (double)snapshot[coordinate], coordinate + 1 == SERVER_WIDTH ? '\n' : ' ');
        if (fflush(stdout) || ferror(stdout)) { result = 1; break; }
    }
    if (result) fputs("server: incomplete, invalid or failed input vector\n", stderr);
    server_sequence_close(sequence);
    return result;
}

int main(int argc, char **argv)
{
    if (argc != 4 || (strcmp(argv[1], "--inspect") && strcmp(argv[1], "--stream"))) {
        fputs("usage: server --inspect|--stream QKV_FILE TRUNK_DIRECTORY\n", stderr);
        return 2;
    }
    Server *server = NULL;
    if (!server_open(argv[2], argv[3], &server)) {
        fputs("server: invalid or unreadable layer-0 datasets\n", stderr);
        return 1;
    }
    if (!strcmp(argv[1], "--stream")) {
        int result = server_stream(server);
        server_close(server);
        return result;
    }
    puts("layer 0: QKV operator and 13 additional prepared records loaded");
    for (unsigned id = 0; id < SERVER_RECORDS; id++) {
        const ServerRecord *record = &server->records[id];
        if (record->data) printf("record %u: %u x %u, %llu bytes\n", id, record->rows, record->columns,
            (unsigned long long)record->length);
    }
    server_close(server);
    return 0;
}
#endif