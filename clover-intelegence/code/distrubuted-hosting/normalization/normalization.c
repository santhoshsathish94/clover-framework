#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <fenv.h>
#include <math.h>
#include "json.h"

enum { NORMALIZATION_WIDTH = 7168, NORMALIZATION_SNAPSHOTS = 8,
    NORMALIZATION_SOURCES = 9, NORMALIZATION_TENSOR_BYTES = 14336 };

typedef struct {
    float gains[3][NORMALIZATION_WIDTH];
    float fold[NORMALIZATION_WIDTH];
} Normalization;

static int normalization_integer(JsonReader *reader, uint64_t *result)
{
    json_space(reader);
    if (reader->cursor == reader->end || *reader->cursor < '0' || *reader->cursor > '9') return 0;
    if (*reader->cursor == '0' && reader->cursor + 1 < reader->end && reader->cursor[1] >= '0' && reader->cursor[1] <= '9') return 0;
    uint64_t value = 0;
    while (reader->cursor < reader->end && *reader->cursor >= '0' && *reader->cursor <= '9') {
        unsigned digit = (unsigned)(*reader->cursor++ - '0');
        if (value > (UINT64_MAX - digit) / 10) return 0;
        value = value * 10 + digit;
    }
    *result = value;
    return 1;
}

static int normalization_string(JsonReader *reader, const char *expected)
{
    unsigned char *text = NULL;
    size_t length = 0;
    if (!json_string(reader, &text, &length)) return 0;
    int valid = key_is(text, length, expected);
    free(text);
    return valid;
}

static int normalization_shape(JsonReader *reader, unsigned tensor)
{
    uint64_t dimension;
    if (!json_take(reader, '[') || !normalization_integer(reader, &dimension)) return 0;
    if (tensor == 1) {
        if (dimension != 1 || !json_take(reader, ',') || !normalization_integer(reader, &dimension)) return 0;
    }
    return dimension == NORMALIZATION_WIDTH && json_take(reader, ']');
}

static int normalization_decode_tensor(JsonReader *reader, float *values)
{
    unsigned char *encoded = NULL;
    size_t length = 0, written = 0;
    if (!json_string(reader, &encoded, &length)) return 0;
    unsigned char decoded[NORMALIZATION_TENSOR_BYTES + 1];
    int valid = decode_base64(encoded, length, decoded, sizeof decoded, &written) && written == NORMALIZATION_TENSOR_BYTES;
    free(encoded);
    if (!valid) return 0;
    for (unsigned coordinate = 0; coordinate < NORMALIZATION_WIDTH; coordinate++) {
        uint32_t bits = ((uint32_t)decoded[coordinate * 2] | (uint32_t)decoded[coordinate * 2 + 1] << 8) << 16;
        memcpy(values + coordinate, &bits, sizeof bits);
        if (!isfinite(values[coordinate])) return 0;
    }
    return 1;
}

static int normalization_tensor(JsonReader *reader, unsigned tensor, Normalization *model)
{
    if (!json_take(reader, '{') || json_take(reader, '}')) return 0;
    unsigned seen = 0;
    for (;;) {
        unsigned char *key = NULL;
        size_t length;
        if (!json_string(reader, &key, &length)) return 0;
        unsigned field = key_is(key, length, "dtype") ? 1 : key_is(key, length, "shape") ? 2 :
            key_is(key, length, "nbytes") ? 4 : key_is(key, length, "data_base64") ? 8 : 0;
        free(key);
        if (!json_take(reader, ':') || (seen & field)) return 0;
        uint64_t bytes;
        int valid;
        if (field == 1) valid = normalization_string(reader, "BF16");
        else if (field == 2) valid = normalization_shape(reader, tensor);
        else if (field == 4) valid = normalization_integer(reader, &bytes) && bytes == NORMALIZATION_TENSOR_BYTES;
        else if (field == 8) valid = normalization_decode_tensor(reader, model->gains[tensor]);
        else valid = json_skip(reader);
        if (!valid) return 0;
        seen |= field;
        if (json_take(reader, '}')) return seen == 15;
        if (!json_take(reader, ',')) return 0;
    }
}

static int normalization_tensors(JsonReader *reader, Normalization *model)
{
    static const char *names[] = {"language_model.model.output_attn_res_norm.weight",
        "language_model.model.output_attn_res_proj.weight", "language_model.model.norm.weight"};
    if (!json_take(reader, '{') || json_take(reader, '}')) return 0;
    unsigned seen = 0;
    for (;;) {
        unsigned char *key = NULL;
        size_t length;
        if (!json_string(reader, &key, &length)) return 0;
        unsigned tensor = 0;
        while (tensor < 3 && !key_is(key, length, names[tensor])) tensor++;
        free(key);
        if (tensor == 3 || (seen & (1U << tensor)) || !json_take(reader, ':') || !normalization_tensor(reader, tensor, model)) return 0;
        seen |= 1U << tensor;
        if (json_take(reader, '}')) return seen == 7;
        if (!json_take(reader, ',')) return 0;
    }
}

static int normalization_parse(const unsigned char *data, size_t bytes, Normalization *model)
{
    JsonReader reader = {data, data + bytes, 0};
    if (!json_take(&reader, '{') || json_take(&reader, '}')) return 0;
    unsigned seen = 0;
    for (;;) {
        unsigned char *key = NULL;
        size_t length;
        if (!json_string(&reader, &key, &length)) return 0;
        unsigned field = key_is(key,length,"format") ? 1 : key_is(key,length,"byte_order") ? 2 :
            key_is(key,length,"encoding") ? 4 : key_is(key,length,"tensor_count") ? 8 :
            key_is(key,length,"payload_bytes") ? 16 : key_is(key,length,"tensors") ? 32 : 0;
        free(key);
        if (!json_take(&reader, ':') || (seen & field)) return 0;
        uint64_t value;
        int valid;
        if (field == 1) valid = normalization_string(&reader,"clover-bf16-tensors-v1");
        else if (field == 2) valid = normalization_string(&reader,"little");
        else if (field == 4) valid = normalization_string(&reader,"base64");
        else if (field == 8) valid = normalization_integer(&reader,&value) && value == 3;
        else if (field == 16) valid = normalization_integer(&reader,&value) && value == 3 * NORMALIZATION_TENSOR_BYTES;
        else if (field == 32) valid = normalization_tensors(&reader,model);
        else valid = json_skip(&reader);
        if (!valid) return 0;
        seen |= field;
        if (json_take(&reader, '}')) break;
        if (!json_take(&reader, ',')) return 0;
    }
    json_space(&reader);
    return seen == 63 && reader.cursor == reader.end;
}

static int normalization_environment(void)
{
    const float one = 1.0f;
    uint32_t bits = 0;
    if (sizeof(float) != 4 || sizeof(double) != 8 || FLT_RADIX != 2 || FLT_MANT_DIG != 24 ||
        DBL_MANT_DIG != 53 || FLT_EVAL_METHOD != 0 || fegetround() != FE_TONEAREST) return 0;
    memcpy(&bits, &one, 4);
    return bits == UINT32_C(0x3f800000);
}

static int normalization_prepare_fold(Normalization *model)
{
    for (unsigned coordinate = 0; coordinate < NORMALIZATION_WIDTH; coordinate++) {
        model->fold[coordinate] = model->gains[0][coordinate] * model->gains[1][coordinate];
        if (!isfinite(model->fold[coordinate])) return 0;
    }
    return 1;
}

void normalization_close(Normalization *model)
{
    free(model);
}

int normalization_open(const char *path, Normalization **result)
{
    if (!result) return 0;
    *result = NULL;
    if (!path || !normalization_environment()) return 0;
    FILE *file = fopen(path, "rb");
    if (!file) return 0;
    if (fseek(file,0,SEEK_END)) { fclose(file); return 0; }
    long size = ftell(file);
    if (size <= 0 || size > 1024 * 1024 || fseek(file,0,SEEK_SET)) { fclose(file); return 0; }
    unsigned char *data = malloc((size_t)size);
    Normalization *model = calloc(1,sizeof *model);
    if (!data || !model) { free(data); free(model); fclose(file); return 0; }
    int valid = fread(data,1,(size_t)size,file) == (size_t)size && !ferror(file);
    if (fclose(file)) valid = 0;
    if (valid) valid = normalization_parse(data,(size_t)size,model) && normalization_prepare_fold(model);
    free(data);
    if (!valid) { normalization_close(model); return 0; }
    *result = model;
    return 1;
}

static int normalization_finite(const float *values, size_t count)
{
    for (size_t coordinate = 0; coordinate < count; coordinate++) if (!isfinite(values[coordinate])) return 0;
    return 1;
}

static float normalization_inverse_rms(const float *input)
{
    double squares = 0.0;
    for (unsigned coordinate = 0; coordinate < NORMALIZATION_WIDTH; coordinate++)
        squares += (double)input[coordinate] * (double)input[coordinate];
    return (float)(1.0 / sqrt(squares / (double)NORMALIZATION_WIDTH + (double)1e-5f));
}

static int normalization_score_sources(const Normalization *model, const float *const sources[9], float scores[9])
{
    for (unsigned source = 0; source < NORMALIZATION_SOURCES; source++) {
        float inverse = normalization_inverse_rms(sources[source]);
        double score = 0.0;
        for (unsigned coordinate = 0; coordinate < NORMALIZATION_WIDTH; coordinate++) {
            float normalized = sources[source][coordinate] * inverse;
            score += (double)normalized * (double)model->fold[coordinate];
        }
        scores[source] = (float)score;
    }
    return normalization_finite(scores,NORMALIZATION_SOURCES);
}

static int normalization_softmax(const float scores[9], float weights[9])
{
    float maximum = scores[0], exponentials[NORMALIZATION_SOURCES];
    for (unsigned source = 1; source < NORMALIZATION_SOURCES; source++)
        if (scores[source] > maximum) maximum = scores[source];
    double total = 0.0;
    for (unsigned source = 0; source < NORMALIZATION_SOURCES; source++) {
        exponentials[source] = expf(scores[source] - maximum);
        total += (double)exponentials[source];
    }
    if (!(total > 0.0) || !isfinite(total)) return 0;
    for (unsigned source = 0; source < NORMALIZATION_SOURCES; source++)
        weights[source] = (float)((double)exponentials[source] / total);
    return 1;
}

static void normalization_aggregate(const float *const sources[9], const float weights[9], float output[7168])
{
    for (unsigned coordinate = 0; coordinate < NORMALIZATION_WIDTH; coordinate++) {
        float value = 0.0f;
        for (unsigned source = 0; source < NORMALIZATION_SOURCES; source++)
            value = value + weights[source] * sources[source][coordinate];
        output[coordinate] = value;
    }
}

static void normalization_apply_rms(const Normalization *model, float values[7168])
{
    float inverse = normalization_inverse_rms(values);
    for (unsigned coordinate = 0; coordinate < NORMALIZATION_WIDTH; coordinate++)
        values[coordinate] = (model->gains[2][coordinate] * values[coordinate]) * inverse;
}

int normalization_process(const Normalization *model, const float residual[7168],
    const float *snapshots, unsigned snapshot_count, float output[7168])
{
    if (!model || !residual || !snapshots || !output || snapshot_count != NORMALIZATION_SNAPSHOTS ||
        !normalization_environment() || !normalization_finite(residual,NORMALIZATION_WIDTH) ||
        !normalization_finite(snapshots,NORMALIZATION_WIDTH * NORMALIZATION_SNAPSHOTS)) return 0;
    const float *sources[NORMALIZATION_SOURCES];
    for (unsigned source = 0; source < NORMALIZATION_SNAPSHOTS; source++) sources[source] = snapshots + source * NORMALIZATION_WIDTH;
    sources[NORMALIZATION_SNAPSHOTS] = residual;
    float scores[NORMALIZATION_SOURCES], weights[NORMALIZATION_SOURCES], result[NORMALIZATION_WIDTH];
    if (!normalization_score_sources(model,sources,scores) || !normalization_softmax(scores,weights)) return 0;
    normalization_aggregate(sources,weights,result);
    if (!normalization_finite(result,NORMALIZATION_WIDTH)) return 0;
    normalization_apply_rms(model,result);
    if (!normalization_finite(result,NORMALIZATION_WIDTH)) return 0;
    memcpy(output,result,sizeof result);
    return 1;
}

#ifndef NORMALIZATION_NO_MAIN
static int normalization_receive(float residual[7168], float *snapshots)
{
    int status = scanf("%f",residual);
    if (status == EOF && !ferror(stdin)) return 0;
    if (status != 1) return -1;
    for (unsigned coordinate = 1; coordinate < NORMALIZATION_WIDTH; coordinate++)
        if (scanf("%f",residual + coordinate) != 1) return -1;
    for (unsigned coordinate = 0; coordinate < NORMALIZATION_WIDTH * NORMALIZATION_SNAPSHOTS; coordinate++)
        if (scanf("%f",snapshots + coordinate) != 1) return -1;
    return 1;
}

static int normalization_send(const float output[7168])
{
    for (unsigned coordinate = 0; coordinate < NORMALIZATION_WIDTH; coordinate++)
        printf("%a%c",(double)output[coordinate],coordinate + 1 == NORMALIZATION_WIDTH ? '\n' : ' ');
    return fflush(stdout) == 0 && !ferror(stdout);
}

static int normalization_stream(const Normalization *model)
{
    float residual[NORMALIZATION_WIDTH], snapshots[NORMALIZATION_WIDTH * NORMALIZATION_SNAPSHOTS], output[NORMALIZATION_WIDTH];
    for (;;) {
        int status = normalization_receive(residual,snapshots);
        if (!status) return 0;
        if (status < 0) { fputs("normalization: incomplete or malformed bundle\n",stderr); return 2; }
        if (!normalization_process(model,residual,snapshots,NORMALIZATION_SNAPSHOTS,output) || !normalization_send(output)) {
            fputs("normalization: invalid input or failed output\n",stderr); return 1;
        }
    }
}

static void normalization_inspect(void)
{
    puts("normalization: three BF16 tensors loaded; 8 snapshots + residual -> 7168 normalized values");
}

int main(int argc, char **argv)
{
    if ((argc != 2 && argc != 3) || (strcmp(argv[1],"--inspect") && strcmp(argv[1],"--stream"))) {
        fputs("usage: normalization --inspect|--stream [LEAVES_JSON]\n",stderr);
        return 2;
    }
    Normalization *model = NULL;
    if (!normalization_open(argc == 3 ? argv[2] : "dataset/leaves.json",&model)) { fputs("normalization: invalid leaves dataset\n",stderr); return 1; }
    int result = 0;
    if (!strcmp(argv[1],"--stream")) result = normalization_stream(model);
    else normalization_inspect();
    normalization_close(model);
    return result;
}
#endif