#ifndef CLOVER_GENERATION_H
#define CLOVER_GENERATION_H
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "json.h"

enum { GENERATION_CAPACITY = 256, GENERATION_VOCAB = 163840 };
typedef struct {
    unsigned max_input_tokens, max_output_tokens, eos_token_id;
    unsigned head_cache_mib, memory_reserve_mib;
    unsigned expert_pipeline_mib;
    unsigned expert_result_cache_mib;
} GenerationConfig;
typedef struct { unsigned tokens[GENERATION_CAPACITY], count, max_new_tokens; } GenerationRequest;

static int generation_uint(JsonReader *reader, unsigned *result)
{
    json_space(reader);
    if (reader->cursor == reader->end || *reader->cursor < '0' || *reader->cursor > '9') return 0;
    unsigned value = 0;
    int leading_zero = *reader->cursor == '0';
    size_t digits = 0;
    while (reader->cursor < reader->end && *reader->cursor >= '0' && *reader->cursor <= '9') {
        unsigned digit = *reader->cursor++ - '0';
        if (value > (UINT32_MAX - digit) / 10 || (leading_zero && digits)) return 0;
        value = value * 10 + digit;
        digits++;
    }
    *result = value;
    return 1;
}

static int generation_config_parse(const unsigned char *data, size_t bytes, GenerationConfig *config)
{
    JsonReader reader = {data, data + bytes, 0};
    GenerationConfig parsed = {.head_cache_mib=2240,.memory_reserve_mib=8192,.expert_pipeline_mib=72,.expert_result_cache_mib=256};
    unsigned seen = 0;
    if (!json_take(&reader, '{')) return 0;
    do {
        unsigned char *key = NULL;
        size_t length;
        if (!json_string(&reader, &key, &length)) return 0;
        unsigned field = key_is(key,length,"max_input_tokens") ? 1 : key_is(key,length,"max_output_tokens") ? 2 :
            key_is(key,length,"eos_token_id") ? 4 : key_is(key,length,"head_cache_mib") ? 8 :
            key_is(key,length,"memory_reserve_mib") ? 64 : key_is(key,length,"expert_pipeline_mib") ? 128 :
            key_is(key,length,"expert_result_cache_mib") ? 256 : 0;
        free(key);
        unsigned value;
        if (!field || (seen & field) || !json_take(&reader, ':') || !generation_uint(&reader, &value)) return 0;
        if (field == 1) parsed.max_input_tokens = value;
        if (field == 2) parsed.max_output_tokens = value;
        if (field == 4) parsed.eos_token_id = value;
        if (field == 8) parsed.head_cache_mib = value;
        if (field == 64) parsed.memory_reserve_mib = value;
        if (field == 128) parsed.expert_pipeline_mib = value;
        if (field == 256) parsed.expert_result_cache_mib = value;
        seen |= field;
        if (json_take(&reader, '}')) break;
        if (!json_take(&reader, ',')) return 0;
    } while (1);
    json_space(&reader);
    if (reader.cursor != reader.end || (seen & 7) != 7 || !parsed.max_input_tokens || !parsed.max_output_tokens ||
        parsed.max_input_tokens > GENERATION_CAPACITY || parsed.max_output_tokens > GENERATION_CAPACITY ||
        parsed.max_input_tokens + parsed.max_output_tokens > GENERATION_CAPACITY || parsed.eos_token_id >= GENERATION_VOCAB ||
        parsed.head_cache_mib>4096 ||
        parsed.memory_reserve_mib<1024 || parsed.memory_reserve_mib>65536 || parsed.expert_pipeline_mib>256 ||
        parsed.expert_result_cache_mib>8192) return 0;
    *config = parsed;
    return 1;
}

static int generation_config_load(const char *path, GenerationConfig *config)
{
    FILE *file = fopen(path,"rb");
    if (!file) return 0;
    unsigned char data[4096];
    size_t length = fread(data,1,sizeof data,file);
    int valid = !ferror(file) && length < sizeof data && generation_config_parse(data,length,config);
    if (fclose(file)) valid = 0;
    return valid;
}

static int generation_request_parse(const unsigned char *data, size_t bytes,
    const GenerationConfig *config, GenerationRequest *request)
{
    JsonReader reader = {data,data + bytes,0};
    GenerationRequest parsed = { .max_new_tokens = config->max_output_tokens };
    unsigned seen = 0;
    if (!json_take(&reader,'{')) return 0;
    do {
        unsigned char *key = NULL;
        size_t length;
        if (!json_string(&reader,&key,&length)) return 0;
        unsigned field = key_is(key,length,"input_ids") ? 1 : key_is(key,length,"max_new_tokens") ? 2 : 0;
        free(key);
        if (!field || (seen & field) || !json_take(&reader,':')) return 0;
        if (field == 1) {
            if (!json_take(&reader,'[')) return 0;
            do {
                unsigned token;
                if (parsed.count >= config->max_input_tokens || !generation_uint(&reader,&token) || token >= GENERATION_VOCAB) return 0;
                parsed.tokens[parsed.count++] = token;
                if (json_take(&reader,']')) break;
                if (!json_take(&reader,',')) return 0;
            } while (1);
        } else if (!generation_uint(&reader,&parsed.max_new_tokens)) return 0;
        seen |= field;
        if (json_take(&reader,'}')) break;
        if (!json_take(&reader,',')) return 0;
    } while (1);
    json_space(&reader);
    if (reader.cursor != reader.end || !(seen & 1) || !parsed.max_new_tokens || parsed.max_new_tokens > config->max_output_tokens) return 0;
    *request = parsed;
    return 1;
}

typedef int (*GenerationStep)(void *state, unsigned token, unsigned position, int project, unsigned *next);
typedef void (*GenerationEmit)(void *state, unsigned token, unsigned index);

static int generation_run(const GenerationConfig *config, const GenerationRequest *request,
    GenerationStep step, GenerationEmit emit, void *state, unsigned *produced, int *eos)
{
    unsigned next = 0;
    *produced = 0; *eos = 0;
    if (!request->count || request->count > config->max_input_tokens || !request->max_new_tokens ||
        request->max_new_tokens > config->max_output_tokens) return 0;
    for (unsigned position = 0; position < request->count; position++)
        if (!step(state,request->tokens[position],position,position + 1 == request->count,&next)) return 0;
    for (unsigned index = 0; index < request->max_new_tokens; index++) {
        if (next >= GENERATION_VOCAB) return 0;
        emit(state,next,index);
        *produced = index + 1;
        if (next == config->eos_token_id) { *eos = 1; return 1; }
        if (*produced < request->max_new_tokens &&
            !step(state,next,request->count + index,1,&next)) return 0;
    }
    return 1;
}
#endif