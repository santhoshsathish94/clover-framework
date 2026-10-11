#ifndef SERVER_ENDS_H
#define SERVER_ENDS_H
/* Token to embedding on the way in, vector to token on the way out. Derived from the
   retired client stage; the server owns both ends of the chain, so nothing else does. */
/* json.h takes its prerequisites from whoever includes it, so they go first here. */
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../normalization/json.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { TOKENIZER_BASE = 163584, TOKENIZER_IDS = 163840, TOKENIZER_HASH = 524288 };

typedef enum {
    TOKENIZER_OK, TOKENIZER_NOT_FOUND, TOKENIZER_INVALID_ARGUMENT,
    TOKENIZER_INVALID_DATA, TOKENIZER_IO_ERROR, TOKENIZER_NO_MEMORY
} TokenizerStatus;

typedef struct {
    const unsigned char *bytes;
    size_t length;
} TokenText;

typedef struct {
    TokenText *tokens;
    uint32_t *lookup;
    unsigned char *vocabulary;
    size_t count;
} Tokenizer;


static uint32_t read_u32(const unsigned char *bytes)
{
    return (uint32_t)bytes[0] | (uint32_t)bytes[1] << 8 |
           (uint32_t)bytes[2] << 16 | (uint32_t)bytes[3] << 24;
}

static TokenizerStatus read_file(const char *path, size_t maximum, unsigned char **data, size_t *length)
{
    FILE *file = fopen(path, "rb");
    if (!file) return TOKENIZER_IO_ERROR;
    if (fseek(file, 0, SEEK_END)) { fclose(file); return TOKENIZER_IO_ERROR; }
    long size = ftell(file);
    if (size < 0 || (unsigned long)size > maximum) { fclose(file); return TOKENIZER_INVALID_DATA; }
    if (fseek(file, 0, SEEK_SET)) { fclose(file); return TOKENIZER_IO_ERROR; }
    unsigned char *buffer = malloc((size_t)size + 1);
    if (!buffer) { fclose(file); return TOKENIZER_NO_MEMORY; }
    size_t received = fread(buffer, 1, (size_t)size, file);
    int failed = ferror(file);
    int closed = fclose(file);
    if (received != (size_t)size || failed || closed) { free(buffer); return TOKENIZER_IO_ERROR; }
    buffer[size] = 0;
    *data = buffer;
    *length = (size_t)size;
    return TOKENIZER_OK;
}

static int decimal_id(const unsigned char *text, size_t length, uint32_t *id)
{
    if (!length) return 0;
    uint32_t value = 0;
    for (size_t index = 0; index < length; index++) {
        if (text[index] < '0' || text[index] > '9' || value > (TOKENIZER_IDS - 1U) / 10) return 0;
        value = value * 10 + (unsigned)(text[index] - '0');
        if (value >= TOKENIZER_IDS) return 0;
    }
    *id = value;
    return 1;
}

static int load_added_tokens(JsonReader *reader, Tokenizer *tokenizer)
{
    if (!json_take(reader, '{')) return 0;
    if (json_take(reader, '}')) return 1;
    do {
        unsigned char *name = NULL; size_t length; uint32_t id;
        if (!json_string(reader, &name, &length)) return 0;
        int valid = decimal_id(name, length, &id);
        free(name);
        if (!valid || id < TOKENIZER_BASE || tokenizer->tokens[id].bytes ||
            !json_take(reader, ':') || !json_take(reader, '{')) return 0;
        unsigned char *content = NULL; size_t content_length = 0;
        if (json_take(reader, '}')) return 0;
        do {
            if (!json_string(reader, &name, &length)) { free(content); return 0; }
            int content_key = key_is(name, length, "content");
            free(name);
            if (!json_take(reader, ':')) { free(content); return 0; }
            if (content_key) {
                if (content || !json_string(reader, &content, &content_length)) { free(content); return 0; }
            } else if (!json_skip(reader)) { free(content); return 0; }
            if (json_take(reader, '}')) break;
            if (!json_take(reader, ',')) { free(content); return 0; }
        } while (1);
        if (!content || !content_length) { free(content); return 0; }
        tokenizer->tokens[id] = (TokenText){content, content_length};
        tokenizer->count++;
        if (json_take(reader, '}')) return 1;
    } while (json_take(reader, ','));
    return 0;
}

static int load_config(Tokenizer *tokenizer, const unsigned char *data, size_t length)
{
    JsonReader reader = {data, data + length, 0};
    if (!json_take(&reader, '{')) return 0;
    int found = 0;
    if (json_take(&reader, '}')) return 0;
    do {
        unsigned char *key = NULL; size_t key_length;
        if (!json_string(&reader, &key, &key_length)) return 0;
        int added = key_is(key, key_length, "added_tokens_decoder");
        free(key);
        if (!json_take(&reader, ':')) return 0;
        if (added) {
            if (found || !load_added_tokens(&reader, tokenizer)) return 0;
            found = 1;
        } else if (!json_skip(&reader)) return 0;
        if (json_take(&reader, '}')) break;
        if (!json_take(&reader, ',')) return 0;
    } while (1);
    json_space(&reader);
    return found && reader.cursor == reader.end;
}

static uint32_t text_hash(const unsigned char *bytes, size_t length)
{
    uint32_t hash = UINT32_C(2166136261);
    for (size_t index = 0; index < length; index++) hash = (hash ^ bytes[index]) * UINT32_C(16777619);
    return hash;
}

void tokenizer_close(Tokenizer *tokenizer)
{
    if (!tokenizer) return;
    if (tokenizer->tokens) {
        for (uint32_t id = TOKENIZER_BASE; id < TOKENIZER_IDS; id++) free((void *)tokenizer->tokens[id].bytes);
    }
    free(tokenizer->tokens); free(tokenizer->lookup); free(tokenizer->vocabulary); free(tokenizer);
}

TokenizerStatus tokenizer_open(const char *model_path, const char *vocabulary_path, const char *config_path, Tokenizer **result)
{
    if (!result) return TOKENIZER_INVALID_ARGUMENT;
    *result = NULL;
    if (!model_path || !vocabulary_path || !config_path) return TOKENIZER_INVALID_ARGUMENT;
    Tokenizer *tokenizer = calloc(1, sizeof(*tokenizer));
    if (!tokenizer) return TOKENIZER_NO_MEMORY;
    unsigned char *model = NULL, *config = NULL;
    size_t model_bytes = 0, vocabulary_bytes = 0, config_bytes = 0;
    TokenizerStatus status = TOKENIZER_NO_MEMORY;
    tokenizer->tokens = calloc(TOKENIZER_IDS, sizeof(*tokenizer->tokens));
    tokenizer->lookup = calloc(TOKENIZER_HASH, sizeof(*tokenizer->lookup));
    if (!tokenizer->tokens || !tokenizer->lookup) goto done;
    status = read_file(model_path, 16 * 1024 * 1024, &model, &model_bytes);
    if (status != TOKENIZER_OK) goto done;
    status = read_file(vocabulary_path, 16 * 1024 * 1024, &tokenizer->vocabulary, &vocabulary_bytes);
    if (status != TOKENIZER_OK) goto done;
    status = read_file(config_path, 1024 * 1024, &config, &config_bytes);
    if (status != TOKENIZER_OK) goto done;
    status = TOKENIZER_INVALID_DATA;
    size_t prefix = (TOKENIZER_IDS + 1ULL) * 4;
    if (vocabulary_bytes < prefix || read_u32(tokenizer->vocabulary) != 0) goto done;
    for (uint32_t id = 0; id < TOKENIZER_IDS; id++) {
        uint32_t start = read_u32(tokenizer->vocabulary + id * 4);
        uint32_t end = read_u32(tokenizer->vocabulary + (id + 1) * 4);
        if (end < start || end > vocabulary_bytes - prefix) goto done;
    }
    if (read_u32(tokenizer->vocabulary + TOKENIZER_IDS * 4) != vocabulary_bytes - prefix) goto done;
    const unsigned char *cursor = model, *end_model = model + model_bytes;
    while (cursor < end_model) {
        const unsigned char *line = cursor;
        while (cursor < end_model && *cursor != '\n') cursor++;
        const unsigned char *end = cursor;
        if (cursor < end_model) cursor++;
        if (end > line && end[-1] == '\r') end--;
        if (end == line) continue;
        const unsigned char *space = memchr(line, ' ', (size_t)(end - line));
        uint32_t id;
        if (!space || !decimal_id(space + 1, (size_t)(end - space - 1), &id) ||
            id >= TOKENIZER_BASE || tokenizer->tokens[id].bytes) goto done;
        unsigned char decoded[4096]; size_t length;
        if (!decode_base64(line, (size_t)(space - line), decoded, sizeof(decoded), &length)) goto done;
        uint32_t start = read_u32(tokenizer->vocabulary + id * 4);
        uint32_t finish = read_u32(tokenizer->vocabulary + (id + 1) * 4);
        if (!length || length != finish - start || memcmp(decoded, tokenizer->vocabulary + prefix + start, length)) goto done;
        tokenizer->tokens[id] = (TokenText){tokenizer->vocabulary + prefix + start, length};
        tokenizer->count++;
    }
    if (tokenizer->count != TOKENIZER_BASE || !load_config(tokenizer, config, config_bytes)) goto done;
    for (uint32_t id = 0; id < TOKENIZER_IDS; id++) {
        TokenText token = tokenizer->tokens[id];
        if (!token.bytes) continue;
        uint32_t bucket = text_hash(token.bytes, token.length) & (TOKENIZER_HASH - 1);
        while (tokenizer->lookup[bucket]) {
            TokenText present = tokenizer->tokens[tokenizer->lookup[bucket] - 1];
            if (present.length == token.length && !memcmp(present.bytes, token.bytes, token.length)) goto done;
            bucket = (bucket + 1) & (TOKENIZER_HASH - 1);
        }
        tokenizer->lookup[bucket] = id + 1;
    }
    status = TOKENIZER_OK;
done:
    free(model); free(config);
    if (status == TOKENIZER_OK) *result = tokenizer;
    else tokenizer_close(tokenizer);
    return status;
}

TokenizerStatus word_to_id(const Tokenizer *tokenizer, const void *text, size_t length, uint32_t *id)
{
    if (!tokenizer || !text || !id) return TOKENIZER_INVALID_ARGUMENT;
    uint32_t bucket = text_hash(text, length) & (TOKENIZER_HASH - 1);
    while (tokenizer->lookup[bucket]) {
        uint32_t candidate = tokenizer->lookup[bucket] - 1;
        TokenText token = tokenizer->tokens[candidate];
        if (token.length == length && !memcmp(token.bytes, text, length)) { *id = candidate; return TOKENIZER_OK; }
        bucket = (bucket + 1) & (TOKENIZER_HASH - 1);
    }
    return TOKENIZER_NOT_FOUND;
}

TokenizerStatus id_to_word(const Tokenizer *tokenizer, uint32_t id, TokenText *text)
{
    if (!tokenizer || !text) return TOKENIZER_INVALID_ARGUMENT;
    if (id >= TOKENIZER_IDS || !tokenizer->tokens[id].bytes) return TOKENIZER_NOT_FOUND;
    *text = tokenizer->tokens[id];
    return TOKENIZER_OK;
}

#include "numeric-table.h"

int server_output(ServerOutput *output,const Tokenizer *tokenizer,
    const float vector[SERVER_TABLE_WIDTH],uint32_t *token,TokenText *text)
{
    if(!tokenizer||!token||!text)return 0;
    uint32_t selected;
    if(!server_output_project(output,vector,NULL,&selected))return 0;
    TokenText result;
    if(id_to_word(tokenizer,selected,&result)!=TOKENIZER_OK)return 0;
    *token=selected;*text=result;return 1;
}
#endif
