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

typedef struct {
    const unsigned char *cursor, *end;
    unsigned depth;
} JsonReader;

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

static int base64_digit(unsigned char value)
{
    if (value >= 'A' && value <= 'Z') return value - 'A';
    if (value >= 'a' && value <= 'z') return value - 'a' + 26;
    if (value >= '0' && value <= '9') return value - '0' + 52;
    if (value == '+') return 62;
    if (value == '/') return 63;
    return -1;
}

static int decode_base64(const unsigned char *text, size_t length, unsigned char *output, size_t capacity, size_t *written)
{
    if (!length || length % 4 || length / 4 > capacity / 3) return 0;
    size_t count = 0;
    for (size_t offset = 0; offset < length; offset += 4) {
        int first = base64_digit(text[offset]), second = base64_digit(text[offset + 1]);
        int third = text[offset + 2] == '=' ? 0 : base64_digit(text[offset + 2]);
        int fourth = text[offset + 3] == '=' ? 0 : base64_digit(text[offset + 3]);
        int padding = (text[offset + 2] == '=') + (text[offset + 3] == '=');
        if (first < 0 || second < 0 || third < 0 || fourth < 0 ||
            (padding && offset + 4 != length) ||
            (text[offset + 2] == '=' && text[offset + 3] != '=') ||
            (padding == 2 && (second & 15)) || (padding == 1 && (third & 3))) return 0;
        uint32_t packed = (uint32_t)first << 18 | (uint32_t)second << 12 | (uint32_t)third << 6 | (uint32_t)fourth;
        output[count++] = (unsigned char)(packed >> 16);
        if (padding < 2) output[count++] = (unsigned char)(packed >> 8);
        if (!padding) output[count++] = (unsigned char)packed;
    }
    *written = count;
    return 1;
}

static void json_space(JsonReader *reader)
{
    while (reader->cursor < reader->end && (*reader->cursor == ' ' || *reader->cursor == '\t' ||
        *reader->cursor == '\n' || *reader->cursor == '\r')) reader->cursor++;
}

static int json_take(JsonReader *reader, unsigned char expected)
{
    json_space(reader);
    if (reader->cursor == reader->end || *reader->cursor != expected) return 0;
    reader->cursor++;
    return 1;
}

static int json_hex4(JsonReader *reader, uint32_t *value)
{
    if ((size_t)(reader->end - reader->cursor) < 4) return 0;
    *value = 0;
    for (unsigned digit = 0; digit < 4; digit++) {
        unsigned char character = *reader->cursor++;
        unsigned number;
        if (character >= '0' && character <= '9') number = character - '0';
        else if (character >= 'a' && character <= 'f') number = character - 'a' + 10;
        else if (character >= 'A' && character <= 'F') number = character - 'A' + 10;
        else return 0;
        *value = *value * 16 + number;
    }
    return 1;
}

static int json_string(JsonReader *reader, unsigned char **text, size_t *length)
{
    if (!json_take(reader, '"')) return 0;
    unsigned char *output = malloc((size_t)(reader->end - reader->cursor) + 1);
    if (!output) return 0;
    size_t count = 0;
    while (reader->cursor < reader->end) {
        unsigned char character = *reader->cursor++;
        if (character == '"') {
            output[count] = 0;
            *text = output; *length = count;
            return 1;
        }
        if (character < 32) break;
        if (character != '\\') { output[count++] = character; continue; }
        if (reader->cursor == reader->end) break;
        character = *reader->cursor++;
        if (character == '"' || character == '/' || character == '\\') output[count++] = character;
        else if (character == 'b') output[count++] = '\b';
        else if (character == 'f') output[count++] = '\f';
        else if (character == 'n') output[count++] = '\n';
        else if (character == 'r') output[count++] = '\r';
        else if (character == 't') output[count++] = '\t';
        else if (character == 'u') {
            uint32_t codepoint;
            if (!json_hex4(reader, &codepoint)) break;
            if (codepoint >= 0xd800 && codepoint <= 0xdbff) {
                uint32_t low;
                if ((size_t)(reader->end - reader->cursor) < 6 || reader->cursor[0] != '\\' || reader->cursor[1] != 'u') break;
                reader->cursor += 2;
                if (!json_hex4(reader, &low) || low < 0xdc00 || low > 0xdfff) break;
                codepoint = 0x10000 + ((codepoint - 0xd800) << 10) + low - 0xdc00;
            } else if (codepoint >= 0xdc00 && codepoint <= 0xdfff) break;
            if (codepoint < 0x80) output[count++] = (unsigned char)codepoint;
            else if (codepoint < 0x800) {
                output[count++] = (unsigned char)(0xc0 | codepoint >> 6);
                output[count++] = (unsigned char)(0x80 | (codepoint & 63));
            } else if (codepoint < 0x10000) {
                output[count++] = (unsigned char)(0xe0 | codepoint >> 12);
                output[count++] = (unsigned char)(0x80 | ((codepoint >> 6) & 63));
                output[count++] = (unsigned char)(0x80 | (codepoint & 63));
            } else {
                output[count++] = (unsigned char)(0xf0 | codepoint >> 18);
                output[count++] = (unsigned char)(0x80 | ((codepoint >> 12) & 63));
                output[count++] = (unsigned char)(0x80 | ((codepoint >> 6) & 63));
                output[count++] = (unsigned char)(0x80 | (codepoint & 63));
            }
        } else break;
    }
    free(output);
    return 0;
}

static int json_skip(JsonReader *reader)
{
    json_space(reader);
    if (reader->cursor == reader->end || reader->depth >= 32) return 0;
    if (*reader->cursor == '"') {
        unsigned char *text = NULL; size_t length;
        int valid = json_string(reader, &text, &length);
        free(text); return valid;
    }
    if (*reader->cursor == '{' || *reader->cursor == '[') {
        int object = *reader->cursor++ == '{';
        unsigned char close = object ? '}' : ']';
        reader->depth++;
        if (json_take(reader, close)) { reader->depth--; return 1; }
        do {
            if (object) {
                unsigned char *name = NULL; size_t length;
                if (!json_string(reader, &name, &length)) return 0;
                free(name);
                if (!json_take(reader, ':')) return 0;
            }
            if (!json_skip(reader)) return 0;
            if (json_take(reader, close)) { reader->depth--; return 1; }
        } while (json_take(reader, ','));
        return 0;
    }
    const char *literals[] = {"true", "false", "null"};
    for (unsigned index = 0; index < 3; index++) {
        size_t length = strlen(literals[index]);
        if ((size_t)(reader->end - reader->cursor) >= length && !memcmp(reader->cursor, literals[index], length)) {
            reader->cursor += length; return 1;
        }
    }
    if (*reader->cursor == '-') reader->cursor++;
    if (reader->cursor == reader->end) return 0;
    if (*reader->cursor == '0') reader->cursor++;
    else {
        if (*reader->cursor < '1' || *reader->cursor > '9') return 0;
        do { reader->cursor++; } while (reader->cursor < reader->end && *reader->cursor >= '0' && *reader->cursor <= '9');
    }
    if (reader->cursor < reader->end && *reader->cursor == '.') {
        const unsigned char *start = ++reader->cursor;
        while (reader->cursor < reader->end && *reader->cursor >= '0' && *reader->cursor <= '9') reader->cursor++;
        if (start == reader->cursor) return 0;
    }
    if (reader->cursor < reader->end && (*reader->cursor == 'e' || *reader->cursor == 'E')) {
        reader->cursor++;
        if (reader->cursor < reader->end && (*reader->cursor == '-' || *reader->cursor == '+')) reader->cursor++;
        const unsigned char *start = reader->cursor;
        while (reader->cursor < reader->end && *reader->cursor >= '0' && *reader->cursor <= '9') reader->cursor++;
        if (start == reader->cursor) return 0;
    }
    return 1;
}

static int key_is(const unsigned char *name, size_t length, const char *expected)
{
    return length == strlen(expected) && !memcmp(name, expected, length);
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

int client_output(ClientOutput *output,const Tokenizer *tokenizer,
    const float vector[CLIENT_WIDTH],uint32_t *token,TokenText *text)
{
    if(!tokenizer||!token||!text)return 0;
    uint32_t selected;
    if(!client_output_project(output,vector,NULL,&selected))return 0;
    TokenText result;
    if(id_to_word(tokenizer,selected,&result)!=TOKENIZER_OK)return 0;
    *token=selected;*text=result;return 1;
}

#ifndef CLIENT_NO_MAIN
static int client_receive_vector(float vector[CLIENT_WIDTH])
{
    for(unsigned coordinate=0;coordinate<CLIENT_WIDTH;coordinate++)
        if(scanf("%f",vector+coordinate)!=1||!isfinite(vector[coordinate]))return 0;
    int character;
    while((character=getchar())!=EOF)if(character!=' '&&character!='\n'&&character!='\r'&&character!='\t')return 0;
    return !ferror(stdin);
}

static int client_output_command(const Tokenizer *tokenizer,const char *path)
{
    float vector[CLIENT_WIDTH];
    if(!client_receive_vector(vector)){fputs("client: expected one finite normalized vector\n",stderr);return 2;}
    ClientOutput *output=client_output_open(path);
    if(!output){fputs("client: invalid output-head dataset\n",stderr);return 1;}
    uint32_t token;TokenText text;
    int valid=client_output(output,tokenizer,vector,&token,&text);
    if(valid)valid=fwrite(text.bytes,1,text.length,stdout)==text.length;
    client_output_close(output);
    if(!valid){fputs("client: output-head evaluation failed\n",stderr);return 1;}
    fprintf(stderr,"output token: %u\n",token);
    return 0;
}

static int client_input_command(const char *path,int count,char **ids)
{
    for(int index=0;index<count;index++){
        uint32_t token;
        if(!decimal_id((const unsigned char *)ids[index],strlen(ids[index]),&token))return 2;
    }
    ClientInput *input=client_input_open(path);if(!input)return 1;
    int result=0;
    for(int index=0;index<count;index++){
        uint32_t token=0;float vector[CLIENT_WIDTH];
        decimal_id((const unsigned char *)ids[index],strlen(ids[index]),&token);
        if(!client_input(input,token,vector)){result=1;break;}
        for(unsigned coordinate=0;coordinate<CLIENT_WIDTH;coordinate++)
            printf("%a%c",(double)vector[coordinate],coordinate+1==CLIENT_WIDTH?'\n':' ');
    }
    client_input_close(input);return result;
}

int main(int argc, char **argv)
{
    if (argc < 6 || (strcmp(argv[4], "word-to-id") && strcmp(argv[4], "id-to-word") &&
        strcmp(argv[4], "input-ids") && strcmp(argv[4], "output-vector")) ||
        (!strcmp(argv[4],"input-ids") && argc < 7) || (!strcmp(argv[4],"output-vector") && argc != 6)) {
        fputs("usage: client MODEL VOCABULARY CONFIG word-to-id TEXT | id-to-word ID... | input-ids INPUT_TABLE ID... | output-vector OUTPUT_TABLE\n", stderr);
        return 2;
    }
    Tokenizer *tokenizer = NULL;
    TokenizerStatus status = tokenizer_open(argv[1], argv[2], argv[3], &tokenizer);
    if (status != TOKENIZER_OK) { fprintf(stderr, "tokenizer load failed (status %d)\n", status); return 1; }
    int result = 0;
    if (!strcmp(argv[4],"output-vector")) result=client_output_command(tokenizer,argv[5]);
    else if (!strcmp(argv[4],"input-ids")) result=client_input_command(argv[5],argc-6,argv+6);
    else if (!strcmp(argv[4], "word-to-id")) {
        uint32_t id;
        if (argc != 6) result = 2;
        else if (word_to_id(tokenizer, argv[5], strlen(argv[5]), &id) != TOKENIZER_OK) {
            fputs("NOT_FOUND: no exact vocabulary entry; this client does not split text\n", stderr);
            result = 3;
        } else printf("%u\n", (unsigned)id);
    } else {
        for (int index = 5; index < argc; index++) {
            uint32_t id; TokenText text;
            if (!decimal_id((const unsigned char *)argv[index], strlen(argv[index]), &id) ||
                id_to_word(tokenizer, id, &text) != TOKENIZER_OK) {
                fputs("NOT_FOUND: token ID has no defined text\n", stderr); result = 3; break;
            }
        }
        if (!result) {
            for (int index = 5; index < argc; index++) {
                uint32_t id = 0; TokenText text = {0};
                decimal_id((const unsigned char *)argv[index], strlen(argv[index]), &id);
                id_to_word(tokenizer, id, &text);
                if (fwrite(text.bytes, 1, text.length, stdout) != text.length) { result = 1; break; }
            }
        }
    }
    if (fflush(stdout) || ferror(stdout)) result = 1;
    tokenizer_close(tokenizer);
    return result;
}
#endif