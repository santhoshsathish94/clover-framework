#ifndef NORMALIZATION_JSON_H
#define NORMALIZATION_JSON_H
typedef struct { const unsigned char *cursor, *end; unsigned depth; } JsonReader;

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

#endif
