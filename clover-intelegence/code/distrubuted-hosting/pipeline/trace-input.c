#include <assert.h>
#include "k3_tok.h"

enum { TRACE_CAPACITY = 256 };

static void trace_token(Tok *tokenizer, int token)
{
    char text[TRACE_CAPACITY];
    int length = tok_decode(tokenizer, &token, 1, text, sizeof text);
    assert(length > 0 && length < (int)sizeof text);
    fprintf(stderr, " %d=\"%.*s\"", token, length, text);
}

static int trace_chunks(Tok *tokenizer, const char *text, int length,
                        const int *encoded, int count)
{
    uint32_t codepoints[TRACE_CAPACITY];
    for (int index = 0; index < length; index++) {
        unsigned char byte = (unsigned char)text[index];
        assert(byte == ' ' || (byte >= 'a' && byte <= 'z') ||
               (byte >= 'A' && byte <= 'Z'));
        codepoints[index] = byte;
    }
    int chunks = 0, position = 0;
    for (int start = 0; start < length;) {
        int end = km_letters(codepoints, length, start);
        assert(end > start && end <= length);
        int tokens[TRACE_CAPACITY], emitted = 0;
        bpe_piece(tokenizer, (const unsigned char *)text, start, end,
                  tokens, &emitted, TRACE_CAPACITY);
        assert(emitted > 0 && position + emitted <= count);
        assert(memcmp(tokens, encoded + position, (size_t)emitted * sizeof *tokens) == 0);
        fprintf(stderr, "CHUNK %d \"%.*s\" ->", chunks, end - start, text + start);
        for (int index = 0; index < emitted; index++) trace_token(tokenizer, tokens[index]);
        if (emitted > 1) {
            char key[TRACE_CAPACITY * 2 + 1];
            int key_length = k3_bytelevel(tokenizer, (const unsigned char *)text + start,
                                          end - start, key);
            fprintf(stderr, " (whole-chunk vocabulary ID: %d)",
                    hm_get(&tokenizer->vocab, key, key_length));
        }
        fputc('\n', stderr);
        position += emitted;
        chunks++;
        start = end;
    }
    assert(position == count);
    return chunks;
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: trace-input TOKENIZER_DIRECTORY ASCII_LETTER_SPACE_TEXT\n");
        return 2;
    }
    size_t length = strlen(argv[2]);
    assert(length > 0 && length < TRACE_CAPACITY);
    Tok tokenizer;
    k3_tok_load(&tokenizer, argv[1]);
    int encoded[TRACE_CAPACITY];
    int count = tok_encode(&tokenizer, argv[2], (int)length, encoded, TRACE_CAPACITY);
    assert(count > 0 && count < TRACE_CAPACITY);
    char decoded[TRACE_CAPACITY];
    int decoded_length = tok_decode(&tokenizer, encoded, count, decoded, sizeof decoded);
    assert(decoded_length == (int)length && memcmp(decoded, argv[2], length) == 0);
    fprintf(stderr, "EXACT_INPUT \"%s\" (%zu bytes)\n", argv[2], length);
    int chunks = trace_chunks(&tokenizer, argv[2], (int)length, encoded, count);
    fprintf(stderr, "RESULT %d pre-tokenizer chunks -> %d final token IDs; exact text roundtrip\n",
            chunks, count);
    for (int index = 0; index < count; index++) printf("%s%d", index ? " " : "", encoded[index]);
    putchar('\n');
    return 0;
}