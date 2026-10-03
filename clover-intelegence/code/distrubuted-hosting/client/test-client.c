#define CLIENT_NO_MAIN
#include "client.c"
#include <assert.h>

static void check_config(const char *json, int expected)
{
    Tokenizer *tokenizer = calloc(1, sizeof(*tokenizer));
    assert(tokenizer);
    tokenizer->tokens = calloc(TOKENIZER_IDS, sizeof(*tokenizer->tokens));
    assert(tokenizer->tokens);
    assert(load_config(tokenizer, (const unsigned char *)json, strlen(json)) == expected);
    tokenizer_close(tokenizer);
}

static void parser_controls(void)
{
    uint32_t id = 0;
    assert(decimal_id((const unsigned char *)"0", 1, &id) && id == 0);
    assert(decimal_id((const unsigned char *)"163839", 6, &id) && id == 163839);
    assert(!decimal_id((const unsigned char *)"163840", 6, &id));
    assert(!decimal_id((const unsigned char *)"99999999999999999999999999", 26, &id));
    assert(!decimal_id((const unsigned char *)"-1", 2, &id));
    assert(!decimal_id((const unsigned char *)"", 0, &id));
    unsigned char output[32]; size_t length = 0;
    assert(decode_base64((const unsigned char *)"AA==", 4, output, sizeof(output), &length));
    assert(length == 1 && output[0] == 0);
    assert(!decode_base64((const unsigned char *)"AB==", 4, output, sizeof(output), &length));
    assert(!decode_base64((const unsigned char *)"AAA", 3, output, sizeof(output), &length));
    assert(!decode_base64((const unsigned char *)"=AAA", 4, output, sizeof(output), &length));
    assert(!decode_base64((const unsigned char *)"AA==AA==", 8, output, sizeof(output), &length));
    assert(!decode_base64((const unsigned char *)"AA=A", 4, output, sizeof(output), &length));
    check_config("{}", 0);
    check_config("{\"added_tokens_decoder\":[]}", 0);
    check_config("{\"added_tokens_decoder\":{}} trailing", 0);
    check_config("{\"added_tokens_decoder\":{},\"added_tokens_decoder\":{}}", 0);
    check_config("{\"added_tokens_decoder\":{\"0\":{\"content\":\"A\"}}}", 0);
    check_config("{\"added_tokens_decoder\":{\"163840\":{\"content\":\"A\"}}}", 0);
    check_config("{\"added_tokens_decoder\":{\"163584\":{}}}", 0);
    check_config("{\"added_tokens_decoder\":{\"163584\":{\"content\":\"A\",\"content\":\"B\"}}}", 0);
    check_config("{\"added_tokens_decoder\":{\"163584\":{\"content\":\"\\uD800\"}}}", 0);
    check_config("{\"added_tokens_decoder\":{\"163584\":{\"content\":\"\\uDC00\"}}}", 0);
    check_config("{\"added_tokens_decoder\":{\"163584\":{\"content\":\"A\"},}}", 0);
    check_config("{\"ignored\": [true,false,null,{\"n\":-12.5e+3}],\"added_tokens_decoder\":{}}", 1);
    check_config("{\"ignored\": 01,\"added_tokens_decoder\":{}}", 0);
    check_config("{\"ignored\": tru,\"added_tokens_decoder\":{}}", 0);
    check_config("{\"ignored\": 1e,\"added_tokens_decoder\":{}}", 0);
    const char *escaped = "\"A\\n\\u0000\\u00e9\\uD83D\\uDE00\"";
    JsonReader reader = {(const unsigned char *)escaped, (const unsigned char *)escaped + strlen(escaped), 0};
    unsigned char *decoded = NULL;
    const unsigned char expected[] = {'A', '\n', 0, 0xc3, 0xa9, 0xf0, 0x9f, 0x98, 0x80};
    assert(json_string(&reader, &decoded, &length));
    assert(length == sizeof(expected) && !memcmp(decoded, expected, length) && reader.cursor == reader.end);
    free(decoded);
}

int main(int argc, char **argv)
{
    if (argc != 4) return 2;
    parser_controls();
    Tokenizer *first = NULL, *second = NULL;
    assert(tokenizer_open(argv[1], argv[2], argv[3], &first) == TOKENIZER_OK);
    assert(tokenizer_open(argv[1], argv[2], argv[3], &second) == TOKENIZER_OK);
    assert(first != second && first->tokens != second->tokens && first->lookup != second->lookup);
    assert(first->vocabulary != second->vocabulary);
    size_t count = 0, missing = 0, binary_count = 0;
    for (uint32_t id = 0; id < TOKENIZER_IDS; id++) {
        TokenText text = {0}; uint32_t back = UINT32_MAX;
        TokenizerStatus status = id_to_word(first, id, &text);
        if (status == TOKENIZER_NOT_FOUND) { assert(id >= TOKENIZER_BASE); missing++; continue; }
        assert(status == TOKENIZER_OK && text.bytes && text.length);
        assert(word_to_id(first, text.bytes, text.length, &back) == TOKENIZER_OK && back == id);
        if (memchr(text.bytes, 0, text.length)) binary_count++;
        count++;
    }
    assert(count == first->count && count == TOKENIZER_BASE + 16 && missing == 240 && binary_count > 0);
    uint32_t id = 0;
    assert(word_to_id(first, " Paris", 6, &id) == TOKENIZER_OK && id == 17374);
    assert(word_to_id(first, " Tokyo", 6, &id) == TOKENIZER_OK && id == 40484);
    assert(word_to_id(first, "[EOS]", 5, &id) == TOKENIZER_OK && id == 163585);
    assert(word_to_id(first, "<|open|>", 8, &id) == TOKENIZER_OK && id == 163587);
    assert(word_to_id(first, "the captial of frace is", 22, &id) == TOKENIZER_NOT_FOUND);
    assert(word_to_id(first, "", 0, &id) == TOKENIZER_NOT_FOUND);
    TokenText text = {0};
    assert(id_to_word(first, TOKENIZER_IDS, &text) == TOKENIZER_NOT_FOUND);
    assert(id_to_word(first, 163592, &text) == TOKENIZER_NOT_FOUND);
    assert(word_to_id(NULL, " Paris", 6, &id) == TOKENIZER_INVALID_ARGUMENT);
    assert(word_to_id(first, NULL, 1, &id) == TOKENIZER_INVALID_ARGUMENT);
    assert(id_to_word(first, 0, NULL) == TOKENIZER_INVALID_ARGUMENT);
    tokenizer_close(first);
    assert(word_to_id(second, " Paris", 6, &id) == TOKENIZER_OK && id == 17374);
    assert(id_to_word(second, 40484, &text) == TOKENIZER_OK && text.length == 6 && !memcmp(text.bytes, " Tokyo", 6));
    tokenizer_close(second);
    Tokenizer *invalid = NULL;
    assert(tokenizer_open(argv[2], argv[1], argv[3], &invalid) == TOKENIZER_INVALID_DATA && !invalid);
    assert(tokenizer_open(NULL, argv[2], argv[3], &invalid) == TOKENIZER_INVALID_ARGUMENT);
    printf("PASS: %zu defined IDs round-trip, %zu undefined IDs reject, binary bytes preserved, independent contexts and parser controls\n", count, missing);
    return 0;
}