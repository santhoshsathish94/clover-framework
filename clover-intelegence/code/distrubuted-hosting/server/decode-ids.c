/* decode-ids.c - token id to text, through the engine's own tokenizer.
 *
 * The head reports a number. Reading "the residual picks 276 and the final vector picks
 * 28202" tells you nothing; reading "the residual picks ' the' and the final vector
 * picks ' Berlin'" tells you what the last step of the model is for. tiktoken is not
 * installed on this host and guessing at ids from a prompt is how a wrong mapping gets
 * written down as evidence, so this uses id_to_word, which is what the tail calls.
 *
 * build: gcc -std=c11 -O2 decode-ids.c -lm -o bin/decode-ids
 * run:   ./bin/decode-ids <server-bin-dir> <id> [id ...]
 *
 * The three paths are the ones pipeline.c passes to layer_93_open. The config does not
 * sit beside the model and vocabulary: it is server/bin/configs, not server/bin/dataset.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ends.h"

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: %s <server-bin-dir> <id> [id ...]\n", argv[0]); return 2; }
    char model[4096], vocabulary[4096], config[4096];
    snprintf(model, sizeof model, "%s/dataset/tiktoken.model", argv[1]);
    snprintf(vocabulary, sizeof vocabulary, "%s/dataset/vocabulary.bin", argv[1]);
    snprintf(config, sizeof config, "%s/configs/tokenizer_config.json", argv[1]);

    Tokenizer *tokenizer = NULL;
    if (tokenizer_open(model, vocabulary, config, &tokenizer) != TOKENIZER_OK) {
        fputs("tokenizer open failed\n", stderr); return 1;
    }
    for (int index = 2; index < argc; index++) {
        uint32_t id = (uint32_t)strtoul(argv[index], NULL, 10);
        TokenText text;
        if (id_to_word(tokenizer, id, &text) != TOKENIZER_OK) { printf("%u\t<not found>\n", id); continue; }
        printf("%u\t\"", id);
        for (size_t byte = 0; byte < text.length; byte++) {
            unsigned char value = text.bytes[byte];
            if (value == '\n') fputs("\\n", stdout);
            else if (value == '\t') fputs("\\t", stdout);
            else if (value == '"') fputs("\\\"", stdout);
            else if (value < 0x20) printf("\\x%02x", value);
            else putchar(value);
        }
        printf("\"\n");
    }
    tokenizer_close(tokenizer);
    return 0;
}
