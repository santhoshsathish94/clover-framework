/* Token ids on stdin, text on stdout. The standalone prints ids and the direction
   is that the output is what decides, so the ids have to be readable. */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../server/ends.h"

int main(int argc, char **argv)
{
    if (argc != 2) { fputs("usage: detokenise CODE_DIRECTORY\n", stderr); return 2; }
    char model[4096], vocabulary[4096], config[4096];
    if (snprintf(model, sizeof model, "%s/server/bin/dataset/tiktoken.model", argv[1]) >= (int)sizeof model ||
        snprintf(vocabulary, sizeof vocabulary, "%s/server/bin/dataset/vocabulary.bin", argv[1]) >= (int)sizeof vocabulary ||
        snprintf(config, sizeof config, "%s/server/bin/configs/tokenizer_config.json", argv[1]) >= (int)sizeof config) {
        fputs("path too long\n", stderr); return 2;
    }
    Tokenizer *tokenizer = NULL;
    if (tokenizer_open(model, vocabulary, config, &tokenizer) != TOKENIZER_OK) {
        fputs("tokenizer could not be opened\n", stderr); return 1;
    }
    unsigned long id;
    while (scanf("%lu", &id) == 1) {
        TokenText text;
        if (id_to_word(tokenizer, (uint32_t)id, &text) != TOKENIZER_OK) { printf("<%lu?>", id); continue; }
        fwrite(text.bytes, 1, text.length, stdout);
    }
    putchar('\n');
    tokenizer_close(tokenizer);
    return 0;
}
