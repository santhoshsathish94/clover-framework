/* Checks that what the engine now reads from the original open weights is what it read before from the converted dataset files.
     check-open-weights OPEN_WEIGHTS_DIR DATASET_DIR
   DATASET_DIR must still hold the old files (inputs/seed.bin, outputs/fruit.bin, leaves.json, vocabulary.bin); they are only read here.
   build: gcc -std=gnu11 -O2 -I src/code/common src/tools/check-open-weights.c -lm -o check-open-weights
   Exit status 0 only if every comparison is identical.                                                                               */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define NORMALIZATION_NO_MAIN
#include "../code/common/final-normalization.c"
#include "../code/common/token-ends.h"
#include "../code/common/embedding-and-head-tables.h"

#define SHARD "model-00094-of-000096.safetensors"
static int failures;
static void report(int same, const char *what, const char *detail)
{
    printf("%s  %s%s%s\n", same ? "SAME" : "DIFFERENT", what, *detail ? ": " : "", detail);
    if (!same) failures++;
}

int main(int argc, char **argv)
{
    if (argc != 3) { fputs("usage: check-open-weights OPEN_WEIGHTS_DIR DATASET_DIR\n", stderr); return 2; }
    char a[4096], b[4096], detail[256];

    /* 1. the tokenizer: opened with the vocabulary file, and with the table built from tiktoken.model. Every token's bytes and the lookup table must be equal.
          (vocabulary.bin also holds 3,584 more bytes, for ids from 163584; the tokenizer never reads them: tokens from that id on come from tokenizer_config.json,
          and every token's bytes are compared here.) */
    char config[4096];
    snprintf(a, sizeof a, "%s/tiktoken.model", argv[1]);
    snprintf(b, sizeof b, "%s/vocabulary.bin", argv[2]);
    snprintf(config, sizeof config, "%s/tokenizer_config.json", argv[1]);
    Tokenizer *rebuilt = NULL, *from_file = NULL;
    const int status_rebuilt = tokenizer_open(a, NULL, config, &rebuilt), status_file = tokenizer_open(a, b, config, &from_file);
    int same_tokens = status_rebuilt == TOKENIZER_OK && status_file == TOKENIZER_OK && rebuilt->count == from_file->count;
    unsigned differing = 0;
    for (uint32_t id = 0; same_tokens && id < TOKENIZER_IDS; id++) {
        const TokenText x = rebuilt->tokens[id], y = from_file->tokens[id];
        if (!x.bytes != !y.bytes || x.length != y.length || (x.bytes && memcmp(x.bytes, y.bytes, x.length))) differing++;
    }
    snprintf(detail, sizeof detail, "%zu tokens, %u differ, status %d and %d", rebuilt ? rebuilt->count : 0, differing, status_rebuilt, status_file);
    report(same_tokens && !differing && !memcmp(rebuilt->lookup, from_file->lookup, TOKENIZER_HASH * sizeof *rebuilt->lookup),
        "tokenizer opened without vocabulary.bin (table built from tiktoken.model) against opened with it: every token and the lookup table", detail);
    tokenizer_close(rebuilt); tokenizer_close(from_file);
    unsigned char *model = NULL, *old = NULL, *built = NULL; size_t model_bytes = 0, old_bytes = 0, built_bytes = 0;
    if (read_file(a, 16 << 20, &model, &model_bytes) || read_file(b, 16 << 20, &old, &old_bytes)) { puts("cannot read tiktoken.model or vocabulary.bin"); return 2; }
    int status = build_vocabulary(model, model_bytes, &built, &built_bytes);
    const size_t prefix = (TOKENIZER_IDS + 1ULL) * 4, bytes_below_added = read_u32(old + TOKENIZER_BASE * 4);
    int same_below_added = status == TOKENIZER_OK && read_u32(built + TOKENIZER_BASE * 4) == bytes_below_added && !memcmp(built + prefix, old + prefix, bytes_below_added);
    for (uint32_t id = 0; same_below_added && id <= TOKENIZER_BASE; id++) if (read_u32(built + id * 4) != read_u32(old + id * 4)) same_below_added = 0;
    snprintf(detail, sizeof detail, "%zu bytes built, %zu in the file (the difference is bytes the file holds for ids from 163584)", built_bytes, old_bytes);
    report(same_below_added, "vocabulary table for ids below 163584, built against vocabulary.bin: offsets and bytes", detail);

    /* 2. the final normalisation tensors */
    snprintf(a, sizeof a, "%s/" SHARD, argv[1]);
    snprintf(b, sizeof b, "%s/leaves.json", argv[2]);
    Normalization *from_shard = NULL, *from_json = NULL;
    int opened = normalization_open(a, &from_shard) && normalization_open(b, &from_json);
    report(opened && !memcmp(from_shard->gains, from_json->gains, sizeof from_json->gains) && !memcmp(from_shard->fold, from_json->fold, sizeof from_json->fold),
        "three normalisation tensors (3 x 7168 bf16) and the folded gain, shard against leaves.json", opened ? "" : "could not open one of them");

    /* 3. the two tables */
    static const char *const tensors[2] = {"language_model.model.embed_tokens.weight", "language_model.lm_head.weight"};
    static const char *const old_files[2] = {"inputs/seed.bin", "outputs/fruit.bin"};
    static const char *const old_sources[2] = {"4a79cdabdab6826b994aff69d90a72aa35dae32e90f8acaf1ce312c7bdc4a487", "11c1f1c09a8e0db55547b5e68ebfd1d8e3b503bee56c4e1312ef55ecd3e5580f"};
    for (int t = 0; t < 2; t++) {
        snprintf(a, sizeof a, "%s/" SHARD "#%s", argv[1], tensors[t]);
        snprintf(b, sizeof b, "%s/%s", argv[2], old_files[t]);
        NumericTable *shard = table_open(a, ""), *file = table_open(b, old_sources[t]);
        int same = shard && file && shard->direct && file->direct && shard->direct_bytes == file->direct_bytes && !memcmp(shard->direct, file->direct, shard->direct_bytes);
        snprintf(detail, sizeof detail, "%s", shard && file ? "" : "could not open one of them");
        char what[160]; snprintf(what, sizeof what, "%s (%zu bytes), shard against %s.direct", tensors[t], shard ? shard->direct_bytes : 0, old_files[t]);
        report(same, what, detail);
        table_close(shard); table_close(file);
    }

    /* 4. a wrong name, a wrong shape and a wrong file are refused */
    snprintf(a, sizeof a, "%s/" SHARD "#language_model.model.norm.weight", argv[1]);
    NumericTable *wrong_shape = table_open(a, "");
    report(wrong_shape == NULL, "a tensor of the wrong shape is refused", "");
    table_close(wrong_shape);
    snprintf(a, sizeof a, "%s/" SHARD "#no.such.tensor", argv[1]);
    NumericTable *wrong_name = table_open(a, "");
    report(wrong_name == NULL, "a tensor that is not in the file is refused", "");
    table_close(wrong_name);
    snprintf(a, sizeof a, "%s/tiktoken.model#language_model.lm_head.weight", argv[1]);
    NumericTable *wrong_file = table_open(a, "");
    report(wrong_file == NULL, "a file that is not safetensors is refused", "");
    table_close(wrong_file);

    printf("%s\n", failures ? "FAIL" : "PASS  everything the engine now reads from the open weights is identical to what it read before");
    return failures != 0;
}
