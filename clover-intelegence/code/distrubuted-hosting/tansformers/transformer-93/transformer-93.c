#define _GNU_SOURCE
/* Layer 93. The last stop: final aggregation and normalisation over the snapshots,
   the head that turns a vector into a token, and the post back to whoever asked.

   It exists so the server owns one end of the chain instead of both. The server takes
   a token in; layer 93 hands an answer out. Nothing routes a reply back through the
   middle, because layer 93 already holds the address it was given.

   Stages 11161..11280 under common/stage.h, which is owner 93. The arithmetic that
   decides the owner already reserved this range, so nothing about routing changes. */

#include <float.h>
#include <fenv.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NORMALIZATION_NO_MAIN
#include "../../normalization/normalization.c"
#include "../../server/ends.h"
#include "stage.h"
#include "deliver.h"
#include "trace.h"

enum {
    TRANSFORMER_93_WIDTH = NORMALIZATION_WIDTH,
    TRANSFORMER_93_TEXT  = 8192,
    TRANSFORMER_93_BODY  = 32768,
    TRANSFORMER_93_URL   = 1024,
    TRANSFORMER_93_STAGE_BASE = CLOVER_STAGE_TAIL_BASE
};

typedef struct {
    Normalization *normalization;
    Tokenizer *tokenizer;
    char head_path[4096];
} Transformer93;

typedef struct {
    const Transformer93 *owner;
    /* The head decodes each block into one scratch buffer inside itself, so it is a
       per-request reader. The gains and the vocabulary are read-only and shared. */
    ServerOutput *head;
    int stage;
    int failed;
    int final;
    unsigned request;
    size_t positions;

    const float *sources[NORMALIZATION_SOURCES];
    float scores[NORMALIZATION_SOURCES];
    float weights[NORMALIZATION_SOURCES];
    float result[TRANSFORMER_93_WIDTH];

    uint32_t token;
    TokenText word;

    char callback[TRANSFORMER_93_URL];
    char text[TRANSFORMER_93_TEXT];
    size_t text_length;
    char body[TRANSFORMER_93_BODY];
    size_t body_length;
    int delivered;
    double normalize_seconds, head_seconds, deliver_seconds, mark;
} Transformer93Sequence;

static double transformer_93_clock(void)
{
    struct timespec moment;
    clock_gettime(CLOCK_MONOTONIC, &moment);
    return (double)moment.tv_sec + (double)moment.tv_nsec / 1e9;
}

void transformer_93_close(Transformer93 *model)
{
    if (!model) return;
    if (model->normalization) normalization_close(model->normalization);
    if (model->tokenizer) tokenizer_close(model->tokenizer);
    free(model);
}

int transformer_93_open(const char *leaves_path, const char *head_path,
    const char *model_path, const char *vocabulary_path, const char *config_path,
    Transformer93 **result)
{
    Transformer93 *model;
    ServerOutput *probe;
    if (!result || !head_path || strlen(head_path) >= sizeof model->head_path) return 0;
    *result = NULL;
    model = calloc(1, sizeof *model);
    if (!model) return 0;
    memcpy(model->head_path, head_path, strlen(head_path) + 1);
    if (!normalization_open(leaves_path, &model->normalization) ||
        tokenizer_open(model_path, vocabulary_path, config_path, &model->tokenizer) != TOKENIZER_OK ||
        !(probe = server_output_open(model->head_path))) {
        transformer_93_close(model);
        return 0;
    }
    server_output_close(probe);
    *result = model;
    return 1;
}

void transformer_93_sequence_close(Transformer93Sequence *sequence)
{
    if (!sequence) return;
    if (sequence->head) server_output_close(sequence->head);
    free(sequence);
}

Transformer93Sequence *transformer_93_sequence_create(const Transformer93 *model)
{
    Transformer93Sequence *sequence;
    if (!model) return NULL;
    sequence = calloc(1, sizeof *sequence);
    if (!sequence) return NULL;
    sequence->owner = model;
    sequence->head = server_output_open(model->head_path);
    if (!sequence->head) { free(sequence); return NULL; }
    return sequence;
}

void transformer_93_sequence_reset(Transformer93Sequence *sequence)
{
    const Transformer93 *owner;
    ServerOutput *head;
    if (!sequence) return;
    owner = sequence->owner;
    head = sequence->head;
    memset(sequence, 0, sizeof *sequence);
    sequence->owner = owner;
    sequence->head = head;
}

int transformer_93_stage(const Transformer93Sequence *sequence)
{
    return sequence ? TRANSFORMER_93_STAGE_BASE + sequence->stage : 0;
}

int transformer_93_text(const Transformer93Sequence *sequence, const char **text, size_t *length)
{
    if (!sequence || !text || !length) return 0;
    *text = sequence->text;
    *length = sequence->text_length;
    return 1;
}

int transformer_93_delivered(const Transformer93Sequence *sequence)
{
    return sequence && sequence->delivered;
}

int transformer_93_timing(const Transformer93Sequence *sequence,
    double *normalize, double *head, double *deliver)
{
    if (!sequence || !normalize || !head || !deliver) return 0;
    *normalize = sequence->normalize_seconds;
    *head = sequence->head_seconds;
    *deliver = sequence->deliver_seconds;
    return 1;
}

/* One condition per stage, one unit of work, one return. Stages 1..8 are the final
   normalisation split at its own seams, in the order normalization_process runs them,
   so the arithmetic is unchanged. 110..112 are the head. 118..120 only run on the last
   token, which is exactly where clover_stage_next stops looping. */
static int transformer_93_step(const Transformer93 *model, Transformer93Sequence *sequence,
    const float *residual, const float *snapshots, unsigned snapshot_count)
{
    const int stage = sequence->stage;

    if (stage == 1) {
        if (!normalization_environment() ||
            !normalization_finite(residual, NORMALIZATION_WIDTH) ||
            !normalization_finite(snapshots, (size_t)snapshot_count * NORMALIZATION_WIDTH)) return 0;
        sequence->stage++; return 1;
    }
    if (stage == 2) {
        for (unsigned source = 0; source < NORMALIZATION_SNAPSHOTS; source++)
            sequence->sources[source] = snapshots + source * NORMALIZATION_WIDTH;
        sequence->stage++; return 1;
    }
    if (stage == 3) { sequence->sources[NORMALIZATION_SNAPSHOTS] = residual; sequence->stage++; return 1; }
    if (stage == 4) {
        if (!normalization_score_sources(model->normalization, sequence->sources, sequence->scores)) return 0;
        sequence->stage++; return 1;
    }
    if (stage == 5) {
        if (!normalization_softmax(sequence->scores, sequence->weights)) return 0;
        sequence->stage++; return 1;
    }
    if (stage == 6) {
        normalization_aggregate(sequence->sources, sequence->weights, sequence->result);
        sequence->stage++; return 1;
    }
    if (stage == 7) {
        if (!normalization_finite(sequence->result, NORMALIZATION_WIDTH)) return 0;
        sequence->stage++; return 1;
    }
    if (stage == 8) { normalization_apply_rms(model->normalization, sequence->result); sequence->stage++; return 1; }
    if (stage == 9) {
        if (!normalization_finite(sequence->result, NORMALIZATION_WIDTH)) return 0;
        sequence->normalize_seconds += transformer_93_clock() - sequence->mark;
        sequence->mark = transformer_93_clock();
        sequence->stage = 110; return 1;
    }
    if (stage == 110) {
        if (!server_output_project(sequence->head, sequence->result, NULL, &sequence->token)) return 0;
        sequence->stage++; return 1;
    }
    if (stage == 111) {
        if (id_to_word(model->tokenizer, sequence->token, &sequence->word) != TOKENIZER_OK) return 0;
        sequence->stage++; return 1;
    }
    if (stage == 112) {
        if (sequence->text_length + sequence->word.length < sizeof sequence->text) {
            memcpy(sequence->text + sequence->text_length, sequence->word.bytes, sequence->word.length);
            sequence->text_length += sequence->word.length;
        }
        sequence->stage++; return 1;
    }
    if (stage == 113) {
        sequence->head_seconds += transformer_93_clock() - sequence->mark;
        sequence->positions++;
        sequence->stage = sequence->final ? 118 : 120;
        return 1;
    }
    if (stage == 118) {
        char escaped[TRANSFORMER_93_TEXT * 6];
        size_t width = clover_deliver_escape(sequence->text, sequence->text_length, escaped, sizeof escaped);
        int written = snprintf(sequence->body, sizeof sequence->body,
            "{\"request\":%u,\"tokens\":%zu,\"text\":\"%.*s\"}",
            sequence->request, sequence->positions, (int)width, escaped);
        if (written <= 0 || (size_t)written >= sizeof sequence->body) return 0;
        sequence->body_length = (size_t)written;
        sequence->stage++; return 1;
    }
    if (stage == 119) {
        /* A callback that cannot be reached is reported, not fatal: the answer exists. */
        double mark = transformer_93_clock();
        sequence->delivered = sequence->callback[0]
            ? clover_deliver(sequence->callback, sequence->body, sequence->body_length) : 0;
        sequence->deliver_seconds += transformer_93_clock() - mark;
        sequence->stage++; return 1;
    }
    if (stage == 120) { sequence->stage = 0; return 1; }

    sequence->stage++;
    return 1;
}

int transformer_93_process(const Transformer93 *model, Transformer93Sequence *sequence,
    const float *residual, const float *snapshots, unsigned snapshot_count,
    int final, const char *callback, uint32_t *token, TokenText *word)
{
    if (!model || !sequence || sequence->owner != model || !residual || !snapshots ||
        snapshot_count != NORMALIZATION_SNAPSHOTS || sequence->failed || !token || !word) return 0;
    if (callback && *callback) {
        size_t length = strlen(callback);
        if (length >= sizeof sequence->callback) return 0;
        memcpy(sequence->callback, callback, length + 1);
    }
    sequence->final = final;
    sequence->mark = transformer_93_clock();
    sequence->stage = 1;
    clover_trace_open();
    while (sequence->stage) {
        const int executed = sequence->stage;
        const double began = clover_trace_active() ? clover_trace_clock() : 0.0;
        if (!transformer_93_step(model, sequence, residual, snapshots, snapshot_count)) {
            sequence->failed = 1;
            return 0;
        }
        if (clover_trace_active())
            clover_trace_stage(93, executed, (unsigned long)sequence->positions,
                clover_trace_clock() - began,
                clover_trace_hash(sequence->result, sizeof sequence->result),
                executed == 110 ? (long)sequence->token : -1);
    }
    *token = sequence->token;
    *word = sequence->word;
    return 1;
}

void transformer_93_set_request(Transformer93Sequence *sequence, unsigned request)
{
    if (sequence) sequence->request = request;
}
