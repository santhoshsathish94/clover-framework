#define _GNU_SOURCE
/* Words -> token ids -> embedding vectors -> layer 0 -> layers 1..92 -> final normalisation -> vocabulary head ->
   next token and its text, in one process, as ONE request over all the tokens of the prompt.

   Stages are separate source files, built as objects and called through their entry points (see ../README.md for the map):
     words-to-embeddings.c   stage1_*   words -> ids -> embeddings
     layer0.c                stage2_*   layer 0 (the server: KDA + dense MLP)
     layers-1-to-92.c        stage3_*   layer 1 (built against its own pod source)
                             lbk_*      layers 2..92 that use KDA attention (one object, any number of opened layers)
                             lbm_*      layers 3,7,..,91 and 92 that use MLA attention
     transformer-93  (the deployed pipeline-stage.so, loaded with dlopen)   final normalisation, head, detokenise,
                     for the LAST position only: that is the one whose next token the prompt asks for.
   No data is copied between layers: one residual buffer is read and the next written, and the snapshots are
   pointers to buffers. At every twelfth layer the layer's input buffer itself becomes the new snapshot.
   One 15.7 GB buffer holds the experts of whichever layer is running; it is allocated once and shared.

     whole-model BUILD_DIR DATASET_DIR OPEN_WEIGHTS_DIR WORDS REFDIR

   REFDIR holds recorded values for the ids WORDS stands for (embed.f32, final.resid.f32, final.snaps.f32: the pod's
   own chain, layer by layer, ending after layer 92) or is "-" when none exist; then nothing is compared.        */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <dlfcn.h>
#include <unistd.h>
#include <sys/mman.h>

/* The original open weights (moonshotai/Kimi-K3): the embedding table, the output head and the three final normalisation tensors are read from
   this shard by their own names, and tiktoken.model and tokenizer_config.json from the same folder. */
#define OPEN_WEIGHTS_SHARD "model-00094-of-000096.safetensors"

double stage1_setup(const char *tokfiles, const char *seed);
int stage1_ids(const char *text, long size, int *ids, int cap);
int stage1_ids_plain(const char *text, long size, int *ids, int cap);
int stage1_id_of(const char *name);
void stage1_lookup(const unsigned *ids, unsigned n, float *out);
double stage2_setup(const char *dataset, unsigned max_n);
void stage2_run(const uint32_t *ids, const float *emb, unsigned n, unsigned window, float *out);
void stage2_run_from(const uint32_t *ids, const float *emb, unsigned n, unsigned window, float *out);
void stage2_step(const uint32_t *id, const float *emb, float *out);
void stage2_stepk(const uint32_t *ids, const float *emb, unsigned n, float *out);
void stage3_stepk(const float *resid, const float *s0, unsigned n, float *out);
void stage3_set_scratch(unsigned char *const *slots, unsigned n);
void stage3_use_buffer(unsigned char *buffer);
double stage3_setup(const char *dataset, unsigned max_n);
void stage3_run(const float *resid, const float *s0, unsigned n, unsigned window, float *out);
void stage3_run_from(const float *resid, const float *s0, unsigned n, unsigned window, float *out);
void stage3_step(const float *resid, const float *s0, float *out);
void stage2_keep_positions(unsigned count); void stage2_use_position(unsigned position); void stage3_keep_positions(unsigned count); void stage3_use_position(unsigned position);
void stage3_route_log(int on);
void stage3_routes(unsigned short *out);
void stage3_hits(unsigned long *out);
void stage3_set_slots(unsigned char *const *slots, unsigned cap);
void stage3_cache_stats(unsigned long *hit, unsigned long *miss);
void stage3_guess_stats(unsigned long *issued, unsigned long *used);
void stage3_laps(double *out);
#define LAYER_KIND(P) \
    void P##_laps(double *out); \
    void P##_guess_stats(unsigned long *issued, unsigned long *used); \
    void P##_set_slots(void *handle, unsigned char *const *slots, unsigned cap); void P##_cache_stats(unsigned long *hit, unsigned long *miss); \
    void P##_route_log(int on); void P##_routes(unsigned short *out); void P##_hits(unsigned long *out); \
    void P##_set_buffer(unsigned char *buffer); double P##_init(unsigned max_n); \
    void *P##_open(const char *dataset, unsigned layer, unsigned k, unsigned max_n); \
    double P##_run(void *handle, const float *resid, const float *const *snaps, unsigned n, unsigned window, float *out); \
    double P##_run_from(void *handle, const float *resid, const float *const *snaps, unsigned n, unsigned window, size_t pos0, float *out); \
    double P##_step(void *handle, const float *resid, const float *const *snaps, size_t pos0, float *out); \
    double P##_stepk(void *handle, const float *resid, const float *const *snaps, unsigned n, size_t pos0, float *out); \
    void P##_set_scratch(unsigned char *const *slots, unsigned n); unsigned long P##_distinct(void); \
    void P##_keep_positions(void *handle, unsigned count); void P##_use_position(void *handle, unsigned position);
LAYER_KIND(lbk)
LAYER_KIND(lbm)

enum { W = 7168, LAYERS = 92, MAX_SNAPSHOTS = 8, POOL = MAX_SNAPSHOTS + 3 };
#define EXPERT_LAYER_BYTES 15722348544ULL
#define EXPERT_BYTES 17547264ULL

static unsigned long vmstat_major_faults(void)
{
    FILE *f = fopen("/proc/vmstat", "r"); char name[64]; unsigned long value, found = 0;
    while (f && fscanf(f, "%63s %lu", name, &value) == 2) if (!strcmp(name, "pgmajfault")) { found = value; break; }
    if (f) fclose(f);
    return found;
}

typedef struct { const unsigned char *bytes; size_t length; } Text;

static double clk(void)
{
    struct timespec m; clock_gettime(CLOCK_MONOTONIC, &m);
    return (double)m.tv_sec + (double)m.tv_nsec / 1e9;
}
static void bail(const char *what) { fprintf(stderr, "pipeline-full: %s\n", what); exit(1); }

static char *slurp(const char *path, long *size)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "pipeline-full: cannot read %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); *size = ftell(f); fseek(f, 0, SEEK_SET);
    char *data = malloc((size_t)*size + 1);
    if (!data || fread(data, 1, (size_t)*size, f) != (size_t)*size) bail("read failed");
    data[*size] = 0; fclose(f);
    return data;
}

static unsigned same_rows(const float *a, const float *b, unsigned n)
{
    unsigned ok = 0;
    for (unsigned i = 0; i < n; i++) ok += !memcmp(a + (size_t)i * W, b + (size_t)i * W, W * sizeof(float));
    return ok;
}

static int is_mla(unsigned layer) { return layer % 4 == 3 || layer == LAYERS; }
static unsigned snapshots_in(unsigned layer) { return 1 + (layer - 1) / 12; }

/* ---- chat mode (PF_CHAT): Kimi K3's XTML format, as in kimi-k3-in-c/src/chat/k3_chat.c ----
   PF_CHAT=1 (or nothink): the model answers at once; PF_CHAT=low|high|max: it thinks first at that effort; PF_SYSTEM adds a system
   message. The control tokens come from the tokenizer by name and the text between them is encoded as plain text only. */
static struct {
    int on, think, open, close, sep, eom, eos, closes, in_tag, to_response, done;
    const char *effort, *system;
    char think_text[1 << 16], response[1 << 16];
    size_t think_len, response_len;
} chat;

static int plain(int *ids, int cap, int n, const char *text)
{
    return n + stage1_ids_plain(text, (long)strlen(text), ids + n, cap - n);
}

static int chat_message(int *ids, int cap, int n, const char *attrs, const char *content, long length)
{
    ids[n++] = chat.open; n = plain(ids, cap, n, "message"); n = plain(ids, cap, n, attrs); ids[n++] = chat.sep;
    if (length > 0) n += stage1_ids_plain(content, length, ids + n, cap - n);
    ids[n++] = chat.close; n = plain(ids, cap, n, "message"); ids[n++] = chat.sep; ids[n++] = chat.eom;
    return n;
}

static int prompt_ids(const char *text, long size, int *ids, int cap)
{
    if (!chat.on) return stage1_ids(text, size, ids, cap);
    int n = 0;
    if (chat.think) {
        char effort[640];
        snprintf(effort, sizeof effort,
            "`thinking_effort` guides on how much to think in your thinking channel (not including the response channel), "
            "supported values include `low`, `medium`, `high`, and `max`.\nNow the system is invoked with `thinking_effort=%s`.", chat.effort);
        n = chat_message(ids, cap, n, " role=\"system\" type=\"thinking-effort\"", effort, (long)strlen(effort));
    }
    if (chat.system && *chat.system) n = chat_message(ids, cap, n, " role=\"system\"", chat.system, (long)strlen(chat.system));
    n = chat_message(ids, cap, n, " role=\"user\"", text, size);
    ids[n++] = chat.open; n = plain(ids, cap, n, "message"); n = plain(ids, cap, n, " role=\"assistant\""); ids[n++] = chat.sep;
    ids[n++] = chat.open; n = plain(ids, cap, n, chat.think ? "think" : "response"); ids[n++] = chat.sep;
    return n;
}

/* Follows what the model writes: the reply ends at the first closing tag (answer at once) or the second (think, then answer). */
static void chat_feed(uint32_t id, const Text *piece)
{
    if (chat.done) return;
    if ((int)id == chat.close) { chat.closes++; chat.in_tag = 1; if (chat.closes >= (chat.think ? 2 : 1)) chat.done = 1; return; }
    if ((int)id == chat.open) { chat.in_tag = 1; return; }
    if ((int)id == chat.sep) { chat.in_tag = 0; if (chat.think && chat.closes == 1) chat.to_response = 1; return; }
    if ((int)id == chat.eom || (int)id == chat.eos) { chat.done = 1; return; }
    if (chat.in_tag) return;
    char *dst = chat.to_response ? chat.response : chat.think_text;
    size_t *len = chat.to_response ? &chat.response_len : &chat.think_len;
    if (*len + piece->length < sizeof chat.response) { memcpy(dst + *len, piece->bytes, piece->length); *len += piece->length; }
}

/* ---- speculative decoding (SPECULATIVE_DRAFT_TOKENS=D) ----
   The text so far is searched for the latest earlier place where its last 3 (else 2) tokens occurred; the tokens that followed there are the
   draft. The last real token and the draft go through the layers as ONE pass; the head then names the model's own next token after each
   of them, and the draft is kept as far as it agrees. The result is the same tokens plain decoding gives, in fewer passes when the draft is
   right. A rejected draft leaves the recurrent state of the KDA layers ahead of where it should be, so every KDA layer keeps its state after each
   token of the pass, and a rejected draft is undone by copying back the state after the last token that was kept. */
enum { PASS_MAX = 16 };
typedef int (*TailRun)(const void *, void *, const float *, const float *, unsigned, int, const char *, uint32_t *, Text *);
typedef struct {
    void *layer[LAYERS + 1];
    const void *tail_model; void *tail_sequence; TailRun tail_run;
    const float *hidden, *snapshot[MAX_SNAPSHOTS];   /* the rows the last forward_tokens call produced */
} Model;
typedef struct { unsigned passes, with_draft, rollbacks, drafted, accepted; double draft_seconds; } SpeculativeStats;
static float pass_buffer[3][PASS_MAX * W], pass_snapshot[MAX_SNAPSHOTS][PASS_MAX * W], pass_embedding[PASS_MAX * W], tail_snapshots[MAX_SNAPSHOTS * W];

/* DSpark draft model (dspark-draft.c): it reads the target's hidden states after layers 2, 23, 47, 71 and 89, one row of 7168 floats per position. */
int dspark_open(const char *checkpoint, const char *head_table, unsigned max_positions);
void dspark_draft(const float *const aux[5], uint32_t anchor, unsigned anchor_position, uint32_t *drafts);
void dspark_reset(void);
static const unsigned dspark_layers[5] = { 2, 23, 47, 71, 89 };
static float *dspark_hidden[5];   /* NULL unless DSpark is used; allocated in main */
static int fixed_draft_width;     /* SPECULATIVE_FIXED_WIDTH=1: every DSpark pass verifies the most draft tokens allowed */
static void keep_for_dspark(unsigned layer, const float *rows, size_t first_position, unsigned count)
{
    for (unsigned a = 0; a < 5; a++)
        if (dspark_hidden[a] && layer == dspark_layers[a]) memcpy(dspark_hidden[a] + first_position * W, rows, (size_t)count * W * sizeof(float));
}

/* k tokens at positions pos0.. through all layers; one token takes the single-token path, which is exactly what plain decoding runs. */
static void forward_tokens(Model *m, const uint32_t *ids, unsigned k, size_t pos0)
{
    stage1_lookup((const unsigned *)ids, k, pass_embedding);
    if (k == 1) { stage2_step(ids, pass_embedding, pass_buffer[0]); stage3_step(pass_buffer[0], pass_embedding, pass_buffer[1]); }
    else { stage2_stepk(ids, pass_embedding, k, pass_buffer[0]); stage3_stepk(pass_buffer[0], pass_embedding, k, pass_buffer[1]); }
    unsigned current = 1, snapshots = 1;
    m->snapshot[0] = pass_embedding;
    for (unsigned l = 2; l <= LAYERS; l++) {
        float *out = pass_buffer[(current + 1) % 3];
        if (k == 1) (is_mla(l) ? lbm_step : lbk_step)(m->layer[l], pass_buffer[current], m->snapshot, pos0, out);
        else (is_mla(l) ? lbm_stepk : lbk_stepk)(m->layer[l], pass_buffer[current], m->snapshot, k, pos0, out);
        keep_for_dspark(l, out, pos0, k);
        if (l % 12 == 0) { memcpy(pass_snapshot[snapshots], pass_buffer[current], (size_t)k * W * sizeof(float)); m->snapshot[snapshots] = pass_snapshot[snapshots]; snapshots++; }
        current = (current + 1) % 3;
    }
    if (snapshots != MAX_SNAPSHOTS) bail("the last layer should hand on eight snapshots");
    m->hidden = pass_buffer[current];
}

/* The model's next token after row `row` of the last pass. */
static void next_token_after(Model *m, unsigned row, uint32_t *token, Text *piece)
{
    for (unsigned s = 0; s < MAX_SNAPSHOTS; s++) memcpy(tail_snapshots + (size_t)s * W, m->snapshot[s] + (size_t)row * W, W * sizeof(float));
    if (!m->tail_run(m->tail_model, m->tail_sequence, m->hidden + (size_t)row * W, tail_snapshots, MAX_SNAPSHOTS, 0, "", token, piece)) bail("decode step failed");
}

static void keep_positions(Model *m, unsigned count)
{
    stage2_keep_positions(count); stage3_keep_positions(count);
    for (unsigned l = 2; l <= LAYERS; l++) (is_mla(l) ? lbm_keep_positions : lbk_keep_positions)(m->layer[l], count);
}

/* Every layer goes back to its state after token `position` of the last pass. */
static void use_position(Model *m, unsigned position)
{
    stage2_use_position(position); stage3_use_position(position);
    for (unsigned l = 2; l <= LAYERS; l++) (is_mla(l) ? lbm_use_position : lbk_use_position)(m->layer[l], position);
}

/* Draft tokens to verify in one pass of the DSpark drafter: the width with the most expected tokens per plain-step time.
   One more position in a pass costs about 0.31 of a plain step (measured 1, 2, 4, 8 positions: 3.33, 4.57, 6.69, 10.65 s, exact trunk);
   each draft token is right with chance q, given the ones before it. 0 means a plain single-token step. */
static unsigned best_draft_width(double q, unsigned most)
{
    unsigned best = 0; double best_rate = 1.0, kept = 0, chance = 1;
    for (unsigned d = 1; d <= most; d++) {
        chance *= q; kept += chance;
        const double rate = (1 + kept) / (1 + 0.31 * d);
        if (rate > best_rate) { best_rate = rate; best = d; }
    }
    return best;
}

static unsigned draft_from_text(const uint32_t *text, unsigned length, unsigned wanted, uint32_t *draft)
{
    for (unsigned n = 3; n >= 2; n--)
        for (long end = (long)length - 2; end >= (long)n - 1; end--) {
            unsigned match = 1;
            for (unsigned i = 0; i < n && match; i++) match = text[end - i] == text[length - 1 - i];
            if (!match) continue;
            const unsigned count = (unsigned)(length - 1 - end) < wanted ? (unsigned)(length - 1 - end) : wanted;
            memcpy(draft, text + end + 1, count * sizeof *draft);
            return count;
        }
    return 0;
}

/* gen[0] is the prompt's next token, not yet fed; fills gen[1..] up to `decode` new tokens; returns how many were made after gen[0]. */
static unsigned speculative_decode(Model *m, const uint32_t *prompt, unsigned N, uint32_t *gen, unsigned decode, unsigned max_draft,
                                   char *text, size_t *text_len, size_t text_capacity, SpeculativeStats *stats)
{
    uint32_t *seen = malloc((N + decode + 2) * sizeof *seen);   /* prompt, then every token made so far */
    memcpy(seen, prompt, N * sizeof *seen);
    seen[N] = gen[0];
    static int positions_kept;   /* a chat session calls this once per turn; the checkpoint buffers are made once */
    if (!positions_kept) { keep_positions(m, max_draft); positions_kept = 1; }
    unsigned produced = 1;
    double hit = 7, tried = 10;   /* DSpark draft tokens right / tried, decayed by 0.9 per pass; starts at 0.7 */
    while (produced < decode + 1 && !chat.done) {
        const unsigned fed = produced - 1, remaining = decode + 1 - produced;
        uint32_t pass_ids[PASS_MAX];
        pass_ids[0] = gen[fed];
        const unsigned wanted = remaining > 1 ? (remaining - 1 < max_draft ? remaining - 1 : max_draft) : 0;
        unsigned drafted = 0;
        uint32_t guess[8];
        const int dspark = wanted && dspark_hidden[0];
        if (dspark) {   /* the draft model guesses 8 tokens from the target's hidden states; the first few are verified, as many as pay off */
            const double t0 = clk();
            dspark_draft((const float *const *)dspark_hidden, gen[fed], (unsigned)(N + fed), guess);
            stats->draft_seconds += clk() - t0;
            drafted = fixed_draft_width ? (wanted < 8 ? wanted : 8) : best_draft_width(hit / tried, wanted < 8 ? wanted : 8);
            memcpy(pass_ids + 1, guess, drafted * sizeof *guess);
        } else if (wanted) drafted = draft_from_text(seen, N + produced, wanted, pass_ids + 1);
        forward_tokens(m, pass_ids, drafted + 1, N + fed);
        unsigned j = 0;
        for (;; j++) {   /* the head is asked only for tokens that are kept, in order, so its own text state never sees a rejected one */
            uint32_t next; Text piece = {0, 0};
            next_token_after(m, j, &next, &piece);
            gen[produced] = next; seen[N + produced] = next; produced++;
            if (chat.on) chat_feed(next, &piece);
            if (*text_len + piece.length < text_capacity) { memcpy(text + *text_len, piece.bytes, piece.length); *text_len += piece.length; }
            if (chat.done || j == drafted || pass_ids[j + 1] != next) break;
        }
        stats->passes++; stats->drafted += drafted; stats->with_draft += drafted > 0;
        stats->accepted += j == drafted ? drafted : j;
        if (dspark) {   /* with no draft verified, the first guess is still compared with the token the model made */
            hit = hit * 0.9 + (drafted ? j : guess[0] == gen[produced - 1]);
            tried = tried * 0.9 + (drafted ? j + (j < drafted) : 1);
        }
        if (j < drafted) {   /* positions after j were fed on a wrong guess (or after the reply ended): go back to the state after token j */
            use_position(m, j);
            stats->rollbacks++;
        }
    }
    free(seen);
    return produced - 1;
}

/* ---- chat session (PF_SESSION_DIR=dir): several turns in one process ----
   A turn is the file dir/turn-NNNN.txt (the user's message, written under another name and renamed into place); the reply is written to
   dir/reply-NNNN.txt, one line about the turn is appended to dir/status.txt, dir/ready appears when the model is loaded, dir/quit ends the session.
   The layers keep their state between turns, so a new turn only runs its own tokens through the prompt pass. When the conversation would not
   leave room for a reply in the context window, the oldest turns are replaced by a summary the model writes and the state is rebuilt from the
   summary and the newest exchange. A message longer than PF_INPUT_MAX tokens is summarised first and the reply is made from the summary.
   Answers at once (PF_CHAT=1) only. */
typedef struct { int assistant; char *text; } ChatTurn;
static struct {
    unsigned context, chunk, reply_reserve, input_max, summary_tokens, max_reply, max_draft;
    int verify;                   /* PF_SESSION_VERIFY=1: after a continued prompt pass, also run one pass over the whole conversation and compare */
    unsigned pos;                 /* positions the layers hold: fed[0..pos) */
    int valid;                    /* 0: the layers hold something else (a summary was just made), the state must be rebuilt */
    uint32_t *fed;
    uint32_t pending;             /* the last token of the previous reply: its closing tag, not fed yet */
    int has_pending;
    char *summary;
    ChatTurn *turn; unsigned turns, turns_cap;
} session;
static float *prefill_buffer[POOL];
static float prefill_last_row[W], prefill_last_snapshots[MAX_SNAPSHOTS * W];   /* what the head read after the last prompt pass */

/* ids[0..count) at positions pos0.. through all 93 layers, at most `chunk` at a time; the head then names the next token after the last one.
   pos0 == 0 starts the layers' state over, anything else continues from the state the earlier tokens left. */
static uint32_t prefill(Model *m, const uint32_t *ids, unsigned count, unsigned pos0, Text *piece)
{
    uint32_t token = 0;
    for (unsigned done = 0; done < count; done += session.chunk) {
        const unsigned n = count - done < session.chunk ? count - done : session.chunk, pos = pos0 + done;
        const int fresh = pos == 0;
        float *pool[POOL]; unsigned free_n = POOL;
        memcpy(pool, prefill_buffer, sizeof pool);
        float *s0 = pool[--free_n], *cur = pool[--free_n], *next = pool[--free_n];
        stage1_lookup((const unsigned *)(ids + done), n, s0);
        if (fresh) stage2_run(ids + done, s0, n, n, cur); else stage2_run_from(ids + done, s0, n, n, cur);
        if (fresh) stage3_run(cur, s0, n, n, next); else stage3_run_from(cur, s0, n, n, next);
        { float *t = cur; cur = next; next = t; }
        pool[free_n++] = next;
        const float *snaps[MAX_SNAPSHOTS] = { s0 };
        unsigned k = 1;
        for (unsigned l = 2; l <= LAYERS; l++) {
            float *out = pool[--free_n];
            if (is_mla(l)) { if (fresh) lbm_run(m->layer[l], cur, snaps, n, n, out); else lbm_run_from(m->layer[l], cur, snaps, n, n, pos, out); }
            else { if (fresh) lbk_run(m->layer[l], cur, snaps, n, n, out); else lbk_run_from(m->layer[l], cur, snaps, n, n, pos, out); }
            keep_for_dspark(l, out, pos, n);
            if (l % 12 == 0) snaps[k++] = cur; else pool[free_n++] = cur;
            cur = out;
        }
        if (k != MAX_SNAPSHOTS) bail("the last layer should hand on eight snapshots");
        if (done + n == count) {
            for (unsigned s = 0; s < MAX_SNAPSHOTS; s++) memcpy(tail_snapshots + (size_t)s * W, snaps[s] + (size_t)(n - 1) * W, W * sizeof(float));
            memcpy(prefill_last_row, cur + (size_t)(n - 1) * W, W * sizeof(float));
            memcpy(prefill_last_snapshots, tail_snapshots, sizeof prefill_last_snapshots);
            if (!m->tail_run(m->tail_model, m->tail_sequence, cur + (size_t)(n - 1) * W, tail_snapshots, MAX_SNAPSHOTS, 0, "", &token, piece)) bail("tail failed");
        }
    }
    return token;
}

static unsigned count_tokens(const char *text, size_t length)
{
    if (!length) return 0;
    int *ids = malloc((length + 16) * sizeof *ids);
    if (!ids) bail("allocation failed");
    const unsigned n = (unsigned)stage1_ids_plain(text, (long)length, ids, (int)(length + 16));
    free(ids);
    return n;
}

static void turn_push(int assistant, const char *text)
{
    if (session.turns == session.turns_cap) {
        session.turns_cap = session.turns_cap ? 2 * session.turns_cap : 16;
        session.turn = realloc(session.turn, session.turns_cap * sizeof *session.turn);
        if (!session.turn) bail("allocation failed");
    }
    session.turn[session.turns++] = (ChatTurn){ assistant, strdup(text) };
}

static int user_message_ids(int *ids, int cap, int n, const char *text)
{
    return chat_message(ids, cap, n, " role=\"user\"", text, (long)strlen(text));
}

static int assistant_message_ids(int *ids, int cap, int n, const char *text)
{
    ids[n++] = chat.open; n = plain(ids, cap, n, "message"); n = plain(ids, cap, n, " role=\"assistant\""); ids[n++] = chat.sep;
    ids[n++] = chat.open; n = plain(ids, cap, n, "response"); ids[n++] = chat.sep;
    if (*text) n += stage1_ids_plain(text, (long)strlen(text), ids + n, cap - n);
    ids[n++] = chat.close; n = plain(ids, cap, n, "response"); ids[n++] = chat.sep;
    ids[n++] = chat.close; n = plain(ids, cap, n, "message"); ids[n++] = chat.sep; ids[n++] = chat.eom;
    return n;
}

static int reply_open_ids(int *ids, int cap, int n)
{
    ids[n++] = chat.open; n = plain(ids, cap, n, "message"); n = plain(ids, cap, n, " role=\"assistant\""); ids[n++] = chat.sep;
    ids[n++] = chat.open; n = plain(ids, cap, n, "response"); ids[n++] = chat.sep;
    return n;
}

/* What follows the last reply's closing tag in the transcript: the rest of the assistant message. A reply that was cut short is closed here. */
static int reply_end_ids(int *ids, int cap, int n)
{
    const int pending = (int)session.pending;
    if (pending != chat.eos && pending != chat.eom) ids[n++] = pending;
    if (pending != chat.close) ids[n++] = chat.close;
    n = plain(ids, cap, n, "response"); ids[n++] = chat.sep;
    ids[n++] = chat.close; n = plain(ids, cap, n, "message"); ids[n++] = chat.sep; ids[n++] = chat.eom;
    return n;
}

static int system_ids(int *ids, int cap, int n)
{
    if (chat.system && *chat.system) n = chat_message(ids, cap, n, " role=\"system\"", chat.system, (long)strlen(chat.system));
    if (session.summary) {
        const size_t size = strlen(session.summary) + 64;
        char *text = malloc(size);
        if (!text) bail("allocation failed");
        snprintf(text, size, "Summary of the earlier conversation:\n%s", session.summary);
        n = chat_message(ids, cap, n, " role=\"system\"", text, (long)strlen(text));
        free(text);
    }
    return n;
}

/* The whole conversation as the model reads it, ending where its next reply starts. */
static int history_ids(int *ids, int cap, const char *new_user)
{
    int n = system_ids(ids, cap, 0);
    for (unsigned i = 0; i < session.turns; i++)
        n = session.turn[i].assistant ? assistant_message_ids(ids, cap, n, session.turn[i].text) : user_message_ids(ids, cap, n, session.turn[i].text);
    return reply_open_ids(ids, cap, user_message_ids(ids, cap, n, new_user));
}

static int ids_capacity(const char *new_user)
{
    size_t bytes = strlen(new_user) + 4096 + (session.summary ? strlen(session.summary) : 0) + (chat.system ? strlen(chat.system) : 0);
    for (unsigned i = 0; i < session.turns; i++) bytes += strlen(session.turn[i].text) + 64;
    return (int)bytes;   /* a token is at least one byte */
}

/* A prompt pass over `ids`, then the reply, up to max_new tokens: gen[0..steps] (the last one is not fed), the text in chat.response. */
static unsigned reply_to(Model *m, uint32_t *ids, unsigned n, unsigned max_new, uint32_t *gen, SpeculativeStats *stats, double *prefill_seconds)
{
    Text piece = {0, 0};
    const double t0 = clk();
    gen[0] = prefill(m, ids, n, 0, &piece);
    *prefill_seconds = clk() - t0;
    chat.done = 0; chat.closes = 0; chat.in_tag = 0; chat.to_response = 1; chat.think_len = 0; chat.response_len = 0;
    chat_feed(gen[0], &piece);
    static char text[1 << 16]; size_t text_len = 0;
    return speculative_decode(m, ids, n, gen, max_new, session.max_draft, text, &text_len, sizeof text, stats);
}

/* The model's own summary of one piece of text that fits the context window; it destroys the layers' state. */
static char *summarize_piece(Model *m, const char *text)
{
    char instruction[512];
    snprintf(instruction, sizeof instruction, "Summarize the following text in at most %u words. Start with every fact the user stated about themselves, their goals or their requirements (names, preferences, constraints), word for word, then give the main points, decisions, numbers and open questions. Write plain prose.\n\n", session.summary_tokens * 3 / 4);
    const size_t size = strlen(instruction) + strlen(text) + 1;
    char *request = malloc(size);
    if (!request) bail("allocation failed");
    snprintf(request, size, "%s%s", instruction, text);
    const int cap = (int)size + 4096;
    int *ids = malloc((size_t)cap * sizeof *ids);
    uint32_t *gen = malloc((session.summary_tokens + 1) * sizeof *gen);
    if (!ids || !gen) bail("allocation failed");
    const char *role = "You write faithful, compact summaries.";
    int n = chat_message(ids, cap, 0, " role=\"system\"", role, (long)strlen(role));
    n = reply_open_ids(ids, cap, user_message_ids(ids, cap, n, request));
    session.valid = 0; session.pos = 0; session.has_pending = 0; dspark_reset();
    SpeculativeStats stats = {0}; double prefill_seconds;
    reply_to(m, (uint32_t *)ids, (unsigned)n, session.summary_tokens, gen, &stats, &prefill_seconds);
    char *summary = strndup(chat.response, chat.response_len);
    free(request); free(ids); free(gen);
    return summary ? summary : strdup("");
}

/* A summary of any length of text: pieces that fit the window are summarised one by one, and the joined summaries again if they are still too long. */
static char *summarize(Model *m, const char *text)
{
    const size_t length = strlen(text);
    const unsigned piece_tokens = session.context / 2, total = count_tokens(text, length);
    if (total <= piece_tokens) return summarize_piece(m, text);
    const size_t piece_bytes = (size_t)((double)length / total * piece_tokens * 0.9);
    char *joined = strdup(""); size_t joined_size = 1;
    for (size_t at = 0; at < length; ) {
        size_t end = at + piece_bytes < length ? at + piece_bytes : length;
        if (end < length) {   /* end the piece at the last line (else space) break in its final fifth */
            size_t cut = end;
            while (cut > at + piece_bytes * 4 / 5 && text[cut - 1] != '\n') cut--;
            if (text[cut - 1] != '\n') { cut = end; while (cut > at + piece_bytes * 4 / 5 && text[cut - 1] != ' ') cut--; }
            if (cut > at + piece_bytes * 4 / 5) end = cut;
        }
        char *piece = strndup(text + at, end - at);
        char *part = summarize_piece(m, piece);
        joined_size += strlen(part) + 2;
        joined = realloc(joined, joined_size);
        strcat(joined, part); strcat(joined, "\n\n");
        free(piece); free(part);
        at = end;
    }
    char *result = count_tokens(joined, strlen(joined)) > piece_tokens ? summarize(m, joined) : joined;
    if (result != joined) free(joined);
    return result;
}

/* The first `count` turns, after the summary so far, as text to be summarised. */
static char *transcript(unsigned count)
{
    size_t size = 64 + (session.summary ? strlen(session.summary) : 0);
    for (unsigned i = 0; i < count; i++) size += strlen(session.turn[i].text) + 16;
    char *text = malloc(size);
    if (!text) bail("allocation failed");
    text[0] = 0;
    if (session.summary) { strcat(text, "Earlier summary: "); strcat(text, session.summary); strcat(text, "\n\n"); }
    for (unsigned i = 0; i < count; i++) { strcat(text, session.turn[i].assistant ? "Assistant: " : "User: "); strcat(text, session.turn[i].text); strcat(text, "\n\n"); }
    return text;
}

static void write_file(const char *path, const char *text, size_t length)
{
    char temporary[4200];
    snprintf(temporary, sizeof temporary, "%s.tmp", path);
    FILE *f = fopen(temporary, "wb");
    if (!f) bail("cannot write a session file");
    fwrite(text, 1, length, f);
    fclose(f);
    rename(temporary, path);
}

/* Every summary the model writes is kept in dir/<kind>-NNNN.txt, so what the conversation forgot can be read. */
static void keep_summary(const char *dir, const char *kind, unsigned number, const char *text)
{
    char path[4200];
    snprintf(path, sizeof path, "%s/%s-%04u.txt", dir, kind, number);
    write_file(path, text, strlen(text));
}

static void session_turn(Model *m, const char *dir, unsigned number, char *user)
{
    const double began = clk();
    char path[4200], status[1500];
    int summarised_input = 0;
    unsigned user_tokens = count_tokens(user, strlen(user));
    if (user_tokens > session.input_max) {
        char *summary = summarize(m, user);
        keep_summary(dir, "input-summary", number, summary);
        free(user); user = summary; summarised_input = 1;
        user_tokens = count_tokens(user, strlen(user));
    }
    int cap = ids_capacity(user);
    int *ids = malloc((size_t)cap * sizeof *ids);
    if (!ids) bail("allocation failed");
    int n = 0;
    const int can_continue = session.valid && session.pos > 0;
    if (can_continue) {
        if (session.has_pending) n = reply_end_ids(ids, cap, n);
        n = reply_open_ids(ids, cap, user_message_ids(ids, cap, n, user));
    }
    unsigned continued_from = can_continue ? session.pos : 0;
    unsigned dropped_turns = 0;
    if (!can_continue || session.pos + (unsigned)n + session.reply_reserve > session.context) {
        /* start the state over from the text: with the oldest turns replaced by a summary when the conversation does not fit */
        continued_from = 0;
        for (int attempt = 0;; attempt++) {
            cap = ids_capacity(user);
            ids = realloc(ids, (size_t)cap * sizeof *ids);
            n = history_ids(ids, cap, user);
            if ((unsigned)n + session.reply_reserve <= session.context || session.turns == 0) break;
            unsigned keep = attempt == 0 && session.turns > 2 ? 2 : 0, old = session.turns - keep;
            char *text = transcript(old);
            char *summary = summarize(m, text);
            keep_summary(dir, "conversation-summary", number, summary);
            free(text); free(session.summary); session.summary = summary;
            for (unsigned i = 0; i < old; i++) free(session.turn[i].text);
            memmove(session.turn, session.turn + old, keep * sizeof *session.turn);
            session.turns = keep; dropped_turns += old;
        }
        if ((unsigned)n + 24 > session.context) bail("the message does not fit the context window even after summarising");
        session.pos = 0; session.has_pending = 0; dspark_reset();
    }
    Text piece = {0, 0};
    const double t_prefill = clk();
    uint32_t first = prefill(m, (const uint32_t *)ids, (unsigned)n, continued_from, &piece);
    const double prefill_seconds = clk() - t_prefill;
    memcpy(session.fed + continued_from, ids, (size_t)n * sizeof *ids);
    session.pos = continued_from + (unsigned)n; session.valid = 1;
    free(ids);
    char verified[300] = "";
    if (session.verify && continued_from) {   /* the state a continued pass left, against the state one pass over everything leaves */
        static float row_before[W], snapshots_before[MAX_SNAPSHOTS * W];
        memcpy(row_before, prefill_last_row, sizeof row_before); memcpy(snapshots_before, prefill_last_snapshots, sizeof snapshots_before);
        Text again = {0, 0};
        dspark_reset();
        const double t_check = clk();
        const uint32_t token_again = prefill(m, session.fed, session.pos, 0, &again);
        unsigned different = 0; double largest = 0;
        for (unsigned i = 0; i < W; i++) { const double d = fabs((double)row_before[i] - prefill_last_row[i]); different += row_before[i] != prefill_last_row[i]; if (d > largest) largest = d; }
        for (unsigned i = 0; i < MAX_SNAPSHOTS * W; i++) { const double d = fabs((double)snapshots_before[i] - prefill_last_snapshots[i]); different += snapshots_before[i] != prefill_last_snapshots[i]; if (d > largest) largest = d; }
        snprintf(verified, sizeof verified, "; continued pass against one pass over all %u positions (%.1f s): %s, next token %s", session.pos, clk() - t_check,
            different ? "DIFFERENT" : "bit-identical", token_again == first ? "same" : "DIFFERENT");
        if (different) { const size_t used = strlen(verified); snprintf(verified + used, sizeof verified - used, " (%u of %u values differ, largest difference %g)", different, (unsigned)(W + MAX_SNAPSHOTS * W), largest); }
    }

    chat.done = 0; chat.closes = 0; chat.in_tag = 0; chat.to_response = 1; chat.think_len = 0; chat.response_len = 0;
    chat_feed(first, &piece);
    const unsigned room = session.context - session.pos - 24;
    const unsigned max_new = session.max_reply < room ? session.max_reply : room;
    uint32_t *gen = malloc((max_new + 1) * sizeof *gen);
    if (!gen) bail("allocation failed");
    gen[0] = first;
    static char text[1 << 16]; size_t text_len = 0;
    SpeculativeStats stats = {0};
    const double t_reply = clk();
    const unsigned steps = speculative_decode(m, session.fed, session.pos, gen, max_new, session.max_draft, text, &text_len, sizeof text, &stats);
    const double reply_seconds = clk() - t_reply;
    memcpy(session.fed + session.pos, gen, steps * sizeof *gen);
    session.pos += steps; session.pending = gen[steps]; session.has_pending = 1;
    free(gen);

    char *reply = strndup(chat.response, chat.response_len);
    turn_push(0, user); turn_push(1, reply);
    snprintf(path, sizeof path, "%s/reply-%04u.txt", dir, number);
    write_file(path, reply, strlen(reply));
    snprintf(status, sizeof status,
        "turn %u: message %u tokens%s; %s from position %u, prefill %u tokens in %.1f s; reply %u tokens in %.1f s (%.2f s per token, %u passes, %u of %u draft tokens kept)%s%s; "
        "positions now %u of %u; %u turns kept%s; wall %.1f s%s\n",
        number, user_tokens, summarised_input ? " (summarised first)" : "", continued_from ? "continued" : "rebuilt", continued_from, (unsigned)n, prefill_seconds,
        steps + 1, reply_seconds, steps ? reply_seconds / steps : 0.0, stats.passes, stats.accepted, stats.drafted,
        chat.done ? "" : " [cut at the token limit]", dropped_turns ? " [oldest turns summarised]" : "",
        session.pos, session.context, session.turns, session.summary ? " plus a summary" : "", clk() - began, verified);
    snprintf(path, sizeof path, "%s/status.txt", dir);
    FILE *log = fopen(path, "ab");
    if (log) { fputs(status, log); fclose(log); }
    fputs(status, stdout); fflush(stdout);
    free(reply); free(user);
}

static void run_session(Model *m, const char *dir)
{
    char path[4200];
    session.fed = malloc((session.context + 64) * sizeof *session.fed);
    for (unsigned i = 0; i < POOL; i++) {
        prefill_buffer[i] = aligned_alloc(64, ((size_t)session.chunk * W * sizeof(float) + 63) / 64 * 64);
        if (!prefill_buffer[i]) bail("allocation failed");
        memset(prefill_buffer[i], 0, (size_t)session.chunk * W * sizeof(float));
    }
    snprintf(path, sizeof path, "%s/ready", dir);
    write_file(path, "ready\n", 6);
    printf("SESSION ready in %s: context %u positions, chunk %u, reply reserve %u, input limit %u tokens\n", dir, session.context, session.chunk, session.reply_reserve, session.input_max);
    fflush(stdout);
    for (unsigned number = 1;;) {
        char quit[4200], turn_file[4200];
        snprintf(quit, sizeof quit, "%s/quit", dir);
        snprintf(turn_file, sizeof turn_file, "%s/turn-%04u.txt", dir, number);
        while (access(turn_file, F_OK) != 0) {
            if (access(quit, F_OK) == 0) { printf("SESSION ended\n"); return; }
            const struct timespec pause = { 0, 300 * 1000 * 1000 };
            nanosleep(&pause, NULL);
        }
        long size; char *user = slurp(turn_file, &size);
        while (size > 0 && (user[size - 1] == '\n' || user[size - 1] == '\r' || user[size - 1] == ' ' || user[size - 1] == '\t')) size--;
        user[size] = 0;
        session_turn(m, dir, number, user);
        number++;
    }
}

int main(int argc, char **argv)
{
    if (argc != 6) { fputs("usage: whole-model BUILD_DIR DATASET_DIR OPEN_WEIGHTS_DIR WORDS REFDIR\n", stderr); return 2; }
    const char *build = argv[1], *dataset = argv[2];   /* build: the folder holding pipeline-stage.so */
    const int compare = strcmp(argv[5], "-") != 0;
    const int verbose = getenv("PF_VERBOSE") && *getenv("PF_VERBOSE");
    char seed[4096], path[4096];
    snprintf(seed, sizeof seed, "%s/" OPEN_WEIGHTS_SHARD "#language_model.model.embed_tokens.weight", argv[3]);


    long words_size; char *words = slurp(argv[4], &words_size);
    /* A prompt that ends in a space tokenises to a dangling space token the model reads as the start of a word, and answers badly. */
    { const long before = words_size;
      while (words_size > 0 && (words[words_size - 1] == ' ' || words[words_size - 1] == '\t')) words_size--;
      words[words_size] = 0;
      if (words_size != before) printf("prompt: dropped %ld trailing space(s)\n", before - words_size); }
    unsigned long word_count = 0; int inside = 0;
    for (long i = 0; i < words_size; i++) {
        const int space = words[i] == ' ' || words[i] == '\n' || words[i] == '\t' || words[i] == '\r';
        if (!space && !inside) word_count++;
        inside = !space;
    }

    /* ---- one-time setup ---- */
    const double setup0 = clk();
    /* PF_DECODE=G generates G more tokens after the prompt; PF_EXTRA=file appends the ids in it to the prompt (a check that the
       prompt pass over a longer prompt gives what the decode steps gave); PF_SAVE_ROWS / PF_CMP_ROWS save / compare the
       layer-92 output rows of the decode steps. */
    const unsigned decode = getenv("PF_DECODE") ? (unsigned)atoi(getenv("PF_DECODE")) : 0;
    /* PF_SWEEP=k: after the prompt, feed the ids in PF_SWEEP_IDS (a PF_SAVE_IDS file from an earlier run of this prompt) k at a time, as one
       pass each, the way a draft of k tokens would be verified; PF_SWEEP_N = the most tokens to feed. */
    const unsigned sweep_k = getenv("PF_SWEEP") && *getenv("PF_SWEEP") ? (unsigned)atoi(getenv("PF_SWEEP")) : 0;
    const unsigned sweep_n = getenv("PF_SWEEP_N") && *getenv("PF_SWEEP_N") ? (unsigned)atoi(getenv("PF_SWEEP_N")) : 0;
    if (sweep_k > 16) bail("PF_SWEEP is at most 16");
    const unsigned max_draft = getenv("SPECULATIVE_DRAFT_TOKENS") && *getenv("SPECULATIVE_DRAFT_TOKENS") ? (unsigned)atoi(getenv("SPECULATIVE_DRAFT_TOKENS")) : 0;
    if (max_draft >= PASS_MAX) bail("SPECULATIVE_DRAFT_TOKENS is at most 15");
    fixed_draft_width = getenv("SPECULATIVE_FIXED_WIDTH") && *getenv("SPECULATIVE_FIXED_WIDTH");
    /* PF_SESSION_DIR=dir: a chat with several turns in one process (see run_session); PF_CONTEXT = positions the layers can hold, PF_CHUNK = tokens per prompt pass */
    const char *session_dir = getenv("PF_SESSION_DIR") && *getenv("PF_SESSION_DIR") ? getenv("PF_SESSION_DIR") : NULL;
    if (session_dir) {
        const char *v;
        session.context = (v = getenv("PF_CONTEXT")) && *v ? (unsigned)atoi(v) : 8192;
        session.chunk = (v = getenv("PF_CHUNK")) && *v ? (unsigned)atoi(v) : 1024;
        session.reply_reserve = (v = getenv("PF_REPLY_RESERVE")) && *v ? (unsigned)atoi(v) : 1024;
        session.input_max = (v = getenv("PF_INPUT_MAX")) && *v ? (unsigned)atoi(v) : session.context / 2;
        session.summary_tokens = (v = getenv("PF_SUMMARY_TOKENS")) && *v ? (unsigned)atoi(v) : 320;
        session.max_reply = decode; session.max_draft = max_draft;
        session.verify = getenv("PF_SESSION_VERIFY") && *getenv("PF_SESSION_VERIFY") && atoi(getenv("PF_SESSION_VERIFY"));
        if (!max_draft) bail("a chat session needs SPECULATIVE_DRAFT_TOKENS (its replies are made by the speculative decoder)");
        if (!getenv("PF_CHAT") || (strcmp(getenv("PF_CHAT"), "1") && strcmp(getenv("PF_CHAT"), "nothink"))) bail("a chat session needs PF_CHAT=1 (answers at once)");
        if (session.chunk < 64 || session.context < 2 * session.chunk || session.reply_reserve + 256 > session.context) bail("PF_CHUNK, PF_CONTEXT and PF_REPLY_RESERVE do not fit each other");
    }
    if (max_draft && sweep_k) bail("SPECULATIVE_DRAFT_TOKENS and PF_SWEEP are separate experiments");
    const unsigned pass_k = sweep_k > max_draft + 1 ? sweep_k : max_draft + 1;
    const int routes_on = getenv("PF_ROUTES") && *getenv("PF_ROUTES");
    unsigned long *hits = calloc((size_t)LAYERS * 896, sizeof *hits);          /* prompt pass: how often each expert was chosen, layers 1..92 */
    unsigned short *route_rec = calloc((size_t)(decode + 1) * LAYERS * 48, sizeof *route_rec);   /* decode steps: 16 chosen + 32 guessed per layer */
    unsigned extra[512], extra_n = 0;
    if (getenv("PF_EXTRA") && *getenv("PF_EXTRA")) {
        long l; char *list = slurp(getenv("PF_EXTRA"), &l);
        for (char *tok = strtok(list, ", \n\r"); tok && extra_n < 512; tok = strtok(NULL, ", \n\r")) extra[extra_n++] = (unsigned)strtoul(tok, NULL, 10);
    }
    if (extra_n && compare) bail("recorded values do not exist for a prompt with extra ids");
    stage1_setup(argv[3], seed);
    {   const char *mode = getenv("PF_CHAT");
        if (mode && *mode) {
            chat.on = 1;
            chat.think = !strcmp(mode, "low") || !strcmp(mode, "high") || !strcmp(mode, "max");
            if (!chat.think && strcmp(mode, "1") && strcmp(mode, "nothink")) bail("PF_CHAT must be 1, nothink, low, high or max");
            chat.effort = mode;
            chat.system = getenv("PF_SYSTEM");
            chat.open = stage1_id_of("<|open|>"); chat.close = stage1_id_of("<|close|>"); chat.sep = stage1_id_of("<|sep|>");
            chat.eom = stage1_id_of("<|end_of_msg|>"); chat.eos = stage1_id_of("[EOS]");
            if (chat.open < 0 || chat.close < 0 || chat.sep < 0 || chat.eom < 0 || chat.eos < 0) bail("the tokenizer lacks the chat control tokens");
            chat.to_response = !chat.think;
            if (!decode) bail("PF_CHAT needs PF_DECODE, the most new tokens to generate");
        }
    }
    const int cap = (int)words_size + 16 + (int)extra_n + 1024 + (chat.system ? 2 * (int)strlen(chat.system) : 0);
    int *ids = malloc((size_t)cap * sizeof *ids);
    const int found = prompt_ids(words, words_size, ids, cap);
    if (found <= 0) bail("the words gave no tokens");
    for (unsigned i = 0; i < extra_n; i++) ids[found + i] = (int)extra[i];
    if (found <= 128) { printf("prompt ids:"); for (int i = 0; i < found; i++) printf(" %d", ids[i]); printf("\n"); }
    const unsigned N0 = (unsigned)found, N = session_dir ? session.chunk : N0 + extra_n;   /* a session sizes its buffers for one prompt chunk */
    const unsigned layer_positions = session_dir ? session.context + 16 : N + decode + sweep_n + 16;   /* what the attention caches and the draft hold */
    float *pool[POOL]; unsigned free_n = 0;
    for (unsigned i = 0; i < POOL; i++) {
        pool[i] = aligned_alloc(64, ((size_t)N * W * sizeof(float) + 63) / 64 * 64);
        if (!pool[i]) bail("allocation failed");
        memset(pool[i], 0, (size_t)N * W * sizeof(float));
        free_n++;
    }
    float *s0 = pool[--free_n];
    float *ref_embed = NULL, *ref_resid = NULL, *ref_snaps = NULL;
    if (compare) {
        long l;
        snprintf(path, sizeof path, "%s/embed.f32", argv[5]);        ref_embed = (float *)slurp(path, &l); if ((size_t)l < (size_t)N * W * 4) bail("embed.f32 too short");
        snprintf(path, sizeof path, "%s/final.resid.f32", argv[5]);  ref_resid = (float *)slurp(path, &l); if ((size_t)l < (size_t)N * W * 4) bail("final.resid.f32 too short");
        snprintf(path, sizeof path, "%s/final.snaps.f32", argv[5]);  ref_snaps = (float *)slurp(path, &l); if ((size_t)l < (size_t)N * W * 4 * MAX_SNAPSHOTS) bail("final.snaps.f32 too short");
    }
    /* the experts of the running layer: one buffer for all layers, on huge pages */
    unsigned char *experts = mmap(NULL, EXPERT_LAYER_BYTES, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (experts == MAP_FAILED) bail("cannot allocate the expert buffer");
    madvise(experts, EXPERT_LAYER_BYTES, MADV_HUGEPAGE);
    memset(experts, 0, EXPERT_LAYER_BYTES);
    const double t_buffer = clk() - setup0;
    /* PF_CACHE=C: every layer gets C expert slots for decode steps (the 896 slots of the prompt buffer are reused, the rest is allocated here). */
    const unsigned cache_cap = getenv("PF_CACHE") && *getenv("PF_CACHE") ? (unsigned)atoi(getenv("PF_CACHE")) : 0;
    unsigned char *cache_extra = NULL, *slot_ptrs[64];
    if (cache_cap && (size_t)LAYERS * cache_cap > 896) {
        const size_t extra_bytes = ((size_t)LAYERS * cache_cap - 896) * EXPERT_BYTES;
        cache_extra = mmap(NULL, extra_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (cache_extra == MAP_FAILED) bail("cannot allocate the expert cache");
        madvise(cache_extra, extra_bytes, MADV_HUGEPAGE);
        memset(cache_extra, 0, extra_bytes);
    }
#define FILL_SLOTS(layer_index) for (unsigned j_ = 0; j_ < cache_cap; j_++) { \
        const size_t idx_ = (size_t)(layer_index) * cache_cap + j_; \
        slot_ptrs[j_] = idx_ < 896 ? experts + idx_ * EXPERT_BYTES : cache_extra + (idx_ - 896) * EXPERT_BYTES; }
    stage2_setup(dataset, N + 16);
    stage3_use_buffer(experts);
    stage3_setup(dataset, N + 16);
    if (cache_cap) { FILL_SLOTS(0) stage3_set_slots(slot_ptrs, cache_cap); }
    lbk_set_buffer(experts); lbm_set_buffer(experts);
    lbk_init(N + 16); lbm_init(N + 16);
    if (pass_k > 1) {   /* experts a window needs beyond the layer's cache slots: one pool shared by all layers */
        const unsigned n_scratch = 16 * pass_k < 128 ? 16 * pass_k : 128;
        unsigned char *scratch_mem = mmap(NULL, (size_t)n_scratch * EXPERT_BYTES, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (scratch_mem == MAP_FAILED) bail("cannot allocate the scratch pool");
        madvise(scratch_mem, (size_t)n_scratch * EXPERT_BYTES, MADV_HUGEPAGE);
        memset(scratch_mem, 0, (size_t)n_scratch * EXPERT_BYTES);
        unsigned char *scratch_ptrs[128];
        for (unsigned j = 0; j < n_scratch; j++) scratch_ptrs[j] = scratch_mem + (size_t)j * EXPERT_BYTES;
        stage3_set_scratch(scratch_ptrs, n_scratch); lbk_set_scratch(scratch_ptrs, n_scratch); lbm_set_scratch(scratch_ptrs, n_scratch);
    }
    const double t_edges = clk() - setup0;
    void *layer[LAYERS + 1] = { NULL };
    for (unsigned l = 2; l <= LAYERS; l++) {
        layer[l] = is_mla(l) ? lbm_open(dataset, l, snapshots_in(l), layer_positions) : lbk_open(dataset, l, snapshots_in(l), layer_positions);
        if (cache_cap) { FILL_SLOTS(l - 1) (is_mla(l) ? lbm_set_slots : lbk_set_slots)(layer[l], slot_ptrs, cache_cap); }
    }
    if (getenv("SPECULATIVE_DRAFTER") && !strcmp(getenv("SPECULATIVE_DRAFTER"), "dspark")) {   /* the DSpark draft model instead of the prompt-lookup draft */
        const char *checkpoint = getenv("DSPARK_CHECKPOINT");
        if (!max_draft) bail("SPECULATIVE_DRAFTER=dspark needs SPECULATIVE_DRAFT_TOKENS (up to 8)");
        if (!checkpoint || !*checkpoint) bail("DSPARK_CHECKPOINT is not set");
        snprintf(path, sizeof path, "%s/" OPEN_WEIGHTS_SHARD "#language_model.lm_head.weight", argv[3]);
        for (unsigned a = 0; a < 5; a++) dspark_hidden[a] = calloc((size_t)(session_dir ? layer_positions : N + decode + 16) * W, sizeof(float));
        if (!dspark_open(checkpoint, path, session_dir ? layer_positions : N + decode + 16)) bail("cannot open the DSpark draft model");
    }
    const double t_layers = clk() - setup0;

    char leaves[4096], head[4096], tik[4096], config[4096];
    snprintf(path, sizeof path, "%s/pipeline-stage.so", build);
    void *tail = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!tail) { fprintf(stderr, "tail dlopen: %s\n", dlerror()); return 1; }
    int (*tail_open)(const char *, const char *, const char *, const char *, const char *, void **) = dlsym(tail, "transformer_93_open");
    void *(*tail_make)(const void *) = dlsym(tail, "transformer_93_sequence_create");
    void (*tail_drop)(void *) = dlsym(tail, "transformer_93_sequence_close");
    int (*tail_run)(const void *, void *, const float *, const float *, unsigned, int, const char *, uint32_t *, Text *) = dlsym(tail, "transformer_93_process");
    if (!tail_open || !tail_make || !tail_drop || !tail_run) bail("tail symbols missing");
    snprintf(leaves, sizeof leaves, "%s/" OPEN_WEIGHTS_SHARD, argv[3]);
    snprintf(head, sizeof head, "%s/" OPEN_WEIGHTS_SHARD "#language_model.lm_head.weight", argv[3]);
    snprintf(tik, sizeof tik, "%s/tiktoken.model", argv[3]);
    snprintf(config, sizeof config, "%s/tokenizer_config.json", argv[3]);
    void *tail_model = NULL;
    /* no vocabulary file: the tokenizer builds its table from tiktoken.model */
    if (!tail_open(leaves, head, tik, NULL, config, &tail_model)) bail("tail open failed");
    {   /* touch the head's table once, with finite dummy vectors, so the request finds it in memory */
        static float one[W], eight[MAX_SNAPSHOTS * W];
        for (unsigned i = 0; i < W; i++) one[i] = 0.01f * (float)(i % 7);
        for (unsigned i = 0; i < MAX_SNAPSHOTS * W; i++) eight[i] = 0.01f * (float)(i % 5);
        uint32_t token; Text word;
        void *seq = tail_make(tail_model);
        if (!seq || !tail_run(tail_model, seq, one, eight, MAX_SNAPSHOTS, 0, "", &token, &word)) bail("tail warm-up failed");
        tail_drop(seq);
    }
    const double setup = clk() - setup0;
    {   /* which arithmetic this process runs */
        const char *experts_8bit = getenv("LB_EXPK"), *trunk_4bit = getenv("USE_4BIT_TRUNK"), *trunk_8bit_activations = getenv("LB_A8T");
        const int use_8bit_experts = experts_8bit && atoi(experts_8bit) == 3, use_4bit_trunk = trunk_4bit && atoi(trunk_4bit) != 0;
        const int use_8bit_trunk_activations = trunk_8bit_activations && atoi(trunk_8bit_activations) != 0;
        printf("ARITHMETIC: %s%s%s\n", use_4bit_trunk ? "trunk 4-bit in decode steps (all 93 layers), prompt pass exact (decode not bit-identical)" : "trunk exact",
            use_8bit_experts ? "; prompt-pass experts with 8-bit activations (not bit-identical)" : "; experts exact",
            use_8bit_trunk_activations ? "; prompt-pass trunk matrices of layers 1-92 with 8-bit activations (not bit-identical)"
                : (use_4bit_trunk || use_8bit_experts) ? "" : "  -> bit-identical to the pod");
    }

    /* ---- the request: words in memory -> next token and text in memory ---- */
    if (session_dir) {
        void *session_sequence = tail_make(tail_model);
        if (!session_sequence) bail("tail sequence failed");
        Model session_model = { .tail_model = tail_model, .tail_sequence = session_sequence, .tail_run = tail_run };
        memcpy(session_model.layer, layer, sizeof session_model.layer);
        run_session(&session_model, session_dir);
        return 0;
    }
    double layer_time[LAYERS + 1] = {0};
    const double a = clk();
    const int got = prompt_ids(words, words_size, ids, cap);
    const double b = clk();
    if (got != (int)N0) bail("the second tokenisation gave a different token count");
    stage1_lookup((const unsigned *)ids, N, s0);
    const double c = clk();
    float *cur = pool[--free_n], *next = pool[--free_n];
    stage2_run((const uint32_t *)ids, s0, N, N, cur);
    const double d = clk();
    stage3_run(cur, s0, N, N, next);
    if (routes_on) stage3_hits(hits);
    { float *t = cur; cur = next; next = t; }
    const double e = clk();
    pool[free_n++] = next;
    const float *snaps[MAX_SNAPSHOTS] = { s0 };
    unsigned k = 1;
    for (unsigned l = 2; l <= LAYERS; l++) {
        if (snapshots_in(l) != k) bail("snapshot count does not match the layer");
        float *out = pool[--free_n];
        const double began = clk();
        if (is_mla(l)) lbm_run(layer[l], cur, snaps, N, N, out); else lbk_run(layer[l], cur, snaps, N, N, out);
        keep_for_dspark(l, out, 0, N);
        layer_time[l] = clk() - began;
        if (routes_on) (is_mla(l) ? lbm_hits : lbk_hits)(hits + (size_t)(l - 1) * 896);
        if (l % 12 == 0) snaps[k++] = cur;      /* the layer's input becomes the next snapshot: a pointer, no copy */
        else pool[free_n++] = cur;
        cur = out;
    }
    const double f = clk();
    if (k != MAX_SNAPSHOTS) bail("the last layer should hand on eight snapshots");
    static float last_snaps[MAX_SNAPSHOTS * W];
    for (unsigned s = 0; s < MAX_SNAPSHOTS; s++) memcpy(last_snaps + (size_t)s * W, snaps[s] + (size_t)(N - 1) * W, W * sizeof(float));
    uint32_t token = 0; Text word = {0, 0};
    void *seq = tail_make(tail_model);
    if (!seq || !tail_run(tail_model, seq, cur + (size_t)(N - 1) * W, last_snaps, MAX_SNAPSHOTS, 1, "", &token, &word)) bail("tail failed");
    const double g = clk();

    printf("one-time setup %.2f s (expert buffer %.2f, layers 0-1 + buffers %.2f, opening layers 2-92 %.2f, tail %.2f)\n", setup,
        t_buffer, t_edges - t_buffer, t_layers - t_edges, setup - t_layers);
    printf("REQUEST, words -> next token: %lu words = %u tokens (%ld bytes): %.1f ms = %.3f s in total\n",
        word_count, N, words_size, 1000 * (g - a), g - a);
    double kda = 0, mla = 0, kmin = 1e9, kmax = 0, mmin = 1e9, mmax = 0; unsigned kn = 0, mn = 0;
    for (unsigned l = 2; l <= LAYERS; l++) {
        if (is_mla(l)) { mla += layer_time[l]; mn++; if (layer_time[l] < mmin) mmin = layer_time[l]; if (layer_time[l] > mmax) mmax = layer_time[l]; }
        else { kda += layer_time[l]; kn++; if (layer_time[l] < kmin) kmin = layer_time[l]; if (layer_time[l] > kmax) kmax = layer_time[l]; }
        if (verbose) printf("   layer %2u %s %7.1f ms\n", l, is_mla(l) ? "MLA" : "KDA", 1000 * layer_time[l]);
    }
    printf("   tokenise %.2f ms, embedding lookup %.2f ms, layer 0 %.1f ms, layer 1 %.1f ms\n", 1000 * (b - a), 1000 * (c - b), 1000 * (d - c), 1000 * (e - d));
    printf("   layers 2-92: %.1f ms in total (%u KDA layers %.1f ms, %.1f-%.1f each; %u MLA layers %.1f ms, %.1f-%.1f each)\n",
        1000 * (f - e), kn, 1000 * kda, 1000 * kmin, 1000 * kmax, mn, 1000 * mla, 1000 * mmin, 1000 * mmax);
    printf("   tail (last position: final normalisation, head, text) %.1f ms\n", 1000 * (g - f));
    printf("   next token id %u, text \"%.*s\"\n", token, (int)word.length, (const char *)word.bytes);
    if (compare) {
        unsigned snap_ok = 0;
        for (unsigned s = 0; s < MAX_SNAPSHOTS; s++) snap_ok += same_rows(snaps[s], ref_snaps + (size_t)s * N * W, N);
        printf("   against the pod's own chain: embeddings %u of %u rows, layer 92 output %u of %u rows, 8 snapshots %u of %u rows bit-identical\n",
            same_rows(s0, ref_embed, N), N, same_rows(cur, ref_resid, N), N, snap_ok, N * MAX_SNAPSHOTS);
    } else
        printf("   not compared: no recorded values exist for these words\n");

    if (getenv("PF_CMP_ROWS") && *getenv("PF_CMP_ROWS")) {
        long l; float *rows = (float *)slurp(getenv("PF_CMP_ROWS"), &l);
        if (!extra_n || (size_t)l < (size_t)extra_n * W * 4) bail("PF_CMP_ROWS needs PF_EXTRA and as many rows as extra ids");
        printf("   layer 92 output at positions %u..%u of this prompt pass against the decode steps' rows: %u of %u bit-identical\n",
            N0, N - 1, same_rows(cur + (size_t)N0 * W, rows, extra_n), extra_n);
    }

    /* ---- PF_SWEEP: the continuation, k tokens per pass ---- */
    if (sweep_k) {
        if (!getenv("PF_SWEEP_IDS") || !*getenv("PF_SWEEP_IDS") || !sweep_n) bail("PF_SWEEP needs PF_SWEEP_IDS and PF_SWEEP_N");
        long l; char *list = slurp(getenv("PF_SWEEP_IDS"), &l);
        uint32_t *cont = malloc((sweep_n + 2) * sizeof *cont); unsigned cont_n = 0;
        for (char *tok = strtok(list, ", \n\r"); tok && cont_n < sweep_n + 1; tok = strtok(NULL, ", \n\r")) cont[cont_n++] = (uint32_t)strtoul(tok, NULL, 10);
        if (cont_n < sweep_k + 1) bail("PF_SWEEP_IDS has too few ids");
        if (cont[0] != token) { fprintf(stderr, "pipeline-full: the first id of PF_SWEEP_IDS (%u) is not the prompt's next token (%u)\n", cont[0], token); return 1; }
        enum { KMAX = 16 };
        static float sb[3][KMAX * W], snapbuf[MAX_SNAPSHOTS][KMAX * W], embk[KMAX * W], tail_snaps[MAX_SNAPSHOTS * W];
        void *dseq = tail_make(tail_model);
        if (!dseq) bail("tail sequence failed");
        const unsigned long faults0 = vmstat_major_faults();
        unsigned long h3, m3, hk, mk, hm, mm;
        stage3_cache_stats(&h3, &m3); lbk_cache_stats(&hk, &mk); lbm_cache_stats(&hm, &mm);
        const unsigned long hit0 = h3 + hk + hm, miss0 = m3 + mk + mm;
        double total = 0, layers_total = 0, tail_total = 0, distinct_sum = 0, late = 0; unsigned sweeps = 0, late_n = 0, right = 0, tested = 0;
        for (unsigned i = 0; i + sweep_k + 1 <= cont_n && i + sweep_k <= sweep_n; i += sweep_k) {
            const double s_begin = clk();
            const uint32_t *wid = cont + i;
            stage1_lookup((const unsigned *)wid, sweep_k, embk);
            stage2_stepk(wid, embk, sweep_k, sb[0]);
            stage3_stepk(sb[0], embk, sweep_k, sb[1]);
            unsigned ci = 1, kk = 1;
            const float *sn[MAX_SNAPSHOTS] = { embk };
            unsigned long distinct = 0;
            for (unsigned l = 2; l <= LAYERS; l++) {
                float *out = sb[(ci + 1) % 3];
                if (is_mla(l)) { lbm_stepk(layer[l], sb[ci], sn, sweep_k, N + i, out); distinct += lbm_distinct(); }
                else { lbk_stepk(layer[l], sb[ci], sn, sweep_k, N + i, out); distinct += lbk_distinct(); }
                if (l % 12 == 0) { memcpy(snapbuf[kk], sb[ci], (size_t)sweep_k * W * sizeof(float)); sn[kk] = snapbuf[kk]; kk++; }
                ci = (ci + 1) % 3;
            }
            const double s_layers = clk();
            for (unsigned j = 0; j < sweep_k; j++) {
                for (unsigned s = 0; s < MAX_SNAPSHOTS; s++) memcpy(tail_snaps + (size_t)s * W, sn[s] + (size_t)j * W, W * sizeof(float));
                uint32_t next = 0; Text piece = {0, 0};
                if (kk != MAX_SNAPSHOTS || !tail_run(tail_model, dseq, sb[ci] + (size_t)j * W, tail_snaps, MAX_SNAPSHOTS, 0, "", &next, &piece)) bail("sweep step failed");
                tested++; right += next == cont[i + j + 1];
            }
            const double s_end = clk();
            total += s_end - s_begin; layers_total += s_layers - s_begin; tail_total += s_end - s_layers;
            distinct_sum += (double)distinct / (LAYERS - 1);
            sweeps++;
            if (sweeps > 3) { late += s_end - s_begin; late_n++; }
            if (verbose) printf("   sweep %3u (tokens %u..%u): %.3f s, layers %.3f s, tail %.3f s, mean distinct experts per layer %.1f\n", sweeps, i, i + sweep_k - 1,
                s_end - s_begin, s_layers - s_begin, s_end - s_layers, (double)distinct / (LAYERS - 1));
        }
        stage3_cache_stats(&h3, &m3); lbk_cache_stats(&hk, &mk); lbm_cache_stats(&hm, &mm);
        const unsigned long hit = h3 + hk + hm - hit0, miss = m3 + mk + mm - miss0;
        printf("SWEEP, %u tokens per pass, %u passes after the %u-token prompt: %.3f s in total, %.3f s per pass (layers %.3f, tail %.3f), %.3f s per token; "
               "passes after the first three: %.3f s per pass\n", sweep_k, sweeps, N, total, sweeps ? total / sweeps : 0.0,
               sweeps ? layers_total / sweeps : 0.0, sweeps ? tail_total / sweeps : 0.0, sweeps ? total / sweeps / sweep_k : 0.0, late_n ? late / late_n : 0.0);
        printf("   distinct experts per layer per pass %.1f; expert cache: %lu of %lu needed were in RAM (%.1f%%); major page faults %lu\n",
            sweeps ? distinct_sum / sweeps : 0.0, hit, hit + miss, hit + miss ? 100.0 * hit / (hit + miss) : 0.0, vmstat_major_faults() - faults0);
        printf("   the next token at each of the %u positions equals the recorded continuation: %u of %u\n", tested, right, tested);
    }

    /* ---- more tokens: each one goes through the layers alone, on the state the prompt left ---- */
    if (decode && !sweep_k) {
        static float bufs[3][W], snapbuf[MAX_SNAPSHOTS][W], emb1[W], tail_snaps[MAX_SNAPSHOTS * W];
        float *saved = malloc((size_t)decode * W * sizeof *saved);
        uint32_t *gen = malloc((decode + 1) * sizeof *gen);
        static char text[1 << 16];
        size_t text_len = 0;
        gen[0] = token;
        if (word.length < sizeof text) { memcpy(text, word.bytes, word.length); text_len = word.length; }
        void *dseq = tail_make(tail_model);
        if (!dseq) bail("tail sequence failed");
        double layers_s = 0, tail_s = 0, first_step = 0;
        unsigned steps = 0;
        if (chat.on) chat_feed(token, &word);
        if (routes_on) { stage3_route_log(1); lbk_route_log(1); lbm_route_log(1); }
        const unsigned long faults0 = vmstat_major_faults();
        double laps0[3][24], laps1[3][24];
        stage3_laps(laps0[0]); lbk_laps(laps0[1]); lbm_laps(laps0[2]);
        const double d0 = clk();
        SpeculativeStats speculation = {0};
        Model model = { .tail_model = tail_model, .tail_sequence = dseq, .tail_run = tail_run };
        memcpy(model.layer, layer, sizeof model.layer);
        if (max_draft) steps = speculative_decode(&model, (const uint32_t *)ids, N, gen, decode, max_draft, text, &text_len, sizeof text, &speculation);
        else
        for (unsigned i = 0; i < decode && !chat.done; i++) {
            const double s_begin = clk();
            const uint32_t id = gen[i];
            stage1_lookup(&id, 1, emb1);
            stage2_step(&id, emb1, bufs[0]);
            stage3_step(bufs[0], emb1, bufs[1]);
            if (routes_on) stage3_routes(route_rec + ((size_t)i * LAYERS + 0) * 48);
            unsigned ci = 1, kk = 1;
            const float *sn[MAX_SNAPSHOTS] = { emb1 };
            for (unsigned l = 2; l <= LAYERS; l++) {
                float *out = bufs[(ci + 1) % 3];
                if (is_mla(l)) lbm_step(layer[l], bufs[ci], sn, N + i, out); else lbk_step(layer[l], bufs[ci], sn, N + i, out);
                if (routes_on) (is_mla(l) ? lbm_routes : lbk_routes)(route_rec + ((size_t)i * LAYERS + (l - 1)) * 48);
                if (l % 12 == 0) { memcpy(snapbuf[kk], bufs[ci], W * sizeof(float)); sn[kk] = snapbuf[kk]; kk++; }
                ci = (ci + 1) % 3;
            }
            const double s_layers = clk();
            memcpy(saved + (size_t)i * W, bufs[ci], W * sizeof(float));
            for (unsigned s = 0; s < MAX_SNAPSHOTS; s++) memcpy(tail_snaps + (size_t)s * W, sn[s], W * sizeof(float));
            uint32_t next = 0; Text piece = {0, 0};
            if (kk != MAX_SNAPSHOTS || !tail_run(tail_model, dseq, bufs[ci], tail_snaps, MAX_SNAPSHOTS, 0, "", &next, &piece)) bail("decode step failed");
            gen[i + 1] = next;
            if (chat.on) chat_feed(next, &piece);
            steps = i + 1;
            if (text_len + piece.length < sizeof text) { memcpy(text + text_len, piece.bytes, piece.length); text_len += piece.length; }
            const double s_end = clk();
            layers_s += s_layers - s_begin; tail_s += s_end - s_layers;
            if (i == 0) first_step = s_end - s_begin;
            if (verbose) printf("   step %3u: token %u in %.0f ms\n", i + 1, next, 1000 * (s_end - s_begin));
        }
        const double dec = clk() - d0;
        printf("DECODE, %u more tokens after the %u-token prompt: %.3f s in total (first step %.3f s, mean %.3f s per token; layers %.3f s, tail %.3f s)\n",
            steps, N, dec, first_step, steps ? dec / steps : 0.0, layers_s, tail_s);
        printf("   major page faults during the decode steps: %lu\n", vmstat_major_faults() - faults0);
        if (max_draft)
            printf("   SPECULATIVE DECODING, up to %u draft tokens: %u new tokens in %u passes (%u with a draft, %u of them rejected and rolled back); "
                   "%u draft tokens proposed, %u kept (%.1f%%); %.3f s per new token\n", max_draft, steps, speculation.passes, speculation.with_draft,
                   speculation.rollbacks, speculation.drafted, speculation.accepted,
                   speculation.drafted ? 100.0 * speculation.accepted / speculation.drafted : 0.0, steps ? dec / steps : 0.0);
        if (max_draft && speculation.draft_seconds > 0) printf("   DSpark draft model: %.3f s in total (%.3f s per pass), %s\n",
            speculation.draft_seconds, speculation.draft_seconds / speculation.passes, fixed_draft_width ? "fixed width" : "draft width chosen per pass from the share of draft tokens that were right");
        if (!max_draft) {   /* seconds per decode token, summed over layers: stages before the router (0-10), latent projection (11), experts (12), the rest (13-18), early guess (19) */
            stage3_laps(laps1[0]); lbk_laps(laps1[1]); lbm_laps(laps1[2]);
            double part[5] = {0};
            for (int o = 0; o < 3; o++) for (int id = 0; id < 20; id++) {
                const double v = laps1[o][id] - laps0[o][id];
                part[id <= 10 ? 0 : id == 11 ? 1 : id == 12 ? 2 : id <= 18 ? 3 : 4] += v;
            }
            if (steps) printf("   per decode token (s): before the router %.3f, latent %.3f, experts %.3f, after %.3f, readers start + guess %.3f\n",
                part[0] / steps, part[1] / steps, part[2] / steps, part[3] / steps, part[4] / steps);
        }
        if (cache_cap) {
            unsigned long h3, m3, hk, mk, hm, mm;
            stage3_cache_stats(&h3, &m3); lbk_cache_stats(&hk, &mk); lbm_cache_stats(&hm, &mm);
            const unsigned long hit = h3 + hk + hm, miss = m3 + mk + mm;
            printf("   expert cache, %u experts per layer (%.1f GB): %lu of %lu experts needed were already in RAM (%.1f%%), %lu read from disk (%.1f GB)\n",
                cache_cap, (double)LAYERS * cache_cap * EXPERT_BYTES / 1e9, hit, hit + miss, hit + miss ? 100.0 * hit / (hit + miss) : 0.0,
                miss, (double)miss * EXPERT_BYTES / 1e9);
            unsigned long gi3, gu3, gik, guk, gim, gum;
            stage3_guess_stats(&gi3, &gu3); lbk_guess_stats(&gik, &guk); lbm_guess_stats(&gim, &gum);
            if (gi3 + gik + gim)
                printf("   early reads from the guess: %lu started, %lu of them were needed (%.1f%%)\n", gi3 + gik + gim, gu3 + guk + gum,
                    100.0 * (gu3 + guk + gum) / (gi3 + gik + gim));
        }
        printf("   generated ids:");
        for (unsigned i = 0; i <= steps; i++) printf(" %u", gen[i]);
        printf("\n   generated text (%u tokens):\n-----\n%.*s\n-----\n", steps + 1, (int)text_len, text);
        if (chat.on) {
            printf("CHAT (%s%s), %s\n", chat.think ? "thinking effort " : "no thinking", chat.think ? chat.effort : "",
                chat.done ? "the reply ended" : "stopped at the token limit before the reply ended");
            if (chat.think) printf("   thinking:\n-----\n%.*s\n-----\n", (int)chat.think_len, chat.think_text);
            printf("   response:\n-----\n%.*s\n-----\n", (int)chat.response_len, chat.response);
        }
        if (getenv("PF_SAVE_ROWS") && *getenv("PF_SAVE_ROWS")) {
            FILE *rf = fopen(getenv("PF_SAVE_ROWS"), "wb");
            if (!rf || fwrite(saved, W * sizeof(float), steps, rf) != steps) bail("cannot save the rows");
            fclose(rf);
            printf("   layer 92 output rows of the %u decode steps saved\n", steps);
        }
        if (routes_on) {
            FILE *rf = fopen(getenv("PF_ROUTES"), "wb");
            const unsigned header[2] = { steps, LAYERS };
            const size_t rn = (size_t)steps * LAYERS * 48;
            if (!rf || fwrite(header, sizeof header, 1, rf) != 1 || fwrite(hits, sizeof *hits, (size_t)LAYERS * 896, rf) != (size_t)LAYERS * 896 ||
                fwrite(route_rec, sizeof *route_rec, rn, rf) != rn) bail("cannot save the routes");
            fclose(rf);
        }
        if (getenv("PF_SAVE_IDS") && *getenv("PF_SAVE_IDS")) {
            FILE *rf = fopen(getenv("PF_SAVE_IDS"), "w");
            if (!rf) bail("cannot save the ids");
            for (unsigned i = 0; i <= steps; i++) fprintf(rf, "%u%c", gen[i], i == steps ? '\n' : ',');
            fclose(rf);
        }
    }
    return 0;
}
