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

     whole-model POD_DIR DATASET_DIR TOKFILES WORDS REFDIR

   REFDIR holds recorded values for the ids WORDS stands for (embed.f32, final.resid.f32, final.snaps.f32: the pod's
   own chain, layer by layer, ending after layer 92) or is "-" when none exist; then nothing is compared.        */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <dlfcn.h>
#include <sys/mman.h>

double stage1_setup(const char *tokfiles, const char *seed);
int stage1_ids(const char *text, long size, int *ids, int cap);
int stage1_ids_plain(const char *text, long size, int *ids, int cap);
int stage1_id_of(const char *name);
void stage1_lookup(const unsigned *ids, unsigned n, float *out);
double stage2_setup(const char *code_dir, unsigned max_n);
void stage2_run(const uint32_t *ids, const float *emb, unsigned n, unsigned window, float *out);
void stage2_step(const uint32_t *id, const float *emb, float *out);
void stage2_stepk(const uint32_t *ids, const float *emb, unsigned n, float *out);
void stage3_stepk(const float *resid, const float *s0, unsigned n, float *out);
void stage3_set_scratch(unsigned char *const *slots, unsigned n);
void stage3_use_buffer(unsigned char *buffer);
double stage3_setup(const char *dataset, unsigned max_n);
void stage3_run(const float *resid, const float *s0, unsigned n, unsigned window, float *out);
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
    keep_positions(m, max_draft);
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
        if (j < drafted && !chat.done) {   /* positions after j were fed on a wrong guess: go back to the state after token j */
            use_position(m, j);
            stats->rollbacks++;
        }
    }
    free(seen);
    return produced - 1;
}

int main(int argc, char **argv)
{
    if (argc != 6) { fputs("usage: pipeline-full CODE_DIR DATASET TOKFILES WORDS REFDIR\n", stderr); return 2; }
    const char *code = argv[1], *dataset = argv[2];
    const int compare = strcmp(argv[5], "-") != 0;
    const int verbose = getenv("PF_VERBOSE") && *getenv("PF_VERBOSE");
    char seed[4096], dataset1[4096], path[4096];
    snprintf(seed, sizeof seed, "%s/server/bin/dataset/inputs/seed.bin", code);
    snprintf(dataset1, sizeof dataset1, "%s/tansformers/transformer-1/bin/dataset", code);

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
    const unsigned N0 = (unsigned)found, N = N0 + extra_n;
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
    stage2_setup(code, N + 16);
    stage3_use_buffer(experts);
    stage3_setup(dataset1, N + 16);
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
        layer[l] = is_mla(l) ? lbm_open(dataset, l, snapshots_in(l), N + decode + sweep_n + 16) : lbk_open(dataset, l, snapshots_in(l), N + decode + sweep_n + 16);
        if (cache_cap) { FILL_SLOTS(l - 1) (is_mla(l) ? lbm_set_slots : lbk_set_slots)(layer[l], slot_ptrs, cache_cap); }
    }
    if (getenv("SPECULATIVE_DRAFTER") && !strcmp(getenv("SPECULATIVE_DRAFTER"), "dspark")) {   /* the DSpark draft model instead of the prompt-lookup draft */
        const char *checkpoint = getenv("DSPARK_CHECKPOINT");
        if (!max_draft) bail("SPECULATIVE_DRAFTER=dspark needs SPECULATIVE_DRAFT_TOKENS (up to 8)");
        if (!checkpoint || !*checkpoint) bail("DSPARK_CHECKPOINT is not set");
        snprintf(path, sizeof path, "%s/server/bin/dataset/outputs/fruit.bin", code);
        for (unsigned a = 0; a < 5; a++) dspark_hidden[a] = calloc((size_t)(N + decode + 16) * W, sizeof(float));
        if (!dspark_open(checkpoint, path, N + decode + 16)) bail("cannot open the DSpark draft model");
    }
    const double t_layers = clk() - setup0;

    char leaves[4096], head[4096], tik[4096], vocab[4096], config[4096];
    snprintf(path, sizeof path, "%s/tansformers/transformer-93/bin/pipeline-stage.so", code);
    void *tail = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!tail) { fprintf(stderr, "tail dlopen: %s\n", dlerror()); return 1; }
    int (*tail_open)(const char *, const char *, const char *, const char *, const char *, void **) = dlsym(tail, "transformer_93_open");
    void *(*tail_make)(const void *) = dlsym(tail, "transformer_93_sequence_create");
    void (*tail_drop)(void *) = dlsym(tail, "transformer_93_sequence_close");
    int (*tail_run)(const void *, void *, const float *, const float *, unsigned, int, const char *, uint32_t *, Text *) = dlsym(tail, "transformer_93_process");
    if (!tail_open || !tail_make || !tail_drop || !tail_run) bail("tail symbols missing");
    snprintf(leaves, sizeof leaves, "%s/server/bin/dataset/leaves.json", code);
    snprintf(head, sizeof head, "%s/server/bin/dataset/outputs/fruit.bin", code);
    snprintf(tik, sizeof tik, "%s/server/bin/dataset/tiktoken.model", code);
    snprintf(vocab, sizeof vocab, "%s/server/bin/dataset/vocabulary.bin", code);
    snprintf(config, sizeof config, "%s/server/bin/configs/tokenizer_config.json", code);
    void *tail_model = NULL;
    if (!tail_open(leaves, head, tik, vocab, config, &tail_model)) bail("tail open failed");
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
