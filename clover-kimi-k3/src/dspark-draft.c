#define _GNU_SOURCE
/* The Kimi K3 DSpark draft model: guesses the next 8 tokens in one pass, so the target model can check them together.

   What it is (config.json of Inferact/Kimi-K3-DSpark, vLLM: models/kimi_k3/nvidia/dspark_mla.py and v1/worker/gpu/spec_decode/dspark/speculator.py):
     - 5 dense MLA layers (hidden 7168, 64 heads, q_lora 1536, kv_lora 512, nope 128, rope 64, value 128, MLP 14336), RMSNorm eps 1e-5,
       YaRN rotary (theta 50000, factor 32, original length 32768, interleaved pairs on the 64 rope dims).
     - Context: the target model's hidden states after its layers 2, 23, 47, 71 and 89, joined (5 x 7168), projected to 7168 and normalised;
       each draft layer turns every context position into one latent key/value (kv_a_proj, kv_a_layernorm, rotary on the rope part).
     - Query block: 8 tokens at the positions after the context: the last real token (the anchor), then 7 mask tokens. All 8 attend to the
       context and to each other (not causal). Each query position predicts the token after it, so the block gives 8 draft tokens.
     - Output: final RMSNorm, then the TARGET's vocabulary head gives the base logits of each position; a Markov head adds, at each position, a
       bias from the previously drafted token (markov_w2 . markov_w1[previous], rank 256); the draft token is the largest logit.
   Attention is computed in the latent space (kv_b_proj absorbed into the query and the output), as vLLM does.
   Weights stay bf16 in the memory-mapped checkpoint; activations are float. A wrong draft only lowers the acceptance rate: the target verifies. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <float.h>
#include <fenv.h>
#include <immintrin.h>
#include <omp.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include "numeric-table.h"

enum { WIDTH = 7168, LAYERS = 5, HEADS = 64, Q_LORA = 1536, KV_LORA = 512, ROPE = 64, NOPE = 128, VALUE = 128, INTERMEDIATE = 14336,
       QUERIES = 8, TARGETS = 5, MARKOV_RANK = 256, VOCAB = 163840, MASK_TOKEN = 163837 };
static const float NORM_EPS = 1e-5f;

typedef struct { const uint16_t *data; unsigned rows, cols; } Bf16Matrix;
typedef struct {
    Bf16Matrix q_a, q_b, kv_a, kv_b, o, gate, up, down;
    const uint16_t *input_norm, *post_attention_norm, *q_a_norm, *kv_a_norm;
    float *latent, *rope_key;   /* context + block keys of this layer, per position: 512 latent floats, 64 rotated rope floats */
} DraftLayer;

static struct {
    int ready;
    unsigned max_positions, context_length;
    DraftLayer layer[LAYERS];
    Bf16Matrix context_proj, embedding, markov_w1, markov_w2, head;
    const uint16_t *context_norm, *final_norm;
    float inverse_frequency[ROPE / 2];
    float attention_scale;
    float *logits;   /* QUERIES x VOCAB base logits */
} D;

static void bail(const char *what) { fprintf(stderr, "dspark: %s\n", what); exit(1); }
static float bf16(uint16_t bits) { uint32_t b = (uint32_t)bits << 16; float f; memcpy(&f, &b, 4); return f; }
static __m512 bf16_16(const uint16_t *p) { return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)p)), 16)); }

/* ---- the checkpoint: one safetensors file, a JSON header, then the raw tensors ---- */
static const char *g_header; static const unsigned char *g_data;
static Bf16Matrix tensor(const char *name)
{
    char key[200]; snprintf(key, sizeof key, "\"%s\":{", name);
    const char *at = strstr(g_header, key);
    if (!at) { fprintf(stderr, "dspark: tensor %s missing\n", name); exit(1); }
    const char *shape = strstr(at, "\"shape\":["), *offsets = strstr(at, "\"data_offsets\":[");
    unsigned rows = 0, cols = 1; unsigned long begin = 0, end = 0;
    if (!shape || !offsets) bail("bad header");
    if (sscanf(shape + 9, "%u,%u", &rows, &cols) < 1) bail("bad shape");
    if (sscanf(offsets + 16, "%lu,%lu", &begin, &end) != 2) bail("bad offsets");
    return (Bf16Matrix){ (const uint16_t *)(g_data + begin), rows, cols };
}
static const uint16_t *vector_of(const char *name) { return tensor(name).data; }

/* Y[t][row] = sum over columns of X[t][column] * W[row][column]; X is n rows of `cols` floats. Eight tokens share every weight row read. */
static void project(const Bf16Matrix *m, const float *X, unsigned n, float *Y)
{
    const unsigned cols = m->cols;
    for (unsigned first = 0; first < n; first += 8) {
        const unsigned count = n - first < 8 ? n - first : 8;
#pragma omp parallel for schedule(static)
        for (unsigned row = 0; row < m->rows; row++) {
            const uint16_t *w = m->data + (size_t)row * cols;
            __m512 sum[8];
            for (int t = 0; t < 8; t++) sum[t] = _mm512_setzero_ps();
            for (unsigned c = 0; c < cols; c += 16) {
                const __m512 weights = bf16_16(w + c);
                _Pragma("GCC unroll 8") for (int t = 0; t < 8; t++)
                    if ((unsigned)t < count) sum[t] = _mm512_fmadd_ps(weights, _mm512_loadu_ps(X + (size_t)(first + t) * cols + c), sum[t]);
            }
            for (unsigned t = 0; t < count; t++) Y[(size_t)(first + t) * m->rows + row] = _mm512_reduce_add_ps(sum[t]);
        }
    }
}

static void rms_norm(float *x, const uint16_t *weight, unsigned n)
{
    double sum = 0;
    for (unsigned i = 0; i < n; i++) sum += (double)x[i] * x[i];
    const float scale = 1.0f / sqrtf((float)(sum / n) + NORM_EPS);
    for (unsigned i = 0; i < n; i++) x[i] = x[i] * scale * bf16(weight[i]);
}

/* Rotary on 64 dims, interleaved pairs (x[2i], x[2i+1]), YaRN frequencies. */
static void rotate(float *x, size_t position)
{
    for (unsigned i = 0; i < ROPE / 2; i++) {
        const float angle = (float)position * D.inverse_frequency[i], c = cosf(angle), s = sinf(angle);
        const float a = x[2 * i], b = x[2 * i + 1];
        x[2 * i] = a * c - b * s; x[2 * i + 1] = a * s + b * c;
    }
}

static void compute_frequencies(void)
{
    const double base = 50000.0, factor = 32.0, original = 32768.0, two_pi = 6.283185307179586;
    const int dim = ROPE;
    double low = floor(dim * log(original / (32.0 * two_pi)) / (2 * log(base)));
    double high = ceil(dim * log(original / (1.0 * two_pi)) / (2 * log(base)));
    if (low < 0) low = 0;
    if (high > dim - 1) high = dim - 1;
    if (low == high) high += 0.001;
    for (int i = 0; i < dim / 2; i++) {
        const double pos_freq = pow(base, (double)(2 * i) / dim), extrapolation = 1.0 / pos_freq, interpolation = 1.0 / (factor * pos_freq);
        double ramp = (i - low) / (high - low); ramp = ramp < 0 ? 0 : ramp > 1 ? 1 : ramp;
        const double mask = 1.0 - ramp;
        D.inverse_frequency[i] = (float)(interpolation * (1.0 - mask) + extrapolation * mask);
    }
    const double mscale = 0.1 * 1.0 * log(factor) + 1.0;
    D.attention_scale = (float)(pow(NOPE + ROPE, -0.5) * mscale * mscale);
}

/* checkpoint: the safetensors file; head_table: the target's vocabulary head (the pod's fruit.bin); max_positions: the longest sequence */
int dspark_open(const char *checkpoint, const char *head_table, unsigned max_positions)
{
    int fd = open(checkpoint, O_RDONLY);
    struct stat st;
    if (fd < 0 || fstat(fd, &st) != 0) { fprintf(stderr, "dspark: cannot open %s\n", checkpoint); return 0; }
    const unsigned char *file = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (file == MAP_FAILED) return 0;
    uint64_t header_size; memcpy(&header_size, file, 8);
    char *header = malloc(header_size + 1); memcpy(header, file + 8, header_size); header[header_size] = 0;
    g_header = header; g_data = file + 8 + header_size;
    D.max_positions = max_positions;
    char name[160];
    for (int l = 0; l < LAYERS; l++) {
        DraftLayer *L = &D.layer[l];
#define T(field, suffix) snprintf(name, sizeof name, "layers.%d." suffix, l); L->field = tensor(name)
#define V(field, suffix) snprintf(name, sizeof name, "layers.%d." suffix, l); L->field = vector_of(name)
        T(q_a, "self_attn.q_a_proj.weight"); T(q_b, "self_attn.q_b_proj.weight"); T(kv_a, "self_attn.kv_a_proj_with_mqa.weight");
        T(kv_b, "self_attn.kv_b_proj.weight"); T(o, "self_attn.o_proj.weight");
        T(gate, "mlp.gate_proj.weight"); T(up, "mlp.up_proj.weight"); T(down, "mlp.down_proj.weight");
        V(input_norm, "input_layernorm.weight"); V(post_attention_norm, "post_attention_layernorm.weight");
        V(q_a_norm, "self_attn.q_a_layernorm.weight"); V(kv_a_norm, "self_attn.kv_a_layernorm.weight");
#undef T
#undef V
        L->latent = calloc((size_t)(max_positions + QUERIES) * KV_LORA, sizeof(float));
        L->rope_key = calloc((size_t)(max_positions + QUERIES) * ROPE, sizeof(float));
        if (!L->latent || !L->rope_key) bail("allocation failed");
    }
    D.context_proj = tensor("context_proj.weight"); D.embedding = tensor("embed_tokens.weight");
    D.markov_w1 = tensor("markov_head.markov_w1.weight"); D.markov_w2 = tensor("markov_head.markov_w2.weight");
    D.context_norm = vector_of("context_norm.weight"); D.final_norm = vector_of("final_norm.weight");
    ServerOutput *output = server_output_open(head_table);
    if (!output || !output->table->direct) { fputs("dspark: cannot open the vocabulary head\n", stderr); return 0; }
    D.head = (Bf16Matrix){ (const uint16_t *)output->table->direct, VOCAB, WIDTH };
    D.logits = malloc((size_t)QUERIES * VOCAB * sizeof(float));
    if (!D.logits) bail("allocation failed");
    compute_frequencies();
    D.ready = 1;
    return 1;
}

/* The target's sequence starts over (a chat that was rebuilt from a summary): the draft forgets the context it has read. */
void dspark_reset(void) { D.context_length = 0; }

/* Context: the target's hidden states of positions [D.context_length, end) after the 5 chosen layers; aux[a] + position * 7168 is the row of layer a. */
static void extend_context(const float *const aux[TARGETS], unsigned end)
{
    static float joined[8 * TARGETS * WIDTH], state[8 * WIDTH], kv[8 * (KV_LORA + ROPE)];
    while (D.context_length < end) {
        const unsigned first = D.context_length, count = end - first < 8 ? end - first : 8;
        for (unsigned t = 0; t < count; t++)
            for (int a = 0; a < TARGETS; a++) memcpy(joined + ((size_t)t * TARGETS + a) * WIDTH, aux[a] + (size_t)(first + t) * WIDTH, WIDTH * sizeof(float));
        project(&D.context_proj, joined, count, state);
        for (unsigned t = 0; t < count; t++) rms_norm(state + (size_t)t * WIDTH, D.context_norm, WIDTH);
        for (int l = 0; l < LAYERS; l++) {
            DraftLayer *L = &D.layer[l];
            project(&L->kv_a, state, count, kv);
            for (unsigned t = 0; t < count; t++) {
                float *latent = L->latent + (size_t)(first + t) * KV_LORA, *rope = L->rope_key + (size_t)(first + t) * ROPE;
                memcpy(latent, kv + (size_t)t * (KV_LORA + ROPE), KV_LORA * sizeof(float));
                rms_norm(latent, L->kv_a_norm, KV_LORA);
                memcpy(rope, kv + (size_t)t * (KV_LORA + ROPE) + KV_LORA, ROPE * sizeof(float));
                rotate(rope, first + t);
            }
        }
        D.context_length += count;
    }
}

/* One head of attention for the query block: all QUERIES queries against `keys` positions (context and block). */
static void attend_head(const DraftLayer *L, unsigned head, const float *q, unsigned keys, float *vcat)
{
    const unsigned qwidth = NOPE + ROPE;
    float *scores = malloc((size_t)QUERIES * keys * sizeof(float));
    float latent_query[QUERIES][KV_LORA], latent_out[QUERIES][KV_LORA], row[KV_LORA];
    memset(latent_query, 0, sizeof latent_query);
    const uint16_t *kv_b = L->kv_b.data + (size_t)head * (NOPE + VALUE) * KV_LORA;
    for (unsigned d = 0; d < NOPE; d++) {   /* query into latent space: sum over d of q_nope[d] * W_UK[d][:] */
        for (unsigned c = 0; c < KV_LORA; c++) row[c] = bf16(kv_b[(size_t)d * KV_LORA + c]);
        for (unsigned t = 0; t < QUERIES; t++) {
            const float weight = q[(size_t)t * HEADS * qwidth + head * qwidth + d];
            for (unsigned c = 0; c < KV_LORA; c++) latent_query[t][c] += weight * row[c];
        }
    }
    for (unsigned t = 0; t < QUERIES; t++) {
        const float *query_rope = q + (size_t)t * HEADS * qwidth + head * qwidth + NOPE;
        float *s = scores + (size_t)t * keys, maximum = -INFINITY;
        for (unsigned j = 0; j < keys; j++) {
            const float *latent = L->latent + (size_t)j * KV_LORA, *rope = L->rope_key + (size_t)j * ROPE;
            float dot = 0;
            for (unsigned c = 0; c < KV_LORA; c++) dot += latent_query[t][c] * latent[c];
            for (unsigned c = 0; c < ROPE; c++) dot += query_rope[c] * rope[c];
            s[j] = dot * D.attention_scale;
            if (s[j] > maximum) maximum = s[j];
        }
        float total = 0;
        for (unsigned j = 0; j < keys; j++) { s[j] = expf(s[j] - maximum); total += s[j]; }
        memset(latent_out[t], 0, sizeof latent_out[t]);
        for (unsigned j = 0; j < keys; j++) {
            const float p = s[j] / total, *latent = L->latent + (size_t)j * KV_LORA;
            for (unsigned c = 0; c < KV_LORA; c++) latent_out[t][c] += p * latent[c];
        }
    }
    for (unsigned d = 0; d < VALUE; d++) {   /* back to values: sum over c of W_UV[d][c] * latent_out[c] */
        for (unsigned c = 0; c < KV_LORA; c++) row[c] = bf16(kv_b[(size_t)(NOPE + d) * KV_LORA + c]);
        for (unsigned t = 0; t < QUERIES; t++) {
            float sum = 0;
            for (unsigned c = 0; c < KV_LORA; c++) sum += row[c] * latent_out[t][c];
            vcat[(size_t)t * HEADS * VALUE + head * VALUE + d] = sum;
        }
    }
    free(scores);
}

/* Drafts QUERIES tokens after `anchor`, the last real token, which sits at position `anchor_position` (the target has processed positions below it).
   aux[a] holds the target's hidden states after layer target_layer_ids[a], one row of 7168 floats per position. */
void dspark_draft(const float *const aux[TARGETS], uint32_t anchor, unsigned anchor_position, uint32_t *drafts)
{
    if (!D.ready) bail("not opened");
    if (anchor_position + QUERIES > D.max_positions + QUERIES) bail("sequence longer than the draft was opened for");
    extend_context(aux, anchor_position);
    static float residual[QUERIES * WIDTH], hidden[QUERIES * WIDTH], q_a[QUERIES * Q_LORA], q[QUERIES * HEADS * (NOPE + ROPE)], kv[QUERIES * (KV_LORA + ROPE)],
        vcat[QUERIES * HEADS * VALUE], attn[QUERIES * WIDTH], gate[QUERIES * INTERMEDIATE], up[QUERIES * INTERMEDIATE], mlp[QUERIES * WIDTH];
    uint32_t ids[QUERIES];
    ids[0] = anchor;
    for (unsigned t = 1; t < QUERIES; t++) ids[t] = MASK_TOKEN;
    for (unsigned t = 0; t < QUERIES; t++)
        for (unsigned c = 0; c < WIDTH; c++) residual[(size_t)t * WIDTH + c] = bf16(D.embedding.data[(size_t)ids[t] * WIDTH + c]);
    for (int l = 0; l < LAYERS; l++) {
        DraftLayer *L = &D.layer[l];
        if (l > 0) for (size_t i = 0; i < (size_t)QUERIES * WIDTH; i++) residual[i] += mlp[i];
        memcpy(hidden, residual, sizeof hidden);
        for (unsigned t = 0; t < QUERIES; t++) rms_norm(hidden + (size_t)t * WIDTH, L->input_norm, WIDTH);
        project(&L->q_a, hidden, QUERIES, q_a);
        for (unsigned t = 0; t < QUERIES; t++) rms_norm(q_a + (size_t)t * Q_LORA, L->q_a_norm, Q_LORA);
        project(&L->q_b, q_a, QUERIES, q);
        project(&L->kv_a, hidden, QUERIES, kv);
        for (unsigned t = 0; t < QUERIES; t++) {
            const size_t position = anchor_position + t;
            float *latent = L->latent + position * KV_LORA, *rope = L->rope_key + position * ROPE;
            memcpy(latent, kv + (size_t)t * (KV_LORA + ROPE), KV_LORA * sizeof(float));
            rms_norm(latent, L->kv_a_norm, KV_LORA);
            memcpy(rope, kv + (size_t)t * (KV_LORA + ROPE) + KV_LORA, ROPE * sizeof(float));
            rotate(rope, position);
            for (unsigned h = 0; h < HEADS; h++) rotate(q + (size_t)t * HEADS * (NOPE + ROPE) + h * (NOPE + ROPE) + NOPE, position);
        }
        const unsigned keys = anchor_position + QUERIES;
#pragma omp parallel for schedule(dynamic, 1)
        for (unsigned h = 0; h < HEADS; h++) attend_head(L, h, q, keys, vcat);
        project(&L->o, vcat, QUERIES, attn);
        for (size_t i = 0; i < (size_t)QUERIES * WIDTH; i++) residual[i] += attn[i];
        memcpy(hidden, residual, sizeof hidden);
        for (unsigned t = 0; t < QUERIES; t++) rms_norm(hidden + (size_t)t * WIDTH, L->post_attention_norm, WIDTH);
        project(&L->gate, hidden, QUERIES, gate);
        project(&L->up, hidden, QUERIES, up);
        for (size_t i = 0; i < (size_t)QUERIES * INTERMEDIATE; i++) gate[i] = gate[i] / (1.0f + expf(-gate[i])) * up[i];
        project(&L->down, gate, QUERIES, mlp);
    }
    for (size_t i = 0; i < (size_t)QUERIES * WIDTH; i++) residual[i] += mlp[i];
    for (unsigned t = 0; t < QUERIES; t++) rms_norm(residual + (size_t)t * WIDTH, D.final_norm, WIDTH);
    project(&D.head, residual, QUERIES, D.logits);
    /* the Markov head: each position's logits get a bias from the token drafted just before it */
    static float markov_embed[MARKOV_RANK], bias[VOCAB];
    uint32_t previous = anchor;
    for (unsigned t = 0; t < QUERIES; t++) {
        for (unsigned k = 0; k < MARKOV_RANK; k++) markov_embed[k] = bf16(D.markov_w1.data[(size_t)previous * MARKOV_RANK + k]);
        project(&D.markov_w2, markov_embed, 1, bias);
        const float *base = D.logits + (size_t)t * VOCAB;
        uint32_t best = 0; float best_value = -INFINITY;
        for (uint32_t v = 0; v < VOCAB; v++) { const float value = base[v] + bias[v]; if (value > best_value) { best_value = value; best = v; } }
        drafts[t] = best; previous = best;
    }
}
