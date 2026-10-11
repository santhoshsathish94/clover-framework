/* The MLA cache stored K, V and the positional tail interleaved per position, so each
   of the 96 heads walked the sequence at a 98,560 byte stride and the bus delivered
   whole cache lines to use 512 bytes of each. Capacity becomes the per-head stride, so
   a head's positions are contiguous, and growing the cache relocates every block. The
   head loop also had no pragma: 96 independent heads on one core.

   Arithmetic and its order are untouched, so every pod must stay bit-identical.
   The block is present in every pod but only compiled where TRANSFORMER_MLA is 1. */
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';

const MARK = 'Head-major, so capacity is the per-head stride';

const edits = [
[`        if (capacity > SIZE_MAX / stride / sizeof(float)) return 0;
        float *cache = realloc(sequence->mla_cache, capacity * stride * sizeof(float));
        if (!cache) return 0;
        sequence->mla_cache = cache;
        sequence->mla_capacity = capacity;
    }`,
`        if (capacity > SIZE_MAX / stride / sizeof(float)) return 0;
        float *cache = realloc(sequence->mla_cache, capacity * stride * sizeof(float));
        if (!cache) return 0;
        /* Head-major, so capacity is the per-head stride and growing relocates every block.
           Highest offset first: each block only moves up, so nothing unread is overwritten. */
        if (sequence->mla_capacity && sequence->positions) {
            const size_t old = sequence->mla_capacity, used = sequence->positions;
            memmove(cache + 2 * (size_t)TRANSFORMER_ROWS * capacity,
                cache + 2 * (size_t)TRANSFORMER_ROWS * old, used * 64 * sizeof(float));
            for (unsigned head = TRANSFORMER_HEADS; head-- > 0; )
                memmove(cache + ((size_t)TRANSFORMER_ROWS + head * 128) * capacity,
                    cache + ((size_t)TRANSFORMER_ROWS + head * 128) * old, used * 128 * sizeof(float));
            for (unsigned head = TRANSFORMER_HEADS; head-- > 0; )
                memmove(cache + (size_t)head * 128 * capacity,
                    cache + (size_t)head * 128 * old, used * 128 * sizeof(float));
        }
        sequence->mla_cache = cache;
        sequence->mla_capacity = capacity;
    }`],

[`    float *scores = malloc(count * 2 * sizeof(float));
    if (!scores) return 0;
    float *exponentials = scores + count;
    transformer_project(transformer, 20, sequence->normalized, sequence->mla_query_latent);`,
`    const int threads = omp_get_max_threads();
    if ((size_t)threads > SIZE_MAX / count / (2 * sizeof(float))) return 0;
    float *scratch = malloc((size_t)threads * count * 2 * sizeof(float));
    if (!scratch) return 0;
    transformer_project(transformer, 20, sequence->normalized, sequence->mla_query_latent);`],

[`    float *current = sequence->mla_cache + sequence->positions * stride;
    for (unsigned head = 0; head < TRANSFORMER_HEADS; head++) {
        memcpy(current + head * 128, sequence->mla_expanded + head * 256, 128 * sizeof(float));
        memcpy(current + TRANSFORMER_ROWS + head * 128, sequence->mla_expanded + head * 256 + 128, 128 * sizeof(float));
    }
    memcpy(current + 2 * TRANSFORMER_ROWS, sequence->mla_kv_latent + 512, 64 * sizeof(float));`,
`    float *const cache = sequence->mla_cache;
    const size_t capacity = sequence->mla_capacity, slot = sequence->positions;
    for (unsigned head = 0; head < TRANSFORMER_HEADS; head++) {
        memcpy(cache + (size_t)head * 128 * capacity + slot * 128,
            sequence->mla_expanded + head * 256, 128 * sizeof(float));
        memcpy(cache + ((size_t)TRANSFORMER_ROWS + head * 128) * capacity + slot * 128,
            sequence->mla_expanded + head * 256 + 128, 128 * sizeof(float));
    }
    memcpy(cache + 2 * (size_t)TRANSFORMER_ROWS * capacity + slot * 64,
        sequence->mla_kv_latent + 512, 64 * sizeof(float));`],

[`    for (unsigned head = 0; head < TRANSFORMER_HEADS; head++) {
        const float *query = sequence->mla_query + head * 192;
        for (size_t position = 0; position < count; position++) {
            const float *cached = sequence->mla_cache + position * stride;
            double score = 0.0;
            for (unsigned coordinate = 0; coordinate < 128; coordinate++)
                score += (double)query[coordinate] * (double)cached[head * 128 + coordinate];
            for (unsigned coordinate = 0; coordinate < 64; coordinate++)
                score += (double)query[128 + coordinate] * (double)cached[2 * TRANSFORMER_ROWS + coordinate];
            scores[position] = (float)score * scale;
        }`,
`    const float *const rope = cache + 2 * (size_t)TRANSFORMER_ROWS * capacity;
    /* Heads share only the query and the cache, both read-only, so the walk spreads
       across cores instead of leaving fifteen of sixteen idle for its whole length. */
#pragma omp parallel for schedule(static)
    for (int head = 0; head < TRANSFORMER_HEADS; head++) {
        float *scores = scratch + (size_t)omp_get_thread_num() * count * 2;
        float *exponentials = scores + count;
        const float *query = sequence->mla_query + head * 192;
        const float *const keys = cache + (size_t)head * 128 * capacity;
        for (size_t position = 0; position < count; position++) {
            const float *cached = keys + position * 128;
            const float *positional = rope + position * 64;
            double score = 0.0;
            for (unsigned coordinate = 0; coordinate < 128; coordinate++)
                score += (double)query[coordinate] * (double)cached[coordinate];
            for (unsigned coordinate = 0; coordinate < 64; coordinate++)
                score += (double)query[128 + coordinate] * (double)positional[coordinate];
            scores[position] = (float)score * scale;
        }`],

[`        float *output = sequence->attention + head * 128;
        memset(output, 0, 128 * sizeof(float));
        for (size_t position = 0; position < count; position++) {
            float weight = (float)((double)exponentials[position] / total);
            const float *value = sequence->mla_cache + position * stride + TRANSFORMER_ROWS + head * 128;`,
`        float *output = sequence->attention + head * 128;
        const float *const values = cache + ((size_t)TRANSFORMER_ROWS + head * 128) * capacity;
        memset(output, 0, 128 * sizeof(float));
        for (size_t position = 0; position < count; position++) {
            float weight = (float)((double)exponentials[position] / total);
            const float *value = values + position * 128;`],

[`    free(scores);
    transformer_project(transformer, 6, sequence->normalized, sequence->gate);`,
`    free(scratch);
    transformer_project(transformer, 6, sequence->normalized, sequence->gate);`],
];

let changed = 0, already = 0, absent = 0;
for (let layer = 2; layer <= 92; layer++) {
    const path = new URL(`transformer-${layer}/transformer-${layer}.c`, import.meta.url);
    let source = readFileSync(path, 'utf8');
    if (source.includes(MARK)) { already++; continue; }
    if (!source.includes(edits[0][0])) { absent++; continue; }
    for (const [before, after] of edits) {
        assert.equal(source.split(before).length - 1, 1, `layer ${layer}: anchor not unique`);
        source = source.replace(before, after);
    }
    writeFileSync(path, source);
    changed++;
}
console.log(`PASS: ${changed} pods store the MLA cache head-major and walk heads in parallel, ` +
    `${already} already done, ${absent} without the block`);
