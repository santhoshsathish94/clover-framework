/* Routing scored 896 experts one after another, each a 7168-long dependent double
   accumulation over full float32 router weights: about 28.7k cycles of latency per
   expert, 896 of them, which is the 4.36 ms/position that per-stage timing showed for
   stage 24. It runs on every layer, MLA and KDA alike.

   A pragma alone would be wrong. Scoring an expert is independent, but choosing the top
   sixteen carries state across experts in best[], selected[] and count, and racing on
   those would corrupt the choice. So the scoring is parallel and the selection stays
   serial, walked in index order so that ties still resolve to the lower expert id.

   Per-expert arithmetic and its order are untouched, so every pod must stay
   bit-identical. The only behavioural difference is on the failure path: a non-finite
   choice now aborts after all scores are computed rather than at the offending expert,
   and the return value is the same. */
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';

const BEFORE = `    float scores[896], best[16];
    unsigned count = 0;
    for (unsigned expert = 0; expert < 896; expert++) {
        const unsigned char *row = transformer->records[29].data + (size_t)expert * TRANSFORMER_WIDTH * 4;
        double score = 0.0;
        for (unsigned coordinate = 0; coordinate < TRANSFORMER_WIDTH; coordinate++)
            score += (double)transformer_f32(row + coordinate * 4) * (double)sequence->postnorm[coordinate];
        scores[expert] = transformer_sigmoid((float)score);
        float choice = scores[expert] + transformer_f32(transformer->records[30].data + expert * 4);
        if (!isfinite(choice)) return 0;
`;

const AFTER = `    float scores[896], choices[896], best[16];
    unsigned count = 0;
    /* Scoring an expert is independent; picking the top sixteen is not. */
#pragma omp parallel for schedule(static)
    for (unsigned expert = 0; expert < 896; expert++) {
        const unsigned char *row = transformer->records[29].data + (size_t)expert * TRANSFORMER_WIDTH * 4;
        double score = 0.0;
        for (unsigned coordinate = 0; coordinate < TRANSFORMER_WIDTH; coordinate++)
            score += (double)transformer_f32(row + coordinate * 4) * (double)sequence->postnorm[coordinate];
        scores[expert] = transformer_sigmoid((float)score);
        choices[expert] = scores[expert] + transformer_f32(transformer->records[30].data + expert * 4);
    }
    /* Index order, so a tie still goes to the lower expert id. */
    for (unsigned expert = 0; expert < 896; expert++) {
        float choice = choices[expert];
        if (!isfinite(choice)) return 0;
`;

let changed = 0, already = 0, absent = 0;
for (let layer = 1; layer <= 92; layer++) {
    const path = new URL(`transformer-${layer}/transformer-${layer}.c`, import.meta.url);
    let source = readFileSync(path, 'utf8');
    if (source.includes('float scores[896], choices[896], best[16];')) { already++; continue; }
    /* transformer-1 carries CRLF in some working trees; match what the file actually has. */
    const crlf = source.includes('\r\n');
    const before = crlf ? BEFORE.replaceAll('\n', '\r\n') : BEFORE;
    const after = crlf ? AFTER.replaceAll('\n', '\r\n') : AFTER;
    if (!source.includes(before)) { absent++; continue; }
    assert.equal(source.split(before).length - 1, 1, `layer ${layer}: anchor not unique`);
    writeFileSync(path, source.replace(before, after));
    changed++;
}
console.log(`PASS: ${changed} pods score the router across cores, ${already} already done, ${absent} without it`);
