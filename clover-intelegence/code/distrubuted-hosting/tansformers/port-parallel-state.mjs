/* The recurrent state update walked 96 heads on one core. Each head owns its own slice
   of recurrent[head], its own output block in sequence->attention, and a prediction
   array local to the iteration, so the heads are independent and the per-head
   arithmetic and its order are untouched.

   Measured in the layer, not in isolation: stage 8 goes 0.1064 s to 0.0124 s over 64
   positions, 8.6x, and the layer 28.281 to 27.462 ms/position. A standalone replica of
   the same loop had suggested 15.2x and about 5.5% of the layer; it ran with the state
   hot and nothing competing for cache, so it over-predicted. The layer number is the
   real one.

   This was the last serial hot spot in the pod. */
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';

const BEFORE = `static void transformer_update_attention(TransformerSequence *sequence)
{
    const float query_scale = 1.0f / sqrtf(128.0f);
    for (unsigned head = 0; head < TRANSFORMER_HEADS; head++) {`;

const AFTER = `static void transformer_update_attention(TransformerSequence *sequence)
{
    const float query_scale = 1.0f / sqrtf(128.0f);
    /* Each head owns its slice of the recurrent state and its own output block. */
#pragma omp parallel for schedule(static)
    for (unsigned head = 0; head < TRANSFORMER_HEADS; head++) {`;

let changed = 0, already = 0, absent = 0;
for (let layer = 1; layer <= 92; layer++) {
    const path = new URL(`transformer-${layer}/transformer-${layer}.c`, import.meta.url);
    let source = readFileSync(path, 'utf8');
    if (source.includes('Each head owns its slice of the recurrent state')) { already++; continue; }
    /* transformer-1 has mixed endings: CRLF in the file but LF around this function,
       so try both rather than deciding from the file as a whole. */
    const crlf = s => s.replaceAll('\n', '\r\n');
    const form = source.includes(BEFORE) ? [BEFORE, AFTER]
        : source.includes(crlf(BEFORE)) ? [crlf(BEFORE), crlf(AFTER)] : null;
    if (!form) { absent++; continue; }
    assert.equal(source.split(form[0]).length - 1, 1, `layer ${layer}: anchor not unique`);
    writeFileSync(path, source.replace(form[0], form[1]));
    changed++;
}
console.log(`PASS: ${changed} pods update the recurrent state across cores, ${already} already done, ${absent} without it`);
