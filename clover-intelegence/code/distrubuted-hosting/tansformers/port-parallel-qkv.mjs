/* transformer_qkv is the one trunk projection that does not go through
   transformer_project, because KDA's Q, K and V weights are interleaved per row with a
   20-byte prefix carrying the row scale and its convolution taps. port-parallel-project
   added the pragma to transformer_project and this loop was missed, so 3 x 12288 row
   dot-products of width 7168 ran on one core while fifteen sat idle.

   Measured on layer 46, 64 positions, experts pinned, with my own per-stage clocks:
   stage 6 was 6.869 s of the layer's 8.438 s, 81.4%. With the pragma the layer goes
   131.877 to 31.633 ms/position, 4.17x, and the checksum is unchanged at -466.877678.
   That puts a KDA layer level with an MLA layer, which costs 31.5.

   Rows are independent: each writes its own sequence->qkv[component][row] and advances
   its own sequence->history[component][row]. Per-row arithmetic and its order are
   untouched, so every pod must stay bit-identical.

   Anchored on the function signature, not the loop: transformer-1 contains the same two
   loop lines again inside its qkv validation, which returns early and must stay serial. */
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';

const BEFORE = `static void transformer_qkv(const Transformer *transformer, TransformerSequence *sequence)
{
    for (unsigned row = 0; row < TRANSFORMER_ROWS; row++)
        for (unsigned component = 0; component < 3; component++) {`;

const AFTER = `static void transformer_qkv(const Transformer *transformer, TransformerSequence *sequence)
{
    /* Each row owns its output and its own convolution history, so rows are independent. */
#pragma omp parallel for schedule(static)
    for (unsigned row = 0; row < TRANSFORMER_ROWS; row++)
        for (unsigned component = 0; component < 3; component++) {`;

let changed = 0, already = 0, absent = 0;
for (let layer = 1; layer <= 92; layer++) {
    const path = new URL(`transformer-${layer}/transformer-${layer}.c`, import.meta.url);
    let source = readFileSync(path, 'utf8');
    if (source.includes(AFTER)) { already++; continue; }
    if (!source.includes(BEFORE)) { absent++; continue; }
    assert.equal(source.split(BEFORE).length - 1, 1, `layer ${layer}: anchor not unique`);
    writeFileSync(path, source.replace(BEFORE, AFTER));
    changed++;
}
console.log(`PASS: ${changed} pods project Q, K and V across cores, ${already} already done, ${absent} without it`);
