/* Issues the reads for all sixteen experts as soon as routing picks them, so the device
   has many requests in flight instead of one demand fault at a time. */
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';

const BEFORE = `    memset(sequence->mixture, 0, sizeof sequence->mixture);
    for (unsigned rank = 0; rank < 16; rank++) {
        if (!transformer_expert(transformer, sequence, sequence->selected[rank])) return 0;`;
const AFTER = `    memset(sequence->mixture, 0, sizeof sequence->mixture);
    for (unsigned rank = 0; rank < 16; rank++) root_prefetch(transformer->root, sequence->selected[rank]);
    for (unsigned rank = 0; rank < 16; rank++) {
        if (!transformer_expert(transformer, sequence, sequence->selected[rank])) return 0;`;

let changed = 0, already = 0, missing = [];
for (let layer = 1; layer <= 92; layer++) {
    const path = new URL(`transformer-${layer}/transformer-${layer}.c`, import.meta.url);
    let source = readFileSync(path, 'utf8');
    if (source.includes('root_prefetch')) { already++; continue; }
    if (!source.includes(BEFORE)) { missing.push(layer); continue; }
    assert.equal(source.split(BEFORE).length, 2, `layer ${layer}`);
    writeFileSync(path, source.replace(BEFORE, AFTER));
    changed++;
}
if (missing.length) console.log(`anchor not found in layers: ${missing.join(',')}`);
console.log(`PASS: ${changed} pods prefetch their experts, ${already} already done`);
