/* The dense trunk projections (QKV, attention output, routing gate, routed output and all
   three shared-expert matrices) ran on one core while the OpenMP team waited at the
   barrier. Rows are independent and each row's reduction order is untouched. */
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';

const BEFORE = `    const unsigned char *ids = record->data + (size_t)record->rows * 4;
    for (unsigned row = 0; row < record->rows; row++)`;
const AFTER = `    const unsigned char *ids = record->data + (size_t)record->rows * 4;
#pragma omp parallel for schedule(static)
    for (unsigned row = 0; row < record->rows; row++)`;

let changed = 0, already = 0;
for (let layer = 2; layer <= 92; layer++) {
    const path = new URL(`transformer-${layer}/transformer-${layer}.c`, import.meta.url);
    let source = readFileSync(path, 'utf8');
    if (source.includes('#pragma omp parallel for schedule(static)\n    for (unsigned row = 0; row < record->rows; row++)')) { already++; continue; }
    assert.equal(source.split(BEFORE).length - 1, 1, `layer ${layer}`);
    writeFileSync(path, source.replace(BEFORE, AFTER));
    changed++;
}
console.log(`PASS: ${changed} pods parallelise the dense projections, ${already} already done`);
