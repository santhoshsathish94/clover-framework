/* Points every layer pod at the shared direct-storage reader. Transformer-1 is produced
   by its own generator and is skipped here. Asserts one occurrence of each edit per pod. */
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';

const INCLUDE_BEFORE = '#include "root.h"';
const INCLUDE_AFTER = '#include "live-root.h"';
const OPEN_BEFORE = 'root_open(root_directory)';

let changed = 0, already = 0;
for (let layer = 2; layer <= 92; layer++) {
    const path = new URL(`transformer-${layer}/transformer-${layer}.c`, import.meta.url);
    const source = readFileSync(path, 'utf8');
    if (source.includes(INCLUDE_AFTER)) { already++; continue; }
    assert.equal(source.split(INCLUDE_BEFORE).length, 2, `layer ${layer}: include`);
    assert.equal(source.split(OPEN_BEFORE).length, 2, `layer ${layer}: root_open`);
    const updated = source
        .replace(INCLUDE_BEFORE, INCLUDE_AFTER)
        .replace(OPEN_BEFORE, `root_open(root_directory, ${layer})`);
    assert.equal(updated.split('root_open(root_directory, ').length, 2, `layer ${layer}: rewrite`);
    writeFileSync(path, updated);
    changed++;
}
console.log(`PASS: ${changed} pods pointed at the shared reader, ${already} already done`);
