/* Record pointers are non-const, so the mapping must be too. */
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';

const edits = [
    ['    const unsigned char *group_map[3];', '    unsigned char *group_map[3];'],
    ['transformer->group_map[group] = transformer_map(path, (size_t)lengths[group]);',
     'transformer->group_map[group] = (unsigned char *)transformer_map(path, (size_t)lengths[group]);'],
];

let changed = 0, already = 0;
for (let layer = 1; layer <= 92; layer++) {
    const path = new URL(`transformer-${layer}/transformer-${layer}.c`, import.meta.url);
    let source = readFileSync(path, 'utf8');
    if (source.includes('unsigned char *group_map[3];') && !source.includes('const unsigned char *group_map[3];')) { already++; continue; }
    for (const [before, after] of edits) {
        assert.equal(source.split(before).length, 2, `layer ${layer}: ${before.slice(0, 40)}`);
        source = source.replace(before, after);
    }
    writeFileSync(path, source);
    changed++;
}
console.log(`PASS: ${changed} pods corrected, ${already} already correct`);
