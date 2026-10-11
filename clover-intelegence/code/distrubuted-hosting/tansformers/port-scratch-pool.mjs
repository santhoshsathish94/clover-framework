/* The reader indexes the scratch pool by OpenMP thread number, so a single embedded
   struct is only safe at one thread. This gives each pod a pool the size of the team,
   matching clover-one.c. */
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';

const edits = [
    ['    RootScratch root_scratch;', '    RootScratch *root_scratch;', 1],
    [`    if (sequence) sequence->owner = transformer;
    return sequence;`,
     `    if (!sequence) return NULL;
    /* One scratch per thread: the reader indexes this pool by OpenMP thread number. */
    sequence->root_scratch = calloc((size_t)omp_get_max_threads(), sizeof *sequence->root_scratch);
    if (!sequence->root_scratch) { free(sequence); return NULL; }
    sequence->owner = transformer;
    return sequence;`, 1],
    ['    const Transformer *owner = sequence->owner;',
     '    const Transformer *owner = sequence->owner;\n    RootScratch *scratch = sequence->root_scratch;', 1],
    ['    memset(sequence, 0, sizeof *sequence);\n    sequence->owner = owner;',
     '    memset(sequence, 0, sizeof *sequence);\n    sequence->owner = owner;\n    sequence->root_scratch = scratch;', 1],
    ['&sequence->root_scratch', 'sequence->root_scratch', 3],
];

const CLOSE = /void transformer_sequence_close\(TransformerSequence \*sequence\)\n\{\n/;

let changed = 0, skipped = 0;
for (let layer = 2; layer <= 92; layer++) {
    const path = new URL(`transformer-${layer}/transformer-${layer}.c`, import.meta.url);
    let source = readFileSync(path, 'utf8');
    if (source.includes('RootScratch *root_scratch;')) { skipped++; continue; }
    for (const [from, to, count] of edits) {
        assert.equal(source.split(from).length - 1, count, `layer ${layer}: "${from.slice(0, 40)}"`);
        source = source.split(from).join(to);
    }
    assert.match(source, CLOSE, `layer ${layer}: close`);
    source = source.replace(CLOSE, (m) => m + '    if (sequence) free(sequence->root_scratch);\n');
    writeFileSync(path, source);
    changed++;
}
console.log(`PASS: ${changed} pods size the scratch pool to the thread team, ${skipped} already done`);
