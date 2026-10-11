/* Applies the stage machine proven on layer 46 to every other layer. The converted
   region is copied from layer 46 verbatim rather than regenerated, so all 92 layers run
   the same code; only the TRANSFORMER_LAYER and TRANSFORMER_MLA macros differ. */
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';

const here = new URL('./', import.meta.url);
const template = readFileSync(new URL('transformer-46/transformer-46.c', here), 'utf8');

const MARK = '/* Every step of a position is numbered.';
const START = 'int transformer_process(Transformer *transformer, TransformerSequence *sequence,';
const END = '    return 1;\n}\n\nint transformer_routes(';

const block = template.slice(template.indexOf(MARK), template.indexOf(END) + '    return 1;\n}\n'.length);
assert(block.includes('transformer_step'), 'template missing the stage machine');
assert(block.includes('stage == 120'), 'template missing the final stage');

const OLD_TAIL = `    sequence->positions++;
    return 1;
}`;

/* Once a layer is staged the block ends at the process loop instead. Both endings are
   recognised so the template can be re-applied, not only applied once. */
const STAGED_TAIL = `    return 1;
}`;

let changed = 0, already = 0, skipped = [];
for (let layer = 2; layer <= 92; layer++) {
    if (layer === 46) { already++; continue; }
    const path = new URL(`transformer-${layer}/transformer-${layer}.c`, here);
    let source = readFileSync(path, 'utf8');
    const staged = source.includes('transformer_step');

    if (!source.includes('#include "live-root.h"')) { skipped.push(layer); continue; }
    if (!staged && !source.includes('    size_t positions;\n    int failed;')) { skipped.push(layer); continue; }
    if (!source.includes('#include "stage.h"'))
        source = source.replace('#include "live-root.h"', '#include "live-root.h"\n#include "stage.h"');
    if (!source.includes('#include "trace.h"'))
        source = source.replace('#include "stage.h"', '#include "stage.h"\n#include "trace.h"');
    if (!source.includes('    int stage;'))
        source = source.replace('    size_t positions;\n    int failed;', '    size_t positions;\n    int stage;\n    int failed;');
    // The aggregate weights live outside the copied block, so they are patched in here.
    if (!source.includes('fold_weights'))
        source = source.replace('    size_t positions;\n    int stage;\n    int failed;',
            '    size_t positions;\n    int stage;\n    float fold_weights[9];\n    unsigned fold_count;\n    int failed;');
    const WEIGHTS = '    for (unsigned source = 0; source < count; source++) weights[source] = (float)((double)exponentials[source] / total);';
    const HARDMAX = `/* The winning source takes the whole weight instead of its softmax share.

   A layer runs 108 stages and only two of them fold: stage 3 before attention
   (record 37) and stage 21 before the MLP (record 38). Addressing a layer moves
   both, which is why they were never told apart. CLOVER_HARDMAX_STAGES picks one
   or both; CLOVER_HARDMAX_LAYERS picks the owners; CLOVER_SOFTMAX=1 disables it.

   Both default to the configuration that was actually measured good, layer 46 with
   both folds. That default is an accident of a guard bug being preserved on purpose
   until a sweep replaces it, not a design choice. */
static int transformer_listed(const char *list, int value)
{
    if (!strcmp(list, "all")) return 1;
    while (*list) {
        char *after;
        long low = strtol(list, &after, 10), high = low;
        if (after == list) return 0;
        if (*after == '-') { list = after + 1; high = strtol(list, &after, 10); }
        if (value >= low && value <= high) return 1;
        list = *after == ',' ? after + 1 : after;
    }
    return 0;
}

static int transformer_hardmax(unsigned fold)
{
    static int decided, layer_ok, stage3, stage21;
    if (!decided) {
        const char *off = getenv("CLOVER_SOFTMAX");
        const char *layers = getenv("CLOVER_HARDMAX_LAYERS");
        const char *stages = getenv("CLOVER_HARDMAX_STAGES");
        int on = !(off && *off && *off != '0');
        layer_ok = on && transformer_listed(layers && *layers ? layers : "46", TRANSFORMER_LAYER);
        stages = stages && *stages ? stages : "3,21";
        stage3 = transformer_listed(stages, 3);
        stage21 = transformer_listed(stages, 21);
        decided = 1;
    }
    return layer_ok && (fold == 37 ? stage3 : stage21);
}

static void transformer_aggregate(`;
    if (!source.includes('transformer_hardmax(unsigned fold)'))
        source = source.replace(
            /\/\* The winning source takes the whole weight[\s\S]*?static int transformer_hardmax\([^)]*\)\n\{[\s\S]*?\n\}\n\n/, '');
    if (!source.includes('static int transformer_hardmax(unsigned fold)'))
        source = source.replace('static void transformer_aggregate(', HARDMAX);
    /* Each switch gets its own guard, keyed on its own name. Sharing a guard means a
       newly added switch is silently skipped in every layer that already carried the
       older one, and the layer then fails to compile. */
    const FOLD = `/* Across five prompts the fold picked the residual in 146 of the 172 calls whose
   winner never moved, and when the residual wins at weight 1 the whole aggregate
   step is a copy. CLOVER_FOLD=residual asserts that everywhere and skips the
   scoring. That is a claim about the model, not an optimisation, so it has to be
   judged by reading what comes out. */
static int transformer_fold_residual(void)
{
    static int decided, value;
    if (!decided) {
        const char *setting = getenv("CLOVER_FOLD");
        value = setting && !strcmp(setting, "residual");
        decided = 1;
    }
    return value;
}

static void transformer_aggregate(`;
    if (!source.includes('transformer_fold_residual(void)'))
        source = source.replace('static void transformer_aggregate(', FOLD);
    const SOURCES = '    sources[sequence->snapshot_count] = sequence->residual;';
    if (!source.includes('transformer_fold_residual()) {'))
        source = source.replace(SOURCES, SOURCES +
            '\n    if (transformer_fold_residual()) {' +
            '\n        memcpy(sequence->aggregate, sequence->residual, TRANSFORMER_WIDTH * sizeof(float));' +
            '\n        sequence->fold_weights[0] = 1.0f;' +
            '\n        sequence->fold_count = 1;' +
            '\n        (void)fold;' +
            '\n        return;' +
            '\n    }');
    const LOAD = '    sequence->snapshot_count = snapshot_count;';
    if (!source.includes('clover_freeze_apply'))
        source = source.replace(LOAD, LOAD + '\n    clover_freeze_apply(TRANSFORMER_LAYER, &sequence->snapshots[0][0], snapshot_count);');
    const RECORD = '    /* Kept only so the blend can be observed; nothing downstream reads these. */' +
        '\n    memcpy(sequence->fold_weights, weights, sizeof weights);' +
        '\n    sequence->fold_count = count;';
    if (!source.includes('memcpy(sequence->fold_weights'))
        source = source.replace(WEIGHTS, WEIGHTS + '\n' + RECORD);
    /* Strip every application first, in either calling form, then add exactly one.
       Guarding "insert if absent" let an earlier rename of the call site hide the
       block from its own guard, and 90 layers ended up with two copies. */
    source = source.replace(
        /    if \(transformer_hardmax\((?:fold)?\)\) \{\n        unsigned best = 0;\n[^\n]*\n[^\n]*\n    \}\n/g, '');
    source = source.replace(RECORD,
        '    if (transformer_hardmax(fold)) {' +
        '\n        unsigned best = 0;' +
        '\n        for (unsigned source = 1; source < count; source++) if (weights[source] > weights[best]) best = source;' +
        '\n        for (unsigned source = 0; source < count; source++) weights[source] = source == best ? 1.0f : 0.0f;' +
        '\n    }\n' + RECORD);

    const start = source.indexOf(staged ? MARK : START);
    const tailMark = staged ? STAGED_TAIL : OLD_TAIL;
    const tail = source.indexOf(tailMark, source.indexOf(START, start));
    if (start < 0 || tail < 0) { skipped.push(layer); continue; }
    const finish = tail + tailMark.length;
    source = source.slice(0, start) + block.trimEnd() + source.slice(finish);

    writeFileSync(path, source);
    changed++;
}
if (skipped.length) console.log(`anchors not found in layers: ${skipped.join(',')}`);
console.log(`PASS: ${changed} layers converted to the stage machine, ${already} already staged`);
