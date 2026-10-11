/* Hardmax made the fold hand the whole weight to its winning source. It defaulted to
   layer 46 with both folds on, so one layer silently disagreed with the numerical
   reference and the full campaign could not run: `all` aborted at layer 46 on the route
   assertion, which meant no regression anywhere else could be seen either.

   Measured before removing it, layer 46 at 256 positions with experts pinned, four runs
   each: hardmax 132.21 ms/position mean against softmax 134.22, but softmax's best run
   was 132.145, below hardmax's mean. At most about 1.5% on one layer of 93.

   transformer_listed stays: transformer_fold_residual and transformer_dropped use it. */
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';

const COMMENT_BEFORE = `/* The winning source takes the whole weight instead of its softmax share.

   A layer runs 108 stages and only two of them fold: stage 3 before attention
   (record 37) and stage 21 before the MLP (record 38). CLOVER_HARDMAX_STAGES picks
   one or both and CLOVER_HARDMAX_LAYERS picks the owners, but the safe set is not
   the same at the two folds. Measured as cos(aggregate, residual), stage 3 clears
   0.9 at sixteen owners and stage 21 at twenty-five, and they overlap only in part.
   CLOVER_HARDMAX_LAYERS_3 and CLOVER_HARDMAX_LAYERS_21 override the owner list for
   one fold, each falling back to CLOVER_HARDMAX_LAYERS. CLOVER_SOFTMAX=1 disables
   all of it.

   Defaults stay at layer 46 with both folds, the configuration measured good. */
static int transformer_listed(const char *list, int value)`;

const COMMENT_AFTER = `/* A comma list of numbers and low-high ranges, or "all". */
static int transformer_listed(const char *list, int value)`;

/* Layer 1 is derived separately and folds two sources, so it carries its own wording. */
const COMMENT_BEFORE_ONE = `/* Layer 1 folds two sources, not nine, but it answers to the same switches as
   layers 2-92 or "the model" means two different things in one run. Record 37 is
   the stage 3 fold, record 38 the stage 21 fold. The default list excludes layer 1,
   which is what the measured-good distributed run actually did. */
static int transformer_listed(const char *list, int value)`;

const FUNCTION = `
static int transformer_hardmax(unsigned fold)
{
    static int decided, on, stage3, stage21, layer3, layer21;
    if (!decided) {
        const char *off = getenv("CLOVER_SOFTMAX");
        const char *layers = getenv("CLOVER_HARDMAX_LAYERS");
        const char *stages = getenv("CLOVER_HARDMAX_STAGES");
        const char *at3 = getenv("CLOVER_HARDMAX_LAYERS_3");
        const char *at21 = getenv("CLOVER_HARDMAX_LAYERS_21");
        on = !(off && *off && *off != '0');
        if (!layers || !*layers) layers = "46";
        stages = stages && *stages ? stages : "3,21";
        stage3 = transformer_listed(stages, 3);
        stage21 = transformer_listed(stages, 21);
        layer3 = transformer_listed(at3 && *at3 ? at3 : layers, TRANSFORMER_LAYER);
        layer21 = transformer_listed(at21 && *at21 ? at21 : layers, TRANSFORMER_LAYER);
        decided = 1;
    }
    return on && (fold == 37 ? (stage3 && layer3) : (stage21 && layer21));
}
`;

const CALL_BEFORE = `    if (transformer_hardmax(fold)) {
        unsigned best = 0;
        for (unsigned source = 1; source < count; source++) if (weights[source] > weights[best]) best = source;
        for (unsigned source = 0; source < count; source++) weights[source] = source == best ? 1.0f : 0.0f;
    }
`;

const CALL_BEFORE_ONE = `    if (transformer_hardmax(fold)) {
        unsigned best = weights[1] > weights[0] ? 1 : 0;
        weights[0] = best == 0 ? 1.0f : 0.0f;
        weights[1] = best == 1 ? 1.0f : 0.0f;
    }
`;

let changed = 0, already = 0, absent = 0;
for (let layer = 1; layer <= 92; layer++) {
    const path = new URL(`transformer-${layer}/transformer-${layer}.c`, import.meta.url);
    let source = readFileSync(path, 'utf8');
    if (!source.includes('transformer_hardmax')) { already++; continue; }
    const comment = source.includes(COMMENT_BEFORE) ? COMMENT_BEFORE
        : source.includes(COMMENT_BEFORE_ONE) ? COMMENT_BEFORE_ONE : null;
    const call = source.includes(CALL_BEFORE) ? CALL_BEFORE
        : source.includes(CALL_BEFORE_ONE) ? CALL_BEFORE_ONE : null;
    if (!comment || !call) { absent++; continue; }
    for (const [before, after] of [[comment, COMMENT_AFTER], [FUNCTION, ''], [call, '']]) {
        assert.equal(source.split(before).length - 1, 1, `layer ${layer}: anchor not unique`);
        source = source.replace(before, after);
    }
    assert(!source.includes('transformer_hardmax'), `layer ${layer}: reference survived`);
    assert(source.includes('transformer_listed('), `layer ${layer}: helper lost`);
    writeFileSync(path, source);
    changed++;
}
console.log(`PASS: hardmax removed from ${changed} pods, ${already} already without it, ${absent} unmatched`);
