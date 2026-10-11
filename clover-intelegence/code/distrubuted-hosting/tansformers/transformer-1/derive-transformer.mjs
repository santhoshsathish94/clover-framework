import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync, existsSync } from 'node:fs';

const original = readFileSync(new URL('../../server/server.c', import.meta.url));
assert.equal(createHash('sha256').update(original).digest('hex'),
  'b39966a7f8b0cb4714f30234762a90859bdf3aeb144e3200c0f080443cc391c3');
let source = original.toString('utf8').replace(/\r\n/g, '\n')
  .replaceAll('SERVER_', 'TRANSFORMER_').replaceAll('Server', 'Transformer')
  .replaceAll('server_', 'transformer_').replaceAll(/\bserver\b/g, 'transformer');
function replace(before, after) {
  assert.equal(source.split(before).length, 2, `anchor must be unique: ${before.slice(0, 80)}`);
  source = source.replace(before, after);
}
function section(start, end, replacement) {
  const begin = source.indexOf(start), finish = source.indexOf(end, begin + start.length);
  assert(begin >= 0 && finish > begin);
  source = source.slice(0, begin) + replacement + source.slice(finish);
}
source = '#define _FILE_OFFSET_BITS 64\n#define _POSIX_C_SOURCE 200809L\n' + source;
replace('#include <string.h>', '#include <string.h>\n#include "live-root.h"\n#include "stage.h"\n#include "trace.h"');
replace('TRANSFORMER_HEADS = 96, TRANSFORMER_DENSE = 33792, TRANSFORMER_RECORDS = 40,',
  'TRANSFORMER_HEADS = 96, TRANSFORMER_LATENT = 3584, TRANSFORMER_EXPERT = 3072,\n    TRANSFORMER_LAYER = 1,\n    TRANSFORMER_SHARED = 6144, TRANSFORMER_RECORDS = 40,');
replace('    size_t group_bytes[3];\n} Transformer;', '    size_t group_bytes[3];\n    Root *root;\n} Transformer;');
replace('    float dense_gate[TRANSFORMER_DENSE], dense_up[TRANSFORMER_DENSE], dense_output[TRANSFORMER_WIDTH];',
`    float incoming[TRANSFORMER_WIDTH], latent[TRANSFORMER_LATENT], mixture[TRANSFORMER_LATENT];
    float expert_gate[TRANSFORMER_EXPERT], expert_up[TRANSFORMER_EXPERT], expert_down[TRANSFORMER_LATENT];
    float latent_norm[TRANSFORMER_LATENT], routed[TRANSFORMER_WIDTH];
    float shared_gate[TRANSFORMER_SHARED], shared_up[TRANSFORMER_SHARED], shared_output[TRANSFORMER_WIDTH];
    unsigned selected[16];
    float weights[16];
    RootScratch *root_scratch;`);
replace(`    if (sequence) sequence->owner = transformer;
    return sequence;`,
`    if (!sequence) return NULL;
    /* One scratch per thread: the reader indexes this pool by OpenMP thread number. */
    sequence->root_scratch = calloc((size_t)omp_get_max_threads(), sizeof *sequence->root_scratch);
    if (!sequence->root_scratch) { free(sequence); return NULL; }
    sequence->owner = transformer;
    return sequence;`);
replace(`    const Transformer *owner = sequence->owner;
    memset(sequence, 0, sizeof *sequence);
    sequence->owner = owner;`,
`    const Transformer *owner = sequence->owner;
    RootScratch *scratch = sequence->root_scratch;
    memset(sequence, 0, sizeof *sequence);
    sequence->owner = owner;
    sequence->root_scratch = scratch;`);
replace(`    const unsigned char *ids = record->data + (size_t)record->rows * 4;
    for (unsigned row = 0; row < record->rows; row++)`,
`    const unsigned char *ids = record->data + (size_t)record->rows * 4;
#pragma omp parallel for schedule(static)
    for (unsigned row = 0; row < record->rows; row++)`);
replace(`void transformer_sequence_close(TransformerSequence *sequence)
{
    free(sequence);`,
`void transformer_sequence_close(TransformerSequence *sequence)
{
    if (sequence) free(sequence->root_scratch);
    free(sequence);`);
replace('    const Transformer *owner;\n    size_t positions;\n    int failed;',
  '    const Transformer *owner;\n    size_t positions;\n    int stage;\n    int failed;');
/* Runs last: the moe splice and the signature rewrite below must land first. */
const stageLayerOne = () => {
replace('int transformer_process(Transformer *transformer, TransformerSequence *sequence,',
`/* One condition per stage, one unit of work, one return. The numbering is the shared
   one, so a caller can tell layer 1 stage 7 from layer 46 stage 7. */
enum {
    TRANSFORMER_STAGE_BASE = CLOVER_STAGE_LAYER_BASE + (TRANSFORMER_LAYER - 1) * CLOVER_STAGE_STRIDE,
    TRANSFORMER_STAGE_EXPERT_FIRST = 30,
    TRANSFORMER_STAGE_EXPERT_LAST = 109
};

int transformer_stage(const TransformerSequence *sequence)
{
    return sequence ? TRANSFORMER_STAGE_BASE + sequence->stage : 0;
}

static int transformer_step(Transformer *transformer, TransformerSequence *sequence,
    const float *input, float output[TRANSFORMER_WIDTH])
{
    const int stage = sequence->stage;

    if (stage == 1) { memcpy(sequence->incoming, input, sizeof sequence->incoming); sequence->stage++; return 1; }
    if (stage == 2) { memcpy(sequence->residual, input, sizeof sequence->residual); sequence->stage++; return 1; }
    if (stage == 3) { transformer_aggregate(transformer, sequence, 37); sequence->stage++; return 1; }
    if (stage == 4) { sequence->stage++; return 1; }
    if (stage == 5) {
        transformer_normalize(sequence->normalized, sequence->aggregate, transformer->qkv + 88, TRANSFORMER_WIDTH);
        sequence->stage++; return 1;
    }
    if (stage == 6) { transformer_qkv(transformer, sequence);              sequence->stage++; return 1; }
    if (stage == 7) { transformer_decay(transformer, sequence);            sequence->stage++; return 1; }
    if (stage == 8) { transformer_update_attention(sequence);              sequence->stage++; return 1; }
    if (stage == 9) { transformer_attention_output(transformer, sequence); sequence->stage = 20; return 1; }
    if (stage == 20) {
        for (unsigned coordinate = 0; coordinate < TRANSFORMER_WIDTH; coordinate++)
            sequence->residual[coordinate] = sequence->incoming[coordinate] + sequence->residual[coordinate];
        sequence->stage++; return 1;
    }
    if (stage == 21) { transformer_aggregate(transformer, sequence, 38); sequence->stage++; return 1; }
    if (stage == 22) {
        transformer_normalize(sequence->postnorm, sequence->aggregate, transformer->records[5].data, TRANSFORMER_WIDTH);
        sequence->stage++; return 1;
    }
    if (stage == 23) { if (!transformer_finite(sequence->postnorm, TRANSFORMER_WIDTH)) return 0; sequence->stage++; return 1; }
    if (stage == 24) { if (!transformer_route(transformer, sequence)) return 0; sequence->stage++; return 1; }
    if (stage == 25) { transformer_project(transformer, 31, sequence->postnorm, sequence->latent); sequence->stage++; return 1; }
    if (stage == 26) { memset(sequence->mixture, 0, sizeof sequence->mixture); sequence->stage++; return 1; }
    if (stage == 27) {
        for (unsigned rank = 0; rank < 16; rank++) root_prefetch(transformer->root, sequence->selected[rank]);
        sequence->stage = TRANSFORMER_STAGE_EXPERT_FIRST; return 1;
    }
    if (stage >= TRANSFORMER_STAGE_EXPERT_FIRST && stage <= TRANSFORMER_STAGE_EXPERT_LAST) {
        const unsigned rank = (unsigned)(stage - TRANSFORMER_STAGE_EXPERT_FIRST) / 5;
        const int step = (stage - TRANSFORMER_STAGE_EXPERT_FIRST) % 5;
        const unsigned expert = sequence->selected[rank];
        if (step == 0) {
            if (!root_project(transformer->root, sequence->root_scratch, expert, 0, sequence->latent, sequence->expert_gate)) return 0;
            sequence->stage++; return 1;
        }
        if (step == 1) {
            if (!root_project(transformer->root, sequence->root_scratch, expert, 1, sequence->latent, sequence->expert_up)) return 0;
            sequence->stage++; return 1;
        }
        if (step == 2) {
            transformer_activation(sequence->expert_gate, sequence->expert_up, TRANSFORMER_EXPERT);
            sequence->stage++; return 1;
        }
        if (step == 3) {
            if (!root_project(transformer->root, sequence->root_scratch, expert, 2, sequence->expert_gate, sequence->expert_down)) return 0;
            sequence->stage++; return 1;
        }
        if (step == 4) {
            for (unsigned coordinate = 0; coordinate < TRANSFORMER_LATENT; coordinate++)
                sequence->mixture[coordinate] = sequence->mixture[coordinate] + sequence->weights[rank] * sequence->expert_down[coordinate];
            sequence->stage++; return 1;
        }
    }
    if (stage == 110) {
        transformer_normalize(sequence->latent_norm, sequence->mixture, transformer->records[33].data, TRANSFORMER_LATENT);
        sequence->stage++; return 1;
    }
    if (stage == 111) { transformer_project(transformer, 32, sequence->latent_norm, sequence->routed); sequence->stage++; return 1; }
    if (stage == 112) { transformer_project(transformer, 34, sequence->postnorm, sequence->shared_gate); sequence->stage++; return 1; }
    if (stage == 113) { transformer_project(transformer, 35, sequence->postnorm, sequence->shared_up); sequence->stage++; return 1; }
    if (stage == 114) { transformer_activation(sequence->shared_gate, sequence->shared_up, TRANSFORMER_SHARED); sequence->stage++; return 1; }
    if (stage == 115) { transformer_project(transformer, 36, sequence->shared_gate, sequence->shared_output); sequence->stage++; return 1; }
    if (stage == 116) {
        for (unsigned coordinate = 0; coordinate < TRANSFORMER_WIDTH; coordinate++)
            sequence->residual[coordinate] = sequence->residual[coordinate] +
                (sequence->routed[coordinate] + sequence->shared_output[coordinate]);
        sequence->stage++; return 1;
    }
    if (stage == 117) {
        if (!transformer_finite(sequence->residual, TRANSFORMER_WIDTH) ||
            !transformer_finite(&sequence->recurrent[0][0][0], TRANSFORMER_HEADS * TRANSFORMER_HEAD * TRANSFORMER_HEAD) ||
            !transformer_finite(&sequence->history[0][0][0], 3 * TRANSFORMER_ROWS * 3)) return 0;
        sequence->stage++; return 1;
    }
    if (stage == 118) { memcpy(output, sequence->residual, sizeof sequence->residual); sequence->stage++; return 1; }
    if (stage == 119) { sequence->stage++; return 1; }
    if (stage == 120) { sequence->positions++; sequence->stage = 0; return 1; }

    sequence->stage++;
    return 1;
}

int transformer_process(Transformer *transformer, TransformerSequence *sequence,`);
/* Layer 1 keeps the shared stage numbering but has its own shape: one snapshot rather
   than an array, no twelfth-layer branches, and extra finiteness checks on the
   recurrent and convolution state. */
replace(`    memcpy(sequence->snapshot, snapshot, sizeof sequence->snapshot);
    memcpy(sequence->incoming, input, sizeof sequence->incoming);
    memcpy(sequence->residual, input, sizeof sequence->residual);
    transformer_aggregate(transformer, sequence, 37);
    transformer_normalize(sequence->normalized, sequence->aggregate, transformer->qkv + 88, TRANSFORMER_WIDTH);
    transformer_qkv(transformer, sequence);
    transformer_decay(transformer, sequence);
    transformer_update_attention(sequence);
    transformer_attention_output(transformer, sequence);
    for (unsigned coordinate = 0; coordinate < TRANSFORMER_WIDTH; coordinate++)
        sequence->residual[coordinate] = sequence->incoming[coordinate] + sequence->residual[coordinate];
    transformer_aggregate(transformer, sequence, 38);
    if (!transformer_moe(transformer, sequence)) { sequence->failed = 1; return 0; }
    if (!transformer_finite(sequence->residual, TRANSFORMER_WIDTH) ||
        !transformer_finite(&sequence->recurrent[0][0][0], TRANSFORMER_HEADS * TRANSFORMER_HEAD * TRANSFORMER_HEAD) ||
        !transformer_finite(&sequence->history[0][0][0], 3 * TRANSFORMER_ROWS * 3)) {
        sequence->failed = 1;
        return 0;
    }
    memcpy(output, sequence->residual, sizeof sequence->residual);
    sequence->positions++;
    return 1;
}`,
`    memcpy(sequence->snapshot, snapshot, sizeof sequence->snapshot);
    sequence->stage = 1;
    clover_trace_open();
    while (sequence->stage) {
        const int executed = sequence->stage;
        const double began = clover_trace_active() ? clover_trace_clock() : 0.0;
        if (!transformer_step(transformer, sequence, input, output)) {
            sequence->failed = 1;
            return 0;
        }
        if (clover_trace_active())
            clover_trace_stage(TRANSFORMER_LAYER, executed, (unsigned long)sequence->positions,
                clover_trace_clock() - began,
                clover_trace_hash(sequence->residual, sizeof sequence->residual),
                executed >= TRANSFORMER_STAGE_EXPERT_FIRST && executed <= TRANSFORMER_STAGE_EXPERT_LAST
                    ? (long)sequence->selected[(executed - TRANSFORMER_STAGE_EXPERT_FIRST) / 5] : -1);
    }
    return 1;
}`);
};
replace(`    {26, 2, 1, 33792, 7168}, {27, 2, 1, 33792, 7168},
    {28, 2, 1, 7168, 33792}, {37, 0, 2, 7168, 1},`,
`    {29, 2, 3, 896, 7168}, {30, 2, 2, 896, 1},
    {31, 2, 1, 3584, 7168}, {32, 2, 1, 7168, 3584},
    {33, 2, 2, 3584, 1}, {34, 2, 1, 6144, 7168},
    {35, 2, 1, 6144, 7168}, {36, 2, 1, 7168, 6144}, {37, 0, 2, 7168, 1},`);
replace('case 19: case 26: case 27: case 28: case 38: case 39: return 1;',
 'case 19: case 29: case 30: case 31: case 32: case 33: case 34:\n    case 35: case 36: case 37: case 38: case 39: return 1;');
replace('const uint32_t fields[] = {1, 0,', 'const uint32_t fields[] = {1, 1,');
replace('!transformer_u32(index + 12)', 'transformer_u32(index + 12) == 1');
replace('const char *names[] = {"common.bin", "kda.bin", "dense.bin"};',
 'const char *names[] = {"common.bin", "attention.bin", "routed-shared.bin"};');
replace('    if (transformer->qkv) munmap(transformer->qkv, transformer->qkv_bytes);',
  '    root_close(transformer->root);\n    if (transformer->qkv) munmap(transformer->qkv, transformer->qkv_bytes);');
replace('int transformer_open(const char *qkv_path, const char *trunk_directory, Transformer **result)',
 'int transformer_open(const char *dataset, Transformer **result)');
replace('    if (!qkv_path || !trunk_directory || sizeof(float) != 4 || sizeof(double) != 8 ||',
`    char qkv_path[4096], trunk_directory[4096], root_directory[4096];
    if (!dataset || !transformer_path(qkv_path, sizeof qkv_path, dataset, "operators/qkv-all/layer-1/operator.bin") ||
        !transformer_path(trunk_directory, sizeof trunk_directory, dataset, "trunk-1") ||
        !transformer_path(root_directory, sizeof root_directory, dataset, "root-1") ||
        sizeof(float) != 4 || sizeof(double) != 8 ||`);
replace('!transformer_load_trunk(transformer, trunk_directory)) {',
 '!transformer_load_trunk(transformer, trunk_directory) || !(transformer->root = root_open(root_directory, TRANSFORMER_LAYER))) {');
replace('static void transformer_aggregate(const Transformer *transformer, TransformerSequence *sequence)',
`/* Layer 1 folds two sources, not nine, but it answers to the same switches as
   layers 2-92 or "the model" means two different things in one run. Record 37 is
   the stage 3 fold, record 38 the stage 21 fold. The default list excludes layer 1,
   which is what the measured-good distributed run actually did. */
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

static void transformer_aggregate(const Transformer *transformer, TransformerSequence *sequence, unsigned fold)`);
replace('    const float *sources[] = {sequence->snapshot, sequence->residual};',
`    const float *sources[] = {sequence->snapshot, sequence->residual};
    if (transformer_fold_residual()) {
        memcpy(sequence->aggregate, sequence->residual, TRANSFORMER_WIDTH * sizeof(float));
        (void)fold;
        return;
    }`);
replace('    for (unsigned source = 0; source < 2; source++) weights[source] = (float)((double)exponentials[source] / total);',
`    for (unsigned source = 0; source < 2; source++) weights[source] = (float)((double)exponentials[source] / total);
    if (transformer_hardmax(fold)) {
        unsigned best = weights[1] > weights[0] ? 1 : 0;
        weights[0] = best == 0 ? 1.0f : 0.0f;
        weights[1] = best == 1 ? 1.0f : 0.0f;
    }`);
replace('transformer->records[38].data + coordinate * 4', 'transformer->records[fold].data + coordinate * 4');
section('static void transformer_dense_activation(', 'int transformer_process(', readFileSync(new URL('moe.inc', import.meta.url), 'utf8') + '\n');
replace('int transformer_process(const Transformer *transformer, TransformerSequence *sequence,\n    const float input[TRANSFORMER_WIDTH], float output[TRANSFORMER_WIDTH], float snapshot[TRANSFORMER_WIDTH])',
`int transformer_process(Transformer *transformer, TransformerSequence *sequence,
    const float input[TRANSFORMER_WIDTH], const float snapshot[TRANSFORMER_WIDTH], float output[TRANSFORMER_WIDTH])`);
replace('        output == snapshot || sequence->failed || sequence->positions == SIZE_MAX ||',
 '        sequence->failed || sequence->positions == SIZE_MAX ||');
replace('!transformer_finite(input, TRANSFORMER_WIDTH)) return 0;',
 '!transformer_finite(input, TRANSFORMER_WIDTH) || !transformer_finite(snapshot, TRANSFORMER_WIDTH)) return 0;');
replace(`    memcpy(sequence->snapshot, input, sizeof sequence->snapshot);
    transformer_normalize(sequence->normalized, sequence->snapshot, transformer->qkv + 88, TRANSFORMER_WIDTH);`,
`    memcpy(sequence->snapshot, snapshot, sizeof sequence->snapshot);
    memcpy(sequence->incoming, input, sizeof sequence->incoming);
    memcpy(sequence->residual, input, sizeof sequence->residual);
    transformer_aggregate(transformer, sequence, 37);
    transformer_normalize(sequence->normalized, sequence->aggregate, transformer->qkv + 88, TRANSFORMER_WIDTH);`);
replace(`    transformer_aggregate(transformer, sequence);
    transformer_dense(transformer, sequence);`,
`    for (unsigned coordinate = 0; coordinate < TRANSFORMER_WIDTH; coordinate++)
        sequence->residual[coordinate] = sequence->incoming[coordinate] + sequence->residual[coordinate];
    transformer_aggregate(transformer, sequence, 38);
    if (!transformer_moe(transformer, sequence)) { sequence->failed = 1; return 0; }`);
replace('    memcpy(snapshot, sequence->snapshot, sizeof sequence->snapshot);\n', '');
stageLayerOne();
section('#ifndef TRANSFORMER_NO_MAIN', '\n#endif', readFileSync(new URL('cli.inc', import.meta.url), 'utf8'));
const destination = new URL('transformer-1.c', import.meta.url);
if (existsSync(destination)) assert.equal(readFileSync(destination, 'utf8'), source, 'refuse to overwrite a modified transformer');
else writeFileSync(destination, source, { flag: 'wx' });
console.log('PASS: derived standalone layer-1 functions from pinned layer-0 primitives; no runtime source dependency');