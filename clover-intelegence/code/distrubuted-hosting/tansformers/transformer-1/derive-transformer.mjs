import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync, existsSync } from 'node:fs';

const original = readFileSync(new URL('../../server/server.c', import.meta.url));
assert.equal(createHash('sha256').update(original).digest('hex'),
  '734ef995e0e05866b50e71572722129d84f970f661007d0d2e88e4e95d726446');
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
replace('#include <string.h>', '#include <string.h>\n#include "root.h"');
replace('TRANSFORMER_HEADS = 96, TRANSFORMER_DENSE = 33792, TRANSFORMER_RECORDS = 40,',
  'TRANSFORMER_HEADS = 96, TRANSFORMER_LATENT = 3584, TRANSFORMER_EXPERT = 3072,\n    TRANSFORMER_SHARED = 6144, TRANSFORMER_RECORDS = 40,');
replace('    float palette[256];\n} Transformer;', '    float palette[256];\n    Root *root;\n} Transformer;');
replace('    float dense_gate[TRANSFORMER_DENSE], dense_up[TRANSFORMER_DENSE], dense_output[TRANSFORMER_WIDTH];',
`    float incoming[TRANSFORMER_WIDTH], latent[TRANSFORMER_LATENT], mixture[TRANSFORMER_LATENT];
    float expert_gate[TRANSFORMER_EXPERT], expert_up[TRANSFORMER_EXPERT], expert_down[TRANSFORMER_LATENT];
    float latent_norm[TRANSFORMER_LATENT], routed[TRANSFORMER_WIDTH];
    float shared_gate[TRANSFORMER_SHARED], shared_up[TRANSFORMER_SHARED], shared_output[TRANSFORMER_WIDTH];
    unsigned selected[16];
    float weights[16];
    RootScratch root_scratch;`);
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
replace('    free(transformer->qkv);', '    root_close(transformer->root);\n    free(transformer->qkv);');
replace('int transformer_open(const char *qkv_path, const char *trunk_directory, Transformer **result)',
 'int transformer_open(const char *dataset, Transformer **result)');
replace('    if (!qkv_path || !trunk_directory || sizeof(float) != 4 || sizeof(double) != 8 ||',
`    char qkv_path[4096], trunk_directory[4096], root_directory[4096];
    if (!dataset || !transformer_path(qkv_path, sizeof qkv_path, dataset, "operators/qkv-all/layer-1/operator.bin") ||
        !transformer_path(trunk_directory, sizeof trunk_directory, dataset, "trunk-1") ||
        !transformer_path(root_directory, sizeof root_directory, dataset, "root-1") ||
        sizeof(float) != 4 || sizeof(double) != 8 ||`);
replace('!transformer_load_trunk(transformer, trunk_directory)) {',
 '!transformer_load_trunk(transformer, trunk_directory) || !(transformer->root = root_open(root_directory))) {');
replace('static void transformer_aggregate(const Transformer *transformer, TransformerSequence *sequence)',
 'static void transformer_aggregate(const Transformer *transformer, TransformerSequence *sequence, unsigned fold)');
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
section('#ifndef TRANSFORMER_NO_MAIN', '\n#endif', readFileSync(new URL('cli.inc', import.meta.url), 'utf8'));
const destination = new URL('transformer-1.c', import.meta.url);
if (existsSync(destination)) assert.equal(readFileSync(destination, 'utf8'), source, 'refuse to overwrite a modified transformer');
else writeFileSync(destination, source, { flag: 'wx' });
console.log('PASS: derived standalone layer-1 functions from pinned layer-0 primitives; no runtime source dependency');