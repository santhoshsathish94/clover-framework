import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';
const here = new URL('./', import.meta.url);
const read = name => readFileSync(new URL(name, here), 'utf8').replace(/\r\n/g, '\n');
const json = name => JSON.parse(read(name));
const layerOne = read('../transformer-1/transformer-1.c');
const rootOne = read('../transformer-1/root.h');
const decoder = read('../transformer-1/decode.h');
const layers = process.argv.slice(2).length ? process.argv.slice(2).map(Number) : Array.from({length:91}, (_, index) => index + 2);
function save(url, content) {
  mkdirSync(new URL('./', url), { recursive:true });
  if (existsSync(url)) assert.equal(readFileSync(url,'utf8'), content, `refuse overwrite ${url}`);
  else writeFileSync(url, content, {flag:'wx'});
}
for (const layer of layers) {
  assert(Number.isInteger(layer) && layer >= 2 && layer <= 92);
  const mla = layer === 92 || layer % 4 === 3;
  const trunk = json(`trunk-${layer}/manifest.json`), maps = json(`root-${layer}/maps.json`);
  const operator = json(`operators/qkv-all/layer-${layer}/manifest.json`);
  assert.equal(trunk.layer, layer); assert.equal(maps.layer, layer); assert.equal(operator.layer, layer);
  assert.equal(trunk.attention, mla ? 'MLA' : 'KDA');
  const counts = [maps.palette_count,maps.unique_maps,maps.unique_templates,maps.template_refs];
  assert(counts.every(value => Number.isInteger(value) && value > 0));
  assert(counts[0] <= 255 && counts[1] < 65536 && counts[2] <= 2688 && counts[3] < 65536);
  const constants = counts[0]*2 + counts[1]*16 + (counts[2]+1)*2 + counts[3]*2;
  let root = rootOne.replace('ROOT_CONSTANT_BYTES = 15880, ROOT_PALETTE = 66, ROOT_MAPS = 465,',
    `ROOT_CONSTANT_BYTES = ${constants}, ROOT_PALETTE = ${counts[0]}, ROOT_MAPS = ${counts[1]},`)
    .replace('ROOT_TEMPLATES = 515, ROOT_REFS = 3638,',`ROOT_TEMPLATES = ${counts[2]}, ROOT_REFS = ${counts[3]},`);
  let source = layerOne;
  const replace = (before, after) => { assert.equal(source.split(before).length, 2, `anchor ${before.slice(0,90)}`); source = source.replace(before,after); };
  const section = (start,end,content) => {const begin=source.indexOf(start),finish=source.indexOf(end,begin+start.length); assert(begin>=0&&finish>begin);source=source.slice(0,begin)+content+source.slice(finish);};
  replace('#include "root.h"', `#include "root.h"\n#define TRANSFORMER_LAYER ${layer}\n#define TRANSFORMER_MLA ${Number(mla)}\n#define TRANSFORMER_INPUT_SNAPSHOTS ${Math.ceil(layer/12)}\n#define TRANSFORMER_OUTPUT_SNAPSHOTS ${Math.floor(layer/12)+1}`);
  replace('    float palette[256];', '    float palette[256];\n    TransformerRecord operator_records[40];\n    const unsigned char *input_gains;');
  replace('    float snapshot[TRANSFORMER_WIDTH], normalized[TRANSFORMER_WIDTH];',
    '    float snapshots[8][TRANSFORMER_WIDTH], normalized[TRANSFORMER_WIDTH];\n    unsigned snapshot_count;');
  replace('    RootScratch root_scratch;', `    RootScratch root_scratch;
    float *mla_cache;
    size_t mla_capacity;
    float mla_query_latent[1536], mla_query[18432], mla_kv_latent[576], mla_expanded[24576];`);
  const groups = {'common.bin':0,'attention.bin':1,'routed-shared.bin':2};
  const layout = trunk.records.map(record=>`    {${record.id}, ${groups[record.group]}, ${record.kind}, ${record.rows}, ${record.columns}}`).join(',\n');
  section('static const TransformerLayout transformer_layout[]', 'static uint32_t transformer_u32', `static const TransformerLayout transformer_layout[] = {\n${layout}\n};\n\n`);
  section('static int transformer_needs(', 'static int transformer_load_qkv(', `static int transformer_needs(unsigned id)\n{\n    if (TRANSFORMER_MLA && id >= 8 && id < 29) return 0;\n    switch (id) {\n    case 5: case 6: case 7: case 11: case 12: case 13: case 18:\n    case 19: case 29: case 30: case 31: case 32: case 33: case 34:\n    case 35: case 36: case 37: case 38: case 39: return 1;\n    default: return 0;\n    }\n}\n\n`);
  section('static int transformer_load_qkv(', 'static int transformer_load_trunk(', read('load-operator.inc')+'\n');
  replace('transformer_u32(index + 12) == 1', 'transformer_u32(index + 12) == TRANSFORMER_LAYER');
  source=source.replaceAll('operators/qkv-all/layer-1/operator.bin',`operators/qkv-all/layer-${layer}/operator.bin`).replaceAll('"trunk-1"',`"trunk-${layer}"`).replaceAll('"root-1"',`"root-${layer}"`);
  replace('    memset(sequence, 0, sizeof *sequence);', '    free(sequence->mla_cache);\n    memset(sequence, 0, sizeof *sequence);');
  replace('    free(sequence);', '    if (sequence) free(sequence->mla_cache);\n    free(sequence);');
  replace('    const TransformerRecord *record = &transformer->records[id];\n    const unsigned char *ids',
    '    const TransformerRecord *record = transformer->operator_records[id].data ? &transformer->operator_records[id] : &transformer->records[id];\n    const unsigned char *ids');
  replace('static float transformer_convolve(', '#if !TRANSFORMER_MLA\nstatic float transformer_convolve(');
  replace('static void transformer_aggregate(', '#endif\n\n#if TRANSFORMER_MLA\n'+read('mla.inc')+'\n#endif\n\nstatic void transformer_aggregate(');
  replace('    const float *sources[] = {sequence->snapshot, sequence->residual};\n    float scores[2], exponentials[2], weights[2];',
    '    const float *sources[9];\n    unsigned count = sequence->snapshot_count + 1;\n    for (unsigned source = 0; source < sequence->snapshot_count; source++) sources[source] = sequence->snapshots[source];\n    sources[sequence->snapshot_count] = sequence->residual;\n    float scores[9], exponentials[9], weights[9];');
  const agBegin=source.indexOf('static void transformer_aggregate('),agEnd=source.indexOf('static void transformer_activation(',agBegin);
  let aggregate=source.slice(agBegin,agEnd).replaceAll('source < 2','source < count').replace('float maximum = scores[1] > scores[0] ? scores[1] : scores[0];',
    'float maximum = scores[0];\n    for (unsigned source = 1; source < count; source++) if (scores[source] > maximum) maximum = scores[source];');
  source=source.slice(0,agBegin)+aggregate+source.slice(agEnd);
  source=source.slice(0,source.indexOf('int transformer_process('))+read('process.inc');
  const target = new URL(`../transformer-${layer}/`,here);
  save(new URL(`transformer-${layer}.c`,target),source);
  save(new URL('root.h',target),root); save(new URL('decode.h',target),decoder);
  save(new URL('.gitignore',target),`/bin/transformer-${layer}\n/bin/*.exe\n/bin/test-*\n`);
  save(new URL('CONTEXT.md',target),`# Transformer ${layer}\n\nRequested function-oriented ${mla?'MLA':'KDA'} layer ${layer}. Sources generated from validated layer1 primitives and original MLA semantics. Runtime data will live in bin/dataset. Snapshot inputs ${Math.ceil(layer/12)}, outputs ${Math.floor(layer/12)+1}. Root counts ${counts.join('/')}. Numerical validation and dataset moves are pending; see ../remaining/CONTEXT.md.\n`);
  const metadata={layer,attention:mla?'MLA':'KDA',input_snapshots:Math.ceil(layer/12),output_snapshots:Math.floor(layer/12)+1,
    root_counts:counts,files:{}};
  for (const [name,content] of [[`transformer-${layer}.c`,source],['root.h',root],['decode.h',decoder]]) metadata.files[name]=createHash('sha256').update(content).digest('hex');
  writeFileSync(new URL('generation.json',target),JSON.stringify(metadata,null,2)+'\n');
  save(new URL('README.md',target),`# Transformer ${layer}\n\nStandalone layer-${layer} ${metadata.attention} implementation with separate functions for loading, aggregation, normalization, projection, attention, routing, expert computation and I/O. Main coordinates only. Standard C/math and platform 64-bit file I/O, no third-party runtime. Stored values are reused; input-dependent expert and QKV outputs are computed live.\n\n## Runtime\n\nFrom this directory, build with:\n\n\`\`\`sh\nmkdir -p bin\nulimit -c 0\ngcc -std=c11 -O2 -march=native -Wall -Wextra -Werror -pedantic -ffp-contract=off -fno-fast-math transformer-${layer}.c -lm -o bin/transformer-${layer}\ncd bin\n./transformer-${layer} --inspect\n./transformer-${layer} --stream\n\`\`\`\n\nAll runtime data is under bin/dataset: trunk-${layer}, root-${layer}, operators/qkv-all/layer-${layer}. Old model paths remain compatibility aliases. Large payloads remain on AX102; local folders carry metadata only. Runtime does not read observations or external model data.\n\n## Interface\n\nEach token input is 7168 residual floats followed by ${metadata.input_snapshots} snapshots of 7168 floats each, oldest first. Output is residual followed by ${metadata.output_snapshots} snapshots, one hexadecimal-float line per vector. EOF ends the sequence. Snapshot boundary updates occur after pre-attention aggregation. ${mla?'MLA keeps a dynamically growing causal key/value/positional cache.':'KDA keeps recurrent state and raw convolution history.'}\n\nThe API transformer_process accepts model, sequence, input, input snapshots, input snapshot count, output and output snapshots. Callers allocate enough output storage and use nonoverlapping buffers outside internal state. Each sequence owns state; all calls sharing a model must be serialized because its expert-file cursor is shared. Reset starts a new sequence; close sequences before their model. No new client/network connection is implemented.\n\n## Verification\n\nGeneration is not numerical verification. See [campaign evidence](../remaining/README.md) for current status, tested scope and per-layer results. [generation.json](generation.json) pins source hashes and layer-specific metadata. Existing transformer1/server sources are unchanged.\n`);
  save(new URL('bin/dataset/README.md',target),`# Layer ${layer} Dataset Location\n\nFull payloads on AX102: /opt/clover-k3/clover-intelegence/code/tansformers/transformer-${layer}/bin/dataset. This workspace contains metadata only, not numeric payloads. Runtime groups: trunk-${layer}, root-${layer}, operators/qkv-all/layer-${layer}. See [program](../../README.md).\n`);
  save(new URL(`bin/dataset/trunk-${layer}/manifest.json`,target),readFileSync(new URL(`trunk-${layer}/manifest.json`,here),'utf8'));
  save(new URL(`bin/dataset/root-${layer}/maps.json`,target),readFileSync(new URL(`root-${layer}/maps.json`,here),'utf8'));
  save(new URL(`bin/dataset/root-${layer}/build-verification.json`,target),readFileSync(new URL(`root-${layer}/build-verification.json`,here),'utf8'));
  save(new URL(`bin/dataset/operators/qkv-all/layer-${layer}/manifest.json`,target),readFileSync(new URL(`operators/qkv-all/layer-${layer}/manifest.json`,here),'utf8'));
  console.log(`GENERATED ${layer} ${metadata.attention} snapshots ${metadata.input_snapshots}->${metadata.output_snapshots}`);
}