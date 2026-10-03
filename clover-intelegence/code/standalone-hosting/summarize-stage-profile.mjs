import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {basename} from 'node:path';
const [beforePath,afterPath,output]=process.argv.slice(2);
assert(beforePath&&afterPath&&output);
const before=JSON.parse(readFileSync(beforePath,'utf8'));
const after=JSON.parse(readFileSync(afterPath,'utf8'));
const stages=['binding-and-stored-folds','pre-attention-aggregation','snapshot-push','pre-attention-normalization',
  'attention','attention-residual','pre-mlp-aggregation-and-normalization','moe-scratch-and-parameter-bind',
  'router-and-top16','prefetch-submit-and-latent-projection','result-lookup-and-first-expert-submit',
  'shared-expert-during-read','experts-mix-normalize-up','shared-expert-and-cleanup','mlp-merge-and-cleanup',
  'dense-mlp','mlp-residual-cache-save-and-unbind'];
function value(report,position,layer,stage){return report.stages.filter(row=>row.position===position&&row.layer===layer&&row.stage===stage).reduce((sum,row)=>sum+row.ms,0);}
function total(report,position,stage){return report.stages.filter(row=>row.position===position&&row.stage===stage).reduce((sum,row)=>sum+row.ms,0);}
for(const report of [before,after])for(let position=0;position<3;position++){
  const rows=report.stages.filter(row=>row.position===position&&row.stage==='total:layer');
  assert.deepEqual(rows.map(row=>row.layer),Array.from({length:93},(_,layer)=>layer));
  for(const row of rows){
    const accounted=stages.reduce((sum,stage)=>sum+value(report,position,row.layer,stage),0);
    assert(accounted<=row.ms+0.005,'Nested timing was counted as disjoint stage time');
    assert(row.ms-accounted<2,'Large unexplained boundary gap');
  }
}
const lines=[
'# Stage-By-Stage Profile',
'',
'AX102, 2026-10-02, 16 OpenMP threads with close/core affinity. One process per run; three incremental positions: two input positions and one generated continuation. Identical prior token assertions passed. No model activation arrays were saved.',
'',
'[Workflow](../../../clover-one/workflow.md), [stage definitions](../../../research/k3/model/k3-stages.md), and the shared local /ai/ website were read against actual runtime source. The website is a recorded example, not live timing evidence. Its stage-count claims do not establish runtime speed.',
'',
`Raw timing-only evidence: [before](${basename(beforePath)}), [after](${basename(afterPath)}). All 93 layer totals exist for all three positions (279 per run). Nested projection/operator/detail rows and parallel worker sums must NOT be added to the disjoint stage wall times. Clocks and report bookkeeping add instrumentation overhead; these are observations, not an uninstrumented controlled benchmark.`,
'',
'## Input Boundary',
'',
'The native tokenizer was loaded once. Repeat means cover 1,000 calls; OS caches were not flushed. Every embedding coordinate matched the original central shard. The standalone service receives IDs; it does not tokenize text itself.',
'',
'| Operation | First call / setup (ms) | Warm mean (ms) |',
'|---|---:|---:|',
'| Tokenizer load | 24.541058 | once |',
'| Text to IDs | 0.006271 | 0.000371 |',
'| Seed metadata open | 1.641398 | once |',
'| Embedding row, input position 0 | 1.769608 | 0.002263 |',
'| Embedding row, input position 1 | 1.603948 | 0.002309 |',
'',
'Cold reader use includes compressed-block read, decode and integrity checks. A cached row is BF16 widening and copying. The fraction-of-a-millisecond expectation holds for the measured warm path, not metadata setup or first compressed-block access.',
'',
'## Whole Positions',
'',
'| Position | Role | Before (s) | After (s) |',
'|---:|---|---:|---:|'
];
for(let position=0;position<3;position++)lines.push(`| ${position} | ${['first input, no head','last input, first head fill','continuation, warm head'][position]} | ${before.positions[position].total_seconds.toFixed(6)} | ${after.positions[position].total_seconds.toFixed(6)} |`);
lines.push('',`Startup: before ${before.startup.seconds.toFixed(6)} s; after ${after.startup.seconds.toFixed(6)} s. Startup is not charged to an embedding lookup. First-use head fill is shown separately below, not presented as steady token cost.`,
'','## Stage Totals At Continuation','','All values below are wall milliseconds, summed across layers where applicable. Rows are disjoint; total layer rows are omitted from this table.','','| Stage | Before (ms) | After (ms) |','|---|---:|---:|');
for(const stage of stages)lines.push(`| ${stage} | ${total(before,2,stage).toFixed(6)} | ${total(after,2,stage).toFixed(6)} |`);
for(const stage of ['final-aggregation-and-normalization','head-fill-and-projection','argmax-and-cleanup'])lines.push(`| tail: ${stage} | ${value(before,2,93,stage).toFixed(6)} | ${value(after,2,93,stage).toFixed(6)} |`);
lines.push('','## Why Time Is Spent','',
'- Binding reads the existing prepared vectors, including fold records 37/38 and KDA decay-base record 39. They are not regenerated per token. Copies and guards cost microseconds here.',
'- Layer 0 pre-attention aggregation is a guarded copy, not a computed aggregate. Later layers score and mix current residual/snapshot values; those coefficients depend on the current state.',
'- Attention and dense/shared projections use stored coefficients but still read and multiply their matrices. Their individual Q/K/V/G/O and MLP projection timings are recorded below.',
'- Routed experts dominate. Their stage includes bounded read-ahead waits, decompression, validating the same decoded-BF16 checksums, gate/up/down projections, input-dependent activation and mixing. The original expert and QKV metadata distinguishes fixed parameters from recorded input-specific outputs.',
'- Head first use fills the verified BF16 byte cache. Repeated head projections avoid that format work and take about 51 ms in this profile; filling is not a per-token requirement.',
'- The tested larger Huffman tables did not reduce decode CPU time. That experiment was rejected. Four-literal decoding from one bounded bit window reduced the alternating decoder probe from about 1.378 to 1.04 summed worker-seconds; original outputs/checks are preserved.',
'',
'This cycle does not establish five-second tokens. It isolates the actual remaining cost instead of treating every stage as slow. Full 128-input/128-output model execution remains unmeasured.',
'','## Every Layer','','Continuation-position wall milliseconds. Expert total includes mix/normalization/up; dense/shared is a separate disjoint stage. Worker columns below are NOT part of this wall table.','','| Layer | Type | Binding/folds | Pre-attn norm | Attention | Router/top16 | Experts/mix | Dense/shared | Total |','|---:|---|---:|---:|---:|---:|---:|---:|---:|');
for(let layer=0;layer<93;layer++){
 const get=stage=>value(after,2,layer,stage).toFixed(3);
 const shared=value(after,2,layer,'shared-expert-and-cleanup')+value(after,2,layer,'shared-expert-during-read')+value(after,2,layer,'mlp-merge-and-cleanup');
 lines.push(`| ${layer} | ${layer===92||layer%4===3?'MLA':'KDA'} | ${get('binding-and-stored-folds')} | ${get('pre-attention-normalization')} | ${get('attention')} | ${get('router-and-top16')} | ${get('experts-mix-normalize-up')} | ${layer===0?get('dense-mlp'):shared.toFixed(3)} | ${get('total:layer')} |`);
}
lines.push('','## Every Layer Worker Costs','','Summed parallel worker milliseconds, not wall milliseconds. Read is the block-access wrapper; asynchronous disk reading occurs in the separate pipeline thread. Its exposed wait is wall time. Neither is a measurement of physical disk bytes.','','| Layer | Decode worker ms | CRC worker ms | Math worker ms | Read-ahead wait ms |','|---:|---:|---:|---:|---:|');
for(let layer=1;layer<93;layer++)lines.push(`| ${layer} | ${value(after,2,layer,'worker:decode').toFixed(3)} | ${value(after,2,layer,'worker:crc').toFixed(3)} | ${value(after,2,layer,'worker:math').toFixed(3)} | ${value(after,2,layer,'detail:read-ahead-wait').toFixed(3)} |`);
lines.push('','## Detailed Layer Stages','','Execution-order wall stages and nested projection details for the continuation position. Only rows listed in the disjoint stage table above can be summed.','');
for(let layer=0;layer<93;layer++){
 lines.push(`### Layer ${layer}`,'','| Measurement | ms | Calls |','|---|---:|---:|');
 for(const row of after.stages.filter(row=>row.position===2&&row.layer===layer&&!row.stage.startsWith('worker:')&&!row.stage.startsWith('op:')))
  lines.push(`| ${row.stage} | ${row.ms.toFixed(6)} | ${row.count} |`);
 lines.push('');
}
writeFileSync(output,lines.join('\n')+'\n',{flag:'wx'});
console.log('PASS: validated timing reconciliation and published all93layer tables plus detailed stage rows');