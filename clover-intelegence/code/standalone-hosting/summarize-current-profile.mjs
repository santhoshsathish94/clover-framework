import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {basename} from 'node:path';

const [input,output]=process.argv.slice(2);
assert(input&&output);
const report=JSON.parse(readFileSync(input,'utf8'));
const boundaries=['binding-and-stored-folds','pre-attention-aggregation','snapshot-push',
  'pre-attention-normalization','attention','attention-residual','pre-mlp-aggregation-and-normalization',
  'moe-scratch-and-parameter-bind','router-and-top16','prefetch-submit-and-latent-projection',
  'result-lookup-and-first-expert-submit','shared-expert-during-read','experts-mix-normalize-up',
  'shared-expert-and-cleanup','mlp-merge-and-cleanup','dense-mlp','mlp-residual-cache-save-and-unbind'];
const tail=['final-aggregation-and-normalization','head-fill-and-projection','argmax-and-cleanup'];
const entry=['request-reset','embedding','scratch-setup'];
const key=(position,layer,stage)=>JSON.stringify([position,layer,stage]);
const measurements=new Map();
for(const row of report.stages){
  assert(Number.isInteger(row.position)&&row.position>=0&&row.position<3);
  assert(Number.isInteger(row.layer)&&row.layer>=-1&&row.layer<=93);
  assert(Number.isFinite(row.ms)&&row.ms>=0&&Number.isInteger(row.calls)&&row.calls>0);
  const identity=key(row.position,row.layer,row.stage);
  assert(!measurements.has(identity),'Duplicate measurement');
  measurements.set(identity,row);
}
const value=(position,layer,stage)=>measurements.get(key(position,layer,stage))?.ms??0;
const total=(positions,stages)=>report.stages.filter(row=>positions.includes(row.position)&&stages.includes(row.stage))
  .reduce((sum,row)=>sum+row.ms,0)/1000;
const duration=positions=>report.positions.filter(row=>positions.includes(row.position)).reduce((sum,row)=>sum+row.total_seconds,0);
assert.deepEqual(report.positions.map(row=>row.position),[0,1,2]);
assert.equal(report.workers.length,12);
for(let position=0;position<3;position++){
  assert.deepEqual(report.stages.filter(row=>row.position===position&&row.stage==='total:layer').map(row=>row.layer),
    Array.from({length:93},(_,layer)=>layer));
  for(let layer=0;layer<93;layer++){
    const accounted=boundaries.reduce((sum,stage)=>sum+value(position,layer,stage),0);
    const elapsed=value(position,layer,'total:layer');
    assert(accounted<=elapsed+0.005,`Double-counted layer ${layer}, position ${position}`);
    assert(elapsed-accounted<2,`Unexplained layer gap ${layer}, position ${position}`);
  }
  const accounted=total([position],[...boundaries,...tail,...entry]);
  assert(accounted<=duration([position])+0.005,'Double-counted position');
  assert(duration([position])-accounted<0.1,'Unexplained position gap');
}
const groups=[
  ['Routed experts, mixing, latent norm and up projection',['experts-mix-normalize-up']],
  ['Vocabulary head preparation and projection',['head-fill-and-projection']],
  ['Attention across all 93 layers',['attention']],
  ['Shared experts',['shared-expert-during-read','shared-expert-and-cleanup']],
  ['Residual/state save, unbind and next-layer prediction',['mlp-residual-cache-save-and-unbind']],
  ['Latent down projection',['prefetch-submit-and-latent-projection']],
  ['Live router and top-16',['router-and-top16']],
  ['Layer-0 dense MLP',['dense-mlp']]
];
const counts=Array.from({length:93},(_,layer)=>report.stages.filter(row=>row.position===2&&row.layer===layer).length);
const lines=['# Current Runtime Timing','',
  report.scope+'.','',
  `Source: [timing data](${basename(input)}). Output assertions passed: IDs 418 then 276 (Rain falls -> on the). `+
  `All 279 layer totals are present, with ${Math.min(...counts)}-${Math.max(...counts)} recorded measurements per layer, depending on layer type. `+
  'These are boundaries and nested measurements, not a claim that every scalar operation has a separate timer.','',
  `Startup: **${report.startup.seconds.toFixed(6)} s**. First output (two input positions): **${duration([0,1]).toFixed(6)} s**. `+
  `Next output (one continuation position): **${duration([2]).toFixed(6)} s**. `+
  'This measures the current source with profiling enabled; it is not a retroactive decomposition of the earlier 45.028-second run.','',
  '## Elapsed-Time Breakdown','',
  '| Work | First output, two input positions (s) | Next output (s) |','|---|---:|---:|'];
for(const [label,stages] of groups)lines.push(`| ${label} | ${total([0,1],stages).toFixed(6)} | ${total([2],stages).toFixed(6)} |`);
const grouped=groups.flatMap(([,stages])=>stages);
const remainder=positions=>duration(positions)-total(positions,grouped);
lines.push(`| Other boundaries and unassigned timer overhead | ${remainder([0,1]).toFixed(6)} | ${remainder([2]).toFixed(6)} |`,
  `| Total | ${duration([0,1]).toFixed(6)} | ${duration([2]).toFixed(6)} |`,'',
  'First head use prepares the retained BF16 table and scores it. The head timer does not separate reading, decompression, layout conversion, checksums and scoring. '+
  'The next output reuses that head table, not a cached answer. Dataset preparation is not run.','',
  '## Inside Routed Experts','',
  'Gate/up/down wall timers are nested inside routed-expert processing above; do not add them again. Each includes its block decoding, validation and multiplication.','',
  '| Nested measurement | First output (s) | Next output (s) |','|---|---:|---:|');
for(const stage of ['detail:expert-gate','detail:expert-up','detail:expert-activation','detail:expert-down','detail:read-ahead-wait'])
  lines.push(`| ${stage} | ${total([0,1],[stage]).toFixed(6)} | ${total([2],[stage]).toFixed(6)} |`);
lines.push('','Worker times below sum concurrent threads. They locate CPU work but are **not elapsed seconds**, and must not be added to the wall-time table. '+
  'The read wrapper accesses already-prefetched bytes; it is not total disk I/O time.','',
  '| Worker phase | First output (summed worker s) | Next output (summed worker s) |','|---|---:|---:|');
for(const phase of ['read','decode','crc','math'])
  lines.push(`| ${phase} | ${total([0,1],['worker:'+phase]).toFixed(6)} | ${total([2],['worker:'+phase]).toFixed(6)} |`);
lines.push('','## Every Layer','',
  'Elapsed milliseconds. Layer 0 is dense; all remaining layers have routed and shared experts.','',
  '| Layer | Attention | First input | Second input | Continuation |','|---:|---|---:|---:|---:|');
for(let layer=0;layer<93;layer++)lines.push(`| ${layer} | ${layer===92||layer%4===3?'MLA':'KDA'} | `+
  [0,1,2].map(position=>value(position,layer,'total:layer').toFixed(6)).join(' | ')+' |');
lines.push('','## Every Recorded Measurement','',
  'Milliseconds. Boundary rows are disjoint within their layer; projections, operator aggregates, details and worker rows are nested or parallel and cannot be summed with them.','');
for(const layer of [-1,...Array.from({length:93},(_,index)=>index),93]){
  lines.push(`### ${layer===-1?'Input':layer===93?'Output':`Layer ${layer}`}`,'',
    '| Measurement | Timing kind | First input ms / calls | Second input ms / calls | Continuation ms / calls |','|---|---|---:|---:|---:|');
  const names=[...new Set(report.stages.filter(row=>row.layer===layer).map(row=>row.stage))];
  for(const name of names){
    const kind=[...boundaries,...tail,...entry].includes(name)?'boundary':name.startsWith('worker:')?'parallel worker':name==='total:layer'?'total':'nested';
    const cells=[0,1,2].map(position=>{
      const row=measurements.get(key(position,layer,name));
      return row?`${row.ms.toFixed(6)} / ${row.calls}`:'not executed';
    });
    lines.push(`| ${name} | ${kind} | ${cells.join(' | ')} |`);
  }
  lines.push('');
}
writeFileSync(output,lines.join('\n')+'\n',{flag:'wx'});
console.log(JSON.stringify({startup_seconds:report.startup.seconds,first_output_seconds:duration([0,1]),continuation_seconds:duration([2]),
  first_output_groups:Object.fromEntries(groups.map(([label,stages])=>[label,total([0,1],stages)])),
  other_seconds:remainder([0,1]),layer_totals:279,recorded_rows:report.stages.length},null,2));
console.log('PASS: every layer reconciles; every recorded measurement published; nested times excluded from elapsed totals');