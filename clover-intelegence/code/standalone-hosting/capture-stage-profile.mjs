import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {createInterface} from 'node:readline';
import {writeFileSync} from 'node:fs';
const [ssh,host,output]=process.argv.slice(2);
assert(ssh&&host&&output);
const child=spawn(ssh,['-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ServerAliveInterval=30',host,
  'ulimit -c 0; cd /opt/clover-k3/clover-intelegence/code/standalone-hosting && OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores exec ./bin/profile-stages ../../dataset'],{stdio:['ignore','pipe','pipe']});
const closed=new Promise((resolve,reject)=>{child.once('error',reject);child.once('close',(code,signal)=>resolve({code,signal}));});
let tail='',passed=false,startup;
const stages=[],positions=[],workers=[];
child.stderr.setEncoding('utf8');child.stderr.on('data',chunk=>{tail=(tail+chunk).slice(-8000);});
for await(const line of createInterface({input:child.stdout})) {
  if(line.startsWith('STAGE_JSON '))stages.push(JSON.parse(line.slice(11)));
  if(line.startsWith('POSITION_JSON ')){const value=JSON.parse(line.slice(14));positions.push(value);console.log('POSITION',JSON.stringify(value));}
  if(line.startsWith('WORKER_JSON '))workers.push(JSON.parse(line.slice(12)));
  if(line.startsWith('STARTUP_JSON ')){startup=JSON.parse(line.slice(13));console.log('STARTUP',JSON.stringify(startup));}
  if(line.startsWith('PASS:'))passed=true;
}
const exit=await closed;assert.equal(exit.code,0,tail);assert.equal(exit.signal,null);assert(passed&&startup);
assert.equal(positions.length,3);assert.equal(workers.length,12);
for(let position=0;position<3;position++){
  const layers=stages.filter(row=>row.position===position&&row.stage==='total:layer');
  assert.equal(layers.length,93);assert.deepEqual(layers.map(row=>row.layer),Array.from({length:93},(_,index)=>index));
  assert(stages.filter(row=>row.position===position).every(row=>row.ms>=0&&Number.isFinite(row.ms)));
}
const report={scope:'AX102 16 OpenMP threads; three incremental positions; timing values only, no model vectors',startup,positions,workers,stages};
writeFileSync(output,JSON.stringify(report,null,2),{flag:'wx'});
for(let position=0;position<3;position++){
  const entries=stages.filter(row=>row.position===position);
  const sums=new Map();
  for(const row of entries)sums.set(row.stage,(sums.get(row.stage)||0)+row.ms);
  console.log('LAYER0',position,JSON.stringify(entries.filter(row=>row.layer===0)));
  console.log('SUMS',position,JSON.stringify([...sums].sort((left,right)=>right[1]-left[1]).slice(0,14)));
}
console.log('PASS: all279layer totals and timing rows captured; model output assertions passed; no activation files');