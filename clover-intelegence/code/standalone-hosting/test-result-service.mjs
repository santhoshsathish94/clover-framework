import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {createInterface} from 'node:readline';
const [ssh,host]=process.argv.slice(2);assert(ssh&&host);
const prompts=[{ids:[91019,25528],expected:[418,276],name:'Rain falls, first run'},
  {ids:[91019,25528],expected:[418,276],name:'Rain falls, exact repeat'},
  {ids:[149058,39466],expected:[1517,15600],name:'Tea tastes, changed input'}];
const root='/opt/clover-k3/clover-intelegence/code/standalone-hosting';
const child=spawn(ssh,['-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ServerAliveInterval=30',host,
  `cd ${root} && exec bash run.sh`],{stdio:['pipe','pipe','pipe']});
const closed=new Promise((resolve,reject)=>{child.once('error',reject);child.once('close',(code,signal)=>resolve({code,signal}));});
let ready=null, stopped=0, tail='';
const reports=[],caches=[],tokens=new Map(),steps=[];
createInterface({input:child.stderr}).on('line',line=>{
  tail=(tail+line+'\n').slice(-8000);
  if(line.startsWith('STEP_JSON '))steps.push(JSON.parse(line.slice(10)));
});
for await(const line of createInterface({input:child.stdout})){
  if(line.startsWith('READY_JSON ')){
    assert.equal(ready,null); ready=JSON.parse(line.slice(11));
    assert(ready.result_cache_entries>0 && ready.result_cache_bytes<=256*1024*1024);
    assert.equal(ready.expert_cache_budget_bytes,0);
    console.log('READY',JSON.stringify(ready));
    for(const prompt of prompts)child.stdin.write(JSON.stringify({input_ids:prompt.ids,max_new_tokens:2})+'\n');
    child.stdin.end();
  }else if(line.startsWith('TOKEN_JSON ')){
    const event=JSON.parse(line.slice(11)),out=tokens.get(event.request)||[];
    assert.equal(event.index,out.length);out.push(event.token);tokens.set(event.request,out);
  }else if(line.startsWith('DONE_JSON '))reports.push(JSON.parse(line.slice(10)));
  else if(line.startsWith('CACHE_JSON ')){
    const cache=JSON.parse(line.slice(11));caches.push(cache);console.log('CACHE',JSON.stringify(cache));
  }else if(line.startsWith('STOP_JSON ')){
    const event=JSON.parse(line.slice(10));assert.equal(event.requests,prompts.length);assert.equal(event.fixed_mapping_releases,372);stopped++;
  }else if(line.startsWith('REQUEST_ERROR '))throw Error(line);
}
const exit=await closed;assert.equal(exit.code,0,tail);assert.equal(exit.signal,null);
assert(ready);assert.equal(stopped,1);assert.equal(reports.length,3);assert.equal(caches.length,3);assert.equal(steps.length,6);
for(let index=0;index<3;index++){
  assert.deepEqual(tokens.get(index+1),prompts[index].expected);
  assert.equal(reports[index].input_tokens,2);assert.equal(reports[index].output_tokens,2);
  assert.equal(caches[index].result_hits+caches[index].result_misses,3*92*16);
  assert.equal(caches[index].pipeline_reads,caches[index].result_misses,'Prefetch only missing expert results');
  assert.equal(caches[index].result_stores,caches[index].result_misses);
  assert.equal(caches[index].result_cache_bytes,ready.result_cache_bytes);
  console.log('RESULT',JSON.stringify({input:prompts[index].name,output_ids:tokens.get(index+1),seconds:reports[index].seconds,
    reused:caches[index].result_hits,computed:caches[index].result_misses,reads:caches[index].pipeline_reads}));
}
assert(caches[1].result_hits>0.9*3*92*16,'Exact repeat should reuse its working set');
assert(caches[1].pipeline_reads<caches[0].pipeline_reads);
assert(caches[2].result_misses>0,'Changed input must not be treated as a completed identical input');
for(const step of steps){assert.equal(step.expert_matches,0);assert.equal(step.expert_projection_calls+step.expert_result_hits*3,4416);}
console.log('PASS: exact repeat skips expert calculations and reads, changed input stays exact, RAM bounded and shutdown clean');