import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {readFileSync} from 'node:fs';
import {createInterface} from 'node:readline';
const [ssh,host]=process.argv.slice(2); assert(ssh&&host);
const prompts=[{text:'Rain falls',ids:[91019,25528],expected:[418,276]},
  {text:'Tea tastes',ids:[149058,39466],expected:[1517,15600]}];
const vocab=new Map();
for(const line of readFileSync(new URL('../../dataset/tiktoken.model',import.meta.url),'utf8').trim().split(/\r?\n/)){
  const [encoded,id]=line.split(' ');vocab.set(Number(id),Buffer.from(encoded,'base64'));
}
for(const prompt of prompts)assert.equal(Buffer.concat(prompt.ids.map(id=>vocab.get(id))).toString('utf8'),prompt.text);
const root='/opt/clover-k3/clover-intelegence/code/standalone-hosting';
const child=spawn(ssh,['-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ServerAliveInterval=30',host,
  `cd ${root} && exec bash run.sh`],{stdio:['pipe','pipe','pipe']});
const closed=new Promise((resolve,reject)=>{child.once('error',reject);child.once('close',(code,signal)=>resolve({code,signal}));});
const tokens=new Map(),reports=[],caches=[],steps=[];let ready=0,stopped=0,tail='';
const stderr=createInterface({input:child.stderr});
stderr.on('line',line=>{tail=(tail+line+'\n').slice(-6000);if(line.startsWith('STEP_JSON ')){const event=JSON.parse(line.slice(10));steps.push(event);console.log('STEP',JSON.stringify(event));}});
for await(const line of createInterface({input:child.stdout})){
  if(line.startsWith('READY_JSON ')){
    const event=JSON.parse(line.slice(11));assert.equal(event.expert_cache_budget_bytes,0);assert.equal(event.expert_pipeline_bytes,72*1024*1024);
    ready++;console.log('READY',JSON.stringify(event));
    for(const prompt of prompts)child.stdin.write(JSON.stringify({input_ids:prompt.ids,max_new_tokens:2})+'\n');
    child.stdin.end();
  }else if(line.startsWith('TOKEN_JSON ')){
    const event=JSON.parse(line.slice(11));const output=tokens.get(event.request)||[];
    assert.equal(event.index,output.length);output.push(event.token);tokens.set(event.request,output);
  }else if(line.startsWith('DONE_JSON '))reports.push(JSON.parse(line.slice(10)));
  else if(line.startsWith('CACHE_JSON ')){const event=JSON.parse(line.slice(11));caches.push(event);console.log('PIPELINE',JSON.stringify(event));}
  else if(line.startsWith('STOP_JSON ')){const event=JSON.parse(line.slice(10));assert.equal(event.requests,2);assert.equal(event.fixed_mapping_releases,372);stopped++;}
  else if(line.startsWith('REQUEST_ERROR '))throw Error(line);
}
const exit=await closed;assert.equal(exit.code,0,tail);assert.equal(exit.signal,null);
assert.equal(ready,1);assert.equal(stopped,1);assert.equal(reports.length,2);assert.equal(caches.length,2);assert.equal(steps.length,4);
for(let index=0;index<2;index++){
  const output=tokens.get(index+1);assert.deepEqual(output,prompts[index].expected);
  assert.equal(caches[index].pipeline_reads,3*92*16);assert.equal(caches[index].pipeline_ram_bytes,72*1024*1024);
  console.log('RESULT',JSON.stringify({input:prompts[index].text,output:Buffer.concat(output.map(id=>vocab.get(id))).toString('utf8'),
    tokens:output,request_seconds:reports[index].seconds,pipeline_wait_seconds:caches[index].pipeline_wait_seconds}));
}
for(const step of steps){assert.equal(step.expert_matches,0);assert.equal(step.expert_projection_calls,4416);}
console.log('PASS: prior exact output IDs retained, real asynchronous expert reads, no expert cache, changed-input correctness and clean shutdown');