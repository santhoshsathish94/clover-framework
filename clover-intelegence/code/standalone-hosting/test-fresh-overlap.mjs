import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {createInterface} from 'node:readline';
const [ssh,host]=process.argv.slice(2);assert(ssh&&host);
const root='/opt/clover-k3/clover-intelegence/code/standalone-hosting';
async function run(binary){
  const command=`ulimit -c 0; cd ${root} && exec env -i PATH="$PATH" CLOVER_CONFIG=${root}/bin/configs/fresh-request.json OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores ./bin/${binary}`;
  const child=spawn(ssh,['-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ServerAliveInterval=30',host,command],{stdio:['pipe','pipe','pipe']});
  const closed=new Promise((resolve,reject)=>{child.once('error',reject);child.once('close',(code,signal)=>resolve({code,signal}));});
  let ready,done,cache,stopped,tail='';const tokens=[];
  child.stderr.setEncoding('utf8');child.stderr.on('data',chunk=>{tail=(tail+chunk).slice(-6000);});
  for await(const line of createInterface({input:child.stdout})){
    if(line.startsWith('READY_JSON ')){
      assert(!ready);ready=JSON.parse(line.slice(11));assert.equal(ready.result_cache_bytes,0);
      child.stdin.end(JSON.stringify({input_ids:[91019,25528],max_new_tokens:1})+'\n');
    }else if(line.startsWith('TOKEN_JSON '))tokens.push(JSON.parse(line.slice(11)).token);
    else if(line.startsWith('DONE_JSON '))done=JSON.parse(line.slice(10));
    else if(line.startsWith('CACHE_JSON '))cache=JSON.parse(line.slice(11));
    else if(line.startsWith('STOP_JSON '))stopped=JSON.parse(line.slice(10));
    else if(line.startsWith('REQUEST_ERROR '))throw Error(line);
  }
  const exit=await closed;assert.equal(exit.code,0,tail);assert.equal(exit.signal,null);
  assert(ready&&done&&cache&&stopped);assert.deepEqual(tokens,[418]);
  assert.equal(done.input_tokens,2);assert.equal(done.output_tokens,1);
  assert.equal(cache.result_hits,0);assert.equal(cache.result_misses,2944);assert.equal(cache.pipeline_reads,2944);
  assert.equal(stopped.requests,1);assert.equal(stopped.fixed_mapping_releases,372);
  const result={binary,startup_seconds:ready.startup_seconds,request_seconds:done.seconds,
    pipeline_wait_seconds:cache.pipeline_wait_seconds,result_hits:cache.result_hits,output_ids:tokens};
  console.log('FRESH_RESULT',JSON.stringify(result));return result;
}
await run('clover-one-before-overlap');
await run('clover-one');
console.log('PASS: fresh-process output unchanged; no repeated-request results reused in either schedule');