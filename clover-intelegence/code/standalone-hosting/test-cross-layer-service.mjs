import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {createInterface} from 'node:readline';

const [ssh,host]=process.argv.slice(2);
assert(ssh&&host);
const root='/opt/clover-k3/clover-intelegence/code/standalone-hosting';

async function run(binary,enabled){
  const command=`ulimit -c 0; cd ${root} && exec env -i PATH="$PATH" CLOVER_CONFIG=${root}/bin/configs/fresh-request.json OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores ./bin/${binary}`;
  const child=spawn(ssh,['-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ServerAliveInterval=30',host,command],{stdio:['pipe','pipe','pipe']});
  const closed=new Promise((resolve,reject)=>{
    child.once('error',reject);
    child.once('close',(code,signal)=>resolve({code,signal}));
  });
  let ready,done,cache,stopped,tail='';
  const tokens=[];
  child.stderr.setEncoding('utf8');
  child.stderr.on('data',chunk=>{tail=(tail+chunk).slice(-6000);});
  try {
    for await(const line of createInterface({input:child.stdout})){
      if(line.startsWith('READY_JSON ')){
        assert(!ready);
        ready=JSON.parse(line.slice(11));
        assert.equal(ready.result_cache_bytes,0);
        child.stdin.end(JSON.stringify({input_ids:[91019,25528],max_new_tokens:1})+'\n');
      }else if(line.startsWith('TOKEN_JSON ')) tokens.push(JSON.parse(line.slice(11)).token);
      else if(line.startsWith('DONE_JSON ')) done=JSON.parse(line.slice(10));
      else if(line.startsWith('CACHE_JSON ')) cache=JSON.parse(line.slice(11));
      else if(line.startsWith('STOP_JSON ')) stopped=JSON.parse(line.slice(10));
      else if(line.startsWith('REQUEST_ERROR ')) throw Error(line);
    }
    const exit=await closed;
    assert.equal(exit.code,0,tail); assert.equal(exit.signal,null);
    assert(ready&&done&&cache&&stopped);
    assert.deepEqual(tokens,[418]);
    assert.equal(done.input_tokens,2); assert.equal(done.output_tokens,1);
    assert.equal(done.stop_reason,'length');
    assert.equal(cache.result_hits,0); assert.equal(cache.result_misses,2944);
    assert.equal(cache.pipeline_ram_bytes,72*1024*1024);
    assert.equal(cache.cross_layer_submitted,enabled?184:0);
    assert.equal(cache.cross_layer_hits+cache.cross_layer_misses,cache.cross_layer_submitted);
    assert.equal(cache.pipeline_reads,2944+cache.cross_layer_misses);
    assert.equal(cache.head_decoded_blocks,10240);
    if(enabled) assert(cache.cross_layer_hits>0 && cache.cross_layer_prediction_seconds>0);
    assert.equal(stopped.requests,1); assert.equal(stopped.fixed_mapping_releases,372);
    console.log('CROSS_LAYER_RESULT',JSON.stringify({binary,startup_seconds:ready.startup_seconds,
      request_seconds:done.seconds,pipeline_wait_seconds:cache.pipeline_wait_seconds,
      prediction_seconds:cache.cross_layer_prediction_seconds,candidates:cache.cross_layer_submitted,
      hits:cache.cross_layer_hits,misses:cache.cross_layer_misses,pipeline_reads:cache.pipeline_reads,
      pipeline_bytes:cache.pipeline_bytes,result_hits:cache.result_hits,output_ids:tokens}));
  }finally{
    if(child.exitCode===null && child.signalCode===null){
      child.stdin.destroy(); child.kill(); await closed.catch(()=>{});
    }
  }
}

await run('clover-one-without-cross-layer',false);
await run('clover-one',true);
console.log('PASS: fresh-request output unchanged; cross-layer matches reused and all speculative misses accounted for');