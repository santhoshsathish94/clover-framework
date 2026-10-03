import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {createInterface} from 'node:readline';

const [ssh,host]=process.argv.slice(2);
assert(ssh && host);
const command='cd /opt/clover-k3/clover-intelegence/code/standalone-hosting && bash run.sh';
const child=spawn(ssh,['-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ServerAliveInterval=30',host,command],{stdio:['pipe','pipe','pipe']});
const results=[], tokens=[];
let ready=0, errors=0, stopped=0, stderrTail='';
child.stderr.setEncoding('utf8');
child.stderr.on('data',chunk=>{stderrTail=(stderrTail+chunk).slice(-6000);});
child.on('error',error=>{throw error;});
for await(const line of createInterface({input:child.stdout})){
  if(line.startsWith('READY_JSON ')){
    ready++;
    const event=JSON.parse(line.slice(11));
    assert.equal(event.fixed_mappings,372);
    child.stdin.write(JSON.stringify({input_ids:Array(129).fill(387)})+'\n');
    child.stdin.write(JSON.stringify({input_ids:[387],max_new_tokens:129})+'\n');
    child.stdin.write(JSON.stringify({input_ids:[387],max_new_tokens:1})+'\n');
    child.stdin.end(JSON.stringify({input_ids:[387],max_new_tokens:1})+'\n');
  }else if(line.startsWith('REQUEST_ERROR '))errors++;
  else if(line.startsWith('TOKEN_JSON '))tokens.push(JSON.parse(line.slice(11)));
  else if(line.startsWith('DONE_JSON ')){
    const event=JSON.parse(line.slice(10));
    results.push(event);
    console.log('Completed real JSON request',JSON.stringify(event));
  }else if(line.startsWith('STOP_JSON ')){
    const event=JSON.parse(line.slice(10));
    assert.equal(event.requests,2); assert.equal(event.fixed_mapping_releases,372); stopped++;
  }
}
const code=await new Promise(resolve=>{if(child.exitCode!==null)resolve(child.exitCode);else child.once('close',resolve);});
assert.equal(code,0,stderrTail);
assert.equal(ready,1);assert.equal(stopped,1);assert.equal(errors,2);
assert.equal(results.length,2);assert.equal(tokens.length,2);
for(let index=0;index<2;index++){
  assert.equal(results[index].request,index+1);assert.equal(results[index].input_tokens,1);
  assert.equal(results[index].output_tokens,1);
  assert.equal(tokens[index].request,index+1);assert.equal(tokens[index].index,0);
  assert(tokens[index].token>=0 && tokens[index].token<163840);
}
assert.equal(tokens[0].token,tokens[1].token,'New request must not retain the earlier attention state');
console.log('PASS: real one-token requests, repeated-request reset, 129-input/output rejection, bounded streaming and EOF shutdown');