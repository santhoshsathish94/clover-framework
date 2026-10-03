import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { spawnSync,spawn } from 'node:child_process';
const [ssh,host,prompt,mode='run']=process.argv.slice(2);
assert(ssh&&host&&prompt);
assert(/^[A-Za-z]+(?: [A-Za-z]+)*$/.test(prompt),'This launcher validates ASCII letter/space prompts only, not a general tokenizer');
assert(mode==='run'||mode==='check');
const model=readFileSync(new URL('../client/bin/dataset/tiktoken.model',import.meta.url),'utf8');
const ranks=new Map(),tokens=new Map();
for(const line of model.trim().split(/\r?\n/)){
  const [encoded,rank]=line.split(' ');const bytes=Buffer.from(encoded,'base64');
  ranks.set(bytes.toString('hex'),Number(rank));tokens.set(Number(rank),bytes);
}
const parts=prompt.match(/ ?[A-Z]*[a-z]+| ?[A-Z]+[a-z]*/g);
assert(parts&&parts.join('')===prompt);
const ids=[];
for(const part of parts){
  const input=Buffer.from(part);let pieces=Array.from(input,byte=>Buffer.from([byte]));
  const whole=ranks.get(input.toString('hex'));
  if(whole!==undefined){ids.push(whole);continue;}
  for(;;){
    let selected=-1,minimum=Infinity;
    for(let index=0;index+1<pieces.length;index++){
      const rank=ranks.get(Buffer.concat([pieces[index],pieces[index+1]]).toString('hex'));
      if(rank!==undefined&&rank<minimum){minimum=rank;selected=index;}
    }
    if(selected<0)break;
    pieces.splice(selected,2,Buffer.concat([pieces[selected],pieces[selected+1]]));
  }
  for(const piece of pieces){const id=ranks.get(piece.toString('hex'));assert(id!==undefined);ids.push(id);}
}
assert.deepEqual(Buffer.concat(ids.map(id=>tokens.get(id))),Buffer.from(prompt));
assert(ids.length<=256);
const quote=value=>`'${value.replaceAll("'","'\\''")}'`;
const root='/opt/clover-k3/clover-intelegence/code/distrubuted-hosting';
const oracle=`set -eu
ulimit -c 0
temporary=$(mktemp -d /tmp/clover-tokenizer-check.XXXXXXXX)
trap 'rm -f "$temporary/tiktoken.model" "$temporary/tokenizer_config.json"; rmdir "$temporary"' EXIT
ln -s ${root}/client/bin/dataset/tiktoken.model "$temporary/tiktoken.model"
ln -s ${root}/client/bin/configs/tokenizer_config.json "$temporary/tokenizer_config.json"
${root}/pipeline/bin/test-tokenizer "$temporary" ${quote(prompt)}`;
const args=['-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15'];
const reference=spawnSync(ssh,[...args,host,oracle],{encoding:'utf8',maxBuffer:1024*1024});
assert.ifError(reference.error);assert.equal(reference.status,0,reference.stderr);
assert.equal(reference.stdout.trim(),ids.join(','),'Native tokenizer and ASCII rank-BPE must agree');
console.log(`Exact prompt: ${JSON.stringify(prompt)}`);
console.log(`Verified token IDs (${ids.length}): ${ids.join(',')}`);
if(mode==='run'){
  const command=`ulimit -c 0; exec ${root}/pipeline/bin/pipeline ${root} ${ids.join(' ')}`;
  const child=spawn(ssh,[...args,'-o','ServerAliveInterval=30','-o','ServerAliveCountMax=6',host,command],{stdio:['ignore','pipe','inherit']});
  let output='';child.stdout.setEncoding('utf8');child.stdout.on('data',chunk=>{output+=chunk;assert(output.length<4096);});
  child.on('error',error=>{console.error(error);process.exitCode=1;});
  child.on('close',(code,signal)=>{if(code!==0||signal){console.error(`Pipeline failed: ${code} ${signal}`);process.exitCode=1;return;}
    console.log(`Generated next token: ${JSON.stringify(output)}`);});
}