import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';

const [executable, leaves, ssh, host] = process.argv.slice(2);
assert(executable && leaves);
assert.equal(Boolean(ssh),Boolean(host));
const quote = value => `'${value.replaceAll("'", "'\\''")}'`;
const invoke = (mode,input) => {
  const args = ssh ? ['-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15',host,
    `ulimit -c 0; exec ${quote(executable)} ${quote(mode)} ${quote(leaves)}`] : [mode,leaves];
  const result = spawnSync(ssh ?? executable,args,{input,encoding:'utf8',maxBuffer:2*1024*1024});
  assert.ifError(result.error);
  assert.equal(result.signal,null);
  return result;
};
const inspect = invoke('--inspect','');
assert.equal(inspect.status,0,inspect.stderr);
assert(inspect.stdout.includes('8 snapshots + residual -> 7168 normalized values'));
const vector = Array(9*7168).fill('0').join(' ') + '\n';
const valid = invoke('--stream',vector + vector);
assert.equal(valid.status,0,valid.stderr);
assert.equal(valid.stderr,'');
const lines = valid.stdout.trim().split('\n');
assert.equal(lines.length,2);
for (const line of lines) {
  const values = line.trim().split(/\s+/);
  assert.equal(values.length,7168);
  assert(values.every(value=>/^-?0x0p\+0$/.test(value)));
}
for (const input of ['0\n','invalid\n',Array(8*7168).fill('0').join(' ')]) {
  const result = invoke('--stream',input);
  assert.equal(result.status,2,result.stderr);
  assert.equal(result.stdout,'');
}
const nonfinite = invoke('--stream',Array(9*7168).fill('nan').join(' '));
assert.equal(nonfinite.status,1,nonfinite.stderr);
assert.equal(nonfinite.stdout,'');
const empty = invoke('--stream','');
assert.equal(empty.status,0,empty.stderr);
assert.equal(empty.stdout,'');
console.log('PASS: inspect, two nine-vector bundles -> two normalized vectors, EOF, malformed/incomplete/nonfinite inputs');