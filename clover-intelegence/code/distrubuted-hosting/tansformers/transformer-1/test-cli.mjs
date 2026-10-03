import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
const [ssh, host, executable, dataset] = process.argv.slice(2);
assert(ssh && host && executable && dataset);
const quote = value => `'${value.replaceAll("'", "'\\''")}'`;
const invoke = (mode, input, path = dataset) => {
  const command = `ulimit -c 0; exec ${quote(executable)} ${quote(mode)} ${quote(path)}`;
  const result = spawnSync(ssh, ['-T', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', host, command], {
    input, encoding: 'utf8', maxBuffer: 2 * 1024 * 1024,
  });
  assert.ifError(result.error);
  assert.equal(result.signal, null);
  return result;
};
const inspect = invoke('--inspect', '');
assert.equal(inspect.status, 0, inspect.stderr);
assert(inspect.stdout.includes('19 additional trunk records, 896 computed-value experts'));
const valid = invoke('--stream', Array(14336).fill('0').join(' ') + '\n');
assert.equal(valid.status, 0, valid.stderr);
const lines = valid.stdout.trim().split('\n');
assert.equal(lines.length, 2);
for (const line of lines) {
  const values = line.trim().split(/\s+/);
  assert.equal(values.length, 7168);
  assert(values.every(value => /^-?0x0p\+0$/.test(value)));
}
for (const input of ['0\n', 'bad\n', Array(14336).fill('nan').join(' ')]) {
  const rejected = invoke('--stream', input);
  assert.notEqual(rejected.status, 0);
  assert.equal(rejected.stdout, '');
}
const empty = invoke('--stream', '');
assert.equal(empty.status, 0);
assert.equal(empty.stdout, '');
const missing = invoke('--inspect', '', '/dev/null');
assert.equal(missing.status, 1);
assert.equal(missing.stdout, '');
console.log('PASS: packaged CLI inspect, vector-pair framing, residual/S0 output, EOF and malformed/nonfinite/missing-data controls');