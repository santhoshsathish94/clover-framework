import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const [executable, qkv, trunk, ssh, host] = process.argv.slice(2);
assert(executable && qkv && trunk, 'expected executable, QKV file and trunk-0 directory');
assert.equal(Boolean(ssh), Boolean(host), 'provide both SSH executable and host');
const invoke = (mode, input, qkvPath = qkv) => {
  const command = ssh ?? executable;
  const args = ssh
    ? ['-T', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', host, executable, mode, qkvPath, trunk]
    : [mode, qkvPath, trunk];
  const result = spawnSync(command, args, {
    input, encoding: 'utf8', maxBuffer: 4 * 1024 * 1024,
  });
  assert.ifError(result.error);
  assert.equal(result.signal, null);
  return result;
};
const inspect = invoke('--inspect', '');
assert.equal(inspect.status, 0, inspect.stderr);
assert(inspect.stdout.includes('13 additional prepared records'));
const valid = invoke('--stream', Array(7168).fill('0').join(' ') + '\n');
assert.equal(valid.status, 0, valid.stderr);
const lines = valid.stdout.trim().split('\n');
assert.equal(lines.length, 2);
for (const line of lines) {
  const values = line.trim().split(/\s+/);
  assert.equal(values.length, 7168);
  assert(values.every(value => /^-?0x0p\+0$/.test(value)));
}
for (const input of ['0\n', 'invalid\n', Array(7168).fill('nan').join(' ')]) {
  const rejected = invoke('--stream', input);
  assert.notEqual(rejected.status, 0);
  assert.equal(rejected.stdout, '');
}
const empty = invoke('--stream', '');
assert.equal(empty.status, 0);
assert.equal(empty.stdout, '');
if (ssh) {
  const rejected = invoke('--inspect', '', '/dev/null');
  assert.equal(rejected.status, 1);
  assert.equal(rejected.stdout, '');
} else {
  const scratch = mkdtempSync(join(tmpdir(), 'clover-server-schema-'));
  try {
    const malformed = join(scratch, 'truncated-qkv.bin');
    writeFileSync(malformed, 'K3QKV001', { flag: 'wx' });
    const rejected = invoke('--inspect', '', malformed);
    assert.equal(rejected.status, 1);
    assert.equal(rejected.stdout, '');
  } finally {
    rmSync(scratch, { recursive: true });
  }
}
console.log('PASS: stream framing, exact zero vector/S0, empty EOF, malformed/truncated/nonfinite input, truncated dataset');