import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { writeFileSync } from 'node:fs';

const [ssh, host] = process.argv.slice(2);
assert(ssh && host);
const results = [];
for (const layer of [2, 3, 12, 92]) {
  const inputSnapshots = Math.ceil(layer / 12);
  const outputSnapshots = Math.floor(layer / 12) + 1;
  const root = `/opt/clover-k3/clover-intelegence/code/tansformers/transformer-${layer}/bin`;
  const invoke = (mode, input) => {
    const command = `ulimit -c 0; cd ${root} && exec ./transformer-${layer} ${mode}`;
    const result = spawnSync(ssh, ['-T', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
      '-o', 'ConnectTimeout=15', host, command], { input, encoding: 'utf8', maxBuffer: 4 * 1024 * 1024 });
    assert.ifError(result.error);
    assert.equal(result.signal, null);
    return result;
  };
  const inspect = invoke('--inspect', '');
  assert.equal(inspect.status, 0, inspect.stderr);
  assert(inspect.stdout.includes(`snapshots ${inputSnapshots} -> ${outputSnapshots}`));
  const valid = invoke('--stream', Array((inputSnapshots + 1) * 7168).fill('0').join(' ') + '\n');
  assert.equal(valid.status, 0, valid.stderr);
  assert.equal(valid.stderr, '');
  const lines = valid.stdout.trim().split('\n');
  assert.equal(lines.length, outputSnapshots + 1);
  for (const line of lines) {
    const values = line.trim().split(/\s+/);
    assert.equal(values.length, 7168);
    assert(values.every(value => /^-?0x0p\+0$/.test(value)));
  }
  for (const input of ['0\n', 'invalid\n']) {
    const rejected = invoke('--stream', input);
    assert.equal(rejected.status, 2, rejected.stderr);
    assert.equal(rejected.stdout, '');
  }
  const nonfinite = invoke('--stream', Array((inputSnapshots + 1) * 7168).fill('nan').join(' '));
  assert.equal(nonfinite.status, 1, nonfinite.stderr);
  assert.equal(nonfinite.stdout, '');
  const empty = invoke('--stream', '');
  assert.equal(empty.status, 0, empty.stderr);
  assert.equal(empty.stdout, '');
  results.push({ layer, input_snapshots: inputSnapshots, output_snapshots: outputSnapshots,
    inspect: true, zero_bundle: true, incomplete_and_malformed: true, nonfinite: true, empty_eof: true });
  console.log(`CLI PASS: layer ${layer}, ${inputSnapshots}->${outputSnapshots} snapshots and error controls`);
}
writeFileSync(new URL('cli-results.json', import.meta.url), JSON.stringify({ gate: 'PASS', results,
  scope: 'Four representative packaged CLI variants; no vector payloads persisted' }, null, 2) + '\n', { flag: 'wx' });