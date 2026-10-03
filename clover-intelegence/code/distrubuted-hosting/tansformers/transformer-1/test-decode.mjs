import assert from 'node:assert/strict';
import { deflateSync, constants } from 'node:zlib';
import { spawnSync } from 'node:child_process';
const [executable, ssh, host] = process.argv.slice(2);
assert(executable);
const cases = [];
const add = (data, length, expected) => cases.push({ data, length, expected });
let state = 12345;
const random = Buffer.alloc(121856);
for (let index = 0; index < random.length; index++) {
  state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
  random[index] = state >>> 24;
}
const patterns = [Buffer.alloc(0), Buffer.from('a'), Buffer.alloc(121856, 7), random,
  Buffer.from('prepared values and exact positions '.repeat(3000)),
  Buffer.from(Array.from({ length: 121856 }, (_, index) => index % 251))];
let passed = 0;
for (const input of patterns) for (const options of [{ level: 0 }, { strategy: constants.Z_FIXED }, { level: 6 }, { strategy: constants.Z_HUFFMAN_ONLY }]) {
  const encoded = deflateSync(input, options);
  add(encoded, input.length, input);
  for (const damaged of [encoded.subarray(0, encoded.length - 1), Buffer.concat([encoded, Buffer.from([0])]),
    Buffer.from(encoded).fill(encoded.at(-1) ^ 1, encoded.length - 1)]) add(damaged, input.length, null);
  add(encoded, input.length + 1, null);
  if (input.length) add(encoded, input.length - 1, null);
  passed++;
}
add(deflateSync(Buffer.from('dictionary'), { dictionary: Buffer.from('dictionary') }), 10, null);
add(Buffer.from([0x78,0x9c,7,0,0,0,1]), 0, null);
const framed = cases.flatMap(test => {
  const header = Buffer.alloc(8);
  header.writeUInt32LE(test.data.length);
  header.writeUInt32LE(test.length, 4);
  return [header, test.data];
});
const args = ssh ? ['-T', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=15', host,
  `ulimit -c 0; exec ${executable} --batch`] : ['--batch'];
const result = spawnSync(ssh ?? executable, args, { input: Buffer.concat(framed), maxBuffer: 16 * 1024 * 1024 });
assert.ifError(result.error);
assert.equal(result.signal, null);
assert.equal(result.status, 0, result.stderr.toString());
assert.equal(result.stderr.length, 0, result.stderr.toString());
let offset = 0;
for (const test of cases) {
  assert(offset < result.stdout.length, 'missing decoder result');
  assert.equal(result.stdout[offset++], test.expected === null ? 1 : 0);
  if (test.expected !== null) {
    assert.deepEqual(result.stdout.subarray(offset, offset + test.length), test.expected);
    offset += test.length;
  }
}
assert.equal(offset, result.stdout.length, 'unexpected decoder output');
console.log(`PASS: ${passed} oracle cases and ${cases.length - passed} rejection controls; single batch, transport errors rejected`);