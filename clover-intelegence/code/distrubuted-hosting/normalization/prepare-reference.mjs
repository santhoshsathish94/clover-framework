import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';
const base = new URL('../tansformers/remaining/', import.meta.url);
const locks = readFileSync(new URL('SOURCE-LOCK.sha256',base),'utf8');
const readLocked = name => {
  const bytes = readFileSync(new URL(name,base));
  const path = '/opt/clover-k3/clover-intelegence/code/tansformers/remaining/' + name;
  const expected = locks.split(/\r?\n/).find(line => line.endsWith('  ' + path));
  assert(expected);
  assert.equal(createHash('sha256').update(bytes).digest('hex'),expected.slice(0,64));
  return bytes.toString('utf8').replace(/\r\n/g, '\n');
};
const reference = readLocked('reference-all.inc');
const harness = readLocked('test-reference.c');
const boundary = harness.indexOf('static void test_finish_layer(int layer)\n{');
assert(boundary > 0);
const prefix = '#define _GNU_SOURCE\n#define NORMALIZATION_NO_MAIN\n#include "normalization.c"\n' +
  harness.slice(0,boundary).replace('#include "reference-all.inc"','#include "reference-tail.inc"')
    .replace('dataset_root, *packages_root, *prompt_name','dataset_root, *prompt_name');
const final = prefix + readFileSync(new URL('reference-check.inc',import.meta.url),'utf8');
for (const [name,content] of [['reference-tail.inc',reference],['test-reference.c',final]]) {
  const output = new URL(name,import.meta.url);
  if (existsSync(output)) assert.equal(readFileSync(output,'utf8'),content);
  else writeFileSync(output,content,{flag:'wx'});
}
console.log('PASS: verified original-layer oracle copied unchanged; normalization compared only after layer92');