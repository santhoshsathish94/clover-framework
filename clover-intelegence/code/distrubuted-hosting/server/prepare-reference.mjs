import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync, existsSync } from 'node:fs';

const sourceUrl = new URL('../../../clover-k3/clover-k3.c', import.meta.url);
const outputUrl = new URL('reference-zero.inc', import.meta.url);
const bytes = readFileSync(sourceUrl);
assert.equal(createHash('sha256').update(bytes).digest('hex'),
  '5628f7c3d2b7932b2b8776dcb498d54bbc730b1785ba7819989bdb51b6604e65');
const source = bytes.toString('utf8');
const boundary = '    /* ---------------------------------------------------------------- tail */';
assert.equal(source.split(boundary).length, 2);
const prefix = source.slice(0, source.indexOf(boundary));
assert(prefix.includes('for (int L = 0; L <= last; L++) {'));
assert(prefix.includes('resid[t][i] = resid[t][i] + ffn[t][i];'));
const reference = 'static int test_compare_reference(void);\n' + prefix +
  '    free(fa); free(fm);\n    return test_compare_reference();\n}\n';
if (existsSync(outputUrl)) assert.equal(readFileSync(outputUrl, 'utf8'), reference);
else writeFileSync(outputUrl, reference, { flag: 'wx' });
console.log('PASS: hash-pinned original layer code retained; reference stops before tail');