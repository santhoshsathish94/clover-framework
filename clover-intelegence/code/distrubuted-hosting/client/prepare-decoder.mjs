import assert from 'node:assert/strict';
import { readFileSync,writeFileSync,existsSync } from 'node:fs';
const source=readFileSync(new URL('../tansformers/transformer-1/decode.h',import.meta.url));
const target=new URL('decode.h',import.meta.url);
if(existsSync(target))assert.deepEqual(readFileSync(target),source);else writeFileSync(target,source,{flag:'wx'});
console.log('PASS: client decoder matches tested project C decoder');