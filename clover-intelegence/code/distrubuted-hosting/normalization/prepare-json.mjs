import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
const source = readFileSync(new URL('../client/client.c', import.meta.url), 'utf8');
const start = source.indexOf('static int base64_digit(');
const end = source.indexOf('static int load_added_tokens(', start);
assert(start >= 0 && end > start);
const content = '#ifndef NORMALIZATION_JSON_H\n#define NORMALIZATION_JSON_H\n' +
  'typedef struct { const unsigned char *cursor, *end; unsigned depth; } JsonReader;\n\n' +
  source.slice(start, end).replace(/\r\n/g, '\n') + '#endif\n';
const output = new URL('json.h', import.meta.url);
if (existsSync(output)) assert.equal(readFileSync(output, 'utf8'), content);
else writeFileSync(output, content, { flag: 'wx' });
console.log('PASS: copied existing JSON/Base64 helpers; no client source change');