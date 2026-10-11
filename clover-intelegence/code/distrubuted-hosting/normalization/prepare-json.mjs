import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
/* These helpers came verbatim from the retired client stage, which was also their
   provenance check. json.h is the only copy now, so the check is a pinned digest. */
const content = readFileSync(new URL('json.h', import.meta.url));
assert.equal(createHash('sha256').update(content).digest('hex'),
  '981e98d52eabfedb81040ef4b1fba30cee4c754379108d86d372d99c8067f809');
console.log('PASS: JSON/Base64 helpers unchanged');