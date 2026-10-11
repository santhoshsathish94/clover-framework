/* Locks each pod's trunk and QKV resident. Experts stay unlocked so the page cache
   can keep whichever ones are actually hot. */
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';

const INCLUDE_BEFORE = '#include <string.h>\n#include <fcntl.h>';
const INCLUDE_AFTER = '#include <string.h>\n#include <errno.h>\n#include <fcntl.h>';

const MAP_BEFORE = `        void *address = mmap(NULL, expected, PROT_READ, MAP_SHARED, handle, 0);
        if (address != MAP_FAILED) mapped = address;`;
const MAP_AFTER = `        void *address = mmap(NULL, expected, PROT_READ, MAP_SHARED, handle, 0);
        if (address != MAP_FAILED) {
            const char *lock = getenv("CLOVER_LOCK_TRUNK");
            if ((!lock || strcmp(lock, "0")) && mlock(address, expected))
                fprintf(stderr, "trunk not locked (%s): %s\\n", path, strerror(errno));
            mapped = address;
        }`;

let changed = 0, already = 0;
for (let layer = 2; layer <= 92; layer++) {
    const path = new URL(`transformer-${layer}/transformer-${layer}.c`, import.meta.url);
    let source = readFileSync(path, 'utf8');
    if (source.includes('CLOVER_LOCK_TRUNK')) { already++; continue; }
    for (const [before, after] of [[INCLUDE_BEFORE, INCLUDE_AFTER], [MAP_BEFORE, MAP_AFTER]]) {
        assert.equal(source.split(before).length, 2, `layer ${layer}: ${before.slice(0, 40)}`);
        source = source.replace(before, after);
    }
    writeFileSync(path, source);
    changed++;
}
console.log(`PASS: ${changed} pods lock their trunk resident, ${already} already done`);
