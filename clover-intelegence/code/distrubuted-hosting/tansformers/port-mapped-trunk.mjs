/* Maps each pod's own trunk and QKV instead of copying them into the heap. The bytes
   then live once in the page cache rather than once per pod, which is what let 92
   resident stages fit. Record pointers move from private buffers into the mapping. */
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';

const READ_FN = `static int transformer_read(FILE *file, uint64_t offset, size_t length, unsigned char **output)
{
    if (offset > LONG_MAX || fseek(file, (long)offset, SEEK_SET)) return 0;
    unsigned char *data = malloc(length);
    if (!data) return 0;
    if (fread(data, 1, length, file) != length) { free(data); return 0; }
    *output = data;
    return 1;
}
`;

const MAP_FN = READ_FN + `
static const unsigned char *transformer_map(const char *path, size_t expected)
{
    int handle = open(path, O_RDONLY);
    if (handle < 0) return NULL;
    struct stat info;
    const unsigned char *mapped = NULL;
    if (!fstat(handle, &info) && (size_t)info.st_size == expected) {
        void *address = mmap(NULL, expected, PROT_READ, MAP_SHARED, handle, 0);
        if (address != MAP_FAILED) mapped = address;
    }
    close(handle);
    return mapped;
}
`;

const STRUCT_BEFORE = `    Root *root;
} Transformer;`;
const STRUCT_AFTER = `    Root *root;
    size_t qkv_bytes;
    const unsigned char *group_map[3];
    size_t group_bytes[3];
} Transformer;`;

const CLOSE_BEFORE = `    if (!transformer) return;
    for (unsigned id = 0; id < TRANSFORMER_RECORDS; id++) free(transformer->records[id].data);
    root_close(transformer->root);
    free(transformer->qkv);
    free(transformer);`;
const CLOSE_AFTER = `    if (!transformer) return;
    for (unsigned group = 0; group < 3; group++)
        if (transformer->group_map[group]) munmap((void *)transformer->group_map[group], transformer->group_bytes[group]);
    root_close(transformer->root);
    if (transformer->qkv) munmap(transformer->qkv, transformer->qkv_bytes);
    free(transformer);`;

const QKV_BEFORE = `    FILE *file = transformer_open_sized(path, bytes);
    if (!file) return 0;
    int loaded = transformer_read(file, 0, bytes, &transformer->qkv);
    if (fclose(file)) loaded = 0;
    if (!loaded) return 0;`;
const QKV_AFTER = `    FILE *file = transformer_open_sized(path, bytes);
    if (!file) return 0;
    if (fclose(file)) return 0;
    transformer->qkv = (unsigned char *)transformer_map(path, bytes);
    if (!transformer->qkv) return 0;
    transformer->qkv_bytes = bytes;`;

const GROUP_BEFORE = `        file = transformer_open_sized(path, lengths[group]);
        if (!file) return 0;
        for (unsigned id = 0; valid && id < TRANSFORMER_RECORDS; id++) {
            TransformerRecord *record = &transformer->records[id];
            if (!transformer_needs(id) || record->group != group) continue;
            if (!transformer_read(file, record->offset, (size_t)record->length, &record->data)) { valid = 0; break; }`;
const GROUP_AFTER = `        file = transformer_open_sized(path, lengths[group]);
        if (!file) return 0;
        if (fclose(file)) return 0;
        file = NULL;
        transformer->group_map[group] = transformer_map(path, (size_t)lengths[group]);
        if (!transformer->group_map[group]) return 0;
        transformer->group_bytes[group] = (size_t)lengths[group];
        for (unsigned id = 0; valid && id < TRANSFORMER_RECORDS; id++) {
            TransformerRecord *record = &transformer->records[id];
            if (!transformer_needs(id) || record->group != group) continue;
            record->data = transformer->group_map[group] + record->offset;`;

const GROUP_TAIL_BEFORE = `        }
        if (fclose(file)) valid = 0;
        if (!valid) return 0;
    }
    return 1;
}`;
const GROUP_TAIL_AFTER = `        }
        if (!valid) return 0;
    }
    return 1;
}`;

const INCLUDE_BEFORE = '#include <string.h>\n#include "live-root.h"';
const INCLUDE_AFTER = '#include <string.h>\n#include <fcntl.h>\n#include <unistd.h>\n#include <sys/mman.h>\n#include <sys/stat.h>\n#include "live-root.h"';

let changed = 0, already = 0;
for (let layer = 1; layer <= 92; layer++) {
    const path = new URL(`transformer-${layer}/transformer-${layer}.c`, import.meta.url);
    let source = readFileSync(path, 'utf8');
    if (source.includes('transformer_map')) { already++; continue; }
    const edits = [[INCLUDE_BEFORE, INCLUDE_AFTER], [READ_FN, MAP_FN], [STRUCT_BEFORE, STRUCT_AFTER],
        [CLOSE_BEFORE, CLOSE_AFTER], [QKV_BEFORE, QKV_AFTER], [GROUP_BEFORE, GROUP_AFTER],
        [GROUP_TAIL_BEFORE, GROUP_TAIL_AFTER]];
    for (const [before, after] of edits) {
        assert.equal(source.split(before).length, 2, `layer ${layer}: ${before.slice(0, 48)}`);
        source = source.replace(before, after);
    }
    writeFileSync(path, source);
    changed++;
}
console.log(`PASS: ${changed} pods now map their own trunk and QKV, ${already} already done`);
