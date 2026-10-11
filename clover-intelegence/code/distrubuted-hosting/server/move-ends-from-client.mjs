/* Moves the two outer ends of the chain out of the client and into the server: token to
   embedding on the way in, vector to token on the way out. Names are rebased to the
   server, with the table's own extents kept distinct from the server's layer extents. */
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, existsSync } from 'node:fs';

const client = new URL('../client/', import.meta.url);
const server = new URL('./', import.meta.url);

const renames = [
    ['CLIENT_NUMERIC_TABLE_H', 'SERVER_NUMERIC_TABLE_H'],
    ['CLIENT_WIDTH', 'SERVER_TABLE_WIDTH'],
    ['CLIENT_BLOCK_ROWS', 'SERVER_TABLE_BLOCK_ROWS'],
    ['CLIENT_BLOCKS', 'SERVER_TABLE_BLOCKS'],
    ['CLIENT_ROWS', 'SERVER_TABLE_ROWS'],
    ['ClientInput', 'ServerInput'],
    ['ClientOutput', 'ServerOutput'],
    ['client_input', 'server_input'],
    ['client_output', 'server_output'],
    ['client_head_row', 'server_head_row'],
    ['client_numeric_environment', 'server_numeric_environment'],
];
const rebase = (text) => renames.reduce((acc, [from, to]) => acc.split(from).join(to), text);

for (const name of ['decode.h', 'sha256.h']) {
    const target = new URL(name, server);
    assert(!existsSync(target), `${name} already exists in server/`);
    writeFileSync(target, readFileSync(new URL(name, client)));
}

const table = new URL('numeric-table.h', server);
assert(!existsSync(table), 'numeric-table.h already exists in server/');
writeFileSync(table, rebase(readFileSync(new URL('numeric-table.h', client), 'utf8')));

/* The tokenizer and the two edge converters, without the client's command line. */
const source = readFileSync(new URL('client.c', client), 'utf8').replace(/\r\n/g, '\n');
const cut = source.indexOf('#ifndef CLIENT_NO_MAIN');
assert(cut > 0, 'client main guard not found');
let body = rebase(source.slice(0, cut).trimEnd());

/* The client carried its own copy of the JSON reader. normalization/json.h already has
   the same seven helpers byte for byte, and server.c includes it, so keep one copy. */
const reader = body.match(/typedef struct \{[^}]*\} JsonReader;\n/);
assert(reader, 'JsonReader typedef not found');
body = body.replace(reader[0], '');
const from = body.indexOf('static int base64_digit');
const to = body.indexOf('static int load_added_tokens');
assert(from > 0 && to > from, 'duplicated json block not found');
body = body.slice(0, from) + body.slice(to);

const ends = new URL('ends.h', server);
assert(!existsSync(ends), 'ends.h already exists in server/');
writeFileSync(ends, `#ifndef SERVER_ENDS_H
#define SERVER_ENDS_H
/* Token to embedding on the way in, vector to token on the way out. Derived from the
   retired client stage; the server owns both ends of the chain, so nothing else does. */
/* json.h takes its prerequisites from whoever includes it, so they go first here. */
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../normalization/json.h"
${body}
#endif
`);

console.log(`PASS: server/ends.h, numeric-table.h, decode.h and sha256.h written (${body.length} bytes of body)`);
