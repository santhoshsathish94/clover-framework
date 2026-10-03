# Client Package

[client.c](client.c) implements the human-selected **exact vocabulary lookup**
contract. It uses only the C standard library: no Python, vendored headers,
tokenizer library or model inference. This is not a full BPE tokenizer.

## Functions

```c
TokenizerStatus tokenizer_open(const char *model_path,
    const char *vocabulary_path, const char *config_path, Tokenizer **result);

TokenizerStatus word_to_id(const Tokenizer *tokenizer,
    const void *text, size_t length, uint32_t *id);

TokenizerStatus id_to_word(const Tokenizer *tokenizer,
    uint32_t id, TokenText *text);

void tokenizer_close(Tokenizer *tokenizer);
```

`word_to_id` finds one exact byte sequence and returns its ID. It does not split
sentences, run BPE, trim whitespace, change case or insert control tokens. Unknown
text returns `TOKENIZER_NOT_FOUND`; no approximate or invented ID is returned.

`id_to_word` returns borrowed immutable bytes and their length in `TokenText`.
They remain valid until `tokenizer_close`. A token may contain spaces, punctuation,
partial UTF-8 bytes or NUL, so use the length, not `strlen`. Undefined reserved
IDs return `TOKENIZER_NOT_FOUND`; they do not return placeholder labels.

`tokenizer_open` owns all loaded state; there are no mutable globals. Lookups do
not modify that context or call one another. Independent contexts are separately
allocated and can be closed independently. Future input/output numeric functions
can own separate contexts if needed; neither is implemented in this increment.

The loader cross-checks all163584base token-byte/rank entries against the binary
vocabulary. It reads16added-token spellings from tokenizer_config.json, including
configured entries not marked special. JSON escapes and surrogate pairs are
decoded. Duplicate IDs/text, invalid offsets, incomplete base vocabulary and
malformed data are rejected. Reserved IDs absent from the config stay undefined.

## Files

Source and development documentation are at
`/opt/clover-k3/clover-intelegence/code/client`. All compiled programs, client
datasets and configuration are exclusively inside its `bin` directory:

```text
client.c
test-client.c
README.md
CONTEXT.md
bin/
    client
    test-client
    test-client-sanitized
    dataset/inputs/seed.bin
    dataset/inputs/...existing metadata and reader files...
    dataset/outputs/fruit.bin
    dataset/outputs/...existing metadata and reader files...
    dataset/tiktoken.model
    dataset/vocabulary.bin
    configs/tokenizer_config.json
```

The C mapper's complete data/config dependencies are inside bin. It does not
depend on the source folder or any parent dataset/config paths at runtime. The
input and output numeric tables and every existing companion file are also
inside bin on AX102; the current exact mapper does not yet consume those tables.

The workspace has the same source, tests, vocabulary and config layout, with
Windows executables. Its [inputs](bin/dataset/inputs/) and
[outputs](bin/dataset/outputs/) folders contain location notes only, not the
large numeric payloads. The full package is on AX102.

All twelve previous client data/config compatibility links were removed with
explicit approval, including the older clover-data/direct-equation/k3model and
seed/fruit paths. Old experiment commands using those paths must be updated;
there are no replacement aliases. The runtime contains no symlinks. Its Linux
executable is now `code/client/bin/client`, not `code/client/client`.

## Build and Run

Build from the client source directory, then run from bin:

```bash
gcc -std=c11 -O2 -Wall -Wextra -Werror -pedantic client.c -o bin/client

cd bin
./client dataset/tiktoken.model dataset/vocabulary.bin configs/tokenizer_config.json word-to-id ' Paris'
# 17374

./client dataset/tiktoken.model dataset/vocabulary.bin configs/tokenizer_config.json id-to-word 40484
#  Tokyo

./client dataset/tiktoken.model dataset/vocabulary.bin configs/tokenizer_config.json word-to-id '[EOS]'
# 163585
```

ID-to-text mode accepts multiple IDs and concatenates their bytes without inserting
spaces or a final newline. It checks all IDs before printing, so an unknown ID
does not produce a partial decoded result. Text-to-ID mode requires one quoted
argument. CLI exit3means no exact entry; exit1means a load or IO failure;
exit2means invalid usage.

`Paris` and ` Paris` are different vocabulary strings. Not every complete word
has a single entry, and an arbitrary sentence will usually return NOT_FOUND.
This limit was explicitly selected instead of importing a third-party tokenizer.

## Verification

[test-client.c](test-client.c) runs without a test framework. From the client
source directory:

```bash
gcc -std=c11 -O2 -Wall -Wextra -Werror -pedantic test-client.c -o bin/test-client
./bin/test-client bin/dataset/tiktoken.model bin/dataset/vocabulary.bin bin/configs/tokenizer_config.json
```

Windows MinGW and Linux strict builds passed. All163600defined IDs round-tripped;
240undefined IDs were rejected. Known base/special IDs, binary token bytes,
independent context lifetimes, JSON escapes/surrogates, malformed JSON/Base64,
crossed vocabulary files and bad arguments were checked. The Linux test also
passed AddressSanitizer and UndefinedBehaviorSanitizer with leak detection enabled.
No model run, numerical-vector conversion or input/output data-content change occurred.

The first package move preserved all 30 original server files by full SHA256 and
device/inode/size/mtime checks before these documentation updates. The C source,
tests, executables and numeric payloads were moved, not regenerated. The local
move preserved all 10 existing files by hash/size/mtime; package-relative strict
compilation, whole-vocabulary tests and the CLI lookup passed on Windows.
Fresh AX102 strict builds and the unchanged sanitized test also passed using
package-internal paths. Both lookup directions and all five original data/config
alias chains passed at that stage. Verification builds used a temporary
directory, leaving the relocated executables unchanged.

The subsequent bin-only move preserved all 27 runtime files by full SHA256 and
device/inode/size/mtime before documentation updates. Eight local runtime files
retained hashes, sizes and mtimes. Unchanged whole-vocabulary tests and the CLI
passed from bin on both machines. All twelve approved aliases and all previous
runtime paths directly under client were absent after the move. C source, test
source, model parameters and historical evidence were not changed.
The relocated AX102 sanitizer passed with leak detection enabled, both CLI
directions passed, and strict source compilation remained warning-free.

The earlier request for7168-value input/output functions is not fulfilled by this
mapping increment. Those matrices are separate from text token spellings; no
fake embedding or inverse-word operation was added. See [CONTEXT.md](CONTEXT.md).