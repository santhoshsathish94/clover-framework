# Layer 0 Server

[server.c](server.c) implements layer 0 only, with separate functions for each
job and explicit model/sequence ownership. It uses the C standard library and
standard math functions, no third-party libraries, Python, OpenMP or model engine.
It is a stdin/stdout program and C API, not a network listener.

## Data

Only one layer-0 QKV operator payload is retained: bin/dataset/trunk-0-qkv/qkv.bin.
The identical main-dataset qkv-all/layer-0 copy was removed with explicit approval,
along with that directory and its legacy alias. Its distinct provenance manifest
is preserved as qkv-all-layer-0-manifest.json beside the canonical payload.
The server's original qkv.bin and manifest.json are unchanged; --inspect passed.
Old commands using the retired duplicate path need the canonical path below.
The [removal plan](QKV-DEDUP-PLAN.tsv) and [journal](QKV-DEDUP-JOURNAL.tsv) record
the exact identities; [remove-duplicate-qkv.sh](remove-duplicate-qkv.sh) retains
the guarded cleanup procedure. Earlier move inventories remain historical.

On AX102:

```text
clover-intelegence/
  code/server/
    server.c
    bin/
      server
      test-server-reference
      test-server-sanitized
      dataset/
        trunk-0-qkv/
          qkv.bin
          manifest.json
        trunk-0/
          index.bin
          common.bin
          kda.bin
          dense.bin
          build-values.json
          manifest.json
```

The QKV folder was moved here unchanged. Its previous main-dataset path is now
a compatibility symlink, and the older clover-data operator path still resolves.
Both datasets now live under server/bin/dataset, alongside the compiled programs
in bin, matching the client runtime layout. No sibling server/dataset directory
remains. Old model aliases point directly into bin; they do not duplicate data.
The server has no separate runtime config file.
No client aliases were restored. External raw data is used only by the reference
test, not by server.c.
See [dataset locations](bin/dataset/README.md).

The QKV file provides input-normalization gains, coefficient palette, Q/K/V
coefficient IDs, row scales and convolution taps. The reader loads only these
additional prepared record payloads from trunk-0:

| IDs | Used For |
|---|---|
| 5 | Post-attention normalization gains |
| 6, 7 | Attention gate and output projections |
| 11, 12, 13 | KDA write and decay projections |
| 18, 19, 39 | Decay offset, head normalization, prepared exp(ALOG) |
| 26, 27, 28 | Dense gate, up and down projections |
| 38 | Prepared post-attention aggregation fold fM |

Layer 0 has no MLA, routed experts, router or shared-expert MLP. Pre-attention
aggregation is bypassed, so no fA calculation or consumption is needed. It does
have KDA recurrent attention and a dense MLP; the smaller QKV file alone does not
represent those remaining operations. No new data preparation is performed.

## Functions

```c
int server_open(const char *qkv_path, const char *trunk_directory, Server **result);
void server_close(Server *server);
ServerSequence *server_sequence_create(const Server *server);
void server_sequence_reset(ServerSequence *sequence);
void server_sequence_close(ServerSequence *sequence);
int server_process(const Server *server, ServerSequence *sequence,
    const float input[7168], float output[7168], float snapshot[7168]);
```

`server_open` loads fixed data once. A sequence owns raw Q/K/V history and KDA
recurrent state; successive calls process tokens in causal order. Reset starts a
new sequence, not another model load. Close sequences before their owning server.
Use one sequence per request; calls on a shared sequence must be serial.

Input is one raw embedding vector, not a token ID or normalized vector. Output
is the layer-0 residual. Snapshot is the original embedding S0, which later layers
also need. Callers retain each returned pair for its token position. Output and
snapshot must be separate, nonoverlapping caller-owned arrays; do not pass internal
state buffers. The CLI does not execute layer 1, the output head or token decoding.

Individual jobs are implemented by `server_normalize`, `server_project_row`,
`server_project`, `server_convolve`, `server_sigmoid`, `server_l2_heads`,
`server_qkv`, `server_decay`, `server_update_attention`, `server_attention_output`,
`server_aggregate`, `server_dense_activation` and `server_dense`. No mutable
global model or sequence state. History and results stay in memory.

## Build And Run

From `/opt/clover-k3/clover-intelegence/code/server`:

```bash
mkdir -p bin
ulimit -c 0
gcc -std=c11 -O2 -Wall -Wextra -Werror -pedantic \
  -ffp-contract=off -fno-fast-math server.c -lm -o bin/server

cd bin
./server --inspect dataset/trunk-0-qkv/qkv.bin dataset/trunk-0
./server --stream dataset/trunk-0-qkv/qkv.bin dataset/trunk-0
```

Disable core dumps before runtime/tests so a host crash handler cannot persist
input or sequence memory. The server itself does not create runtime data files.

`-lm` links the platform's standard C math implementation, not a third-party
library. `-ffp-contract=off` preserves separate operations; explicit `fmaf` calls
retain the original 16-lane projection rounding. Do not use fast-math. The tested
environment uses IEEE binary32/binary64, nearest rounding and gradual underflow.

Stream input is whitespace-separated numeric values, exactly 7168 per token.
For each complete vector, stdout contains two lines of 7168 hexadecimal floats:
first the residual, then S0. EOF ends the sequence. Invalid or incomplete input
exits nonzero without emitting that vector's result; prior completed responses
are already delivered. Nothing is written to input, output or activation files.
The current client has not yet implemented embedding-vector output or transport
to this server; no end-to-end client/server integration is claimed.

## Verification

The reader validates exact file sizes, headers, layout, record IDs/shapes, offsets,
palette and finite stored constants. It does not authenticate payload SHA256
inside server.c; use the preserved dataset identities and verification scripts
for that check. Model files must remain immutable while loaded.

- [test-reference.sh](test-reference.sh) verifies all five dataset hashes and
  compares against the hash-pinned original code, stopping after layer 0.
- [prepare-reference.mjs](prepare-reference.mjs) mechanically retains the original
  executed code; it does not rewrite the original reference or its arithmetic.
- [test-server-reference.c](test-server-reference.c) passed both existing France
  and Japan five-token sequences: all residual and S0 values exact, plus sequence
  isolation, reset and invalid-input controls. No intermediate equality required.
- [test-server.c](test-server.c) passed standalone AddressSanitizer and
  UndefinedBehaviorSanitizer with leak detection, primitive checks and two zero-input
  steps. The original all-layer reference is not linked into this test.
- [test-stream.mjs](test-stream.mjs) passed actual CLI framing, zero-vector output,
  EOF, malformed/truncated/nonfinite input and empty-dataset rejection over SSH.

Node and the original reference's OpenMP are validation tooling only. No new
full-model run or latency/throughput comparison was performed. Broader inputs,
long-running sequences and end-to-end generation remain unverified. Windows
source compilation passed; real-data numeric tests ran on AX102, not Windows.

[QKV-MOVE-PLAN.tsv](QKV-MOVE-PLAN.tsv) and
[QKV-MOVE-JOURNAL.tsv](QKV-MOVE-JOURNAL.tsv) record the two-file same-filesystem
move. Full SHA256/device/inode/size/mtime checks passed, as did both old alias
chains. [move-qkv.sh](move-qkv.sh) provides the guarded plan/move/verify steps.
[CONTEXT.md](CONTEXT.md) records the implementation, corrected record-5 binding,
evidence and limits. Existing numeric data, original code and tests were preserved.

The later complete trunk-0 move is recorded by
[TRUNK0-MOVE-PLAN.tsv](TRUNK0-MOVE-PLAN.tsv) and
[TRUNK0-MOVE-JOURNAL.tsv](TRUNK0-MOVE-JOURNAL.tsv). All six files retained their
SHA256 and device/inode/size/mtime; directory identity also matched. Both old
trunk paths resolve to the same files. The unchanged compiled server passed
`--inspect` with both datasets under server/dataset. No numerical rerun or rebuild
was needed for this path-only move. [move-trunk-zero.sh](move-trunk-zero.sh) holds
the guarded plan/move/verify operations.

The final dataset-to-bin move is recorded by
[BIN-DATASET-PLAN.tsv](BIN-DATASET-PLAN.tsv) and
[BIN-DATASET-JOURNAL.tsv](BIN-DATASET-JOURNAL.tsv), using
[move-dataset-to-bin.sh](move-dataset-to-bin.sh). All nine original dataset files
retained full hashes and file identities before the dataset note was updated.
The unchanged binary passed --inspect from inside bin. Both existing model aliases
were retargeted; server/dataset was not replaced by an alias. The three local
metadata files moved into bin/dataset as well; large payloads remain AX102-only.
Earlier move plans and folder-tree snapshots describe their original locations,
not the current runtime layout. No new numerical comparison or rebuild was needed.