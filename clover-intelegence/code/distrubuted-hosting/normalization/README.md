# Final Normalization

[normalization.c](normalization.c) implements the aggregation and normalization
after transformer 92. The original program labels its tail diagnostics as 93;
there are still 93 transformer layers numbered 0 through 92, not another
attention/expert layer here. Standard C and standard math only, no third-party
runtime libraries. Each action has its own function; main only selects the CLI
mode and manages the loaded parameters.

## Data Ownership

| File | Role |
|---|---|
| leaves.json | Three global BF16 parameter tensors for tail aggregation and final RMSNorm. The only runtime data needed by this package. |
| eqidx.bin | The original whole-model index of parameter shapes, offsets and file paths. Not specific to stage 93 and not opened by normalization.c. |
| model-00094-of-000096.safetensors | Original shard containing input embeddings, the LM head and the three global tail tensors. Remains shared; normalization.c uses the existing leaf extraction instead. |

The original reference reads mrec[1..3] from shard 94. This stage reads those
same values from the existing leaves JSON, without regenerating parameters.
It decodes 21,504 BF16 values and prepares the aggregation fold once when opened.
Live source scores, mixing weights and normalized output still depend on input.

```text
normalization/
  normalization.c
  json.h
  README.md
  CONTEXT.md
  bin/
    normalization
    dataset/
      leaves.json
```

The 58,615-byte leaves file moved unchanged into this package on AX102. Its old
main-dataset and clover-data paths remain aliases. The shared index and shard
were not moved or changed. An identical small leaves file is also included locally;
the large shared checkpoint is not downloaded.

## Functions

```c
int normalization_open(const char *path, Normalization **result);
void normalization_close(Normalization *model);
int normalization_process(const Normalization *model,
    const float residual[7168], const float *snapshots,
    unsigned snapshot_count, float output[7168]);
```

Pass the selected token position's layer-92 residual and eight contiguous
snapshots, oldest first: S0, S12, S24, S36, S48, S60, S72, S84. The stage scores
those eight sources followed by the residual, mixes them in original order, and
returns one 7,168-value normalized vector. For next-token prediction the caller
selects the final token position, as the original tail does.

Separate functions handle schema parsing, Base64/BF16 decoding, fixed fold
preparation, `normalization_score_sources`, `normalization_softmax`,
`normalization_aggregate`, `normalization_inverse_rms` and
`normalization_apply_rms`. `normalization_receive`, `normalization_send` and
`normalization_stream` handle the protocol. There is no shared mutable runtime
state or persistent sequence cache; working results are local to each call.
Keep the model alive during calls. Caller buffers must be full-sized valid arrays
outside the model; a failed call leaves the output unchanged.

## Build And Run

From the normalization source directory on AX102:

```bash
mkdir -p bin
ulimit -c 0
gcc -std=c11 -O2 -Wall -Wextra -Werror -pedantic \
  -ffp-contract=off -fno-fast-math normalization.c -lm -o bin/normalization
cd bin
./normalization --inspect
./normalization --stream
```

The default data path is dataset/leaves.json relative to the working directory;
an explicit path may follow either option. `-lm` is the platform's standard math
library. Preserve nearest rounding, IEEE binary32/binary64 and gradual underflow;
do not enable fast-math or fused contraction. Disable core dumps before execution
so crash handlers do not persist process input memory.

Stream input is one residual vector followed by eight snapshot vectors, each
7,168 whitespace-separated floats. Each complete bundle produces one line of
7,168 hexadecimal floats. EOF finishes cleanly; malformed/incomplete input exits
2, invalid numeric input or processing failure exits 1. No result is emitted for
the failed bundle. Earlier completed bundles have already been delivered.

There is no network listener, output-head projection, argmax or text decoding in
this stage. The normalized vector is the input to the separate LM-head step.
The current client still lacks its numeric output-head function; no end-to-end
client/server/transformer/normalization integration is claimed.

## Verification

- The original tail comparison passed both existing five-token sequences at all
  five positions: 71,680 normalized float32 values exactly matched. Checking earlier
  positions is additional coverage beyond the original last-position tail.
- All 21,504 decoded leaf values matched independent reads from the original
  shard through mrec[1..3]; the fixed aggregation fold also matched.
- [test-normalization.c](test-normalization.c) passed standalone ASan/UBSan with
  leak detection, zero-input output, independent model lifetimes, invalid counts,
  nonfinite inputs and unchanged outputs on failure.
- [test-schema.c](test-schema.c) passed 15 in-memory schema, shape, tensor-name,
  framing, Base64, nonfinite-parameter and integer rejection checks with sanitizers.
- [test-cli.mjs](test-cli.mjs) passed two bundles, exact zero-output framing, EOF
  and explicit rejection codes against both the deployed Linux CLI and a strict
  Windows build using the identical small leaves file. Real reference-sequence
  numerical comparisons ran on AX102, not Windows.

[prepare-json.mjs](prepare-json.mjs) copies existing project JSON/Base64 helpers
without changing client.c. [prepare-reference.mjs](prepare-reference.mjs),
[reference-check.inc](reference-check.inc) and [test-reference.sh](test-reference.sh)
use the hash-verified original-layer test oracle. Its upstream expert calculations
use exact-input-bound historical observations only in the reference; normalization
itself receives no expected outputs. This is tail equivalence, not a fresh run of
retired expert tensors or a chained standalone pipeline. No vectors were persisted.

The runtime checks schema and finite values, not cryptographic payload authenticity.
The unchanged leaves SHA256 is checked separately by the test/move scripts. The
JSON's source offsets and hashes are provenance, not extra runtime dependencies.
All three decoded Base64 tensor payloads were independently SHA256-checked locally;
the [recorded tail results](reference-results.log) contain both successful cases,
not vector payloads.
The [move plan](MOVE-PLAN.tsv) and [journal](MOVE-JOURNAL.tsv) record the original
identity; [move-leaves.sh](move-leaves.sh) verified full SHA256/inode/size/mtime
and both old alias chains. See [CONTEXT.md](CONTEXT.md) for outcomes and limits.