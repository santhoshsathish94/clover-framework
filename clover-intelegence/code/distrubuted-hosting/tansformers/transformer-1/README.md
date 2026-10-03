# Transformer 1

[transformer-1.c](transformer-1.c) implements layer 1 only: KDA attention,
live routing, sixteen selected experts, and the shared expert MLP. It uses the
existing precomputed values with a bounded C decoder for their placement format.
The runtime needs only standard C/math and platform 64-bit file I/O. No zlib,
OpenMP, Python, Node or third-party runtime library is linked into the transformer.

## Layout

```text
tansformers/
  transformer-1/
    transformer-1.c
    root.h
    decode.h
    bin/
      transformer-1
      dataset/
        trunk-1/
        root-1/
        operators/qkv-all/layer-1/
```

All three real dataset directories are inside bin on AX102. Forty original files
moved unchanged, with full SHA256/device/inode/size/mtime and directory-identity
checks. Both old model path families remain compatibility aliases. The local
workspace has source, a Windows executable and small metadata only, not the large
numeric payloads. [Dataset locations](bin/dataset/README.md) distinguish them.

## Separate Functions

`main` only validates arguments, opens the model, dispatches inspect/stream and
closes it. Each computational action has its own function:

| Action | Functions |
|---|---|
| Parameter loading and ownership | `transformer_open`, `transformer_load_qkv`, `transformer_load_trunk`, `transformer_close` |
| Sequence state | `transformer_sequence_create`, `transformer_sequence_reset`, `transformer_sequence_close` |
| Aggregation and normalization | `transformer_aggregate`, `transformer_inverse_rms`, `transformer_normalize` |
| Prepared matrix projection | `transformer_project`, `transformer_project_row` |
| Q/K/V and convolution | `transformer_qkv`, `transformer_convolve`, `transformer_sigmoid`, `transformer_l2_heads` |
| KDA state and output | `transformer_decay`, `transformer_update_attention`, `transformer_attention_output` |
| Routing and expert work | `transformer_route`, `transformer_expert`, `transformer_mix_experts`, `transformer_shared`, `transformer_activation` |
| Layer coordination | `transformer_moe`, `transformer_process` |
| CLI protocol | `transformer_receive`, `transformer_send`, `transformer_stream`, `transformer_inspect` |
| Root stored-value access | `root_open`, `root_block`, `root_project`, `root_crc`, `root_close` in [root.h](root.h) |
| Placement decoding | `decode_bits`, `decode_tree`, `decode_symbol`, `decode_tables`, `decode_adler`, `decode_zlib` in [decode.h](decode.h) |

There is no mutable global model or sequence state. Fixed values remain loaded
until close. Each sequence owns its KDA state, raw convolution history and working
buffers. The model owns one seekable root file: serialize calls sharing a model,
including calls on different sequences. Close sequences before their model.

## Precomputed Values

The layer-1 operator supplies input gains, Q/K/V coefficient IDs, row scales and
convolution taps. Nineteen additional prepared trunk records supply postnorm,
attention G/O, KDA write/decay, router/bias, latent projections/norm, shared MLP,
prepared fA/fM and prepared exp(ALOG): IDs 5,6,7,11,12,13,18,19,29-39.
No layer-0 dense matrices or MLA parameters are loaded.

Root-1 has 66 precomputed BF16 values, maps and templates. Its expert placement
codes/selectors are compressed. The decoder unpacks one bounded block and verifies
the zlib frame and original decoded-value CRC; arithmetic directly reads those
stored values. It does not reconstruct original checkpoint tensors, scale arrays
or full expert matrices, and does not rebuild the old pair-value tables.

The implementation evaluates experts for the current input. Existing observations
move only as historical evidence and are never a runtime fallback. Router ties
prefer lower expert IDs; the sixteen outputs mix in original rank order.

## Interface

```c
int transformer_open(const char *dataset, Transformer **result);
void transformer_close(Transformer *transformer);
TransformerSequence *transformer_sequence_create(const Transformer *transformer);
void transformer_sequence_reset(TransformerSequence *sequence);
void transformer_sequence_close(TransformerSequence *sequence);
int transformer_process(Transformer *transformer, TransformerSequence *sequence,
    const float input[7168], const float snapshot[7168], float output[7168]);
```

Input is the previous layer's residual plus original S0, not a token ID. The
layer preserves S0 and returns its new residual. Process successive token positions
on the same sequence; reset between independent requests. Caller buffers must be
valid full arrays and must not alias internal state. A failure after computation
starts poisons that sequence until reset; rejected nonfinite inputs do not advance it.

## Build And Run

From the transformer-1 source directory on AX102:

```bash
mkdir -p bin
ulimit -c 0
gcc -std=c11 -O2 -Wall -Wextra -Werror -pedantic \
  -ffp-contract=off -fno-fast-math transformer-1.c -lm -o bin/transformer-1
cd bin
./transformer-1 --inspect
./transformer-1 --stream
```

The default dataset directory is `dataset`, relative to the current directory;
an explicit directory can follow either command. Standard math is linked with
`-lm`. Preserve IEEE binary32/binary64, nearest rounding and gradual underflow.
Do not enable fast-math or reassociation; only explicit `fmaf` projections fuse.
Disable process core dumps so crash handlers do not persist vectors.

Stream input is 7168 floats of residual followed by 7168 floats of S0 per token,
with whitespace separators. Output is two hex-float lines of 7168 entries each:
new residual followed by unchanged S0. EOF closes the sequence. Incomplete or
invalid input exits nonzero without emitting that pair; earlier successful pairs
have already been delivered. It is a stdin/stdout process, not a network listener.
No client integration or later-layer generation is included.

## Evidence And Limits

- [test-decode.mjs](test-decode.mjs) compares 24 stored/fixed/dynamic cases against
  Node's independent zlib output; checksum, framing, bounds, dictionary and reserved
  block rejection controls pass locally and with Linux ASan/UBSan. The final batch
  checks 24 valid cases and 118 explicit rejections in one process; SSH errors
  cannot count as decoder rejections. The C implementation has no zlib dependency.
- [test-root.c](test-root.c) checks 27 real blocks across three experts, all three
  matrices and first/middle/last blocks, plus complete expert-0 zero-input projections.
- [test-reference.c](test-reference.c) compares layer-1 residuals, all routes and S0
  for both existing five-token sequences. All five positions and reset controls pass.
  The reference uses hash-pinned original layer code and original raw trunk data;
  retired original expert tensors are replaced only in the reference by authenticated
  observations with exact full-input matches. The candidate computes experts live.
- [test-transformer.c](test-transformer.c) passed standalone ASan/UBSan with leak
  detection, full zero-input layer-1 computation, live routes, state/reset and invalid
  inputs. No original all-layer code is linked into this standalone test.
- [test-cli.mjs](test-cli.mjs) passed packaged inspect, vector-pair framing, zero
  residual/S0 output, EOF and malformed/nonfinite/missing-data checks over SSH.

These are layer-1 checks, not a new full-model run, arbitrary-input proof or speed
benchmark. Windows strict compilation is checked; real numeric data tests run on
AX102. The reader checks structural contracts, lengths, finite constants and root
checksums, not cryptographic authenticity at every inference. The full relocation
hashes bind the actual packaged files; keep them immutable while the model is open.

[derive-transformer.mjs](derive-transformer.mjs) mechanically derives the common
KDA functions from the hash-pinned earlier server source with layer-1 additions
in [moe.inc](moe.inc) and [cli.inc](cli.inc). It verifies existing output rather
than overwriting manual changes. This is development tooling, not a runtime
dependency. [prepare-reference.mjs](prepare-reference.mjs) and
[test-reference.sh](test-reference.sh) build the separate original-code test oracle.
The original server, reference source and existing tests are unchanged.

[MOVE-PLAN.tsv](MOVE-PLAN.tsv), [MOVE-JOURNAL.tsv](MOVE-JOURNAL.tsv) and
[move-datasets.sh](move-datasets.sh) record the requested three-directory move.
[CONTEXT.md](CONTEXT.md) records the choices, observed results and remaining limits.