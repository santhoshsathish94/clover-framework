# seed model integration

User direction: integrate the exact stored seed input table into the model and
test it. Work is isolated at `/opt/clover-k3/seed-integrated-20260929-a` on AX102.
The frozen final thirteen-item implementation, original checkpoint, seed file
and installed reference are unchanged.

## Data path

The native reader validates seed metadata, payload CRC and reconstructed-block
SHA256. It supports all six codecs in format version1. The model loads requested
  BF16 rows from seed, widens them into initial residuals, and gives the same seed
rows to the existing entry-normalization consumer. No source-table lookup is
used in either seed-enabled embedding path. One block is cached during loading;
the reader releases its index/block buffers afterwards, retaining prompt rows.

The shared checkpoint shard remains needed for output-head and global parameters.
Its embedding payload's page-aligned interior is protected with PROT_NONE after
mapping. Partial boundary pages remain readable because they also hold other
tensors. A child probe must receive SIGSEGV on a protected read. This protects
the process mapping only; it never modifies the source file. It is not a claim
that the process cannot access any other mapping or file descriptor.

## Gates

- Native reader warning-free compile, six codecs and all65536BF16 patterns.
- Thirty fixture checks, ten malformed/out-of-range cases rejected.
- Seven actual boundary/token rows matched the verified Python reader.
- Full native decode of2348810240bytes matched the original input-table SHA256
  without opening the checkpoint.
- Next: compile model hook, fresh reference and seed-disabled control, then
  seed-enabled France/Japan forward runs with full outputs/routes and every
  existing representation gate unchanged.

Native reader gates have passed. France passed fresh reference, prior combined
binary, new seed-disabled control and seed-enabled forward comparisons. The
seed-enabled run passed all thirteen existing stage gates, consumed35840input
values from five decoded blocks (811509compressed bytes), and retained71680BF16
row bytes. The original embedding mapping had2348806144bytes protected;4096bytes
on shared boundary pages remained readable. The child probe faulted as required.

Japan also passed its fresh reference and all thirteen combined-stage checks.
Its five rows required811642compressed bytes. Both full norm/logit outputs and
route traces match. A repeated France reference still matches, and a missing
seed path is rejected explicitly, without fallback.

## Final result

**PASS.** The final verifier compared the actual BF16 rows emitted by the
seed-enabled model with both the original tensor and the independent Python
seed decoder. All 71680 values across the two runs match exactly.

| Prompt | Input rows | Input values | Seed compressed bytes read | Full-output MD5 | Token |
|---|---:|---:|---:|---|---:|
| France | 5 | 35840 | 811509 | 23d162dcefb18211a7540ef12948f1eb | 17374 |
| Japan | 5 | 35840 | 811642 | 4b2a7b96fb6323e5639feced66f54a8e | 40484 |

For each prompt, all 7168 final norm values, all 163840 logits and the route
trace match the unchanged reference. All thirteen prior representation gates
remain active and passed. Their combined sample remains 1926 scale rows,
214 layer/expert pairs, 5778 code groups and 4320 projection comparisons.

The native-reader full-table pass independently reconstructed all 1174405120
input values from seed without opening the checkpoint. Invalid metadata,
payloads, shapes and row IDs were rejected; a missing seed file fails rather
than falling back to the original embedding table. Both model-run guard probes
received SIGSEGV as expected. Partial shared pages remain a stated guard limit.

The [final report](seed-integrated/results.json),
[tested model source](seed-integrated/model/candidate.c),
[reader test report](seed-integrated/reader-results.json), and
[usage instructions](seed-integrated/README.md) are preserved locally. The
[local-only verified archive](README.md#local-only-artifacts) contains the tested sources,
binaries and run evidence. Its SHA256 is
`fba4b7b5ca72ba167199a54620355bcf9ee21f4886608b438c6e7dbd6be3fe00`.

The final result SHA256 is
`cbc0d7cc4a72f7d019235cc1357062c7360960f7e0b431e33a8d762d9d7d4792`.
Installed source/binary/gate/index and the seed dataset hashes are unchanged.
The model uses a private OpenSSL static archive from
`libssl-dev 3.0.13-0ubuntu3.15`; no system package was installed.

This is a correctness integration, not a new speed or peak-RAM benchmark.
The 71680 retained BF16 input-row bytes per run are a buffer size, not process
RSS. Reference computations and other checkpoint tensors remain required;
the seed storage reduction is not a claim about whole-model memory savings.