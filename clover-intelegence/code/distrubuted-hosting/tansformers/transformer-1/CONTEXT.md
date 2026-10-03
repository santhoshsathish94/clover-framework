# Transformer 1

## Direction

Create code/tansformers/transformer-1/transformer-1.c, with its executable and
trunk-1, root-1, operators/qkv-all/layer-1 dependencies beneath bin/dataset.
The spelling tansformers follows the human's requested path. The human explicitly
approved a C decoder for the existing compressed placement format, while retaining
direct consumption of precomputed values. No third-party runtime libraries.
Separate functions for each action; main is only a command-line coordinator.

## Established Context

Layer 1 is KDA plus routed and shared experts, not layer 0's dense MLP. Inputs
are incoming residual and S0 for each token, with sequence-owned KDA history.
Pre-attention aggregation uses S0 and incoming residual; attention residual adds
instead of replacing. Root constants contain precomputed BF16 values and maps;
experts.bin stores zlib-compressed placement codes/selectors. Decode that format,
then use stored values directly without recreating original checkpoint/scales or
full weight matrices. Do not substitute recorded observations for live computation.

Original client/server sources and historical tests remain unchanged. Datasets
will move unchanged with full hashes/identities and old model aliases retained,
as with the server. Large payloads stay AX102-only, local metadata documents them.
No new input/output/activation persistence; disable core dumps before tests.

## First Check

Implement a bounded zlib/DEFLATE reader separately from model arithmetic. Compare
stored/fixed/dynamic blocks against the standard Node zlib oracle and malformed
stream controls before integrating it. The output length and checksums must match;
no trailing frame, preset dictionary, invalid code or out-of-window reference
may be accepted. Then bind the real root-1 index/constants and layer-1 parameters.

## Decoder And Root Reader Passed

The C decoder passed 24 independent Node-zlib generated cases (stored, fixed,
dynamic, Huffman-only), exact bytes, checksum/truncation/extra-data/output bounds,
reserved-type and dictionary rejection. Strict compile caught an extra code-order
entry before runtime; corrected that table without changing the checks. Windows
binary stdio is explicit in the test driver.

The direct root reader then passed 27 real blocks spread across experts0/498/895,
all3matrices and first/middle/last blocks, checking original decoded-BF16 CRCs.
Complete expert0 zero-input projections passed all rows; address controls passed.
It loads the 66 stored constants, maps/template tables and small index only.
Root payload remains on disk; sequence scratch decodes one block, validates it,
then supplies stored values directly to the original 16-double-lane arithmetic.

derive-transformer.mjs pins the prior server source and mechanically copies
unchanged KDA primitives into standalone transformer-1.c. Layer-specific code
adds pre-attention fA aggregation, additive attention residual, live router/top16,
root projections, rank-ordered mixture and shared expert functions. Nineteen
additional trunk records are loaded; QKV uses the layer-1 operator file. No
runtime include or dependency on server.c; generation is development tooling.

## Layer 1 Comparison Passed

The independently compiled original C runs layer0 and layer1, with reference-only
expert projections sourced from authenticated historical observations. Every
reference projection requires matching layer/expert and complete input vector;
no permissive replay. The candidate never reads observations and computes live
router, top16, latent, gate/up/SiTU/down and shared outputs from stored parameters.

Both five-token France/Japan sequences passed all five final layer1 residuals
(7168F32 each), all16route IDs and S0 bytes. Reset-first-position and invalid-input
controls passed. The reference used original raw trunk values, not the candidate
prepared loader. The two original expert input/result file pairs were SHA256-pinned.
Core dumps disabled, no new vector/result files. This is layer1 correctness for
these sequences, not a full-model run or speed benchmark.

Proceed with only the requested three data-directory moves. Keep original
metadata and observations as historical files but never runtime lookup inputs.
Use separate move plan/journal, full SHA256, inode/size/mtime, active-user checks,
same-directory identity, and old model aliases. No client-alias restoration.

## Package Moves Passed

All forty original files moved: trunk-1 seven, root-1 thirty-one including its
historical observations, layer-1 operators two. Full SHA256 and device/inode/size/
mtime passed per group before advancing, with original directory identity retained.
All three old main-dataset paths now alias the package; earlier clover-data paths
resolve too. No internal dataset symlinks. Candidate sources/binary stayed unchanged.
From bin, transformer-1 --inspect (default dataset) loaded all dependencies.

Standalone sanitizer/CLI controls follow using the final paths, with core dumps
disabled and no runtime vector files. Source functions remain individually scoped;
main only selects inspect/stream and manages model lifetime.

## Final Package Validation

Standalone transformer ASan/UBSan with leak detection passed a full live zero-input
layer1 computation, nineteen records, distinct routes, state isolation/reset and
invalid inputs. Its first SSH attempt failed to connect before compilation; after
a separate CLI connection succeeded, the bounded retry passed. Packaged CLI passed
inspect, residual/S0 framing, EOF and malformed/nonfinite/missing-data controls.

Decoder oracle checks also passed on Linux under ASan/UBSan. The first per-case
SSH suite completed, but repeated connections made validation unnecessarily slow.
Changed only the test transport to one framed batch, preserving the 24 valid cases
and 118 rejection controls and requiring successful transport plus explicit result
statuses. This prevents an SSH failure being mistaken for expected decoder failure.
The batch passed on Windows and Linux sanitizers; runtime decoder code unchanged.

Local bin/transformer-1.exe compiled with strict warnings and standard C/math.
Local bin/dataset mirrors requested directories with notes and three unchanged
manifest/report copies only; the forty-file move inventory and transferred metadata
hashes were checked. No large numeric files downloaded. All numerical candidate
runs used live stored values, not recorded observations; no new runtime vectors
or crash reports created. Process core limit was zero throughout remote tests.

Functions are separated by action in transformer-1.c, root.h and decode.h;
main only owns CLI selection and lifetime. No later layer implemented, no new
training/preparation, no source changes to existing client/server, no commit/push.
Runtime calls sharing one model must be serialized because its root FILE cursor
is shared. This limitation is documented, not a thread-safety claim.

Final consistency checks passed candidate and independent-reference source
reproduction, sixty documentation links, eight source/metadata/build ignore
expectations and the ordered three-move journal. Editor diagnostics are clean;
reserved clover-one/context.md remains empty. Sync only source/tests and shared
documentation; do not overwrite original dataset companions with local location
notes. The requested layer-1 package is complete within the documented test scope.