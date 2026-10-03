# Final Normalization Stage

## Direction And Boundary

The human asks where eqidx.bin, leaves.json and model shard94 are used, and
requests a stage called normalization after transformer92, described as layer93.
Original clover-k3.c has93transformer layers numbered0..92. The tail is tagged93
for diagnostics, not another attention/expert layer. Its jobs are snapshot
aggregation and final RMSNorm followed by a separate output-head projection.

eqidx.bin is the original program's whole-model offset/shape index, used at
startup and across the model. Shard94 contains input embeddings, LM head and
three global tail tensors. leaves.json already stores those three tail tensors
losslessly as BF16/base64: output_attn_res_norm, output_attn_res_proj and norm.
The original C reads them from shard94; the new stage will read existing leaves.
No need to copy the index or4.70GBshard into a normalization-only package.

Implement code/normalization/normalization.c, owned immutable parameters, separate
load/decode/fold/score/softmax/aggregate/RMSNorm/protocol functions. Standard C/math
only, no third-party runtime. Input is last-position layer92residual plus8snapshots
oldest first; output is7168normalized floats. Do not implement LM head/argmax or
token mapping in this stage, or claim client integration. Fixed leaf values are
decoded once, live aggregation and normalization still compute per input.

Use the tested client JSON/Base64 helper code as an independent copied header,
not a runtime dependency on client.c. Validate actual leaf schema and lengths.
First check compiles the loader against actual leaves; then compare original
tail computation and independently read source tensor bytes. No new vector files
or core dumps. Keep original client/server/transformers and historical tests intact.
Package leaves under normalization/bin/dataset; retain old model aliases after
verification. All other main datasets remain untouched.

## Leaves Loader Passed

The actual58615-byte leaves file loaded successfully under strict Windows/Linux
compilation. The C parser requires exact format, endian, encoding, tensor count,
payload size, all three tensor names, dtype, shape and decoded length; rejects
duplicate required fields and nonfinite values. It copies existing tested JSON/
Base64 helpers without modifying client.c. Full JSON SHA and payload/source checks
are separate verification gates, not implemented cryptography in the runtime.

Added individual source scoring, softmax, ordered weighted aggregation and final
gain-before-inverse RMSNorm functions. Each call requires8snapshots plus residual
and returns only the normalized7168vector. Immutable loaded model owns the fixed
gains/fold; working results are per call. Main dispatches inspect/stream only.

Reference generation first stopped before execution because its source anchor
expected LF but the hash-verified local harness retained CRLF. Normalize text
line endings only after original byte-hash verification, preserving arithmetic
and comparisons. No candidate/runtime/data change was required.

## Original Tail Comparison Passed

Both France/Japan five-token reference sequences passed final normalized vectors
at all five positions:35840float32values per case,71680total. The last position
matches the original generation tail; earlier positions are extra causal checks.
Every one of21504BF16leaf parameter values compared exactly against shard94 through
original mrec1..3 offsets, and the prepared fold matched too. The reference uses
original all-layer arithmetic and exact-input-bound historical expert observations
as TEST-ONLY oracle, as earlier transformer validation; no candidate reads them.
No logits, normalized-vector or activation files were written. No end-to-end
standalone pipeline or output-head integration is claimed.

Proceed with moving only leaves.json into normalization/bin/dataset, fullSHA256
and device/inode/size/mtime checks with old model aliases retained. eqidx and the
large shared shard stay in main dataset, because this runtime opens neither.

## Leaves Move Passed

The original leaves.json moved to normalization/bin/dataset/leaves.json with
unchanged full SHA256 and device/inode/size/mtime. Both old main-dataset and
clover-data paths now resolve to the new real file. Candidate source, JSON helper
and executable hashes were unchanged; bin-local --inspect passed. Shared index
and shard identities were checked before/after and unchanged. New schema/CLI
controls check malformed data and vector framing without altering real datasets.

## Packaged CLI And Schema Checks Passed

The deployed CLI passed inspect, two nine-vector bundles yielding two7168value
lines, emptyEOF, malformed/incomplete bundles and nonfinite rejection with exact
exit codes. Separate schema test passed15shape/type/format/name/framing/Base64/
integer and nonfinite-parameter controls underASan/UBSan with leak detection.
All malformed file variants existed only in memory; the real leavesfile unchanged.
The sanitizer command was moved to background by the terminal tool and completed
successfully; no running test remains. Next finish local build and shared docs.

## Completed Package

The identical leaves.json and small move/reference records were downloaded into
the local package. Strict Windows executable build and the same CLI checks passed.
All three decoded leaf payload hashes match their original identities; the two
reference-result records, move journal and helper/reference source reproduction
checks passed. Original clover-k3.c remains hash-identical and reserved context
is empty. All62documentation links resolved before adding the result-log link.

Normalization runtime is normalization.c/json.h compiled into bin/normalization
(normalization.exe on Windows) plus bin/dataset/leaves.json. Final docs record
the role of shared eqidx and shard94, which were not moved. No LM-head, output
token function or end-to-end pipeline is delivered by this stage. Reference
numerical coverage is AX102 only; Windows covers build/loader/CLI behavior.
No background test remains, no new input/output/activation files, no commit/push.