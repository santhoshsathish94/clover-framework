# Layer 0 Server

## Remaining QKV-All Layer 0 Duplicate Confirmed

The human asks whether main-dataset operators/qkv-all/layer-0 should belong with
the server and whether it supplies additional useful values. Fresh AX102 full
SHA256 and cmp comparison confirms its operator.bin is byte-identical to the
server's bin/dataset/trunk-0-qkv/qkv.bin: both265008216bytes, SHA256
4925ea619a4fd268b441e8008e4c1cd714f681b328b03cccd52c69b6a9b6f654.
Both contain the same input gains, coefficient palette/IDs, row scales, convolution
taps and arithmetic contract. These are fixed operator parameters, not completed
input-dependent Q/K/V vectors. Server already reads them; a second copy adds no
new computation or parameter coverage.

The two manifests differ in provenance: first construction from prepared trunk0,
later all-layer construction from original runtime trunk slots. Preserve both
histories if consolidating. Logical owner is server/bin/dataset, matching the
current package layout, not a recreated server/dataset sibling. This was a question
and read-only payload comparison, not authorization to delete a duplicate or alter
aliases. Neither payload nor manifest was moved, deleted or changed. A future
consolidation can retain one canonical payload and preserve both provenance records.

## Duplicate Removal Authorized

The human now requests deleting the redundant copy to remove confusion. After
the exact folder and link were inspected, they explicitly selected removing the
folder AND its one legacy link, accepting that old qkv-all/layer-0 commands stop
working. This retires only main-dataset operators/qkv-all/layer-0 and the older
clover-data/operators/qkv-all/layer-0 link; all other model aliases remain.

Fresh full-hash/cmp checks match both265008216-byte payloads, with no active users.
The canonical file and duplicate have different inodes and each has one link.
Preserve the duplicate's distinct manifest by moving it into canonical
server/bin/dataset/trunk-0-qkv/qkv-all-layer-0-manifest.json, then unlink only the
duplicate payload, remove its empty directory and unlink the exact legacy alias.
The canonical qkv.bin and original manifest must remain unchanged. Separate
QKV-DEDUP-PLAN/JOURNAL.tsv record identities and operations; no historical plan
rewrites or server arithmetic changes. Verify by hashes and server --inspect.

## Duplicate Removal Completed

The approved redundant operator.bin was unlinked, its now-empty layer-0 directory
removed, and only the exact legacy clover-data/operators/qkv-all/layer-0 symlink
unlinked. Both old paths verified absent. The1863-byte provenance manifest was
first moved unchanged beside the canonical qkv.bin under its distinct filename;
its SHA256/inode/size/mtime matched. Canonical payload and original manifest both
retained hashes and identities. Source and executable hashes stayed unchanged.
The server passed --inspect with its canonical bin-local datasets afterward.

Removed265008216logicalduplicatebytes; the original inventory now has3399files/
1,387,808,622,300bytes, retaining both provenance records. This is not a measured
whole-filesystem free-space delta. The duplicate had a separate inode and one
hardlink before deletion. No other data or aliases were removed or recreated.
QKV-DEDUP-PLAN/JOURNAL.tsv and a small provenance copy were downloaded locally;
there was no local numeric duplicate to delete. Historical inventory/check scripts
that require the retired paths are no longer current verification commands.
Their assertions and records were not weakened or rewritten. No model computation,
rebuild, commit or push was needed for this cleanup.

## Direction and Evidence

The human requests server/server.c with individual functions for each job, no
third-party libraries, and operators/trunk-0-qkv moved into server/dataset.
They ask to distinguish common all-layer code from actual layer-0 dependencies
and point to existing clover-intelegence/dataset/trunk-0 for remaining values.

Read the original clover-k3.c L=0 branch, qkv-operator.c, the prepared reader and
both real AX102 manifests. Layer 0 skips MLA, MoE, pre-attention AR and the unused
fA fold. It still executes KDA beta/decay/state/output and the dense MLP after
two-source pre-MLP aggregation. These values already exist in trunk-0, not in
the smaller QKV operator file. No missing dataset needs to be reconstructed.

Use the QKV dataset for input gains, Q/K/V coefficients, row scales and raw-history
convolution. The remaining runtime uses thirteen prepared records: POST_LN, G, O,
B, FA, FB, DTB, ONORM, MGATE, MUP, MDOWN, prepared fM and prepared exp(ALOG).
Do not read or compute later-layer router/shared-expert/MLA work. Standard C and
its math functions are allowed; no OpenMP, external engine or third-party library
in server.c. Keep explicit instance-owned model and sequence state.

The requested QKV directory will move to code/server/dataset/trunk-0-qkv on
AX102 with full hashes and file identity verification. Leave the other trunk-0
data at its existing location and pass that directory explicitly. Preserve old
QKV paths as aliases: the earlier removal approval applied only to twelve client
aliases, not this operator used by other experiments. Do not move other datasets
or download large payloads locally without further direction.

## First Check

First implement the independent C dataset reader and inspect both actual formats.
Its exact header/shape/bounds/palette checks can disconfirm the selected thirteen
records before arithmetic is added. Then validate the layer-0 final result against
the original layer-0 code, without persisting input/output/activation datasets.
No network transport or client vector implementation is implied by server.c.

## Reader Passed

Strict Windows/Linux C11 compilation passed. The real AX102 datasets loaded:
QKV plus the thirteen specified record IDs, with expected shapes and file sizes.
No numeric computation or dataset move was required for that check. Added separate
functions for RMSNorm, palette projection, convolution/history, SiLU, Q/K L2,
beta/decay, KDA recurrence, attention output, two-source pre-MLP aggregation,
dense activation/projections and residual addition. Each sequence owns state,
and each model instance owns fixed values. No all-layer loop or MoE/MLA code.

Standard C fmaf preserves the original sixteen float accumulators and fixed
reduction; compile with -ffp-contract=off so other operations do not fuse. Standard
math functions require -lm on Linux, not a third-party library. The command-line
stream reads one whitespace-separated 7168-float vector at a time and returns
two hex-float lines: residual then original S0. No runtime file writes. It is
a stdin/stdout process, not an invented network protocol or transport service.

## First Numerical Check Found A Binding Error

The original-versus-server layer-0 residual comparison failed all 7168 values
at the first position. Read the original slot enum: POST_LN is 5, not 1. The
first reader mistakenly selected ARP (1), also 7168 floats, so shape checks alone
could not detect semantic misbinding. Corrected selected/read POST_LN to 5;
the same full residual assertion remains unchanged. No numerical tolerance or
reference edits. Derive record roles from the actual enum, not position guesses.

The test abort reported core handling; no core file exists in the server work
directory. Host core policy routes to apport. Added a test-process-only zero
core limit before rerunning, without host configuration or assertion changes.

## Layer 0 Numerical Comparison Passed

Corrected POST_LN binding passed the unchanged original-reference test on both
five-token France/Japan sequences. Each of five layer-0 residuals and original
S0 snapshots matched byte-for-byte, with independent sequence and reset-first-token
controls and invalid-input checks. Inputs came from the original embedding
lookup in memory; reference stops before layer 1 and tail. This is layer-0 output
verification, not a new full-model/token-output test or performance comparison.

The reference harness pins original clover-k3.c SHA256
5628f7c3d2b7932b2b8776dcb498d54bbc730b1785ba7819989bdb51b6604e65,
preserves all executed layer arithmetic and compares using separate original raw
trunk values. The prepared index and all three payloads plus QKV hashes were
verified before tests. Reference OpenMP is test-only; server.c still uses only
standard C/math and no OpenMP. No new activation or output dataset was saved.

Proceed with the requested QKV-only directory move using move-qkv.sh. Same-device
rename, full hashes, unchanged inode/size/mtime, active-reader checks and old-path
aliases; other prepared trunk data stays in place. No client aliases restored.

## QKV Move Passed

Both qkv.bin and manifest.json moved to code/server/dataset/trunk-0-qkv with
identical full SHA256 and device/inode/size/mtime. The former main-dataset path
is now an alias; the older clover-data operator alias resolves through it. No
internal QKV symlinks, regenerated payload or other moved trunk data. Record:
QKV-MOVE-PLAN.tsv and QKV-MOVE-JOURNAL.tsv beside server.c.

Add standalone sanitizer/primitive and CLI framing/error checks against the
relocated paths. These test files contain no third-party libraries; Node is a
validation driver only, never a requirement of server.c. Runtime output remains
stdout/in-memory, with no new prompt/activation/result files.

## Final Validation And Crash-Report Correction

The standalone ASan/UBSan test passed with leak detection, real relocated data,
primitive controls, two zero-input steps, reset and nonfinite rejection. CLI
tests over SSH passed residual/S0 framing, exact zero output, clean EOF, incomplete/
malformed/nonfinite inputs and an empty dataset. No Node installation on AX102;
the driver runs locally and keeps input/output in process memory.

The final original-reference comparison was rerun without -march=native, matching
the standalone server's portable arithmetic flags. Both five-position sequences
again matched all layer-0 residuals and S0 snapshots exactly, with independent/
reset sequence controls. Full QKV and all prepared dataset hashes passed first.
Final Windows source/test strict compilation, 50 documentation links, relocation
manifest hash, unchanged original reference hash and reserved empty context passed.

Correction: the initial failed assertion DID create a 1373164343-byte apport
report under /var/crash for this test executable. The earlier work-directory-only
check was insufficient. Verified its ExecutablePath and unchanged identity with
no active writer, then unlinked ONLY that test-generated report; absence verified,
no copy retained. It briefly persisted process memory, contrary to the intended
boundary. Subsequent reference/sanitizer tests ran with process core limit zero;
document this setting before runtime too. Host apport settings were not changed.
No claim that checking the work directory alone establishes no persistence.

All requested implementation and QKV relocation work is complete. Keep source
and model identities distinct from execution evidence. Broader sequences and
full client/server/model integration remain unverified; no performance claim,
third-party runtime, input-dependent dataset, git commit or push is delivered.

## Trunk 0 Relocation Requested

The human now requests moving the existing trunk-0 into server/dataset as well.
AX102 inspection confirms six real files, no destination collision, same device
2306, and an older clover-data/trunk-0 alias. Move the complete directory without
rebuilding data or changing server.c. Retain old model paths as aliases, consistent
with the QKV move; do not restore any removed client aliases. Full file hashes and
device/inode/size/mtime plus directory identity will verify preservation.

The CLI already accepts an explicit trunk directory. A successful --inspect using
dataset/trunk-0 alongside dataset/trunk-0-qkv, plus the unchanged file hashes,
can check the relocation without another model computation. Update commands and
current layout docs after the move. Local workspace remains metadata-only for
these large payloads. Record this move separately from historical inventories.

## Trunk 0 Move Completed

All six files moved to /opt/clover-k3/clover-intelegence/code/server/dataset/trunk-0.
Full SHA256 and device/inode/size/mtime checks passed before and after; the directory
identity also matched. Both old trunk paths resolve to the same files, with no
internal links. No active readers were found in preflight. No contents changed.

The unchanged server binary passed --inspect with dataset/trunk-0-qkv/qkv.bin
and dataset/trunk-0, loading its thirteen prepared records. Source and executable
hashes matched before/after. This path-only check is not a new numerical inference
run; prior layer-0 comparisons remain their recorded evidence. The server now
owns both real runtime dataset directories. Large payloads remain AX102-only.

TRUNK0-MOVE-PLAN.tsv and TRUNK0-MOVE-JOURNAL.tsv were downloaded with the unchanged
manifest as metadata only. Current commands/docs now use internal dataset paths;
prior context entries retain the history of the previously external dependency.
No client aliases restored, other datasets moved, rebuild, commit or push.

Documentation links, six-file relocation record and downloaded manifest SHA256
passed local checks; reserved context stays empty. Shared dataset notes distinguish
local metadata copies from the real server directories. Sync only current notes;
the moved original manifest and all other dataset files remain unchanged.

## Match Client Bin Layout

The human points out that the server layout does not match the client's bin.
The server has executables in bin but its dataset is still a sibling. Move the
complete nine-file server/dataset directory beneath bin, keeping C sources,
tests and development documentation outside. Retarget the two existing model
aliases directly into bin; no new server/dataset alias or duplicate data. The
prior twelve client-alias removals remain unchanged.

Actual AX102 inspection found both real datasets, same device 2306 and no target
collision. Full hashes/file identities and a server --inspect from inside bin
can verify this layout-only change; no arithmetic change or model rerun needed.
Local dataset notes and manifests follow the same move. Numeric payloads remain
server-only. Preserve older move plans as historical snapshots.

## Bin Layout Corrected

The nine-file dataset directory moved by same-filesystem rename into server/bin.
Full SHA256/device/inode/size/mtime checks passed for every file, and directory
identity matched. Both existing model aliases were atomically retargeted to the
new real subdirectories; both older clover-data chains still resolve. No new
server/dataset alias, internal symlink or duplicate payload was created.

The unchanged source and executable hashes were verified; --inspect from inside
bin successfully loaded both datasets and all thirteen active prepared records.
All three local metadata files moved with original hashes/sizes/mtimes before
the location note was updated. The full numeric payloads remain on AX102.

README paths and ignore rules now match the client arrangement: runtime in bin,
source/tests/docs outside, compiled files ignored individually rather than all
of bin. BIN-DATASET-PLAN/JOURNAL.tsv preserve the original nine-file snapshot and
one directory move plus two alias updates. Existing numerical tests are unchanged;
this was a path-only correction, not new arithmetic or model validation.