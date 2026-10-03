# Remaining Transformer Campaign

The requested packages are transformer-2 through transformer-92, with individually
scoped C functions and complete runtime datasets beneath each bin directory.
All 91 packages are built and their datasets relocated. Every layer passed both
France and Japan five-token cases: 182 comparisons covering 910 positions,
6,522,880 residual values, all 14,560 selected route IDs and complete snapshot
bundles. [Layer index](LAYERS.md) links each package and verification record;
[results.json](results.json) reconciles source identities, cases and moves.

The implementation specializes actual dataset metadata, including each root's
palette/maps/templates. Layers3,7,...,91 and92 use MLA; other remaining layers use
KDA. Snapshot counts advance at12,24,...,84; the original ordering, residual rule
and rounding are preserved. Stored values are reused; live projections are not
replaced with historical activations. No third-party runtime decoder or model
engine. Main only coordinates argument handling, opening and dispatch.

Validation uses the original all-layer C with reference-only expert observations
matched by complete live input and expert ID. Candidates independently decode
placement and compute outputs from stored values; observations never feed them.
Every tested position must match residual, routes and the complete snapshot bundle.
The test oracle is not a fresh reconstruction of retired expert checkpoints.
Both France/Japan five-token sequences are required for every layer. Representative
reset and malformed bundle checks accompany these tests. Sanitizers cover both
attention families and snapshot boundaries. This is not a full deployed-model or
long-context performance result.

The campaign writes only source/build identities, validation summaries and move
metadata. No input/output/activation files. All test processes disable core dumps.
Moves are same-filesystem renames with hashes/identities and old-path aliases.
Original datasets, tests, transformer1, server and client remain unchanged.
Large payloads are server-only. See [context](CONTEXT.md) for progress and limits.

## Completed Checks

All 91 variants passed strict Windows executable builds and strict AX102 builds
and dataset loading. The 67 KDA and 24 MLA variants bind their own root metadata,
not layer 1's palette dimensions. The original-reference campaign compared each
candidate independently; candidate outputs did not feed later reference layers.
This does not claim a separately tested chained standalone pipeline.

Standalone ASan/UBSan with leak detection passed layers 2, 3, 12 and 92, including
two sequential zero-input positions, reset and invalid snapshot-count/nonfinite
controls. Packaged CLI checks passed those same four variants for vector-bundle
framing, output snapshot counts, zero outputs, EOF and explicit malformed-input
exit codes. Sanitizer/CLI coverage is representative, not all 91 layers.

All 273 dependency directories moved in layer order only after both cases passed.
The 3,325 files (1,363,192,610,753 bytes) retained full SHA256 and device/inode/size/
mtime; directory identities were preserved and old model aliases still resolve.
Every package passed bin-local --inspect after moving. A final audit rechecked
all moved file identities/aliases and the locked source/binary hashes. No payload
regeneration or large local download occurred.

## Evidence

- [preflight.log](preflight.log): 91 successful build/data checks.
- [validation.log](validation.log): all 182 per-layer case results.
- [validation-run.log](validation-run.log): representative sanitizer results and campaign completion.
- [move.log](move.log): 91 packaged-layer results and 273 completed directory moves.
- [SOURCE-LOCK.sha256](SOURCE-LOCK.sha256): exact candidate/test source and binary identities.
- [cli-results.json](cli-results.json): four packaged CLI cases; no vectors retained.
- [local-builds.json](local-builds.json): 91 Windows executable hashes; compilation, not Windows numerical validation.
- [results.json](results.json): independently reconciled counts and per-layer evidence.
- Each package's MOVE-PLAN.tsv and MOVE-JOURNAL.tsv retain original hashes and paths.

The small evidence archive's SHA256 is
`a2ddb36492b805f33842fe565b7cd095ed941ce34f47dbf66f313fcf81016e04`.
It was transferred and hash-verified before local reconciliation. The numerical
reference uses original arithmetic with existing exact-input-bound observations
for retired expert tensors; this is not a fresh run of those original tensors.

Generation uses [generate.mjs](generate.mjs), [mla.inc](mla.inc),
[load-operator.inc](load-operator.inc) and [process.inc](process.inc). These are
development tools; each emitted package is standalone. [campaign.sh](campaign.sh)
retains the original gated workflow, not a command to repeat over completed moves.
[finalize.mjs](finalize.mjs) checks the preserved evidence and refuses changed
verification records. Sources, runtime parameters and historical tests of the
client, server and transformer 1 were not modified. No commits or pushes were made.