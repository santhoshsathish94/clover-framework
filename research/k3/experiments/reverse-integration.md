# Reverse-order cumulative integration

Direction (2026-09-29): combine inventory items 13 through 1 in reverse order,
one at a time. Each item requires its own implementation change, validation
and recorded outcome before proceeding. Do not substitute an all-at-once build.

## Boundaries and checks

- New isolated AX102 experiment only; installed reference and previous evidence
  remain unchanged. Performance is deferred, correctness is not.
- Preserve the existing sampled expert rows/groups and two five-token prompts.
  This is cumulative integration, not expansion to the entire checkpoint.
- Earlier alternative representations may be composed as exact encode/decode
  stages; their descriptor sizes are not additive compression gains.
- The previous adaptive candidate already consumed joint code ranks. The new
  starting point must use packed codes until item 11 is explicitly introduced.
- Retain raw traces, compare reconstructed data and affected projections, and
  require complete final norm/logit bytes and routes to match the references.
- A failed check stops progression. Do not replace a failing candidate with
  original data and call it a pass.

## Ordered checklist

- [x] 13. Adaptive scale exceptions with packed-code consumers.
- [x] 12. Base/offset scale representation composed with item 13.
- [x] 11. Joint code rank added to the scale path.
- [x] 10. Constrained histogram rank added as an explicit code stage.
- [x] 9. Counts/arrangement descriptor added as an explicit code stage.
- [x] 8. Mask count/position rank added to code reconstruction.
- [x] 7. Four-mask weight decoding added after code reconstruction.
- [x] 6. Tail source-expression consumer.
- [x] 5. Entry row-reference consumer.
- [x] 4. Rounded source-expression router consumer.
- [x] 3. Lossless vector byte layouts.
- [x] 2. Shared-exponent vector blocks.
- [x] 1. Exact component equation.

## Evidence log

Preparation: read actual adaptive candidate and coverage hook. Its code path
already includes joint ranks and exact value decoding; merely copying it would
not establish a step-13-only baseline. Plan a packed-code consumer branch,
retaining scale reconstruction and projection/output checks. No new run yet.

Step 13 smoke: new packed-code consumer compiled and passed a fresh France
reference comparison on AX102. Nine rows, 960 scales, 30720 decoded-scale
weights and nine projections; complete norm/logit bytes and routes identical.
Adaptive scale payload 123 bytes. Joint-rank consumer disabled, zero joint
decoder controls; order/value controls still run as validation, not consumers.
Patch application required LF normalization and explicit replacement hunks.
Broader step-13 checks are pending; no item12 implementation started.

Step 13 complete: independent smoke/France/Japan checks PASS. Union 1926 rows,
214 layer/expert pairs, 5778 packed groups, 205440 scales, 23830 adaptive bytes.
Broad runs compare 4320 projections and both full norm/logit and route outputs.
Evidence: AX102 reverse-integration-20260929-a/step13/results.json. Item12 now
adds an explicit base-offset decode feeding the adaptive encoder; not yet run.

Step 12 complete: both prompt gates PASS with unchanged coverage and full
outputs/routes. Every base-offset payload independently decoded and matched
before checking the adaptive payload. Step12/results.json retains evidence.
Item11 now explicitly requires joint-rank consumption; its gate is pending.

Step 11 complete: France/Japan cumulative gates PASS, same coverage and full
outputs/routes. Actual 17-byte joint payloads matched saved exact ranks.
Item10 now serializes decoded counts into a five-byte histogram rank, clears
the counts, reconstructs them, then supplies arrangement decoding. Gate pending.

Step 10 complete: both cumulative model gates PASS. All histogram payloads
matched the independently computed constrained rank; 105 controls and four
invalid cases retained. Item9 now adds actual serialization of scale/counts/
arrangement after histogram reconstruction, before code decoding. Gate pending.

Step 9 complete: both cumulative model gates and independently reconstructed
complete descriptors PASS. Item8 now applies mask count/rank to plane2 across
the existing sampled groups, with 5249 controls and 34 invalid-input checks.
Reconstructed bits replace the selected plane before projection. Gate pending.

Step 8 complete: both cumulative gates PASS. First build also passed but
warned about an implicit mask buffer bound; added an explicit five-byte guard
and reran the same full gates without warnings. First evidence preserved in
step8-before-bound. Item7 now adds four-mask payloads and exact weight-equation
consumption, leaving original projection arithmetic unchanged. Gate pending.

Step 7 complete: both cumulative expert paths PASS, including actual four-mask
payloads, regenerated weights, projections, routes and complete outputs.
Item6 now adds only the tail source-expression consumer and its unchanged
endpoint checks; no entry hook yet. Gate pending.

Step 6 complete: both cumulative gates PASS, including 7168 tail aggregate,
7168 tail norm and 163840 head-logit comparisons per prompt. Item5 now adds
the entry row-reference consumer; entry and tail will be enabled together on
top of the unchanged expert chain. Gate pending.

Step 5 complete: both cumulative gates PASS, including 35840 embedding and
35840 entry-norm comparisons per prompt. Item4 now adds the rounded router
source expression at layer84/position0, with all896 projections and sigmoid
scores checked. Failed Gram mode is not enabled. Gate pending.

Step 4 complete: both cumulative gates PASS, all896 router projections/scores
per prompt exact along with endpoints and expert paths. Item3 now introduces
both zlib byte layouts for the generated L84/pos0 vector. Decoded values, not
the original array, feed the same router norm/projections. Scope one complete
7168-value vector per prompt; private zlib dependency, no system installation.

Step 3 complete: both cumulative gates PASS; raw-byte and byte-plane zlib
payloads independently decompressed to the consumed vector. Private zlib1g-dev
1:1.3.dfsg-3.1ubuntu2.2, package SHA256 e9152a08af21ab22bc99e8bfa98f1eb83955b2cc3653bcefecab394ecf9d1b63.
Item2 now adds both existing DYA1 block formats with exact GMP integers,
finite/extreme/raw-block controls, and unchanged Python payload comparison.

Step 2 complete: both cumulative gates PASS, and actual32/128 block payloads
match the original Python encoder byte-for-byte and decode exactly. Item1 now
adds an explicit component-equation payload after both blocks: eight-byte
header (O1, width, minimum exponent, exponent/code widths), then sign, zero
flag, exponent offset and odd-integer code. Zero flag extends the earlier
nonzero-only analytical layout to preserve signed zero. No table lookup or
new compression claim; full final step gate pending.

Step 1 complete: both full cumulative gates PASS. Every component field agrees
with the original Python factorization, and all restored float32 bits match.
All13 checklist items were individually introduced and gated in reverse order.
The repeated unchanged France reference also matches after the campaign.
Final provenance/archive verification is in progress; no production/reference
files were replaced and no persistent storage or performance claim is made.

## Completed outcome

**PASS: all13 items combined one at a time in reverse order.** Each step has
its own candidate, source snapshot, raw traces, two-prompt model gate and
independent representation checks. No later item was activated before the
preceding step passed. Failed patch attempts and the first warning-bearing
step8 build are retained separately, not counted as completed steps.

- 13 passing cumulative candidates, 26 broad model runs, 56160 broad projection
  comparisons across the sequence. These repeat the same scoped sites; they
  do not multiply unique coverage. Smoke and warning-repair runs are additional.
- Final unique expert scope:1926rows,214layer/expert pairs,5778code groups,
  205440scales and6574080weights using reconstructed scales.
- Vector codecs: one generated7168-value vector atL84/pos0 per prompt, with
  both zlib layouts, both dyadic block sizes, and the component equation active.
- Per prompt:35840embedding and35840entry-normalized values,896router
  projections/scores,7168tail aggregate/norm values and163840logits match.
- France and Japan full output MD5s remain23d162dcefb18211a7540ef12948f1eb and
  4b2a7b96fb6323e5639feced66f54a8e. All routing traces match; repeated France
  reference after the campaign matches. Installed source/binary/gate/index
  hashes are unchanged.

The [campaign report](reverse-integration-results.json) verifies ordered gates,
protected/dependency hashes and per-step result identities. The
[final tested source](reverse-integration/step1/candidate.c) and
[final step report](reverse-integration/step1/results.json) are local, alongside
all prior frozen snapshots. Raw projection/group traces remain on AX102.
The [local-only verified archive](README.md#local-only-artifacts) SHA256 is
f0ee1aa5a0c42f26fa6f57cf5e327fb750f7c7f02a8a3229c5581dc90eb8611d.
Extraction and519recorded file hashes were verified locally.

The first local extraction command found no tar on PATH but continued because
PowerShell errors were non-terminating. Its success text was invalid. Repeated
with ErrorActionPreference=Stop and the explicit Windows tar path; extraction
and all hashes then passed. Remote model evidence was unaffected.

Alternative encodings are explicit serial reconstruction stages, not cumulative
compression gains. The latest scale descriptor still occupies23830bytes on
the sample, but the harness retains other intermediate descriptors and original
data. This is not a persistent checkpoint format, an all-rows integration,
generation/decode coverage, or measured storage/RAM/speed improvement.

## Subsequent speed and RAM measurement

The user subsequently authorized [as-is performance measurement](integrated-performance.md)
of this final candidate against the unchanged reference. On AX102, three
measured trials per prompt/variant showed 3.4-3.6% longer median end-to-end
time and essentially unchanged peak RSS near 54.18 GiB, with all correctness
gates still passing. These results include retained reference work and
diagnostics; they do not estimate an optimized deployment or a new weight store.