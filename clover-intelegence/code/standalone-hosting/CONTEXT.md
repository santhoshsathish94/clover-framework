# Standalone Hosting Source

## Current: End-To-End Text, Not Just Token Ids

Everything measured so far was token ids. To read the output as language the ids
had to be mapped back through the vocabulary. `test-decode` is the zlib decoder,
not a detokenizer, and the host has no `tiktoken`, `regex` or `transformers`, so
a minimal encoder/decoder was written at `/tmp/k3tok.py` on AX102, outside the
repository. It reads `dataset/tiktoken.model` (163,584 base64 rank pairs; the
remaining 256 of the 163,840 vocabulary are special tokens) and applies the Kimi
pretokenizer pattern restricted to ASCII plus tiktoken byte-pair merging.

It is a reimplementation, not the official tokenizer. It was accepted only
because it reproduced ids already observed from other evidence: `Rain falls`
encodes to 91019, 25528 and `The capital` to 1008, 10484, both exact. Non-ASCII
prompts are outside what was checked and should not be trusted to it.

What the model produced, with the batched build:

- `The capital of France is` -> ` Paris.",` then a newline and
  `"The Eiffel Tower is located in`. Correct, then drifting into the JSON shape
  the five-field france records were stored in.
- `Rain falls` -> ` on the roof of a house and collects in a rain gutter that
  drains into a rain barrel. The base of the barrel is a circle with an area of`
- `In 1969, humans first walked on` -> ` the moon. The Apollo 11 mission was a
  historic achievement, but it was`

Determinism, unplanned but worth keeping: the 32-token `Rain falls` run
reproduced the first 12 tokens of the earlier 12-token run exactly, ids and all.

Cost at these lengths, which the short benchmarks did not show: 6.61 s per token
for 2 in and 32 out, 10.00 s per token for 9 in and 16 out, against roughly 2.3 s
per position measured on a warm two-token request. Prefill batching does not
touch this. Every generated token routes to its own sixteen experts per layer, so
a long generation keeps pulling previously unread expert data off disk, and 124 GB
of page cache cannot hold a working set drawn from 1.45 TB. The per-token figure
rises with how much new expert territory the continuation visits, which is a
property of the routing and the storage, not of the batching change.

## Current: Batched Prefill Positions Share One Weight Sweep

Intended outcome: stop re-reading the dense weights once per prefill position.
Measurement had already shown roughly 70.68 GB of coefficients streamed per
token, so several positions evaluated together should pay that once.

What changed. `NPOS` was a compile-time macro fixed at 1. It is now two things:
`NPOS_SLOTS`, a compile-time bound that sizes every fixed array, and `NPOS`, a
runtime count of the positions active in the current pass. `evaluate_token` became
`evaluate_tokens(tokens, count, position, project)` with a one-token wrapper kept
for the tests that call it directly. `resident_step` buffers the prefill positions,
which do not project, and evaluates them together with the final input position in
a single pass. Decode steps are unaffected and still run one position at a time.
`build.sh` takes `CLOVER_NPOS_SLOTS`, default 8.

What the system showed, on AX102, warm page cache, three runs each:

- Two input tokens, one output: 4.81/4.86 s unbatched against 3.65/3.68/3.66 s
  batched. Roughly a quarter faster, and the spread does not overlap.
- Eight input tokens, one output: 52.6/55.2 s at two slots against 49.6/52.3 s at
  eight. Direction favours eight but the run-to-run spread is wider than the gap,
  so this is not a result, only an absence of harm.

Correctness. Eight input tokens and three outputs produced tokens 198, 1008, 12981
at one, two and eight slots, identical. The France reference check passes with all
7,360 reference expert matches and the same continuation token 20829.

What this does not cover. Batching amortises the dense weights only. Each position
selects its own sixteen experts per layer, so a batch of eight needs the union of
up to 128 experts per layer and the routed share does not shrink. That is the
reason the eight-token case barely moves, and it is a property of the routing, not
of this change. Cold and warm page cache still differ by about a factor of two
against 1.45 TB of expert data behind 124 GB of RAM; every number above is warm.

Finding, not caused by this change: `test-cross-layer-model` fails its assertion
`resident_cross_layer.submitted==(position+1)*92`. The pre-batching snapshot
`bin/clover-one.c.known-good` fails the same assertion in the same place, and a
live request reports `cross_layer_submitted` 0, hits 0, misses 0. Cross-layer
candidate prefetch stages experts for the old packed format; with `experts.direct`
mapped there is nothing to stage, so the mechanism is inert and the assertion
encodes a removed behaviour. The numerical assertion earlier in that same test,
that every expert input and output is byte-exact, passes. The test has been left
failing rather than adjusted, because deciding whether to retire the mechanism or
the assertion is a direction question.

## Current Full-Model Timing: Where Time Goes

The human asks where the current run spends time. Recompiled the existing
profile-stages.c from current source, with no decoder experiment or runtime change.
The first SSH capture disconnected after startup (5.147800179 s), exit255; no
complete report was created. Process inspection found no surviving profiler and
no new kernel memory-kill event; the network-reset cause remains unknown.

Recovered with a finite run that ignores HUP and filters stdout to timing events
on AX102, retaining an explicit exit status. No activations/logits were written.
Local raw evidence: stage-profile-current-20261002.jsonl; structured evidence:
stage-profile-current-20261002.json. Remote original resides under
bin/profile-current-20261002-recovery.jsonl. Current default generation.json,
16 threads close/core affinity, no prior requests. Existing output assertions
passed for Rain falls ->418 then276; both output-producing steps reported zero
result-cache hits and4,416live expert projections. Process exited0.

Startup5.132054012s, position0 14.772339065s, position1 30.096404195s,
continuation16.532759167s. First output from2inputpositions44.868743260s;
startup is separate. This is a new instrumented observation, not a retroactive
breakdown of the earlier45.028s request. No claim of restored six-second latency.

First-output disjoint wall costs: expert/mix/norm/up26.340245765s;
head first preparation/projection15.295461546s; allattention2.176674446s;
sharedexperts0.615059500s; residual/state-save/unbind/nextprediction0.161231742s;
latentdown0.111182938s; router/top160.103850052s; layer0dense0.033484472s;
otherboundaries/unassignedtimer overhead0.031552799s. Exposedreaderwait nested
inside these stages0.000273103s. Continuationexpert/mix14.760273914s,
attention1.178713111s, head0.051457970s, exposedwait0.075617652s.

Expert worker sums across first2positions: decode197.348521710s,
CRC/selectorvalidation110.288644238s, math66.935093551s, readwrapper0.018348936s.
These sum concurrent workers, NOT elapsed seconds. Roughly53/29/18percent of
measured expert worker time is decode/validation/math. Readwrapper touches staged
bytes, not totaldiskI/O. First-head timer does not separate read/decode/layout/hash
from scoring. Do not invent an exact split inside that combined timer.

New summarize-current-profile.mjs reconciles every layer with the existing
disjoint boundary set (same0.005ms double-count and2ms unexplained-gap limits),
checks position coverage, and writes CURRENT-RUNTIME-TIMING.md. All279layer totals
passed;13,560recorded rows published,35-50measurements per layer by type. Includes
all93layer totals for3positions and allrecorded boundaries/projections/operators/
details/workers with callcounts, not merely3samplelayers. Measurement rows are NOT
a claim every scalar/substage has a separate timer. Runtime and datasets unchanged.

## Experimental Decoders Removed Completely

The human explicitly requests removal, not dormant code. Removed the optional
system-zlib/libdeflate includes, macros, wrappers and renamed portable entry point
from decode.h, restoring its pre-experiment implementation. Restored the original
capture-stage-profile.mjs interface/command by removing the experiment-only binary
selector and associated configuration changes. No runtime configuration changed.

Deleted exactly six experiment binaries from AX102 bin: test-root-phases-libdeflate,
test-decode-system-zlib, test-head-cache-system-zlib, test-root-phases-portable,
test-decode-libdeflate and test-root-phases-system-zlib. Deleted both private
dependency trees bin/zlib-dev and bin/libdeflate-dev, including downloaded packages
and extracted headers/libraries. No system packages had been installed.

Verification: normal standalone build succeeds without those dependencies;
unchanged decoder oracle24cases and118rejection controls pass; restored profile
driver passes Node syntax check. Source searches and explicit artifact checks
found no experimental decoder references/artifacts locally or on AX102. The
pre-existing decoder, tests, historical evidence and dataset remain intact.
This cleanup is not a performance fix or a completed no-decoding model path.

## Latest Direction: No Decoding, All Layers And Stages

The human explicitly says decoding is not required anywhere and requires attention
to all 93 layers and their individual stages, not a three-layer probe. The decoder
optimization branch and its artifacts have now been removed as recorded above.
No complete new profile was produced. The interrupted full-profile command had
unknown tool outcome; subsequent file/process checks found no new baseline report
and no running profiler. Do not claim it completed.

Read-only AX102 audit of the configured dataset examined ALL 93 trunk indexes,
all prepared record bounds/group sizes and all 93 operator headers, not samples
of those layers. Found 2,710 prepared parameter records: 1,459 float32 vectors,
92 float32 router matrices and 1,159 palette-indexed matrices. These are parameter
records, NOT a count or verification of the human's 40+ computation stages.
Layer0 has26 records; 68 routed KDA layers have31 each; 24 MLA layers have24 each.
All operators identify as K3QKV001 or K3MLA001. Direct float records are already
read without decompression by the current fixed-parameter binding path.

All 92 root expert files identify as K3MAPS01; every first expert block checked
has a valid zlib header. This checked each container's header/size and FIRST block
marker, not every block's integrity. All 10,240 input block descriptors use codec4;
outputs have10,239 codec4 plus1 codec5. Current numeric-table.h interprets those
as compressed planes. Existing uncompressed observations total7,360records per
France/Japan set across92roots, with matching input-vector file sizes. No runtime
inference, decoding or dataset writes were performed during this audit.

The configured files therefore do not establish a complete no-decoding data path.
This is a concrete mismatch to resolve with the human, not authority to substitute
a faster decoder or rebuild another dataset. Ask which directly usable computed
dataset/stage-value path is intended. Full stage-by-stage execution remains open.

Historical probe, now removed: decode.h temporarily had optional
CLOVER_SYSTEM_ZLIB and CLOVER_LIBDEFLATE wrappers, portable default unchanged.
Private development packages were downloaded/extracted into bin/zlib-dev and
bin/libdeflate-dev on AX102, with no system package installation; both trees are
now deleted. Both experimental backends passed
unchanged24oracle/118rejection tests; zero-output and small-window streams retain
the portable path. System zlib's zero-output edge case was caught by the original
tests and corrected. libdeflate three-layer wall~.090-.092s vsportable~.176-.178s,
not full-model evidence. System-zlib head check passed all163840scores for2inputs:
firstcache12.457s/warm.050575s. The initial head invocation passed a directory and
failed before correcting to outputs/fruit.bin. No such probe restores six seconds.
capture-stage-profile.mjs temporarily accepted a binary name and clean fresh-request
config; it has now been restored to its pre-experiment interface. No completed
all-layer decoder comparison or production enabling is claimed.

## Performance Regression: Baseline Takes Priority

The human rejects treating the measured 45-second request as success when the
earlier implementation achieved about six seconds. That earlier measurement is
valid; uncertainty about generalizing a newer timing does not invalidate either
observed outcome. The assistant failed to preserve the performance objective and
kept optimizing the slower implementation instead of resolving the regression.

This turn re-read original clover-k3.c Xm: direct packed-weight/scales, DQ2 pair
loads and explicit AVX2 double multiply/add with the original reduction. Current
live-root.h instead reads compressed blocks, calls decode_zlib, validates decoded
BF16 CRC and selectors, then runs palette/map lookup arithmetic on cache misses.
head-cache.h allocates at startup but prepares its bytes inside the first head
projection. These are material execution-path differences, not evidence that
the original six-second result was merely recorded-answer replay.

Extracted existing stage-profile-after-corrected.json, not a new timing run:
expert/mix totals 15.437, 15.185 and 15.141 s at positions 0/1/2; first head
fill/projection 15.667 s, warm head 0.050909 s; warm full position 16.688 s.
This profile predates the latest prefetch changes. The latest 45.028 s is a
two-input/one-output first request, not one warm decode step. State both timing
boundaries correctly without using the distinction to dismiss the regression.
Latest prefetch pair reduced exposed wait by only about 0.282 s: it does not
resolve the dominant expert execution and first-head-fill costs.

No runtime/config/dataset change or inference run in this turn. Recovery must
return to the original fast expert reader/kernel as the reference, retain exact
outputs and existing data constraints, and compare the same workload/timing
boundary. Do not silently rebuild data, remove validation, replace the source
wholesale, or move request costs into startup and call that restored throughput.
Six-second performance has not been restored; the dominant path is the priority,
not another repeat-result-cache headline or speculative-read optimization.

## Current: Cross-Layer Candidate Prefetch

The human prioritizes implementing cross-layer selected-expert prefetch before
the broader computed-result-cache and 46-second timing discussion. Exact next
selection requires next-layer attention; early reads are explicitly speculative,
not a substitute router. At the end of layer N, normalize its final residual with
N+1's post-MLP gain and score N+1's stored router/bias, omitting its not-yet-computed
attention/aggregation. Submit one highest-scoring candidate into existing slot 0.
Its read overlaps N+1 attention. No prior prompt, saved routes or extra staging
allocation; this is a bounded first implementation, not prefetch of all top 16.

After the live router and exact-result lookup, accept only the same root and an
actually selected cache-missing expert. Evaluate a matching candidate first, copy
all expert down outputs to rank-indexed RAM, then mix in original rank order.
Wrong/unneeded candidates are drained and discarded, with normal reads afterward.
Original format/CRC checks and live numerical routing remain intact. Predictions
never supply expert outputs or change selected IDs/weights. A candidate cannot
escape a token boundary. Pipeline close precedes dataset release as before.

Initial handoff tests passed match/miss/wrong-layer/cached-result skip/disabled/
shutdown paths. Integrated fresh two-position France check passed all 2,944
independent expert input/output pairs across all 93 layers. 184 candidates:
137 matched, 47 missed; extra read accounting matched misses, zero result reuse.
This establishes numerical correctness for that workload, not a speedup. Next:
verify result-cache coexistence and measure new fresh-request baseline/overlap
using test-cross-layer-service.mjs. Older tests and historical timing reports
remain unchanged; their read-count assumptions predate speculative traffic.

Completed: all test targets built. Additional reference-checked result-cache
requests passed; the repeated one reused all 1,472 expert results with zero live
projections. Predictions on that path were safely discarded when unnecessary.
The new handoff test passed ASan/UBSan/leak checks; the unchanged pipeline CRC,
zlib corruption and EOF test passed. No full test-suite execution is claimed.

Fresh Rain falls 2-input/1-output AX102 pair, 16 threads, close/core affinity,
separate processes and result cache disabled. Both emitted [418], released all
372 mappings, decoded 10,240 head blocks and used 72 MiB pipeline staging.
Without cross-layer: startup 4.923923982 s; request 47.116389686 s;
wait 0.286417521 s; 2,944 reads / 46,508,564,103 logical bytes.
With cross-layer: startup 4.937876670 s; request 45.028196234 s;
wait 0.004082860 s; prediction 0.108840578 s; 184 candidates, 142 hits, 42 misses;
2,986 reads / 47,172,510,507 logical bytes. Extra 663,946,404 bytes from misses.
These are logical reads, not physical I/O. The whole-request gap includes other
variation; saved exposed wait alone is about 0.282 s, not the whole 2.09 s gap.
Baseline ran first and OS caches were not flushed. No statistically established
speedup, no five-second claim, no new complete stage profile, no full 128 run.

Growth: a dependency preventing early exact routing does not prevent speculative
I/O with live validation and a correct fallback. Keep these two claims distinct.
Speculation can waste traffic, including on an exact-result hit; report both hits
and misses and prediction cost. Default computed-result caching remains enabled.
No historical fixtures, dataset values or recorded timing reports were changed.
The new test's initial pread declaration warning was fixed by including unistd.h.

## Current Direction: Prefetch Within A Fresh Request

The human clarifies that prefetch must overlap stages/layers of a fresh request,
not depend on submitting the same input again. The previous RAM result cache
remains an optional exact-repeat optimization, but its 7.9 s repeat is not evidence
of the desired fresh-input latency. Expanding it is not this request's fix.

Read startup: all fixed non-routed layer parameters are mapped/touched before
input, not physically locked. Next-layer expert IDs are unknown until its input
and router are evaluated. No future-selection assumption. The first expert read
was followed immediately by a wait while the independent shared branch waited
until routed experts finished. Moved the same Qm/SiTU shared branch into that read
window, before acquire. Next-missing-expert reads still overlap the current expert.
No arithmetic sequence inside either branch or ordered routed sum changed.
Result lookup still precedes reads when enabled; the comparison disables it.

The focused test's first layer outputs matched, but the existing operator coverage
guard stopped because the test did not consume QKV records. Corrected the new
test to use actual record consumers, without counter edits or relaxed guards.
Layers 1, 3 and 92 passed: shared gate/up/output and expert output byte-exact
against the original calculations. The original coverage/lifetime guards passed.

Fresh AX102 comparison, 16 OpenMP threads, close/core affinity, result cache zero:
CLOVER_DEFER_SHARED baseline versus new default, separate processes, one request
each, two input IDs [91019,25528], one output [418]. Both performed 2,944 expert
reads/misses, zero result hits and normal shutdown with 372 mapping releases.
Baseline startup 4.967708844 s, request 49.310048407 s, pipeline wait 0.936958583 s.
Overlap startup 4.930079536 s, request 46.989308169 s, pipeline wait 0.308189591 s.
First head fill remains included in each request. Baseline ran first; OS cache
was not flushed. One sequential pair is an observation, not proof of a repeatable
2.32 s end-to-end gain. It does demonstrate the fresh-request path without prior
computed results and less measured reader wait in this pair.

Profiling separates result lookup/first submit, shared work during read, routed
expert work and final merge/cleanup. Summarizer retains old stage-name support;
historical reports stay unchanged. New tests and the baseline build are wired to
build.sh --tests. No dataset generation or activation persistence was introduced.
Five-second fresh tokens remain unresolved. Do not return to repeat-input cache
timings as the headline; distinguish CPU decode/validation, exposed read wait and
first head fill when choosing further work. No full 128-input/128-output run.

Final validation: build.sh --tests compiled all targets; the focused layer 1/3/92
test passed again on the final source. The report summarizer accepted unchanged
historical reports and synthetic new stage boundaries in memory, and still
rejected double-counted timing. No synthetic report was saved as measurement.
No new complete profile capture or full-suite execution is claimed.

## Result Cache Capacity Question

The human asks totalRAM and why not retainall1400+computedexpertresults. Fresh
AX102/proc/meminfo: MemTotal130982176KiB=124.914GiB(134.126decimalGB), available
129181120KiB=123.197GiB with NOstandaloneprocessrunning. AvailableidleRAMisnot
headroomaftermodelstartup. Fixedmappedpayload75,891,522,488B=70.679GiB; head2.188GiB,
pipeline72MiB, reserve8GiB plusstate/OSrequirements remain distinct quantities.

Currententry=28,696bytes(input3584F32+down3584F32+24bytesmetadata). Oneprocessed
token uses92*16=1472expert-evaluations,40.284MiBentries; previous3positions4416
entries120.851MiB. Currentactualallocation224.188MiB8192entries CANholdthese, but
8waysetconflicts caused prematureeviction(63firstrequestevictions,333repeatmisses).
This is a cacheorganization limitation introduced here, notinsufficientphysicalRAM.
Full255processedpositionsat128in+128outmayneed375360distinctentries10.032GiB
beforeindex/allocatoroverhead. All4intermediatesplusinputfor1472wouldbe92.034MiB;
keepingonlyfinaldownissufficienttoskipentireexpertchainandpreservemixing.

Recommendation: globalcapacity/sharedpool eviction onlywhenfull, thensizeto
working-setwithinliveheadroom. Merelyincreasingeightwaybudgetdoesnoteliminate
setconflicts. Resultsarekeyedbyinputaswellasexpert; storingoneentryperIDdoesnot
coverallfutureinputs. Thisturnmeasured/calculatedonly; no cache/config/runtime
change or inference. Do not claimall-resultretentionwasalreadyimplemented.

## Current: Bounded Computed Expert Results In RAM

The human explicitly requests storing/reusing computed values in RAM, not only
expert IDs. Added ExpertResults owned by one model process, initialized atstartup
and freed atshutdown. Key includes layer,expert,inputhash AND all3584inputfloat
bytes. Hash is only a filter; forced collisions cannot bypass fullcomparison.
Cached payload is final3584downvector, sufficient toskipgate/up/activation/down.
Mix still uses currentroutingweights and originalrankorder. No resultfileswritten.

Configexpert_result_cache_mib defaults256; effectivebudgetfits existingheadroom
policy.8wayset-associative cache uses235077632B/8192entries atdefault, LRUwithinset;
zero/lowbudget/allocationfailure disables gracefully. Cache scopedtoimmutable
loadedmodel; close/reopen clears results. No crossmodel or persistent reuse.
Unittestpassed fullvector, layer/expert/cacheownerisolation, forcedcollision,
signedzero, eviction, copiedhitlifetime anddisabledcache understrictcompile.

Integrationchecks all16results afterlatentprojection and BEFOREweightprefetch.
Hitscopiedtolocalbufferssoevictionduringmissfillcannotinvalidatependinghits.
Missingranksaloneenterdouble-bufferpipeline; resultsstoredonlyafterallthree
projectionscomplete. Coverageaccountsforlivecalls+3*resulthits=4416/position,
recordedobservationmatchesremain0. Cache retainedbetweenrequests, attentionstate
stillresets. Initialintegratedcompile/inspectpassed. Nextrealrepeat/changedinput
comparison verifies identicaloutputs, skippedreads/calculations andboundedRAM.

RealAX102same-processcheckpassed: Rainfallsfirstrequest2input/2output emitted
418,276in65.785218016s,0resulthits/4416misses; exactrepeatemittedsameIDs
in7.902487353s,4083hits/333missesand333prefetchreads. ChangedTeatastes emitted
1517,15600in50.906658683s,0hits/4416misses. Headcoldfillonlyfirstrequest; coldvs
repeatgapincludesheadwarmup,notpureexpertcachebenefit. Alloutputstepcoverage
liveprojectioncalls+3*RAMresulthits=4416,recordedobservationmatches0. Fixed
235077632Ballocated<=256MiB,8192entries. Setcollisions/evictioncanmissdespite
unusedcapacityelsewhere; noclaimallrepeatsarecompletehits. Bothrepeat/newinput
outputscheckedagainstprioruncachedruns. Exactlyoneprefetchperresultmiss, none
forhits. Processclosednormally,released372fixedmaps/cacheonshutdown. No model
vectorspersisted; datasetsnotmodified. Nextfinishsanitizer/configanddocs.

Finalchecks passed: originalgeneration/pipelinebudgets andlegacyweightcacheconfig,
result-cachelowmemory/disable/oversize/actualallocation controls, fulltestbuild,
andASan/UBSan/leak exactcacheunit suite. Originalcanonicaldatasetunitidentities,
lengths/nsmtime/mode/ownershipunchanged. Defaultresultcache256MiB; legacy/disabled
comparisonconfigsset0to preserve their meaning. No originaltestcriteria changed.
Currentdocslabelolderprofilereportsasmiss-pathbeforeRAMmemoization, notwarming
repeatlatency. Timingsoneobservation, notuniversalperformanceguarantee.

## Exact Result Reuse Clarification

The human asks why expert computation/mixing remain when prior routes and stored
gate/up/activation/down exist. Read current runtime: resident_expert is called
unconditionally for selected experts, no exact-output lookup precedes it. For an
identical expert input, this is avoidable work. Route-ID prefetch and output reuse
are different optimizations; the old route cache is keyed by full prompt IDs and
checks live selections. A new continuation is not automatically the same key.

Read-only comparison of existing layer1France/Japan records in memory found48
sameexpert+same3584input matches with identicaldown. Also foundexpert76 atFrance
record19/Japanrecord20 with DIFFERENTinputvectors and DIFFERENTdownvectors. This
directly shows expertIDalone cannot identify a reusable output. No inference,
parameter reconstruction, runtime edit or newvectorfile occurred in this check.

An exact-output cache should bind model identity, layer, expert and exact input
bytes; existing outputs can then bypassgate/up/activation/down. The following
mix still applies currentroutingweights in originalrankorder unless thatcomplete
mixedresult is separatelymatched. Currentruntime has no such memoization, and
this explanation does not claim it was implemented. Preserve arbitrary-input
correctness and distinguish exacthits from uncomputed/newinputs.

## Completed Stage Measurement Cycle

Complete human-readable STAGE-PROFILE.md now links timing-only before/after JSON,
279layer totals perrun and detailedcontinuationwallstage/slot/operator/worker rows.
Read workflow,stageguideandsharedsite; alltimingsfromactual source path. Model
outputIDs418/276 unchanged. Inputwarmlookupmicroseconds, firstrowdecode~1.6-1.77ms.

Originalafterreportwascompletebutitsreconciliationcaughtupto0.01627msnegative
gaps: previouslayerprofiling-summaryoverheadenterednextbindingtimer. Corrected
byresettingboundaryclockatlayerentry; originalreportsretained. Newcapture
stage-profile-after-corrected.json passes unchangedcoverageandtimingreconciliation.
Beforepositions17.572909/32.630372/17.023243s; correctedafter16.962130/32.361147/
16.687541s. Additionalworker/opreportingaddsmeasurementoverhead, solimitcomparison
toobservations. Aftercontinuationallattention1.018026s,expertsmix15.140517s,
router54.333ms,shared314.354ms,layer0total28.913ms,finalnorm0.052ms,head50.909ms.
No5secondclaim. Mainremainingcoststaysinsideexpertdecode/validation.

Four-literalfastpath retainsoriginalshort/longcodefallbacks and allchecks. Full
unchanged24decoderoracle/118rejectioncasespassedunderASan/UBSan; allfruitbytes/
163840scores,sixseedrows,21504leaves,186retainedbindings,14720observationreader
controls passed. Production andprofilebuildscompile; profiling-onlyclocks now
compileoutnormally. No originalsource/testcriteria or datasetvalueschanged.
First9/10/11bitexperimentshowedwarmerreadnotdecodegain, reverted. Keepcorrected
timer/capturelessons; neverderivecausalityfromincomplete terminaltail.

## Current: Stage-By-Stage Measurement

The human requests source/workflow/site-grounded measurement starting at text to
IDs to vectors, then every layer/stage, before choosing the next optimization.
Read clover-one/workflow.md, research/k3/model/k3-stages.md and the shared running
/ai/page. The page labels recorded examples and distinguishes fixed coefficients
from input-specific outputs; stagecounts are nottimings. Layer0preA bypassesAR.

profile-input.c runs the original native tokenizer and actual seedreader, with
originalshardrow comparisons. AX102 'Rain falls': tokenizerload24.541058ms,
firstencode0.006271ms,1000repeatmean0.000371ms; seedmetadataopen1.641398ms;
firstrow1.769608/1.603948ms, cachedrowmean0.002263/0.002309ms. Both7168rows exact.
This confirms submillisecondwarm lookup, not submillisecondinitialdecode. No layer
ran inthisslice. Profilers include clocks/validation; no coldOS-cacheclaim.

Added compile-time-only layerboundaries/projections/expertphase instrumentation.
First all93layers/3positions run passed original418/276outputassertions butterminal
capture retainedonlylast21layerrows. Do not treat truncatedcaptureascompleteevidence.
capture-stage-profile.mjs streams alltimingJSONdirectly and validates279layertotals
before saving timing-onlyreport, noactivation/outputfiles. A rerun is justified
to close this concrete evidence gap, not torepeatforfavorabletimings.

Completebaseline captured stage-profile-before.json: all279layer totals present.
Threepositionwall17.572909/32.630372/17.023243s. Layer0wall31.864/29.468/29.074ms;
warmL0attention12.069ms, denseMLP16.588ms, norm0.0118ms. All93attention~0.986s,
experts+mix~15.530s, sharedexperts~0.294s oncontinuation. Firstheadfill14.994s,
warmhead~0.051s. Thuslookup/binding/normalization isnottheseconds-scaleproblem.

Focuseddecoderexperiment9vs10vs11bitfasttables:decodeworker~1.4sunchanged; apparent
wallimprovementwaswarmerreads. Rejectedlargertables,retained9bit. Nextbatchedfour
consecutiveliteralsfromonebounded64bitwindow:unchanged24oracle/118rejectioncases
PASS. Alternatingbatch/off/batchprobe layers1/2/92 improved~0.163to0.141s wall;
decodeworker~1.378to1.04s, math/CRCessentiallyunchanged. No newvalues or changed
equations/checks. Fullafterprofileaddsperlayeroperatorandworkerbreakdown; worker
sumsmustnotbeaddedtowall/nestedprojections. RuntimecostclaimsremainAX102scoped.

## Current: Diagnose Historical Five-Second Path

The human challenges the latency regression, requests n-1prefetch rather than
anotherlargecache, and asks what precomputedexpert/QKVvalues do. Read actual
mainXm/gen.py/pl_start/nx_begin pluscurrentreaderandmetadata. Historical5.1-5.93s
decode used14threadpipelinedrawreads,AVX2pairloads,prefixreuse; notthe1.86sexpert
observationvariant. gen.py doesNOTsetcrosslayerK3_NX; thatseparateoptionrequires
prompt-keyedcachedroutes. CurrentWILLNEEDafterroutingwasnotanequivalentpipeline.
Currentportableexpertmath is scalar-lookups, butmeasurementisneededbeforeblame.

Added opt-in phaseprobe,48projections/16experts perlayer1,2,92 withallchecks.
AX102originalwall~0.456s/layer; workerread~0.53s,decode~4.14s,CRC~1.75s,
math~0.36s. Worker sumsareNOTwalltime. DEFLATEwasbit-at-a-timeandAdlermodulo
perbyte. Added9bitshortHuffmantable,boundedmultibitreads,5552byteAdlerchunks:
originalunchanged24oracle+118rejectiontestsPASS;wall~0.239s,decode~1.37worker-s.
Equivalent8byteCRCupdatewithsmallstartupCRCslicetablesretainsstoredBF16checksum
andallbounds;independentliveprojectionslayer1/2/92exact;wall~0.151s,CRC~0.448s.
Notmodelweightregeneration,nocheckremoved,andnoexternalcodecdependency.

Added persistentone-reader/twocompressed-bufferexpertpipeline. Afterrouting,prime
firstexpert; whileexpert(n-1)computes,readexpert(n);joinbeforeusingbytes. Current
decoderstillvalidatesallstageddata.72MiBtotalstaging, releasedonshutdown. Nopredictive
expertresults; exactnextlayerselectionunavailableforunseeninputbeforeitsrouter.
Warm48projectionprobeexact:direct0.161894s,pipeline0.171690s,wait0.007035s,
readworker0.048096s. DoNOTclaimisolatedprefetchspeedupfromthis;fullrequestnext.
Defaultconfigexpertcache0,WILLNEED0,pipeline72MiB,headcache2240MiBretained.
Memorybudgetincludespipeline;startupinspectandlowmemorycontrolsPASS.

Actualmetadata: rootobservationsgate/up/activation/downareinput-specificrecords;
operatorQKVfilesstorefixedcoefficients/scales/gains/taps,live_inputs_stored=false.
Storedpairtablesstill exist outsidecentralroot; inspectedlocationonly, notusedor
regenerated. Mainoldweightsraw vscurrentcompressedrootformat ismaterial; historical
five-secondresultisvalidinitscontext, notreproducedbythischangedreader.

Fullrequestcomparison afterdecoder/CRC/pipelinechange: Rainfalls->' on the'
IDs418,276 in72.029005281s vsprior156.673926435s; Tea tastes->' like tea'
IDs1517,15600 in55.714099523s vsprior121.281682103s. Samepreviousoutputs,2input/
2outputeach; threeprocessedpositions/request. Warmoutputsteps18.6933/19.1319/
17.0570s, not5seconds. Firstheadfill15.2636s then~0.051s. Expertstep15.27-16.99s.
Eachrequest4416asyncwhole-expertreads (~69.77GBlogicalread),pipelinewait1.669s/
1.371s acrossrequest, staging75497472B, expertcache0, hints0,head2348810240B.
These are combinedchanges, notproofpipelineitselfwins;warmone-layerprobeabove
showedslightoverhead. Cold/disk behavior notisolated. Allprocessesclosedcleanly.

Reproduciblegeneratedreadercheckcaught tabsinmultilineJSreplacementliterals;
normalizedreplacementindentationwithoutchanginggeneratedC/criteria. ASan/UBSan/
leaktestpipelineintegrity passed badBF16CRC,badzlibheader,readerEOF,shutdown,
disabledmode; unchangeddecoderoracle24valid+118negativepassed under sanitizers.
Legacycacheconfigretainedseparatelyforoldcachetests, notshipdefault. No original
datasetsororiginalsources/fixtureschanged. Finishglobalcontainercomparisonand
payloadidentitychecks, thenreportimprovementwithoutclaimingfive-secondrestoration.

Finalstored-readerregressionpassed: sixseedrows,21504leafvalues,allfruitbytesand
163840scoresvsoriginaldata/arithmetic;186retainedbindings39154876416IDsand21780480
vectors/taps;14720observationrecordsreadercheck;372mappingreleases. Current128+128
controller,defaultpipelinebudgetandlegacycachebudgettestsPASS. Canonicalpayload
identities/size/mtime/ownershipunchanged;nocentrallinks. Productionbinaryrebuilt.
OriginalC/testfixturesandmodeldataunchanged. Remaininglatencygapismeasured, not
explainedaway; don'tlabelthisrestored5sectokens. Thefirstcache-centricresponse
missedthemoreimportantdecoderandschedulingregression; phasecostsshouldleadnextwork.

## Current: Reuse Fixed Data With Bounded Caches

The human clarifies that arbitrary input must work using existing computed values,
with RAM/disk caching and prefetch to avoid repeated fixed-data work. This resumes
live inference optimization, not recorded-answer substitution. Input-dependent
arithmetic remains necessary; no dataset reconstruction or persisted activation
cache is authorized. Use existing disk containers as backing and bounded RAM.

First local hypothesis: the full output head is decoded again for every output.
Read numeric-table.h confirmed a single16-row cache and a full10240-block walk per
projection. New head-cache.h retains verified decoded BF16bytes inRAM; original
numeric reader and arithmetic remain unchanged. Changed-input test compares all
163840scores and argmax against originalstreamingreader. ActualAX102results:
firstcachefill25.082345s vsstream24.285989s; nextdifferentinput0.050833s vsstream
24.378498s. Both exact, decodedblocksremain10240, secondcallreusesall10240blocks.
RAM2348810240bytes. This removes repeatedformatwork, not nextinputcalculations.
Zero-budget cache leaves original reader available. No modelrequest ordatawrites
in this firstcheck. Next: bounded expert-block caching and selected-weight prefetch.

Expertcache tests passed changed-input exact gate/up/down results onlayers1,2,92;
152blockmissesforfirstexpertuse, then152hits/zero newdecodes. Admissionprefers more
frequently selected experts with age decay and oldest-use tie-breaking; eviction
clears allvalidbits.32MiBtestbudget used18,522,288bytes. Prefetchissuesbounded
POSIX_FADV_WILLNEED on actually selected expert ranges, not predictedanswers.

Integratedlivecomparison onAX102: exactnative-tokenized "Rain falls" IDs91019,25528,
max_new_tokens2. Cacheoff emitted418,276(" on the") in179.911460913s; cacheon
emittedidenticalIDs in156.673926435s. Coldrequestheaddecoded10240blocks and reused
10240; experthits22648/misses648584. Sameenabledprocess thenprocesseddifferent
"Tea tastes" IDs149058,39466, emitted1517,15600(" like tea") in121.281682103s;
headdecodes0/reused20480;experthits76000/misses595232. Logicalcompressedbytesread
69,767,192,157off;67,411,693,798firston;61,864,271,869changedinput. These are logical
readbytes, not physicaldeviceIO; prefetchrequestedbytes are hints, not provenhits.
ExpertallocatedRAM17,040,504,960B<=16GiBbudget, head2,348,810,240B. Startup4.887soff,
4.984son. Singlepairedobservation, not acontrolledstatisticalspeedup or isolated
prefetchbenefit. No output replay; changedinputscores stillcomputed.

Configcachefields optional withdefaults2240MiBhead,16384MiBexperts,64MiBprefetch,
8192MiBreserve. Atstartup budget subtractsfixedmappingbytes, reserve and2GiBstate
allowancefromMemAvailable; lowmemoryreduces/disablescaches. Notdynamicevictionunder
latermemorypressure. Rootbudgetsdividedequallyacross92layers; OSpagecacheis disk
backing, noexpandedpersistentcachefiles. Cachevalidonlywhilemodeldataimmutable;
restartafterdatasetreplacement.128input/outputconfigunchanged. Nextfinishbuild/
reproducibility/sanitizerchecksanddocuments, noadditionalinferenceforreassurance.

Finalchecks: derive-root.mjs reproduces cachedlive-root.h fromtheexistingreader
with explicitcachewrappersinroot-cache-io.h; alltestsbuild. ASan/UBSan/leakchecks
onrootcachepassedchangedinputs, eviction, disabledcapacity andprefetchlimits.
Defaultconfigtests, originalgenerationbudgettests andintegratedinspectpassed.
Originalcentralpayloadunitidentities/size/nsmtime/ownership match frozenrecords;
no linksindataset. Cachecomparisonmodelprocessesclosedwith372mappingreleases.
Sourcecachefiles/config/tests/docsupdatedonlyinstanalone; originalnumericreader,
referenceC, residentexperiment, existingtestcriteria anddatasetsunchanged.
No full128-tokenrunor universalperformanceclaim. Headallocation is fullverified
BF16bytecache, not regeneratedparameters; originalcompressedfilesstayondisk.

## Paused: Unseen Input Without Live Expert Calculation

The human asks to avoid live expert calculations and test a different input.
Fresh AX102 inspection found the existing france/japan observation sets; Japan
metadata explicitly describes input-specific records, not an all-input cache.
The runtime currently calls resident_expert/root_project, while recorded_down
requires exact expert ID and all3584input coordinates. An explicit recorded-only
Japan test was offered with a stop on any miss; the human rejected it and selected
an arbitrary unseen prompt. No recorded-only mode was added and no test was run.

The unresolved requirement is how to obtain results for an unseen expert input
without live calculation or a matching stored result. Fixed parameter values
are not input-dependent outputs. Do not substitute a covered prompt or answer,
generate new observation files, or silently continue live inference. Clarify
whether the intended optimization is avoiding repeated parameter loading/decoding
while retaining necessary input-dependent arithmetic. Existing runtime/config/
datasets remain unchanged; current behavior is not claimed to meet this request.

## Random Input Demonstration

2026-10-02: the human requested a fresh random input/output test. PowerShell
Get-Random selected "Snow feels" from six short prompts. Original native K3
tokenizer check produced89446,16323; vocabulary decoding recovered the exact text.
Ran the current AX102 standalone run.sh with max_new_tokens2, without changing
the configured128input/128output budgets or any runtime source/data.

Actual emitted IDs1517,261 decode to " like a". Combined text: "Snow feels like a".
Stop reason length: two requested output tokens, not a complete sentence or EOS.
Startup4.975919447s; request182.480339416s; observed client process lifetime
194.009366s including shutdown/transport. Each output-producing step reported
4416liveexpertprojections and zeroexpertmatches; cleanEOFshutdown released372maps.
No recorded-output substitution, saved numerical vectors or dataset modification.
This is one observed demonstration, not a model-quality or latency guarantee.

## Current Direction: Configurable Live Generation

The human explicitly approves replacing recorded-result-only inference with live
expert calculations from the existing root datasets, and chooses separate limits:
128inputtokens plus128outputtokens. No dataset reconstruction, new observations,
saved activations or old backend substitution. Use startup-owned fixed parameters
and per-layer attention state in RAM. This supersedes the historical five-token/
observation-only contract, not any data-preservation or numerical check.

generation.h and bin/configs/generation.json parse independent budgets, bounded by
256total capacity, and JSON-lines input_ids with optional max_new_tokens. Controller
tests cover every1..128inputlength,128outputs,EOS,overflowandfailure using a synthetic
engine. That is not evidence of128realmodeloutputs. SharedJSONhelper unused-function
warnings initially blocked strictcompile; helpers exercised without disablingwarnings.

live-root.h derives from the tested layer1reader: metadata-driven counts, pread
instead of shared FILEcursor, parallel independent blocks, unchanged scalar-lane
math and decoded-value CRC. First test exposed layer1's older maps.json schema,
which omits layer/palettecount. Palettecount is recovered from existingconstants
length minus recordedmaps/templates/refs; exact layout/CRC and presentlayerchecks
remain. All92rootopens passed, and livegate/up/down match independentrecorded
projections for layers1,2,92. No storeddatawaschanged or regenerated.

clover-one.c now evaluates one active token perstep, keeping per-layer KDAstate/
convhistory and MLAkeys/values/positions across tokens, clearing them between
requests. NPOS=1means stepwidth, NOTinputlimit. Runtimeexpertpath calls root_project
three times perselectedexpert with originalSiTU between; no recordedresultfallback.
The controller prefillsallinputpositions, projectsheadonlyatlastinput, thenfeeds
generatedIDsforwarduntilEOSoroutputbudget. Configuration lives inbin/configs;
no France/Japanlabel in the newrequestprotocol. Priorrecordedreader remains only
for independenttests. Nextvalidatefullfive-tokenreference/livecontinuation and
the realCLI; do not claim full128-tokenmodelverification fromcontroller tests.

Real incremental test passed on AX102: at each of five historical input positions,
live expert inputs and outputs matched all1472independent observations for that
position,7360total. Prefill selected17374. Feeding that generated token atposition5
selected20829with zero runtimeobservationmatches and4416liveprojections. Alllayer
cachelengths advanced, then cleared for a newrequest. Fullrequest state staysRAM.
Old tests and recordeddata unchanged. The two measured outputsteps were70.524s
and69.947s: about43.7sexpertwork and24.8-25.4sheaddecode/projection. Not a controlled
benchmark, but a material limitation: the earlier1.86srecorded-result request does
NOTdescribe livegeneration. A full128-outputmodelrun has not been performed.

Real JSONservice check passed: one startup, two identical one-token requests with
max_new_tokens1 emitted identical tokens after independentcache resets;129input
and129outputrequests rejected beforeinference, twoTOKEN/DONEevents and cleanEOF
shutdown released372mappings. Requesttimes69.994sand67.110sonAX102. Controller
ASan/UBSan/leakchecks passed; alloriginalcentraldataunitidentities/size/mtime/mode/
ownership unchanged andnocentrallinks. CurrentREADYreports128input,128output,
256contextcapacity. A reporting-only missingprintfargumenterror was caught by
-Werror=format beforeexecution and corrected; samecheck/rebuild/inspect passed.

Currentrun.sh forwardsCLOVER_CONFIG(defaultbin/configs/generation.json) in a clean
environment. build.sh --tests compiles controller,root,incremental and retained
readertests. Configreadatstartup; JSON input_ids replaces oldresult-set protocol.
No full128-tokenrealmodelrun, no automatictexttokenizer and no performancepromise.
No originaltests/fixtureschanged, no datasetwrites or savedruntimevectors.

## Current: Resident Bound To Existing Computed Datasets

The human now selects the resident implementation and requests metadata-first,
stepwise integration with the current datasets, without regenerating or
reconstructing data. This supersedes the earlier exact-main-source copy direction.
Original main and resident sources stay unchanged. Todo tracking covers source/
metadata, adoption, bindings, validation and outcome. Arbitrary-input support and
full generation are not added; exact recorded expert matching remains required.

Read actual AX102 trunk build-values/manifest records, KDA/MLA operator manifests,
observation record-format/functions/launch metadata, and seed/fruit results. Trunks
use K3TRK001 records: kind1stored scales+palette IDs, kind2F32vectors, kind3F32router
rows. Operator metadata explicitly says live-input outputs are not stored. Expert
observations contain80records of51204bytes, each with int32ID, three3072F32arrays
and3584down-values, paired with3584F32inputs; not all-input or all-expert coverage.
Seed/fruit are16-row K3SEED1containers with distinct source tensor hashes. Leaves
are three BF16tensors in the existing JSON/base64 schema. Historical paths inside
metadata are provenance, not paths to recreate or execution instructions.

Adopted hash-verified resident source1e55e1f0794ad62e217803e029e5e2142033a85a6d1d2ef695fb7ac40074e67d
as clover-one.c. Original arithmetic flags compile passed before reader changes.
Copied five existing format-reader files byte-for-byte from distributed client
and normalization packages. No original helper/test files modified.

The new root configuration resolves explicit directory/CLOVER_DATASET/bin dataset
link, then binds K3_INDEX, K3_PREPARED_DATA, K3_OPERATOR_DIRECTORY and K3_TRUNK0_QKV
to central paths. Original prepared/operator/observation readers remain. Global
embedding/tail/head accesses now use seed.bin/fruit.bin/leaves.json directly;
old index checkpoint paths are not mapped. Containers are decoded in bounded
blocks, never reconstructed into full matrices or saved. Original final AR/RMS
arithmetic remains; the existing leaves reader prepares the same fold once.

First integrated --inspect passed372fixed mappings/69tap layouts and shutdown,
zero requests. Its1.134546095sstartup was without page-touch warmup, not a5.8s
comparison or new performance claim. Then test-datasets.c passed original Bf
versus newhead all163840scores/argmax, every decoded fruit byte, six seed rows,
all21504leafvalues/fold,186forward/reverse retained bindings with39154876416IDs
and21780480vector/tapvalues, and14720existing expert records. All372mappings
released. The original central shard is used only by this test as independent
data, not by runtime. No full model request or persisted vectors occurred.

No dataset payload generation/reconstruction or move occurred. Full old numerical
performance is unmeasured: head streaming/decode differs from old resident raw
BF16mmap. Remain explicit about five-token/observation-match limits. Build/run
scripts add a bin dataset link only and keep compiled programs in bin; no Windows
runtime port is claimed. Earlier sections below are historical observations.

Final packaging: build.sh compiles only code and creates a relative bin/dataset
link to ../../../dataset onAX102. Local bin/dataset is a junction to the same
central root. run.sh clears inherited instrumentation, uses16OpenMPthreads with
close/core affinity, and disables core dumps. Executable-relative default mapping
passed --inspect without an explicit path. All central payload unit identities,
size/mtime/mode/ownership still match the frozen original-path relocation records;
dataset still contains no links. Original resident source hash is unchanged.

Direct source comparison confirmed prepared_load, operator_load, recorded_down,
result_options, request_reset and the entire layer arithmetic body are unchanged.
test-boundaries passed seed/fruit identity crossing, invalid token index, exact
expert-input mismatch with exit3and nofallback, and missing root rejection before
startup. These are reader/control checks, not a full forward request. Current
verification stops here; no numerical tolerance or original test was modified.
The original shard was read only as independent test evidence. No new model
input/output dataset or complete expanded parameter table was produced.

2026-10-02: the human requests copying the main repository clover-k3.c into
code/standalone-hosting as clover-one.c. This is a copy, not a move; the original
clover-k3/clover-k3.c and all experimental variants remain untouched.

The copied source matches repository revision
0eb364c12cade5697090688d56c8e224c80501ec. Complete SHA256 of the original and both
local/AX102 copies:

```text
5628f7c3d2b7932b2b8776dcb498d54bbc730b1785ba7819989bdb51b6604e65
```

AX102 path: /opt/clover-k3/clover-intelegence/code/standalone-hosting/clover-one.c.
No source edits, dataset mappings, build, runtime link, inference or performance
measurement were made. The source still expects the original checkpoint and
packed-trunk formats; the prepared datasets are not automatically compatible.
The distributed packages and central dataset layout are unchanged. Any adaptation
of this source is separate work; the exact copy is the verified outcome here.

## Source And Version Analysis

2026-10-02: requested analysis, not code changes. git diff --no-index confirms
clover-one.c still matches the main repository source. Read its actual main,
attention/MoE/tail operations, route cache guards and prefix IO, plus Git history
and the separate experimental readers. No model run or new benchmark occurred.

The main lineage has six source revisions in available local Git refs:
- c6dbc1f: initial committed optimized one-shot forward implementation; already
	has batched projections, dequantization tables, threaded reads, KDA/MLA and
	prefix save/load. It is not the earliest unoptimized equation experiment.
- 5952a34: source header/build naming cleanup; no C execution changes.
- 55415ce: require explicit K3_INDEX/K3_TRUNKPATH, centralize script config and
	change default output location; arithmetic unchanged.
- f4f7844: optional prompt-keyed routing cache plus dual-arena next-layer weight
	reads. Live router still runs and cached rows are checked. Expert projections
	still run; the cache stores IDs, not expert answers. K3_NX defaults off.
- a74dfd0: optional CPU-cache prefetch of the next distinct expert's weight data
	during Xm. K3_PFCACHE defaults off; recorded tests showed no gain or slowdown.
- 0eb364c: K3_PFXOUT writes updated KDA/MLA state, including loaded prefix state,
	enabling the separate gen.py driver to continue prefix reuse across steps.
	The C program remains one-shot; generation restarts it and exchanges state files.

Current input is numeric K3_IDS with compile-time NPOS (default5), not text/BPE.
It embeds each ID, runs93layers (69KDA/24MLA; dense layer0, routed/shared experts
layers1..92), aggregates snapshots, normalizes and projects the BF16 output head,
then prints an argmax ID. It uses original packed trunk and indexed expert shards,
not the central prepared formats. Linux APIs, pthreads/OpenMP and optional AVX2
mean the exact copy is not a portable/reentrant multi-request service.

Important unadapted behavior: main unconditionally opens K3_LOGITS or default
clover-k3-logits.bin and writes the normalized vector plus logits. Prefix/route
options also persist data. No-persistence was NOT implemented by copying it; do
not run its default path under the current RAM/console-only boundary. Its printed
engine token17374 comparison is fixed France-example text, not an independent
correctness check for arbitrary inputs. Header claims about no prefetching are
stale relative to the actual implementation.

Separate experiment variants are not further commits to the main source:
- computed-values: prepared trunks and compressed-root/pair-table readers;
	performs live expert gate/up/down projections, with zlib for placement decoding.
- recorded-results/expert-only-probe: replaces expert projections by exact expert
	ID + full3584-value input matches; QKV/shared arithmetic remains live.
- recorded-results combined: also consumes completed QKV/shared outputs; recorded
	run stopped on a layer1expert-input mismatch, without a fallback or final token.
- operator-values: derives from the expert-only probe, uses fixed operator values
	for live QKV, and retains exact-match recorded expert results; per-layer mappings.
- resident: retains operator-values fixed mappings/taps from startup to shutdown,
	serves serial independent five-token requests and resets attention state between
	requests. Expert-result matching remains; this is not incremental generation.

The historical resident5.8196sstartup/1.8585swarm request numbers therefore do not
describe the main live-expert implementation or the copied standalone source.
All timing references remain recorded hardware/configuration-specific observations,
not a new controlled comparison. Analysis changes only this context record.

## Resident Startup Versus User Input

Follow-up question asks whether the measured5.8second resident startup depends on
static user input. Fresh source trace: main calls resident_startup and prints
READY before fgets reads any request. Startup maps all fixed trunk/operator and
global embedding/tail/head parameters, copies taps and optionally touches pages.
It does not embed a supplied prompt or choose experts for it. This startup is
input-value-independent;5.8seconds is one recorded observation, not fixed latency.

The limitation is request-time behavior, not startup: the protocol requires an
observation-set label and exactly5IDs; evaluate_request resets state and embeds
those supplied IDs. Live routing later calls recorded_down, which requires an
exact expert ID and full3584-coordinate input match, exiting on a miss. Expert
observations are opened then, not during startup. Thus this resident does not
support arbitrary user prompts even though its fixed-parameter startup is not
prompt-dependent. General support would need live expert computation and a
variable-length token interface; the1.86second measured request is not a promise
for that different workload. No code change or execution occurred in this check.