# Standalone Resident Hosting

## Cross-Layer Prefetch

The default now starts one next-layer candidate read at the end of the current
layer, overlapping the next layer's attention. The predictor applies the next
layer's stored router and bias to the current final residual normalized with the
next post-MLP gain. This omits the next attention/aggregation, so it is explicitly
a prediction, not the model's final selection. It requires no earlier request or
prompt-specific routing file.

The live router still computes all scores and the actual top 16. Only a candidate
matching the correct layer and a selected expert with no exact-result cache hit
is reused. It is evaluated first, but outputs are retained by rank and mixed in
the original order. A wrong or unnecessary candidate is drained and discarded;
normal selected-expert reads follow. Original decompression, CRC and shape guards
apply before computation. A prediction never supplies an answer or changes routing.

The existing two buffers still total 72 MiB. This bounded implementation prefetches
one candidate across each layer boundary, not all 16 experts. Later reads overlap
individual expert calculations as before. Setting `expert_pipeline_mib` to 0
disables both read-ahead paths; `CLOVER_NO_CROSS_LAYER_PREFETCH` at compile time
disables only cross-layer speculation for comparison.

### Fresh-Request Measurement

AX102, 16 threads, separate fresh processes, two input tokens (Rain falls), one
output (` on`), result caching disabled in both:

| Schedule | Request time | Exposed pipeline wait | Expert reads |
|---|---:|---:|---:|
| Shared-stage overlap, no cross-layer prediction | 47.116 s | 0.286418 s | 2,944 |
| Shared-stage plus cross-layer prediction | 45.028 s | 0.004083 s | 2,986 |

Of 184 candidates, 142 matched the live selection and 42 missed. Prediction took
0.108841 s; misses added 663,946,404 logical read bytes. Total logical bytes were
46,508,564,103 versus 47,172,510,507, not measurements of physical disk traffic.
Startup was separate, 4.924/4.938 s. Each request included first head-cache fill;
both decoded 10,240 head blocks. Output IDs matched with zero result-cache hits.

This was one sequential pair, baseline first, without flushing OS caches. It
shows correct cross-layer reuse and lower exposed reader wait in this run, not a
repeatable 2.09-second gain. Prediction adds work and wrong reads can cost time.
Five-second fresh tokens and full 128-input/128-output execution remain unverified.

### Verification

`bash build.sh --tests` builds the new checks and a comparison binary with only
cross-layer prediction disabled. On AX102, from this directory, run:

```text
OMP_NUM_THREADS=16 ./bin/test-cross-layer-prefetch ../../dataset
OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores ./bin/test-cross-layer-model ../../dataset
```

From the local machine, `node test-cross-layer-service.mjs SSH_EXECUTABLE HOST`
runs the fresh-process pair using [fresh-request.json](bin/configs/fresh-request.json).
The model check compared 2,944 independent expert input/output pairs exactly,
then checked result-cache coexistence: a repeated single-token computation reused
all 1,472 expert results with zero live expert projections and unchanged values.
That repeat is a correctness check, not the fresh-request performance measurement.
Candidate lifecycle tests passed address/undefined-behavior/leak sanitizers. The
unchanged pipeline corruption/EOF test passed. All test targets compiled; a full
test-suite run or new complete stage profile is not claimed.

CACHE_JSON adds `cross_layer_submitted`, `cross_layer_hits`, `cross_layer_misses`
and `cross_layer_prediction_seconds`. With the pipeline enabled, expert reads
equal result-cache misses plus discarded cross-layer candidates. Exact-result
hits avoid live expert calculations, but an early speculative read may already
have occurred before that hit is known. Older service drivers assume no such
speculative traffic; they and the prior reports are preserved as historical
checks, not the current cross-layer comparison driver.

## Earlier Shared-Stage Overlap

After the current layer routes its input, the reader starts loading the first
missing expert. The independent shared-expert branch now runs during that read,
before the consumer waits for the expert bytes. Later expert reads continue to
overlap the current expert's calculation. This scheduling does not require a
previous request, cached route, or repeated prompt. Arithmetic inside each branch
and the final ordered merge are unchanged.

AX102, 16 OpenMP threads, one sequential comparison with result caching disabled:

| Schedule | Request time | Pipeline wait | Result hits |
|---|---:|---:|---:|
| Shared branch after routed experts | 49.310 s | 0.937 s | 0 |
| Shared branch during first expert read | 46.989 s | 0.308 s | 0 |

Each separate process received `[91019,25528]` (Rain falls), generated `[418]`
(` on`), and performed all 2,944 selected expert reads. Startup was separate:
4.968/4.930 s. Both request times include first-use head-cache fill. The processes
were fresh, but OS caches were not flushed, and the baseline ran first. This pair
shows less measured reader wait with identical output; it does not establish a
repeatable end-to-end speedup or restore five-second tokens.

The fixed non-routed parameters for all layers are already mapped and touched at
startup, not locked into physical RAM or CPU cache. An unseen input's next-layer
expert selection is known only after that layer's router runs. This earlier
measurement predates the validated speculative cross-layer path described above.

`bash build.sh --tests` builds both schedules and the focused stage test. Run
`OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores ./bin/test-stage-overlap ../../dataset`
from this directory on AX102 for byte-exact shared/expert comparisons on layers
1, 3 and 92. The historical test-fresh-overlap.mjs recorded the fresh-process pair
above before cross-layer speculation. It uses [fresh-request.json](bin/configs/fresh-request.json), which keeps
the head cache and read pipeline enabled but disables expert-result caching.
The default result-cache setting is unchanged. Profile accounting includes the
new shared-during-read and merge boundaries; existing recorded reports are retained.

## Computed Expert Results In RAM

The runtime now retains completed expert `down` vectors, not just IDs or weights.
An exact hit skips gate, up, activation, down, and demand expert-weight reads and
decode. An already-started speculative read is discarded when unnecessary.
Mixing still applies the current routing weights in rank
order, followed by the original normalization and projection.

The key is the loaded model's cache instance, layer, expert ID, and all 3,584 input
float bytes. A hash narrows the lookup; full byte comparison is required, including
signed zero. Thus a different input with the same expert or hash cannot receive
the wrong result. Cache hits are copied before later inserts can evict entries.

`expert_result_cache_mib` defaults to 256 in
[generation.json](bin/configs/generation.json). The effective allocation is 235,077,632
bytes for 8,192 input/output entries. It fits within the existing memory-headroom
budget and can be disabled with 0. Entries use bounded eight-way sets with LRU
eviction; collisions or capacity pressure can cause repeat misses. Allocation
failure disables this optional cache rather than changing model results.

The cache survives independent requests in the same process, but is empty at
startup and freed on shutdown. It is never written to disk and does not preload
the historical observation datasets. Dataset files must remain immutable while
the model is loaded; restart after a model change. Only `down` is retained because
that is what the next stage consumes; storing gate/up/activation too is unnecessary
for this reuse. The existing attention-cache reset between requests is unchanged.

### Measured Reuse

Actual AX102 run, two input tokens and two output tokens for each request:

| Request | Actual continuation | Time | Result hits | Computed misses / expert reads |
|---|---|---:|---:|---:|
| Rain falls, first run | ` on the` | 65.79 s | 0 | 4,416 |
| Rain falls, exact repeat | ` on the` | 7.90 s | 4,083 | 333 |
| Tea tastes, changed input | ` like tea` | 50.91 s | 0 | 4,416 |

All output IDs matched the earlier uncached runs. Exact repeat reused about 92.5%
of expert results, and each hit avoided its prefetch entirely. The first request
also pays for cold head-cache fill, so these whole-request timings are not an
isolated expert-cache speedup. They do not imply unseen prompts will run in 7.90 s
or establish performance for a full 128-token generation.

READY_JSON reports result-cache bytes and capacity. CACHE_JSON distinguishes
`result_hits`, `result_misses`, `result_stores`, and `result_evictions` from the
older fixed-weight-cache counters. STEP_JSON reports `expert_result_hits` and
actual projection calls. Live projections plus three per result hit must account
for all 4,416 projections per processed position; no selected expert is omitted.

`bash build.sh --tests` builds test-expert-results and test-result-config. Tests
passed exact input/layer/expert/model-owner isolation, forced hash collisions,
signed zero, eviction, copied-hit lifetime, disabled mode and low-memory limits,
including address/undefined-behavior/leak checks. test-result-service.mjs verifies
the real repeated/changed-input behavior above. Original datasets remain unchanged.

## Stage-By-Stage Evidence

[STAGE-PROFILE.md](STAGE-PROFILE.md) records input timings, all 93 layer totals,
individual stage/projection timings and per-layer decode/CRC/math worker times.
It follows the documented workflow and real executable path, not the website's
recorded example as a substitute for measurements. The 279 layer totals per run
cover two input positions and a continuation, with the same output-ID assertions.
No activation arrays were saved.
That profiling report predates the computed-result RAM cache, shared-stage
overlap and cross-layer prefetch; it describes the earlier miss path, not a warmed exact-repeat request
or a measurement of the new schedule.

On AX102 the warm native text-to-ID path averaged 0.000371 ms and cached embedding
lookup about 0.0023 ms/token. First compressed-row reads were 1.60-1.77 ms. Layer 0
took about 29 ms; all layer attention about 1.02 s and routed expert/mix work about
15.14 s for the measured continuation. The warm head was 50.91 ms; first head fill
was about 15 s and is reported separately. Startup is not charged to lookup.

One measured decoder change batches four literal symbols from a bounded bit window.
It passed the unchanged 24-case oracle and 118 rejection controls under sanitizers,
all output-head byte/score and stored-reader tests, and the full profile's output
checks. The corrected continuation profile moved from 17.02 to 16.69 s, not to
five seconds. A larger Huffman lookup did not reduce decode CPU time and was rejected.
Per-layer worker sums and nested projection times must not be added to wall totals.
These instrumented observations are not a controlled performance guarantee.

Build the numerical profiler with `bash build.sh --tests`. From this directory,
the capture and report tools retain timing JSON only and refuse to overwrite an
existing report:

```text
node capture-stage-profile.mjs SSH_EXECUTABLE HOST NEW_TIMING_JSON
node summarize-stage-profile.mjs BEFORE_JSON AFTER_JSON NEW_REPORT_MD
```

The capture driver currently profiles the recorded two-position test and one
continuation on AX102. The tokenizer-only profile is test tooling using the original
native K3 headers, not a new tokenizer dependency in production. All profiling
hooks compile out of normal builds. See [context](CONTEXT.md) for the corrected
capture/timer issues and remaining expert decoding bottleneck.

## Current Live Generation

[clover-one.c](clover-one.c) now accepts variable-length token-ID requests and
generates output using exact cached expert results or live calculations on misses
from the existing stored root values.
Fixed parameters stay startup-owned. Per-layer KDA/MLA state stays in RAM across
tokens and resets between requests. Production does not read expert observations.
No dataset values are regenerated. Fixed decoded bytes may be cached in bounded
RAM; there is no expanded float-weight matrix or persistent reconstructed dataset.

[bin/configs/generation.json](bin/configs/generation.json) sets independent budgets:

```json
{
  "max_input_tokens": 128,
  "max_output_tokens": 128,
  "eos_token_id": 163585,
  "head_cache_mib": 2240,
  "expert_cache_mib": 0,
  "prefetch_mib": 0,
  "expert_pipeline_mib": 72,
  "expert_result_cache_mib": 256,
  "memory_reserve_mib": 8192
}
```

The sum cannot exceed the compiled context capacity of 256. Restart after editing
the config; CLOVER_CONFIG can select an alternative. NPOS=1 is the active step
width, not the input limit. The prompt is processed incrementally, and continuation
does not recompute its earlier positions.

Build with `bash build.sh --tests`; inspect with `bash run.sh --inspect`. To serve
requests, run `bash run.sh`, wait for READY_JSON, then send one JSON object per line:

```json
{"input_ids":[2108,11989,682,318,2225,876,387],"max_new_tokens":2}
```

Omitting max_new_tokens uses the configured maximum. Empty input, invalid IDs,
oversized input and invalid output budgets are rejected before inference.
TOKEN_JSON streams each generated ID. DONE_JSON reports counts and stop_reason
of eos or length. EOS is included in the output count. EOF shuts down the process.
The interface accepts token IDs, not raw text; tokenization/decoding is the caller's
responsibility. No France/Japan dataset label is required.

The live reader uses root-N/maps.json, constants.bin and experts.bin with original
CRC checks and reduction order. Reads are cursor-free and independent blocks can
run in parallel. Layer1's older metadata schema is supported. Existing seed/fruit,
leaves and prepared trunk/operator readers remain; only bin/dataset is a link.

### Current Read Pipeline

The default no longer allocates a 16 GiB expert-weight cache or relies on WILLNEED hints.
A persistent reader uses two compressed buffers, 72 MiB total: after routing
selects the experts and exact-result lookups identify misses, it primes the first
missing expert while the shared branch is calculated, then fetches the next
missing expert while the current one is being evaluated. A consumer waits for
completion and applies the unchanged
format/integrity checks before using the bytes. The buffers are reused, not a
persistent weight cache. The small pipeline budget participates in the same
startup memory-headroom check. Set expert_pipeline_mib to 0 to disable it.

This within-layer pipeline now also accepts the verified cross-layer candidate
described above. The historical cross-layer nx_begin instead required a
prompt-keyed routing cache. The historical generation driver instead enabled
threaded raw-weight reads, AVX2 expert arithmetic and prefix-state reuse; it did
not itself enable K3_NX. Its recorded five-second decode must not be conflated
with the later 1.86-second recorded-expert-result request.

The decoder now uses a 9-bit short Huffman lookup, bounded multi-bit reads and
chunked Adler arithmetic. Decoded expert BF16 CRC uses eight-byte updates with
small checksum tables. These are format/checksum optimizations, not new weights
or weaker validation. All original 24 oracle cases and 118 rejection controls
passed, including under sanitizers. Original root CRC values and independent
expert results still match. Corrupted prefetched data and short reads fail.

The measured three-layer probe changed from about 0.456 to 0.151 seconds per layer.
Originally the summed worker times were approximately 4.14 s decompressing,
1.75 s validating, and 0.36 s doing projections; thus missing AVX2 was not the
dominant measured phase. Afterward, decode was about 1.37 s and CRC 0.448 s.
Worker-time sums are not wall times and cannot be added to predict end-to-end time.

Full AX102 requests, each with two input and two output tokens:

| Input | Actual continuation | Previous large-cache version | Current pipeline/decoder |
|---|---|---:|---:|
| Rain falls | ` on the` | 156.67 s | 72.03 s |
| Tea tastes | ` like tea` | 121.28 s | 55.71 s |

Output IDs are identical to the prior tests. The second output and warmed-process
output steps took about 17-19 s; five-second token latency is NOT restored.
First head fill took 15.26 s, subsequent head projections about 0.051 s. Read-ahead
wait totaled 1.67/1.37 s per request. Both used zero expert-cache RAM, 72 MiB staging
and 2.35 GB head-cache RAM. These are combined-change observations, not isolated
proof that pipelining helps: a warm one-layer probe measured 0.172 s pipelined
versus 0.162 s direct. No fresh cold-disk or historical raw-reader comparison ran.

### What Stored Values Mean

| Stored content | What can be reused |
|---|---|
| Root constants, maps and placement blocks | Learned expert weights for any input; no weight regeneration is needed |
| Observed gate/up/activation/down vectors | Results for an exact recorded expert/input vector; useful as independent checks or exact-match reuse, not answers for unrelated inputs |
| Current QKV operator files | Fixed coefficients, row scales, gains and convolution taps used directly by every request; metadata explicitly says no live input values are stored |
| Historical completed QKV captures | Input-specific intermediate outputs, not interchangeable with the fixed operator files |

Gate and up apply learned projections to the current input. Activation combines
those results through SiTU. Down projects that activation back to the expert's
output width. Changing the input can change all four outputs. Precomputed fixed
parameters avoid rebuilding the weights, but do not remove these input-dependent
operations. Production does not read another prompt's observed outputs.

Build/tests include test-expert-pipeline, test-pipeline-integrity, test-root-phases,
test-pipeline-config and the unchanged decoder oracle harness. test-pipeline-service.mjs
checks the real output/counter comparison. test-datasets passed all original head
bytes/scores and retained-value checks with the faster decoder. Original model
data identities, lengths, timestamps and ownership remain unchanged.

### Earlier Fixed-Data Cache Approach

The following records the preceding optional large-cache implementation, now off
by default. Its measurements predate the optimized decoder and real read pipeline.
The head cache remains enabled. Legacy cache tests use
[cache-legacy.json](bin/configs/cache-legacy.json); historical test-cache-service.mjs
assumes the previous defaults and is not the current pipeline comparison driver.

- The head cache holds 2,348,810,240 verified BF16 bytes when the budget permits.
  It fills on the first output projection, then reuses those bytes for new inputs.
  Row calculations are parallel; the arithmetic/reduction order within a row is
  unchanged. It does not cache logits, generated tokens or prompt answers.
- The expert cache keeps verified decoded placement blocks, not expanded weights.
  Its total 16 GiB limit is divided across 92 layers, with lazy allocation.
  Frequency-aware admission and aging retain frequently selected experts; eviction
  invalidates their blocks before reuse. Each cold block still passes the original
  decoded-value CRC checks. A hit avoids both the file read and decompression.
- The existing compressed disk files remain the backing store. The OS page cache
  services misses where possible. Once the router selects experts, bounded
  POSIX_FADV_WILLNEED hints request up to 64 MiB per layer of their actual ranges;
  fully cached experts are skipped. Hints do not guarantee physical prefetch.
- Caches live until process shutdown and are reused across independent requests.
  Attention state resets between requests, while fixed weight caches do not.
  Dataset files must stay immutable while running; restart after replacing them.

At startup, available RAM minus the fixed mappings, an 8 GiB reserve and a 2 GiB
state allowance bounds the effective cache budgets. Insufficient headroom reduces
or disables caches. This is a startup allocation policy, not a guarantee against
memory pressure from other programs later. Set a cache budget to 0 to disable it;
set prefetch_mib to 0 to disable hints. No host memory settings are changed.

READY_JSON reports effective cache budgets. CACHE_JSON reports per-request expert
hits/misses/evictions, logical compressed bytes read, prefetch bytes requested,
allocated RAM, and cached-path head fills/reuses. Logical read bytes are not
physical disk traffic; disabled head-cache mode does not count its streaming
reader's decodes in the head-cache counters.

### Cache Measurements

On AX102, every one of 163,840 head scores and argmax matched the original reader
for two different vectors. Initial fill cost 25.08 s; a second different vector
took 0.051 s from cache versus 24.38 s streaming, with no new decodes. Expert tests
at layers 1, 2 and 92 matched cached/uncached projections on different inputs, with
152 misses initially then 152 hits and zero additional decodes. Memory bounds,
eviction invalidation, prefetch limits, low-memory budgets and disabled mode passed.
The expert-cache tests also passed address/undefined-behavior/leak checks.

Two-token live output comparison (each prompt had two input tokens):

| Input | Mode | Actual continuation | Request time |
|---|---|---|---:|
| Rain falls | Cache disabled | ` on the` | 179.91 s |
| Rain falls | Cache enabled, initially empty | ` on the` | 156.67 s |
| Tea tastes | Same enabled process, changed input | ` like tea` | 121.28 s |

The enabled requests used 17,040,504,960 bytes of expert RAM within the 16 GiB
budget, plus the 2.35 GB head cache. The changed input reused all head blocks and
had 76,000 expert-block hits, but 595,232 expert blocks still missed. That remaining
work explains why this is not a universal near-zero-latency cache. Outputs were
computed for each input, not substituted from prior observations.

These are individual measurements, not a statistically controlled speedup or proof
of prefetch's isolated benefit. The two different prompts cannot be compared as
equal workloads. First-use decoding still costs time, and cache allocations are
additional to the existing fixed mappings. No full 128-output model run was added.

`bash build.sh --tests` includes test-head-cache, test-root-cache and test-cache-config.
test-cache-service.mjs runs the live enabled/disabled comparison over SSH using
[cache-disabled.json](bin/configs/cache-disabled.json). No model-data preparation
is part of building or testing these caches.

### Current Verification

- Synthetic controller tests covered input lengths 1..128, 128 outputs, EOS,
  malformed requests, budget overflow and failed steps; strict compilation and
  address/undefined-behavior sanitizers passed.
- All92root readers loaded. Live gate/up/down projections matched independent
  recorded values at layers1,2,92.
- Real five-token incremental prefill matched all7360independent expert inputs
  and outputs, selected17374, then generated20829 with retained state and zero
  runtime observation reuse. Request reset cleared all attention caches.
- Real JSON one-token requests, repeated-request reset,129input/output rejection,
  bounded streaming and EOF shutdown passed.
- Original central dataset identities/lengths/timestamps/ownership are unchanged.

The full128-input/128-output model workload has NOT been executed. Maximum-budget
coverage comes from controller tests, not a long model run. These bounded checks
do not establish general model quality. Existing originals and tests are unchanged.

Before these caches, observed AX102 output-producing steps took about67-71seconds: roughly44seconds
expert work and25seconds head decoding/projection in the reference continuation.
These are individual observations, not a controlled benchmark or latency guarantee.
The old1.86second recorded-result request timing does not apply to live generation.

Run the inexpensive controller check with
`./bin/test-generation bin/configs/generation.json`. test-live-root and
test-incremental perform longer real calculations. test-service.mjs exercises the
service over SSH. run.sh disables core dumps; all runtime vectors stay in RAM.

## Historical Recorded-Result Integration

The sections below describe the earlier bounded implementation and its reader
checks. Their five-token/observation-only/no-generation claims are superseded by
the current live-generation contract above. The recorded reader remains test-only.

[clover-one.c](clover-one.c) now uses the selected resident implementation, with
startup-owned fixed parameters and the current central datasets. The original
main reference and resident experiment remain unchanged. The source originally
copied here from revision0eb364c has been superseded at the human's direction.

## Stored Data

All real payloads remain in ../../dataset. Only bin/dataset is a link to that root.

| Path under dataset | Use |
|---|---|
| eqidx.bin | Existing slot identities/shapes; old checkpoint paths are not opened by this runtime |
| trunk-0 through trunk-92 | Prepared K3TRK001 records, stored palettes/scales, F32 vectors/router rows and folded constants |
| operators/trunk-0-qkv/qkv.bin | Canonical layer0 fixed QKV operator |
| operators/qkv-all/layer-N/operator.bin | Fixed KDA/MLA operator records for layers1..92 |
| inputs/seed.bin | Stored embedding rows in the lossless K3SEED1 container |
| outputs/fruit.bin | Stored BF16 output-head rows in the same container family |
| leaves.json | Three stored global BF16 tensors; decoded once and the original fold prepared once at startup |
| root-N/observations/{france,japan}/ | Existing exact-input expert results and their input records |

The selected resident does not consume root-N/experts.bin or rebuild expert
weights. Fixed QKV files explicitly contain no token-conditioned outputs: applying
their coefficients to live inputs remains necessary. Expert-result lookup retains
the original expert-ID plus full3584-coordinate equality guard and exits on a
miss. It is not arbitrary-prompt inference or a generation loop. The protocol
remains an observation-set label followed by exactly five token IDs.

The copied numeric-table/decode/SHA256 readers and normalization-values/json
readers come from the already-tested distributed packages. They introduce no new
third-party library. Input/head decoding uses bounded16-row buffers, not complete
expanded matrices. Leaf decoding is format conversion, not parameter generation.
Original layer arithmetic and its exact-match expert guards are retained.

## Build And Inspect

Linux, GCC, libm, OpenMP and pthread support are required. Build on the target
machine because the preserved compiler flags include -march=native.

```bash
bash build.sh --tests
bash run.sh --inspect
```

Inspect loads and validates the current datasets, then releases372fixed mappings
without accepting a request or touching every mapped payload page. It is not a
benchmark of normal warmed startup. Normal run.sh starts the resident with its
original page-warmup default and keeps stdin open for serial independent requests.
It disables core dumps and clears inherited diagnostic/persistence options.

The program resolves data from an explicit DATASET_DIRECTORY argument, then
CLOVER_DATASET, then bin/dataset beside its executable. It binds all existing K3
reader paths to that one root. A Windows junction mirrors the bin layout locally,
but no Windows executable is built; the resident uses Linux-specific APIs.

## Verified Scope

- Central-data startup inspection passed:372mappings,69KDA tap layouts, zero requests.
- Six embedding rows, all21504leafvalues and the folded direction matched the
  existing central original shard. The shard is a test oracle only, not runtime data.
- Every output-table byte and all163840scores matched the resident's original Bf
  arithmetic on a deterministic test vector, including argmax.
- The original lifetime comparison's186forward/reverse bindings matched
  39,154,876,416coefficient IDs and21,780,480vector/tap values, without remapping.
- All14,720existing expert observation records resolved and matched across both
  stored sets. This verifies readers, not coverage for an arbitrary new input.
- Crossed seed/fruit containers, out-of-range token IDs, a missing dataset root
  and an unmatched expert input were rejected. Expert misses retain exit3 with
  no projection fallback.

To run the scoped checks on AX102 after building:

```bash
ulimit -c 0
OMP_NUM_THREADS=16 OMP_PROC_BIND=close OMP_PLACES=cores ./bin/test-datasets ../../dataset
./bin/test-boundaries ../../dataset
```

No dataset was regenerated, reconstructed or relocated. Tests held numerical
vectors only in memory. No complete model request, generation or fresh latency
benchmark has run for this integration. The old5.8sstartup/1.86srequest figures
do not characterize this build: input/head now use compressed-container readers,
and head rows are streamed and decoded during projection. See [context](CONTEXT.md).