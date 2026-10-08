# Standalone Hosting Source

## Reduced Expert Computation: Measured, And It Does Not Exist

### Intended outcome

If an expert's transformation needed less than its whole matrix, the 17.5 MB
read per expert could shrink and the data movement would go away rather than
being hidden. Trace input -> computation -> output for gate, up, activation and
down, then look for a reduced form giving the same output.

### What the trace is

`root-N/observations/{france,japan}/` already holds it, 80 records a layer, and
`record-format.json` states the layout rather than leaving it to inference:
`id int32, gate float32[3072], up float32[3072], activation float32[3072],
down float32[3584]`, with `inputs.f32` holding the 3584-float input per record.
No engine run was needed.

### What the system showed

`y = sum_j h_j D[:,j]`, so a zero entry of h never touches its column of down.
L2 mass of the activation in its top-k of 3072:

| | top-128 | top-512 | top-1024 |
|---|---|---|---|
| L1 e498 | 98.3% | 99.5% | 99.9% |
| L1 e545 | 68.7% | 92.6% | 98.6% |
| L45 e460 | 99.8% | 99.9% | 100% |
| L90 e128 | 79.7% | 94.6% | 98.6% |

One sixth of the coordinates carry 93-99.9% of the answer, so five sixths of
`down` is dead weight for any given input.

But not the same sixth twice. Same expert, two different inputs, top-512 overlap
27.7% and 28.7% against a 16.7% random baseline, and the union over two inputs
is already 882 and 877 of 3072.

And gate alone does not identify the carriers: its top-K against the true top-K
is 52-66% at K=512 against 17% random. Three to four times better than chance,
missing about 40%.

### Why this closes the direction

The output is low-information relative to its weight bytes, but which bytes
matter is decided by the input, and the computation that identifies them is the
one being avoided. The stored matrix cannot be pruned once, because the union
over inputs grows toward the whole matrix. Dynamic partial reads would need a
dependent second round trip after gate and up, which adds latency rather than
removing it, and is lossy on top.

### What could not be established

At layer 90 the recorded `activation` differs from `silu(gate)*up` by up to 0.45
absolute against a 3.9 peak, 12%, while layers 1 and 45 agree to under 0.1%. The
reason is unknown. The concentration and stability results do not depend on the
formula, being measured on the recorded activation itself, but no claim is made
that `silu(gate)*up` holds at every depth.

### Sample size

Three layers, 1, 45 and 90. Twelve (layer, expert, input) samples for the gate
prediction and two inputs per expert for stability. Thin for a positive claim;
the three measurements agree with each other and all point the same way.

## Current: Where A Single Token Goes, And Why The Expert Cache Was Rejected Wrongly

### Intended outcome

Batching bought throughput and nothing else. This cycle went after latency: take
one prompt apart stage by stage on the current engine and let the numbers say
what is addressable.

### What was known going in

A decode token was 3.4-3.7 s and an output token was a full forward pass, not a
lookup. `profile-stages.c` existed but predated every recent change and no
longer compiled: it called `resident_sequence_clear`, which the lane work
removed, and it profiled single positions through `evaluate_token` rather than
the batched shapes the engine now runs.

### What the system showed

One decode token, 3474 ms, spread almost evenly over 92 MoE layers at 36.5 ms
mean, range 33-44. No hot layer, so nothing local to attack. Per layer:

| per layer | ms |
|---|---|
| attention, which runs **before** the router | 9.8 |
| shared expert | 3.7 |
| expert projection | 10.4 |
| **expert read stall** | **11.4** |
| rest | 1.2 |

Expert read is 280.8 MB a layer, exactly 16 x 17,547,264 B, no reuse within the
layer. At the measured 13.5 GB/s that needs 20.8 ms, and only 9.4 ms of it is
hidden. Totalled, 1060 ms of stall, **30.5% of the token**.

The arithmetic: a layer can only hide reads behind work that comes after its own
router, which is the shared expert plus the projections, 14.1 ms of cover for
20.8 ms of read. A 6.7 ms structural deficit every layer. The attention block is
9.8 ms and sits before the router, so it cannot help this layer.

Disk is at 7.3 GB/s of the 13.5 GB/s ceiling, so 54% used. There is room to read
speculatively.

### What was ruled out

`cross_layer_submit` and `cross_layer_match` are never called. The predictor is
scaffolding: struct, counters and functions, not wired in. `sel_fp` and
`route_fp` are declared and never assigned, so the route dumps never fired.

Dumping the router's choices through the `sel_fp` hook, over one prompt and 32
output tokens:

| predictor | coverage of the next 16 |
|---|---|
| layer L predicts layer L+1 | **1.6%** |
| token N predicts token N+1, same layer | **42.6%** |
| union of tokens 0..3 predicts token 4 | 66.0% |

Adjacent layers share essentially nothing, so one-layer-ahead prediction is dead
however it is implemented.

### What this corrects

An expert cache was measured earlier and rejected: LRU flat at 23.3% for every
capacity, diagnosed as cyclic-sweep pathology. The diagnosis was right and the
conclusion was wrong. The cache was **global**, so one token pushes 92x16 fresh
keys through it and layer L's entries are always evicted before the next token
returns to layer L, 92 layers later. Partition per layer and the competition
disappears: layer L holds its own small set and sees only its own traffic, one
visit per token.

Simulated against the 32-token trace:

| cap/layer | RAM | steady hit | read/token | per-layer read |
|---|---|---|---|---|
| 0, today | 0 | 0% | 25.8 GB | 20.8 ms |
| 16 | 25.8 GB | 26.6% | 18.9 GB | 15.2 ms |
| 24 | 38.7 GB | 41.2% | 15.1 GB | 12.2 ms |
| 32 | 51.7 GB | 48.0% | 13.3 GB | 10.7 ms |

At the same 51.7 GB where the global LRU measured 23.3%, per-layer measures
48.0%. The curve is no longer flat; the flatness was the structure, not the data.
Warm 41.5% against steady 41.2% at cap 24, so it is a real steady state and not
a warm-up artefact.

Free RAM is about 48 GB, 124 total less the 75.9 GB payload, so cap 24 at
38.7 GB fits and cap 32 does not.

### What it was worth, built and measured

Eight different prompts, eight output tokens each, one request at a time, each
prompt reported separately because an average would hide a cache that helps only
a few. Mean decode-step latency in seconds:

| cap/layer | p1 | p2 | p3 | p4 | p5 | p6 | p7 | p8 | mean |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 3.604 | 3.608 | 3.605 | 3.627 | 3.603 | 3.592 | 3.604 | 3.612 | 3.607 |
| 16 | 3.067 | 3.076 | 3.060 | 3.072 | 3.046 | 3.001 | 3.053 | 3.006 | 3.048 |
| 24 | 3.046 | 2.969 | 2.954 | 2.981 | 3.002 | 2.916 | 2.972 | 2.938 | 2.972 |

Speedup at cap 24: 1.18, 1.22, 1.22, 1.22, 1.20, 1.23, 1.21, 1.23. Mean 1.21x
and no prompt left behind. Tokens identical at every capacity. Measured hit rate
42.1% against 41.2% simulated, expert read 2163 -> 1554 GB over the run, wall
282.9 -> 250.4 s.

The prediction made before building was 3474 -> ~2800 ms, about 1.25x, on the
grounds that 616 ms of the 1060 ms stall was structural and the rest scheduling.
Measured 3607 -> 2972 ms, 1.21x, inside that range.

cap 16 returns 1.18x for 25.8 GB against cap 24's 1.21x for 38.7 GB, so most of
the benefit arrives at two thirds of the memory.

### The first attempt made prompt one slower, and why

Before pre-faulting, cap 24 measured 0.87x on the first prompt and 1.17-1.21x on
every later one. A fresh 38.7 GB mapping charges a fault and a 2 MB zero-fill on
first touch and that landed inside the first request. `root_cache_open` now
walks the mapping one byte per huge page at startup, paid once per process.
This is why the requirement was "every prompt" and not "on average": the average
over eight prompts was already 1.14x while one of them was a 15% regression.

### The limit to carry forward

This is a low-concurrency lever and it does not compose with batching. At B
lanes a layer needs up to 16B distinct experts, and a shared cap-24 cache
behaves like 24/B per lane, so the hit rate collapses as lanes are added.
Holding 41% at eight lanes would need 192/layer, about 310 GB, which does not
fit. The cache is bypassed outright when a batch asks for more distinct experts
than a layer holds, which keeps it correct at any lane count. Batching is the
throughput lever at high concurrency; this is the latency lever at low
concurrency. They are alternatives, not a stack.

### Harnesses

`profile-stages.c` rewritten for the lane API and the batched shapes, now also
recording per-layer expert stall, drain and read volume. `stage-report.py`,
`route-probe.c` (read only, drives the existing `sel_fp` hook),
`route-predict.py`, `layer-cache-sim.py`.

## Current: Cross-Request Batching, One Weight Sweep For Many Lanes

### Intended outcome

An output token moved 53.83 GB of weights to serve one position, at 2.00
FLOP/byte; a five-position prompt pass moved the same bytes at 9.99. The weight
traffic is identical and only the useful work differs, so the question was
whether several in-flight requests could share one sweep.

### What was known going in

`Qm(Y, Xs, T, W, in, out)` already reads each weight byte once and applies it to
all `T` positions, and `NPOS_SLOTS=8` already allows `T <= 8`. What was missing
was a scheduler, and lanes: every slot array was already per-position, but the
attention cache and the KDA recurrent state were a single set shared by whatever
request was running.

### What was built

A lane is one in-flight request. `resident_sequences` gained a lane dimension,
and `pos_lane[]` / `pos_abs[]` say which lane owns each slot and at what absolute
position. `St` and `convbuf` became pointers into the owning lane's storage
rather than scratch copied in and out per layer, which also removed 12 MB of
memcpy per layer. Prefill runs a lane at a time because it already fills every
slot; decode then runs all lanes in one pass.

Requests are split out of a buffer the engine owns rather than via `fgets`,
because stdio would absorb a whole pipe write into its own buffer and leave
`poll` with nothing to report. The engine blocks for the first request and then
takes whatever has already arrived, up to `K3_LANES`.

### What the system showed

Six prompts, six output tokens, submitted together. Tokens identical to running
them one at a time. 161.5 s -> 111.6 s wall.

Per-evaluation telemetry, 36 evaluations at one lane against 11 at eight
(six prefills either way, then 30 single steps against 5 six-wide steps):

| | evaluations | wall | expert read | `op:Q` | `op:X` |
|---|---|---|---|---|---|
| one lane | 36 | 151.6 s | 1326.8 GB | 50.4 s | 52.1 s |
| eight lanes | 11 | 100.8 s | 1135.9 GB | 17.0 s | 51.7 s |

`Q` is the whole of the win and it behaved exactly as the FLOP/byte argument
said: a batched decode step spends 1.58-1.64 s on `Q` for six positions against
1.38 s for one, so six times the work for 1.16 times the time.

`X` did not amortise at all: 52.1 s against 51.7 s. Expert selection is
per-position, so six lanes mostly pull six different experts and the read volume
falls only 14%. Divergence grows as the lanes generate: 100.7 GB on the first
batched step, 127.8 GB by the fifth, because the prompts start alike and the
continuations separate.

### What this changes

Decode went from 3.39 s to 1.77 s per token at six lanes, about 1.9x. The wall
figure of 1.45x is diluted because prefill is still sequential and accounted for
50 s of the 100.8 s of engine time in that run.

### The eight-lane regression, and what caused it

Sweeping lane counts over eight prompts at sixteen output tokens showed the
curve improving to four lanes and then going backwards:

| lanes | wall | `op:Q` | expert read |
|---|---|---|---|
| 1 | 481.4 s | 177.6 s | 3792.0 GB |
| 2 | 380.8 s | 95.2 s | 3598.1 GB |
| 4 | 314.9 s | 55.2 s | 3355.0 GB |
| 8 | 341.6 s | **133.6 s** | 3102.5 GB |

`Q` fell cleanly to four lanes and then more than doubled. Per step it went from
1.45 s at four positions to 8.12 s at eight, 5.6x the time for twice the work.

`Qm` had an output-blocked variant that engaged at `T>=8`. It hoisted the x load
across eight output rows, but it carried the `t` loop outside the weight load, so
it re-read every weight row `T` times, and it never took the direct int8 decode
that the fallback uses when the palette is the identity. Weights are what binds
this kernel, so at `T=8` it asked for eight times the traffic on a path already
at 89% of RAM bandwidth. Its comment claimed the blocked path won above `T=8`;
whenever that was measured, it was before the direct decode landed.

Removed. Both paths were already known to agree, because lanes 1, 2 and 4 used
the fallback and lanes 8 used the blocked path and all four produced identical
tokens. After removal:

| lanes | wall | tok/s | before |
|---|---|---|---|
| 1 | 479.6 s | 0.2669 | 481.4 |
| 2 | 381.9 s | 0.3352 | 380.8 |
| 4 | 318.6 s | 0.4018 | 314.9 |
| 8 | **276.4 s** | **0.4631** | 341.6 |

Tokens identical at every lane count. The one-, two- and four-lane arms landing
within 1% of the previous run is what makes the eight-lane change attributable
to that path rather than to drift. End to end 479.6 -> 276.4 s, 1.74x.

This is not only a batching fix. Prompts are chunked to `NPOS_SLOTS=8`, so every
full chunk of a long prompt was taking the blocked path too.

### What is still unknown

Prefill is sequential and now the larger half of a short request. The expert
read path is untouched by batching: `X` is flat at 141-146 s across every lane
count, so expert projection does not amortise at all.

### What the next cycle should do differently

The scheduler was static: a batch formed, drained, and only then was the next
one admitted. A retired lane sat idle while the sweep it was part of cost the
same, and a request arriving mid-batch waited for the whole batch. Continuous
admission is being built and measured against it.

## Current: Three Simplifications Built And Measured

All three were implemented one at a time and measured against the engine as
committed at 31344a4, built from the same source and run interleaved
(pm, new, new, pm) to cancel drift. Eight-position prefill, `fresh-request.json`
so the result cache is off, two passes per process.

| | `op:Q` dense | `op:X` expert | wall clock |
|---|---|---|---|
| as committed | 14.39 / 17.55 s | 37.33 / 38.82 s | 53.44 / 58.56 s |
| all three | 13.74 / 16.44 s | 36.27 / 38.15 s | 51.59 / 56.64 s |
| difference | \u22124.6% / \u22126.3% | \u22122.8% / \u22121.7% | **\u22123.5% / \u22123.3%** |

Both new runs fell below both old runs in every pass, so the wall-clock figure is
separated rather than inside the noise. Output is unchanged throughout: the France
reference keeps all 8 PASS lines, `test-live-root` and `test-datasets` pass, and
the service still returns 418/276, 198/1008/12981 and a correct recursive
Fibonacci for the Python prompt.

**Expert-major grouping did far less than its own counters suggested.** A new
`root_project_rows` in `derive-root.mjs` projects every position that chose an
expert in one pass, and the expert loop groups by expert. Measured on an
eight-position batch it cut weight passes from 11,776 to 8,266, **29.8% fewer** \u2014
and bought about **2.9%** of expert time. The duplicates it removed were already
page-cache warm from a sibling position microseconds earlier, so what was saved was
a cached read plus the nibble unpack. The cost is the first disk read of each
distinct expert, and the union of distinct experts is unchanged by batching.
Two predictions were wrong and the work corrected them: the routing data suggested
35\u201356% fewer passes (measured 29.8%, this prompt shares less context), and the
estimate treated the saved passes as if they cost what a cold read costs.

**The shared expert was the clean win.** `resident_shared` called `Qm(...,1,...)`
three times per position; `resident_shared_rows` calls it once for all positions.
`op:Q` fell 14.39 to 13.74 s and 17.55 to 16.44 s, both runs separated from both
baselines. This is the same change the dense MLP already had, and it was the last
dense weight still read once per position.

**Removing the dead read path changed nothing measurable, as predicted.** The
staged-read branch, the `primed`/`staged` conditionals and `resident_prefetch_next`
are gone; the per-layer stage count drops and a duplicated 45-line loop with it.
`expert-pipeline.h` and `cross-layer-prefetch.h` are still included and still
reported in `CACHE_JSON`, because removing them would invalidate
`test-expert-pipeline`, `test-pipeline-integrity`, `test-cross-layer-prefetch`,
`test-cross-layer-model` and the `clover-one-without-cross-layer` build variant for
no measured gain. Their counters now read zero permanently, which they already did.

A claim from the previous cycle was wrong and is withdrawn. The expert result cache
does **not** "never fire". It hit immediately when a harness re-evaluated the same
prefix inside one process, and that silently contaminated the first grouping
measurement until the run was repeated with the cache off. It does not fire during
ordinary generation, because every position presents a different input vector. It
was left in place.

What is still unknown: whether any of this changes with a warm expert store. Every
measurement here sits on a machine where an eight-position prefill takes 51\u201359 s and
repeats are not faster, so the page cache is not retaining the working set between
runs. On a machine that held the experts in RAM the balance between read, unpack
and multiply would differ, and the expert-major result in particular could look
quite different.

## Current: What The Cycle Observation Says Can Be Simplified

Asked whether the cycles can be simplified given the caches already present. The
answer came from the profile and the routing observations, not from the stage names.

A correction first, made to `cycle.md`: the expert cost was reported as 36.9% of a
position. That divided by the sum of every boundary stage, which double-counts the
`detail:` stages nested inside them. Against the sum of `total:layer` (6,696.8 ms)
or the position total (7,018.5 ms), `experts-mix-normalize-up` is 4,983.5 ms \u2014
**about 71% of a position**. `op:X` alone is 4,883.1 ms. Nothing else is close:
attention 16.7%, shared expert 5.0%, head 4.5%.

Four candidates, with what was measured for each.

The dead read-ahead machinery costs almost nothing to keep. Measured at position 1:
`detail:cross-layer-predict-and-submit` 0.0 ms, `detail:read-ahead-wait` 0.0 ms,
`result-lookup-and-first-expert-submit` 1.6 ms, against 6,696.8 ms. Removing it
would take the per-layer stage count from 15 to about 12 and delete two headers,
but it is a code simplification and not a speedup. `shared-expert-during-read` is
not overhead at all: its 351.1 ms is `SH1`+`SH2`+`SH3` = 349.8 ms, work that has to
happen. Only its placement was chosen to hide a read that no longer occurs.

The expert result cache never fired. Zero hits across 61 positions and 9 runs. It
keys on an exact match of the 3,584-float input, and that input differs at every
position, so it cannot hit during ordinary generation. It holds 256 MiB.

The shared expert is the last dense weight still read once per position.
`resident_shared` calls `Qm(...,1,...)` three times. Every other dense weight,
including the layer-0 MLP, was already converted to a batched `Qm` over `NACT`
rows. At 349.8 ms per position this is a bounded, well-precedented change.

The one change that touches the dominant cost is making the expert loop
expert-major instead of position-major. Measured from the routing observations,
the distinct experts a batch of positions needs per layer:

| batch | distinct experts needed | worst case | reduction |
|---|---|---|---|
| 2 positions | 25.6 \u2013 27.6 | 32 | 14 \u2013 20% |
| 4 positions | 38.3 \u2013 48.8 | 64 | 24 \u2013 40% |
| 8 positions | 55.9 \u2013 83.0 | 128 | 35 \u2013 56% |

The loop currently runs `for position { for rank { resident_expert(...) } }`, so an
expert chosen by three positions is read and unpacked three times. The router was
already restructured this way \u2014 there is a comment at the top of the routing block
saying each gate row is read once and applied to every position \u2014 so the precedent
is in this file.

What is not established: how much of the expert projection is the read and the
mxfp4 unpack, which batching would share, versus the multiply-accumulate, which it
would not. The only evidence is that `op:X` was 4.883 s cold and 0.856 s warm for
the same arithmetic, a 5.71x spread, which says the read dominates when cold. The
unpack in `root_project` is per input row, so batching would save it in both
states, but by an unmeasured amount. This should be measured before the work, not
after.

Two constraints on that change. `live-root.h` is generated by `derive-root.mjs`, so
`root_project` must be changed in the generator. And the arithmetic contract must
hold: each row keeps its own 16 lanes and the same reduction tree, so bit-exactness
is preservable and checkable against `test-expert-independence` and the France
reference. Batching helps prefill only; decode positions depend on the previous
output and cannot be batched.

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