# Context — one layer, properly

A context file for the distributed hosting work. Newest cycle first. Written during the
work, not after it. Offered, not imposed: this is my account of what happened and what
it does and does not establish.

---

## Cycle: the pieces I had skipped (2026-10-11)

### What prompted it

I had been reporting the chain as though it were finished. The human counted it out:
input to tokens, embedding, layer 0, 92 layers, normalisation, token to output. I had
fixed layer 0, the 92 layers and the head, and had **not looked at** the input side, the
final normalisation, or the detokenise. That was a fair catch; "nothing conspicuously
out of line any more" was a claim about the parts I happened to have measured.

### The three missing pieces, measured

**Input embedding.** `server_input` reads one 14 KB row through `table_block`, which
without a direct mapping would decompress and SHA-256 verify a 229 KB block per token.
The direct mapping is present. 148.976 us scattered, 4.080 us warm, per prompt token.
Against a 1,988 ms token that is nothing.

**Final normalisation.** `tail-cost.c` reads transformer-93's own accumulated counters
rather than subtracting one harness from another:

| | ms/token | share |
|---|---|---|
| whole tail | 50.440 | |
| normalisation and fold | **0.149** | 0.3% |
| head and detokenise | 50.290 | 99.7% |
| deliver | 0.000 | 0.0% |

The fold over nine sources of 7,168 floats is about 64k multiply-accumulates. It was
never going to be large, and it is not.

**Detokenise.** Inside the head counter, and since the head alone measures 50.0 ms it is
about 0.3 ms. `id_to_word` is a direct array index.

**Text to token ids.** `word_to_id` is one open-addressed probe into a 524,288-bucket
table plus a memcmp. There is no merge loop, and the pipeline is driven by token ids
rather than raw text, so it runs at most once per prompt word. **This one is read, not
timed**, and I am marking it as such rather than quoting a number I did not measure.

### The chain, end to end

Per generated token, everything on this box with experts pinned:

| piece | cost | state |
|---|---|---|
| text to token ids | one hash probe per word | read, not timed |
| token id to embedding | 0.149 ms scattered, 0.004 warm | measured, negligible |
| layer 0 | 26.1 ms | fixed, was 427 |
| layers 1 to 92 | 1,912 ms | fixed |
| final normalisation and fold | 0.149 ms | measured, negligible |
| vocabulary head | 50.0 ms | fixed, was 231 |
| detokenise | ~0.3 ms | measured, negligible |
| deliver | 0.000 ms | measured |
| **total** | **~1,988 ms** | was ~10,905 |

The edge instrument confirms the two edges independently through the production shared
objects: server 26.17 ms/position, tail 52.36 ms/token.

So every piece is now accounted for. Three were worth fixing; the other four are small
enough that the measurement is the whole result.

### Worth keeping

I reported a chain as finished while three of its pieces had never been looked at. The
parts I had measured were genuinely done, which is exactly what made the claim easy to
make. Count the pieces from the architecture, not from the list of things already
touched.

---

## Cycle: the server, both ends of it (2026-10-11)

### Layer 0, fixed

The four changes already proven on the pods, applied to `server.c`: `-fopenmp` on its
build line, a pragma on `server_project`, a pragma on `server_qkv`, a pragma on
`server_update_attention`, and the AVX2 gather in `server_project_row`. `server.c` does
not include `live-root.h`, so it needed its own guarded `immintrin.h`.

| | ms/position |
|---|---|
| before | 427 |
| after | **26.1** |
| single thread, same binary | 319.8 |

**16.3x**, and every checksum identical to before the change: 1.404196 at 16 positions,
0.315960 at 32, 0.272570 at 8. The single-thread control separates the two effects: the
gather alone is 422 to 320, and the threads take 320 to 26.

Validated on a gate I had to get working first, since I would not change layer 0 without
one. `server/test-reference.sh` needs four arguments: the QKV path, the prepared
trunk-0, the original index and the raw trunk, all with pinned hashes. It passed before
the change and after.

**It was run twice on purpose.** The gate compiles `test-server-reference.c` without
`-march=native`, so `__AVX2__` is undefined and it exercises the scalar fallback rather
than the gather that production runs. Rebuilt with `-march=native` it passes too, so
both code paths are covered rather than only the one the gate happens to compile.

Also green afterwards: VERIFIED 91 of 91 on both prompts and two NORMALIZATION_PASS,
because the full build rebuilt every layer as well.

### Where a token stands

Per generated token, everything measured on this box with experts pinned:

| | session start | now |
|---|---|---|
| 68 KDA pods | 9,492 ms | 1,497 |
| 24 MLA pods | 755 | 415 |
| layer 0 | 427 | 26 |
| vocabulary head | 231 | 50 |
| **total** | **10,905 ms** | **1,988 ms** |

About **5.5x**, every step bit-identical and verified against a reference.

### What the server turned out to be

Every optimisation this session landed on the 92 layer pods. The server owns layer 0
and the edges, and the tail owns final normalisation and the vocabulary head, and
**none of them were ever touched**. `server.c` has zero OpenMP regions against a pod's
five, and `pipeline/build.sh` compiles the coordinator, the server stage and layer 93
**without -fopenmp** while layers 1..92 get it. The pragmas would have been dropped
silently even if they had been written.

### The head, fixed

`server_output_project` scored 163,840 rows one after another over 2.35 GB of table,
scalar and double, once per generated token. Rows are independent and the tie rule,
highest score then lowest id among equals, is associative, so the winner cannot depend
on the split. Only the direct mapping is safe to share, because `table_load` caches one
block at a time.

| | seconds | GB/s |
|---|---|---|
| warm sequential | 0.231 | 10.15 |
| warm parallel | **0.050** | 47.16 |

4.6x, same token 70480, at the memory ceiling. An `OMP_NUM_THREADS=1` control on the
same binary gives 0.219 s, so the gain is the parallelism and nothing else.

`head-cost.c` reports 18.9x for this. That compares its first sequential call, which
also faults in the 2.35 GB mapping, against a parallel arm running afterwards on warm
pages. `head-steady.c` measures both warm. The instrument was not wrong about the
mechanism, only about the size, and I would have repeated its number.

Verified: NORMALIZATION_PASS on the tail reference, VERIFIED 91 of 91 on the layer
campaign for both prompts. The server's own reference script needs a QKV path argument
and was not run, so it is not part of this evidence.

### Layer 0, observed

`server-context.c` drives `server_process` the way `pod-context` drives a layer, so the
two are comparable. Stable across sizes: 422.1, 427.2, 428.5 ms/position at 8, 16 and
32 positions, no first-touch artefact.

**A pod layer is 22.0 ms. Layer 0 is 427. Nineteen times.**

Phases, from my own clocks in a diagnostic copy:

| phase | ms/position | share | MACs | rate |
|---|---|---|---|---|
| dense | 282.9 | 62.0% | 727M | 2.57 GMAC/s |
| qkv | 102.0 | 22.3% | 264M | 2.59 |
| attention output | 68.1 | 14.9% | 176M | 2.58 |
| update_attention | 1.5 | 0.3% | | |
| decay | 1.3 | 0.3% | | |
| normalize, aggregate, checks | ~0.6 | 0.1% | | |

97.2% is three trunk projections, and **all three run at 2.58 GMAC/s**, which is the
single-threaded scalar palette rate measured independently in `kernel-cost.c`. The pods
run the identical kernel at about 48. Layer 0 has no experts at all; it is dense, so its
1.17 GMACs are entirely trunk-style projection.

The gap is not an algorithm. It is the four things already proven on the pods and never
applied here: `-fopenmp` in the build, a pragma on `server_project`, a pragma on
`server_qkv`, and the gather in `server_project_row`. On the pods those took the same
kernel from 2.58 to about 48 GMAC/s, which would put layer 0 near 25 ms/position.

### Not done

Layer 0 itself is unchanged. The measurement says what it would be worth; applying it
is the next cycle, and it needs the server's own reference gate working first, which
currently needs an argument I have not supplied.

---

## Cycle: can the trunk be four bits (2026-10-11)

### Intended outcome

The gather left the trunk at about 50 GMAC/s against the expert path's 80, and I had
said the rest was the format. Direction was to try the data change on a small scope and
measure before doing it everywhere. The first question is not how to pack it but
whether packing it is lossless at all, because a lossy version is a model change and
not an optimisation.

### What the weights actually look like

`trunk-probe.c` reads the real trunk and counts distinct palette indices, writing
nothing. The expert path carries a selector per 32 coordinates, so 32 coordinates is
the span that would have to fit in sixteen values:

| record | shape | distinct in record | 32-coord groups needing more than 16 | worst row |
|---|---|---|---|---|
| 6 | 12288x7168 | 256 | 2,752,365 of 2,752,512 | 244 |
| 7 | 7168x12288 | 256 | 2,752,365 of 2,752,512 | 254 |
| 8, 9, 10 qkv | 12288x7168 | 255 | 2,752,473 of 2,752,512 | 236 |
| 31, 32 | 3584x7168 | 256 | 802,814 of 802,816 | 245 |
| 34, 35, 36 shared | 6144x7168 | 256 | 1,375,803 of 1,376,256 | 245 |

The worst group uses all 32 of its coordinates on distinct values, and 99.99% of groups
exceed sixteen. The weights are spread across the whole 256-entry palette with no
sub-palette structure to exploit.

**So the trunk cannot be packed to four bits losslessly.** Doing it means folding more
than sixteen distinct values into sixteen slots, which changes every affected output and
would fail the reference. The experts are four-bit because they were quantised that way
upstream and the trunk was quantised to an eight-bit palette; that is a model decision
already made, not an implementation oversight I can correct. Re-quantising the trunk is
a question about model fidelity and belongs to whoever owns that, with an evaluation
behind it, not to a performance cycle.

### The lossless alternative, also measured and also rejected

This is znver4 with full AVX-512, so sixteen indices can be gathered at once instead of
eight, and a single 512-bit accumulator maps lane j to element c+j exactly as the
scalar `lanes[j]` did. Bit-identical, mismatch 0 at every size.

| rows | scalar | gather 256 | gather 512 | expert 4-bit |
|---|---|---|---|---|
| 4,096 | 25.98 | 50.60 | 55.49 | 83.34 |
| 16,384 | 26.84 | 52.40 | 56.50 | 83.71 |
| 65,536 | 28.09 | 50.03 | 52.37 | 80.73 |

About 5 to 8% in the harness. In the layer, two runs each: AVX-512 17.163 and 17.242
ms/position against AVX2's 17.228 and 17.513. **About 1%, inside the noise.** Not
ported. Zen 4's AVX-512 is double-pumped and a gather does not widen.

That is isolation mispredicting for the third time today, and over-predicting after
under-predicting in the previous cycle. The rule that survives is only that the layer
decides.

### Outcome

The trunk projection path is finished at about 50 GMAC/s on this hardware unless the
model is re-quantised. Both remaining routes were measured and both are closed: the
format change is not lossless, and the wider gather is not worth anything in the layer.

---

## Cycle: can the trunk palette path be optimised (2026-10-11)

### Intended outcome

I had written that the trunk path runs about 37 GMAC/s where the expert path runs about
70, and implied the gap was the format. Instructed not to jump to a conclusion but to
do the runs. Both numbers were derived by dividing stage times by my own MAC counts,
and a separate run of the trunk kernel alone had given 23.8. Three numbers from three
methods is not a comparison.

### Both kernels on one harness

`kernel-cost.c` runs them side by side, same rows and width so the MAC count is
identical, swept from 0.9 MB to 235 MB:

| rows | A scalar palette | A2 gather | B 4-bit AVX2 |
|---|---|---|---|
| 256 | 25.26 GMAC/s | 45.05 | 71.30 |
| 4,096 | 28.83 | 50.43 | 82.89 |
| 16,384 | 29.06 | 50.62 | 81.93 |
| 65,536 | 30.26 | 50.69 | 79.21 |

**The ratio holds when everything fits in L2**, so it is not bandwidth and the "half
the bytes" explanation I was heading towards is wrong. It is one scalar multiply-add
and one table lookup per element against four in a vector. A2 gathers eight palette
indices at a time; mismatch against A was 0 at every size, so it is bit-identical.

### In the layer, which is the measurement that counts

MLA layer 3, same binary type, same conditions, only the kernel differing:

| stage | scalar | gather | ratio |
|---|---|---|---|
| layer | 27.336 ms/position | **17.273** | 1.58x |
| 6, the whole MLA attention | 0.6670 s | 0.3070 | 2.17x |
| 25, 111 | 0.0724, 0.0730 | 0.0327, 0.0327 | 2.2x |
| 112, 113, 115 shared expert | 0.1241, 0.1244, 0.1238 | 0.0557, 0.0560, 0.0555 | 2.2x |
| experts 30-109 | 0.5164 | 0.5182 | 1.00 |
| 24 routing | 0.0309 | 0.0304 | 1.00 |

Experts and routing are untouched, which is the check that the attribution is right.

**The layer shows 2.2x where the harness showed 1.68x.** Last cycle isolation
over-predicted; this cycle it under-predicted. The lesson is not "isolation
over-predicts", it is that isolation predicts neither direction reliably and only the
layer settles it. My harness fed uniformly random palette indices; real weights are not
uniform.

KDA's qkv gained only 1.43x against the 2.2x everything else got. The difference is
layout: `transformer_project` walks contiguous rows, while `transformer_qkv` reads
`transformer->qkv` interleaved with a 20-byte per-row prefix.

### Result

KDA layer 2 about 27.9 to 22.013 ms/position; MLA layer 3 about 27.7 to 17.296.
VERIFIED 91 of 91 for france and japan, checksums unchanged, zero build failures. Only
the tail lacks the kernel, having no trunk projection.

Across the session: a KDA layer 139.585 to 22.013, **6.3x**; an MLA layer 31.457 to
17.296, **1.8x**. Over 68 KDA and 24 MLA pods, roughly 10,247 ms/position to 1,912,
about **5.4x**, every step bit-identical and verified.

### What is still unknown

- The remaining gap to the expert kernel, 50 against 79 GMAC/s, is the format: one byte
  of palette index per weight against four bits. That is a data change, re-quantising
  the trunk, not a code change, and it has not been attempted.
- `server.c` carries the same `server_project_row` pattern for layer 0 and the
  vocabulary head. Not looked at.
- Still nothing measured under concurrency.

---

## Cycle: the last serial loop (2026-10-11)

### Intended outcome

Observe `transformer_update_attention` before touching it, and before going anywhere
near concurrency.

### What the layer looked like first

Clocked copy of the current source, KDA layer 46, 64 positions, experts pinned, 28.281
ms/position:

| stage | what | ms/position | share |
|---|---|---|---|
| 30-109 | sixteen routed experts | 8.11 | 28.7% |
| 6 | qkv | 7.33 | 26.0% |
| 9 | attention output | 4.99 | 17.7% |
| 112/113/115 | shared expert | 3.67 | 13.0% |
| **8** | **state update** | **1.66** | **5.9%** |
| 24 | routing | 0.53 | 1.9% |

Routing confirmed at 4.36 to 0.53 ms, 8.2x, from the previous cycle. Stage 8 was the
only serial hot spot left: everything above it already runs parallel.

### Observing it

Every reference in the head loop is per-head and disjoint: `recurrent[head]`, an output
block at `attention + head * 128`, and a `prediction` array local to the iteration.
6.29 MB of state over 96 heads is 65.5 KB each, so six heads per thread sit in L2.
Per-head arithmetic and order are untouched, so it must stay bit-identical.

One thing worth noticing and not acting on: the two passes over the state could be
fused, since pass one's write of the decayed value could be recomputed in pass two from
the same operands, halving the 25.2 MB of traffic. But parallel the stage is already
about 0.12 ms, so halving its traffic would save around 0.2% of the layer. Not worth the
risk of touching the arithmetic.

### Result, and a correction to my own instrument

In the layer: stage 8 goes 0.1064 s to 0.0124 s over 64 positions, **8.6x**, 1.47
ms/position, and the stage falls from 5.9% to 0.7%. Checksum unchanged.

The standalone replica had measured 15.2x and implied about 5.5% of the layer. The real
figures are 8.6x and nearer 4%. The replica ran the update repeatedly with the state hot
and nothing else competing for cache; the layer streams 281 MB of experts between
positions. **Isolation over-predicts, and the layer number is the one that counts.**

Three runs each after the change: KDA 28.167, 27.392, 28.068; MLA 28.032, 27.629,
27.425. MLA is unchanged, which it must be since the state update is KDA-only, and that
is a useful check that nothing leaked. KDA and MLA are now indistinguishable at 64
positions.

VERIFIED 91 of 91 for france and japan, zero build failures.

### Worth keeping

The port skipped transformer-1 twice. First because that file carries CRLF and my
anchor used LF. Then, after I "fixed" it by converting the anchor when the file contains
CRLF, it skipped again: **transformer-1 has mixed endings**, CRLF in the file but LF
around this particular function. The fix that works is to try both forms rather than
decide from the file as a whole. I had diagnosed the first failure correctly and then
generalised it wrongly.

### Where the session ended

At 64 positions, experts pinned: a KDA layer 139.585 to about 27.9 ms/position, roughly
**5x**; an MLA layer 31.457 to about 27.7. Over 68 KDA and 24 MLA pods that is about
10,247 ms/position to 2,562, close to **4x** on the layer stack, every step
bit-identical and verified against the reference.

### What is still unknown

- The trunk palette-gather path runs about 37 GMAC/s where the expert 4-bit AVX2 path
  runs about 70. Same machine, half the rate. That is now the largest unexplained thing
  left, and qkv plus the shared expert plus stage 9 are all on the slow path.
- Nothing has been measured under concurrency. Every number in this file is one layer
  with the whole machine to itself and its experts pinned.

---

## Cycle: survey the remaining serial loops, then the router (2026-10-11)

### Intended outcome

I had left "whether any other loop was missed the same way" as an open question after
the qkv finding. A sample is not a survey, so: list every OpenMP region in a pod against
every heavy loop, and see what is uncovered.

### What the survey showed

The whole pod had **three** parallel regions: `transformer_project`, and the two added
earlier today. Mapping those against every loop over `TRANSFORMER_ROWS`,
`TRANSFORMER_WIDTH`, `TRANSFORMER_HEADS` or `record->rows` left one obvious candidate by
measured cost: `transformer_route`, stage 24, 4.36 ms/position, and it runs on **every**
layer rather than only the KDA ones.

It scores 896 experts, each a 7168-long dependent double accumulation over full float32
router weights. That is roughly 28.7k cycles of latency per expert and 896 of them,
about 6 ms at 4.2 GHz, which is where the measured 4.36 comes from. Latency-bound, not
bandwidth-bound, so splitting across experts should scale nearly linearly.

### Why a pragma alone would have been wrong

Scoring an expert is independent, but choosing the top sixteen is not: the loop carries
`best[]`, `selected[]` and `count` across iterations as an insertion sort. A parallel
for would have raced on all three and quietly corrupted the choice, and the reference
campaign would have caught it only as a route mismatch with no obvious cause.

So the port splits the function: scoring parallel, selection serial and still walked in
index order so a tie resolves to the lower expert id exactly as before. Per-expert
arithmetic and its order are untouched. The one behavioural difference is on the failure
path, where a non-finite choice now aborts after all scores are computed rather than at
the offending expert; the return value is the same.

### Result

| layer | before | after |
|---|---|---|
| KDA layer 2 | 32.612 ms/position | **29.020** |
| MLA layer 3 | 31.048 | **28.020** |

About 3.0 to 3.6 ms off every layer, roughly 11%. Checksums unchanged, VERIFIED 91 of
91 for france and japan, zero build failures.

### Across the whole session

A KDA layer went 139.585 to 29.020 ms/position, **4.81x**; an MLA layer 31.457 to
28.020. Over 68 KDA and 24 MLA pods that is about 10,247 ms/position to 2,646, roughly
**3.9x** on the layer stack, all of it bit-identical and verified against the reference.

### Worth keeping

`transformer-1` was skipped on the first run, reported honestly by the script as "1
without it", and that contradicted my own survey which had found the block in all 92.
The cause was line endings: that file carries CRLF in this working tree, my PowerShell
survey normalised before matching and node did not. The port now matches whatever the
file actually contains. Two instruments disagreeing is a fact to chase, not a rounding
error to pick a side on.

### What is still unknown

- `transformer_update_attention` is still serial, 1.8 ms/position, now about 6% of a KDA
  layer. Measured in isolation at 15.2x available; not applied.
- The trunk palette-gather path runs about 37 GMAC/s where the expert 4-bit AVX2 path
  runs about 70. Same machine, half the rate, unexplained.
- Nothing has been measured under concurrency. Every number here is one layer with the
  whole machine and its experts pinned.

---

## Cycle: what KDA is for (2026-10-11)

### Intended outcome

Observe KDA rather than time it. I had been treating it as "the slow kind of layer",
which is backwards.

### What the System showed

A KDA layer's entire per-sequence state, read from the struct:

```c
float history[3][TRANSFORMER_ROWS][3];                                  /*   442 KB */
float recurrent[TRANSFORMER_HEADS][TRANSFORMER_HEAD][TRANSFORMER_HEAD]; /*  6.29 MB */
```

**6.73 MB, and it does not depend on the position count.** An MLA layer instead keeps
98,560 bytes *per position*. The two cross at about 68 positions, and after that MLA
only grows.

The time behaves the same way. Both layers, experts pinned, same instrument:

| positions | KDA layer 2 | MLA layer 3 |
|---|---|---|
| 64 | 32.612 ms/position | 31.048 |
| 256 | 32.246 | 31.275 |
| 1,024 | 32.302 | 32.250 |
| 4,096 | **32.313** | **35.262** |

KDA varies 1.1% across a 64-fold change in context and shows no trend. MLA rises and
the rise accelerates, because each position attends over every earlier one. They cross
near 1,024.

### Why the model is built this way

KDA is O(1) per token in both time and memory; MLA is O(N) in both. Every fourth layer
is MLA and the other three are KDA, so three quarters of the depth costs the same at a
million tokens as at sixty-four. At 1M positions that is the difference between

- 24 MLA layers x 103.3 GB = **2.48 TB** of cache, and
- 68 KDA layers x 6.73 MB = **458 MB**.

The KDA layers contribute 0.02% of the long-context memory. The whole long-context
problem lives in the 24 MLA layers.

### What this says about today's work

Both changes were right, for different regimes, and I did not understand that when I
made them.

- Head-major K/V, 23.8x on MLA attention at 1M, touches only the 24 layers whose cost
  grows with context. At long context those layers are nearly all of the time, so it is
  the right target there and nowhere else.
- The qkv pragma, 4.19x on KDA, touches 68 layers whose cost is flat. It is the right
  target at every context length, and it was worth more overall.

I spent most of the day on the MLA layers believing they were the expensive ones. They
are only expensive once the context is long, which is exactly the case the KDA layers
exist to avoid.

### Where a layer's time goes now

Per-stage, layer 46, 64 positions, after the fix: qkv 7.2 ms, the sixteen routed experts
7.5, attention output 4.9, routing 4.4, shared expert 4.4, state update 1.8. No single
dominator left; the layer is roughly balanced.

One thing stands out for later. The routed experts run their 528M MACs at about 70
GMAC/s on the 4-bit AVX2 path, while the trunk projections run at about 37 GMAC/s on the
palette-gather path. Same machine, half the rate. Not investigated.

### What is still unknown

- Whether the KDA/MLA crossover near 1,024 holds for the whole model rather than these
  two layers.
- Why the palette-gather trunk path is half the rate of the expert path.
- `transformer_update_attention` is still serial, worth about 1.7 ms/position in
  isolation, now 5.6% of a KDA layer rather than a rounding error.

---

## Cycle: why a KDA layer cost four times an MLA layer (2026-10-11)

### Intended outcome

Understand the gap rather than optimise around it. Instructed not to trust the existing
instrumentation but to see it myself, which was the right call: the shipped tracer would
have given the answer, but I would have been trusting its placement and its clock.

### Three of my explanations were wrong

I had written that layer 46 cost 132 ms/position against layer 3's 31 because it is
"KDA carrying four input snapshots". That compares two layers differing in two things at
once and attributes the result to one of them. A 2x2 across pods separates them:

| | 1 snapshot | 4 snapshots |
|---|---|---|
| MLA | layer 3: 31.457 | layer 43: 31.472 |
| KDA | layer 2: 139.585 | layer 46: 139.518 |

**Snapshot count costs nothing measurable.** The whole 4.4x is KDA against MLA.

Then, in order, measured with my own code:

- **The recurrent state walk is not it.** `kda-state.c` replicates
  `transformer_update_attention` exactly: 1.852 ms/position serial, 0.122 parallel,
  checksums equal. Under 2% of the 108 ms gap. The missing pragma on its 96 heads is
  real but small. (First attempt produced `-nan`, because the layer L2-normalises key
  and query per head and my replica did not; a checksum of NaN compares equal to
  nothing, so the instrument was fixed before its timing was believed.)
- **The projection shapes are not it.** `proj-cost.c` measures the real record shapes:
  KDA's three 12288x7168 at 3.701 ms each parallel, MLA's four at 2.35 ms total. An 8.8
  ms difference.

That left about 90% of the gap unexplained by anything I had reasoned about.

### What it actually is

A diagnostic copy of the layer with my own per-stage clocks, 64 positions, experts
pinned:

```
STAGE_TOTAL 8.4382 s
STAGE   6     6.869 s   81.4%
```

Stage 6 is 107.3 ms/position, which is the entire gap. Stage 6 for KDA is
`transformer_qkv`, and it is the one trunk projection that does **not** go through
`transformer_project`:

```c
static void transformer_qkv(...)
{
    for (unsigned row = 0; row < TRANSFORMER_ROWS; row++)
        for (unsigned component = 0; component < 3; component++) {
```

No pragma. KDA's Q, K and V weights are interleaved per row with a 20-byte prefix
holding the row scale and its convolution taps, so the loop could not call
`transformer_project` and was written out longhand. `port-parallel-project.mjs` added
the pragma to `transformer_project` and this loop was missed. 3 x 12288 row
dot-products of width 7168, on one core, per position.

MLA escapes it because `transformer_mla` uses `transformer_project`, which is parallel.

### The fix and what it gave

One pragma, `port-parallel-qkv.mjs`, 92 pods. Rows are independent: each writes its own
`qkv[component][row]` and advances its own `history[component][row]`. Per-row arithmetic
and order untouched.

| layer | kind | before | after |
|---|---|---|---|
| 3 | MLA | 31.457 | 31.395 |
| 43 | MLA | 31.472 | 31.667 |
| 2 | KDA | 139.585 | **33.253** |
| 46 | KDA | 139.518 | **33.348** |

**4.19x on a KDA layer**, every checksum unchanged, and KDA now sits level with MLA.
68 of the 92 pods are KDA, so across the layer stack that is roughly 10,242 ms/position
to 3,020, about **3.4x**. Verified: VERIFIED 91 of 91 for france and 91 of 91 for japan,
zero build failures.

### Worth keeping

Anchoring on the loop alone would have been wrong. `transformer-1` contains the same two
loop lines a second time inside its qkv validation, which returns early and cannot be a
parallel region. The port anchors on the function signature, which is unique in exactly
92 pods and absent only from the tail. Counting before acting caught it.

Everything I did earlier today was on the 24 MLA pods, which were never the expensive
ones. The instruction to understand before optimising found more than the optimisation
did.

### What is still unknown

- KDA remains 5.7% above MLA (33.3 against 31.5). The state update accounts for about
  1.85 ms of that; the rest is unexamined.
- `transformer_update_attention` is still serial, worth about 1.73 ms/position by its own
  measurement. Not applied, because it was measured in isolation rather than in the layer.
- Whether any other loop in the tree was missed the same way. Only `transformer_project`
  and now `transformer_qkv` have been checked.

---

## Cycle: remove hardmax (2026-10-11)

### Direction

From the human, after the investigation below: remove the hardmax machinery and make
layer 46 behave like every other layer. Not needed for now.

### What was done

`port-drop-hardmax.mjs` removes the comment, the `transformer_hardmax` function and its
call site in `transformer_aggregate`. `transformer_listed` stays, because
`transformer_fold_residual` and `transformer_dropped` still use it.

Scope was counted before acting rather than sampled: 93 pod sources, **92 contain
`transformer_hardmax`, exactly 2 occurrences each**; nothing in `server.c` or
`transformer-93`. Layer 1 needed its own pair of anchors — it is derived separately,
folds two sources instead of nine, and carries different wording and a different call
site. The script asserts each anchor is unique, that no reference survives, and that the
helper is still present.

92 files, 3,488 deletions. All 92 pods rebuilt with no failures.

### What the System showed

**The full campaign now passes on defaults: VERIFIED 91 of 91 for france and 91 of 91
for japan, with no `CLOVER_SOFTMAX` set.** Before this change it aborted at layer 46.

That gate had been unusable, which mattered more than the arithmetic: no regression
anywhere in 91 layers could have been detected while it was off. It is now a gate again,
and the head-major attention work is verified through it.

Stale scratch copies on the host (`transformer-3.orig.c`, `transformer-3.hm.c`) were
removed; git holds both.

### What this gave up

At most about 1.5% on layer 46, and possibly nothing — hardmax measured 132.21
ms/position mean against softmax's 134.22, but softmax's best run was 132.145, below
hardmax's mean. One layer of 93. The configuration and its measurements are preserved in
the cycle below and in the commit, so it can be reconstructed if it is ever wanted.

---

## Cycle: why layer 46 fails the reference (2026-10-11)

### Intended outcome

Understand the layer 46 route mismatch before working around it. It was suggested the
failure might be informative rather than merely broken. It was.

### What the System showed

The harness checks each layer independently against the reference's own input for that
layer, so nothing accumulates: layer 46's pod simply disagrees with the reference about
which sixteen experts to route to, from identical input.

The cause is in the pod, in plain sight:

```c
/* Defaults stay at layer 46 with both folds, the configuration measured good. */
if (!layers || !*layers) layers = "46";
stages = stages && *stages ? stages : "3,21";
on = !(off && *off && *off != '0');
```

`transformer_hardmax` defaults its owner list to the literal string `"46"`, both folds,
on unless `CLOVER_SOFTMAX` is set. Layer 46 alone replaces the fold's softmax blend with
winner-takes-all at stages 3 and 21. The aggregate changes, so `postnorm` changes, so
the router selects different experts. The harness runs `env -i`, so the experiment is
always on. That is why exactly one layer fails, and why it is that one.

Tested rather than assumed:

| run | result |
|---|---|
| layer 46, default | VERIFIED 0, aborts on the route assertion |
| layer 46, `CLOVER_SOFTMAX=1` | **VERIFIED 1** |
| all layers, `CLOVER_SOFTMAX=1` | **VERIFIED 91, last 92** |

So there is no regression anywhere. The whole campaign passes once the experiment is
off, which also means the head-major port is verified across all 91 layers, not the 24
MLA layers alone.

**What the default buys.** Layer 46 through `pod-context.c`, 256 positions, experts
pinned, four runs each:

| arm | runs, ms/position | mean |
|---|---|---|
| hardmax | 131.788, 132.162, 132.429, 132.478 | 132.21 |
| softmax | 133.140, 135.513, 132.145, 136.094 | 134.22 |

Checksums are exactly reproducible within each arm and differ between them
(-2523.027578 against -4549.359472), so the output difference is real and
deterministic. The time difference is not solidly real: softmax's best run, 132.145, is
below hardmax's mean. Hardmax is tight and softmax is noisy. At most about 1.5%, on one
layer of 93, so on the order of 0.02% of model time.

### What this is really about

The cost is not the 1.5%. It is that the full campaign has been unrunnable, so no other
regression across 91 layers could have been detected either. The gate was off and
nothing said so. That it now passes cleanly is good news bought cheaply.

The prior work behind the default is sound on its own terms: `scripts/fold-cosine.sh`
derived owner lists from cos(aggregate, residual), and `run-hardmax-cos.sh` compared
generated text. "Measured good" meant the text did not change on the prompts tried. It
did not mean the numerical reference still matched, and those are different claims.

### Recommendation, not a change made

Make hardmax opt-in: default the owner list to empty so `CLOVER_HARDMAX_LAYERS=46`
enables it explicitly. A default should be the configuration that can be verified; an
experiment should have to be asked for. I have not changed it, because it alters shipped
model output and that is a Direction decision.

### What is still unknown

- Whether hardmax at layer 46 changes generated text on prompts beyond those already
  tried. The checksum says the residual differs materially; only text comparison over a
  real prompt set answers quality, and that needs the fleet running.
- Why layer 46 costs 132 ms/position where layer 3 costs 31. Four times, and layer 46 is
  KDA carrying four input snapshots against layer 3's one. Not investigated, and it
  bears on where optimisation effort belongs, since 68 of 92 pods are KDA.

---

## Cycle: join the kernel number to a real layer, then port it (2026-10-11)

### Intended outcome

The previous cycle left its central claim unjoined: 23.8x was an isolated kernel, and
the layer had only ever run at five positions. Close that, then decide from the result
whether the other MLA pods should have the change.

### What the System showed

`pod-context.c` drives a real pod through `transformer_process` with its 15.72 GB of
experts pinned by `mlock`, so expert arrival is identical in every arm and only the
attention walk moves. Output bit-identical in every pair:

| positions | original | head-major | layer | attention component |
|---|---|---|---|---|
| 64 | 31.265 ms/pos | 30.864 | 1.01x | negligible |
| 1,024 | 40.035 | 32.106 | 1.25x | 9.1 -> 1.2 ms |
| 4,096 | **71.546** | **35.245** | **2.03x** | 40.6 -> 4.3 ms, 9.4x |

Experts cost about 30.9 ms per position whatever the length, so below roughly 1k
positions the change is noise — the cache still fits in L3. At 4,096 the old layout
spends 57% of the layer in attention and the new one 12%. That justified the port.

**Ported to all pods and verified.** `port-head-major-mla.mjs` follows the house pattern:
unique-anchor assert, idempotent, layers 2..92. The block exists in every pod but is
only compiled where `TRANSFORMER_MLA` is 1, so 24 layers change behaviour and 66 keep a
dead copy consistent rather than leaving a latent defect. All 24 MLA layers verify
against the reference for france and japan, 24 of 24, matching the pre-port baseline.

### The mistake, which was the same one twice

I shipped 90 stale sources to the host and overwrote the deployed tree.

Last cycle I found the repository was behind what runs, pulled `transformer-3.c` from
the host, and wrote in this file: *"Read the deployed artefact first, not the repository
copy."* Then I ran the port against the other 91 pods **without pulling them**, because I
had quietly assumed that fixing one file had fixed the problem. It had not: those pods
still included `root.h` instead of `common/live-root.h`, a gap of **30,159 insertions**,
not the 332 I had measured on a single file.

The build caught it — 23 pods failed with `omp_get_max_threads` implicitly declared,
which is exactly the set of MLA pods, because only they compile the block. The host was
restored from a backup taken before the extraction, rebuilt, and re-verified. Nothing
was lost, because the script took that backup before writing. That was luck of habit
rather than design, and the habit is worth keeping.

The correction: a sample is not a survey. When one artefact is found to be stale, the
question is how many others are, and that is a cheap thing to check.

### What was ruled out

**The layer 46 route mismatch is not mine.** The full campaign (`all`, layers 2..92)
aborts at layer 46 on `!memcmp(routes, captured_routes[position])`. Controlled test:
identical runs differing only in transformer-3's stage object, original and head-major
both reach VERIFIED 44, last layer 45, same assertion. So it predates this work. The
historical `remaining/validation.log` holds 182 VERIFIED lines, a full pass over 91
layers times two prompts, so layer 46's routing changed at some point between that
campaign and now — most likely in one of the ports, none of which re-ran the campaign.
Selecting only MLA layers steps past it, because the harness checks only selected layers.

### What is still unknown

- Why layer 46's routes no longer match the stored reference. Not investigated. Until it
  is, the full campaign cannot pass and only subsets can be verified.
- Everything in the previous cycle's unknowns still stands: no end-to-end long context,
  the latent K/V form unread, expert compute unprofiled at stage granularity, the DIMMs
  at 3600 against a 4800 rating.
- The 4,096-position result is one layer on one machine with experts pinned. A pod under
  real traffic shares the box.

---

## Cycle: make layer 3 use the machine (2026-10-10)

### Intended outcome

One layer — not the fleet, not the model — correct and actually using the CPU cores, the
RAM and the disk it has. Then a stage map from input through to layer 92.

### What was known going in

The standalone engine's MLA K/V cache had been changed from position-major to head-major
earlier the same day, measured at 1.68x at 1M positions on an isolated kernel. The open
question was whether the distributed layers carried the same defect.

### What was tried, and what the System showed

**The repo was not what runs.** The local `transformer-3.c` included `root.h`; the
deployed one on k3 included `live-root.h`, `stage.h` and `trace.h`. A diff put the repo
**332 insertions and 41 deletions behind** the deployed source. `common/` is untracked in
git and was never committed; the port scripts write in place but their output never
reached a commit. I had already read the stale file and begun reasoning about it. Pulled
the deployed copy down and worked from that.

**The defect was there, and worse.** `transformer_mla` walked every head across every
position at a 98,560-byte stride — K, V and the positional tail interleaved per position.
Double the standalone's stride, because here all three share one block.

**Two things were wrong, not one.** The head loop also had no OpenMP pragma: 96
independent heads on a single core while fifteen sat idle.

Measured with `tansformers/mla-stride.c`, which isolates the attention step at real
sizes. All arms produce identical checksums at every size.

At 1,048,576 positions, 103.35 GB for one layer:

| arm | time | GB/s |
|---|---|---|
| position-major serial — what the layer did | 62.40 s | 1.66 |
| position-major parallel | 5.16 s | 20.03 |
| head-major serial | 12.51 s | 8.26 |
| head-major parallel | **2.62 s** | 39.45 |

23.8x together. Layout alone 4.99x, threads alone 12.1x. The result sits at 39.45 GB/s
against a measured 42.8 GB/s RAM ceiling, so it is bus-bound.

**Thread count, measured on the real layer through the reference harness:** 1 thread
2.49 s, 4 threads 0.73 s, 8 threads 0.44 s, 16 threads 0.30 s, 32 threads 0.31 s. Every
run verified. Sixteen is the optimum for a layer alone; SMT does not help.

**Correctness.** Three gates, all passed:
- `VERIFIED` for france and japan at all five positions against the independent
  reference, for both the layout change and the parallel head loop.
- Bit-identical output over a 40-position stream against the unmodified binary. This
  matters because the cache starts at capacity 8 and the reference prompts are 5
  positions, so **the reference never grows the cache and never executes the re-layout
  code**. Forty positions crosses 8 to 16 to 32 to 64.
- `CONTROLS 3 PASS` under AddressSanitizer and UndefinedBehaviorSanitizer.

Installed as `transformer-3.c` on k3 and rebuilt as `pipeline-stage.so`; re-verified at
16 threads. The original is kept at `transformer-3.orig.c`.

### What was ruled out

**The disk is not the bottleneck, and the current I/O path is the best of three.** I
believed expert reads were running at roughly 380 MB/s against a device measured at
13.51 GB/s. That number was wrong and so was the reasoning: I divided whole-layer wall
time by expert bytes and called the quotient an I/O rate. Measured properly, with the
page cache dropped before each arm and disjoint expert ranges:

| request shape | pass 1 | pass 2 |
|---|---|---|
| mmap + `posix_madvise(WILLNEED)` — current | 8.58 GB/s | 8.06 GB/s |
| one `pread` per expert across threads | 7.02 | 7.18 |
| chunked `pread`, more requests in flight | 7.02 | 6.86 |

281 MB of experts arrives in 33 ms. The existing path beats both alternatives. There is
no gain here, and the plan to port the standalone engine's dedicated pread readers into
the distributed layer is withdrawn on this hardware.

**`OMP_NUM_THREADS=4` in HOSTING.md is not wrong.** I measured 4 threads as 2.4x slower
than 16 and nearly reported the documented guidance as a defect. It is not: that table
is a resident fleet serving 1 to 32 concurrent requests, where four threads per request
times sixteen concurrent requests already saturates the box. One layer alone is a
different regime. The thread count belongs to the deployment shape, not to the code.

### Mistakes worth keeping

- **I read the repo and assumed it was the system.** The deployed source had moved on by
  332 lines. Pull the running artefact before reasoning about behaviour.
- **I treated `--inspect` agreement as a correctness result.** It prints one summary line
  and never touches the attention path. Caught it before reporting it as a pass, but I
  had already run it as though it were a gate.
- **I assumed the existing test covered the code I wrote.** It could not: 5 positions
  never grows an 8-slot cache. A test that cannot reach the change is not evidence about
  the change.
- **PowerShell `\$?` through ssh returns the local shell's status, not the remote
  command's.** Several "exit=True" lines earlier in the session were meaningless. The
  `VERIFIED` lines came from the harness itself and were real.

### What is still unknown

- No layer has run end to end at long context. The 1M figures are an isolated kernel.
  Expert cost per position makes a full long-context layer run impractical on one box,
  so the two measurements have not been joined.
- Only layer 3 has these changes. The other 23 MLA pods are unmodified. The 24 pods are
  generated, so landing this across them is a generator change, not 23 edits.
- The latent K/V form is 2,304 bytes per position per layer against 98,560 expanded,
  42.8x fewer bytes. Since attention is now bus-bound, this is the only remaining large
  factor — and it needs the absorb matrices, which have not been read.
- Expert compute, not expert I/O, dominates a layer at short context. 80 of 120 stages.
  Not yet profiled at stage granularity on the distributed path.
- The DIMMs run at 3600 MT/s against a 4800 rating. Never investigated.
- The `detail:expert-read-gb` counter in the standalone profiler prints gigabytes through
  a milliseconds field. Still unfixed.

### What the next cycle should do differently

Read the deployed artefact first, not the repository copy, and check whether a test can
actually reach the code being changed before treating a pass as evidence. Both failures
this cycle were of the same kind: accepting a proxy for the thing itself.

### Repository state

`STAGES.md` is new. `tansformers/mla-stride.c` and `tansformers/expert-io.c` are new
instruments. `tansformers/transformer-3/transformer-3.c` now holds the deployed source
plus both changes — note that committing it mixes a 332-line catch-up with a small
functional change, and the catch-up should land separately. `common/` is still untracked
and still needs committing; it is the reason the repo cannot currently rebuild what runs.
