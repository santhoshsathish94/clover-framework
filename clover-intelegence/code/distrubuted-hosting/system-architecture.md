# System Architecture

How Clover-K3 is split across machines, why the split makes the hardware usable, and
what has actually been measured. Every number here came from a run on AX102
(124 GB RAM, 16 cores) on 2026-10-03. Where something has not been measured, it says so.

## The problem the split solves

The model's expert weights are **1,430.9 GB**. A single 124 GB machine can hold about
8.7% of that. Everything else is read from disk, again, for every token.

That is not a theory about the machine; it is what the machine does. The same 93 layers,
timed with the weights already in memory and then timed with them coming off disk:

| per-layer stage time | min | median | p90 | max |
|---|---|---|---|---|
| RAM-resident | 19.1 ms | 26.9 ms | 27.3 ms | **28.6 ms** |
| read from disk | 49.5 ms | 74.8 ms | 85.0 ms | 909.2 ms |

Two things matter in that table. The disk row is three times slower at the median and
thirty times slower at the tail. The RAM row is **tight** — 19 to 29 ms across all 93
layers, a spread of about 1.5×. A pipeline runs at the speed of its slowest stage, so a
narrow spread is worth as much as a low average.

Running the whole model on one box, the result is a machine that cannot use itself:
CPU sits around 75% idle while the disk moves 1.9 GB/s, because every token is waiting
for experts the machine has no room to keep.

## The shape

```
caller  ──►  server  ──►  layer 1  ──►  layer 2  ──►  …  ──►  layer 92  ──►  server  ──►  caller
 (token      (tokenise,                                                   (final normalisation,
  ids)        embed, trunk 0)                                              vocabulary head, decode)
```

The server owns every edge of the problem. Text to token ids, token id to embedding row,
the final vocabulary projection and the choice of the next token all live in
`server/ends.h`; layer 0 is different from the rest — no MLA, no routed experts, no
router — so it lives with the server rather than being a pod that looks like none of the
others; and the final aggregation and normalisation after layer 92 lives there too.

There was once a separate client stage holding the tokenizer and the two edge
converters. It was retired: one service now owns entry and exit, and a caller only ever
talks to one address.

Each of the 92 remaining layers is a pod holding its own trunk and its own 896 experts.
A pod computes one layer and nothing else.

## What each machine holds

Measured by following the symlinks in each stage's own dataset directory:

| stage | contents | size |
|---|---|---|
| server | `tiktoken.model`, `vocabulary.bin`, `inputs/seed.bin`, `outputs/fruit.bin`, `trunk-0`, `trunk-0-qkv`, `leaves.json` | **4.432 GB** |
| layer pod | `root-N` (experts 14.643 GB), `trunk-N`, `operators/qkv-all/layer-N` | **15.512 GB** |

The client stage was retired; the server owns both ends of the chain, which is why its
share grew from 1.338 GB to 4.432 GB.

A pod also needs per-request state: 7.41 MB per in-flight sequence, of which about
6.7 MB is genuine history (`recurrent` 6.00 MB, `history` 432 KB, `snapshots` 224 KB)
and the rest scratch. Four concurrent requests therefore cost a pod roughly 30 MB. On a
single box running all 93 stages the same request costs 689 MB, because it needs one
sequence in every stage at once.

**A 20 GB node per layer is enough**, with headroom. The point is not the headroom, it
is that 15.5 GB fits inside 20 GB, so after warm-up the pod never reads disk again.

Each stage can reach only its own data. Before this, every pod's `bin/dataset` was a
single symlink to the whole 1,430.9 GB tree.

## Why this is an efficient way to host

The fleet needs 92 × 20 GB = **1,840 GB** of RAM against 124 GB on one box. That is
about 15× the memory and 93× the cores, so the honest accounting is:

| | one box, measured | 93 pods, projected |
|---|---|---|
| per-token latency | ~46 s per position at peak load | ~2.5 s (93 x 27 ms) |
| aggregate throughput | **0.343 positions/s** | ~35 tokens/s pipelined |

The left column is measured on this box at its throughput peak of sixteen concurrent
requests: 64 positions in 186.4 s. That is roughly **twice** the 0.155-0.172 positions/s
the standalone engine reaches with a single request, so the fleet now exceeds the
standalone rather than trailing it. The right column remains a projection and has never
been run.

Most of the 200× is simply more hardware. **The per-machine gain is roughly 2–3×.**

The real argument is not the ratio. It is that on one box, adding cores changes nothing,
because the bottleneck is a disk the cores are waiting on. Split this way, every
machine's working set fits in its own memory and the cores finally have something to do.
The split does not make the model faster; it makes the hardware usable.

Throughput also scales with the number of pods in a way a single box cannot. With 93
stages, 93 requests can be in flight at once, each occupying a different layer. The
pipeline produces one token per slowest-stage time, 28.6 ms, regardless of how long any
single request takes end to end.

### The network is not a problem

A hop carries one 7,168-float residual (28,672 bytes) plus up to eight snapshots
(229,376 bytes): **252 KB worst case**. At 35 tokens/s that is 8.8 MB/s on any one link
and about 810 MB/s across the whole fabric. Added latency is roughly 23 ms against
~2.5 s of compute. Ordinary 10 GbE is ample.

## Two memory decisions that made the difference

**Map the weights, do not copy them.** Every stage originally read its trunk and its
265 MB QKV operator into private heap with `malloc` and `fread`. With 93 stages resident
that was 54 GB of memory the kernel could never reclaim — and 54 GB the page cache no
longer had for experts, which are the thing that must stay cached. Mapping the same
files read-only changed that:

| | copied | mapped |
|---|---|---|
| unreclaimable memory | 54 GB | **3 GB** |
| reclaimable page cache | — | 88 GB |
| fleet open | 39.8 s | 14.6 s, then **1.8 s** warm |

**Lock the trunk, let the experts stream.** The trunk is read at every layer of every
token, so it should never be evicted. The experts are 896 per layer with 16 chosen per
token, so caching them is mostly luck. Locking the trunk and QKV resident pins 72 GB and
leaves the page cache to hold whatever experts happen to be hot. Set
`CLOVER_LOCK_TRUNK=0` to disable it.

On a single box this is a trade, not a free win: locking 72 GB alongside 123 GB of page
cache over-subscribes 124 GB, and 3.2 GB of swap appeared under four concurrent
requests. On a 20 GB pod the question does not arise, because the pod's whole 15.5 GB
working set is resident anyway.

## Concurrency

Stages are not threads. An idle stage costs nothing at all: it is mapped memory and a
function pointer, and it runs only when a vector reaches it. A request contributes one
thread of its own plus its OpenMP team, so four concurrent requests at four threads each
run the whole 93-stage fleet on about seventeen threads. Under Kubernetes each pod would
add one thread blocked on a socket read, which consumes no CPU.

What cannot be shared is per-request state: the KDA recurrence and the MLA cache. So a
request builds one sequence in every stage when it arrives and frees them when it
finishes, and it opens its own embedding and head readers, because those hold a `FILE *`
cursor and would corrupt each other.

There is no configured concurrency. Admission is decided against a memory budget: the
coordinator prices one request by opening a probe session at startup and measuring what
it maps, reads `MemAvailable`, and admits while the next request still fits. It reports
what it derived:

```
Fleet resident: server + 92 layers in 16.887s; request costs 875 MB, budget 26.1 GB, so 30 run at once
```

### What concurrency actually buys, and where it stops

One resident fleet stepped from 1 to 32 concurrent requests without restarting, same
prompt each time so only concurrency varies:

| concurrent | wall | per request | throughput | step gain | CPU of 3200% | disk read |
|---|---|---|---|---|---|---|
| 1 | 123.9 s | 123.9 s | 0.0081 req/s | — | — | — |
| 2 | 130.8 s | 129.5 s | 0.0153 | 1.89x | — | 68.3 GB |
| 4 | 138.2 s | 137.9 s | 0.0289 | 1.89x | 716% | 60.8 GB |
| 8 | 152.3 s | 152.1 s | 0.0525 | 1.82x | 1,541% | 62.9 GB |
| **16** | 186.4 s | 185.6 s | **0.0858** | 1.63x | 2,221% | 68.0 GB |
| 32 | 416.9 s | 291 s | 0.0767 | **0.89x** | 2,262% | **134.5 GB** |

Output is byte-identical at every concurrency. Throughput peaks near sixteen and then
falls. The disk column explains it: flat to sixteen, doubling at thirty-two, while CPU
barely moves. Thirty requests holding ~875 MB each evict the page cache that was holding
experts, so the fleet re-reads them.

**The derived budget is a safety bound, not a performance one.** Half of `MemAvailable`
admits about thirty here, which is past the peak. It stops the box dying; it does not
make it fast, because it spends memory the expert cache was already using well. Set
`CLOVER_MEMORY_BUDGET_MB` to roughly sixteen requests' worth on this hardware. A budget
that adapts from observed throughput is not built.

Every request above used the identical prompt, so they share experts and page cache
perfectly — this is the best case. Eight *distinct* prompts gave 60% efficiency at eight
concurrent where this shows 81%. Real traffic lands between the two.

Concurrency was a compiled-in constant until it was measured: first 4, then 8, with a
claim that the ceiling sat near eight. Measurement put the useful figure at four times
that, so the constant is gone rather than re-tuned.

What a request costs is measured, not estimated:

| | per layer | x 93 stages |
|---|---|---|
| `TransformerSequence` | 7.41 MB | **689 MB** |
| of which `recurrent` | 6.00 MB | 558 MB |
| of which `history` | 432 KB | 39 MB |
| of which `snapshots` | 224 KB | 20 MB |

Roughly 90% of that is genuine conversation memory rather than scratch. The KDA
recurrent state and the MLA cache exist because token N attends to tokens 1..N-1; they
cannot be shared between requests and cannot be recomputed. That, and not CPU or disk,
is what bounds how many requests run at once.

**None of this survives individual hosting.** Slots, sessions and admission are
consequences of 93 stages sharing one process on one box. With each layer on its own
machine there is no shared process to allocate anything in: a layer service holds the
state for the conversations passing through it, and concurrency comes from pipelining,
layer N working on one request while layer N+1 works on another. The numbers above are
properties of this deployment, not of the architecture.

The comparison that matters is against the standalone engine on the same hardware, which
serves a single request at a time and reaches **0.161 positions/s**. Eight concurrent
requests here reach 0.141. Concurrency recovered parity with one well-tuned process; it
did not exceed it. The standalone wins per request because it runs 16 threads instead of
4 and batches positions with expert-major grouping, so each expert it reads serves
several positions at once. That raises work done per byte read; concurrency only raises
bytes read.

## Evidence that the split computes the same thing

| check | result |
|---|---|
| layers 2–92 against `clover-one.c` | **91 of 91 bit-exact**, residual and snapshots, 5 positions each |
| layer 1 against its pinned original fixtures | 10 of 10 positions, france and japan, residual, 16 routes and S0 exact |
| server against its own layer-0 reference | exact at 5 positions |
| full chain, `The capital of France is` | token 17374, ` Paris` |
| full chain, six tokens | `17374, 20829, 10, 427, 414, 1008` — identical to the standalone engine |

For layers 2–92 the oracle is `clover-one.c` itself: a harness dumps every layer's input
residual, input snapshots, output residual and output snapshots for all five france
positions, and each pod replays those records and must match every float. That is a
stronger check than the pinned originals it replaced, because it compares against the
engine actually in use.

## What is not built yet

This is an honest list, not a caveat. None of it is hard; none of it exists.

- **No network transport.** A search of every `.c` and `.h` for
  `socket|bind|listen|accept|AF_INET|grpc` returns nothing. The 93 stages are
  `dlopen`'d into one process, and the only other interface is a `--stream` mode that
  reads whitespace-separated floats from stdin. Nothing can answer a Kubernetes Service.
- **The stream format is ASCII floats parsed by `scanf`.** A hop would carry about
  900 KB of text where 252 KB of binary would do. A length-prefixed binary frame is
  needed.
- **No session routing.** Concurrency today is requests inside one process, each holding
  its own sequence in every stage. Across pods, each frame needs a session id so a pod
  can pick the right sequence, with creation and eviction.
- **No images, manifests or volumes.** 92 × 15.512 GB = 1,427 GB has to be provisioned
  per stage, and the datasets are currently symlinks into a local path.
- **Ordering is unenforced.** Positions must reach each layer in order; nothing
  guarantees that over a network.

The minimum to host this is one small service wrapper shared by all stages: listen,
binary frames, a session id per frame, readiness probes. That is one new file, not 92
changes.
