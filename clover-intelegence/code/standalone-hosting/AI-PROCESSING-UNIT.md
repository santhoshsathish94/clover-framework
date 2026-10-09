# The machine this model wants

Why memory is the constraint, why arithmetic is not, why neither a CPU nor a GPU
is the right shape for it, and what a machine built for it would look like.

Everything in the **measured** columns is from `k3` (Ryzen 9 7950X3D, 16 of 32
threads, 4 x 32 GB DDR5 across 2 channels, RAID1 of two KIOXIA CD8). Everything
else is arithmetic on those measurements and is labelled derived. Nothing in the
architecture section has been built.

Read [LATENCY-AND-SCALING.md](LATENCY-AND-SCALING.md) first — this document
starts where that one stops.

---

## 1. The two numbers that decide everything

One decode token, single stream, per-layer expert cache at 24:

| | measured |
|---|---|
| time | **3.07 s** |
| bytes crossing the memory bus | **98.6 GB** |
| arithmetic performed | **208 GFLOP** (derived, see below) |

Those give the workload's **arithmetic intensity**:

```
208 GFLOP / 98.6 GB  =  2.11 FLOP per byte
```

Two operations for every byte fetched. That number is not an implementation
choice. A matrix-by-vector product reads each weight exactly once and does one
multiply and one add with it. For an int8 weight that is 2 FLOP per byte; for a
4-bit weight, two weights per byte, 4 FLOP per byte. Both confirmed against the
engine's own counters:

```
Q trunk, int8    107.58 GFLOP / 53.8251 GB  =  2.00 FLOP/byte
X experts, 4-bit   161.1 GFLOP/s at 42.8 GB/s  =  3.76 FLOP/byte
```

**There is no algorithm that lowers this.** You cannot do fewer than one
multiply-add per weight, and you cannot avoid reading a weight you are going to
multiply by. The ratio is a property of the operation, not of the code.

### Where the 208 GFLOP comes from

| | bytes | FLOP/byte | GFLOP |
|---|---|---|---|
| `Q` trunk, int8 | 53.83 | 2.00 (measured) | 107.58 |
| `X` experts, 4-bit | 25.83 | 3.76 (measured) | 97.1 |
| `B` lm_head, bf16 | 2.35 | 1.00 | 2.35 |
| router, int8 | 0.59 | 2.00 | 1.18 |
| **total** | **82.6 GB read** | | **208 GFLOP** |

Plus ~16 GB of concurrent reader DMA writes, giving the 98.6 GB of bus traffic.

---

## 2. The arithmetic unit is idle 82% of the time

The highest rate this kernel has ever been measured at on this box is **377.9
GFLOP/s** — `expert-bound.c` at `count=8`, where one expert's weights serve
eight positions so the bus is not binding. That is the chip's demonstrated
ceiling for exactly this arithmetic.

```
208 GFLOP / 377.9 GFLOP/s  =  0.55 s of arithmetic
measured token                3.07 s
```

**The arithmetic is 18% of the token. The other 82% the multipliers have nothing
to multiply.**

The memory side, measured over the same token:

```
98.6 GB / 47.8 GB/s observed during X  =  2.06 s  =  67% of the token
```

Memory time is **3.7x** the arithmetic time. The remaining 1.0 s is the part
that does not overlap: expert read stall plus arithmetic that cannot hide under
the stream.

This is confirmed from the other direction by the thread-scaling sweep:

| threads | GFLOP/s | GB/s | scaling |
|---|---|---|---|
| 1 | 11.6 | 3.1 | 1.00x |
| 8 | 95.2 | 25.3 | 8.24x |
| 16 | **161.1** | **42.8** | 13.93x |
| 32 | 134.3 | 35.7 | **11.61x** |

**Doubling the threads made it 18% slower.** More arithmetic units made the
machine worse, because they contend for the same memory ports. That is the whole
argument in one row: on this workload, compute is not the scarce thing — adding
it costs rather than pays.

### A note on this box specifically

`dmidecode` reports the DIMMs rated **4800 MT/s** and configured at **3600
MT/s** — four dual-rank modules across two channels, which on AM5 routinely
forces the controller down. The 42.8 GB/s ceiling above is therefore ~25% below
what these parts are rated for. Worth fixing, and it does not change any
conclusion here: it moves the ceiling, not the ratio.

---

## 3. Why a CPU is the wrong shape

Every machine has a **balance point** — the arithmetic intensity at which its
compute and its memory saturate together. Below it, the memory system is the
limit and the arithmetic units idle. Above it, the reverse.

```
balance = peak FLOP/s / peak bytes/s
```

| | FLOP/s | bytes/s | balance | vs this workload (2.11) |
|---|---|---|---|---|
| 7950X3D, FP64, no FMA | ~538 G (derived) | 42.8 G (measured) | **12.6** | 6.0x too high |

The CPU is built to do 12.6 operations per byte it can fetch. This workload has
2.11 operations to offer. **Five out of every six arithmetic slots have nothing
to do**, which is exactly the 82% idle measured above, arrived at independently.

A CPU is a latency-optimised machine: caches, branch prediction, out-of-order
execution, speculative loads. All of that exists to make *reused* data fast.
This workload has **no reuse** — the trunk is read exactly once per token, at
page granularity, 100% coverage, measured. A cache hierarchy has nothing to hold.
The 128 MB of L3 on this part, its most expensive feature, is serving a working
set of 98.6 GB that never repeats.

---

## 4. Why a GPU is also the wrong shape, and worse at batch 1

A GPU raises both sides, but it raises compute far faster than bandwidth. That
is the entire point of a tensor core: more arithmetic per unit of memory
traffic. The published H100 SXM figures (**vendor spec, not measured here**):

| | FLOP/s | bytes/s | balance | vs 2.11 |
|---|---|---|---|---|
| H100, FP64 tensor | 67 T | 3.35 T | **20** | 9.5x too high |
| H100, BF16 tensor | 989 T | 3.35 T | **295** | **140x too high** |

Even granting the model fitted in 80 GB — it does not, it is ~1.5 TB — a token
on an H100 would be:

```
memory   98.6 GB / 3.35 TB/s  =  29.4 ms
compute  208 GFLOP / 67 TFLOP/s  =  3.1 ms
```

**Still memory-bound by 9.5x. The tensor cores would be ~10% occupied.** The HBM
is a genuine 70x improvement over this box's DDR5 and would be worth having; the
tensor cores are not the reason, and are mostly not used.

Three further problems, all structural:

1. **It does not fit.** 1.5 TB of weights against 80 GB of HBM. The weights
   would stream in over PCIe or NVLink, both slower than this box's DDR5 bus.
   You would be building a GPU-shaped machine and then feeding it at DDR speeds.
2. **The only way to use a GPU well is batching**, which raises FLOP/byte
   linearly by reusing one weight read across many sequences. To reach 295
   FLOP/byte from 2.11 you need roughly 140 concurrent sequences.
3. **Batching makes single-request latency worse, and we measured it.** At 8
   lanes the per-step time went 3.40 s to 10.2 s. Per position it halved; per
   request it tripled. A request inside a batch of 140 waits for 140.

**So the GPU answer is: be fast only when you are serving many users at once.**
If what you want is one prompt answered quickly, a GPU at batch 1 is an
extremely expensive way to be memory-bound.

---

## 5. What the machine should be instead

Not fewer FLOPs per byte of silicon. **More bytes per second per FLOP** — a part
whose balance point is the workload's balance point:

```
design rule:  bandwidth and arithmetic matched at ~2.6 FLOP per byte
```

(2.6 against the 82.6 GB actually read; 2.11 against the 98.6 GB of total bus
traffic including DMA. Use 2.6 for sizing a unit's internal pair.)

And the hard part is not the ratio. It is that nobody can build **1.5 TB at
800 GB/s** as one memory system. The observation that makes this tractable is
that nobody needs to.

### The structure is already there: 93 independent working sets

| | measured |
|---|---|
| layers | 93 (92 MoE + layer 0 dense) |
| expert pool per layer | 896 x 17,547,264 B = **15.72 GB** |
| trunk slice per layer | 423 MB (MLA) / 635 MB (KDA) / 1172 MB (L0), mean **586 MB** |
| **held per layer** | **~16.3 GB** |
| expert bytes read per token | 16 x 17,547,264 B = **281 MB** |
| trunk bytes read per token | **586 MB** (every byte, once — 100% coverage measured) |
| **read per layer per token** | **867 MB — 5.3% of what it holds** |
| arithmetic per layer per token | 208 / 93 = **2.24 GFLOP** |

A layer never reads another layer's weights. Not once, in any measurement taken.
The 1.5 TB is **93 disjoint 16 GB working sets that happen to share a bus.**

Sharing that bus is the only reason the bandwidth problem looks unsolvable.

### What actually has to travel between layers

| | bytes |
|---|---|
| residual vector, 7168 x f32 | 28,672 |
| with the full snapshot stack (deepest layers, 9 vectors) | **258,048** |

Measured: 57,344 B per position at layer 1, growing to 258,048 B at layer 85.

```
read per layer   867,000,000 B
emitted          258,048 B  (worst case)
ratio            3,360x
```

**To compute one layer, 867 MB must meet the token. Today the 867 MB travels. It
is 3,360x cheaper to move the token.**

### The unit

93 of them, one per layer, chained:

```
+----------+     +----------+     +----------+          +----------+
| layer 0  | --> | layer 1  | --> | layer 2  | -- ... -->| layer 92 |
| 16 GB    |     | 16 GB    |     | 16 GB    |          | 16 GB    |
| resident |     | resident |     | resident |          | resident |
+----------+     +----------+     +----------+          +----------+
      \_____________/ \_____________/ \_____________/
         <= 252 KB       <= 252 KB       <= 252 KB
```

Each unit holds its layer's weights **permanently**. They are loaded once, at
power-on, and never move again. A token arrives as ~252 KB, is transformed, and
leaves as ~252 KB.

What one unit needs, derived from the per-layer measurements:

| token target | per-layer budget | local bandwidth | local arithmetic |
|---|---|---|---|
| 3.07 s (today) | 33.0 ms | 26 GB/s | 68 GFLOP/s |
| 1.0 s | 10.8 ms | 80 GB/s | 207 GFLOP/s |
| 0.5 s | 5.4 ms | 161 GB/s | 415 GFLOP/s |
| **0.1 s** | **1.08 ms** | **803 GB/s** | **2.07 TFLOP/s** |

**16 GB at 800 GB/s is a buildable part.** That is roughly two HBM2e stacks. 1.5
TB at 800 GB/s is not a part at all. The architecture's entire contribution is
turning the second problem into 93 instances of the first.

And the arithmetic beside it is modest and dull: ~2 TFLOP/s of FP64
multiply-accumulate. No transcendentals in the hot path, no branching, no
speculation, no coherence protocol, no cache hierarchy — reuse is measured at
1.00, there is nothing to cache. The expensive parts of both a CPU and a GPU are
the parts this workload does not use.

### Link traffic is negligible, and that is the point

```
92 hops x ~154 KB mean payload  =  14.2 MB per token moving between units
                                   98.6 GB per token moving inside them today
                                   6,900x less data in motion
```

At the 0.1 s target, giving a hop 10% of a layer's budget: 252 KB in 0.108 ms =
**2.3 GB/s per link.** A single 25 GbE link covers it with room to spare. The
interconnect is not where the difficulty is, which is unusual and worth saying
plainly.

### Latency and throughput come from the same hardware

Layers are sequential — layer N+1 needs layer N's output — so one token walks all
93 in order and latency is the sum. But a unit is free the moment its token
leaves, so token N+1 enters unit 0 while token N is at unit 1.

```
at 1.08 ms per layer:   latency    100 ms per token
                        throughput 926 tok/s, 93 tokens in flight
```

**Both, from the same machine, with no batching.** That is the property a GPU
cannot offer: there, utilisation *requires* batching, and batching costs
latency. Here the pipeline fills itself.

### Two thirds of each unit's reads are deterministic

Within a layer, measured:

| | bytes | known before the token arrives? |
|---|---|---|
| trunk | 586 MB (68%) | **yes** — same weights every token |
| experts | 281 MB (32%) | no — chosen by this layer's own router |

The trunk half can be streaming into the arithmetic unit before the token even
arrives. Only the expert half is data-dependent, and it cannot be predicted
earlier: measured, layer L predicts layer L+1's experts at **1.6%**.

But it can be covered. Attention runs *before* the router and is 27% of layer
time — 0.29 ms of cover at the 1.08 ms budget against 0.35 ms of expert read.
**Nearly covered, and only because the deterministic 68% was prefetched.** On the
current monolith the same structure gives 14.1 ms of cover for 20.8 ms of read,
a 6.7 ms deficit every layer. Per-unit residency is what closes it.

---

## 6. What this does not solve

Stated plainly, because a named gap is a limitation and a filled-in guess is a
defect.

1. **Nothing here is built.** Every figure past the measured column is
   arithmetic on measured inputs. The architecture is a proposal.
2. **It does not break the sequential chain.** Latency is still 93 steps in
   order. This makes each step cheap; it does not make them concurrent. The only
   identified lever on *that* is head-wise splitting inside a layer — 96 heads
   across N units working the same token, then an all-reduce. Also unbuilt, and
   a different change.
3. **The weights still have to be read.** 867 MB per layer per token does not go
   away. The architecture makes that read local and private instead of
   serialised on one shared bus. It is a bandwidth-provisioning argument, not a
   bytes-reduction argument.
4. **94.7% of each unit's memory is idle per token.** You pay for 16 GB to read
   867 MB. Batching raises the touch rate — more distinct experts per layer per
   batch, measured 16.0 distinct at batch 1 rising to 69.7 at batch 8 — which is
   where that capacity starts earning its cost. Single-stream, it mostly sits.
5. **2.6 FLOP/byte is a property of this engine's path**: int8 trunk, 4-bit
   experts, FP64 accumulate, `-ffp-contract=off`. Change the numeric format and
   the design point moves. A different path computing the same input-to-output
   map could have a different ratio.
6. **Measured on one machine, one engine.** Every performance verdict above is
   scoped to `k3` and to this code. None of it is a lower bound on all paths.

---

## 7. What to test first, on hardware that exists

The architecture makes two separable claims. One is testable now; one is not.

**Testable now — the pipeline claim.** `distrubuted-hosting` already shards
layer-wise: 92 stages, each with its own `trunk-N`, `root-N` and
`operators/qkv-all/layer-N`, ~16-20 GB per stage. Running it with each stage's
working set fully RAM-resident on its own node tests:

- that the inter-unit payload really is ~252 KB and the hops really are free
- that per-layer residency removes the expert read stall (predicted ~33 ms to
  ~25 ms per layer, ~1.28x latency, ~40 tok/s aggregate)
- that the pipeline fills and gives throughput without batching

That is the part that is *architecture*, and it needs no new silicon — only
rented nodes.

**Not testable here — the bandwidth claim.** Nothing on this box, or on any
rentable box, provides 16 GB at 800 GB/s to a 2 TFLOP/s FP64 unit. The 0.1 s
projection stands or falls on a part that does not exist yet. What the test above
*can* establish is whether the rest of the design holds up, so that the
bandwidth question is the only one left open.

---

## 8. The finding, in three sentences

This workload performs **2.11 operations per byte it fetches**. Every machine
available to run it is built for 12 to 295, so between 82% and 90% of their
arithmetic sits idle while the memory system does all the work — measured on a
CPU, and structural on a GPU.

The 1.5 TB of weights is not one memory system; it is **93 disjoint 16 GB
working sets**, and the token that must visit all of them is **252 KB**.

Therefore: stop moving 98.6 GB of weights to the token, and move the 252 KB
token to the weights — **93 small, cheap, balanced memory-plus-arithmetic units
in a chain, holding their layer forever and passing only the token.** 3,360x less
data in motion per layer, latency and throughput from the same hardware, and
every unit a part that can actually be built.
