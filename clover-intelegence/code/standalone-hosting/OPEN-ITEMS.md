# Open items — standalone-hosting

State after 7b6fd26. Each item says what is known, what is not, and what would
settle it. Measurements are on `k3` (Ryzen 9 7950X3D, 16 threads of 32, 124 GB,
RAID1 of two KIOXIA CD8 on PCIe 4.0 x4) unless stated otherwise.

## 1. `X` expert projection is at its floor — measured, closed

0.913 s of a 3474 ms decode token, and flat at 141-146 s across every lane
count, because expert selection is per position so N lanes pull N different
experts. The per-layer cache does not help it either: a hit saves the disk read,
but the bytes still stream from RAM and are still decoded.

I assumed this was decode-instruction-bound. It is not. Holding one expert fixed
and varying `count`, 60 repetitions, layer 1 expert 498:

| count | wall | s/position | GFLOP/s |
|---|---|---|---|
| 1 | 0.0263 | 0.02626 | 150.9 |
| 2 | 0.0422 | 0.02109 | 187.9 |
| 4 | 0.0601 | 0.01504 | 263.6 |
| 8 | 0.0839 | 0.01049 | 377.9 |

`wall = 0.0283 + 0.0070 * count` over count 2..8, so one expert at one position
is **0.471 ms fixed and 0.116 ms arithmetic, 80% fixed**. The model predicts
0.587 x 1472 = 864 ms against 913 ms measured, so it holds.

The fixed part runs at **37.2 GB/s**, which is the 17.5 MB of 4-bit codes being
streamed out of RAM. Q achieves 41.7 GB/s on a simpler kernel and the ceiling is
about 47. So X is already at 80-90% of what the bus will give, and the decode
instructions are riding underneath the stream rather than setting the rate.

Perfect bandwidth would take X from 913 to roughly 800 ms, about 4% of a token.
Not worth the risk to a numerically exact kernel.

**The only real lever is moving fewer expert bytes, and that is closed:**
reduced computation does not exist (see below), 4-bit is already the format, and
fewer experts changes the output.

### What this says about the whole token

Per token the engine moves 53.8 GB for Q and 25.8 GB for X, 79.6 GB total. At
~45 GB/s that is ~1769 ms of pure memory traffic against a 2972 ms token with
the cache on. Q and X cannot overlap with each other — within a layer the order
is attention, router, experts, and each depends on the last — so the bus is the
constraint and it is already ~85% busy during compute.

The token decomposes roughly as 1769 ms of memory traffic, ~425 ms of residual
expert read stall, and the remainder in arithmetic that does not hide under the
stream.

## 2. Residual expert read stall — measured, mostly already gone

Split into launch (submit to first byte landing) and tail (first to last reader)
by new counters `expert_launch_seconds` and `expert_tail_seconds`. One prompt,
six output tokens, steady state:

| | total | launch | tail | stall | read | hit |
|---|---|---|---|---|---|---|
| cache off | 3.600 s | 1051.7 ms | 1255.6 ms | 898.3 ms | 25.8 GB | 0% |
| cache 24 | 2.767 s | 283.0 ms | 989.9 ms | 179.2 ms | 10.7 GB | 59% |

Launch was not wake-up latency. It is the time for the first single 17.5 MB
expert read to finish, because each expert is assigned to one reader and one
O_DIRECT stream runs at about 1.5 GB/s. The tail of 990 ms is real but largely
hidden behind compute, since only 179 ms of it actually blocks.

Remaining value here is about 180 ms, 6% of a token. Striping one expert across
several readers would land the first expert sooner and let projection start
earlier, but the ceiling is small.

Note the hit rate here is 59%, against 42% over eight different prompts, because
one prompt generating six tokens is maximal reuse. The 42% figure is the honest
general one.

## 3. The engine is now memory-bandwidth-bound end to end

Per token it moves 53.8 GB for Q and 25.8 GB for X, 79.6 GB, at about 35 GB/s
effective. That is the whole remaining cost:

  2767 ms token = ~2263 ms Q and X, bandwidth-bound
                + ~179 ms residual read stall
                + ~325 ms everything else

Q and X cannot overlap: within a layer the order is attention, router, experts,
each depending on the last. So the bus is the constraint.

    t_token  >=  bytes_per_token / aggregate_memory_bandwidth

79.6 GB a token is fixed by the model and top-16 routing. Nothing in software
changes it: reduced computation does not exist, 4-bit is already the format, and
fewer experts changes the output. **The remaining lever is hardware, and it is
bandwidth, not capacity.**

## 3. Nothing addresses latency at high concurrency

The two levers are alternatives, not a stack. Batching is throughput at high
concurrency; the per-layer cache is latency at low concurrency, and it is
bypassed outright once a batch wants more distinct experts than a layer holds.
At 8 lanes a cap-24 cache behaves like 3 per lane. Holding 41% at eight lanes
would need 192/layer, about 310 GB.

## 4. Layer 90's recorded activation does not match `silu(gate)*up`

Off by 0.45 absolute against a 3.9 peak, 12%, while layers 1 and 45 agree to
under 0.1%. Reason unknown. Does not affect the concentration results, which are
measured on the recorded activation itself, but it means no claim can be made
that the formula holds at every depth.

## 3. The lane dimension in the expert kernel is only a fifth used

`root_project_rows` takes `count` position vectors against one expert's weights,
so positions picking the same expert stream its 17.5 MB once instead of once
each. That is the 80% fixed cost. The grouping exists and works: positions are
merged into groups by expert before the call. `resident_group_passes` and
`resident_group_pairs` counted it all along and were never reported; they are
in STEP_JSON now.

Eight different prompts, four output tokens each:

| | passes | pairs | avg count | `op:X` |
|---|---|---|---|---|
| 1 lane | 76,175 | 89,792 | 1.18 | 54.8 s |
| 8 lanes | 63,352 | 89,792 | 1.42 | 51.7 s |

Per decode step: one lane is 1472 passes for 1472 pairs, count 1.00. Eight lanes
is 7023-7759 passes for 11776 pairs, count 1.52-1.68. `pairs` identical either
way confirms the same total work is counted.

So eight lanes fill 1.6 of 8 slots, about 20%. A layer's 128 position-expert
pairs land on roughly 80 distinct experts because eight different prompts route
differently. Predicted 1.84 from consecutive tokens of a single prompt, which
overlap more than separate prompts do, so the direction was right and the
magnitude slightly optimistic.

**This is why `X` looked flat across lane counts.** 54.8 -> 51.7 s is 5.7%,
inside the +-3.5% spread of the earlier 141-146 s figures. Not a contradiction,
just too small to see at that resolution.

If `count` reached 8 the fixed cost would amortise eight ways, worth roughly 3x
on X. It cannot be forced: the route-sensitivity branch showed that making
positions share experts they did not choose changes the answer. Genuine sharing
is bounded by how different the concurrent prompts are, and `count` only grows
with batch width once the batch is far wider than 56, since 16 x lanes pairs
must spread over at most 896 experts.

## 4. Pods would not give N times the speed, and the trunk is the reason

Measured on the real 32-token route trace, modelling expert parallelism: each
pod holds a slice of every layer's 896 experts, a token selects 16, they land
where they land, and the layer is not done until the slowest pod is. So the
cost is `max over pods of count`, not `16/N`.

| pods | mean deepest pod | speedup | ideal | efficiency |
|---|---|---|---|---|
| 2 | 9.52 | 1.68 | 2 | 84% |
| 4 | 6.07 | 2.64 | 4 | 66% |
| 8 | 4.12 | 3.88 | 8 | 49% |
| 14 | 3.17 | 5.05 | 14 | 36% |
| 16 | 3.02 | 5.30 | 16 | 33% |
| 56 | 2.00 | 8.01 | 16 | 50% |

Balls into bins. Sixteen experts over fourteen pods leaves the deepest pod
holding about three, so the critical path is three deep rather than 1.14.
Frequency-balanced placement (LPT on observed usage) helps a little, 5.05 ->
5.80 at fourteen pods. Contiguous blocks are the same as round robin.

Sixteen experts cannot occupy more than sixteen pods, so past N=16 the critical
path cannot shrink further for a single request however many pods are added.

**Batching and pods compose**, which the per-layer cache and batching do not:

| pods | batch | distinct experts | deepest pod | speedup | efficiency |
|---|---|---|---|---|---|
| 14 | 1 | 16.0 | 3.17 | 5.05 | 36% |
| 14 | 2 | 26.2 | 4.40 | 5.94 | 42% |
| 14 | 4 | 43.0 | 6.23 | 6.90 | 49% |
| 14 | 8 | 69.7 | 8.85 | 7.88 | 56% |
| 28 | 8 | 69.7 | 5.84 | 11.94 | 43% |

Eight concurrent tokens need 69.7 distinct experts rather than 128, which is
the 42.6% cross-request overlap appearing again, this time as a benefit.

### The part that decides the architecture

A token moves 79.6 GB: **53.8 GB of trunk for Q and only 25.8 GB of experts.**
Expert parallelism addresses the smaller half.

If each pod simply holds a copy of the trunk, every pod does the full Q and
nothing about Q improves. The token floor becomes roughly `1290 ms of Q +
913/5 ms of experts = 1473 ms`, about 2x better than the 2972 ms measured
today. Two, not fourteen.

To get more, the trunk has to be sharded as well, by attention head, with an
all-reduce of the residual per layer. That is a second parallelism axis and a
different piece of work. The vector traffic stays small either way: the
residual is 7168 floats, 28.7 KB a layer, against 79.6 GB of weights.

**So the honest projection for pods is: about 2x if only the experts are
distributed, and much more only if the trunk is distributed too.** Not
measured on real hardware; this is arithmetic on measured inputs.

## 5. Distributed model still defaults hardmax to 46

The standalone engine had hardmax removed entirely, so the two have now drifted
the other way. Aligning means regenerating 92 `transformer-N.c` via
`port-stages.mjs`.

## 6. Dead scaffolding left in place

`cross_layer_submit` and `cross_layer_match` are never called. `sel_fp` and
`route_fp` are declared and never assigned; `route-probe.c` drives `sel_fp` from
a harness instead, so the engine is unchanged. Decide whether to wire or delete.

## 7. No reader parses the 4000 aggregate capture range

Data exists, uninterpreted.

## 8. Server owner 0 has no stage machine

Stages 1-120 reserved, never implemented.

## 9. Understanding the problem is not finished

Everything measured so far characterises the path this engine takes: read the
weights the model specifies, in the order the architecture specifies, and the
cost is 79.6 GB a token. What has *not* been established is a lower bound on
any path producing the same input-to-output map.

Only the input and the output are fixed. Every intermediate transformation is
one implementation of the map, and a cheaper one may exist. The measurements
rule out a particular family — per-matrix low-rank, shared bases across
experts, activation truncation, adjacent-layer route prediction, faster
multiplication — all of which keep the existing computational structure and try
to shrink a piece of it. They say nothing about a different structure.

Specific things never tested:

- structure *across layers* rather than within one
- nonlinear relationships between experts, as opposed to the linear ones ruled
  out by the orthogonality result
- whether the realised map, for a given input, is reachable by a materially
  shorter computation
- **route sensitivity**: how exactly must the 16 of 896 be right? Forcing token
  N's route onto token N+1 and sweeping how many of the sixteen must be exact
  would say whether the routing carries as much information as it costs. This
  one is cheap, the machinery half exists in `K3_ROUTESAVE`/`K3_ROUTELOAD`, and
  it is the obvious next experiment.

The negative results above are worth having because they are specific and
repeatable, not because they settle the general question.

---

# Closed by measurement — do not retry

**Scope note.** Everything below is a measurement of *the path this engine
currently takes*, not a lower bound on every path that computes the same
input-to-output map. "Closed" here means this specific approach was tried and
does not pay. It does not mean no cheaper computation exists. An earlier version
of this file claimed the 4-bit matrix was "near the minimum representation for
this model" — that was a far larger claim than the evidence supports and it has
been removed.

- **A shared base across the experts of a layer.** If the 896 experts of a layer
  were a common matrix plus small per-expert deltas, a layer could read the base
  once and only sixteen deltas, which would attack the bandwidth problem
  directly. Measured on eight experts of layer 1, gate matrix:

  pairwise cosine between flattened experts is 0.000 to 0.001, so they are
  mutually orthogonal. `|B|/mean|W_j| = 0.3537` against `1/sqrt(8) = 0.3536`
  predicted for independent matrices, and `|W_j - B|/|W_j| = 0.931-0.943`
  against `sqrt(7/8) = 0.9354` predicted. The match to the independence
  prediction is exact, so the mean carries no shared signal. The residual's
  effective rank is the same or *higher* than the original (1629 -> 1629,
  1451 -> 1498, 2210 -> 2344), so subtracting a base does not simplify anything.

  MoE training appears to have made the experts maximally decorrelated, which is
  what a mixture should do: no redundancy to exploit. This closes linear shared
  structure *across experts within a layer*. It does not address nonlinear
  relationships, structure across layers, or a different functional form.

- **Collapsing the experts into one matrix per layer.** Not available exactly.
  A layer computes `sum over j in TopK(x) of w_j(x) * down_j(silu(gate_j(x)) *
  up_j(x))`, which depends on the input three separate ways: which `j` (a
  discrete choice), the mixing weights, and SiLU. A fixed matrix cannot select a
  different subspace per input. As an approximation across many inputs this is
  distillation: a different model, which the "tokens identical" gate rules out.

  On why top-16 exists: sparse MoE is a tractability compromise, chosen because
  a dense model at the same capacity is infeasible to train and run. That is a
  fair reading. The narrower fact is that *these* weights were trained under
  top-16 routing, so running all 896 at inference would be out of distribution
  for this checkpoint. That is a statement about the checkpoint, not a defence
  of the architecture as optimal.

- **Low-rank factorisation of individual expert matrices.** Measured on real
  weights, extracted exactly by projecting unit basis vectors through the
  engine's own kernel, then SVD:

  | layer | expert | condition | stable rank | r@90% | r@99% | rank-413 energy | fp16 at 99% |
  |---|---|---|---|---|---|---|---|
  | 1 | 498 | 29.1 | 743 | 1629 | 2498 | 39.7% | 33.3 MB |
  | 45 | 107 | 27.9 | 742 | 1665 | 2522 | 38.6% | 33.6 MB |
  | 90 | 128 | 56.9 | 181 | 1644 | 2511 | 40.7% | 33.4 MB |

  Storing gate as rank-r fp16 factors beats the 5.5 MB 4-bit form only below
  r = 413, 13% of 3072. Rank 413 captures 39-41% of the energy. Reaching 99%
  needs rank ~2500 and costs 33 MB, six times *more* than the current form. A
  condition number near 29 means the smallest singular value is only 29x below
  the largest: there is no tail to truncate and the matrix is close to
  isotropic. Consistent at layers 1, 45 and 90.
- **Faster multiplication algorithms.** Strassen and relatives reduce the
  multiplication count for matrix-by-matrix, which pays only when a matrix is
  reused across many vectors. The expert projection is matrix-by-vector at
  count ~1.29, where every element must be read at least once regardless of
  algorithm. The kernel sits at 21% of FLOP peak and 80-90% of memory
  bandwidth, so removing arithmetic changes nothing. The only currency is bytes
  read.
- **One-layer-ahead expert prefetch.** Layer L predicts layer L+1 at **1.6%**.
  Adjacent layers share essentially nothing, so no implementation of this works.
- **Reduced expert computation.** The activation is concentrated — top-512 of
  3072 holds 93-99.9% of L2 mass — but not the same 512 twice: overlap between
  two inputs of one expert is 27.7-28.7% against a 16.7% random baseline, and
  `gate` alone identifies the carriers at 52-66% against 17% random. Which bytes
  matter is decided by the input, and the computation that identifies them is
  the one being avoided.
- **Global LRU expert cache.** Flat at 23.3% for every capacity. The diagnosis
  (cyclic-sweep eviction) was right; the conclusion that caching does not pay
  was wrong. Per-layer partitioning is what works.
- **Route-driven prefetch from a saved route.** 30% ceiling and prompt-specific,
  the same limitation as the prefix cache.
- **More read concurrency.** 16 readers is the knee; 32, 48 and 64 are worse.
- **Four or more concurrent processes.** 470 s against 62 s at two workers.
- **`K3_RESIDENT_WARMUP=1`.** Costs 19.74 s, saves 3.1.
- **Trunk in `/dev/shm`.** Starves the expert page cache.
- **Reader thread using OpenMP.** Its team competes with the compute team.
- **`Qm`'s output-blocked path at `T>=8`.** Re-read every weight row T times and
  never took the direct int8 decode. Removed.
