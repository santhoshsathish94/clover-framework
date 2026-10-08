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

---

# Closed by measurement — do not retry

- **Collapsing the experts into one matrix per layer.** Not available exactly.
  A layer computes `sum over j in TopK(x) of w_j(x) * down_j(silu(gate_j(x)) *
  up_j(x))`, which depends on the input three separate ways: which `j` (a
  discrete choice), the mixing weights, and SiLU. A fixed matrix cannot select a
  different subspace per input. Note also that top-16 is not an approximation of
  a dense 896-expert computation — the model was trained with top-16 routing, so
  running all 896 would be a different and wrong answer, not a more exact one.
  As an approximation across many inputs this is distillation: a different
  model, which the "tokens identical" gate rules out.
- **Low-rank factorisation of the expert matrices.** Measured on real weights,
  extracted exactly by projecting unit basis vectors through the engine's own
  kernel, then SVD:

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
