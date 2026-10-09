# Latency and scaling — what is measured, and what follows from it

Everything here is measured on `k3` (Ryzen 9 7950X3D, 16 of 32 threads, 124 GB,
RAID1 of two KIOXIA CD8 on PCIe 4.0 x4) unless marked as derived. Derived
figures are arithmetic on measured inputs and are labelled.

For what follows from all of this about hardware — why the arithmetic unit is
idle 82% of the time, and what a machine matched to this workload would look
like — see [AI-PROCESSING-UNIT.md](AI-PROCESSING-UNIT.md).

---

## 1. One token moves 79.6 GB, and it splits in two

This single fact explains almost everything below.

| | bytes a token | scales with positions in a sweep? |
|---|---|---|
| trunk, `op:Q` | 53.8 GB | **no** — read once per sweep |
| experts, `op:X` | 25.8 GB | **yes** — per position, ~1.6x sharing at 8 lanes |

The proof is in the engine's own `OPSTAT_JSON`, same build, same run:

```
prompt pass, 5 positions   Q: 1.561 s   537.88 GFLOP   wgb 53.8251 GB   calls 1159
decode step,  1 position   Q: 1.351 s   107.58 GFLOP   wgb 53.8251 GB   calls 1159
```

`wgb` is weight bytes. Identical to four decimals. Five times the arithmetic for
the same bytes and 1.16 times the time.

**68% of the traffic is shareable across positions. 32% is not.**

---

## 2. Where a decode token actually goes

3474 ms without the expert cache, spread evenly over 92 MoE layers at 36.5 ms
mean, range 33-44. There is no hot layer, so nothing local to attack.

| per layer | ms |
|---|---|
| attention, runs **before** the router | 9.8 |
| shared expert | 3.7 |
| expert projection | 10.4 |
| expert read stall | 11.4 |
| rest | 1.2 |

A layer can only hide its expert reads behind work that comes *after* its own
router: 14.1 ms of cover for 20.8 ms of read. A 6.7 ms structural deficit every
layer, 1060 ms of stall, 30.5% of the token. The per-layer expert cache closes
the structural part; residual stall is ~179 ms.

---

## 3. The two levers, and why they do not stack

**Batching** shares the trunk. **The per-layer cache** reduces the expert reads.
Different halves of the 79.6 GB.

| lever | what it touches | measured |
|---|---|---|
| batching, 8 lanes | the 53.8 GB trunk | 479.6 -> 276.4 s for 8 prompts x 16 tokens, 1.74x throughput |
| per-layer cache, 24/layer | the 25.8 GB experts | 3.608 -> 2.983 s a token, 1.21x latency, every prompt |

They do not compose: the cache is bypassed once a batch wants more distinct
experts than a layer holds, which 8 lanes always does.

### Batching improves throughput and makes single-request latency worse

| | measured |
|---|---|
| per step, width 1 -> 6 | 3.40 s -> 10.2 s |
| per position | 3.40 s -> 1.70 s |

Both are true. The step got slower because total bytes rose from 79.6 to ~180 GB;
per position it got faster because the 53.8 GB trunk was paid once instead of six
times. A request inside a six-wide batch waits **longer** for each of its tokens.

**Batching and the cache must never be quoted together as one speedup.**

---

## 4. A single five-token prompt

Single stream, cache at 24, measured:

```
prompt pass (5 tokens, one sweep)  ->  9.9 s      produces output token 1
each further output token          ->  3.05 s
t = 9.9 + 3.05 * (N - 1)
```

| output tokens | time |
|---|---|
| 1 | 9.9 s |
| 8 | 31.3 s |
| 32 | 104 s |
| 64 | 202 s |

Process startup is ~9.4 s and is paid once per resident process, not per request.

For this case the cache is the only lever that applies. Batching does not, being
single stream, and neither does the `Qm` blocked-path removal, which only
engaged at `T>=8` while a five-token prompt pass runs at `T=5`.

---

## 5. The hardware is already saturated

One expert projection, count=1, scaling threads:

| threads | wall | GFLOP/s | GB/s | scaling |
|---|---|---|---|---|
| 1 | 0.3426 | 11.6 | 3.1 | 1.00x |
| 2 | 0.1854 | 21.4 | 5.7 | 1.85x |
| 4 | 0.0858 | 46.2 | 12.3 | 3.99x |
| 8 | 0.0416 | 95.2 | 25.3 | 8.24x |
| 16 | 0.0246 | 161.1 | 42.8 | **13.93x** |
| 32 | 0.0295 | 134.3 | 35.7 | 11.61x |

87% of ideal scaling to 16 threads, then the memory bus. 42.8 GB/s against Q's
41.7 and a ~47 ceiling. **32 threads is 18% worse** — this is a 16-core part and
the SMT siblings contend for the same memory ports.

So the limit is bytes per second, not operations per second. The 16 experts of a
layer are already parallel across threads; restructuring them to run
concurrently would be worse, because the engine currently computes expert *n*
while expert *n+1* is still being read.

---

## 6. Sharding: layer-wise and head-wise do different things

The trunk **is** already sharded by layer in `distrubuted-hosting`: each of 92
layer stages holds its own `trunk-N`, `root-N` and `operators/qkv-all/layer-N`,
20 GB a node.

Layers are sequential — layer N+1 needs layer N's residual — so one token still
walks all 92 in order: one node working, 91 idle, plus 91 network hops.

**Layer-wise, derived from measured per-layer cost:**

```
per layer today, one box   36.5 ms, of which 11.4 ms is expert read stall
per layer on a 20 GB node  ~25 ms, its 15.5 GB is RAM-resident so the stall goes
one token   92 x 25 ms + 91 hops x ~0.12 ms  =  ~2320 ms  against 2972 ms
```

About 1.28x on latency, and that comes from RAM residency, not parallelism.
Throughput is the real prize: a token can enter the pipeline every ~25 ms with
92 others in flight, roughly 40 tok/s aggregate against 0.34 tok/s on one box.

**Head-wise, not built, derived:** split one layer's 96 heads across N machines
so they work the same token simultaneously, then all-reduce.

```
53.8 GB / 14 / 42 GB/s        ~  91 ms of trunk compute
186 all-reduces x 28.7 KB     ~  28 ms of collectives
```

That is the axis that cuts single-request latency, because it divides the trunk
rather than relocating it.

### Expert parallelism alone is not enough

Measured against the real 32-token route trace, modelling pods holding slices of
each layer's 896 experts. A layer finishes when the slowest pod does, so the cost
is the deepest pod, not 16/N.

| pods | deepest pod | speedup | ideal | efficiency |
|---|---|---|---|---|
| 2 | 9.52 | 1.68 | 2 | 84% |
| 8 | 4.12 | 3.88 | 8 | 49% |
| 14 | 3.17 | 5.05 | 14 | 36% |
| 16 | 3.02 | 5.30 | 16 | 33% |
| 56 | 2.00 | 8.01 | 16 | 50% |

Balls into bins. Sixteen experts cannot occupy more than sixteen pods, so past
N=16 a single request cannot go faster however many are added.

Batching and pods **do** compose, unlike batching and the cache:

| pods | batch | distinct | deepest | speedup |
|---|---|---|---|---|
| 14 | 1 | 16.0 | 3.17 | 5.05 |
| 14 | 8 | 69.7 | 8.85 | 7.88 |
| 28 | 8 | 69.7 | 5.84 | 11.94 |

Expert parallelism addresses only the 25.8 GB. If every pod keeps a copy of the
trunk, nothing about the 53.8 GB improves, leaving roughly `1290 ms of Q +
913/5 ms of experts = 1473 ms`, about 2x. **Two, not fourteen.**

---

## 7. Speculative decoding, and its budget

Token N+1 needs token N — not just the state, the token itself, as the input
embedding for position N+1. And token N only exists after the full 93-layer pass
at position N produces logits. That is autoregression, and it is the model's
definition rather than an implementation choice.

The standard escape is to guess k tokens cheaply and verify all k in one
multi-position pass, which the engine already supports because that is what a
prompt pass is. The cost of a k-position sweep of one sequence against k
one-position decodes is the entire budget. Measured:

| T | pass | s/position | vs T=1 |
|---|---|---|---|
| 1 | 4.055 s | 4.0554 | 1.00x |
| 2 | 5.508 s | 2.7540 | 1.47x |
| 4 | 8.311 s | 2.0777 | 1.95x |
| 8 | 12.406 s | 1.5507 | 2.62x |

Break-even acceptance:

```
k=2 : need 1.4 of 2 accepted
k=4 : need 2.0 of 4 accepted
k=8 : need 3.1 of 8 accepted
```

**3.1 of 8 is a low bar.** The obstacle is not the budget, it is that there is
no draft model here to produce the guesses.

---

## 8. What is closed, and what is not

Closed by measurement, with the scope stated:

- Per-matrix low-rank. Rank 413 is break-even against the 4-bit form and
  captures 39-41% of the energy; 99% needs rank ~2500 at six times the bytes.
  Condition number ~29, no tail to truncate.
- A shared base across a layer's experts. Pairwise cosine 0.000-0.001, and
  `|B|/mean|W|` is 0.3537 against `1/sqrt(8) = 0.3536` predicted for independent
  matrices. Subtracting the mean leaves the residual no simpler.
- Adjacent-layer route prediction: layer L predicts L+1 at 1.6%.
- Route reuse: carrying 4 of 16 moves the logits 6.7%; the routing is not
  over-precise.
- Faster multiplication algorithms. Matrix-by-vector must read every element,
  and the kernel is at 21% of FLOP peak against 80-90% of bandwidth.

**Scope note.** Every item above measures *the path this engine takes*. None of
them is a lower bound on all paths computing the same input-to-output map. Only
the input and the output are fixed; the intermediate transformations are one
implementation.

Still open:

- whether consecutive tokens' per-layer residuals are similar enough to exploit
- structure across layers rather than within one
- nonlinear relationships between experts
- a draft model, which would make speculative decoding reachable at 3.1 of 8
- head-wise trunk sharding, the only identified lever on single-request latency
- the hardware shape the workload wants, worked out in
  [AI-PROCESSING-UNIT.md](AI-PROCESSING-UNIT.md): 2.11 FLOP per byte against
  machines built for 12 to 295, and 1.5 TB that is really 93 disjoint 16 GB
  working sets joined only by a 252 KB token
