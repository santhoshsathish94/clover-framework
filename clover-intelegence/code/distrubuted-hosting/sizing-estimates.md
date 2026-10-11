# Sizing estimates

Per-pod CPU, RAM and capacity for the 95-pod fleet: one request pod, one server pod,
transformer layers 1 to 92, and layer 93 for normalisation and output.

Every number below was measured on the box, not modelled. Where a figure is an
extrapolation from a sample rather than a direct measurement it says so.

## How these were measured

Hardware: AMD Ryzen 9 7950X3D, 16 cores / 32 threads, 124 GB RAM, 2x KIOXIA
KCD8XRUG1T92 NVMe behind md2.

- `tansformers/pod-cost.c` loads one layer through `dlopen`, runs six positions with a
  different input vector per position so the router picks a different sixteen experts
  each time, then locks that layer's `experts.direct` into RAM with `mlock` and runs the
  same six positions again. A single input reused would touch only ~268 MB and would
  not represent a real prompt.
- `scripts/sweep-pods.sh` does that for all 92 layers, one at a time. One at a time is
  forced: 92 x 15.72 GB is 1,446 GB and the box has 124 GB.
- `scripts/sweep-threads.sh` repeats one KDA layer and one MLA layer at 1, 2, 4, 8, 16
  and 32 threads with the experts pinned each time.
- `tansformers/edge-cost.c` and `scripts/sweep-edges.sh` do the same for the server pod
  and the layer 93 pod.
- `server/residency.c` reports what is actually resident using `mincore`, rather than
  assuming that a mapped file is a loaded file.

Raw output is in `measurements/pod-sweep.csv`, `measurements/thread-scaling.csv` and
`measurements/edge-scaling.csv`.

## RAM per pod

Measured with `stat` on the files each pod opens.

| Pod | Holds | Bytes | GB |
|---|---|---:|---:|
| Request / queue | 4,096 queued submissions | 6,586,368 | 0.007 |
| | per in-flight request: 256 frames x 258,072 | 66,066,432 | 0.066 |
| Server | trunk-0 qkv | 265,008,216 | 0.265 |
| | trunk-0 records | 1,171,710,871 | 1.172 |
| | `seed.bin.direct`, token to vector | 2,348,810,240 | 2.349 |
| | **total** | **3,785,529,327** | **3.79** |
| Layer 1-92, each | `experts.direct` | 15,722,348,544 | 15.722 |
| | constants, maps, headers | ~3,990,000 | 0.004 |
| | **total each** | **~15,726,338,544** | **15.73** |
| Layer 93 | `fruit.bin.direct`, vector to token | 2,348,810,240 | 2.349 |
| | tiktoken model | 2,795,286 | 0.003 |
| | vocabulary | 1,764,931 | 0.002 |
| | leaves.json | 58,615 | 0.00006 |
| | **total** | **2,353,429,072** | **2.35** |

Every layer pod is exactly the same size. There is no large or small layer.

On top of the weights, each layer pod holds a `TransformerSequence` of **7,771,448
bytes (7.41 MB) per concurrent request**. Across 92 layers that is 681 MB of
conversation state for every request in flight.

**Practical pod sizing: 18 GB for a layer pod** (15.73 GB of weights, plus sequence
state and headroom), **6 GB for the server pod**, **4 GB for layer 93**, **1 GB for the
request pod**.

A layer pod loads its 15.72 GB from local NVMe at a measured **2.61 GB/s**, so expect
about **6.0 s of start-up** before it can serve. That is the mean `mlock` time across
all 92 layers.

## CPU: how each pod responds to cores

Measured on layer 46 (KDA) and layer 7 (MLA), experts pinned, six positions.

| Threads | KDA ms/position | KDA speedup | KDA efficiency | MLA ms/position | MLA speedup | MLA efficiency |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 319.12 | 1.00x | 100% | 335.14 | 1.00x | 100% |
| 2 | 214.56 | 1.49x | 74% | 171.10 | 1.96x | 98% |
| 4 | 164.55 | 1.94x | 48% | 90.79 | 3.69x | 92% |
| 8 | 138.32 | 2.31x | 29% | 48.78 | 6.87x | 86% |
| 16 | 131.99 | 2.42x | 15% | 30.89 | 10.85x | 68% |
| 32 | 135.79 | 2.35x | 7% | 32.12 | 10.43x | 33% |

The two layer types behave very differently and should not get the same pod shape.

- **KDA layers stop paying for cores early.** Past 4 threads each extra core buys little,
  and 32 threads is slower than 16. Recommended: **4 cores**, 8 if latency matters more
  than density.
- **MLA layers scale almost linearly to 8 and usefully to 16.** Recommended: **8 cores**,
  16 if latency matters. 32 regresses.

The server pod and layer 93 pod are flat from 1 to 32 threads:

| Threads | Server ms/position | Layer 93 ms/token |
|---:|---:|---:|
| 1 | 431.52 | 286.84 |
| 2 | 432.18 | 285.78 |
| 4 | 431.66 | 285.90 |
| 8 | 432.01 | 285.41 |
| 16 | 431.88 | 285.25 |
| 32 | 432.11 | 285.26 |

Neither moves at all. `server.c` and `transformer-93.c` contain **zero** `#pragma omp`
directives, so this is not a missing build flag — there is no parallel region to enable.
**Give these pods 1 core each.** More cores are wasted.

## Capacity per pod

From the service times above.

| Pod | Cores | Service time | Capacity |
|---|---:|---:|---:|
| Server | 1 | 431.89 ms / position | **2.32 positions/s** |
| KDA layer | 4 | 164.55 ms / position | 6.08 positions/s |
| KDA layer | 8 | 141.99 ms / position | **7.04 positions/s** |
| KDA layer | 16 | 131.99 ms / position | 7.58 positions/s |
| MLA layer | 8 | 50.76 ms / position | **19.70 positions/s** |
| MLA layer | 16 | 30.89 ms / position | 32.37 positions/s |
| Layer 93 | 1 | 285.74 ms / token | **3.50 tokens/s** |
| Request / queue | 1 | 0.046 ms / submission | >20,000 submissions/s |

KDA at 8 and MLA at 8 are means across all 68 and all 24 layers. The 1/2/4/16/32 rows
come from one representative layer each and are extrapolated to the group.

Note the unit difference: the server and the layers run **once per position**; layer 93
runs **once per generated token**.

## The bottleneck is the server pod

At one pod each, the fleet is limited to **2.32 positions/s** by the server, and
everything behind it idles:

| Pod | Busy at 2.32 positions/s |
|---|---:|
| Server | 100% |
| KDA layer | 32.9% |
| MLA layer | 11.8% |
| Request | <0.1% |

The server costs **431.89 ms per position, three times a KDA layer and eight times an
MLA layer**, and it cannot use a second core. It needs replicas before anything else
does.

## Replica counts by target throughput

`replicas = ceil(target / pod capacity)`, using 8 cores for layers.

| Target | Server | Each KDA (x68) | Each MLA (x24) | Layer 93 | Total pods | Total cores | Total RAM |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2.32 pos/s | 1 | 1 | 1 | 1 | 95 | 738 | 1,453 GB |
| 10 pos/s | 5 | 2 | 1 | 3 | 169 | 1,288 | 2,543 GB |
| 25 pos/s | 11 | 4 | 2 | 8 | 341 | 2,595 | 5,146 GB |
| 50 pos/s | 22 | 8 | 3 | 15 | 656 | 5,014 | 9,939 GB |

Layer 93 replicas assume one generated token per position, which is the worst case. A
prompt of 5 tokens generating 2 needs layer 93 only twice for six positions, so in that
shape a third of the figure above is enough.

Worked example at 10 positions/s:

- 5 server pods, 1 core, 6 GB each
- 136 KDA pods (68 layers x 2), 8 cores, 18 GB each
- 24 MLA pods (24 layers x 1), 8 cores, 18 GB each
- 3 layer 93 pods, 1 core, 4 GB each
- 1 request pod, 1 core, 1 GB

Switching KDA pods to 4 cores costs 14% throughput per pod but halves their cores: 3
replicas each instead of 2, so 204 pods and 816 cores rather than 136 pods and 1,088
cores. Denser in cores, heavier in RAM. Which is cheaper depends on your machine prices,
and I have not costed it.

## What the pod split is actually worth

Single request, "the capital of france is", 5 prompt tokens, 2 generated, 6 positions.

| | Measured |
|---|---:|
| Monolith today, one box, experts streaming from NVMe | 96.00 s |
| Projected pods, 8 cores per layer, experts resident | **68.40 s** |

Built from: server 6 x 431.89 ms = 2.591 s, layers 65.240 s measured, layer 93
2 x 285.74 ms = 0.571 s.

That is **29% faster**, not an order of magnitude. The honest reason: across all 92
layers the cold total is 70.624 s and the resident total is 65.240 s, so **waiting on
storage is only 7.6%** of layer time. The prefetch already overlaps most of the 120 GB
a prompt reads.

The real gains from the pod split are elsewhere:

1. **It removes cache thrashing.** On one box a prompt's working set is ~120 GB against
   124 GB of RAM, so a second identical prompt read 120 GB again and ran no faster
   (96.0 s then 97.1 s). Per-pod residency makes the second request cheap.
2. **It scales past one box.** 1,453 GB cannot fit in 124 GB. This is the only way to
   hold the whole model in memory.
3. **It lets you scale the bottleneck alone.** Adding server pods costs 6 GB each
   instead of re-provisioning a whole box.

## Per-layer measurements

Six positions, 8 threads, `experts.direct` locked in RAM. Type from each layer's
`TRANSFORMER_MLA` define, not inferred from its timing.

| Layer | Type | Resident s | ms/position | Layer | Type | Resident s | ms/position |
|---:|---|---:|---:|---:|---|---:|---:|
| 1 | KDA | 0.853 | 142.15 | 47 | MLA | 0.308 | 51.28 |
| 2 | KDA | 0.849 | 141.54 | 48 | KDA | 0.880 | 146.68 |
| 3 | MLA | 0.307 | 51.10 | 49 | KDA | 0.838 | 139.65 |
| 4 | KDA | 0.841 | 140.13 | 50 | KDA | 0.877 | 146.24 |
| 5 | KDA | 0.834 | 139.00 | 51 | MLA | 0.303 | 50.52 |
| 6 | KDA | 0.834 | 138.99 | 52 | KDA | 0.866 | 144.35 |
| 7 | MLA | 0.299 | 49.86 | 53 | KDA | 0.854 | 142.40 |
| 8 | KDA | 0.848 | 141.36 | 54 | KDA | 0.841 | 140.15 |
| 9 | KDA | 0.835 | 139.20 | 55 | MLA | 0.302 | 50.36 |
| 10 | KDA | 0.873 | 145.46 | 56 | KDA | 0.853 | 142.12 |
| 11 | MLA | 0.294 | 48.94 | 57 | KDA | 0.844 | 140.60 |
| 12 | KDA | 0.848 | 141.32 | 58 | KDA | 0.857 | 142.86 |
| 13 | KDA | 0.860 | 143.35 | 59 | MLA | 0.301 | 50.22 |
| 14 | KDA | 0.863 | 143.87 | 60 | KDA | 0.854 | 142.26 |
| 15 | MLA | 0.295 | 49.22 | 61 | KDA | 0.844 | 140.64 |
| 16 | KDA | 0.917 | 152.84 | 62 | KDA | 0.843 | 140.56 |
| 17 | KDA | 0.873 | 145.43 | 63 | MLA | 0.299 | 49.87 |
| 18 | KDA | 0.838 | 139.74 | 64 | KDA | 0.862 | 143.69 |
| 19 | MLA | 0.294 | 49.05 | 65 | KDA | 0.843 | 140.46 |
| 20 | KDA | 0.872 | 145.42 | 66 | KDA | 0.845 | 140.82 |
| 21 | KDA | 0.835 | 139.21 | 67 | MLA | 0.316 | 52.72 |
| 22 | KDA | 0.849 | 141.57 | 68 | KDA | 0.840 | 139.99 |
| 23 | MLA | 0.328 | 54.59 | 69 | KDA | 0.881 | 146.90 |
| 24 | KDA | 0.867 | 144.58 | 70 | KDA | 0.847 | 141.09 |
| 25 | KDA | 0.839 | 139.89 | 71 | MLA | 0.313 | 52.12 |
| 26 | KDA | 0.866 | 144.32 | 72 | KDA | 0.842 | 140.34 |
| 27 | MLA | 0.308 | 51.41 | 73 | KDA | 0.859 | 143.10 |
| 28 | KDA | 0.841 | 140.11 | 74 | KDA | 0.848 | 141.32 |
| 29 | KDA | 0.853 | 142.10 | 75 | MLA | 0.296 | 49.27 |
| 30 | KDA | 0.851 | 141.90 | 76 | KDA | 0.841 | 140.11 |
| 31 | MLA | 0.295 | 49.15 | 77 | KDA | 0.842 | 140.27 |
| 32 | KDA | 0.851 | 141.85 | 78 | KDA | 0.838 | 139.72 |
| 33 | KDA | 0.851 | 141.90 | 79 | MLA | 0.296 | 49.35 |
| 34 | KDA | 0.876 | 145.99 | 80 | KDA | 0.838 | 139.61 |
| 35 | MLA | 0.321 | 53.48 | 81 | KDA | 0.843 | 140.53 |
| 36 | KDA | 0.870 | 144.96 | 82 | KDA | 0.842 | 140.40 |
| 37 | KDA | 0.847 | 141.22 | 83 | MLA | 0.303 | 50.56 |
| 38 | KDA | 0.861 | 143.55 | 84 | KDA | 0.843 | 140.46 |
| 39 | MLA | 0.305 | 50.83 | 85 | KDA | 0.842 | 140.33 |
| 40 | KDA | 0.858 | 143.04 | 86 | KDA | 0.837 | 139.58 |
| 41 | KDA | 0.891 | 148.51 | 87 | MLA | 0.307 | 51.20 |
| 42 | KDA | 0.838 | 139.59 | 88 | KDA | 0.843 | 140.45 |
| 43 | MLA | 0.300 | 50.01 | 89 | KDA | 0.843 | 140.57 |
| 44 | KDA | 0.853 | 142.16 | 90 | KDA | 0.842 | 140.40 |
| 45 | KDA | 0.845 | 140.75 | 91 | MLA | 0.311 | 51.77 |
| 46 | KDA | 0.839 | 139.84 | 92 | MLA | 0.308 | 51.25 |

| Group | Layers | Total | Mean | Slowest | Fastest |
|---|---:|---:|---:|---|---|
| KDA | 68 | 57.931 s | 141.99 ms/pos | 16 at 152.84 | 6 at 138.99 |
| MLA | 24 | 7.309 s | 50.76 ms/pos | 23 at 54.59 | 11 at 48.94 |
| All | 92 | 65.240 s | | | |

**No layer is an outlier worth its own replica count.** The spread inside KDA is
138.99 to 152.84 ms, a 10% band, and inside MLA 48.94 to 54.59 ms, an 11% band. Scale
by type, not per layer. If one layer later becomes slow, it will be a change in that
layer, not a property of the model as it stands.

Layer 40 specifically, which prompted this: 143.04 ms/position, within 0.7% of the KDA
mean. It needs no more pods than any other KDA layer.

## Throughput: how many requests the fleet can handle

Everything in this section is arithmetic over the measured service times above. No
95-pod fleet has been run, so these are projections from measured parts.

### The two costs are not the same

A request is `P` prompt tokens and `G` generated tokens.

- **Prefill** processes the `P` prompt tokens. They are all known in advance, so they can
  flow through the pods as a pipeline: position 1 can be at layer 2 while position 2 is
  at layer 1. Prefill is limited by the **slowest single stage**.
- **Decode** produces the `G` tokens one at a time. Token `n+1` cannot start until token
  `n` has gone through every pod and come back, because it *is* the next input. Decode
  is limited by the **sum of all stages** and cannot be pipelined.

Positions through the server and layers: `P + G - 1`. Calls to layer 93: `G`.

### Measured building blocks

| Quantity | Value | From |
|---|---:|---|
| Server, per position | 431.89 ms | measured |
| All 92 layers, per position | 10,873.5 ms | 68 x 141.99 + 24 x 50.76 |
| **Full traversal, server + 92 layers** | **11,305.4 ms** | sum |
| Layer 93, per token | 285.74 ms | measured |
| **One decode token, end to end** | **11,591.1 ms** | traversal + layer 93 |
| Slowest single stage | 431.89 ms | the server |

**A generated token costs 11.59 seconds.** That is the number that governs everything
else, and no amount of replication reduces it. Replicas add throughput, never speed to
one token.

### Single request latency

`latency = traversal + (P-1) x 431.89ms + 285.74ms + (G-1) x 11,591.1ms`

| Request | P | G | Prefill | Decode | Total latency |
|---|---:|---:|---:|---:|---:|
| "the capital of france is" | 5 | 2 | 13.03 s | 11.88 s | **24.9 s** |
| Short chat | 100 | 50 | 54.06 s | 568.2 s | **10.4 min** |
| Typical | 500 | 200 | 226.8 s | 2,306.9 s | **42.2 min** |
| Long | 2,000 | 500 | 874.7 s | 5,784.2 s | **1.85 h** |

Decode is 95% of the time on anything but a trivial request. Prompt length is nearly
free by comparison; **generated length is what costs**.

### Fleet throughput, one pod of each

The server is the binding stage at 431.89 ms per position.

| Request shape | Positions | Server time per request | Throughput | Per hour |
|---|---:|---:|---:|---:|
| P=5, G=2 | 6 | 2.59 s | 0.386 req/s | **1,390** |
| P=100, G=50 | 149 | 64.35 s | 0.0155 req/s | **56** |
| P=500, G=200 | 699 | 301.9 s | 0.0033 req/s | **11.9** |
| P=2000, G=500 | 2,499 | 1,079.3 s | 0.0009 req/s | **3.3** |

Aggregate token rate with one pod of each: **2.32 tokens/s** across all concurrent
requests, set by the server.

By Little's law the P=5 case holds 0.386 x 24.9 = **9.6 requests in flight**, which is
9.6 x 7.41 MB = 71 MB of sequence state per layer pod. Not a constraint.

### Fleet throughput, balanced

Replicate until the next stage binds. Capacities per pod: server 2.32, KDA 7.04,
MLA 19.70 positions/s; layer 93 3.50 tokens/s.

| Configuration | Binding stage | Aggregate positions/s | Tokens/s | P=500,G=200 req/hour |
|---|---|---:|---:|---:|
| 1 of each (95 pods) | server | 2.32 | 2.32 | 11.9 |
| 3 servers | server | 6.95 | 6.95 | 35.8 |
| 4 servers | KDA layers | 7.04 | 7.04 | 36.3 |
| 5 servers, 2x KDA | MLA layers | 11.59 | 11.59 | 59.7 |
| 11 servers, 4x KDA, 2x MLA | KDA layers | 28.17 | 28.17 | 145.1 |
| 22 servers, 8x KDA, 3x MLA | MLA layers | 51.03 | 51.03 | 262.8 |

Going from 95 pods to 4 servers triples throughput for the cost of 3 extra small pods
and 18 GB. That is the cheapest move available and it is the first one to make.

After that each doubling costs roughly a doubling of the whole layer tier, because
68 KDA replicas is 1,070 GB of RAM per round.

### What dominates, and what to do about it

| Stage | Share of one decode token |
|---|---:|
| KDA layers (68) | 83.3% |
| MLA layers (24) | 10.5% |
| Server | 3.7% |
| Layer 93 | 2.5% |

The 68 KDA layers are the whole problem. They are also the layers that scale worst with
cores: 2.42x on sixteen. Making one KDA layer 2x faster would take 11.59 s per token
down to about 7.4 s; nothing else on this list moves the needle.

Two things are worth measuring before buying hardware:

1. **Why KDA stalls at 2.4x on cores while MLA reaches 10.9x.** Same single-thread cost,
   very different scaling. If the serial part of a KDA layer can be parallelised, it is
   worth more than any amount of replication.
2. **The network cost**, which is still unmeasured and is noted below as the weakest
   assumption in this document.

## What these numbers do not cover

- **One run per layer, no variance.** Each figure is a single six-position measurement.
  I have not repeated them, so I cannot separate a real difference from run-to-run
  noise. The 10% spread inside each group is within what I would expect from noise
  alone. Three runs per layer would settle it.
- **Synthetic input.** `pod-cost` feeds pseudo-random vectors, so expert selection is
  random rather than whatever a real prompt induces. The count is right at sixteen
  experts per position, but a real prompt may reuse experts across positions more than
  random vectors do, which would make a pod slightly faster than shown.
- **Measured alone, not under load.** Every pod figure was taken with nothing else
  running. Real pods share a host, and cores, caches and memory bandwidth are contended.
  Expect worse than these figures, not better.
- **No network cost.** The frame is 258,072 bytes and crosses a pod boundary 94 times
  per position. At 1 GbE that is roughly 2 ms per hop and about 194 ms per position
  added, which would be larger than an MLA layer's entire compute. **This has not been
  measured and could change the sizing materially.** It should be measured before the
  fleet is built.
- **Start-up not in the capacity figures.** A pod needs ~6.0 s to read its 15.72 GB
  before serving. Rolling a 136-pod tier is minutes, not seconds.
- **Layer 93 replica counts assume one token per position**, the worst case.- **The throughput section assumes prefill pipelines across pods and decode does not.**
  Prefill pipelining is sound in principle, because a layer pod is free again after
  141.99 ms while the server only feeds it every 431.89 ms, but it depends on the
  transport actually overlapping stages. It has not been demonstrated. If it does not
  overlap, prefill falls back to the sum of stages and the P=500 latency goes from
  42.2 min to about 2.2 h.- The 29% projected improvement is arithmetic over separately measured parts, not a
  running 95-pod fleet. The parts are measured; the assembly is not.
