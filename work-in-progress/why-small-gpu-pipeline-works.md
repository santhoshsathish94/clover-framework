# Why a small-GPU pipeline works

**A model that does not fit on one machine has to be cut somewhere. Cutting it by layer
lets ordinary cards on ordinary networking run it. This is the mechanism, and its price.**

Exploratory. Measurements are from one CPU machine and the released Kimi K3 checkpoint.
Nothing here has been run on a GPU; where a figure is modeled rather than measured it says
so, and the last section lists what that leaves unsettled.

---

## The situation that forces a choice

Kimi K3 holds **1,555.3 GB** for inference — 108.81 GB of trunk and 1,446.5 GB of routed
experts, the experts being **93.0%** of it. Read from `trunk.json` and the safetensors
headers.

No single device comes close. The largest accelerator available holds 141 GB; our CPU
machine holds 124 GiB. So the weights have to be divided across machines, and a transformer
gives you exactly two places to cut.

---

## Cut A — by layer. This is the pipeline.

Device *i* holds whole layers. A token enters device 1, comes out, enters device 2, and so
on to the end.

```
device 1        device 2        device 3              device N
layers 0..k     layers k+1..    layers ..             layers ..92
   |               ^               ^                     ^
   +-- 28 KB ------+-- 28 KB ------+ ......... 28 KB -----+
```

**What crosses:** the hidden state. 7168 values at 4 bytes = **28 KB** per boundary,
2.67 MB end to end. Against 2.24 GB read locally per layer, that is about **1 : 80,000**.

A 1 Gbit/s port carries roughly **4,360 tokens per second** of that traffic — more than the
devices at either end can consume. Ordinary ethernet is sufficient. No special fabric.

## Cut B — within the layer. This is what needs the fabric.

Every device holds a slice of **every** layer. All devices work on layer 0 together, then
layer 1 together, and so on.

```
             layer 0          layer 1          ...      layer 92
device 1     slice            slice                     slice
device 2     slice            slice                     slice
device N     slice            slice                     slice
             \___ combine ___/ \___ combine ___/         \___ combine
```

**What crosses:** partial results, which must be reconciled **at every layer** before the
next can start. Each device computed part of the answer; none of them has it.

This is why such systems are built with 900 GB/s between cards rather than 1 Gbit/s. The
interconnect is not a convenience — the arrangement does not function without it.

---

## The measured quantity that decides between them

**Weights are read per batch, not per token.** Read a layer's weights once and you can
apply them to every token in the batch at that moment. This is the single most important
fact about serving cost, and we measured it on our own machine.

Prefill processes many positions in one sweep; decode processes one. Same engine, same
weights, same run — so the difference is a direct measurement of what batch size is worth:

| positions in the batch | seconds per token | advantage |
|---:|---:|---:|
| 5 | 3.878 | 1.36× |
| 46 | 3.056 | 1.74× |
| 64 | 2.841 | 1.87× |
| 399 | **2.462** | **2.21×** |

Monotonic, and still improving at 399. The same effect shows in the expert reads: **in-layer
expert diversity falls from 77.2% at 5 positions to 14.3% at 225**, so a bigger batch touches
proportionally fewer distinct experts and reads fewer bytes for each token it produces.

### And this is exactly where the two cuts differ

With **N** concurrent requests and **S** stages:

- **Cut A** spreads the requests across the stages, so each stage only ever sees **N/S** of
  them at once.
- **Cut B** keeps all **N** in one batch, because every device participates in every layer.

So Cut A divides the batch by the number of stages. **That is its cost, and it is the only
real one.** Everything else about Cut A is cheaper.

Modeled at 744 concurrent requests, using our measured byte counts:

| per token | Cut A, 93 stages (batch 8) | Cut B, one batch of 744 |
|---|---:|---:|
| trunk | 13.60 GB | 0.15 GB |
| experts | 24.27 GB | 1.94 GB |
| per-request state and KV | 0.54 GB | 0.54 GB |
| **total** | **38.41 GB** | **2.63 GB** |

Same layers, same order, same routing. To make the same 744 tokens, the first device in
Cut A reads layer 0's weights 93 times; in Cut B they are read once.

---

## The trade, stated once

> **Cut A buys commodity networking by paying batch efficiency.
> Cut B buys batch efficiency by paying for an interconnect.**

Neither is free and neither is wrong. They are the same physics arranged for different
constraints.

The cost of Cut A is also adjustable, and this is the part worth knowing before building
anything. Stage count is a dial, not a fixed property — put more layers on each device and
the batch per stage rises:

| stages | layers each | batch per stage at N=744 | GB per token |
|---:|---:|---:|---:|
| 93 | 1 | 8 | 38.41 |
| 24 | ~4 | 31 | 24.02 |
| 12 | ~8 | 62 | 17.99 |
| 4 | ~23 | 186 | 8.63 |
| 1 | 93 | 744 | 2.63 |

Fewer stages is always cheaper per token, and at one stage Cut A *is* Cut B. But each step
down that table needs a device large enough to hold more layers, or a faster link between
the devices that share them. **That is the whole design space.**

---

## What else differs, measured

**Balance.** Layers are not the same size. From the trace joined to `trunk.json`:

```
min   1.722 GB   layer 83 (MLA attention)
max   2.566 GB   layer 1  (KDA attention)
mean  2.242 GB          spread 1.49x
```

MLA sits at layers 3, 7, 11 … 87, then 91 and 92 back to back, and its slice is smaller than
KDA's — so the cost oscillates with period 4. Layer 0, which carries a dense feed-forward
instead of experts, is 2.341 GB on its own.

Cut A inherits this directly: **the rate is set by the heaviest stage**, so a one-layer-per-
device arrangement is unbalanced by 1.49× before it starts, and splitting layer 0 buys more
than a faster card. Cut B is even by construction, because every device holds a slice of
everything.

**Behavior at one request.** Cut A parallelizes across layers, so with a single request in
flight there is nothing to spread and all but one device idles. Cut B parallelizes inside
each layer, so one request uses every device. For a single user Cut A is at its worst, not
its best — and autoregression means you cannot fill it with that user's own next token
either, since token *N+1* needs token *N*.

**What neither escapes.** Both walk all 93 layers in order. Both must run each layer's
router before they know which 16 of its 896 experts to read — we tested three ways to
predict that in advance and two were dead on arrival. And both face an arithmetic intensity
of about **1.5 FLOP per byte** at one token, against hardware built for 5 to 200. Every
machine is starved on this workload; the faster the machine, the more of it sits waiting.

---

## Why the pipeline works, in one place

Pulling the measured facts together, this is the case:

1. **The model never needs to be resident anywhere as a whole.** A layer needs its own
   trunk slice and its 16 chosen experts — a **1.451 GB** working set, 1,071× smaller than
   the 1,555.3 GB model.
2. **One layer plus all 896 of its experts is 17.0 GB**, which fits a 24 GB card with room
   to spare. Hold that and storage leaves the critical path entirely — on our machine,
   waiting for SSD is 35% of every token.
3. **Only 28 KB crosses between layers**, so the link between devices can be ordinary
   ethernet. This is the fact that makes the whole arrangement possible, and it is the one
   people expect to be the obstacle.
4. **The bandwidth arrives with the cards.** Each card brings its own — a 432 GB/s card is
   9× our CPU's entire memory bandwidth, and there are as many as you choose to rent.
5. **It scales by adding machines, not by replacing one**, and the model is already sliced
   93 ways by construction.

**And the price, stated in the same breath:** the batch is divided by the number of stages,
so each stage reads its weights for fewer tokens than a single large batch would. That is
real, it is the only significant cost, and the stage-count dial above is how it is managed.

## Which reality this suits

Not a ranking. The constraints are different, so the answers are.

**The pipeline fits when** you can rent or buy modest cards but cannot build a fabric; when
the model must be resident with no storage in the loop; when concurrency is modest, so the
batch was never going to be large anyway; and when you would rather add a machine than
replace one.

**Cutting within the layer fits when** the fabric already exists; when there is enough
concurrent demand to fill a large batch, which is where its advantage comes from; and when
per-request latency matters, because every device works on every token.

The honest summary of the measurements is that **the value of cutting within the layer
rises with concurrency, and the cost of the pipeline rises with stage count**. Someone
serving a handful of requests on rented cards and someone serving thousands on owned
hardware are not making the same trade, and neither is making a mistake.

---

## What is not settled

- **No GPU has been measured.** Every rate here is a 7950X3D with 47.9 GB/s of DRAM and two
  NVMe drives. The per-token byte counts are measured and transfer directly; the throughput
  figures are those bytes divided by vendor bandwidth numbers, which is a model.
- **Cut B's interconnect cost is assumed, not measured.** At small batches it is the
  dominant term, and the numbers move a long way on it.
- **The engine cannot do either of these today.** It is one process walking all 93 layers.
  Whether it can be split at all is untested.
- **Concurrent-user routing is modeled as independent.** We measured 36.8% expert overlap
  between consecutive tokens of one stream against about 1.8% under an independence
  assumption, so real overlap is much higher — but between *different users* it is unmeasured.
- **Every price is an assumption** except Hetzner's GEX45 at €214.00/month, read from their
  page on 2026-09-25.

---

**Evidence** — `CONTEXT-kimi-k3-benchmark.md` (the measurement campaign, including the
predictions that turned out wrong), `k3-flow/cost-notes.md` (per-layer costs, the sawtooths,
and the routing-prediction dead ends), `residency-not-speed.md` (why the model has to be
split at all).
