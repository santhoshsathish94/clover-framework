# The client-server shape, and why the cut is per layer

The single-machine work answered the computation question: the model is an
equation, the equation runs, and it reproduces the reference byte for byte.
What it did not answer is the **data movement** question. One machine reading
1.5 TB of experts off its own disks is bounded by those disks — measured at
14.6 GB/s, and already running at 94% of it.

This document states the shape that follows from that, and separates what has
been measured from what has not.

> The equation solved the computation problem.
> The system has to solve the data movement problem.
> If each pod already holds every byte it needs, only vectors travel.

---

## 1. The cut

```
  client machine            head, layer 0, tail          5.87 GB, no experts
        |  vector
  router                    which layer runs next
        |  vector
  pod 1 ... pod 92          one layer each               ~16.2 GB per pod
```

Two different things get called routing, and they are kept apart deliberately:

- **The router routes between layers.** It takes a request and decides which
  layer runs next. It holds no weights.
- **The MoE gate routes to experts.** Top-k selection inside a single layer,
  driven by that layer's own trunk slice, and it never leaves the pod.

---

## 2. What the client holds — measured

The two ends need the checkpoint's five non-layer tensors, and **neither
touches the trunk or any expert**:

```
  embed_tokens          2,348,810,240   [163840, 7168]   HEAD  lookup
  output_attn_res_norm         14,336   [7168]           TAIL  residual fold
  output_attn_res_proj         14,336   [1, 7168]        TAIL  residual fold
  norm                         14,336   [7168]           TAIL  final rmsnorm
  lm_head               2,348,810,240   [163840, 7168]   TAIL  matmul
                        -------------
                              4.70 GB
```

**Layer 0 can sit with the client too, and only layer 0.** It is the one layer
in the model with no experts at all — `isMoE = (L >= 1)` — so it is a dense
MLP and needs nothing but its own trunk slice, measured at **1.172 GB**. That
is the largest slice in the model, because the dense MLP it carries is 727 MB
that the MoE layers do not have.

```
  client total    4.70 GB  +  1.172 GB  =  5.87 GB,  zero experts
```

That is a laptop. The head is a genuine lookup — 14,336 B per token out of a
2.35 GB table. The tail is a full matmul over the whole vocabulary, 2.35 GB and
2.35 GFLOP, run **once per prompt** on the last position only.

**Layer 1 cannot join them.** It is a MoE layer and needs its 896 experts. The
494,592 expert records divide as 5,376 per layer for layers 1 to 92 and zero
for layer 0. Note also the numbering: there are 93 layers, 0 to 92, and what
the code labels `cur_L = 93` is the tail rather than a layer.

---

## 3. What a pod holds — measured

Each layer's weights are a contiguous range of the trunk, and each layer's
experts belong to that layer alone.

```
  trunk slice     0.423 - 0.635 GB   layers 1 to 92, measured from trunk.json
  its experts        15.72 GB        896 experts x 17,547,264 B
                  ----------
  per pod          ~16.2 GB
```

**That is the number that matters.** A pod is a commodity machine, not a
special one. Nothing in a pod needs to reach another pod's storage, and no pod
ever reads the 1.5 TB whole.

---

## 4. What crosses a boundary — measured

Not token ids. Each layer consumes the residual for every position **plus
every snapshot taken so far**, and snapshots accumulate where `L % 12 == 0`,
so the payload grows with depth:

```
  layer    vectors    bytes/position
  L1        2          57,344
  L25       4         114,688
  L48       5         143,360
  L85       9         258,048
```

| positions | all 92 hops | mean per hop |
|---|---|---|
| 1 | 14.11 MB | 0.15 MB |
| 5 | 70.53 MB | 0.77 MB |
| 16 | 225.71 MB | 2.45 MB |
| 64 | 902.82 MB | 9.81 MB |

---

## 5. Why this shape could win, and what would sink it

**The argument for it.** A single machine spends its time moving expert bytes
from its own disk into its own memory: 99.86 GB per prompt at a measured
ceiling of 14.6 GB/s. If every pod already holds its own layer's 16.2 GB, that
movement does not happen at request time at all. What travels instead is a
vector of tens to hundreds of kilobytes. The expensive thing stops moving and
the cheap thing moves.

**The argument against it, and it is not small.** At 64 positions the hops
total 902.82 MB against a 47 ms stage. On one node, over shared memory, that is
free. Across nodes at 10 Gb/s it is **~0.7 s per prompt in network alone**. So
co-scheduling is a real constraint rather than a preference, and the deepest
layers are the expensive hops because the snapshot stack has grown.

**The claim that is not established.** That this beats a large GPU server on
tokens per second is a hypothesis, not a result. Nothing here has been
measured on a GPU, across nodes, or under concurrent load. What has been
measured is the single-machine ceiling and the sizes above.

---

## 6. How correctness would be established

Every piece is gated against the reference's own output, so nothing is judged
by whether it looks plausible.

`K3_DUMPLAY` makes the reference write each layer's input, so a pod holding
layer N can be checked against the reference's dump for layer N+1, one layer at
a time, **without modifying the reference**. The end of the chain is checked
against the preserved baseline:

```
The capital of France is  ->  17374 (' Paris')
logits md5 23d162dcefb18211a7540ef12948f1eb
```

A layer that disagrees with the reference is wrong. There is no second opinion
and no tolerance to argue about.

---

## 7. Status

**This is a shape, not a running system.** An earlier single-process version on
a per-layer data layer did run and did reproduce the gate, but it was removed
at v4.0 once the SQLite store it read from was deleted to put the checkpoint
back; it survives in git history at `abdcde7^:clover-server-k3/`.

What exists today is [`clover-k3`](../clover-k3/), the single-machine
reference, and the measurements above. The sizes are measured. The topology is
not built.

The step-by-step record, including the parts of this that were tried and
abandoned, is in
[clover-scaling-architecture.md](clover-scaling-architecture.md) steps 4 to 11.
